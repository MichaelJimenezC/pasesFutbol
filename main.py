from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import csv, os

PROCESS_SHORT_SIDE = 720
PROCESS_FPS = 10

APPLY_VIEW_TRANSFORM = False    
CALC_SPEED_DISTANCE  = False   
DRAW_CAMERA_MOVEMENT = False     
DRAW_ALL_PLAYERS     = False     


def compute_pass_counts_over_time(tracks, min_confirm=3, missing_grace=4):
    """
    Pase = cambio de JUGADOR dentro del MISMO EQUIPO, confirmado por min_confirm frames.
    missing_grace: si se pierde el balón pocos frames, mantenemos la última posesión.
    Devuelve:
      - team_passes: {1:int, 2:int}
      - timeline: [{1:int,2:int} por frame]
      - events: [{'frame', 'team', 'from_id', 'to_id'}]
    """
    team_passes = {1: 0, 2: 0}
    timeline = []
    events = []

    current = None     
    candidate = None
    cand_count = 0
    missing = 0

    n_frames = len(tracks['players'])
    for f in range(n_frames):
        holder = None
        for pid, p in tracks['players'][f].items():
            if p.get('has_ball', False):
                holder = (p.get('team', None), pid)
                break

        if holder is None:
            missing += 1
            if current and missing <= missing_grace:
                timeline.append(team_passes.copy())
                continue
            candidate = None; cand_count = 0
            timeline.append(team_passes.copy())
            continue

        missing = 0
        if current is None:
            current = holder
            timeline.append(team_passes.copy())
            continue

        if holder == current:
            candidate = None; cand_count = 0
            timeline.append(team_passes.copy())
            continue

        if candidate is None or holder != candidate:
            candidate = holder
            cand_count = 1
        else:
            cand_count += 1
            if cand_count >= min_confirm:
                if candidate[0] == current[0] and candidate[0] in (1, 2):
                    team_passes[candidate[0]] += 1
                    events.append({
                        'frame': f,
                        'team': candidate[0],
                        'from_id': current[1],
                        'to_id': candidate[1],
                    })
                current = candidate
                candidate = None
                cand_count = 0

        timeline.append(team_passes.copy())

    return team_passes, timeline, events


def smooth_ball_possession(tracks, min_confirm=2, grace=3):
    """
    Marca 'has_ball' de forma estable:
      - Requiere min_confirm frames para aceptar un cambio de jugador.
      - Si se pierde el balón por 'grace' frames, mantenemos la última posesión.
      - No revienta si el holder actual no existe en el frame.
    """
    current_holder_id = None
    candidate_id = None
    confirm_cnt = 0
    lost_cnt = 0

    n_frames = len(tracks['players'])
    for f in range(n_frames):
        for pid, p in tracks['players'][f].items():
            p['has_ball'] = False

        assigned_id = None
        for pid, p in tracks['players'][f].items():
            if p.get('_assigned_raw', False):
                assigned_id = pid
                break

        if assigned_id is None:
            lost_cnt += 1
            if current_holder_id is not None and lost_cnt <= grace:
                if current_holder_id in tracks['players'][f]:
                    tracks['players'][f][current_holder_id]['has_ball'] = True
            else:
                candidate_id = None
                confirm_cnt = 0
            continue

        lost_cnt = 0
        if current_holder_id is None:
            current_holder_id = assigned_id
            tracks['players'][f][current_holder_id]['has_ball'] = True
            candidate_id = None
            confirm_cnt = 0
            continue

        if assigned_id == current_holder_id:
            if current_holder_id in tracks['players'][f]:
                tracks['players'][f][current_holder_id]['has_ball'] = True
            candidate_id = None
            confirm_cnt = 0
            continue

        if candidate_id is None or candidate_id != assigned_id:
            candidate_id = assigned_id
            confirm_cnt = 1
        else:
            confirm_cnt += 1
            if confirm_cnt >= min_confirm:
                current_holder_id = candidate_id
                candidate_id = None
                confirm_cnt = 0

        if current_holder_id in tracks['players'][f]:
            tracks['players'][f][current_holder_id]['has_ball'] = True


def _ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def export_frame_metrics_csv(path, tracks, team_ball_control, pass_counts_timeline, pass_events, fps):
    _ensure_dir(path)
    events_by_frame = {e['frame']: e for e in pass_events}
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["frame", "time_s", "team_ball", "holder_id", "holder_team", "team1_passes", "team2_passes", "pass_event"])
        n = len(team_ball_control)
        for i in range(n):
            holder_id = -1
            holder_team = -1
            for pid, p in tracks['players'][i].items():
                if p.get('has_ball', False):
                    holder_id = pid
                    holder_team = p.get('team', -1)
                    break
            counts = pass_counts_timeline[min(i, len(pass_counts_timeline)-1)]
            w.writerow([
                i,
                f"{i/float(fps):.2f}",
                int(team_ball_control[i]),
                holder_id,
                holder_team,
                counts.get(1, 0),
                counts.get(2, 0),
                1 if i in events_by_frame else 0
            ])

def export_pass_events_csv(path, pass_events, fps):
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["frame", "time_s", "team", "from_id", "to_id"])
        for e in pass_events:
            w.writerow([e['frame'], f"{e['frame']/float(fps):.2f}", e['team'], e['from_id'], e['to_id']])

def export_possession_segments_csv(path, team_ball_control, fps):
    _ensure_dir(path)
    segs = []
    n = len(team_ball_control)
    if n == 0:
        return
    start = 0
    team = int(team_ball_control[0])
    for i in range(1, n):
        if int(team_ball_control[i]) != team:
            segs.append((start, i-1, team))
            start = i
            team = int(team_ball_control[i])
    segs.append((start, n-1, team))
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["start_frame", "end_frame", "duration_s", "team"])
        for s, e, t in segs:
            w.writerow([s, e, f"{(e - s + 1)/float(fps):.2f}", t])

def export_metrics_csv(path, team_ball_control, pass_counts_timeline, fps):
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["frame", "time_s", "team_ball", "team1_passes", "team2_passes"])
        for i, counts in enumerate(pass_counts_timeline):
            t = i / float(fps)
            w.writerow([i, f"{t:.2f}", int(team_ball_control[i]), counts.get(1,0), counts.get(2,0)])


def main(input_path='input_videos/08fd33_4.mp4', output_path='output_videos/output_video.mp4'):
    print("📹 [1/15] Leyendo video...")
    video_frames = read_video(input_path, target_short_side=PROCESS_SHORT_SIDE, target_fps=PROCESS_FPS)
    print("✅ Video leído. Total de frames:", len(video_frames))

    print("📦 [2/15] Cargando modelo Tracker...")
    tracker = Tracker(
        'models/best.pt',
        imgsz=768,
        conf=0.25,
        iou=0.45,
        detect_stride=1,
        batch_size=16,
        classes=None
    )
    print("✅ Modelo cargado.")

    print("🧠 [3/15] Obteniendo tracks...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path='stubs/track_stubs.pkl'
    )
    print("✅ Tracks obtenidos.")

    print("🧭 [4/15] Añadiendo posiciones a los tracks...")
    tracker.add_position_to_tracks(tracks)
    print("✅ Posiciones añadidas.")

    print("🎥 [5/15] Estimando movimiento de cámara...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    print("✅ Movimiento de cámara estimado.")

    print("🧭 [6/15] Ajustando posiciones según movimiento de cámara...")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print("✅ Posiciones ajustadas.")

    print("📐 [7/15] Transformando perspectiva del campo...")
    if APPLY_VIEW_TRANSFORM:
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)
        print("✅ Perspectiva transformada.")
    else:
        print("⏭️  Vista: saltada (modo pases)")

    print("⚽ [8/15] Interpolando posiciones del balón...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print("✅ Posiciones del balón interpoladas.")

    print("🚀 [9/15] Estimando velocidad y distancia...")
    if CALC_SPEED_DISTANCE:
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
        print("✅ Velocidad y distancia estimadas.")
    else:
        print("⏭️  Velocidad/Distancia: saltado (modo pases)")

    print("👕 [10/15] Asignando colores de equipo...")
    team_assigner = TeamAssigner()

    if len(tracks['players'][0]) == 0:
        for f in range(1, len(tracks['players'])):
            if len(tracks['players'][f]) > 0:
                team_assigner.assign_team_color(video_frames[f], tracks['players'][f])
                break
    else:
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    print("✅ Colores asignados.")

    print("👥 [11/15] Asignando equipos por jugador/frame...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))
    print("✅ Equipos asignados por jugador.")

    print("🥎 [12/15] Asignando posesión del balón (crudo)...")
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_dict = tracks['ball'][frame_num].get(1, None)
        if ball_dict is None:
            continue
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_dict)
        if assigned_player != -1:

            tracks['players'][frame_num].setdefault(assigned_player, {})['_assigned_raw'] = True
    print("✅ Posesión cruda asignada.")

    print("🧪 [12.2/15] Suavizando posesión…")
    smooth_ball_possession(tracks, min_confirm=2, grace=3)
    print("✅ Posesión suavizada.")

    print("🏳️ [12.3/15] Recalculando control de balón…")
    team_ball_control = []
    for f in range(len(tracks['players'])):
        team = -1
        for pid, p in tracks['players'][f].items():
            if p.get('has_ball', False):
                team = p.get('team', -1)
                break
        if team == -1 and team_ball_control:
            team = team_ball_control[-1]
        team_ball_control.append(team)
    team_ball_control = np.array(team_ball_control)
    print("✅ Control de balón recalculado.")

    print("🔢 [12.5/15] Contando pases (robusto)…")
    team_passes_final, pass_counts_timeline, pass_events = compute_pass_counts_over_time(
        tracks, min_confirm=3, missing_grace=4
    )
    print("✅ Pases:", team_passes_final)

    print("🎨 [13/15] Dibujando anotaciones (modo pases)…")
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control, pass_counts_timeline,
        draw_all_players=DRAW_ALL_PLAYERS
    )
    print("✅ Anotaciones dibujadas.")

    print("📹 [14/15] Overlay de movimiento de cámara…")
    if DRAW_CAMERA_MOVEMENT:
        output_video_frames = camera_movement_estimator.draw_camera_movement(
            output_video_frames, camera_movement_per_frame
        )
        print("✅ Movimiento de cámara dibujado.")
    else:
        print("⏭️  Overlay de movimiento de cámara: saltado")

    print("📏 [15/15] Overlay velocidad/distancia…")
    if CALC_SPEED_DISTANCE:
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
        print("✅ Velocidad y distancia dibujadas.")
    else:
        print("⏭️  Overlay de velocidad/distancia: saltado")

    print("💾 Guardando video de salida…")
    save_video(output_video_frames, output_path, fps=PROCESS_FPS)
    print("🎉 Video guardado como", output_path)


    print("🧾 Exportando métricas…")
    metrics_dir = "output_videos/metrics"
    export_frame_metrics_csv(os.path.join(metrics_dir, "metrics.csv"),
                             tracks, team_ball_control, pass_counts_timeline, pass_events, PROCESS_FPS)
    export_pass_events_csv(os.path.join(metrics_dir, "pass_events.csv"),
                           pass_events, PROCESS_FPS)
    export_possession_segments_csv(os.path.join(metrics_dir, "possession_segments.csv"),
                                   team_ball_control, PROCESS_FPS)
    print(f"✅ Métricas listas en {metrics_dir}/")


if __name__ == '__main__':
    main()
