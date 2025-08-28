from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main(input_path='input_videos/08fd33_4.mp4', output_path='output_videos/output_video.mp4'):
    print("📹 [1/15] Leyendo video...")
    video_frames = read_video(input_path)
    print("✅ Video leído. Total de frames:", len(video_frames))

    print("📦 [2/15] Cargando modelo Tracker...")
    tracker = Tracker('models/best.pt')
    print("✅ Modelo cargado.")

    print("🧠 [3/15] Obteniendo tracks...")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    print("✅ Tracks obtenidos.")

    print("🧭 [4/15] Añadiendo posiciones a los tracks...")
    tracker.add_position_to_tracks(tracks)
    print("✅ Posiciones añadidas.")

    print("🎥 [5/15] Estimando movimiento de cámara...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    print("✅ Movimiento de cámara estimado.")

    print("🧭 [6/15] Ajustando posiciones según movimiento de cámara...")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print("✅ Posiciones ajustadas.")

    print("📐 [7/15] Transformando perspectiva del campo...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    print("✅ Perspectiva transformada.")

    print("⚽ [8/15] Interpolando posiciones del balón...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print("✅ Posiciones del balón interpoladas.")

    print("🚀 [9/15] Estimando velocidad y distancia...")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print("✅ Velocidad y distancia estimadas.")

    print("👕 [10/15] Asignando colores de equipo...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    print("✅ Colores asignados.")

    print("👥 [11/15] Asignando equipos por jugador/frame...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print("✅ Equipos asignados por jugador.")

    print("🥎 [12/15] Asignando posesión del balón...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
    team_ball_control = np.array(team_ball_control)
    print("✅ Posesión del balón asignada.")

    print("🎨 [13/15] Dibujando anotaciones...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    print("✅ Anotaciones dibujadas.")

    print("📹 [14/15] Dibujando movimiento de cámara...")
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    print("✅ Movimiento de cámara dibujado.")

    print("📏 [15/15] Dibujando velocidad y distancia...")
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    print("✅ Velocidad y distancia dibujadas.")

    print("💾 Guardando video de salida...")
    save_video(output_video_frames, output_path)
    print("🎉 Video guardado como output_videos/output_video.mp4")


if __name__ == '__main__':
    main()