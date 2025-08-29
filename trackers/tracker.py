from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import torch
import inspect
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(
        self,
        model_path,
        imgsz=704,
        conf=0.25,
        iou=0.45,
        detect_stride=2,
        batch_size=16,
        classes=None
    ):
        self.model = YOLO(model_path)


        try:
            sig = inspect.signature(sv.ByteTrack.__init__)
            params = sig.parameters
            kwargs = {}
            if "track_thresh" in params:      kwargs["track_thresh"] = 0.25
            elif "track_threshold" in params: kwargs["track_threshold"] = 0.25
            if "match_thresh" in params:      kwargs["match_thresh"] = 0.8
            elif "match_threshold" in params: kwargs["match_threshold"] = 0.8
            if "track_buffer" in params:      kwargs["track_buffer"] = 90
            if "frame_rate"  in params:       kwargs["frame_rate"]  = 10
            self.tracker = sv.ByteTrack(**kwargs)
        except TypeError:
            self.tracker = sv.ByteTrack()
            if hasattr(self.tracker, "track_thresh"):      self.tracker.track_thresh = 0.25
            if hasattr(self.tracker, "track_threshold"):   self.tracker.track_threshold = 0.25
            if hasattr(self.tracker, "match_thresh"):      self.tracker.match_thresh = 0.8
            if hasattr(self.tracker, "match_threshold"):   self.tracker.match_threshold = 0.8
            if hasattr(self.tracker, "track_buffer"):      self.tracker.track_buffer = 90
            if hasattr(self.tracker, "frame_rate"):        self.tracker.frame_rate  = 10


        self.device = "cpu"
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.detect_stride = max(int(detect_stride), 1)
        self.batch_size = batch_size
        self.classes = classes
        self.model.to(self.device)

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions


    def _predict_batch(self, imgs):
        with torch.inference_mode():
            return self.model.predict(
                imgs,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                classes=self.classes
            )

    def detect_frames(self, frames):
        detections = [None] * len(frames)
        batch_imgs, batch_idxs = [], []

        def flush():
            nonlocal batch_imgs, batch_idxs
            if not batch_imgs:
                return
            results = self._predict_batch(batch_imgs)
            for idx, r in zip(batch_idxs, results):
                detections[idx] = r
            batch_imgs, batch_idxs = [], []

        for i, f in enumerate(frames):
            if i % self.detect_stride == 0:
                batch_imgs.append(f)
                batch_idxs.append(i)
                if len(batch_imgs) == self.batch_size:
                    flush()
        flush()
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}
        last_players_count = 0
        miss_ball_streak = 0

        for frame_num in range(len(frames)):

            need_rescue = False
            if frame_num > 0:
                need_rescue = (last_players_count <= 3) or (miss_ball_streak >= 2)
            if detections[frame_num] is None and need_rescue:
                detections[frame_num] = self._predict_batch([frames[frame_num]])[0]


            if detections[frame_num] is None:
                detection_supervision = sv.Detections.empty()
                cls_names = {0: 'player', 1: 'referee', 2: 'ball'}
            else:
                det = detections[frame_num]
                cls_names = det.names
                detection_supervision = sv.Detections.from_ultralytics(det)
                cls_names_inv = {v: k for k, v in cls_names.items()}

                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names.get(class_id, "") == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv.get("player", class_id)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})


            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                name = cls_names.get(cls_id, "")
                if name == 'player':
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif name == 'referee':
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}


            ball_found = False
            if detections[frame_num] is not None:
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    if cls_names.get(cls_id, "") == 'ball':
                        tracks["ball"][frame_num][1] = {"bbox": bbox}
                        ball_found = True
            miss_ball_streak = 0 if ball_found else (miss_ball_streak + 1)

            last_players_count = len(tracks["players"][frame_num])

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def _draw_hud_box(self, frame, anchor='tr', alpha=0.70):
        """
        Dibuja una caja negra semitransparente para el HUD.
        anchor: 'tr' (top-right), 'tl', 'br', 'bl'
        """
        h, w = frame.shape[:2]
        pad = int(0.02 * w)
        box_w = int(0.40 * w)
        box_h = int(0.18 * h)

        if anchor == 'tr':
            x1, y1 = w - box_w - pad, pad
        elif anchor == 'tl':
            x1, y1 = pad, pad
        elif anchor == 'br':
            x1, y1 = w - box_w - pad, h - box_h - pad
        else:
            x1, y1 = pad, h - box_h - pad

        x2, y2 = x1 + box_w, y1 + box_h

        overlay = frame.copy()

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


        bar_h = int(0.28 * box_h)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + bar_h), (40, 40, 40), -1)

        return (x1, y1, x2, y2)
    def _put_text(self, frame, text, org, fs):

        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)

        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)


    def draw_annotations(self, video_frames, tracks, team_ball_control, pass_counts_timeline, draw_all_players=False):
        """
        Modo minimalista:
        - Dibuja SOLO el poseedor (halo) y el balón.
        - HUD con % posesión y pases por equipo (fondo negro).
        - Sin IDs, sin árbitros, sin speed/distance.
        """
        output_video_frames = []
        last_ball_bbox = None

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict   = tracks["ball"][frame_num]


            holder_id = None
            for pid, p in player_dict.items():
                if p.get('has_ball', False):
                    holder_id = pid
                    break


            if draw_all_players:

                for pid, player in player_dict.items():
                    color = player.get("team_color", (0, 0, 255))
                    frame = self.draw_ellipse(frame, player["bbox"], color, track_id=None)
            else:

                if holder_id is not None and holder_id in player_dict:
                    holder = player_dict[holder_id]
                    color = holder.get("team_color", (0, 0, 255))
                    frame = self.draw_ellipse(frame, holder["bbox"], color, track_id=None)

                    frame = self.draw_traingle(frame, holder["bbox"], (0, 0, 255))


            if len(ball_dict) > 0:
                for _, ball in ball_dict.items():
                    last_ball_bbox = ball["bbox"]
                    frame = self.draw_traingle(frame, last_ball_bbox, (0, 255, 0))
            elif last_ball_bbox is not None:
                frame = self.draw_traingle(frame, last_ball_bbox, (0, 255, 0))


            x1, y1, x2, y2 = self._draw_hud_box(frame, anchor='tr', alpha=0.70)
            h, w = frame.shape[:2]
            fs = max(0.8, min(1.2, h / 900.0))


            inner_pad_x = int(0.02 * w)
            inner_pad_y = int(0.03 * h)
            tx = x1 + inner_pad_x
            ty = y1 + inner_pad_y + 28


            tbc = team_ball_control[:frame_num + 1]
            t1 = (tbc == 1).sum()
            t2 = (tbc == 2).sum()
            denom = max(t1 + t2, 1)
            txt1 = f"Team 1 Ball: {t1/denom*100:.1f}%"
            txt2 = f"Team 2 Ball: {t2/denom*100:.1f}%"

            counts = pass_counts_timeline[min(frame_num, len(pass_counts_timeline) - 1)]
            p1 = counts.get(1, 0)
            p2 = counts.get(2, 0)


            self._put_text(frame, txt1, (tx, ty), fs)
            self._put_text(frame, txt2, (tx, ty + 30), fs)
            self._put_text(frame, f"Team 1 Passes: {p1}", (tx, ty + 60), fs)
            self._put_text(frame, f"Team 2 Passes: {p2}", (tx, ty + 90), fs)

            output_video_frames.append(frame)

        return output_video_frames
