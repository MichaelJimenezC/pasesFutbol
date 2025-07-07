import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import ultralytics

# Añadimos la línea de seguridad para los modelos de ultralytics
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

VIDEO_PATH = "../assets/15sec_input_720p.mp4"
MODEL_PATH = "../assets/best.pt"
OUTPUT_DIR = "../output/detected_frames"
OUTPUT_VIDEO_PATH = "../output/detected_video_custom.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables para el conteo de pases
ball_positions = []  # Lista para almacenar las posiciones del balón
player_positions = {}  # Diccionario para almacenar las posiciones de los jugadores
passes_count = 0  # Contador de pases
previous_ball_position = None  # Variable para almacenar la última posición del balón
previous_players_positions = {}  # Para almacenar las posiciones de los jugadores en el último frame

def detect_pass(ball_positions, player_positions, frame_idx, distance_threshold=10):
    global passes_count, previous_ball_position, previous_players_positions

    if len(ball_positions) < 2:
        return None  # No es posible detectar un pase sin al menos dos posiciones

    ball_position_current = ball_positions[-1]
    ball_position_previous = ball_positions[-2]

    # Verificar si el balón se ha movido lo suficiente
    ball_distance = np.linalg.norm(np.array(ball_position_current) - np.array(ball_position_previous))

    if ball_distance > distance_threshold:
        passing_player = None
        receiving_player = None

        # Verificar si el balón fue tocado por algún jugador
        for player_id, player_position in player_positions.items():
            if player_id in previous_players_positions:
                prev_position = previous_players_positions[player_id]
                distance_to_ball = np.linalg.norm(np.array(player_position) - np.array(ball_position_previous))

                # Verificar si el jugador está cerca del balón
                if distance_to_ball < 10:  # Umbral para determinar que un jugador tocó el balón
                    if passing_player is None:
                        passing_player = player_id
                    else:
                        receiving_player = player_id

        if passing_player and receiving_player and passing_player != receiving_player:
            print(f"¡Pase detectado! El jugador {passing_player} pasa el balón al jugador {receiving_player}")
            passes_count += 1
            return (passing_player, receiving_player)

    # Actualizar las posiciones anteriores
    previous_ball_position = ball_position_current
    previous_players_positions = player_positions

    return None

def draw_annotations(frame, results, class_names, frame_idx):
    annotated_frame = frame.copy()
    colors = {
        'player': (255, 255, 255),
        'referee': (32, 165, 218),
        'ball': (203, 192, 255),
        'goalkeeper': (255, 99, 99)
    }
    default_color = (144, 128, 112)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    box_thickness = 2
    text_color = (0, 0, 0)

    # Diccionario para almacenar las posiciones de los jugadores
    player_positions = {}

    # Puntos clave de los jugadores
    player_keypoints = {}

    if results is not None and results.boxes is not None:
        for *xyxy, conf, cls in results.boxes.data:
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            label = class_names.get(class_id, f"Class {class_id}")
            draw_color = colors.get(label, default_color)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), draw_color, box_thickness)
            confidence_str = f"{conf:.2f}"
            text = f"{label} {confidence_str}"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_bg_y1 = max(0, y1 - text_height - 5)
            cv2.rectangle(annotated_frame, (x1, text_bg_y1), (x1 + text_width + 5, y1), draw_color, -1)
            cv2.putText(annotated_frame, text, (x1 + 2, y1 - 5 if y1 - 5 >= text_height else y1 + text_height + 5),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Si es un balón, guardamos su posición
            if label == "ball":
                ball_positions.append(((x1 + x2) / 2, (y1 + y2) / 2))  # Guardar posición del balón
                continue

            # Detectar puntos clave del jugador (pies)
            if label == "player":
                if results.keypoints is not None:
                    keypoints = results.keypoints[class_id]
                    feet = keypoints[14:16]  # Ejemplo: puntos clave para los pies
                    feet_position = np.mean(feet, axis=0)
                    player_keypoints[class_id] = feet_position
                    player_positions[class_id] = (x1 + x2) / 2, (y1 + y2) / 2  # Guardar la posición del jugador

    # Detectar el pase en cada frame
    detect_pass(ball_positions, player_positions, frame_idx)

    # Añadir el texto de cuántos pases se han detectado hasta el momento
    passes_text = f"Total passes detected: {passes_count}"
    cv2.putText(annotated_frame, passes_text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated_frame

def run_detection(video_path, model_path, output_dir, output_video_path):
    model = YOLO(model_path)
    class_names = model.names
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.4, verbose=False)[0]
        annotated_frame = draw_annotations(frame, results, class_names, frame_idx)
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)
        out_writer.write(annotated_frame)
        frame_idx += 1
    cap.release()
    out_writer.release()
    print(f"Processed {frame_idx} frames. Output saved to: {output_video_path}")

if __name__ == "__main__":
    run_detection(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)
