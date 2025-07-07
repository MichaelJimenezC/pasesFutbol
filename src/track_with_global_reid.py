import os
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchreid
import ultralytics
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

VIDEO_PATH = "../assets/15sec_input_720p.mp4"
MODEL_PATH = "../assets/best.pt"
OUTPUT_DIR = "../output/tracked_frames"
OUTPUT_VIDEO_PATH = "../output/tracked_video_reid_final.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

global_id_counter = 0
active_tracks = {}
inactive_gallery = []
ball_positions = []  # Lista para almacenar las posiciones del balón
player_positions = {}  # Diccionario para almacenar las posiciones de los jugadores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.to(device)
reid_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Tkinter Setup
root = tk.Tk()
root.title("Video y Pases Detectados")

# Crear el área para mostrar el video
video_frame = tk.Frame(root)
video_frame.pack(side=tk.LEFT, padx=10)

# Crear el área para la lista de pases detectados
pases_frame = tk.Frame(root)
pases_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

# Etiqueta para mostrar el video
video_label = tk.Label(video_frame)
video_label.pack()

# Lista para mostrar los pases detectados
pases_listbox = tk.Listbox(pases_frame, height=20, width=40, font=('Helvetica', 12), selectmode=tk.SINGLE)
pases_listbox.pack(padx=10, pady=10)

# Contador de pases
pases_counter_label = tk.Label(pases_frame, text="Total de Pases: 0", font=('Helvetica', 14))
pases_counter_label.pack(padx=10, pady=10)

# Función para actualizar la lista de pases y el contador
def update_passes_list(passes_detected, counter):
    pases_listbox.delete(0, tk.END)  # Limpiar la lista
    for pass_info in passes_detected:
        pase_str = f"Jugador {pass_info[0]} pasa el balón al Jugador {pass_info[1]}"
        pases_listbox.insert(tk.END, pase_str)
    
    # Actualizar el contador de pases
    pases_counter_label.config(text=f"Total de Pases: {counter}")

# Extract features using ReID model
def extract_features(image_crop):
    try:
        img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = reid_model(tensor)
        return features.cpu().numpy().flatten()
    except:
        return None

# Match with inactive gallery
def match_in_gallery(features, used_global_ids, threshold=0.7):
    if not inactive_gallery:
        return None
    filtered = [g for g in inactive_gallery if g['global_id'] not in used_global_ids]
    if not filtered:
        return None
    gallery_features = [g['features'] for g in filtered]
    gallery_ids = [g['global_id'] for g in filtered]
    sims = cosine_similarity([features], gallery_features)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] > threshold:
        return gallery_ids[best_idx]
    return None

# Assign global ID to the track
def assign_global_id(track_id, bbox, frame, used_global_ids):
    global global_id_counter
    if track_id in active_tracks:
        global_id = active_tracks[track_id]['global_id']
        if global_id not in used_global_ids:
            return global_id
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    features = extract_features(crop)
    if features is None:
        return None
    matched_global_id = match_in_gallery(features, used_global_ids)
    if matched_global_id is not None:
        global_id = matched_global_id
    else:
        global_id_counter += 1
        global_id = global_id_counter
        inactive_gallery.append({'global_id': global_id, 'features': features})
    active_tracks[track_id] = {'global_id': global_id, 'features': features}
    used_global_ids.add(global_id)
    return global_id

# Retire lost tracks
def retire_lost_tracks(current_track_ids):
    lost_ids = set(active_tracks.keys()) - set(current_track_ids)
    for tid in lost_ids:
        if 'features' in active_tracks[tid]:
            inactive_gallery.append({
                'global_id': active_tracks[tid]['global_id'],
                'features': active_tracks[tid]['features']
            })
    for tid in lost_ids:
        del active_tracks[tid]

# Detect passes
def detect_pass(ball_positions, player_positions, frame_idx, distance_threshold=10):
    if len(ball_positions) < 2:
        return None  # No es posible detectar un pase sin al menos dos posiciones

    # Obtener las últimas dos posiciones del balón
    ball_position_current = ball_positions[-1]
    ball_position_previous = ball_positions[-2]

    # Calcular la distancia recorrida por el balón entre los dos frames
    ball_distance = np.linalg.norm(np.array(ball_position_current) - np.array(ball_position_previous))

    if ball_distance > distance_threshold:
        passing_player = None
        receiving_player = None
        
        for player_id, player_position in player_positions.items():
            # Verificar si el jugador está cerca del balón (por ejemplo, en un rango de 50 píxeles)
            distance_to_ball = np.linalg.norm(np.array(player_position) - np.array(ball_position_previous))
            if distance_to_ball < 50:  # Umbral para determinar que un jugador toca el balón
                passing_player = player_id
            if distance_to_ball < 50:  # El jugador que está recibiendo el pase
                receiving_player = player_id

        if passing_player and receiving_player:
            print(f"¡Pase detectado! El jugador {passing_player} pasa el balón al jugador {receiving_player}")
            return (passing_player, receiving_player)
    return None

# Main function to process each frame and detect passes
def draw_frame(frame, results, class_names, frame_idx):
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_map = {
        'player': (255, 255, 255),
        'referee': (0, 215, 255),
        'goalkeeper': (0, 0, 255)
    }
    used_global_ids = set()
    current_frame_track_ids = []
    player_positions = {}  # Diccionario para almacenar las posiciones de los jugadores
    passes_text = ""  # Variable para almacenar el texto del pase

    if results.boxes.id is not None:
        boxes = results.boxes.data.cpu().numpy()
        for *xyxy, track_id, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label = class_names.get(cls_id, f"class{cls_id}")
            if label == 'ball':
                ball_positions.append(((x1 + x2) / 2, (y1 + y2) / 2))  # Guardar posición del balón
                continue
            current_frame_track_ids.append(track_id)
            global_id = assign_global_id(track_id, (x1, y1, x2, y2), frame, used_global_ids)
            player_positions[track_id] = (x1 + x2) / 2, (y1 + y2) / 2  # Guardar posición del jugador
            color = color_map.get(label, (128, 128, 128))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            text = f"ID: {global_id}"
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # Detectar el pase en cada frame
    pass_info = detect_pass(ball_positions, player_positions, frame_idx)
    if pass_info:
        passing_player, receiving_player = pass_info
        passes_text = f"Pass detected: {passing_player} -> {receiving_player}"

    # Verificar si hay texto de pase y dibujarlo en el video
    if passes_text:
        print(f"Drawing pass text: {passes_text}")  # Depuración
        # Dibujar el texto en la parte superior del video
        cv2.putText(annotated, passes_text, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    retire_lost_tracks(current_frame_track_ids)
    return annotated

# Función para actualizar el video en tiempo real
def update_video_frame(cap, model, frame_idx, video_label, passes_detected, passes_counter):
    ret, frame = cap.read()
    if not ret:
        return False
    
    results = model.track(source=frame, persist=True, conf=0.4, verbose=False)[0]
    annotated_frame = draw_frame(frame, results, model.names, frame_idx)
    frame_idx += 1

    # Convertir la imagen de OpenCV a formato compatible con Tkinter
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_pil)

    # Actualizar la etiqueta del video con el nuevo fotograma
    video_label.config(image=frame_tk)
    video_label.image = frame_tk  # Mantener una referencia de la imagen

    # Actualizar la lista de pases y el contador
    pass_info = detect_pass(ball_positions, player_positions, frame_idx)
    if pass_info:
        passes_detected.append(pass_info)
        passes_counter += 1
        update_passes_list(passes_detected, passes_counter)

    root.after(10, update_video_frame, cap, model, frame_idx, video_label, passes_detected, passes_counter)
    return True

# Main function to run the tracking and video display
def run_tracking(video_path, model_path, output_dir, output_video_path):
    model = YOLO(model_path)
    class_names = model.names
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    passes_detected = []  # Lista para almacenar los pases detectados
    passes_counter = 0  # Contador de pases

    # Llamar a la función para actualizar el video y los pases
    update_video_frame(cap, model, frame_idx, video_label, passes_detected, passes_counter)

    root.mainloop()  # Iniciar el bucle principal de la interfaz de Tkinter

if __name__ == "__main__":
    run_tracking(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)
