import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # Usa 'mp4v' para MP4 o 'XVID' para AVI
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Forzar extensión .mp4 si el usuario olvidó
    if not output_video_path.endswith('.mp4'):
        output_video_path = output_video_path.rsplit('.', 1)[0] + '.mp4'

    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"🎬 Video guardado correctamente en: {output_video_path}")
