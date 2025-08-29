import cv2

def read_video(video_path, target_short_side=720, target_fps=10):
    """
    Lee el video reescalando con buena calidad (INTER_AREA) y submuestreando FPS.
    Devuelve una lista de frames ya redimensionados.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 0:
        in_fps = 24.0
    step = max(int(round(in_fps / target_fps)), 1)

    ret, first = cap.read()
    if not ret:
        cap.release()
        return frames

    h, w = first.shape[:2]
    scale = target_short_side / float(min(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    frames.append(cv2.resize(first, (new_w, new_h), interpolation=cv2.INTER_AREA))
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA))
        idx += 1

    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=10, upscale_to=None):
    """
    Escribe MP4 con FPS correcto.
    - fps: usa el MISMO valor que usaste al leer (target_fps).
    - upscale_to: (w, h) si quieres reescalar el video final a un tamaño mayor.
                  Usa LANCZOS4 para máxima nitidez.
    """
    assert len(output_video_frames) > 0, "No hay frames para guardar."

    if upscale_to is not None:
        w, h = upscale_to
        output_video_frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4)
                               for f in output_video_frames]

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"🎬 Video guardado correctamente en: {output_video_path} (fps={fps}, size={width}x{height})")
