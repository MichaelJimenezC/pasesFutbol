import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QLabel, QProgressBar, QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QTimer
from main import main as run_processing


class ProcessingThread(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal()

    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        self.progress.emit("🔄 Procesando video...")
        run_processing(self.input_path, self.output_path)
        self.progress.emit("✅ Procesamiento finalizado")
        self.done.emit()


class ClickableVideoWidget(QVideoWidget):
    def __init__(self):
        super().__init__()
        self.isFullScreenMode = False

    def mouseDoubleClickEvent(self, event):
        if self.isFullScreenMode:
            self.setFullScreen(False)
            self.isFullScreenMode = False
        else:
            self.setFullScreen(True)
            self.isFullScreenMode = True


class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎥 Comparador de Videos")
        self.setGeometry(100, 100, 1000, 800)

        # Reproductores y widgets de video
        self.originalPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.originalVideoWidget = ClickableVideoWidget()
        self.outputPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.outputVideoWidget = ClickableVideoWidget()

        # Permitir loop automático
        self.originalPlayer.mediaStatusChanged.connect(self.loop_original)
        self.outputPlayer.mediaStatusChanged.connect(self.loop_output)

        # Botones
        self.loadButton = QPushButton("📂 Abrir video original")
        self.processButton = QPushButton("⚙️ Procesar video")
        self.processButton.setEnabled(False)

        # Estado
        self.statusLabel = QLabel("")
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.hide()

        # Conexiones
        self.loadButton.clicked.connect(self.load_video)
        self.processButton.clicked.connect(self.process_video)

        # Layout
        layout = QVBoxLayout()
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.originalVideoWidget)
        video_layout.addWidget(self.outputVideoWidget)

        layout.addLayout(video_layout)
        layout.addWidget(self.loadButton)
        layout.addWidget(self.processButton)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.statusLabel)
        self.setLayout(layout)

        self.originalPlayer.setVideoOutput(self.originalVideoWidget)
        self.outputPlayer.setVideoOutput(self.outputVideoWidget)

        self.input_path = None
        self.output_path = "output_videos/output_video.mp4"
        self.converted_path = "output_videos/output_video_converted.mp4"

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Videos (*.mp4 *.avi *.mov)")
        if filename:
            self.input_path = filename
            print("📥 Video original seleccionado:", filename)
            self.originalPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.originalPlayer.play()
            self.processButton.setEnabled(True)
            self.statusLabel.setText("▶️ Video original cargado")

    def process_video(self):
        if not self.input_path:
            self.statusLabel.setText("❌ No se ha cargado un video.")
            return

        print("⚙️ Iniciando procesamiento...")
        self.processButton.setEnabled(False)
        self.progressBar.show()
        self.statusLabel.setText("⏳ Procesando...")

        self.thread = ProcessingThread(self.input_path, self.output_path)
        self.thread.progress.connect(self.statusLabel.setText)
        self.thread.done.connect(self.on_processing_done)
        self.thread.start()

    def on_processing_done(self):
        print("✅ Finalizó thread de procesamiento.")
        self.progressBar.hide()

        if not os.path.exists(self.output_path):
            print("❌ No se encontró el video procesado:", self.output_path)
            self.statusLabel.setText("⚠️ El video procesado no se encontró.")
            return

        file_size = os.path.getsize(self.output_path)
        print("📏 Tamaño del video procesado:", file_size, "bytes")
        if file_size == 0:
            self.statusLabel.setText("⚠️ El video procesado está vacío.")
            return

        print("🔄 Convirtiendo video a formato compatible...")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", self.output_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                self.converted_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not os.path.exists(self.converted_path):
                self.statusLabel.setText("❌ Falló la conversión del video.")
                return

            url = QUrl.fromLocalFile(os.path.abspath(self.converted_path))
            self.outputPlayer.setMedia(QMediaContent(url))
            QTimer.singleShot(300, lambda: self.outputPlayer.setPosition(0))
            QTimer.singleShot(400, self.outputPlayer.play)
            self.statusLabel.setText("🎉 Video procesado y mostrado")
        except Exception as e:
            print("❌ Error:", e)
            self.statusLabel.setText("❌ Error al reproducir el video.")

    def loop_original(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.originalPlayer.setPosition(0)
            self.originalPlayer.play()

    def loop_output(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.outputPlayer.setPosition(0)
            self.outputPlayer.play()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
