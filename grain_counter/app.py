import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QSlider
)

PINK = (255, 0, 220)  # BGR-ish visualization color


def list_cameras(max_index=12):
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cams.append(i)
        cap.release()
    return cams


class CounterEngine:
    """
    Hybrid segmentation designed for static objects on a uniform background.

    Pipeline:
      1. Estimate background/foreground by color distance.
      2. Clean mask with morphology.
      3. Distance-transform + marker watershed to split touching objects.
      4. Filter by area/circularity/size.
      5. Produce object masks for pink overlay.

    This is deliberately not dependent on a pretrained object-class model:
    arbitrary grains/items are not necessarily in COCO or another generic
    pretrained vocabulary.
    """

    def __init__(self):
        self.min_area = 80
        self.max_area = 50000
        self.threshold = 28
        self.kernel = 3
        self.min_peak = 0.28
        self.open_iter = 1
        self.close_iter = 2
        self.last_count = 0

    def calibrate(self, frame):
        \"\"\"Estimate sensible segmentation parameters from a clean frame.

        The user should place several representative objects on the surface.
        The method estimates background color from the border and chooses a
        threshold from color-distance statistics, then derives an area range
        from connected components. It is intentionally conservative: the user
        can still fine-tune the values afterwards.
        \"\"\"
        img = cv2.GaussianBlur(frame, (5, 5), 0)
        border = np.concatenate([
            img[0:20].reshape(-1, 3), img[-20:].reshape(-1, 3),
            img[:, 0:20].reshape(-1, 3), img[:, -20:].reshape(-1, 3)
        ], axis=0)
        bg = np.median(border, axis=0).astype(np.float32)
        dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
        bg_dist = dist[:20, :].ravel()
        p99 = float(np.percentile(bg_dist, 99.5))
        self.threshold = max(8, min(100, int(round(max(12, p99 * 1.8)))))

        probe = (dist > self.threshold).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        probe = cv2.morphologyEx(probe, cv2.MORPH_OPEN, k)
        n, _, stats, _ = cv2.connectedComponentsWithStats(probe, 8)
        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= 20]
        if areas:
            med = float(np.median(areas))
            self.min_area = max(10, int(med * 0.20))
            self.max_area = max(self.min_area * 10, int(max(areas) * 4))
        self.min_peak = 0.28
        return self.threshold, self.min_area, self.max_area

    def segment(self, frame):
        # Blur reduces sensor noise without destroying grain boundaries.
        img = cv2.GaussianBlur(frame, (5, 5), 0)

        # Robust background estimate: border pixels are assumed to be mostly
        # the uniform work surface.
        b, g, r = cv2.split(img)
        border = np.concatenate([
            img[0:12].reshape(-1, 3),
            img[-12:].reshape(-1, 3),
            img[:, 0:12].reshape(-1, 3),
            img[:, -12:].reshape(-1, 3)
        ], axis=0)
        bg = np.median(border, axis=0).astype(np.float32)

        dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
        mask = (dist > float(self.threshold)).astype(np.uint8) * 255

        k = max(1, int(self.kernel))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if self.open_iter:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.open_iter)
        if self.close_iter:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)

        # Remove tiny connected components before watershed.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= max(5, self.min_area // 4):
                clean[labels == i] = 255
        mask = clean

        if not np.any(mask):
            return [], mask, frame.copy()

        # Distance transform: local maxima provide seeds for individual grains.
        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        maxv = float(dist_img.max())
        if maxv <= 0:
            return [], mask, frame.copy()

        peak = dist_img > (self.min_peak * maxv)
        peak = peak.astype(np.uint8) * 255
        # Split broad peak plateaus into separated markers.
        peak = cv2.morphologyEx(peak, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

        nmark, markers, stats, _ = cv2.connectedComponentsWithStats(peak, 8)
        if nmark <= 1:
            # No reliable split seeds: connected components are still useful.
            labels = cv2.connectedComponentsWithStats(mask, 8)[1]
        else:
            markers = markers.astype(np.int32) + 1
            # Background marker 1.
            markers[mask == 0] = 1
            # Foreground seeds remain > 1.
            ws = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ws = cv2.cvtColor(ws, cv2.COLOR_RGB2BGR)
            cv2.watershed(ws, markers)
            labels = markers

        objects = []
        overlay = frame.copy()

        ids = np.unique(labels)
        for lab in ids:
            if lab <= 1:
                continue
            obj = (labels == lab).astype(np.uint8) * 255
            area = int(cv2.countNonZero(obj))
            if area < self.min_area or area > self.max_area:
                continue

            contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            circularity = (4*np.pi*area/(peri*peri)) if peri > 0 else 0.0

            # Keep shape filter intentionally permissive: grains can be long,
            # angular, irregular, or nearly round.
            if circularity < 0.08:
                continue

            M = cv2.moments(c)
            if M["m00"]:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                x,y,w,h = cv2.boundingRect(c)
                cx, cy = x+w//2, y+h//2

            objects.append((obj, (cx, cy), area))

        # Pink visualization for every accepted object.
        for obj, center, _ in objects:
            overlay[obj > 0] = PINK
            cv2.circle(overlay, center, 5, (255,255,255), -1, cv2.LINE_AA)

        # Alpha blend makes boundaries visible while retaining camera detail.
        result = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
        for idx, (_, (cx, cy), _) in enumerate(objects, 1):
            cv2.putText(result, str(idx), (cx+7, cy-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(result, str(idx), (cx+7, cy-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,20,20), 1, cv2.LINE_AA)

        return objects, mask, result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GrainCounter — Contagem Inteligente")
        self.resize(1250, 760)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.stable_counts = []
        self.current_count = 0

        self.camera = QComboBox()
        self.refresh_btn = QPushButton("↻ Atualizar")
        self.start_btn = QPushButton("Iniciar câmera")
        self.stop_btn = QPushButton("Parar")
        self.calibrate_btn = QPushButton("✨ Calibrar automaticamente")
        self.stop_btn.setEnabled(False)

        self.min_area = QSpinBox()
        self.min_area.setRange(1, 1_000_000)
        self.min_area.setValue(80)
        self.max_area = QSpinBox()
        self.max_area.setRange(10, 5_000_000)
        self.max_area.setValue(50000)

        self.threshold = QSpinBox()
        self.threshold.setRange(1, 150)
        self.threshold.setValue(28)

        self.peak = QDoubleSpinBox()
        self.peak.setRange(0.05, 0.95)
        self.peak.setSingleStep(0.01)
        self.peak.setValue(0.28)

        self.show_mask = QCheckBox("Mostrar máscara")
        self.show_mask.setChecked(False)

        self.count_label = QLabel("0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet(
            "font-size:52px;font-weight:700;color:#ff00dc;"
            "background:#16161b;border-radius:18px;padding:16px;"
        )
        self.status = QLabel("Pronto. Posicione a câmera perpendicularmente à superfície.")
        self.status.setWordWrap(True)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(800, 560)
        self.preview.setStyleSheet("background:#101014;border-radius:12px;")

        self.build_ui()
        self.refresh_cameras()

        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.calibrate_btn.clicked.connect(self.calibrate_camera)

        for w in [self.min_area, self.max_area, self.threshold, self.peak]:
            w.valueChanged.connect(self.apply_settings)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        controls = QGroupBox("Câmera e processamento")
        grid = QGridLayout(controls)
        grid.addWidget(QLabel("Câmera:"), 0, 0)
        grid.addWidget(self.camera, 0, 1)
        grid.addWidget(self.refresh_btn, 0, 2)
        grid.addWidget(self.start_btn, 1, 0, 1, 2)
        grid.addWidget(self.stop_btn, 1, 2)
        grid.addWidget(self.calibrate_btn, 2, 0, 1, 3)
        grid.addWidget(QLabel("Área mínima:"), 2, 0)
        grid.addWidget(self.min_area, 2, 1)
        grid.addWidget(QLabel("Área máxima:"), 3, 0)
        grid.addWidget(self.max_area, 3, 1)
        grid.addWidget(QLabel("Sensibilidade:"), 4, 0)
        grid.addWidget(self.threshold, 4, 1)
        grid.addWidget(QLabel("Separação (picos):"), 5, 0)
        grid.addWidget(self.peak, 5, 1)
        grid.addWidget(self.show_mask, 6, 0, 1, 3)

        count_box = QGroupBox("CONTAGEM ATUAL")
        cb = QVBoxLayout(count_box)
        cb.addWidget(self.count_label)
        cb.addWidget(QLabel("Objetos identificados na superfície"), alignment=Qt.AlignCenter)

        right = QVBoxLayout()
        right.addWidget(controls)
        right.addWidget(count_box)
        right.addWidget(self.status)
        right.addStretch()

        main = QHBoxLayout(central)
        main.addWidget(self.preview, 4)
        main.addLayout(right, 1)

    def apply_settings(self):
        self.engine.min_area = self.min_area.value()
        self.engine.max_area = self.max_area.value()
        self.engine.threshold = self.threshold.value()
        self.engine.min_peak = self.peak.value()

    @property
    def engine(self):
        if not hasattr(self, "_engine"):
            self._engine = CounterEngine()
        return self._engine

    def refresh_cameras(self):
        was_running = self.cap is not None
        if was_running:
            self.stop_camera()
        self.camera.clear()
        cams = list_cameras()
        for i in cams:
            self.camera.addItem(f"Câmera {i}", i)
        if cams:
            self.status.setText(f"{len(cams)} câmera(s) encontrada(s).")
        else:
            self.status.setText("Nenhuma câmera encontrada. Conecte a webcam e clique em Atualizar.")

    def calibrate_camera(self):
        if self.cap is None:
            QMessageBox.information(self, "Calibração",
                                    "Inicie a câmera e deixe alguns objetos representativos na superfície.")
            return
        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.warning(self, "Calibração", "Não foi possível capturar um frame.")
            return
        th, amin, amax = self.engine.calibrate(frame)
        self.threshold.setValue(th)
        self.min_area.setValue(amin)
        self.max_area.setValue(amax)
        self.status.setText(
            f"Calibração concluída — sensibilidade {th}, área {amin}–{amax} px²."
        )

    def start_camera(self):
        if self.camera.count() == 0:
            QMessageBox.warning(self, "Câmera", "Nenhuma câmera disponível.")
            return
        idx = int(self.camera.currentData())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            QMessageBox.critical(self, "Erro", "Não foi possível abrir a câmera selecionada.")
            return

        self.stable_counts.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera.setEnabled(False)
        self.timer.start(33)
        self.status.setText("Processando em tempo real…")

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera.setEnabled(True)

    def update_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status.setText("Falha ao capturar frame.")
            return

        objects, mask, output = self.engine.segment(frame)
        count = len(objects)

        # Static-object stabilization: median over recent frames prevents a
        # single noisy frame from changing the displayed result.
        self.stable_counts.append(count)
        self.stable_counts = self.stable_counts[-9:]
        stable = int(round(float(np.median(self.stable_counts))))

        self.current_count = stable
        self.count_label.setText(str(stable))

        if self.show_mask.isChecked():
            output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pix)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
