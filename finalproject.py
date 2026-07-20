"""
Edge-AI Surveillance Pipeline

"""

import sys
import os
import time
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QFormLayout, QMessageBox, QHeaderView, QDoubleSpinBox,
)

# Desktop notifications (optional dependency; degrades gracefully if absent)
try:
    from plyer import notification as desktop_notify
    HAVE_PLYER = True
except ImportError:
    HAVE_PLYER = False

# Cross-platform beep fallback for the audible alarm
try:
    import simpleaudio as sa
    HAVE_SIMPLEAUDIO = True
except ImportError:
    HAVE_SIMPLEAUDIO = False

# YOLO inference engine
try:
    from ultralytics import YOLO
    HAVE_ULTRALYTICS = True
except ImportError:
    HAVE_ULTRALYTICS = False


@dataclass
class Config:
    model_path: str = "yolov8n.pt"          
    confidence_threshold: float = 0.5
    device: str = "cpu"                    

    threat_classes: Tuple[str, ...] = ("person", "car", "truck", "drone")
    high_priority_classes: Tuple[str, ...] = ("drone",)

    after_hours_start: dtime = dtime(20, 0)   # 8 PM
    after_hours_end: dtime = dtime(6, 0)      # 6 AM
    # A detection whose CENTRE falls inside this polygon always alerts.
    restricted_zone: Tuple[Tuple[float, float], ...] = (
        (0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)
    )

    alert_cooldown_seconds: float = 5.0  
    snapshot_dir: str = "snapshots"
    db_path: str = "database/detections.db"
    alarm_sound_path: str = "sounds/alarm.wav"  

CONFIG = Config()
class DetectionDB:

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Detection_Log (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Timestamp TEXT NOT NULL,
                    Object TEXT NOT NULL,
                    Confidence REAL NOT NULL,
                    Snapshot_Path TEXT,
                    Alert_Type TEXT NOT NULL
                )
                """
            )
            self.conn.commit()

    def insert(self, obj: str, confidence: float, snapshot_path: str, alert_type: str) -> int:
        ts = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO Detection_Log (Timestamp, Object, Confidence, Snapshot_Path, Alert_Type) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, obj, confidence, snapshot_path, alert_type),
            )
            self.conn.commit()
            return cur.lastrowid

    def recent(self, limit: int = 200) -> List[tuple]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT ID, Timestamp, Object, Confidence, Snapshot_Path, Alert_Type "
                "FROM Detection_Log ORDER BY ID DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    def close(self):
        with self._lock:
            self.conn.close()




class AlertManager:
    """Handles desktop notification + audible alarm, with per-class cooldown."""

    def __init__(self, cooldown_seconds: float, sound_path: str):
        self.cooldown_seconds = cooldown_seconds
        self.sound_path = sound_path
        self._last_alert_time = {}
        self._lock = threading.Lock()

    def should_alert(self, obj_class: str) -> bool:
        now = time.monotonic()
        with self._lock:
            last = self._last_alert_time.get(obj_class, 0.0)
            if now - last >= self.cooldown_seconds:
                self._last_alert_time[obj_class] = now
                return True
        return False

    def fire(self, obj_class: str, confidence: float, alert_type: str):
        threading.Thread(
            target=self._fire_impl, args=(obj_class, confidence, alert_type), daemon=True
        ).start()

    def _fire_impl(self, obj_class: str, confidence: float, alert_type: str):
        title = f"THREAT DETECTED: {obj_class.upper()}"
        message = f"{alert_type} — confidence {confidence:.0%}"

        if HAVE_PLYER:
            try:
                desktop_notify.notify(title=title, message=message, timeout=4)
            except Exception:
                pass  # never let a notification failure break the pipeline

        self._play_alarm()

    def _play_alarm(self):
        try:
            if os.path.exists(self.sound_path) and HAVE_SIMPLEAUDIO:
                wave_obj = sa.WaveObject.from_wave_file(self.sound_path)
                wave_obj.play()
                return
        except Exception:
            pass
        # Fallback: OS-level beep, best-effort, never fatal
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(1500, 300)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass



def is_after_hours(cfg: Config, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    t = now.time()
    start, end = cfg.after_hours_start, cfg.after_hours_end
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end  # window wraps past midnight


def point_in_polygon(x: float, y: float, polygon: Tuple[Tuple[float, float], ...]) -> bool:
    n = len(polygon)
    inside = False
    px, py = polygon[-1]
    for cx, cy in polygon:
        if ((cy > y) != (py > y)) and (
            x < (py - cy) * (px - cx) / (py - cy + 1e-12) + cx
        ):
            inside = not inside
        px, py = cx, cy
    return inside


def classify_alert(cfg: Config, cls_name: str, cx_norm: float, cy_norm: float) -> Optional[str]:
    """Returns an alert-type string if this detection should trigger an
    alert, or None if it's informational only."""
    if cls_name not in cfg.threat_classes:
        return None
    if cls_name in cfg.high_priority_classes:
        return "HIGH_PRIORITY"
    if point_in_polygon(cx_norm, cy_norm, cfg.restricted_zone):
        return "RESTRICTED_ZONE"
    if cls_name == "person" and is_after_hours(cfg):
        return "AFTER_HOURS_PERSON"
    return None



# VIDEO / INFERENCE THREAD


@dataclass
class FrameResult:
    frame: np.ndarray
    fps: float
    latency_ms: float
    detections: List[dict] = field(default_factory=list)


class InferenceThread(QThread):
    frame_ready = pyqtSignal(object)         
    alert_fired = pyqtSignal(str, float, str)  
    error = pyqtSignal(str)

    def __init__(self, source, cfg: Config, db: DetectionDB, alerts: AlertManager):
        super().__init__()
        self.source = source
        self.cfg = cfg
        self.db = db
        self.alerts = alerts
        self._running = False
        self.model = None

    def run(self):
        if not HAVE_ULTRALYTICS:
            self.error.emit(
                "The 'ultralytics' package is not installed. Run: pip install ultralytics"
            )
            return
        try:
            self.model = YOLO(self.cfg.model_path)
        except Exception as e:
            self.error.emit(f"Failed to load model '{self.cfg.model_path}': {e}")
            return

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error.emit(f"Could not open video source: {self.source}")
            return

        os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
        self._running = True
        fps_smooth = 0.0

        while self._running:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            results = self.model.predict(
                frame,
                conf=self.cfg.confidence_threshold,
                device=self.cfg.device,
                verbose=False,
            )
            r = results[0]
            h, w = frame.shape[:2]
            detections = []

            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx_norm, cy_norm = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h

                alert_type = classify_alert(self.cfg, cls_name, cx_norm, cy_norm)
                detections.append(dict(
                    cls=cls_name, conf=confidence, box=(x1, y1, x2, y2),
                    alert_type=alert_type,
                ))

                color = (0, 0, 255) if alert_type else (0, 200, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{cls_name} {confidence:.0%}"
                cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if alert_type and self.alerts.should_alert(cls_name):
                    snapshot_path = os.path.join(
                        self.cfg.snapshot_dir,
                        f"{cls_name}_{int(time.time()*1000)}.jpg",
                    )
                    cv2.imwrite(snapshot_path, frame)
                    self.db.insert(cls_name, confidence, snapshot_path, alert_type)
                    self.alerts.fire(cls_name, confidence, alert_type)
                    self.alert_fired.emit(cls_name, confidence, alert_type)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            inst_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            fps_smooth = inst_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst_fps)

            self.frame_ready.emit(FrameResult(
                frame=frame, fps=fps_smooth, latency_ms=latency_ms, detections=detections
            ))

        cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)



# DASHBOARD (PyQt6)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge-AI Surveillance — Offline Threat Detection")
        self.resize(1200, 750)

        self.cfg = CONFIG
        self.db = DetectionDB(self.cfg.db_path)
        self.alerts = AlertManager(self.cfg.alert_cooldown_seconds, self.cfg.alarm_sound_path)
        self.inference_thread: Optional[InferenceThread] = None

        self._build_ui()
        self._refresh_log_table()

        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._clear_flash)

    # -- UI construction
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: video + controls + metrics
        left = QVBoxLayout()

        controls = QGroupBox("Source")
        form = QFormLayout(controls)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam (0)", "Video file...", "RTSP URL..."])
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("path/to/video.mp4 or rtsp://...")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.cfg.confidence_threshold)
        form.addRow("Source type:", self.source_combo)
        form.addRow("Path / URL:", self.source_input)
        form.addRow("Confidence:", self.conf_spin)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)

        self.video_label = QLabel("No feed running")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(760, 480)
        self.video_label.setStyleSheet("background-color: #111; color: #888;")

        metrics_row = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.latency_label = QLabel("Latency: -- ms")
        for lbl in (self.fps_label, self.latency_label):
            lbl.setFont(QFont("Consolas", 11))
        metrics_row.addWidget(self.fps_label)
        metrics_row.addWidget(self.latency_label)
        metrics_row.addStretch()

        self.flash_label = QLabel("")
        self.flash_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.flash_label.setStyleSheet("color: red;")

        left.addWidget(controls)
        left.addLayout(btn_row)
        left.addWidget(self.video_label, stretch=1)
        left.addLayout(metrics_row)
        left.addWidget(self.flash_label)

        # Right: detection log
        right = QVBoxLayout()
        right.addWidget(QLabel("Detection Log"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Object", "Confidence", "Alert Type", "Snapshot"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right.addWidget(self.table)

        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=2)

    # -- Stream control ---------------------------------------------------
    def start_stream(self):
        choice = self.source_combo.currentIndex()
        if choice == 0:
            source = 0
        else:
            text = self.source_input.text().strip()
            if not text:
                QMessageBox.warning(self, "Missing source", "Enter a file path or RTSP URL.")
                return
            source = text

        self.cfg.confidence_threshold = self.conf_spin.value()

        self.inference_thread = InferenceThread(source, self.cfg, self.db, self.alerts)
        self.inference_thread.frame_ready.connect(self._on_frame)
        self.inference_thread.alert_fired.connect(self._on_alert)
        self.inference_thread.error.connect(self._on_error)
        self.inference_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_stream(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("No feed running")

    # -- Signal handlers ---------------------------------------------------
    def _on_frame(self, result: FrameResult):
        frame_rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.video_label.setPixmap(pix)
        self.fps_label.setText(f"FPS: {result.fps:.1f}")
        self.latency_label.setText(f"Latency: {result.latency_ms:.1f} ms")

    def _on_alert(self, obj_class: str, confidence: float, alert_type: str):
        self.flash_label.setText(f"⚠ {alert_type}: {obj_class} ({confidence:.0%})")
        self._flash_timer.start(2000)
        self._refresh_log_table()

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Pipeline error", message)
        self.stop_stream()

    def _clear_flash(self):
        self.flash_label.setText("")
        self._flash_timer.stop()

    def _refresh_log_table(self):
        rows = self.db.recent(limit=200)
        self.table.setRowCount(len(rows))
        for i, (rid, ts, obj, conf, snap, alert_type) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(ts))
            self.table.setItem(i, 1, QTableWidgetItem(obj))
            self.table.setItem(i, 2, QTableWidgetItem(f"{conf:.0%}"))
            self.table.setItem(i, 3, QTableWidgetItem(alert_type))
            self.table.setItem(i, 4, QTableWidgetItem(snap or ""))

    def closeEvent(self, event):
        self.stop_stream()
        self.db.close()
        super().closeEvent(event)



# ENTRY POINT

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


