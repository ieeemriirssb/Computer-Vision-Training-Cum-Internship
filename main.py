"""
main.py
Edge-AI Surveillance Pipeline — single-file dashboard.

Pipeline:
  Camera / RTSP / Video file
      -> OpenCV VideoCapture (frame grab)
      -> YOLOv8n inference (local, CPU or GPU)
      -> Draw bounding boxes + confidence
      -> Threat check -> cooldown -> alert_db.log_detection()
      -> FPS / latency overlay
      -> Show window

Run:
    python main.py
Quit:
    press 'q' in the video window
"""

import time
import cv2
from ultralytics import YOLO

import config
from alert_db import init_db, log_detection, AlertCooldown, get_recent_logs


def draw_overlay(frame, fps, latency_ms, alert_flash):
    """Draw the performance metrics + status strip on top of the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (150, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    status_text = "STATUS: LIVE"
    status_color = (0, 255, 0)
    if alert_flash:
        status_text = "STATUS: THREAT DETECTED"
        status_color = (0, 0, 255)
    cv2.putText(frame, status_text, (w - 320, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # Red border flash on active alert — visible even from across a room
    if alert_flash:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

    return frame


def draw_detections(frame, boxes, names):
    """Draw bounding boxes + class label + confidence for every detection."""
    threat_hits = []  # list of (class_name, confidence) that count as threats this frame

    for box in boxes:
        conf = float(box.conf[0])
        if conf < config.CONFIDENCE_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        is_threat = cls_name in config.THREAT_CLASSES
        color = (0, 0, 255) if is_threat else (0, 200, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf*100:.0f}%"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if is_threat:
            threat_hits.append((cls_name, conf))

    return frame, threat_hits


def main():
    import sys
    print(f"[init] Python interpreter in use: {sys.executable}")

    print("[init] Setting up database...")
    init_db()

    print("[init] Loading YOLO model (first run will auto-download weights)...")
    model = YOLO(config.MODEL_PATH)
    names = model.names

    print(f"[init] Opening video source: {config.VIDEO_SOURCE}")
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("[error] Could not open video source. Check config.VIDEO_SOURCE.")
        return

    cooldown = AlertCooldown(cooldown_seconds=config.ALERT_COOLDOWN_SECONDS)

    prev_time = time.time()
    fps = 0.0
    alert_flash_until = 0  # timestamp until which the red flash stays on screen

    print("[run] Press 'q' in the video window to quit.\n")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[warn] Frame grab failed — end of stream or camera disconnected.")
            break

        # ---- Inference ----
        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes

        frame, threat_hits = draw_detections(frame, boxes, names)

        # ---- Threat decision + alert pipeline ----
        alert_fired_this_frame = False
        for cls_name, conf in threat_hits:
            if cooldown.should_alert(cls_name):
                alert_type = "HIGH_PRIORITY" if cls_name == "person" else "ALERT"
                log_detection(frame.copy(), cls_name, conf, alert_type=alert_type)
                alert_fired_this_frame = True
                print(f"[ALERT] {cls_name} ({conf*100:.1f}%) logged + notified.")

        if alert_fired_this_frame:
            alert_flash_until = time.time() + 1.5  # keep the red flash visible ~1.5s

        # ---- Performance metrics ----
        now = time.time()
        latency_ms = (now - frame_start) * 1000.0
        instant_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        fps = (fps * 0.9) + (instant_fps * 0.1)  # smoothed FPS
        prev_time = now

        frame = draw_overlay(frame, fps, latency_ms, alert_flash=(now < alert_flash_until))

        cv2.imshow(config.WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[summary] Last 10 detections logged this session (or earlier):")
    for row in get_recent_logs(10):
        print("  ", row)


if __name__ == "__main__":
    main()
