"""
config.py
All the "knobs" for the project live here so you don't have to
dig through main.py to tweak behaviour for your demo.
"""

# ---- Video source ----
# 0 = default webcam
# "video.mp4" = a local video file (good for a reliable demo if no camera available)
# "rtsp://user:pass@ip:port/stream" = an IP camera / CCTV RTSP feed
VIDEO_SOURCE = 0

# ---- Model ----
MODEL_PATH = "models/yolov8n.pt"   # auto-downloads on first run if not present
CONFIDENCE_THRESHOLD = 0.45

# ---- Threat classes ----
# COCO class names that should trigger a HIGH priority alert.
# (COCO has no "drone" class — add a custom-trained model later if you need that.)
THREAT_CLASSES = {"person", "car", "truck", "motorcycle", "bicycle"}

# ---- Alerting ----
ALERT_COOLDOWN_SECONDS = 5   # min seconds between repeat alerts for the same class

# ---- Display ----
WINDOW_NAME = "Edge-AI Surveillance Dashboard"
SHOW_FPS_OVERLAY = True
