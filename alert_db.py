"""
alert_db.py
Handles everything that happens the instant a threat is detected:
  1. Write a structured record to a local SQLite database
  2. Fire an OS-level desktop notification
  3. Play an audible alarm sound
  4. Save a snapshot image of the offending frame

All three "response" actions run in the same call so the whole
alert pipeline stays inside the sub-second latency budget.
"""

import os
import sqlite3
import time
import threading
from datetime import datetime

import cv2

try:
    from plyer import notification
    NOTIFICATIONS_AVAILABLE = True
except Exception:
    NOTIFICATIONS_AVAILABLE = False

try:
    from playsound import playsound
    AUDIO_AVAILABLE = True
except Exception as e:
    AUDIO_AVAILABLE = False
    print(f"[startup] playsound not available — alarm sound disabled. Reason: {e}")


# Use the script's own folder as the base, NOT the current working directory.
# VS Code (Run button / debugger) sometimes launches with a different cwd than
# a plain terminal does, which silently breaks relative paths like "sounds/alarm.wav".
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "database", "logs.db")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "sounds", "alarm.wav")


def init_db():
    """Create the database + table if they don't already exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            object_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            snapshot_path TEXT,
            alert_type TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _play_alarm_async():
    """Play the alarm sound on a background thread so it never blocks the video loop."""
    if not AUDIO_AVAILABLE:
        return

    def _play():
        try:
            if os.path.exists(ALARM_SOUND_PATH):
                playsound(ALARM_SOUND_PATH)
            else:
                print(f"[alarm] sound file not found at: {ALARM_SOUND_PATH}")
        except Exception as e:
            print(f"[alarm] could not play sound: {e}")

    threading.Thread(target=_play, daemon=True).start()


def _notify_async(title, message):
    """Fire a desktop notification on a background thread."""
    if not NOTIFICATIONS_AVAILABLE:
        return

    def _send():
        try:
            notification.notify(title=title, message=message, timeout=4)
        except Exception as e:
            print(f"[notify] could not send notification: {e}")

    threading.Thread(target=_send, daemon=True).start()


def save_snapshot(frame):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SNAPSHOT_DIR, f"threat_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def log_detection(frame, object_class, confidence, alert_type="THREAT"):
    """
    Full alert pipeline for a single detected threat:
    snapshot -> db insert -> notification -> alarm sound.
    Returns the row that was written, useful for the on-screen alert history.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_path = save_snapshot(frame)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO detection_log
           (timestamp, object_class, confidence, snapshot_path, alert_type)
           VALUES (?, ?, ?, ?, ?)""",
        (timestamp, object_class, confidence, snapshot_path, alert_type),
    )
    conn.commit()
    conn.close()

    _notify_async(
        title=f"⚠ {alert_type}: {object_class.upper()} detected",
        message=f"Confidence {confidence*100:.1f}% at {timestamp}",
    )
    _play_alarm_async()

    return {
        "timestamp": timestamp,
        "object_class": object_class,
        "confidence": confidence,
        "snapshot_path": snapshot_path,
        "alert_type": alert_type,
    }


def get_recent_logs(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, object_class, confidence, alert_type "
        "FROM detection_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


class AlertCooldown:
    """
    Prevents alert spam. Without this, a person standing in frame for
    10 seconds at 20 FPS would create ~200 log rows and 200 popups.
    Only allows one alert per object class every `cooldown_seconds`.
    """

    def __init__(self, cooldown_seconds=5):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time = {}

    def should_alert(self, object_class):
        now = time.time()
        last = self._last_alert_time.get(object_class, 0)
        if now - last >= self.cooldown_seconds:
            self._last_alert_time[object_class] = now
            return True
        return False
