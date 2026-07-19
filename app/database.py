import sqlite3
from typing import List, Dict, Any


class DetectionStore:
    def __init__(self, db_path: str = "detections.sqlite"):
        self.db_path = db_path

    def init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    object_class TEXT NOT NULL,
                    confidence_score REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def log_detection(self, object_class: str, confidence_score: float, timestamp: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO detections (timestamp, object_class, confidence_score) VALUES (?, ?, ?)",
                (timestamp, object_class, confidence_score),
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT timestamp, object_class, confidence_score FROM detections ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {"timestamp": ts, "object_class": obj, "confidence_score": conf}
            for ts, obj, conf in rows
        ]
