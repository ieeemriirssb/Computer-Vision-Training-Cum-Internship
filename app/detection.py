from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    object_class: str
    confidence: float


class ThreatDetector:
    def __init__(self):
        self._threat_classes = {"drone", "intruder", "vehicle"}

    def detect_frame(self, frame_id: int) -> List[Detection]:
        if frame_id % 5 == 0:
            return [Detection(object_class="drone", confidence=0.93)]
        return []

    def is_threat(self, detection: Detection) -> bool:
        return detection.object_class in self._threat_classes


