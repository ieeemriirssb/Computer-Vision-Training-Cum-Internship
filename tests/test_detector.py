import unittest

from app.detection import ThreatDetector


class ThreatDetectorTests(unittest.TestCase):
    def test_detects_on_periodic_frames(self):
        detector = ThreatDetector()

        detections = detector.detect_frame(0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].object_class, "drone")

        detections = detector.detect_frame(1)
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
