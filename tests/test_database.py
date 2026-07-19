import os
import tempfile
import unittest

from app.database import DetectionStore


class DetectionStoreTests(unittest.TestCase):
    def test_logs_and_reads_detections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "detections.sqlite")
            store = DetectionStore(db_path)
            store.init_db()

            store.log_detection("drone", 0.91, "2026-07-14T12:00:00")
            rows = store.get_recent_detections(5)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["object_class"], "drone")
            self.assertAlmostEqual(rows[0]["confidence_score"], 0.91)


if __name__ == "__main__":
    unittest.main()
