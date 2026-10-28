"""Face-gallery scoring tests. Run after copying into the project tests folder."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "recognition" / "face_recognizer.py"
spec = importlib.util.spec_from_file_location("face_recognizer_under_test", MODULE_PATH)
face_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face_module)
FaceRecognizer = face_module.FaceRecognizer


class FaceRecognizerScoringTests(unittest.TestCase):
    def setUp(self):
        # Avoid loading InsightFace; these tests exercise pure NumPy ranking.
        self.recognizer = FaceRecognizer.__new__(FaceRecognizer)

    def test_best_reference_per_identity(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        gallery = {
            "Alice": [
                np.array([0.8, 0.6], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            ],
            "Bob": [np.array([0.0, 1.0], dtype=np.float32)],
        }
        ranked = self.recognizer.rank_candidates(query, gallery)
        self.assertEqual(ranked[0]["name"], "Alice")
        self.assertAlmostEqual(ranked[0]["similarity"], 1.0, places=6)
        self.assertEqual(ranked[1]["name"], "Bob")
        self.assertAlmostEqual(ranked[1]["similarity"], 0.0, places=6)

    def test_query_is_normalized(self):
        ranked = self.recognizer.rank_candidates(
            np.array([10.0, 0.0], dtype=np.float32),
            {"Alice": [np.array([1.0, 0.0], dtype=np.float32)]},
        )
        self.assertAlmostEqual(ranked[0]["similarity"], 1.0, places=6)

    def test_empty_gallery_returns_empty_ranking(self):
        self.assertEqual(
            self.recognizer.rank_candidates(
                np.array([1.0, 0.0], dtype=np.float32), {}
            ),
            [],
        )

    def test_invalid_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.recognizer.rank_candidates(
                np.array([0.0, 0.0], dtype=np.float32),
                {"Alice": [np.array([1.0, 0.0], dtype=np.float32)]},
            )


if __name__ == "__main__":
    unittest.main()
