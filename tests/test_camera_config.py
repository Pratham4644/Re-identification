import json
import os
import tempfile
import unittest
from pathlib import Path

from src.camera.camera_config import (
    _parse_cameras_env,
    get_camera_url,
    load_camera_urls,
)


class CameraConfigTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_load_from_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "cameras.json").write_text(
                json.dumps(
                    {
                        "cameras": [
                            {
                                "id": "CAM-A",
                                "url_env": "CAMERA_A_URL",
                                "fallback_env": "CAMERA_A_RTSP",
                            },
                            {"id": "CAM-B", "url": "rtsp://direct/b"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            os.environ["CAMERA_A_URL"] = "rtsp://from-env/a"
            cameras = load_camera_urls(root)
            self.assertEqual(
                cameras,
                {"CAM-A": "rtsp://from-env/a", "CAM-B": "rtsp://direct/b"},
            )

    def test_cameras_json_env_overrides_same_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "cameras.json").write_text(
                json.dumps({"cameras": [{"id": "CAM-1", "url": "rtsp://old"}]}),
                encoding="utf-8",
            )
            os.environ["CAMERAS_JSON"] = json.dumps(
                [{"id": "CAM-1", "url": "rtsp://new"}, {"id": "CAM-2", "url": "rtsp://two"}]
            )
            cameras = load_camera_urls(root)
            self.assertEqual(
                cameras,
                {"CAM-1": "rtsp://new", "CAM-2": "rtsp://two"},
            )

    def test_cameras_env_string(self):
        parsed = _parse_cameras_env("CAM-X|rtsp://x; CAM-Y | rtsp://y ")
        self.assertEqual(parsed, {"CAM-X": "rtsp://x", "CAM-Y": "rtsp://y"})

    def test_get_camera_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "config" / "cameras.json").write_text(
                json.dumps({"cameras": [{"id": "DESK-1", "url": "http://cam1/video"}]}),
                encoding="utf-8",
            )
            self.assertEqual(
                get_camera_url("DESK-1", root),
                "http://cam1/video",
            )
            self.assertIsNone(get_camera_url("MISSING", root))


if __name__ == "__main__":
    unittest.main()
