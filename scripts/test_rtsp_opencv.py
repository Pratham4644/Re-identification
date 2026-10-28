import os
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.camera_config import resolve_single_camera_stream

OUTPUT_DIR = PROJECT_ROOT / "tests" / "rtsp_test_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

camera_id, rtsp_url = resolve_single_camera_stream(PROJECT_ROOT)

if not rtsp_url:
    raise RuntimeError(
        "No camera URL configured. Set CAMERA_URL / CAMERA_RTSP or "
        "config/cameras.json."
    )

output_path = OUTPUT_DIR / f"opencv_{camera_id.replace('/', '_')}_frame.jpg"

print("OpenCV version:", cv2.__version__)
print(f"Opening {camera_id}...")

params = [
    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
]

cap = cv2.VideoCapture()

try:
    opened = cap.open(rtsp_url, cv2.CAP_FFMPEG, params)

    if not opened:
        print("FAILED: OpenCV could not open the stream.")
        raise SystemExit(1)

    print("Stream opened. Waiting for a frame...")

    frame = None

    for attempt in range(1, 11):
        success, candidate = cap.read()

        if success and candidate is not None:
            frame = candidate
            print(f"Frame received on attempt {attempt}")
            break

        print(f"Frame read failed: {attempt}")
        time.sleep(0.5)

    if frame is None:
        print("FAILED: OpenCV opened the stream but received no frames.")
        raise SystemExit(2)

    print("Frame dimensions:", frame.shape[1], "x", frame.shape[0])

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError("Could not save the OpenCV frame.")

    print("SUCCESS: Frame saved to:", output_path)

finally:
    cap.release()