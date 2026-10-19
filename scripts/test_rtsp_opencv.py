import os
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "rtsp_test_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rtsp_url = os.environ.get("CAMERA_202_RTSP")

if not rtsp_url:
    raise RuntimeError("CAMERA_202_RTSP environment variable is not set.")

output_path = OUTPUT_DIR / "opencv_camera_202_frame.jpg"

print("OpenCV version:", cv2.__version__)
print("Opening CAM-202...")

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