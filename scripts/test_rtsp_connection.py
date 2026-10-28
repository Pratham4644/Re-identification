import os
import sys
import time
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera.camera_config import resolve_single_camera_stream


def main():
    camera_id, rtsp_url = resolve_single_camera_stream(ROOT)

    if not rtsp_url:
        raise RuntimeError(
            "No camera URL configured. Set CAMERA_URL / CAMERA_RTSP or "
            "config/cameras.json."
        )

    print(f"Opening {camera_id} stream...")

    # Use FFmpeg for RTSP transport through OpenCV.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

    cap.open(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the RTSP stream. "
            "Check the URL, credentials, network, and camera permissions."
        )

    print("Stream opened. Reading frames...")

    successful_frames = 0
    failed_reads = 0
    start_time = time.time()
    output_dir = Path("tests") / "rtsp_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while successful_frames < 30:
            ok, frame = cap.read()

            if not ok or frame is None:
                failed_reads += 1
                print(f"Frame read failed: {failed_reads}")

                if failed_reads >= 5:
                    print("Too many consecutive read failures.")
                    break

                time.sleep(0.2)
                continue

            failed_reads = 0
            successful_frames += 1

            if successful_frames == 1:
                print(
                    f"Frame dimensions: "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )

            if successful_frames == 30:
                output_path = output_dir / "camera_202_frame.jpg"

                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(
                        f"Failed to save frame: {output_path}"
                    )

                print(f"Saved test frame: {output_path}")

    finally:
        cap.release()

    elapsed = time.time() - start_time

    print("\nRTSP TEST SUMMARY")
    print(f"Successful frames: {successful_frames}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Output folder: {output_dir.resolve()}")

    if successful_frames == 30:
        print("SUCCESS: Received 30 frames from CAM-202.")
    else:
        print("INCOMPLETE: Could not read all 30 frames.")


if __name__ == "__main__":
    main()