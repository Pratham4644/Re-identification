import os
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "live_tracking_output"
LOG_PATH = OUTPUT_DIR / "tracking_events.csv"

CAMERA_ID = "CAM-202"
CAMERA_ENV = "CAMERA_202_RTSP"

CONFIDENCE = 0.35
IMAGE_SIZE = 640

# Log a track when first seen, then periodically while visible.
LOG_INTERVAL_SECONDS = 5.0

# Set to True to save an annotated video as well.
SAVE_VIDEO = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def create_csv_if_needed():
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "camera_id",
                "track_id",
                "event",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ])


def log_track(track_id, confidence, bbox, event="visible"):
    timestamp = datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )

    x1, y1, x2, y2 = bbox

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            CAMERA_ID,
            track_id,
            event,
            f"{confidence:.4f}",
            x1,
            y1,
            x2,
            y2,
        ])


# ============================================================
# Main
# ============================================================

def main():
    rtsp_url = os.environ.get(CAMERA_ENV)

    if not rtsp_url:
        raise RuntimeError(
            f"{CAMERA_ENV} is not set in this PowerShell session."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    create_csv_if_needed()

    print("=" * 60)
    print("LIVE CCTV PERSON TRACKING")
    print("=" * 60)
    print(f"Camera: {CAMERA_ID}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Loading YOLO model on CPU...")

    model = YOLO(str(MODEL_PATH))

    print("Connecting to CCTV stream...")

    cap = cv2.VideoCapture()

    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
    ]

    if not cap.open(rtsp_url, cv2.CAP_FFMPEG, params):
        cap.release()
        raise RuntimeError(
            f"Could not open {CAMERA_ID} RTSP stream."
        )

    print("Camera connected.")
    print("Press Q in the video window to stop.")
    print("Tracking events are being written to CSV.")
    print()

    window_name = f"{CAMERA_ID} - Live Person Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_logged = {}
    last_seen = {}
    previous_time = time.monotonic()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            success, frame = cap.read()

            if not success or frame is None:
                print("Frame read failed. Reconnecting...")

                cap.release()
                time.sleep(1)

                cap = cv2.VideoCapture()

                if not cap.open(rtsp_url, cv2.CAP_FFMPEG, params):
                    print("Reconnect failed. Retrying...")
                    time.sleep(2)
                    continue

                continue

            now = time.monotonic()
            elapsed = now - previous_time
            previous_time = now

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (
                    0.9 * fps + 0.1 * instant_fps
                )

            frame_count += 1

            # Detect and track people. Class 0 is the person class.
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )

            result = results[0]
            annotated = frame.copy()
            visible_ids = set()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()

                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(xyxy)

                for bbox, confidence, track_id in zip(
                    xyxy, confidences, track_ids
                ):
                    x1, y1, x2, y2 = [
                        int(value) for value in bbox
                    ]

                    if track_id is None:
                        label = f"Person | {confidence:.2f}"
                        color = (0, 165, 255)
                    else:
                        track_id = int(track_id)
                        visible_ids.add(track_id)
                        last_seen[track_id] = now

                        label = (
                            f"Track {track_id} | {confidence:.2f}"
                        )
                        color = (0, 220, 0)

                        # Log first sighting, then periodically.
                        if (
                            track_id not in last_logged
                            or now - last_logged[track_id]
                            >= LOG_INTERVAL_SECONDS
                        ):
                            log_track(
                                track_id,
                                float(confidence),
                                (x1, y1, x2, y2),
                            )
                            last_logged[track_id] = now

                            timestamp = datetime.now().strftime(
                                "%H:%M:%S"
                            )
                            print(
                                f"[{timestamp}] "
                                f"{CAMERA_ID} | "
                                f"Track {track_id} | "
                                f"confidence={confidence:.2f}"
                            )

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # Display diagnostic information.
            cv2.putText(
                annotated,
                f"Camera: {CAMERA_ID}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} | Visible tracks: {len(visible_ids)}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("TRACKING STOPPED")
    print(f"CSV log: {LOG_PATH}")
    print(f"Frames processed: {frame_count}")
    print(f"Approximate processing FPS: {fps:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()