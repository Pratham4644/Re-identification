import csv
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.face_quality import FaceQualityChecker

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_ID = "CAM-202"
RTSP_URL = os.environ.get("CAMERA_202_RTSP")

GALLERY_PATH = PROJECT_ROOT / "data" / "staff_gallery"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "live_recognition_output"
CSV_PATH = OUTPUT_DIR / "recognition_events.csv"

PERSON_MODEL = "yolo11n.pt"
PERSON_CONFIDENCE = 0.35
IMAGE_SIZE = 416

FACE_SIMILARITY_THRESHOLD = 0.30  # Experimental, validate before production.
MATCH_WINDOW = 8
MIN_MATCH_VOTES = 5
MIN_SIMILARITY_MARGIN = 0.05
FACE_PROCESS_INTERVAL = 1.0

PERSON_CROP_BOTTOM_RATIO = 0.65
MIN_PERSON_CROP_WIDTH = 100
MIN_PERSON_CROP_HEIGHT = 140

LOG_INTERVAL = 5.0
MIN_FACE_WIDTH = 60
MIN_FACE_HEIGHT = 60
MIN_BLUR_SCORE = 30.0
MIN_FACE_DETECTION_CONFIDENCE = 0.50
IDENTITY_TTL_SECONDS = 10.0

# Print one performance summary every N seconds.
PROFILE_INTERVAL = 10.0


def timestamp_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def create_csv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "timestamp", "camera_id", "track_id", "identity",
                "similarity", "second_best_similarity", "status",
                "quality_reason", "x1", "y1", "x2", "y2",
            ])


def write_event(track_id, identity, similarity, second_similarity,
                status, quality_reason, person_box):
    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            timestamp_now(), CAMERA_ID, track_id, identity,
            "" if similarity is None else round(similarity, 4),
            "" if second_similarity is None else round(second_similarity, 4),
            status, quality_reason, *person_box,
        ])


def print_profile(stats, wall_elapsed, interval_frames, interval_faces):
    if wall_elapsed <= 0:
        return
    def ms(key):
        return 1000.0 * stats[key] / max(interval_frames, 1)

    print(
        f"[PROFILE {wall_elapsed:5.1f}s] "
        f"FPS={interval_frames / wall_elapsed:.2f} | "
        f"YOLO={ms('yolo'):.1f} ms/frame | "
        f"face-pass total={stats['face_total'] * 1000 / max(interval_faces, 1):.1f} "
        f"ms/pass | face calls={stats['face_calls']} "
        f"({stats['face_calls'] / max(wall_elapsed, 0.001):.2f}/s) | "
        f"face detect+embed={stats['face_detect'] * 1000:.1f} ms total | "
        f"quality+match={stats['match'] * 1000:.1f} ms total | "
        f"display+misc≈{stats['misc'] * 1000 / max(interval_frames, 1):.1f} ms/frame"
    )


def main():
    if not RTSP_URL:
        raise RuntimeError("CAMERA_202_RTSP is not set in this PowerShell session.")
    if not GALLERY_PATH.is_dir():
        raise RuntimeError(f"Staff gallery does not exist: {GALLERY_PATH}")

    create_csv()
    print("=" * 65)
    print("LIVE PERSON TRACKING + STAFF FACE RECOGNITION (PROFILED)")
    print("=" * 65)
    print(f"Camera: {CAMERA_ID}")
    print(f"Device: CPU | YOLO image size: {IMAGE_SIZE}")
    print(f"Gallery: {GALLERY_PATH}")
    print(f"CSV: {CSV_PATH}")
    print("Performance summary prints every 10 seconds. Press Q to quit.\n")

    print("[1/4] Loading face recognition model...")
    face_recognizer = FaceRecognizer()

    print("[2/4] Loading staff face gallery...")
    gallery = face_recognizer.load_gallery(GALLERY_PATH)
    if not gallery:
        raise RuntimeError("No valid staff identities were loaded.")
    print("Loaded identities:", ", ".join(sorted(gallery.keys())))

    quality_checker = FaceQualityChecker(
        min_face_width=MIN_FACE_WIDTH,
        min_face_height=MIN_FACE_HEIGHT,
        min_blur_score=MIN_BLUR_SCORE,
        min_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
    )

    print("[3/4] Loading person detector...")
    person_model = YOLO(PERSON_MODEL)

    print("[4/4] Connecting to RTSP stream...")
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
    ]
    cap = cv2.VideoCapture()
    if not cap.open(RTSP_URL, cv2.CAP_FFMPEG, params):
        cap.release()
        raise RuntimeError("Could not open RTSP stream. Check camera and URL.")

    print("Camera connected. Recognition is running. Keep the RTSP URL private.\n")

    match_history = defaultdict(lambda: deque(maxlen=MATCH_WINDOW))
    track_identity = {}
    track_last_face_time = {}
    track_last_log_time = {}

    frame_count = 0
    face_pass_count = 0
    start_time = time.perf_counter()
    last_face_pass = 0.0
    last_profile_time = start_time
    last_profile_frame = 0
    last_profile_face_pass = 0

    # Timings accumulate between profile summaries.
    stats = defaultdict(float)
    total_face_calls = 0

    try:
        while True:
            loop_start = time.perf_counter()
            read_start = time.perf_counter()
            ok, frame = cap.read()
            read_time = time.perf_counter() - read_start
            if not ok or frame is None:
                print("Frame read failed. Stopping the pipeline.")
                break

            frame_count += 1
            frame_height, frame_width = frame.shape[:2]
            current_time = time.time()

            yolo_start = time.perf_counter()
            results = person_model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                classes=[0], conf=PERSON_CONFIDENCE,
                imgsz=IMAGE_SIZE, device="cpu", verbose=False,
            )
            yolo_time = time.perf_counter() - yolo_start
            stats["yolo"] += yolo_time
            stats["read"] += read_time

            people = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    for box, track_id, confidence in zip(xyxy, ids, confidences):
                        person_box = clamp_box(box, frame_width, frame_height)
                        if person_box is not None:
                            people.append({
                                "track_id": int(track_id),
                                "box": person_box,
                                "confidence": float(confidence),
                            })

            face_pass_due = current_time - last_face_pass >= FACE_PROCESS_INTERVAL
            if face_pass_due:
                last_face_pass = current_time
                face_pass_count += 1
                face_pass_start = time.perf_counter()

                for person in people:
                    track_id = person["track_id"]
                    px1, py1, px2, py2 = person["box"]
                    person_width = px2 - px1
                    person_height = py2 - py1
                    if (person_width < MIN_PERSON_CROP_WIDTH or
                            person_height < MIN_PERSON_CROP_HEIGHT):
                        continue

                    crop_y2 = py1 + int(person_height * PERSON_CROP_BOTTOM_RATIO)
                    person_crop = frame[py1:crop_y2, px1:px2]
                    if person_crop.size == 0:
                        continue

                    face_start = time.perf_counter()
                    faces = face_recognizer.app.get(person_crop)
                    face_elapsed = time.perf_counter() - face_start
                    stats["face_detect"] += face_elapsed
                    stats["face_calls"] += 1
                    total_face_calls += 1

                    # Avoid choosing arbitrarily when the crop has zero or multiple faces.
                    if len(faces) != 1:
                        continue

                    match_start = time.perf_counter()
                    face = faces[0]
                    quality = quality_checker.evaluate(person_crop, face)
                    if not quality.acceptable:
                        stats["match"] += time.perf_counter() - match_start
                        continue

                    track_last_face_time[track_id] = current_time
                    embedding = face_recognizer.normalize_embedding(face.normed_embedding)
                    ranked = face_recognizer.rank_candidates(embedding, gallery)
                    if not ranked:
                        stats["match"] += time.perf_counter() - match_start
                        continue

                    best = ranked[0]
                    best_name = best["name"]
                    best_similarity = float(best["similarity"])
                    second_similarity = (
                        float(ranked[1]["similarity"]) if len(ranked) > 1 else None
                    )
                    margin_ok = (
                        second_similarity is None or
                        best_similarity - second_similarity >= MIN_SIMILARITY_MARGIN
                    )
                    candidate_accepted = (
                        best_similarity >= FACE_SIMILARITY_THRESHOLD and margin_ok
                    )
                    match_history[track_id].append(
                        best_name if candidate_accepted else "Unknown"
                    )

                    votes = list(match_history[track_id])
                    confirmed_name = "Unknown"
                    if votes:
                        most_common_name = max(set(votes), key=votes.count)
                        if (most_common_name != "Unknown" and
                                votes.count(most_common_name) >= MIN_MATCH_VOTES):
                            confirmed_name = most_common_name

                    previous_name = track_identity.get(track_id, "Unknown")
                    track_identity[track_id] = confirmed_name
                    stats["match"] += time.perf_counter() - match_start

                    should_log = (
                        confirmed_name != previous_name or
                        current_time - track_last_log_time.get(track_id, 0.0) >= LOG_INTERVAL
                    )
                    if should_log:
                        write_event(
                            track_id, confirmed_name, best_similarity, second_similarity,
                            "confirmed" if confirmed_name != "Unknown" else "unknown_or_unconfirmed",
                            quality.reason, person["box"],
                        )
                        print(
                            f"[{timestamp_now()}] Track {track_id} | "
                            f"Identity: {confirmed_name} | Similarity: {best_similarity:.3f} | "
                            f"Margin OK: {margin_ok}"
                        )
                        track_last_log_time[track_id] = current_time

                stats["face_total"] += time.perf_counter() - face_pass_start

            # Expire identity if no acceptable face has been seen recently.
            for track_id, last_seen in list(track_last_face_time.items()):
                if current_time - last_seen > IDENTITY_TTL_SECONDS:
                    if track_identity.get(track_id) != "Unknown":
                        track_identity[track_id] = "Unknown"
                        match_history[track_id].clear()

            misc_start = time.perf_counter()
            for person in people:
                track_id = person["track_id"]
                x1, y1, x2, y2 = person["box"]
                identity = track_identity.get(track_id, "Unknown")
                label = f"Track {track_id} | {identity}"
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0) if identity != "Unknown" else (0, 165, 255), 2,
                )
                cv2.putText(
                    frame, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
                )

            elapsed = max(time.perf_counter() - start_time, 0.001)
            processing_fps = frame_count / elapsed
            cv2.putText(
                frame, f"Camera: {CAMERA_ID} | FPS: {processing_fps:.1f}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("Live Staff Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            stats["misc"] += time.perf_counter() - misc_start

            now = time.perf_counter()
            if now - last_profile_time >= PROFILE_INTERVAL:
                interval_elapsed = now - last_profile_time
                interval_frames = frame_count - last_profile_frame
                interval_faces = face_pass_count - last_profile_face_pass
                print_profile(stats, interval_elapsed, interval_frames, interval_faces)
                stats.clear()
                last_profile_time = now
                last_profile_frame = frame_count
                last_profile_face_pass = face_pass_count

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = max(time.perf_counter() - start_time, 0.001)
        print("\n" + "=" * 65)
        print("LIVE RECOGNITION STOPPED")
        print(f"Frames processed: {frame_count}")
        print(f"Face recognition passes: {face_pass_count}")
        print(f"Total face-detector calls: {total_face_calls}")
        print(f"Elapsed seconds: {elapsed:.1f}")
        print(f"Average end-to-end FPS: {frame_count / elapsed:.2f}")
        print(f"CSV log: {CSV_PATH}")
        print("=" * 65)


if __name__ == "__main__":
    main()
