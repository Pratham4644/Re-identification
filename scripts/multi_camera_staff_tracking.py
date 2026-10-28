
import csv
import os
import sys
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.camera_config import load_camera_urls
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.face_quality import FaceQualityChecker
GALLERY_PATH = PROJECT_ROOT / "data" / "staff_gallery"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "multi_camera_output"
CSV_PATH = OUTPUT_DIR / "recognition_events.csv"

PERSON_MODEL = str(PROJECT_ROOT / "yolo11n.pt")
PERSON_CONFIDENCE = 0.35
IMAGE_SIZE = 416

FACE_SIMILARITY_THRESHOLD = 0.30  # Experimental, not production-validated.
MIN_SIMILARITY_MARGIN = 0.05
MATCH_WINDOW = 8
MIN_MATCH_VOTES = 5
FACE_PROCESS_INTERVAL = 1.0
PERSON_CROP_BOTTOM_RATIO = 0.65
MIN_PERSON_CROP_WIDTH = 100
MIN_PERSON_CROP_HEIGHT = 140
MIN_FACE_WIDTH = 60
MIN_FACE_HEIGHT = 60
MIN_BLUR_SCORE = 30.0
MIN_FACE_DETECTION_CONFIDENCE = 0.50
TRACK_STATE_TTL_SECONDS = 15.0

STOP_EVENT = threading.Event()
FACE_LOCK = threading.Lock()
CSV_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def timestamp_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def safe_print(message):
    with PRINT_LOCK:
        print(message, flush=True)


def create_csv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "camera_id", "local_track_id", "identity",
                "similarity", "second_best_similarity", "status",
                "x1", "y1", "x2", "y2",
            ])


def write_event(camera_id, track_id, identity, similarity, second_similarity,
                status, box):
    with CSV_LOCK:
        with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                timestamp_now(), camera_id, track_id, identity,
                "" if similarity is None else round(similarity, 4),
                "" if second_similarity is None else round(second_similarity, 4),
                status, *box,
            ])


class CameraWorker(threading.Thread):
    def __init__(self, camera_id, rtsp_url, gallery, face_recognizer,
                 quality_checker):
        super().__init__(daemon=True, name=camera_id)
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.gallery = gallery
        self.face_recognizer = face_recognizer
        self.quality_checker = quality_checker
        self.person_model = None
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.track_identity = {}
        self.match_history = defaultdict(lambda: deque(maxlen=MATCH_WINDOW))
        self.track_last_seen = {}
        self.track_last_face_attempt = {}
        self.track_last_log = {}
        self.frame_count = 0
        self.error = None

    def connect(self):
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
        ]
        self.cap = cv2.VideoCapture()
        if not self.cap.open(self.rtsp_url, cv2.CAP_FFMPEG, params):
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"{self.camera_id}: unable to open RTSP stream")

    def recognize_crop(self, crop):
        # One shared InsightFace instance is protected against concurrent calls.
        with FACE_LOCK:
            faces = self.face_recognizer.app.get(crop)
        if len(faces) != 1:
            return None

        face = faces[0]
        quality = self.quality_checker.evaluate(crop, face)
        if not quality.acceptable:
            return None

        embedding = self.face_recognizer.normalize_embedding(face.normed_embedding)
        ranked = self.face_recognizer.rank_candidates(embedding, self.gallery)
        if not ranked:
            return None

        best = ranked[0]
        best_name = best["name"]
        similarity = float(best["similarity"])
        second = float(ranked[1]["similarity"]) if len(ranked) > 1 else None
        margin_ok = second is None or similarity - second >= MIN_SIMILARITY_MARGIN
        if similarity < FACE_SIMILARITY_THRESHOLD or not margin_ok:
            # Ambiguous/weak matches do not vote for Unknown, and do not refresh
            # identity confirmation.
            return None
        return best_name, similarity, second

    def process_face(self, frame, person, now):
        track_id = person["track_id"]
        if now - self.track_last_face_attempt.get(track_id, 0.0) < FACE_PROCESS_INTERVAL:
            return
        self.track_last_face_attempt[track_id] = now

        x1, y1, x2, y2 = person["box"]
        pw, ph = x2 - x1, y2 - y1
        if pw < MIN_PERSON_CROP_WIDTH or ph < MIN_PERSON_CROP_HEIGHT:
            return

        crop_y2 = y1 + int(ph * PERSON_CROP_BOTTOM_RATIO)
        crop = frame[y1:crop_y2, x1:x2]
        if crop.size == 0:
            return

        result = self.recognize_crop(crop)
        # No face, low-quality face, or ambiguous match is not a negative identity vote.
        if result is None:
            return

        name, similarity, second = result
        self.match_history[track_id].append(name)
        votes = list(self.match_history[track_id])
        counts = {candidate: votes.count(candidate) for candidate in set(votes)}
        winner = max(counts, key=counts.get)

        previous = self.track_identity.get(track_id, "Unknown")
        if counts[winner] >= MIN_MATCH_VOTES:
            self.track_identity[track_id] = winner

        identity = self.track_identity.get(track_id, "Unknown")
        changed = identity != previous
        log_due = now - self.track_last_log.get(track_id, 0.0) >= 5.0
        if changed or log_due:
            write_event(
                self.camera_id, track_id, identity, similarity, second,
                "confirmed" if identity != "Unknown" else "candidate_not_confirmed",
                person["box"],
            )
            safe_print(
                f"[{timestamp_now()}] {self.camera_id} | Local Track {track_id} "
                f"| Identity: {identity} | Similarity: {similarity:.3f}"
            )
            self.track_last_log[track_id] = now

    def run(self):
        try:
            safe_print(f"[{self.camera_id}] Loading detector...")
            self.person_model = YOLO(PERSON_MODEL)
            safe_print(f"[{self.camera_id}] Connecting...")
            self.connect()
            safe_print(f"[{self.camera_id}] Connected.")

            while not STOP_EVENT.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    safe_print(f"[{self.camera_id}] Frame read failed; reconnecting...")
                    if self.cap is not None:
                        self.cap.release()
                    time.sleep(1.0)
                    if STOP_EVENT.is_set():
                        break
                    try:
                        self.connect()
                    except Exception as exc:
                        safe_print(f"[{self.camera_id}] Reconnect failed: {exc}")
                        time.sleep(2.0)
                    continue

                self.frame_count += 1
                now = time.time()
                height, width = frame.shape[:2]
                results = self.person_model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    classes=[0], conf=PERSON_CONFIDENCE,
                    imgsz=IMAGE_SIZE, device="cpu", verbose=False,
                )

                people = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if boxes.id is not None:
                        for box, track_id in zip(
                            boxes.xyxy.cpu().numpy(),
                            boxes.id.int().cpu().tolist(),
                        ):
                            person_box = clamp_box(box, width, height)
                            if person_box is None:
                                continue
                            track_id = int(track_id)
                            people.append({"track_id": track_id, "box": person_box})
                            self.track_last_seen[track_id] = now

                for person in people:
                    self.process_face(frame, person, now)

                # Face loss does not clear identity. Only a local track absent
                # for the configured period loses its cached state.
                for track_id, last_seen in list(self.track_last_seen.items()):
                    if now - last_seen > TRACK_STATE_TTL_SECONDS:
                        self.track_identity.pop(track_id, None)
                        self.match_history.pop(track_id, None)
                        self.track_last_seen.pop(track_id, None)
                        self.track_last_face_attempt.pop(track_id, None)
                        self.track_last_log.pop(track_id, None)

                for person in people:
                    track_id = person["track_id"]
                    x1, y1, x2, y2 = person["box"]
                    identity = self.track_identity.get(track_id, "Unknown")
                    color = (0, 200, 0) if identity != "Unknown" else (0, 165, 255)
                    label = f"{self.camera_id} | Track {track_id} | {identity}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                        2, cv2.LINE_AA,
                    )

                cv2.putText(
                    frame, f"{self.camera_id} | people: {len(people)}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                with self.frame_lock:
                    self.latest_frame = frame.copy()

        except Exception as exc:
            self.error = str(exc)
            safe_print(f"[{self.camera_id}] ERROR: {exc}")
        finally:
            if self.cap is not None:
                self.cap.release()
            safe_print(f"[{self.camera_id}] Worker stopped.")


def combine_camera_frames(workers):
    import numpy as np

    tiles = []
    for worker in workers:
        with worker.frame_lock:
            frame = worker.latest_frame
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"{worker.camera_id} waiting",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        tiles.append(frame)

    if not tiles:
        return None

    target_height = min(img.shape[0] for img in tiles)

    def resize_height(img):
        scale = target_height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_height))

    resized = [resize_height(img) for img in tiles]
    columns = 2 if len(resized) > 1 else 1
    rows = []
    for index in range(0, len(resized), columns):
        row = resized[index : index + columns]
        if len(row) < columns:
            row = row + [np.zeros_like(row[0])] * (columns - len(row))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows) if len(rows) > 1 else rows[0]


def main():
    cameras = load_camera_urls(PROJECT_ROOT)
    if not cameras:
        raise RuntimeError(
            "No cameras configured. Add entries to config/cameras.json or set "
            "CAMERAS_JSON / CAMERAS. Keep credentials out of source code."
        )
    print(f"[CONFIG] Cameras ({len(cameras)}): {', '.join(sorted(cameras.keys()))}")
    if not GALLERY_PATH.is_dir():
        raise RuntimeError(f"Staff gallery does not exist: {GALLERY_PATH}")

    create_csv()
    print("=" * 70)
    print("MULTI-CAMERA STAFF TRACKING")
    print("Local track IDs are per camera. Press Q to stop.")
    print("Body-only cross-camera Re-ID is not implemented in this version.")
    print(f"CSV: {CSV_PATH}")
    print("=" * 70)

    print("[1/3] Loading shared face recognition model...")
    face_recognizer = FaceRecognizer()
    print("[2/3] Loading staff gallery...")
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

    workers = []
    print("[3/3] Starting camera workers...")
    for camera_id, url in cameras.items():
        worker = CameraWorker(
            camera_id, url, gallery, face_recognizer, quality_checker
        )
        workers.append(worker)
        worker.start()

    try:
        while not STOP_EVENT.is_set():
            combined = combine_camera_frames(workers)
            if combined is not None:
                max_width = 1600
                if combined.shape[1] > max_width:
                    scale = max_width / combined.shape[1]
                    combined = cv2.resize(
                        combined, (max_width, int(combined.shape[0] * scale))
                    )
                cv2.imshow("Multi-Camera Staff Tracking", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                STOP_EVENT.set()
                break
            if all(not worker.is_alive() for worker in workers):
                break
    except KeyboardInterrupt:
        STOP_EVENT.set()
    finally:
        STOP_EVENT.set()
        for worker in workers:
            worker.join(timeout=5.0)
        cv2.destroyAllWindows()
        print("\nStopped.")
        print(f"CSV log: {CSV_PATH}")
        for worker in workers:
            print(f"{worker.camera_id}: frames processed={worker.frame_count}")
            if worker.error:
                print(f"{worker.camera_id} error: {worker.error}")


if __name__ == "__main__":
    main()
