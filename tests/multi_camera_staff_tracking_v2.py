"""
Multi-camera staff tracking V2 (experimental baseline)
======================================================
Keeps per-camera YOLO/ByteTrack tracks separate and associates them to cautious
global IDs using OSNet body embeddings when torchreid is installed. Face
recognition can attach a staff name; body similarity alone never invents a name.

Outputs:
  tests/multi_camera_output_v2/track_events.csv
  tests/multi_camera_output_v2/attendance_events.csv  (one first_seen per staff/day)

Environment:
  CAMERA_202_RTSP, CAMERA_302_RTSP
Optional:
  STAFF_GALLERY_PATH, V2_BODY_REID=1, V2_FACE_INTERVAL_SEC=1.5

This is a test baseline, not production attendance software. Tune thresholds with
recorded footage before relying on attendance records. OSNet pretrained weights
may download on first use.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import threading
import queue
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.face_recognizer import FaceRecognizer

CAMERAS = {
    "CAM-202": os.getenv("CAMERA_202_RTSP", ""),
    "CAM-302": os.getenv("CAMERA_302_RTSP", ""),
}
GALLERY_PATH = Path(os.getenv("STAFF_GALLERY_PATH", str(ROOT / "data" / "staff_gallery")))
OUT_DIR = ROOT / "tests" / "multi_camera_output_v2"
TRACK_CSV = OUT_DIR / "track_events.csv"
ATTENDANCE_CSV = OUT_DIR / "attendance_events.csv"

YOLO_MODEL = os.getenv("V2_YOLO_MODEL", str(ROOT / "yolo11n.pt"))
YOLO_IMGSZ = int(os.getenv("V2_YOLO_IMGSZ", "416"))
YOLO_CONF = float(os.getenv("V2_YOLO_CONF", "0.35"))
FACE_INTERVAL = float(os.getenv("V2_FACE_INTERVAL_SEC", "1.5"))
GLOBAL_TTL = float(os.getenv("V2_GLOBAL_TTL_SEC", "45"))
BODY_ENABLED = os.getenv("V2_BODY_REID", "1") != "0"
BODY_MATCH_THRESHOLD = float(os.getenv("V2_BODY_MATCH_THRESHOLD", "0.78"))
FACE_MATCH_THRESHOLD = float(os.getenv("V2_FACE_MATCH_THRESHOLD", "0.45"))
FACE_MARGIN = float(os.getenv("V2_FACE_MARGIN", "0.05"))

stop_event = threading.Event()
face_lock = threading.Lock()
state_lock = threading.RLock()
csv_lock = threading.Lock()
event_queue: queue.Queue = queue.Queue()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    a, b = normalize(a), normalize(b)
    if a.shape != b.shape:
        return -1.0
    return float(np.dot(a, b))


class BodyEmbedder:
    """OSNet person-ReID features via torchreid; fails closed if unavailable."""
    def __init__(self):
        self.model = None
        self.transform = None
        self.torch = None
        if not BODY_ENABLED:
            print("[BODY] Disabled by V2_BODY_REID=0")
            return
        try:
            import torch
            import torchreid
            from torchvision import transforms
            self.torch = torch
            self.model = torchreid.models.build_model(
                name="osnet_x1_0", num_classes=1000, loss="softmax", pretrained=True
            )
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            print("[BODY] OSNet loaded; body matching is enabled.")
        except Exception as exc:
            self.model = None
            print("[BODY] OSNet unavailable. Cross-camera matching will be face-anchor only.")
            print(f"[BODY] Details: {type(exc).__name__}: {exc}")
            print("[BODY] Install in this venv with: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")

    def embed(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or bgr_crop is None or bgr_crop.size == 0:
            return None
        h, w = bgr_crop.shape[:2]
        if h < 70 or w < 25:
            return None
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0)
        try:
            with self.torch.no_grad():
                feature = self.model(tensor)
            if isinstance(feature, (tuple, list)):
                feature = feature[0]
            return normalize(feature.detach().cpu().numpy().reshape(-1))
        except Exception as exc:
            print(f"[BODY] Embedding error: {exc}")
            return None


@dataclass
class GlobalPerson:
    global_id: str
    created_at: float
    last_seen: float
    body_embedding: Optional[np.ndarray] = None
    body_samples: int = 0
    staff_name: str = "Unknown"
    face_scores: deque = field(default_factory=lambda: deque(maxlen=7))
    face_name_votes: Counter = field(default_factory=Counter)
    last_face_at: float = 0.0
    cameras_seen: set = field(default_factory=set)


@dataclass
class LocalTrack:
    camera_id: str
    local_id: int
    global_id: str
    last_seen: float
    last_face_attempt: float = 0.0
    last_body_attempt: float = 0.0
    face_similarity: float = -1.0
    body_similarity: float = -1.0
    staff_name: str = "Unknown"
    last_bbox: tuple = (0, 0, 0, 0)


class SharedRegistry:
    def __init__(self):
        self.globals: dict[str, GlobalPerson] = {}
        self.locals: dict[tuple[str, int], LocalTrack] = {}
        self.counter = 0
        self.attendance_written: set[tuple[str, str]] = set()

    def _new_global(self, t: float) -> GlobalPerson:
        self.counter += 1
        gid = f"G{self.counter:05d}"
        person = GlobalPerson(gid, t, t)
        self.globals[gid] = person
        return person

    def _active_local_ids(self, camera_id: str, t: float) -> set[str]:
        return {lt.global_id for (cam, _), lt in self.locals.items()
                if cam == camera_id and t - lt.last_seen < 2.0}

    def assign(self, camera_id: str, local_id: int, body: Optional[np.ndarray],
               t: float) -> tuple[LocalTrack, float]:
        key = (camera_id, local_id)
        with state_lock:
            lt = self.locals.get(key)
            if lt is not None:
                lt.last_seen = t
                gp = self.globals.get(lt.global_id)
                if gp is not None:
                    gp.last_seen = t
                    gp.cameras_seen.add(camera_id)
                return lt, -1.0

            best_gp = None
            best_sim = -1.0
            blocked = self._active_local_ids(camera_id, t)
            if body is not None:
                for gp in self.globals.values():
                    if t - gp.last_seen > GLOBAL_TTL or gp.global_id in blocked:
                        continue
                    if gp.body_embedding is None:
                        continue
                    sim = cosine(body, gp.body_embedding)
                    # Require a strong OSNet match; identical uniforms can still collide.
                    if sim > best_sim:
                        best_sim, best_gp = sim, gp
            if best_gp is None or best_sim < BODY_MATCH_THRESHOLD:
                best_gp = self._new_global(t)
                best_sim = -1.0

            lt = LocalTrack(camera_id, local_id, best_gp.global_id, t)
            self.locals[key] = lt
            best_gp.last_seen = t
            best_gp.cameras_seen.add(camera_id)
            return lt, best_sim

    def update_body(self, lt: LocalTrack, emb: Optional[np.ndarray], t: float):
        if emb is None:
            return
        with state_lock:
            gp = self.globals.get(lt.global_id)
            if gp is None:
                return
            if gp.body_embedding is None:
                gp.body_embedding = emb.copy()
                gp.body_samples = 1
            else:
                # Running mean reduces noise; normalize after combining.
                gp.body_embedding = normalize(gp.body_embedding * min(gp.body_samples, 8) + emb)
                gp.body_samples += 1
            gp.last_seen = t

    def update_face(self, lt: LocalTrack, name: str, similarity: float, t: float):
        lt.face_similarity = similarity
        if name == "Unknown" or similarity < FACE_MATCH_THRESHOLD:
            return
        with state_lock:
            gp = self.globals.get(lt.global_id)
            if gp is None:
                return
            # Do not silently overwrite one confirmed name with another.
            if gp.staff_name not in ("Unknown", name):
                print(f"[SAFETY] Conflicting face names for {gp.global_id}: "
                      f"{gp.staff_name} vs {name}; leaving name unchanged.")
                return
            gp.face_name_votes[name] += 1
            gp.face_scores.append(similarity)
            gp.last_face_at = t
            # Require repeated consistent face evidence before attaching a staff name.
            if gp.face_name_votes[name] >= 2 and len(gp.face_scores) >= 2:
                gp.staff_name = name
            lt.staff_name = gp.staff_name

    def get_name(self, lt: LocalTrack) -> str:
        with state_lock:
            gp = self.globals.get(lt.global_id)
            return gp.staff_name if gp else "Unknown"

    def prune(self, t: float):
        with state_lock:
            expired = [gid for gid, gp in self.globals.items()
                       if t - gp.last_seen > GLOBAL_TTL]
            for gid in expired:
                self.globals.pop(gid, None)
            # Keep stale local track mappings only while their global record exists.
            for key, lt in list(self.locals.items()):
                if lt.global_id not in self.globals and t - lt.last_seen > GLOBAL_TTL:
                    self.locals.pop(key, None)

    def maybe_attendance(self, lt: LocalTrack, camera_id: str):
        name = self.get_name(lt)
        if name == "Unknown":
            return
        day = datetime.now().astimezone().date().isoformat()
        key = (day, name)
        with state_lock:
            if key in self.attendance_written:
                return
            self.attendance_written.add(key)
        event_queue.put(("attendance", {
            "timestamp": now_iso(), "date": day, "staff_name": name,
            "global_id": lt.global_id, "camera_id": camera_id,
            "event_type": "first_seen", "evidence": "repeated_face_match"
        }))


registry = SharedRegistry()


def init_csvs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRACK_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=[
            "timestamp", "camera_id", "local_track_id", "global_id", "staff_name",
            "face_similarity", "body_similarity", "bbox_x1", "bbox_y1", "bbox_x2",
            "bbox_y2", "status"
        ]).writeheader()
    with ATTENDANCE_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=[
            "timestamp", "date", "staff_name", "global_id", "camera_id",
            "event_type", "evidence"
        ]).writeheader()


def csv_writer():
    while not stop_event.is_set() or not event_queue.empty():
        try:
            kind, row = event_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        path = TRACK_CSV if kind == "track" else ATTENDANCE_CSV
        with csv_lock, path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        event_queue.task_done()


def crop_box(frame: np.ndarray, xyxy) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def recognize_face(face_recognizer, crop: Optional[np.ndarray], gallery):
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 80:
        return "Unknown", -1.0
    try:
        # InsightFace app is shared between camera threads, so serialize calls.
        with face_lock:
            faces = face_recognizer.app.get(crop)
        if not faces:
            return "Unknown", -1.0
        # Use the largest face in the person crop.
        face = max(faces, key=lambda f: max(0, f.bbox[2] - f.bbox[0]) *
                   max(0, f.bbox[3] - f.bbox[1]))
        embedding = face_recognizer.normalize_embedding(face.normed_embedding)
        candidates = face_recognizer.rank_candidates(embedding, gallery)
        if not candidates:
            return "Unknown", -1.0
        best = candidates[0]
        second = candidates[1]["similarity"] if len(candidates) > 1 else -1.0
        score = float(best["similarity"])
        if score >= FACE_MATCH_THRESHOLD and (score - float(second)) >= FACE_MARGIN:
            return str(best["name"]), score
        return "Unknown", score
    except Exception as exc:
        print(f"[FACE] Recognition error: {exc}")
        return "Unknown", -1.0


def camera_worker(camera_id: str, url: str, face_recognizer, gallery, embedder):
    print(f"[{camera_id}] Starting camera worker")
    model = YOLO(YOLO_MODEL)
    tracks_last_seen = {}
    cap = None
    frame_no = 0
    try:
        while not stop_event.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not cap.isOpened():
                    print(f"[{camera_id}] RTSP open failed; retrying in 3s")
                    time.sleep(3)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[{camera_id}] Read failed; reconnecting")
                cap.release()
                cap = None
                time.sleep(1)
                continue
            frame_no += 1
            t = time.monotonic()
            try:
                result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                     classes=[0], conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                                     verbose=False, device="cpu")[0]
            except Exception as exc:
                print(f"[{camera_id}] YOLO error: {exc}")
                continue
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                registry.prune(t)
                continue

            ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.cpu().numpy()
            for local_id, xyxy in zip(ids, coords):
                crop = crop_box(frame, xyxy)
                if crop is None:
                    continue
                key = (camera_id, int(local_id))
                lt = registry.locals.get(key)
                body_emb = None
                if lt is None or t - lt.last_body_attempt >= 1.5:
                    body_emb = embedder.embed(crop)
                    # Existing local track mapping can be read by camera thread; registry writes lock.
                    if lt is not None:
                        lt.last_body_attempt = t
                lt, match_sim = registry.assign(camera_id, int(local_id), body_emb, t)
                lt.body_similarity = match_sim if match_sim >= 0 else lt.body_similarity
                if body_emb is not None:
                    registry.update_body(lt, body_emb, t)
                    lt.last_body_attempt = t

                if t - lt.last_face_attempt >= FACE_INTERVAL:
                    name, score = recognize_face(face_recognizer, crop, gallery)
                    registry.update_face(lt, name, score, t)
                    lt.last_face_attempt = t

                staff_name = registry.get_name(lt)
                lt.staff_name = staff_name
                registry.maybe_attendance(lt, camera_id)
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                color = (0, 200, 0) if staff_name != "Unknown" else (0, 180, 255)
                label = f"{staff_name} | {lt.global_id} | L{local_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                event_queue.put(("track", {
                    "timestamp": now_iso(), "camera_id": camera_id,
                    "local_track_id": int(local_id), "global_id": lt.global_id,
                    "staff_name": staff_name,
                    "face_similarity": round(lt.face_similarity, 4),
                    "body_similarity": round(lt.body_similarity, 4),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "status": "identified" if staff_name != "Unknown" else "tracking"
                }))
            registry.prune(t)
            cv2.imshow(f"V2 {camera_id}", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    except Exception as exc:
        print(f"[{camera_id}] Worker stopped after error: {type(exc).__name__}: {exc}")
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyWindow(f"V2 {camera_id}")
        except Exception:
            pass
        print(f"[{camera_id}] Stopped")


def main():
    configured = {k: v for k, v in CAMERAS.items() if v.strip()}
    if not configured:
        print("No camera URLs found. Set CAMERA_202_RTSP and/or CAMERA_302_RTSP in PowerShell.")
        return 2
    if not Path(YOLO_MODEL).exists() and not Path(YOLO_MODEL).is_absolute():
        print(f"YOLO model not found: {YOLO_MODEL}")
        return 2
    if not GALLERY_PATH.exists():
        print(f"Staff gallery not found: {GALLERY_PATH}")
        return 2

    init_csvs()
    writer_thread = threading.Thread(target=csv_writer, daemon=True)
    writer_thread.start()

    face_recognizer = FaceRecognizer()
    gallery = face_recognizer.load_gallery(str(GALLERY_PATH))
    if not gallery:
        print(f"No staff faces loaded from {GALLERY_PATH}")
        stop_event.set()
        writer_thread.join(timeout=2)
        return 2
    print(f"[FACE] Gallery identities: {', '.join(gallery.keys())}")

    embedder = BodyEmbedder()
    threads = []
    for camera_id, url in configured.items():
        th = threading.Thread(target=camera_worker,
                              args=(camera_id, url, face_recognizer, gallery, embedder),
                              daemon=True, name=camera_id)
        th.start()
        threads.append(th)
    print("V2 running. Press q in a video window or Ctrl+C to stop.")
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
    finally:
        stop_event.set()
        for th in threads:
            th.join(timeout=5)
        event_queue.join()
        writer_thread.join(timeout=2)
        cv2.destroyAllWindows()
    print(f"Track log: {TRACK_CSV}")
    print(f"First-seen attendance log: {ATTENDANCE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
