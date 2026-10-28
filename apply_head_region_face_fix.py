from pathlib import Path
import ast
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "scripts" / "multi_camera_staff_tracking_v5_safe.py"
START_MARKER = "def recognize_face(face_recognizer, crop: Optional[np.ndarray], gallery):"
END_MARKER = "\ndef stream_candidates("

NEW_FUNCTION = """def recognize_face(face_recognizer, crop: Optional[np.ndarray], gallery):
    \"\"\"Reject ambiguous face crops instead of guessing an identity.\"\"\"
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 80:
        return "Unknown", -1.0
    try:
        height, width = crop.shape[:2]

        # Search the upper portion of the person detection, where the head
        # should be, rather than the entire body crop.
        head_height = max(1, int(height * 0.55))
        head_region = crop[:head_height, :]

        with face_lock:
            faces = face_recognizer.app.get(head_region)

        # If there is not exactly one face, do not cast an identity vote.
        if len(faces) != 1:
            if len(faces) > 1:
                print(f"[FACE] Rejecting ambiguous crop: {len(faces)} faces in head region.")
            return "Unknown", -1.0

        face = faces[0]
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if (fw < 20 or fh < 20 or cx < 0 or cx >= width
                or cy < 0 or cy >= head_height):
            return "Unknown", -1.0

        embedding = face_recognizer.normalize_embedding(face.normed_embedding)
        candidates = face_recognizer.rank_candidates(embedding, gallery)
        if not candidates:
            return "Unknown", -1.0

        best = candidates[0]
        second = candidates[1]["similarity"] if len(candidates) > 1 else -1.0
        score = float(best["similarity"])
        # Preserve existing configured score and margin safeguards.
        if score >= FACE_MATCH_THRESHOLD and (score - float(second)) >= FACE_MARGIN:
            return str(best["name"]), score
        return "Unknown", score
    except Exception as exc:
        print(f"[FACE] Recognition error: {type(exc).__name__}: {exc}")
        return "Unknown", -1.0

"""

def main():
    if not TARGET.is_file():
        raise SystemExit(f"[ERROR] Target script not found: {TARGET}")
    source = TARGET.read_text(encoding="utf-8")
    start = source.find(START_MARKER)
    if start < 0:
        raise SystemExit("[ERROR] recognize_face() marker not found. No changes made.")
    end = source.find(END_MARKER, start)
    if end < 0:
        raise SystemExit("[ERROR] stream_candidates() marker not found. No changes made.")
    if "head_height = max(1, int(height * 0.55))" in source[start:end]:
        print("[INFO] This face-selection fix is already applied.")
        return

    updated = source[:start] + NEW_FUNCTION + source[end:]
    ast.parse(updated, filename=str(TARGET))
    backup = TARGET.with_name(TARGET.name + ".pre-head-region-face-fix.bak")
    if backup.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = TARGET.with_name(TARGET.name + f".pre-head-region-face-fix.{stamp}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(updated, encoding="utf-8", newline="")
    ast.parse(TARGET.read_text(encoding="utf-8"), filename=str(TARGET))
    print(f"[OK] Updated: {TARGET}")
    print(f"[OK] Backup:  {backup}")
    print("[INFO] Face detection now uses the upper 55% of each person crop.")
    print("[INFO] Crops with multiple detected faces are rejected.")
    print("[INFO] Face thresholds, conflict safeguards, SQLite, and attendance code were preserved.")
    print("[NOTE] Validate with live footage; this reduces one association risk but cannot guarantee correct identity.")

if __name__ == "__main__":
    main()
