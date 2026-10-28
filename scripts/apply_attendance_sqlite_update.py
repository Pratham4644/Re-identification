from pathlib import Path
import shutil
import sys

target = Path("scripts/multi_camera_staff_tracking_v5_safe.py")
backup = Path("scripts/multi_camera_staff_tracking_v5_safe.py.pre-sqlite-apply.bak")

if not target.is_file():
    sys.exit(f"ABORT: Source file not found: {target}")

source = target.read_text(encoding="utf-8")

if "import sqlite3" in source or "def init_attendance_db(" in source:
    sys.exit("ABORT: SQLite changes appear to exist already. Inspect the source first.")

required = [
    "import csv\n",
    'ATTENDANCE_CSV = OUT_DIR / "attendance_events.csv"',
    "def init_csvs():",
    "def csv_writer():",
    "def crop_box(",
]
for anchor in required:
    if source.count(anchor) != 1:
        sys.exit(f"ABORT: Expected exactly one occurrence of {anchor!r}")

start = source.index("def init_csvs():")
end = source.index("def crop_box(", start)

replacement = '''def init_attendance_db():
    """Create the durable attendance database and migrate legacy CSV rows."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(ATTENDANCE_DB, timeout=10) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_date TEXT NOT NULL,
                staff_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                global_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                PRIMARY KEY (attendance_date, staff_name)
            )
        """)

        # Import legacy CSV data before exporting anything back to the CSV.
        if ATTENDANCE_CSV.exists():
            with ATTENDANCE_CSV.open(
                "r", newline="", encoding="utf-8-sig"
            ) as f:
                for row in csv.DictReader(f):
                    day = (row.get("date") or "").strip()
                    name = (row.get("staff_name") or "").strip()
                    if not day or not name:
                        continue
                    conn.execute("""
                        INSERT OR IGNORE INTO attendance
                        (attendance_date, staff_name, timestamp, global_id,
                         camera_id, event_type, evidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        day, name, row.get("timestamp") or "",
                        row.get("global_id") or "",
                        row.get("camera_id") or "",
                        row.get("event_type") or "",
                        row.get("evidence") or "",
                    ))

    export_attendance_csv()


def export_attendance_csv():
    """Rebuild the compatibility CSV from committed database records."""
    fields = [
        "timestamp", "date", "staff_name", "global_id", "camera_id",
        "event_type", "evidence",
    ]
    temp_path = ATTENDANCE_CSV.with_suffix(".csv.tmp")

    with sqlite3.connect(ATTENDANCE_DB, timeout=10) as conn:
        rows = conn.execute("""
            SELECT timestamp, attendance_date, staff_name, global_id,
                   camera_id, event_type, evidence
            FROM attendance
            ORDER BY attendance_date, timestamp, staff_name
        """).fetchall()

    try:
        with csv_lock:
            with temp_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "timestamp": row[0],
                        "date": row[1],
                        "staff_name": row[2],
                        "global_id": row[3],
                        "camera_id": row[4],
                        "event_type": row[5],
                        "evidence": row[6],
                    })
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, ATTENDANCE_CSV)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def save_attendance_event(row):
    """Insert one attendance event; return True only for a new daily record."""
    with sqlite3.connect(ATTENDANCE_DB, timeout=10) as conn:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO attendance
            (attendance_date, staff_name, timestamp, global_id, camera_id,
             event_type, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row["date"], row["staff_name"], row["timestamp"],
            row["global_id"], row["camera_id"], row["event_type"],
            row["evidence"],
        ))
        return cursor.rowcount == 1


def init_csvs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACKER_CONFIG.write_text(
        "tracker_type: bytetrack\\n"
        "track_high_thresh: 0.25\\n"
        "track_low_thresh: 0.10\\n"
        "new_track_thresh: 0.25\\n"
        "track_buffer: 90\\n"
        "match_thresh: 0.80\\n"
        "fuse_score: True\\n",
        encoding="utf-8",
    )
    with TRACK_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=[
            "timestamp", "camera_id", "local_track_id", "global_id", "staff_name",
            "face_similarity", "body_similarity", "bbox_x1", "bbox_y1", "bbox_x2",
            "bbox_y2", "status", "match_reason", "match_score", "cameras_seen"
        ]).writeheader()

    # Do not truncate attendance history. Migrate it into SQLite, then export.
    init_attendance_db()


def csv_writer():
    while not stop_event.is_set() or not event_queue.empty():
        try:
            kind, row = event_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        try:
            if kind == "attendance":
                last_error = None
                for attempt in range(3):
                    try:
                        save_attendance_event(row)
                        last_error = None
                        break
                    except sqlite3.Error as exc:
                        last_error = exc
                        if attempt < 2:
                            time.sleep(0.25 * (attempt + 1))

                if last_error is not None:
                    print(
                        "[SQLITE CRITICAL] Attendance write failed after retries: "
                        f"{type(last_error).__name__}: {last_error}"
                    )
                    key = (row.get("date"), row.get("staff_name"))
                    with state_lock:
                        registry.attendance_written.discard(key)
                    continue

                # SQLite is committed. An export failure must not undo that record.
                try:
                    export_attendance_csv()
                except Exception as exc:
                    print(
                        "[CSV EXPORT ERROR] Attendance is saved in SQLite, "
                        f"but CSV export failed: {type(exc).__name__}: {exc}"
                    )

            elif kind == "track":
                with csv_lock:
                    with TRACK_CSV.open(
                        "a", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writerow(row)
                        f.flush()
                        os.fsync(f.fileno())
            else:
                print(f"[QUEUE ERROR] Unknown event kind: {kind!r}")

        except Exception as exc:
            print(
                f"[WRITER ERROR] Failed to write {kind} event: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            event_queue.task_done()


'''

updated = source[:start] + replacement + source[end:]
updated = updated.replace("import csv\n", "import csv\nimport sqlite3\n", 1)
updated = updated.replace(
    'ATTENDANCE_CSV = OUT_DIR / "attendance_events.csv"',
    'ATTENDANCE_CSV = OUT_DIR / "attendance_events.csv"\n'
    'ATTENDANCE_DB = OUT_DIR / "attendance.db"',
    1,
)

# Validate the proposed result before writing anything.
compile(updated, str(target), "exec")
if updated.count("def init_attendance_db():") != 1:
    sys.exit("ABORT: Proposed patch failed validation.")
if 'ATTENDANCE_DB = OUT_DIR / "attendance.db"' not in updated:
    sys.exit("ABORT: Database path missing from proposed patch.")

if backup.exists():
    sys.exit(f"ABORT: Backup already exists; refusing to overwrite: {backup}")

shutil.copy2(target, backup)
try:
    target.write_text(updated, encoding="utf-8", newline="")
except Exception:
    shutil.copy2(backup, target)
    raise

print(f"Patch applied. Backup: {backup}")
print("Next: run py_compile and the identity-safety tests.")
