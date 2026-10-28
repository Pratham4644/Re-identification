from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


TARGET_RELATIVE = Path("scripts") / "multi_camera_staff_tracking_v5_safe.py"
BACKUP_SUFFIX = ".pre-local-track-reuse-fix.bak"


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def main() -> int:
    root = Path.cwd()
    target = root / TARGET_RELATIVE

    if not target.is_file():
        fail(
            f"Could not find {TARGET_RELATIVE} from {root}. "
            "Run this patcher from the D:\\person-reid project root."
        )

    source = target.read_text(encoding="utf-8")

    if "class SharedRegistry" not in source or "def assign(" not in source or "def prune(" not in source:
        fail("The target file does not contain the expected SharedRegistry methods; no changes made.")

    changed = False

    if "LOCAL_TRACK_REUSE_SEC" not in source:
        config_pattern = r"(?m)^(GLOBAL_TTL\s*=\s*.*\n)"
        source, count = re.subn(
            config_pattern,
            r'\1LOCAL_TRACK_REUSE_SEC = float(\n'
            r'    os.getenv("V5_LOCAL_TRACK_REUSE_SEC", "5.0")\n'
            r')\n',
            source,
            count=1,
        )
        if count != 1:
            fail("Could not locate GLOBAL_TTL config line; no changes made.")
        changed = True

    if "Expiring stale local mapping" not in source:
        assign_pattern = (
            r"(?m)^(?P<indent>\s*)lt = self\.locals\.get\(key\)\n"
            r"(?P=indent)if lt is not None:\n"
            r"(?P=indent)    lt\.last_seen = t"
        )
        assign_replacement = (
            r"\g<indent>lt = self.locals.get(key)\n"
            r"\g<indent>if (\n"
            r"\g<indent>    lt is not None\n"
            r"\g<indent>    and t - lt.last_seen > LOCAL_TRACK_REUSE_SEC\n"
            r"\g<indent>):\n"
            r'\g<indent>    print(\n'
            r'\g<indent>        f"[SAFETY] Expiring stale local mapping: "\n'
            r'\g<indent>        f"{camera_id}/L{local_id} -> {lt.global_id}; "\n'
            r'\g<indent>        f"inactive for {t - lt.last_seen:.1f}s."\n'
            r"\g<indent>    )\n"
            r"\g<indent>    self.locals.pop(key, None)\n"
            r"\g<indent>    lt = None\n"
            r"\n"
            r"\g<indent>if lt is not None:\n"
            r"\g<indent>    lt.last_seen = t"
        )
        source, count = re.subn(assign_pattern, assign_replacement, source, count=1)
        if count != 1:
            fail("Could not safely patch SharedRegistry.assign(); no changes written.")
        changed = True

    if "stale_local_keys = [" not in source:
        prune_pattern = (
            r"(?ms)^    def prune\(self, t: float\):\n"
            r".*?"
            r"(?=^    def maybe_attendance\()"
        )
        prune_replacement = """    def prune(self, t: float):
        with state_lock:
            # A camera-local tracker ID is not permanent identity evidence.
            # Drop its mapping after inactivity so a recycled ID starts fresh.
            stale_local_keys = [
                key for key, lt in self.locals.items()
                if t - lt.last_seen > LOCAL_TRACK_REUSE_SEC
            ]
            for key in stale_local_keys:
                lt = self.locals.pop(key, None)
                if lt is not None:
                    print(
                        f"[SAFETY] Pruned stale local mapping: "
                        f"{key[0]}/L{key[1]} -> {lt.global_id}."
                    )

            expired = [
                gid for gid, gp in self.globals.items()
                if t - gp.last_seen > GLOBAL_TTL
            ]
            for gid in expired:
                self.globals.pop(gid, None)

            # Remove any remaining local mapping whose global record expired.
            for key, lt in list(self.locals.items()):
                if lt.global_id not in self.globals:
                    self.locals.pop(key, None)

"""
        source, count = re.subn(prune_pattern, prune_replacement, source, count=1)
        if count != 1:
            fail("Could not safely patch SharedRegistry.prune(); no changes written.")
        changed = True

    if not changed:
        print("[INFO] The local-track reuse fix is already present; no changes made.")
        return 0

    try:
        compile(source, str(target), "exec")
    except SyntaxError as exc:
        fail(f"Patched source failed syntax validation at line {exc.lineno}: {exc.msg}. No changes written.")

    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if backup.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = target.with_name(target.name + f".{stamp}" + BACKUP_SUFFIX)

    shutil.copy2(target, backup)
    target.write_text(source, encoding="utf-8", newline="")
    print(f"[OK] Updated: {target}")
    print(f"[OK] Backup:  {backup}")
    print("[OK] Stale local mappings expire after V5_LOCAL_TRACK_REUSE_SEC seconds (default 5.0).")
    print("[INFO] Attendance database/CSV code and conflict safeguards were left unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
