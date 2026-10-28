from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root containing config/cameras.json or src/."""
    start = start or Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "config" / "cameras.json").is_file():
            return candidate
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return start


def _resolve_entry(entry: dict[str, Any]) -> tuple[str, str] | None:
    camera_id = str(entry.get("id", "")).strip()
    if not camera_id:
        return None

    url = str(entry.get("url") or "").strip()
    if not url:
        url_env = entry.get("url_env")
        if url_env:
            url = os.getenv(str(url_env), "").strip()
        if not url:
            fallback_env = entry.get("fallback_env")
            if fallback_env:
                url = os.getenv(str(fallback_env), "").strip()

    if not url:
        return None
    return camera_id, url


def _parse_cameras_env(raw: str) -> dict[str, str]:
    """
    Parse CAMERAS env: id|url pairs separated by semicolons.
    Example: CAM-202|rtsp://a;CAM-302|rtsp://b
    """
    cameras: dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "|" not in part:
            continue
        camera_id, url = part.split("|", 1)
        camera_id = camera_id.strip()
        url = url.strip()
        if camera_id and url:
            cameras[camera_id] = url
    return cameras


def _parse_camera_list(entries: Any) -> dict[str, str]:
    if not isinstance(entries, list):
        return {}
    cameras: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resolved = _resolve_entry(entry)
        if resolved:
            cameras[resolved[0]] = resolved[1]
    return cameras


def load_camera_urls(project_root: Path | None = None) -> dict[str, str]:
    """
    Load camera id -> stream URL mappings.

    Sources (later sources override earlier ones for the same id):
      1. config/cameras.json (or CAMERAS_CONFIG path)
      2. CAMERAS_JSON environment variable (JSON array of camera objects)
      3. CAMERAS environment variable (id|url;id|url)
    """
    root = project_root or find_project_root()
    config_path = Path(
        os.getenv("CAMERAS_CONFIG", str(root / "config" / "cameras.json"))
    ).expanduser()

    cameras: dict[str, str] = {}

    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
        cameras.update(_parse_camera_list(data.get("cameras", [])))

    cameras_json = os.getenv("CAMERAS_JSON", "").strip()
    if cameras_json:
        try:
            payload = json.loads(cameras_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid CAMERAS_JSON: {exc}") from exc
        cameras.update(_parse_camera_list(payload))

    cameras_env = os.getenv("CAMERAS", "").strip()
    if cameras_env:
        cameras.update(_parse_cameras_env(cameras_env))

    return cameras


def get_camera_url(
    camera_id: str,
    project_root: Path | None = None,
) -> str | None:
    """Return the URL for one camera id, or None if it is not configured."""
    camera_id = camera_id.strip()
    if not camera_id:
        return None
    urls = load_camera_urls(project_root)
    url = urls.get(camera_id)
    return url.strip() if url else None


def list_configured_camera_ids(project_root: Path | None = None) -> list[str]:
    return sorted(load_camera_urls(project_root).keys())


_ENV_CAMERA_URL = re.compile(r"^CAMERA_(?P<suffix>.+)_(?:URL|RTSP)$")


def resolve_single_camera_stream(
    project_root: Path | None = None,
    camera_id: str | None = None,
) -> tuple[str, str]:
    """
    Pick one camera for single-stream scripts.

    URL precedence: CAMERA_URL, CAMERA_RTSP, configured url for camera_id.
    camera_id precedence: argument, CAMERA_ID env, sole configured camera,
    else first configured id alphabetically.
    """
    root = project_root or find_project_root()
    chosen_id = (camera_id or os.getenv("CAMERA_ID", "")).strip()
    configured = load_camera_urls(root)

    if not chosen_id:
        if len(configured) == 1:
            chosen_id = next(iter(configured))
        elif configured:
            chosen_id = sorted(configured)[0]

    url = os.getenv("CAMERA_URL", "").strip() or os.getenv("CAMERA_RTSP", "").strip()
    if not url and chosen_id:
        url = configured.get(chosen_id, "")
    if not url and configured and not chosen_id:
        chosen_id, url = sorted(configured.items())[0]

    if not chosen_id:
        chosen_id = "CAM-1"
    return chosen_id, url.strip()


def discover_legacy_env_camera_ids() -> list[str]:
    """
    List suffixes from CAMERA_*_URL / CAMERA_*_RTSP env vars.
    Useful for diagnostics only; prefer config/cameras.json.
    """
    suffixes: set[str] = set()
    for key in os.environ:
        match = _ENV_CAMERA_URL.match(key)
        if match:
            suffixes.add(match.group("suffix"))
    return sorted(suffixes)
