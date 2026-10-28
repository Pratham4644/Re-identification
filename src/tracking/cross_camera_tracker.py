from __future__ import annotations


class CrossCameraTracker:
    """Prototype cross-camera tracker that groups all detections by identity label."""

    def __init__(self):
        self.identities = {}

    def update(self, detections):
        for detection in detections or []:
            identity = detection.get("identity", "Unknown") if isinstance(detection, dict) else detection
            self.identities.setdefault(identity, []).append(detection)
        return dict(self.identities)

    def clear(self):
        self.identities.clear()
