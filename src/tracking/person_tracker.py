from __future__ import annotations


class PersonTracker:
    """Prototype tracker that keeps track of recent detections by ID."""

    def __init__(self):
        self.tracks = {}

    def update(self, detections):
        self.tracks = {idx: detection for idx, detection in enumerate(detections or [])}
        return list(self.tracks.values())

    def clear(self):
        self.tracks.clear()
