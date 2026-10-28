from __future__ import annotations


class CameraManager:
    """Minimal camera manager for a single or multi-camera workflow."""

    def __init__(self, streams=None):
        self.streams = list(streams or [])

    def add_stream(self, stream):
        self.streams.append(stream)
        return stream

    def release(self):
        for stream in self.streams:
            if hasattr(stream, "release"):
                stream.release()
