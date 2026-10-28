from __future__ import annotations


class VideoStream:
    """Simple video-stream wrapper used by prototype pipelines."""

    def __init__(self, source=0):
        self.source = source
        self._capture = None

    def open(self):
        import cv2

        self._capture = cv2.VideoCapture(self.source)
        return self._capture is not None and self._capture.isOpened()

    def read(self):
        if self._capture is None:
            raise RuntimeError("Stream is not open.")
        return self._capture.read()

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
