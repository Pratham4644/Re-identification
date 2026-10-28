from __future__ import annotations


class Dashboard:
    """Minimal dashboard placeholder."""

    def __init__(self, title="Person Re-ID Dashboard"):
        self.title = title

    def render(self, data=None):
        return {"title": self.title, "data": data}
