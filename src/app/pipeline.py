from __future__ import annotations


class ProcessingPipeline:
    """Skeleton pipeline for processing video frames through detection and recognition stages."""

    def __init__(self, stages=None):
        self.stages = list(stages or [])

    def add_stage(self, stage):
        self.stages.append(stage)
        return stage

    def process(self, frame):
        current = frame
        for stage in self.stages:
            if hasattr(stage, "process"):
                current = stage.process(current)
        return current
