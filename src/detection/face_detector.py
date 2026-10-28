from __future__ import annotations

import cv2
import numpy as np


class FaceDetector:
    """Detect faces using OpenCV Haar cascades with an optional YOLO face model."""

    def __init__(self, model_path=None, scale_factor=1.1, min_neighbors=5, confidence=0.5):
        self.model_path = model_path
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.confidence = confidence
        self._model = None

        if model_path:
            try:
                from ultralytics import YOLO

                self._model = YOLO(model_path)
            except Exception:
                self._model = None

        if self._model is None and hasattr(cv2, "CascadeClassifier"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._model = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self.model_path and self._model is not None and hasattr(self._model, "predict"):
            results = self._model.predict(
                source=image,
                conf=self.confidence,
                imgsz=640,
                device="cpu",
                verbose=False,
            )

            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(box.conf[0].cpu().item()),
                        "class_id": int(box.cls[0].cpu().item()) if box.cls is not None else 0,
                    })
            return detections

        if self._model is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not hasattr(self._model, "detectMultiScale"):
            return []

        detections = self._model.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
        )

        results = []
        for x, y, width, height in detections:
            x1, y1, x2, y2 = x, y, x + width, y + height
            results.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": 1.0,
                "class_id": 0,
            })
        return results

    def draw_detections(self, image, detections):
        output = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Face {detection.get('confidence', 1.0):.2f}"
            cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return output
