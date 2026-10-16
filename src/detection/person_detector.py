from pathlib import Path

import cv2
from ultralytics import YOLO


class PersonDetector:
    """
    Detect people in images using YOLO.

    Initial confidence and model settings are experimental.
    """

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.35,
        image_size=640,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, image):
        """
        Args:
            image: OpenCV BGR image.

        Returns:
            List of detected people with bounding boxes,
            confidence scores, and class IDs.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        results = self.model.predict(
            source=image,
            classes=[0],  # COCO class 0 is person.
            conf=self.confidence,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().tolist()
                )

                detections.append({
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ),
                    "confidence": float(
                        box.conf[0].cpu().item()
                    ),
                    "class_id": int(
                        box.cls[0].cpu().item()
                    ),
                })

        return detections

    def draw_detections(self, image, detections):
        """
        Return an annotated copy of the input image.
        """
        output = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"Person {confidence:.2f}"

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return output