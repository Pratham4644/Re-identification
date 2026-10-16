from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class FaceQualityResult:
    acceptable: bool
    reason: str

    width: int
    height: int

    blur_score: float
    detection_confidence: float

    size_ok: bool
    blur_ok: bool
    confidence_ok: bool


class FaceQualityChecker:
    """
    Experimental face quality assessment.

    Thresholds are initial settings only. They must be validated
    against representative CCTV footage before production use.
    """

    def __init__(
        self,
        min_face_width=60,
        min_face_height=60,
        min_blur_score=30.0,
        min_detection_confidence=0.50,
    ):
        self.min_face_width = min_face_width
        self.min_face_height = min_face_height
        self.min_blur_score = min_blur_score
        self.min_detection_confidence = min_detection_confidence

    def evaluate(self, image, face):
        if image is None or image.size == 0:
            return FaceQualityResult(
                False,
                "invalid_image",
                0,
                0,
                0.0,
                0.0,
                False,
                False,
                False,
            )

        image_height, image_width = image.shape[:2]

        bbox = np.asarray(face.bbox, dtype=float)

        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            return FaceQualityResult(
                False,
                "invalid_face_box",
                0,
                0,
                0.0,
                0.0,
                False,
                False,
                False,
            )

        x1, y1, x2, y2 = bbox.tolist()

        confidence = float(face.det_score)

        if (
            not np.isfinite(confidence)
            or x1 < 0
            or y1 < 0
            or x2 > image_width
            or y2 > image_height
            or x2 <= x1
            or y2 <= y1
        ):
            return FaceQualityResult(
                False,
                "invalid_or_outside_face_box",
                0,
                0,
                0.0,
                confidence if np.isfinite(confidence) else 0.0,
                False,
                False,
                False,
            )

        # Use floor/ceil to avoid accidentally shrinking the crop.
        x1, y1 = int(np.floor(x1)), int(np.floor(y1))
        x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))

        width = x2 - x1
        height = y2 - y1

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return FaceQualityResult(
                False,
                "empty_face_crop",
                width,
                height,
                0.0,
                confidence,
                False,
                False,
                confidence >= self.min_detection_confidence,
            )

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        blur_score = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        size_ok = (
            width >= self.min_face_width
            and height >= self.min_face_height
        )

        blur_ok = blur_score >= self.min_blur_score

        confidence_ok = (
            confidence >= self.min_detection_confidence
        )

        failed_checks = []

        if not size_ok:
            failed_checks.append("face_too_small")

        if not blur_ok:
            failed_checks.append("face_too_blurry")

        if not confidence_ok:
            failed_checks.append("low_detection_confidence")

        acceptable = not failed_checks

        reason = (
            "acceptable"
            if acceptable
            else ",".join(failed_checks)
        )

        return FaceQualityResult(
            acceptable=acceptable,
            reason=reason,
            width=width,
            height=height,
            blur_score=blur_score,
            detection_confidence=confidence,
            size_ok=size_ok,
            blur_ok=blur_ok,
            confidence_ok=confidence_ok,
        )