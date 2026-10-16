from pathlib import Path

import cv2
import numpy as np


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: tuple[int, int] = (112, 112),
) -> np.ndarray | None:
    """
    Align a face using the 68-point facial landmarks.

    Args:
        image: Original BGR image.
        landmarks: 68 facial landmark points, shape (68, 2).
        output_size: (width, height) of the aligned face.

    Returns:
        Aligned BGR face image, or None if alignment fails.
    """
    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)

    if len(points) != 68:
        raise ValueError(f"Expected 68 landmarks, got {len(points)}")

    if not np.isfinite(points).all():
        return None

    # Standard eye-center locations for a 112x112 aligned face.
    # Landmark indices follow the conventional 68-point layout.
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)

    source = np.array([left_eye, right_eye], dtype=np.float32)

    width, height = output_size

    destination = np.array(
        [
            [0.30 * width, 0.38 * height],
            [0.70 * width, 0.38 * height],
        ],
        dtype=np.float32,
    )

    transform, _ = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.LMEDS,
    )

    if transform is None:
        return None

    aligned = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return aligned