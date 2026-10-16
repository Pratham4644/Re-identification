from pathlib import Path

import cv2

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.face_aligner import align_face


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lbfmodel.yaml"


def main():
    image_path = input("Enter the face image path: ").strip().strip('"')

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier(
        str(ROOT / "models" / "haarcascade_frontalface_default.xml")
    )

    if detector.empty():
        raise RuntimeError("Could not load the face detector.")

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        raise RuntimeError("No face detected.")

    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(MODEL_PATH))

    success, results = facemark.fit(gray, faces)

    if not success or len(results) == 0:
        raise RuntimeError("Landmark prediction failed.")

    landmarks = results[0].reshape(-1, 2)

    aligned = align_face(image, landmarks)

    if aligned is None:
        raise RuntimeError("Face alignment failed.")

    output_path = ROOT / "tests" / "aligned_face.jpg"

    if not cv2.imwrite(str(output_path), aligned):
        raise RuntimeError("Could not save aligned face.")

    print("Landmarks:", len(landmarks))
    print("Aligned image size:", aligned.shape)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()