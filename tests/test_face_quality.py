from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.face_quality import FaceQualityChecker


GALLERY_PATH = ROOT / "data" / "staff_gallery"
EVALUATION_PATH = ROOT / "data" / "evaluation"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


def main():
    recognizer = FaceRecognizer()
    checker = FaceQualityChecker()

    # Load the model and gallery using the existing working code.
    gallery = recognizer.load_gallery(GALLERY_PATH)

    print(f"\nEnrolled staff: {len(gallery)}")
    print("Face quality test")
    print("=" * 80)

    for person_dir in sorted(EVALUATION_PATH.iterdir()):
        if not person_dir.is_dir():
            continue

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"\n{image_path.name}: unreadable image")
                continue

            faces = recognizer.app.get(image)

            print(f"\nImage: {person_dir.name}/{image_path.name}")

            if not faces:
                print("Result: NO FACE DETECTED")
                continue

            # Evaluate every detected face, not just the first one.
            for index, face in enumerate(faces, start=1):
                quality = checker.evaluate(image, face)

                print(f"Face: {index}")
                print(f"Acceptable: {quality.acceptable}")
                print(f"Reason: {quality.reason}")
                print(
                    f"Bounding box: "
                    f"{quality.width} x {quality.height} pixels"
                )
                print(
                    f"Detection confidence: "
                    f"{quality.detection_confidence:.3f}"
                )
                print(f"Blur score: {quality.blur_score:.2f}")
                print(f"Size check passed: {quality.size_ok}")
                print(f"Blur check passed: {quality.blur_ok}")
                print(
                    "Detection confidence check passed: "
                    f"{quality.confidence_ok}"
                )

                if quality.acceptable:
                    embedding = recognizer.normalize_embedding(
                        face.normed_embedding
                    )

                    result = recognizer.identify(
                        embedding,
                        gallery,
                        threshold=0.30,
                    )

                    print(f"Identity decision: {result['identity']}")
                    print(f"Status: {result['status']}")
                    print(
                        f"Similarity: "
                        f"{result['similarity']:.4f}"
                    )
                else:
                    print("Identity decision: Unknown")
                    print("Status: insufficient_face_quality")


if __name__ == "__main__":
    main()