from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recognition.face_recognizer import FaceRecognizer

GALLERY_PATH = ROOT / "data" / "staff_gallery"
EVALUATION_PATH = ROOT / "data" / "evaluation"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

# Demonstration only. Not a validated production threshold.
DEMO_THRESHOLD = 0.30


def main():
    recognizer = FaceRecognizer()
    gallery = recognizer.load_gallery(GALLERY_PATH)

    print(f"\nDemonstration threshold: {DEMO_THRESHOLD:.2f}")
    print("WARNING: This threshold is not production-validated.")

    for person_dir in sorted(EVALUATION_PATH.iterdir()):
        if not person_dir.is_dir():
            continue

        expected_name = person_dir.name

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                embedding = recognizer.extract_embedding(image_path)
                result = recognizer.identify(
                    embedding,
                    gallery,
                    threshold=DEMO_THRESHOLD,
                )

                print(f"\nImage: {expected_name}/{image_path.name}")
                print(f"Expected: {expected_name}")
                print(f"Predicted: {result['identity']}")
                print(f"Status: {result['status']}")
                print(f"Similarity: {result['similarity']:.4f}")

            except ValueError as exc:
                print(f"\nImage: {image_path}")
                print(f"Skipped: {exc}")


if __name__ == "__main__":
    main()