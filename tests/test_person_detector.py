from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.person_detector import PersonDetector


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

EVALUATION_PATH = ROOT / "data" / "evaluation"
OUTPUT_PATH = ROOT / "tests" / "person_detection_output"


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    detector = PersonDetector(
        model_path="yolo11n.pt",
        confidence=0.35,
        image_size=640,
    )

    print("YOLO person detection test")
    print("=" * 60)

    tested = 0

    for person_dir in sorted(EVALUATION_PATH.iterdir()):
        if not person_dir.is_dir():
            continue

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipped unreadable image: {image_path}")
                continue

            detections = detector.detect(image)

            output_image = detector.draw_detections(
                image,
                detections,
            )

            output_name = (
                f"{person_dir.name}_{image_path.stem}_persons.jpg"
            )

            output_path = OUTPUT_PATH / output_name

            if not cv2.imwrite(str(output_path), output_image):
                print(f"Could not save: {output_path}")
                continue

            print(f"\nImage: {person_dir.name}/{image_path.name}")
            print(f"People detected: {len(detections)}")

            for index, detection in enumerate(detections, start=1):
                print(
                    f"  Person {index}: "
                    f"bbox={detection['bbox']}, "
                    f"confidence={detection['confidence']:.3f}"
                )

            print(f"Saved visualization: {output_path}")
            tested += 1

    print(f"\nImages processed successfully: {tested}")
    print(f"Output directory: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()