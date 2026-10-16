import cv2
from pathlib import Path

from src.detection.person_detector import PersonDetector
from src.detection.face_person_associator import FacePersonAssociator
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.face_quality import FaceQualityChecker


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
GALLERY_DIR = PROJECT_ROOT / "data" / "staff_gallery"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "face_person_association_output"

# Demo threshold only. It must be validated on a larger,
# independent dataset before any real deployment.
FACE_MATCH_THRESHOLD = 0.30


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading person detector...")
    person_detector = PersonDetector(
        model_path="yolo11n.pt",
        confidence=0.35,
        image_size=640,
    )

    print("Loading face recognizer...")
    face_recognizer = FaceRecognizer()

    print("Loading face quality checker...")
    quality_checker = FaceQualityChecker()

    print("Loading face-person associator...")
    associator = FacePersonAssociator(min_containment=0.80)

    print("Loading staff gallery...")
    gallery = face_recognizer.load_gallery(str(GALLERY_DIR))

    if not gallery:
        raise RuntimeError(
            f"No staff face embeddings loaded from {GALLERY_DIR}"
        )

    print(f"Gallery identities: {list(gallery.keys())}")

    image_paths = sorted(
        path
        for path in EVALUATION_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise RuntimeError(
            f"No evaluation images found in {EVALUATION_DIR}"
        )

    total_images = 0
    total_people = 0
    total_faces = 0
    total_associated_faces = 0
    total_quality_rejected = 0
    total_unknown = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"\nSkipping unreadable image: {image_path.name}")
            continue

        total_images += 1

        # 1. Detect people.
        people = person_detector.detect(image)

        # 2. Detect faces and calculate face embeddings.
        faces = face_recognizer.app.get(image)

        # 3. Associate each detected face with a person bounding box.
        associations = associator.associate(faces, people)

        total_people += len(people)
        total_faces += len(faces)

        print(f"\n{'=' * 65}")
        print(f"Image: {image_path.relative_to(EVALUATION_DIR)}")
        print(f"People detected: {len(people)}")
        print(f"Faces detected: {len(faces)}")

        annotated = image.copy()

        # Draw person detections.
        for person_index, person in enumerate(people):
            x1, y1, x2, y2 = map(int, person["bbox"])
            confidence = person["confidence"]

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Person {person_index} ({confidence:.2f})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # Process each detected face.
        for association in associations:
            face_index = association["face_index"]
            person_index = association["person_index"]
            containment = association["containment"]

            face = faces[face_index]
            fx1, fy1, fx2, fy2 = map(int, face.bbox)

            total_associated_faces += (
                1 if person_index is not None else 0
            )

            # Check face quality before attempting identification.
            quality = quality_checker.evaluate(image, face)

            if not quality.acceptable:
                total_quality_rejected += 1
                identity = "Unknown"
                status = f"quality_rejected:{quality.reason}"
                similarity = None

            else:
                # Reuse the embedding already computed by InsightFace.
                embedding = face_recognizer.normalize_embedding(
                    face.normed_embedding
                )

                result = face_recognizer.identify(
                    embedding,
                    gallery,
                    threshold=FACE_MATCH_THRESHOLD,
                )

                identity = result.get("identity", "Unknown")
                status = result.get("status", "unknown")
                similarity = result.get("similarity")

                if identity == "Unknown":
                    total_unknown += 1

            if person_index is None:
                person_text = "No person match"
            else:
                person_text = f"Person {person_index}"

            if similarity is None:
                similarity_text = "N/A"
            else:
                similarity_text = f"{similarity:.3f}"

            print(
                f"Face {face_index}: "
                f"identity={identity}, "
                f"person={person_text}, "
                f"containment={containment:.3f}, "
                f"similarity={similarity_text}, "
                f"status={status}"
            )

            # Draw the face bounding box.
            cv2.rectangle(
                annotated,
                (fx1, fy1),
                (fx2, fy2),
                (255, 255, 0),
                2,
            )

            label = f"{identity} | {person_text}"

            label_y = max(20, fy1 - 8)

            cv2.putText(
                annotated,
                label,
                (fx1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        # Save annotated output.
        output_path = OUTPUT_DIR / (
            f"{image_path.parent.name}_{image_path.stem}_annotated.jpg"
        )

        cv2.imwrite(str(output_path), annotated)
        print(f"Saved output: {output_path}")

    print(f"\n{'=' * 65}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'=' * 65}")
    print(f"Images processed: {total_images}")
    print(f"People detected: {total_people}")
    print(f"Faces detected: {total_faces}")
    print(f"Faces associated with people: {total_associated_faces}")
    print(f"Faces rejected by quality checks: {total_quality_rejected}")
    print(f"Faces left Unknown after recognition: {total_unknown}")
    print(f"Annotated images: {OUTPUT_DIR}")

    print(
        "\nReminder: this is an offline integration test, "
        "not a validated real-time CCTV tracking system."
    )


if __name__ == "__main__":
    main()