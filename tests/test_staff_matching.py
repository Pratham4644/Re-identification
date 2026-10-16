from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recognition.face_recognizer import FaceRecognizer


GALLERY_PATH = ROOT / "data" / "staff_gallery"


def main():
    recognizer = FaceRecognizer()

    print("\nLoading staff gallery...")
    gallery = recognizer.load_gallery(GALLERY_PATH)

    print("\nGallery summary:")
    for staff_name, embeddings in gallery.items():
        print(f"  {staff_name}: {len(embeddings)} reference image(s)")

    print("\nEnrollment self-check:")
    print("This check confirms that enrollment images can be processed.")
    print("It does NOT establish recognition accuracy.")

    for staff_name, reference_embeddings in gallery.items():
        for index, embedding in enumerate(reference_embeddings, start=1):
            ranked = recognizer.rank_candidates(embedding, gallery)

            print(f"\nReference: {staff_name}, image {index}")
            for candidate in ranked:
                print(
                    f"  {candidate['name']}: "
                    f"{candidate['similarity']:.4f}"
                )


if __name__ == "__main__":
    main()