from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recognition.face_recognizer import FaceRecognizer


GALLERY_PATH = ROOT / "data" / "staff_gallery"
EVALUATION_PATH = ROOT / "data" / "evaluation"
OUTPUT_PATH = ROOT / "tests" / "face_evaluation_results.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


def get_evaluation_images():
    images = []

    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {EVALUATION_PATH}"
        )

    for person_dir in sorted(EVALUATION_PATH.iterdir()):
        if not person_dir.is_dir():
            continue

        expected_name = person_dir.name

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            images.append((image_path, expected_name))

    return images


def evaluate():
    recognizer = FaceRecognizer()

    print("Loading enrolled staff gallery...")
    gallery = recognizer.load_gallery(GALLERY_PATH)

    print("\nEnrolled identities:")
    for name, embeddings in gallery.items():
        print(f"  {name}: {len(embeddings)} reference image(s)")

    evaluation_images = get_evaluation_images()

    if not evaluation_images:
        raise RuntimeError(
            f"No evaluation images found in {EVALUATION_PATH}"
        )

    rows = []

    for image_path, expected_name in evaluation_images:
        print(f"\nEvaluating: {expected_name}/{image_path.name}")

        try:
            query_embedding = recognizer.extract_embedding(image_path)
            ranked = recognizer.rank_candidates(query_embedding, gallery)

            best_match = ranked[0]
            best_name = best_match["name"]
            best_score = best_match["similarity"]

            correct_top1 = (
                expected_name != "Unknown"
                and best_name == expected_name
            )

            row = {
                "image": str(image_path.relative_to(ROOT)),
                "expected_identity": expected_name,
                "best_candidate": best_name,
                "best_similarity": round(best_score, 6),
                "correct_top1": (
                    correct_top1 if expected_name != "Unknown" else ""
                ),
                "top2_candidate": (
                    ranked[1]["name"] if len(ranked) > 1 else ""
                ),
                "top2_similarity": (
                    round(ranked[1]["similarity"], 6)
                    if len(ranked) > 1 else ""
                ),
                "status": "processed",
            }

            for candidate in ranked:
                print(
                    f"  {candidate['name']}: "
                    f"{candidate['similarity']:.4f}"
                )

            print(f"  Best candidate: {best_name}")
            print(f"  Best similarity: {best_score:.4f}")

            if expected_name == "Unknown":
                print(
                    "  NOTE: This is an unknown-person test. "
                    "A high-scoring staff candidate may be a false match."
                )
            else:
                print(f"  Expected identity: {expected_name}")
                print(f"  Correct top-1 candidate: {correct_top1}")

        except ValueError as exc:
            print(f"  SKIPPED: {exc}")

            row = {
                "image": str(image_path.relative_to(ROOT)),
                "expected_identity": expected_name,
                "best_candidate": "",
                "best_similarity": "",
                "correct_top1": "",
                "top2_candidate": "",
                "top2_similarity": "",
                "status": f"error: {exc}",
            }

        rows.append(row)

    fieldnames = [
        "image",
        "expected_identity",
        "best_candidate",
        "best_similarity",
        "correct_top1",
        "top2_candidate",
        "top2_similarity",
        "status",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    known_rows = [
        row for row in rows
        if row["expected_identity"] != "Unknown"
        and row["status"] == "processed"
    ]

    unknown_rows = [
        row for row in rows
        if row["expected_identity"] == "Unknown"
        and row["status"] == "processed"
    ]

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    if known_rows:
        correct = sum(
            row["correct_top1"] is True
            for row in known_rows
        )

        print(f"Known-person images processed: {len(known_rows)}")
        print(f"Correct top-1 matches: {correct}/{len(known_rows)}")
        print(
            "Top-1 accuracy on this small test set: "
            f"{correct / len(known_rows):.1%}"
        )
        print(
            "This is a preliminary measurement, not a guarantee "
            "of CCTV performance."
        )
    else:
        print("No known-person evaluation images were processed.")

    if unknown_rows:
        print(f"\nUnknown-person images processed: {len(unknown_rows)}")
        print(
            "Unknown images whose highest-scoring candidate is a staff "
            f"member: {len(unknown_rows)}/{len(unknown_rows)}"
        )
        print(
            "These are candidate rankings, not confirmed identity matches. "
            "We have not applied an acceptance threshold yet."
        )
    else:
        print("\nNo unknown-person evaluation images were processed.")

    print(f"\nDetailed results saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate()
    
def identify(
    self,
    query_embedding,
    gallery,
    threshold=0.30,
):
    """
    Prototype open-set identification.

    The threshold is configurable and has NOT been validated
    for production use.
    """
    ranked = self.rank_candidates(query_embedding, gallery)

    if not ranked:
        return {
            "identity": "Unknown",
            "similarity": None,
            "status": "no_candidates",
            "candidates": [],
        }

    best = ranked[0]

    if best["similarity"] < threshold:
        return {
            "identity": "Unknown",
            "similarity": best["similarity"],
            "status": "below_threshold",
            "candidates": ranked,
        }

    return {
        "identity": best["name"],
        "similarity": best["similarity"],
        "status": "candidate_match",
        "candidates": ranked,
    }