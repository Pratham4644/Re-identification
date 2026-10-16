from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recognition.face_recognizer import FaceRecognizer

GALLERY_PATH = ROOT / "data" / "staff_gallery"
EVALUATION_PATH = ROOT / "data" / "evaluation"
OUTPUT_PATH = ROOT / "tests" / "threshold_results.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

THRESHOLDS = [
    round(value / 100, 2)
    for value in range(20, 71, 5)
]

# Exact duplicates of enrolled images, excluded from evaluation.
EXCLUDED_IMAGES = {
    Path("Dhruval")
    / "WhatsApp Image 2026-10-12 at 5.58.41 PM.jpeg",

    Path("Prathamesh")
    / "WIN_20261012_17_25_11_Pro.jpg",
}


def main():
    recognizer = FaceRecognizer()
    gallery = recognizer.load_gallery(GALLERY_PATH)

    records = []
    skipped = []

    for person_dir in sorted(EVALUATION_PATH.iterdir()):
        if not person_dir.is_dir():
            continue

        expected_name = person_dir.name

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(EVALUATION_PATH)

            if relative_path in EXCLUDED_IMAGES:
                print(f"Excluded duplicate: {relative_path}")
                continue

            try:
                embedding = recognizer.extract_embedding(image_path)

                ranked = recognizer.rank_candidates(
                    embedding,
                    gallery,
                )

                if not ranked:
                    skipped.append((relative_path, "No candidates"))
                    continue

                records.append({
                    "image": str(relative_path),
                    "expected": expected_name,
                    "best_name": ranked[0]["name"],
                    "similarity": ranked[0]["similarity"],
                })

            except ValueError as exc:
                skipped.append((relative_path, str(exc)))

    if not records:
        print("No usable evaluation images found.")
        return

    output_rows = []

    print("\nTHRESHOLD EVALUATION")
    print("=" * 90)

    print(
        f"{'Threshold':<12}"
        f"{'Known accepted':<18}"
        f"{'Known rejected':<18}"
        f"{'Unknown accepted':<20}"
        f"{'Unknown rejected':<20}"
    )

    for threshold in THRESHOLDS:
        known_accepted = 0
        known_rejected = 0
        unknown_accepted = 0
        unknown_rejected = 0
        known_wrong_identity = 0

        for record in records:
            expected = record["expected"]
            predicted = record["best_name"]
            similarity = record["similarity"]

            accepted = similarity >= threshold

            if expected == "Unknown":
                if accepted:
                    unknown_accepted += 1
                else:
                    unknown_rejected += 1

            else:
                if not accepted:
                    known_rejected += 1
                elif predicted == expected:
                    known_accepted += 1
                else:
                    known_wrong_identity += 1

        row = {
            "threshold": threshold,
            "known_accepted_correct": known_accepted,
            "known_rejected": known_rejected,
            "known_accepted_wrong_identity": known_wrong_identity,
            "unknown_accepted": unknown_accepted,
            "unknown_rejected": unknown_rejected,
            "known_images": sum(
                record["expected"] != "Unknown"
                for record in records
            ),
            "unknown_images": sum(
                record["expected"] == "Unknown"
                for record in records
            ),
        }

        output_rows.append(row)

        print(
            f"{threshold:<12.2f}"
            f"{known_accepted:<18}"
            f"{known_rejected:<18}"
            f"{unknown_accepted:<20}"
            f"{unknown_rejected:<20}"
        )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nCSV saved to: {OUTPUT_PATH}")

    print("\nSKIPPED IMAGES")
    for image, reason in skipped:
        print(f"{image}: {reason}")

    print(
        "\nNote: Results use a small evaluation set and do not "
        "validate a production threshold."
    )


if __name__ == "__main__":
    main()