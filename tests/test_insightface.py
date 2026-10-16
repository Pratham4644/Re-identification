from pathlib import Path

import cv2
import insightface
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]

GALLERY = ROOT / "data" / "staff_gallery"

IMAGE_PATHS = {
    "Dhruval": GALLERY / "Dhruval" / "face" / "face.jpeg",
    "Prathamesh": GALLERY / "Prathamesh" / "face" / "face_photo.jpg",
    "Saransh": GALLERY / "Saransh" / "face" / "image.png",
}


def main():
    print("OpenCV:", cv2.__version__)
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())

    print("\nInitializing InsightFace model...")

    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    app.prepare(ctx_id=-1, det_size=(640, 640))

    print("Model initialized successfully.")

    for staff_name, image_path in IMAGE_PATHS.items():
        print(f"\nStaff: {staff_name}")
        print("Image:", image_path)

        if not image_path.is_file():
            print("ERROR: Image file not found.")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("ERROR: OpenCV could not read the image.")
            continue

        faces = app.get(image)

        print("Faces detected:", len(faces))

        for index, face in enumerate(faces):
            embedding = face.normed_embedding

            print(f"Face {index + 1}:")
            print("  Bounding box:", face.bbox.astype(int).tolist())
            print("  Embedding dimensions:", embedding.shape)
            print("  Embedding finite:", bool(__import__("numpy").isfinite(embedding).all()))


if __name__ == "__main__":
    main()