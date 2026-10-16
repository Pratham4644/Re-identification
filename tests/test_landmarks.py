from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lbfmodel.yaml"

# OpenCV's built-in Haar cascade face detector.
CASCADE_PATH = ROOT / "models" / "haarcascade_frontalface_default.xml"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Landmark model missing: {MODEL_PATH}")

    image_path = input("Enter the full path to a test face image: ").strip()
    image_path = image_path.strip('"')

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if detector.empty():
        raise RuntimeError("Could not load Haar cascade face detector.")

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    print("Faces detected:", len(faces))

    if len(faces) == 0:
        print("No face detected. Try a clearer, frontal face image.")
        return

    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(MODEL_PATH))

    success, landmarks = facemark.fit(gray, faces)

    print("Landmark prediction successful:", success)

    if not success or len(landmarks) == 0:
        print("No landmarks returned.")
        return

    for index, points in enumerate(landmarks):
        points = points.reshape(-1, 2)

        print(f"Face {index + 1}: {len(points)} landmark points")

        if len(points) != 68:
            print("WARNING: Expected 68 points.")

        # Draw the face rectangle.
        x, y, w, h = faces[index]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw landmarks.
        for px, py in points:
            px, py = int(px), int(py)
            cv2.circle(image, (px, py), 2, (0, 0, 255), -1)

    output_path = ROOT / "tests" / "landmarks_output.jpg"

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError("Failed to save landmark output.")

    print("Output saved to:", output_path)


if __name__ == "__main__":
    main()