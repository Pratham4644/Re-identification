import time
import cv2
import torch
from ultralytics import YOLO

MODELS = {
    "YOLO11n": r"D:\person-reid\yolo11n.pt",
    "YOLO26s": r"D:\person-reid\yolo26s.pt",
}

IMAGE_DIR = r"D:\person-reid\data\evaluation"
IMAGE_SIZE = 416
CONFIDENCE = 0.35
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


def main():
    print("Device: CPU")
    print(f"PyTorch: {torch.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Image size: {IMAGE_SIZE}")
    print()

    # Use the same image set for both models.
    import os

    images = []
    for root, _, files in os.walk(IMAGE_DIR):
        for filename in files:
            path = os.path.join(root, filename)
            image = cv2.imread(path)
            if image is not None:
                images.append((path, image))

    if not images:
        print(f"No readable images found in {IMAGE_DIR}")
        return

    print(f"Images found: {len(images)}")
    print()

    results_summary = []

    for model_name, model_path in MODELS.items():
        print("=" * 60)
        print(f"Testing {model_name}: {model_path}")

        model = YOLO(model_path)

        # Warm up the model.
        for i in range(min(WARMUP_RUNS, len(images))):
            model.predict(
                source=images[i][1],
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                classes=[0],
                device="cpu",
                verbose=False,
            )

        total_time = 0.0
        total_people = 0
        total_confidence = 0.0
        confidence_count = 0
        timings = []

        for i in range(BENCHMARK_RUNS):
            image_path, image = images[i % len(images)]

            start = time.perf_counter()

            prediction = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                classes=[0],
                device="cpu",
                verbose=False,
            )

            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            total_time += elapsed

            boxes = prediction[0].boxes
            count = len(boxes) if boxes is not None else 0
            total_people += count

            if boxes is not None and count:
                scores = boxes.conf.cpu().tolist()
                total_confidence += sum(scores)
                confidence_count += len(scores)

            print(
                f"Run {i + 1:02d}: "
                f"{elapsed * 1000:.1f} ms, "
                f"people detected: {count}, "
                f"image: {os.path.basename(image_path)}"
            )

        average_ms = (total_time / BENCHMARK_RUNS) * 1000
        average_fps = 1.0 / (total_time / BENCHMARK_RUNS)
        average_confidence = (
            total_confidence / confidence_count
            if confidence_count else 0.0
        )

        results_summary.append(
            (
                model_name,
                average_ms,
                average_fps,
                total_people / BENCHMARK_RUNS,
                average_confidence,
            )
        )

        print(f"\n{model_name} results:")
        print(f"Average inference time: {average_ms:.1f} ms")
        print(f"Approximate inference FPS: {average_fps:.2f}")
        print(f"Average people per image: {total_people / BENCHMARK_RUNS:.2f}")
        print(f"Mean confidence: {average_confidence:.3f}")
        print()

        del model

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("-" * 75)
    print(
        f"{'Model':<12} {'Avg ms':>10} {'FPS':>10} "
        f"{'People/image':>15} {'Mean conf.':>12}"
    )

    for name, ms, fps, people, confidence in results_summary:
        print(
            f"{name:<12} {ms:>10.1f} {fps:>10.2f} "
            f"{people:>15.2f} {confidence:>12.3f}"
        )

    print("\nNote: this measures detector inference on still images,")
    print("not end-to-end RTSP tracking or cross-camera Re-ID.")


if __name__ == "__main__":
    main()