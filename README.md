# Person Re-Identification

A Python prototype for detecting people in images or CCTV streams, matching detected faces against an enrolled staff gallery, and experimenting with tracking across cameras.

The repository contains two related layers:

- Reusable components under `src/` for person detection, face recognition, face-quality checks, face/person association, and face alignment.
- Runnable experiments under `tests/` for gallery evaluation, detector checks, live RTSP tracking/recognition, and multi-camera tracking.

This is research and prototyping code. Similarity thresholds, camera behavior, and cross-camera associations have not been established as reliable for production use.

## Features

- **Person detection:** `PersonDetector` uses Ultralytics YOLO with COCO class `0` (person). The default confidence threshold is `0.35`, inference runs on CPU, and detections include an `(x1, y1, x2, y2)` box, confidence, and class ID.
- **Face recognition:** `FaceRecognizer` uses InsightFace `FaceAnalysis` with the `buffalo_l` model and CPU execution. It creates normalized face embeddings and ranks gallery identities by cosine similarity.
- **Open-set demonstration:** `identify()` returns `Unknown` when the best similarity is below the supplied threshold. The default `0.30` threshold is a demonstration value, not a validated operating point.
- **Face quality checks:** `FaceQualityChecker` reports face size, Laplacian blur score, and detector-confidence checks before recognition.
- **Face/person association:** `FacePersonAssociator` associates face boxes with person boxes by face-box containment. This is a geometric association, not identity tracking.
- **Face alignment:** `align_face()` aligns a face from the conventional 68-point landmark layout.
- **Live and multi-camera experiments:** scripts use OpenCV RTSP capture and Ultralytics tracking; the V4 experiment can optionally use an explicit OSNet person-ReID checkpoint for cross-camera body matching.

## Repository layout

```text
.
├── config/
│   ├── cameras.json
│   └── settings.yaml
├── data/
│   ├── evaluation/          # Evaluation images organized by expected identity
│   └── staff_gallery/       # Enrollment images organized by staff identity
├── models/
│   └── haarcascade_frontalface_default.xml
├── src/
│   ├── detection/           # Person detection, face alignment and association
│   ├── recognition/         # Face embeddings and quality checks
│   ├── camera/              # Camera package
│   ├── tracking/            # Tracking package
│   ├── app/                 # Application package
│   └── main.py
├── tests/                   # Evaluation, benchmark and live-camera scripts
├── yolo11n.pt               # Local YOLO weights
└── yolo26s.pt                # Additional local YOLO weights
```

The currently implemented reusable functionality is concentrated in the detection and recognition modules. The camera, tracking, and app packages and `src/main.py` are scaffolding rather than a complete integrated application; the runnable live workflows are in `tests/`.

## Requirements and setup

- Python 3.10 or newer is recommended. The code uses Python 3.10+ type syntax.
- Windows, Linux, or macOS with the native dependencies required by OpenCV, PyTorch, Ultralytics, and InsightFace.
- A working camera/RTSP endpoint for live scripts.

`requirements.txt` is currently empty, so it does not install the project dependencies. Create and activate a virtual environment, then install the packages needed for the workflows you plan to use. For example, in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy opencv-python ultralytics insightface onnxruntime
```

The multi-camera V4 script can additionally use `torchreid` with its dependencies, but only when body-ReID is enabled and an explicit compatible person-ReID checkpoint is provided. That optional setup is not required for face-based recognition or basic person detection.

InsightFace may download its `buffalo_l` model the first time `FaceRecognizer` is initialized. The YOLO scripts use the local weights in the repository by default; the weights must be present at the paths expected by each script.

## Preparing the data

### Staff enrollment gallery

`FaceRecognizer.load_gallery()` expects one directory per identity and a `face` subdirectory containing enrollment images:

```text
data/
└── staff_gallery/
    ├── Staff_Name/
    │   └── face/
    │       ├── front.jpg
    │       └── side.png
    └── Another_Staff_Member/
        └── face/
            └── enrollment.jpg
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, and `.bmp`. Each accepted enrollment image must contain exactly one detectable face. Images that cannot be read, have no face, or contain multiple faces are skipped with a message. The gallery fails to load if no valid embeddings are found.

### Evaluation images

Evaluation scripts use each immediate subdirectory name as the expected identity. The `Unknown` directory is reserved for people who are not enrolled:

```text
data/
└── evaluation/
    ├── Staff_Name/
    │   └── query.jpg
    ├── Another_Staff_Member/
    │   └── query.png
    └── Unknown/
        └── unregistered_person.jpg
```

Use images you have permission to process. Keep real biometric images and derived data private; `data/` is excluded from Git by `.gitignore`.

## Running the scripts

Run commands from the repository root after activating the virtual environment.

### Offline checks and evaluation

```powershell
python tests\test_person_detector.py
python tests\test_face_recognition.py
python tests\test_face_quality.py
python tests\test_face_person_association.py
python tests\evaluate_face_recognition.py
python tests\evaluate_thresholds.py
```

These are standalone scripts, not a conventional pytest suite. They may load neural-network models, require the gallery/evaluation folders, and in some cases write annotated images or CSV results under `tests/`. Check a script’s constants near its top before running it if you need to change a model path, threshold, or output location.

`evaluate_face_recognition.py` writes `tests/face_evaluation_results.csv` with candidate rankings and a preliminary known-person top-1 result. Its unknown-person rows report candidate scores; they do not constitute an open-set false-accept evaluation. `test_face_recognition.py` is a gallery enrollment self-check and does not measure recognition accuracy.

### Live RTSP tracking

The live scripts expect RTSP URLs in environment variables. Set them in the current PowerShell session; do not put camera credentials or URLs in source code or commit them:

```powershell
$env:CAMERA_202_RTSP = "rtsp://<camera-url>"
python tests\live_person_tracking.py
```

For live face recognition against the staff gallery:

```powershell
$env:CAMERA_202_RTSP = "rtsp://<camera-url>"
python tests\live_person_recognition.py
```

The live recognition script performs person tracking and periodically detects/embeds faces, applies face-quality checks, and logs recognition events. Its threshold and temporal voting settings are experimental. Live scripts display a video window; press **Q** in that window to stop when the script indicates that shortcut.

### Multi-camera V4 experiment

Set both camera variables before starting the script:

```powershell
$env:CAMERA_202_RTSP = "rtsp://<camera-202-url>"
$env:CAMERA_302_RTSP = "rtsp://<camera-302-url>"
python tests\multi_camera_staff_tracking_v4_fixed.py
```

The V4 experiment keeps local tracker IDs per camera and can write track and attendance CSVs under `tests/multi_camera_output_v4/`. Face recognition can attach a staff name; body similarity alone is not intended to assign a staff identity. Body-based cross-camera matching requires `V4_REID_CHECKPOINT` to point to an appropriate person-ReID-trained OSNet checkpoint. It is disabled if no usable checkpoint is supplied. The script also supports optional settings such as `STAFF_GALLERY_PATH`, `V4_BODY_REID`, `V4_YOLO_MODEL`, and matching thresholds; see the constants and environment-variable reads in `tests/multi_camera_staff_tracking_v4_fixed.py` before configuring it.

## Important limitations

- Face recognition is biometric processing. Obtain appropriate authorization and comply with applicable privacy, employment, and data-protection requirements before collecting or using images.
- The default similarity and quality thresholds are experimental. Validate them on representative, independently collected footage before making operational decisions.
- Recognition can fail due to image quality, pose, lighting, occlusion, camera differences, or demographic variation. A similarity score is not proof of identity.
- The image-based `FacePersonAssociator` matches boxes only; it does not maintain track IDs.
- Basic YOLO person detection is separate from the more involved live/multi-camera scripts. The standalone tracker experiments rely on Ultralytics tracking configuration and camera behavior.
- Multi-camera body-ReID matching is not meaningful without a compatible person-ReID checkpoint and has not been validated for production attendance.
- `config/cameras.json` and `config/settings.yaml` are configuration placeholders; the current scripts primarily read settings from their own constants and environment variables.
- `requirements.txt` does not yet pin or declare dependencies, so reproducible environment setup is not currently provided.
