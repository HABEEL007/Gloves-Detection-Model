# Glove Compliance Detection Backend Module

Production-ready computer vision and compliance verification backend for jewellery-store CCTV environments. Supports online RTSP streams, offline video processing at exactly 3 AI inference FPS via timestamp-based scheduling, ByteTrack employee tracking, multi-signal hand-person association, temporal compliance state machines, and dual evidence snapshots.

---

## 1. Architecture Overview

```
backend/
├── api/
│   ├── main.py          # FastAPI application entrypoint & static mount
│   ├── routes.py        # RESTful endpoints (Health, Model, Image, Video Jobs)
│   ├── schemas.py       # Pydantic request/response schemas
│   └── dependencies.py  # API key authentication & model manager injection
├── model/
│   ├── best.pt          # Active RT-DETR-L model weights
│   ├── config.py        # Centralized detection & system configuration
│   ├── model_loader.py  # Thread-safe singleton model loader (RT-DETR-L & YOLO)
│   └── inference.py     # Pre/post-processing timing instrumentation & hand detector
├── detection/
│   ├── tracker.py       # ByteTrack Kalman tracking & wrist-to-hand association
│   ├── video_detector.py# Offline 3 FPS timestamp-based video processing & compliance
│   └── video_processor.py# Online low-latency RTSP capture & streaming worker
└── README.md
```

---

## 2. Verified Model Information

- **Trained Model Location:** `runs/detect/glove_rtdetr_pure_cctv/weights/best.pt`
- **Architecture:** `RT-DETR-L` (HGNetv2 backbone + AIFI Transformer + RepC3 + RTDETRDecoder)
- **Dataset Domain:** 100% genuine jewellery-store CCTV images (2,417 train / 554 validation). Out-of-domain medical noise (30.8%) completely eliminated.
- **Classes:**
  - `0: gloved_hand`
  - `1: bare_hand`

### Empirical Performance Benchmark (554 Untouched CCTV Validation Images)

| Setting | Overall Precision | Overall Recall | mAP@50 | mAP@50-95 | Gloved Precision | Gloved Recall | Bare Precision | Bare Recall | Latency / Img |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **conf=0.10, imgsz=640** | 59.7% | 60.9% | 54.1% | 30.9% | 77.7% | 72.5% | 41.8% | 49.4% | ~25.4 ms |
| **conf=0.15, imgsz=640** | 59.7% | 60.9% | 52.8% | 30.0% | 77.7% | 72.5% | 41.8% | 49.4% | ~25.3 ms |
| **conf=0.20, imgsz=640** *(Recommended)* | **59.7%** | **60.9%** | **51.3%** | **29.4%** | **77.7%** | **72.5%** | **41.8%** | **49.4%** | **~24.6 ms** |
| **conf=0.25, imgsz=640** | 59.7% | 60.9% | 50.7% | 29.2% | 77.7% | 72.5% | 41.8% | 49.4% | ~25.0 ms |
| **conf=0.30, imgsz=640** | 59.7% | 60.9% | 50.7% | 29.2% | 77.7% | 72.5% | 41.8% | 49.4% | ~25.2 ms |
| **conf=0.20, imgsz=800** | 53.3% | 62.8% | 44.5% | 24.5% | 71.9% | 72.3% | 34.6% | **53.2%** | ~40.4 ms |

---

## 3. Key Capabilities

### A. Offline 3 FPS Video Processing (`backend/detection/video_detector.py`)
- **Timestamp-Based Scheduling:** Uses source video Presentation Timestamp (PTS) rather than crude frame skips (`0.000s`, `0.333s`, `0.667s`...).
- **Playback Duration Preservation:** When writing full-duration output video, original video FPS, frame count, and dimensions are strictly preserved.
- **Carried-Forward Visualization:** Non-inference frames display carried-forward boxes with an explicit `[CARRIED-FORWARD OVERLAY]` badge. Stale overlays expire after `OVERLAY_EXPIRY_SECONDS` (0.45s). Carried-forward frames never trigger violation events.
- **Sampled-Only Mode:** Optional `sampled_frames_only=True` outputs only the 3 FPS analyzed frames.

### B. Online Low-Latency CCTV Processing (`backend/detection/video_processor.py`)
- **Decoupled Architecture:** Dedicated capture thread with bounded ring-buffer (`max_buffer_size=2`). Stale frames are discarded automatically if the AI worker falls behind.
- **Frame Age Telemetry:** Tracks true glass-to-display latency from camera grab to inference output.
- **Automatic Reconnection:** Handles RTSP network drops with graceful exponential backoff.

### C. Compliance Verification Engine
- **Supported Rules:**
  - `RIGHT_HAND_MANDATORY` (Rule A): Right hand must wear a glove; bare left hand alone is not a violation.
  - `EITHER_HAND_GLOVED` (Rule B): At least one reliably observed hand must wear a glove.
- **6-State Temporal Logic:** `UNKNOWN` → `VERIFYING` → `COMPLIANT` → `VIOLATION_PENDING` → `VIOLATION_CONFIRMED` → `RECOVERY`. Missing hands are never assumed to be bare or compliant.
- **Alert Deduplication:** Unique event ID per violation episode with configurable cooldown (`ALERT_COOLDOWN_SECONDS=15.0`).
- **Dual Evidence Snapshots:** Writes both raw unannotated frames and legal audit stamped annotated snapshots to disk.

---

## 4. API Reference

Start the server:
```bash
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Authentication: Pass `X-API-Key: gloves-cctv-secure-key-2026` or `?api_key=...`.

### Endpoints
- `GET /api/v1/health`: Service health and GPU availability.
- `GET /api/v1/model/info`: Active model architecture, device, and class mappings.
- `POST /api/v1/detect/image`: Hand detection on raw binary image, base64, or server image path.
- `POST /api/v1/detect/video`: Submits an asynchronous 3 FPS offline video processing job.
- `GET /api/v1/jobs/{job_id}`: Polls job processing status.
- `GET /api/v1/jobs/{job_id}/result`: Retrieves full compliance metrics and violation timestamps.
- `GET /api/v1/jobs/{job_id}/download`: Downloads the annotated output video.

---

## 5. Execution Commands

### Run Offline Video Processing via Python:
```python
from backend.detection.video_detector import OfflineVideoDetector

detector = OfflineVideoDetector(target_fps=3.0, rule_mode="RIGHT_HAND_MANDATORY")
report = detector.process_video(
    video_path="evidence/detected_output.mp4",
    output_video_path="evidence/annotated_run.mp4",
    sampled_frames_only=False
)
print(report)
```

### Run Standalone Image Inference:
```python
import cv2
from backend.model.inference import HandDetector

detector = HandDetector(conf_threshold=0.20)
frame = cv2.imread("test.jpg")
result = detector.detect(frame)
print(f"Detected {result.total_hands} hands in {result.timing['total_ms']}ms")
```

