# Gloves Detection Backend

Simplified, modular backend service for real-time glove compliance verification, worker tracking, and live video streaming.

## Directory Structure

```text
backend/
├── api/
│   ├── main.py              # Application entry point & FastAPI setup
│   ├── routes.py            # Streaming & control API endpoints
│   ├── schemas.py           # Pydantic request/response schemas
│   └── dependencies.py      # Dependency injection & worker lifecycle
│
├── model/
│   ├── best.pt              # Champion YOLO glove detection weights
│   ├── model_loader.py      # Centralized model loader & device manager
│   ├── inference.py         # Forward-pass inference & color evaluation
│   └── config.py            # Hardware, model, & threshold configuration
│
├── detection/
│   ├── video_detector.py    # Pipeline coordinator & compliance evaluator
│   ├── video_processor.py   # RTSP capture & video file stream engine
│   └── tracker.py           # Person tracking & persistent ID association
│
└── README.md
```

## Key Modules

- **API (`api/`)**: Provides HTTP and MJPEG streaming endpoints (`/api/video_feed`, `/api/live_stats`, `/api/stream/set_source`, `/api/stream/control`).
- **Model (`model/`)**: Manages model checkpoint loading (`best.pt`), FP16 acceleration, CUDA/CPU fallback, and color classification.
- **Detection (`detection/`)**: Handles low-latency video decoding, multi-worker tracking, wrist-to-hand association, and configurable compliance business rules.

