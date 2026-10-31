import os
import sys
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

# Windows CUDA memory configuration
if sys.platform.startswith("win"):
    cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in cuda_alloc_conf:
        parts = [p.strip() for p in cuda_alloc_conf.split(",") if not p.strip().startswith("expandable_segments")]
        if parts:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

@dataclass
class BackendConfig:
    BACKEND_DIR: Path = BACKEND_DIR
    PROJECT_ROOT: Path = PROJECT_ROOT

    # Model Weights
    MODEL_WEIGHTS_PATH: Path = BACKEND_DIR / "model" / "best.pt"
    POSE_MODEL_PATH: Path = PROJECT_ROOT / "yolo26n-pose.pt"
    FALLBACK_POSE_PATH: Path = PROJECT_ROOT / "yolov8n-pose.pt"

    # Class mappings
    CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: {
        0: "gloved_hand",
        1: "bare_hand"
    })

    # Device & Precision
    DEVICE: str = "0" if torch.cuda.is_available() else "cpu"
    IMGSZ: int = 640
    HALF_PRECISION: bool = True

    # Detection Thresholds
    HAND_CONF_THRESHOLD: float = 0.25
    BARE_VIOLATION_CONF_THRESHOLD: float = 0.70
    WRIST_CONF_THRESHOLD: float = 0.20
    PERSON_POSE_CONF_THRESHOLD: float = 0.25
    COLOR_CONF_THRESHOLD: float = 0.40

    # Geometric Association
    ASSOCIATION_DISTANCE_MAX_PX: float = 240.0
    ASSOCIATION_CONFIDENCE_MIN: float = 0.45
    RIGHT_HAND_RULE_DEFAULT: bool = False

    # Temporal Compliance
    BARE_PERSISTENCE_SECONDS: float = 1.8
    MINIMUM_STABLE_FRAMES: int = 20
    VIOLATION_COOLDOWN_SECONDS: float = 15.0
    TRACK_EXPIRY_SECONDS: float = 5.0
    MISSING_DETECTION_GRACE_SECONDS: float = 1.5

    # Streaming & Video
    DEFAULT_RTSP_URL: str = "rtsp://admin:Syn@vos@456@192.168.1.128:554/Streaming/Channels/301"
    RTSP_MAX_FAILURES: int = 30
    RTSP_RECONNECT_SEC: float = 2.0
    DISPLAY_WIDTH: int = 1280
    DISPLAY_HEIGHT: int = 720

config = BackendConfig()

