"""
Centralized Configuration for Glove Compliance Detection Module
Defines paths, model hyperparameters, tracking, compliance rules, and API settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os
import sys
import warnings
import torch

# Clean PyTorch Windows allocator warnings
warnings.filterwarnings("ignore", message=".*expandable_segments.*")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@dataclass
class DetectionConfig:
    # Model Weights Paths
    BASE_DIR: Path = BASE_DIR
    LOCAL_BEST_PATH: Path = Path(__file__).resolve().parent / "best.pt"
    PURE_CCTV_MODEL_PATH: Path = BASE_DIR / "runs" / "detect" / "glove_rtdetr_pure_cctv" / "weights" / "best.pt"
    AUGMENTED_MODEL_PATH: Path = BASE_DIR / "runs" / "detect" / "glove_rtdetr_augmented_final" / "weights" / "best.pt"
    CHAMPION_MODEL_PATH: Path = BASE_DIR / "runs" / "detect" / "champion_model" / "weights" / "best.pt"
    POSE_MODEL_PATH: Path = BASE_DIR / "yolo26n-pose.pt"
    FALLBACK_POSE_MODEL_PATH: Path = BASE_DIR / "yolov8n-pose.pt"

    # Active Model Path (configurable, default: local backend/model/best.pt)
    MODEL_WEIGHTS_PATH: Path = LOCAL_BEST_PATH
    
    # Class mapping verified from trained checkpoint
    CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: {
        0: "gloved_hand",
        1: "bare_hand"
    })
    
    # Hardware & Precision
    DEVICE: str = "0" if torch.cuda.is_available() else "cpu"
    HALF_PRECISION: bool = True  # FP16 acceleration where supported
    IMG_SIZE: int = 640          # Default 640x640 input resolution
    
    # Inference Thresholds
    HAND_CONF_THRESHOLD: float = 0.20        # Default confidence for hand detection
    BARE_VIOLATION_CONF_THRESHOLD: float = 0.45  # Confirmed bare hand confidence threshold
    POSE_CONF_THRESHOLD: float = 0.25        # Person pose confidence
    WRIST_CONF_THRESHOLD: float = 0.20       # Wrist keypoint confidence
    
    # Offline Processing Cadence
    OFFLINE_TARGET_FPS: float = 3.0          # Exactly 3 inference frames per second of video time
    OVERLAY_EXPIRY_SECONDS: float = 0.45     # Maximum age before an overlay box is expired/stale
    
    # Tracking Parameters (ByteTrack)
    TRACK_HIGH_THRESH: float = 0.45
    TRACK_LOW_THRESH: float = 0.15
    NEW_TRACK_THRESH: float = 0.50
    TRACK_BUFFER: int = 45                   # Frames to retain lost tracks (~1.5s at 30fps)
    MATCH_THRESH: float = 0.75
    FUSE_SCORE: bool = False
    
    # Hand-to-Person Association Parameters
    MAX_ASSOCIATION_DIST_PX: float = 260.0
    MIN_ASSOCIATION_CONF: float = 0.40
    
    # Compliance Rule Settings
    # Modes: "RIGHT_HAND_MANDATORY" (Rule A), "EITHER_HAND_GLOVED" (Rule B)
    DEFAULT_RULE_MODE: str = "RIGHT_HAND_MANDATORY"
    
    # Temporal State Machine Settings
    # 6 States: UNKNOWN, VERIFYING, COMPLIANT, VIOLATION_PENDING, VIOLATION_CONFIRMED, RECOVERY
    VIOLATION_CONFIRMATION_SECONDS: float = 1.0  # Duration required to confirm violation
    MIN_OBSERVATIONS_FOR_CONFIRMATION: int = 3   # At 3 FPS, 3 consecutive observations = ~1 sec
    ALERT_COOLDOWN_SECONDS: float = 15.0         # Cooldown before a new episode can trigger
    
    # Evidence & Snapshots
    EVIDENCE_DIR: Path = BASE_DIR / "evidence"
    SNAPSHOT_ENABLED: bool = True
    SAVE_ANNOTATED_SNAPSHOT: bool = True
    SAVE_ORIGINAL_SNAPSHOT: bool = True
    
    # Online RTSP Stream Settings
    RTSP_DEFAULT_URL: str = os.getenv(
        "RTSP_URL",
        "rtsp://admin:hanif@admin786@203.99.178.116:8080/cam/realmonitor?channel=1&subtype=0"
    )
    RTSP_RECONNECT_DELAY_SEC: float = 2.0
    RTSP_BUFFER_SIZE: int = 2                 # Bounded latest-frame buffer (prevents latency buildup)
    
    # API Settings
    API_TITLE: str = "Gloves Compliance Detection API"
    API_VERSION: str = "2.0.0"
    API_KEY: Optional[str] = os.getenv("API_KEY", "gloves-cctv-secure-key-2026")
    
    def resolve_model_path(self, override_path: Optional[str] = None) -> Path:
        """Resolves the preferred hand detection model path with graceful fallback."""
        if override_path and Path(override_path).exists():
            return Path(override_path)
        if self.LOCAL_BEST_PATH.exists():
            return self.LOCAL_BEST_PATH
        if self.MODEL_WEIGHTS_PATH.exists():
            return self.MODEL_WEIGHTS_PATH
        if self.PURE_CCTV_MODEL_PATH.exists():
            return self.PURE_CCTV_MODEL_PATH
        if self.AUGMENTED_MODEL_PATH.exists():
            return self.AUGMENTED_MODEL_PATH
        if self.CHAMPION_MODEL_PATH.exists():
            return self.CHAMPION_MODEL_PATH
        raise FileNotFoundError("No valid glove detection model checkpoint found.")

    def resolve_pose_path(self) -> Optional[Path]:
        """Resolves pose detector path if available."""
        if self.POSE_MODEL_PATH.exists():
            return self.POSE_MODEL_PATH
        if self.FALLBACK_POSE_MODEL_PATH.exists():
            return self.FALLBACK_POSE_MODEL_PATH
        return None

# Singleton config instance
config = DetectionConfig()

