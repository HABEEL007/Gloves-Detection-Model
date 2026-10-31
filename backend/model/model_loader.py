import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from ultralytics import YOLO

from .config import config

class ModelLoader:
    """
    Centralized loader for YOLO glove detection & pose estimation models.
    Provides device placement, precision handling, and CUDA OOM fallback.
    """
    def __init__(self, hand_weights_path: Optional[Path] = None, pose_weights_path: Optional[Path] = None):
        self.hand_weights_path = Path(hand_weights_path) if hand_weights_path else config.MODEL_WEIGHTS_PATH
        self.pose_weights_path = Path(pose_weights_path) if pose_weights_path else self._resolve_pose_path()
        self.device = config.DEVICE
        self.hand_model: Optional[YOLO] = None
        self.pose_model: Optional[YOLO] = None

    def _resolve_pose_path(self) -> Path:
        if config.POSE_MODEL_PATH.exists():
            return config.POSE_MODEL_PATH
        if config.FALLBACK_POSE_PATH.exists():
            return config.FALLBACK_POSE_PATH
        return config.PROJECT_ROOT / "yolov8n-pose.pt"

    def load_hand_model(self) -> YOLO:
        if self.hand_model is None:
            if not self.hand_weights_path.exists():
                raise FileNotFoundError(f"Glove model weights not found at: {self.hand_weights_path}")
            self.hand_model = YOLO(str(self.hand_weights_path))
        return self.hand_model

    def load_pose_model(self) -> YOLO:
        if self.pose_model is None:
            if not self.pose_weights_path.exists():
                raise FileNotFoundError(f"Pose model weights not found at: {self.pose_weights_path}")
            self.pose_model = YOLO(str(self.pose_weights_path))
        return self.pose_model

    def load_all(self) -> Tuple[YOLO, YOLO]:
        return self.load_hand_model(), self.load_pose_model()

