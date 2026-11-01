"""
Reusable, thread-safe model loader for Glove Compliance Detection System.
Supports RT-DETR-L and YOLO checkpoints with automatic architecture detection,
CUDA acceleration, CPU fallback, class validation, and shared singleton management.
"""
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch

from .config import config

logger = logging.getLogger("backend.model_loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ModelLoadError(Exception):
    """Raised when model loading or checkpoint validation fails."""
    pass


class GloveModelManager:
    """
    Singleton Manager for hand detection model.
    Guarantees that the checkpoint is loaded once and safely reused across
    multiple threads and inference workers.
    """
    _instance: Optional["GloveModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GloveModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, weights_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        if self._initialized:
            # If path changed and specified, reload
            if weights_path and Path(weights_path) != self._current_weights_path:
                self.load_model(weights_path, device)
            return

        self._inference_lock = threading.Lock()
        self._current_weights_path: Optional[Path] = None
        self._model = None
        self._architecture: str = "Unknown"
        self._device: str = device or config.DEVICE
        self._class_names: Dict[int, str] = {}
        
        target_path = weights_path or config.resolve_model_path()
        self.load_model(target_path, self._device)
        self._initialized = True

    def load_model(self, weights_path: Union[str, Path], device: Optional[str] = None) -> None:
        """
        Loads and validates a detection checkpoint.
        Automatically differentiates between RT-DETR and YOLO architectures.
        """
        path = Path(weights_path).resolve()
        if not path.exists():
            raise ModelLoadError(f"Model checkpoint not found at: {path}")

        chosen_device = device or self._device
        # Validate device compatibility
        if chosen_device != "cpu" and not torch.cuda.is_available():
            logger.warning(f"CUDA requested ('{chosen_device}') but not available. Falling back to CPU.")
            chosen_device = "cpu"

        logger.info(f"Loading checkpoint from: {path} on device: {chosen_device}")

        with self._inference_lock:
            try:
                # 1. Inspect checkpoint metadata
                ckpt_data = torch.load(str(path), map_location="cpu", weights_only=False)
                arch_name = "RT-DETR"
                
                # Check model class name if present
                if "model" in ckpt_data:
                    m_type = str(type(ckpt_data["model"])).lower()
                    if "rtdetr" in m_type:
                        arch_name = "RT-DETR-L"
                    elif "yolo" in m_type:
                        arch_name = "YOLO"

                del ckpt_data # Free temporary dict

                # 2. Instantiate using the specialized class
                if "RT-DETR" in arch_name:
                    from ultralytics import RTDETR
                    model = RTDETR(str(path))
                    self._architecture = "RT-DETR-L"
                else:
                    from ultralytics import YOLO
                    model = YOLO(str(path))
                    self._architecture = "YOLO"

                # 3. Validate class mappings
                raw_names = getattr(model, "names", {})
                if isinstance(raw_names, dict):
                    # Convert keys to int
                    class_names = {int(k): str(v) for k, v in raw_names.items()}
                elif isinstance(raw_names, (list, tuple)):
                    class_names = {i: str(v) for i, v in enumerate(raw_names)}
                else:
                    class_names = {0: "gloved_hand", 1: "bare_hand"}

                # Verification: Must contain gloved and bare classes
                c_vals = [c.lower() for c in class_names.values()]
                has_glove = any("glove" in c for c in c_vals)
                has_bare = any("bare" in c for c in c_vals) or len(class_names) == 2

                if not has_glove:
                    raise ModelLoadError(
                        f"Checkpoint classes {class_names} missing expected glove class."
                    )

                self._model = model
                self._current_weights_path = path
                self._device = chosen_device
                self._class_names = class_names

                logger.info(
                    f"Successfully loaded {self._architecture} from {path.name}. "
                    f"Classes: {self._class_names}, Device: {self._device}"
                )

            except Exception as e:
                logger.error(f"Failed to load checkpoint '{path}': {e}", exc_info=True)
                raise ModelLoadError(f"Could not load checkpoint from {path}: {str(e)}") from e

    @property
    def model(self):
        if self._model is None:
            raise ModelLoadError("Model has not been loaded. Call load_model() first.")
        return self._model

    @property
    def architecture(self) -> str:
        return self._architecture

    @property
    def current_weights_path(self) -> Optional[Path]:
        return self._current_weights_path

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    @property
    def device(self) -> str:
        return self._device

    @property
    def inference_lock(self) -> threading.Lock:
        return self._inference_lock

    def get_model_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the loaded model."""
        return {
            "architecture": self._architecture,
            "weights_path": str(self._current_weights_path) if self._current_weights_path else None,
            "weights_filename": self._current_weights_path.name if self._current_weights_path else None,
            "device": self._device,
            "classes": self._class_names,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }


def get_model_manager(weights_path: Optional[Union[str, Path]] = None) -> GloveModelManager:
    """Convenience getter for the GloveModelManager singleton."""
    return GloveModelManager(weights_path=weights_path)

