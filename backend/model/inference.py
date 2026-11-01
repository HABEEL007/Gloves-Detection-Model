"""
Standalone, reusable hand detection inference module.
Wraps the loaded model manager, handles pre/post-processing timing,
supports color-based skin/glove verification, and returns structured detections.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
import torch

from .config import config
from .model_loader import get_model_manager, GloveModelManager

logger = logging.getLogger("backend.inference")


class DetectedHand:
    """Structured representation of an individual detected hand."""
    def __init__(
        self,
        box: List[float],          # [x1, y1, x2, y2] in pixels
        class_id: int,             # 0 = gloved_hand, 1 = bare_hand
        confidence: float,         # Detection confidence [0.0 - 1.0]
        class_name: str,           # "gloved_hand" or "bare_hand"
        glove_color: str = "unknown"  # "black", "green", "bare", "other", "unknown"
    ):
        self.box = [float(x) for x in box]
        self.class_id = int(class_id)
        self.confidence = float(confidence)
        self.class_name = str(class_name)
        self.glove_color = str(glove_color)
        
        # Derived geometric properties
        self.x1, self.y1, self.x2, self.y2 = self.box
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0
        self.width = max(1.0, self.x2 - self.x1)
        self.height = max(1.0, self.y2 - self.y1)
        self.area = self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(x, 2) for x in self.box],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "glove_color": self.glove_color,
            "center": [round(self.cx, 2), round(self.cy, 2)],
            "dimensions": [round(self.width, 2), round(self.height, 2)]
        }


class InferenceResult:
    """Complete structured output from a single frame inference."""
    def __init__(
        self,
        hands: List[DetectedHand],
        timing: Dict[str, float],
        frame_shape: Tuple[int, int],  # (height, width)
        conf_threshold: float,
        imgsz: int
    ):
        self.hands = hands
        self.timing = timing
        self.frame_shape = frame_shape
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    @property
    def gloved_hands(self) -> List[DetectedHand]:
        return [h for h in self.hands if h.class_id == 0]

    @property
    def bare_hands(self) -> List[DetectedHand]:
        return [h for h in self.hands if h.class_id == 1]

    @property
    def total_hands(self) -> int:
        return len(self.hands)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_hands": len(self.hands),
            "gloved_count": len(self.gloved_hands),
            "bare_count": len(self.bare_hands),
            "detections": [h.to_dict() for h in self.hands],
            "timing_ms": self.timing,
            "frame_shape": list(self.frame_shape),
            "conf_threshold": self.conf_threshold,
            "imgsz": self.imgsz
        }


def evaluate_hand_color(frame: np.ndarray, box: List[float]) -> str:
    """
    Analyzes the hand crop to distinguish skin from black, green, or other gloves.
    Ensures exposed human skin is not misclassified as a dark glove.
    """
    try:
        img_h, img_w = frame.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(img_w, int(box[2]))
        y2 = min(img_h, int(box[3]))

        if (x2 - x1) < 12 or (y2 - y1) < 12:
            return "unknown"

        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        # Sample central 70% of crop to minimize background / sleeve interference
        cy1, cy2 = int(ch * 0.15), int(ch * 0.85)
        cx1, cx2 = int(cw * 0.15), int(cw * 0.85)
        center = crop[cy1:cy2, cx1:cx2] if (cy2 - cy1 >= 8 and cx2 - cx1 >= 8) else crop

        # Downsample before color calculations for speed
        h_, w_ = center.shape[:2]
        if h_ > 48 or w_ > 48:
            scale = 48.0 / max(h_, w_)
            center = cv2.resize(
                center,
                (max(1, int(w_ * scale)), max(1, int(h_ * scale))),
                interpolation=cv2.INTER_NEAREST
            )

        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(center, cv2.COLOR_BGR2YCrCb)

        # 1. Skin tone detector (high Cr, moderate Cb)
        skin_mask = (
            (ycrcb[:, :, 1] >= 130) & (ycrcb[:, :, 1] <= 175) &
            (ycrcb[:, :, 2] >= 75) & (ycrcb[:, :, 2] <= 130) &
            (ycrcb[:, :, 0] > 60) &
            (hsv[:, :, 1] > 18) & (hsv[:, :, 2] > 50)
        )
        skin_ratio = float(np.mean(skin_mask))

        # 2. Green glove: HSV Hue 32-88 with saturation
        green_hsv = (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 35)
        green_bgr = (center[:, :, 1] > center[:, :, 0] * 1.15) & (center[:, :, 1] > center[:, :, 2] * 1.15) & (center[:, :, 1] > 40)
        green_ratio = float(np.mean(green_hsv | green_bgr))

        # 3. Dark/Black glove: V < 120, S < 140 or BGR all < 120
        dark_hsv = (hsv[:, :, 2] < 120) & (hsv[:, :, 1] < 140)
        dark_rgb = (center[:, :, 0] < 120) & (center[:, :, 1] < 120) & (center[:, :, 2] < 120)
        dark_ratio = float(np.mean(dark_hsv | dark_rgb))

        # Decision rules
        if skin_ratio >= 0.12:
            return "bare"
        if green_ratio >= 0.18 and skin_ratio < 0.10:
            return "green"
        if dark_ratio >= 0.38 and skin_ratio < 0.08:
            return "black"
        if skin_ratio >= 0.08:
            return "bare"
        return "other"
    except Exception:
        return "unknown"


class HandDetector:
    """
    Reusable Hand Detector.
    Encapsulates inference execution, timing instrumentation, and color refinement.
    """
    def __init__(
        self,
        model_manager: Optional[GloveModelManager] = None,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None
    ):
        self.manager = model_manager or get_model_manager()
        self.conf_threshold = conf_threshold or config.HAND_CONF_THRESHOLD
        self.imgsz = imgsz or config.IMG_SIZE

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> InferenceResult:
        """
        Executes inference on a single numpy image frame.
        Instruments preprocess, inference, and postprocess latency separately.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid image frame provided for inference.")

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        target_imgsz = imgsz if imgsz is not None else self.imgsz
        h, w = frame.shape[:2]

        t_total_start = time.perf_counter()

        with self.manager.inference_lock:
            try:
                # Execute inference via model predict
                raw_results = self.manager.model.predict(
                    source=frame,
                    device=self.manager.device,
                    conf=conf,
                    imgsz=target_imgsz,
                    verbose=False
                )[0]
            except Exception as e:
                logger.error(f"Inference execution failed: {e}", exc_info=True)
                # Return empty result rather than crashing
                return InferenceResult(
                    hands=[],
                    timing={"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "total_ms": 0.0},
                    frame_shape=(h, w),
                    conf_threshold=conf,
                    imgsz=target_imgsz
                )

        t_post_start = time.perf_counter()

        # Extract timing dict from ultralytics if available
        speed_dict = getattr(raw_results, "speed", {})
        prep_ms = float(speed_dict.get("preprocess", 0.0))
        inf_ms = float(speed_dict.get("inference", 0.0))
        post_ms = float(speed_dict.get("postprocess", 0.0))

        detected_hands: List[DetectedHand] = []

        if raw_results.boxes is not None and len(raw_results.boxes) > 0:
            boxes = raw_results.boxes.xyxy.cpu().numpy()
            confs = raw_results.boxes.conf.cpu().numpy()
            classes = raw_results.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                box = boxes[i].tolist()
                c_score = float(confs[i])
                raw_cid = int(classes[i])

                # Color-based skin/glove verification
                color = evaluate_hand_color(frame, box)

                if color == "bare":
                    # Exposed skin detected -> BARE HAND
                    final_cid = 1
                    final_cname = "bare_hand"
                    glove_color = "bare"
                elif color == "green":
                    final_cid = 0
                    final_cname = "gloved_hand"
                    glove_color = "green"
                elif color == "black":
                    # Dark fabric with zero skin
                    if raw_cid == 0 or c_score >= 0.35:
                        final_cid = 0
                        final_cname = "gloved_hand"
                        glove_color = "black"
                    else:
                        final_cid = 1
                        final_cname = "bare_hand"
                        glove_color = "bare"
                else:
                    # Other / unknown
                    if raw_cid == 0 and c_score >= 0.40:
                        final_cid = 0
                        final_cname = "gloved_hand"
                        glove_color = "other"
                    else:
                        final_cid = raw_cid
                        final_cname = self.manager.class_names.get(raw_cid, "bare_hand")
                        glove_color = "bare" if final_cid == 1 else "other"

                detected_hands.append(DetectedHand(
                    box=box,
                    class_id=final_cid,
                    confidence=c_score,
                    class_name=final_cname,
                    glove_color=glove_color
                ))

        t_total_end = time.perf_counter()
        total_ms = (t_total_end - t_total_start) * 1000.0

        # If speed dict had 0s, estimate
        if inf_ms == 0.0:
            inf_ms = total_ms - prep_ms

        timing = {
            "preprocess_ms": round(prep_ms, 2),
            "inference_ms": round(inf_ms, 2),
            "postprocess_ms": round(post_ms, 2),
            "total_ms": round(total_ms, 2)
        }

        return InferenceResult(
            hands=detected_hands,
            timing=timing,
            frame_shape=(h, w),
            conf_threshold=conf,
            imgsz=target_imgsz
        )

