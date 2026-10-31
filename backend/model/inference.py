from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from .config import config
from .model_loader import ModelLoader

class HandDetectionBox:
    def __init__(self, box: List[float], confidence: float, class_id: int, class_name: str, glove_color: str = "unknown"):
        self.box = box
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.glove_color = str(glove_color)

class PoseDetectionResult:
    def __init__(
        self,
        person_idx: int,
        person_box: List[float],
        person_conf: float,
        left_wrist: Optional[Dict[str, float]] = None,
        right_wrist: Optional[Dict[str, float]] = None,
        left_shoulder: Optional[Dict[str, float]] = None,
        right_shoulder: Optional[Dict[str, float]] = None,
        left_elbow: Optional[Dict[str, float]] = None,
        right_elbow: Optional[Dict[str, float]] = None,
        torso_x: Optional[float] = None
    ):
        self.person_idx = person_idx
        self.person_box = person_box
        self.person_conf = person_conf
        self.left_wrist = left_wrist
        self.right_wrist = right_wrist
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder
        self.left_elbow = left_elbow
        self.right_elbow = right_elbow
        self.torso_x = torso_x

def evaluate_hand_color(frame: np.ndarray, box: List[float]) -> str:
    """Classifies hand crop color into green, black, bare skin, or other."""
    try:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        if (x2 - x1) < 14 or (y2 - y1) < 14:
            return "unknown"
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Green mask
        lower_green = np.array([35, 40, 40], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = float(np.sum(green_mask > 0)) / float(crop.shape[0] * crop.shape[1])
        if green_ratio > 0.15:
            return "green"

        # Dark/black mask
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.sum(gray < 55)) / float(crop.shape[0] * crop.shape[1])
        if dark_ratio > 0.35:
            return "black"

        return "other"
    except Exception:
        return "unknown"

class ModelInferenceEngine:
    """
    Executes hand detection and pose keypoint estimation on frames.
    """
    def __init__(self, loader: Optional[ModelLoader] = None):
        self.loader = loader or ModelLoader()
        self.hand_model: YOLO = self.loader.load_hand_model()
        self.pose_model: YOLO = self.loader.load_pose_model()

    def predict_hands(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[HandDetectionBox]:
        conf = conf_threshold if conf_threshold is not None else config.HAND_CONF_THRESHOLD
        use_half = config.HALF_PRECISION if str(config.DEVICE) != "cpu" else False
        
        try:
            res = self.hand_model.predict(
                source=frame,
                device=config.DEVICE,
                conf=conf,
                imgsz=config.IMGSZ,
                verbose=False,
                half=use_half
            )[0]
        except Exception as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                res = self.hand_model.predict(
                    source=frame,
                    device="cpu",
                    conf=conf,
                    imgsz=config.IMGSZ,
                    verbose=False,
                    half=False
                )[0]
            else:
                raise e

        detections = []
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            cid = int(classes[i])
            c_score = float(confs[i])
            b = boxes[i].tolist()
            cname = config.CLASS_NAMES.get(cid, "gloved_hand" if cid == 0 else "bare_hand")
            glove_color = evaluate_hand_color(frame, b) if cid == 0 else "bare"
            detections.append(HandDetectionBox(
                box=b,
                confidence=c_score,
                class_id=cid,
                class_name=cname,
                glove_color=glove_color
            ))

        return detections

    def predict_pose(self, frame: np.ndarray) -> List[PoseDetectionResult]:
        use_half = config.HALF_PRECISION if str(config.DEVICE) != "cpu" else False
        try:
            res = self.pose_model.predict(
                source=frame,
                device=config.DEVICE,
                conf=config.PERSON_POSE_CONF_THRESHOLD,
                imgsz=config.IMGSZ,
                verbose=False,
                half=use_half
            )[0]
        except Exception as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                res = self.pose_model.predict(
                    source=frame,
                    device="cpu",
                    conf=config.PERSON_POSE_CONF_THRESHOLD,
                    imgsz=config.IMGSZ,
                    verbose=False,
                    half=False
                )[0]
            else:
                raise e

        results = []
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return results

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts_data = res.keypoints.data.cpu().numpy()

        for idx in range(len(boxes)):
            p_box = boxes[idx].tolist()
            p_conf = float(confs[idx])
            kpts = kpts_data[idx]

            lw, rw, ls, rs, le, re = None, None, None, None, None, None
            if len(kpts) > 10:
                if kpts[9][2] >= config.WRIST_CONF_THRESHOLD:
                    lw = {"x": float(kpts[9][0]), "y": float(kpts[9][1]), "conf": float(kpts[9][2])}
                if kpts[10][2] >= config.WRIST_CONF_THRESHOLD:
                    rw = {"x": float(kpts[10][0]), "y": float(kpts[10][1]), "conf": float(kpts[10][2])}
            if len(kpts) > 6:
                if kpts[5][2] >= 0.2:
                    ls = {"x": float(kpts[5][0]), "y": float(kpts[5][1]), "conf": float(kpts[5][2])}
                if kpts[6][2] >= 0.2:
                    rs = {"x": float(kpts[6][0]), "y": float(kpts[6][1]), "conf": float(kpts[6][2])}
            if len(kpts) > 8:
                if kpts[7][2] >= 0.2:
                    le = {"x": float(kpts[7][0]), "y": float(kpts[7][1]), "conf": float(kpts[7][2])}
                if kpts[8][2] >= 0.2:
                    re = {"x": float(kpts[8][0]), "y": float(kpts[8][1]), "conf": float(kpts[8][2])}

            torso_x = (p_box[0] + p_box[2]) / 2.0
            results.append(PoseDetectionResult(
                person_idx=idx,
                person_box=p_box,
                person_conf=p_conf,
                left_wrist=lw,
                right_wrist=rw,
                left_shoulder=ls,
                right_shoulder=rs,
                left_elbow=le,
                right_elbow=re,
                torso_x=torso_x
            ))

        return results

