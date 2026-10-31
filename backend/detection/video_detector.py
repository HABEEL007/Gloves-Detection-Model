import time
import math
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import cv2
import numpy as np

from ..model.config import config
from ..model.inference import ModelInferenceEngine, HandDetectionBox, PoseDetectionResult
from .tracker import PersonTracker, PersonTrack

class AssociatedHand:
    def __init__(self, track_id: Optional[int], hand_box: HandDetectionBox, side: str, distance: float, conf: float):
        self.track_id = track_id
        self.hand_box = hand_box
        self.side = side
        self.distance = distance
        self.conf = conf

class VideoDetector:
    """
    Unified detector orchestrating pose tracking, glove inference,
    wrist-to-hand association, and temporal safety compliance.
    """
    def __init__(self):
        self.inference_engine = ModelInferenceEngine()
        self.tracker = PersonTracker()
        self.right_hand_only_rule = config.RIGHT_HAND_RULE_DEFAULT
        self.conf_threshold = config.HAND_CONF_THRESHOLD

        # Temporal compliance tracking per person
        self.track_states: Dict[int, str] = {}
        self.track_violation_start: Dict[int, float] = {}

    def set_rule_mode(self, right_hand_only: bool):
        self.right_hand_only_rule = bool(right_hand_only)

    def set_hand_conf(self, conf: float):
        self.conf_threshold = max(0.05, min(0.95, float(conf)))

    def _associate_hands_to_tracks(
        self, hands: List[HandDetectionBox], tracks: List[PersonTrack], img_w: int, img_h: int
    ) -> List[AssociatedHand]:
        associated = []
        for h in hands:
            hx = (h.box[0] + h.box[2]) / 2.0
            hy = (h.box[1] + h.box[3]) / 2.0

            best_tid = None
            best_dist = float("inf")
            assigned_side = "UNKNOWN"

            for trk in tracks:
                pose = trk.pose_result
                if not pose:
                    continue

                # Check wrist proximity
                if pose.right_wrist:
                    d_rw = math.hypot(hx - pose.right_wrist["x"], hy - pose.right_wrist["y"])
                    if d_rw < best_dist and d_rw < config.ASSOCIATION_DISTANCE_MAX_PX:
                        best_dist = d_rw
                        best_tid = trk.track_id
                        assigned_side = "RIGHT"

                if pose.left_wrist:
                    d_lw = math.hypot(hx - pose.left_wrist["x"], hy - pose.left_wrist["y"])
                    if d_lw < best_dist and d_lw < config.ASSOCIATION_DISTANCE_MAX_PX:
                        best_dist = d_lw
                        best_tid = trk.track_id
                        assigned_side = "LEFT"

            associated.append(AssociatedHand(
                track_id=best_tid,
                hand_box=h,
                side=assigned_side,
                distance=best_dist if best_tid is not None else 0.0,
                conf=h.confidence
            ))
        return associated

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        h, w = frame.shape[:2]

        # 1. Pose estimation & ByteTrack
        pose_results = self.inference_engine.predict_pose(frame)
        active_tracks = self.tracker.update(pose_results)

        # 2. Hand / Glove Detection
        hands = self.inference_engine.predict_hands(frame, conf_threshold=self.conf_threshold)

        # 3. Geometric Association
        associated = self._associate_hands_to_tracks(hands, active_tracks, w, h)

        # 4. Evaluate Compliance
        gloves_detected = []
        violations = 0
        now = time.time()

        for ah in associated:
            hb = ah.hand_box
            if hb.class_id == 0:  # gloved_hand
                gloves_detected.append({
                    "box": [float(c) for c in hb.box],
                    "conf": float(hb.confidence),
                    "color": hb.glove_color,
                    "side": ah.side,
                    "track_id": ah.track_id
                })
            else:  # bare_hand
                if self.right_hand_only_rule and ah.side == "LEFT":
                    # Left hand exempt under right-hand mandatory rule
                    pass
                else:
                    violations += 1

        # 5. Render HUD onto 1280x720 canvas
        disp_w, disp_h = config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT
        canvas = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        sx, sy = disp_w / float(w), disp_h / float(h)

        # Draw Person tracks
        for trk in active_tracks:
            b = trk.bbox
            cv2.rectangle(
                canvas,
                (int(b[0] * sx), int(b[1] * sy)),
                (int(b[2] * sx), int(b[3] * sy)),
                (255, 255, 255),
                1
            )
            cv2.putText(
                canvas,
                f"Worker #{trk.track_id}",
                (int(b[0] * sx), max(20, int(b[1] * sy) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1
            )

        # Draw Hands
        for ah in associated:
            b = ah.hand_box.box
            is_glove = (ah.hand_box.class_id == 0)
            color = (34, 197, 94) if is_glove else (68, 68, 239)  # Green vs Red
            cv2.rectangle(
                canvas,
                (int(b[0] * sx), int(b[1] * sy)),
                (int(b[2] * sx), int(b[3] * sy)),
                color,
                2
            )
            label = f"{ah.hand_box.class_name} ({int(ah.hand_box.confidence * 100)}%)"
            cv2.putText(
                canvas,
                label,
                (int(b[0] * sx), max(18, int(b[1] * sy) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

        latency_ms = (time.time() - t0) * 1000.0
        fps = 1000.0 / max(1.0, latency_ms)

        stats = {
            "status": "COMPLIANT" if violations == 0 else f"VIOLATION: {violations} Bare Hand(s)",
            "badge_type": "success" if violations == 0 else "danger",
            "active_violations": violations,
            "detected_gloves": gloves_detected,
            "gloves_count": len(gloves_detected),
            "tracked_people": len(active_tracks),
            "latency_ms": round(latency_ms, 1),
            "fps": round(fps, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return canvas, stats

