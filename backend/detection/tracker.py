"""
Person Tracking and Hand-to-Person Association Module.
Wraps ByteTrack multi-person tracking with Kalman filter motion estimation,
and multi-signal hand-to-wrist association with robust geometric fallbacks.
"""
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..model.config import config
from ..model.inference import DetectedHand

logger = logging.getLogger("backend.tracker")

# Re-use core Pose detector and ByteTrack when available
try:
    from core.pose.pose_detector import YOLO26PoseDetector, PoseDetectionResult
    from core.tracking.byte_tracker import PersonByteTracker, PersonTrack
    POSE_AND_TRACKING_AVAILABLE = True
except ImportError:
    POSE_AND_TRACKING_AVAILABLE = False
    logger.warning("Core pose detector / byte tracker could not be imported. Running in hand-only mode.")


class AssociatedHand:
    """Represents a hand detection associated with a tracked person."""
    def __init__(
        self,
        hand: DetectedHand,
        track_id: Optional[int],
        side: str,                  # "RIGHT", "LEFT", "UNCERTAIN"
        association_confidence: float,
        wrist_distance: float,
        is_staff: bool = True,
        notes: str = ""
    ):
        self.hand = hand
        self.track_id = track_id
        self.side = side
        self.association_confidence = float(association_confidence)
        self.wrist_distance = float(wrist_distance)
        self.is_staff = is_staff
        self.notes = notes

    @property
    def class_id(self) -> int:
        return self.hand.class_id

    @property
    def class_name(self) -> str:
        return self.hand.class_name

    @property
    def confidence(self) -> float:
        return self.hand.confidence

    @property
    def box(self) -> List[float]:
        return self.hand.box

    @property
    def glove_color(self) -> str:
        return self.hand.glove_color

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hand": self.hand.to_dict(),
            "track_id": self.track_id,
            "side": self.side,
            "association_confidence": round(self.association_confidence, 3),
            "wrist_distance": round(self.wrist_distance, 1),
            "is_staff": self.is_staff,
            "notes": self.notes
        }


class TrackingAndAssociationPipeline:
    """
    Unified manager for person tracking and hand-to-person association.
    If pose detector is available, extracts body keypoints and tracks people via ByteTrack.
    If pose is unavailable or disabled, gracefully falls back to unassociated hand tracking.
    """
    def __init__(
        self,
        enable_pose: bool = True,
        high_thresh: Optional[float] = None,
        low_thresh: Optional[float] = None,
        match_thresh: Optional[float] = None,
        track_buffer: Optional[int] = None,
        max_dist_px: Optional[float] = None,
        min_assoc_conf: Optional[float] = None
    ):
        self.enable_pose = enable_pose and POSE_AND_TRACKING_AVAILABLE
        self.high_thresh = high_thresh or config.TRACK_HIGH_THRESH
        self.low_thresh = low_thresh or config.TRACK_LOW_THRESH
        self.match_thresh = match_thresh or config.MATCH_THRESH
        self.track_buffer = track_buffer or config.TRACK_BUFFER
        self.max_dist_px = max_dist_px or config.MAX_ASSOCIATION_DIST_PX
        self.min_assoc_conf = min_assoc_conf or config.MIN_ASSOCIATION_CONF

        self.pose_detector = None
        self.tracker = None

        if self.enable_pose:
            try:
                self.pose_detector = YOLO26PoseDetector()
                self.tracker = PersonByteTracker(
                    high_thresh=self.high_thresh,
                    low_thresh=self.low_thresh,
                    match_thresh=self.match_thresh,
                    buffer_frames=self.track_buffer
                )
                logger.info("Pose detector and ByteTrack initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize pose model: {e}. Falling back to hand-only mode.")
                self.enable_pose = False

    def reset(self):
        """Resets tracker state on stream restart, camera change, or video seek."""
        if self.tracker is not None:
            self.tracker.tracked_tracks.clear()
            self.tracker.lost_tracks.clear()
            self.tracker.frame_id = 0
            logger.info("Tracker state reset.")

    def detect_and_track_persons(self, frame: np.ndarray) -> List[Any]:
        """Runs pose estimation and updates ByteTrack person tracks."""
        if self.enable_pose and self.pose_detector is not None and self.tracker is not None:
            try:
                pose_results = self.pose_detector.predict(frame)
                return self.tracker.update(pose_results)
            except Exception as e:
                logger.error(f"Pose tracking error: {e}", exc_info=True)
                return []
        return []

    def associate_hands(
        self,
        frame: np.ndarray,
        detected_hands: List[DetectedHand],
        active_tracks: List[Any]
    ) -> List[AssociatedHand]:
        """Associates detected hands to tracked persons."""
        img_h, img_w = frame.shape[:2]
        return self._associate(detected_hands, active_tracks, img_w, img_h)

    def process(
        self,
        frame: np.ndarray,
        detected_hands: List[DetectedHand],
        frame_idx: int = 0
    ) -> Tuple[List[AssociatedHand], List[Any]]:
        """Unified runner executing person detection, tracking, and hand association."""
        active_tracks = self.detect_and_track_persons(frame)
        associated_hands = self.associate_hands(frame, detected_hands, active_tracks)
        return associated_hands, active_tracks

    def _associate(
        self,
        hands: List[DetectedHand],
        tracks: List[Any],
        img_w: int,
        img_h: int
    ) -> List[AssociatedHand]:
        """
        Multi-signal hand-to-person association:
        1. Person bounding-box containment.
        2. Distance to detected wrist keypoints (left & right).
        3. Keypoint confidence.
        4. Normalized spatial distance relative to person scale.
        5. Ambiguity check (assigns UNCERTAIN when confidence is low or distance too large).
        """
        results: List[AssociatedHand] = []

        if not hands:
            return results

        if not tracks:
            # No persons tracked in frame
            for h in hands:
                results.append(AssociatedHand(
                    hand=h,
                    track_id=None,
                    side="UNCERTAIN",
                    association_confidence=0.0,
                    wrist_distance=999.0,
                    is_staff=False,
                    notes="No person tracked"
                ))
            return results

        for h in hands:
            hx, hy = h.cx, h.cy
            best_match: Optional[Dict[str, Any]] = None
            min_score = float("inf")

            for trk in tracks:
                p_box = trk.box
                px1, py1, px2, py2 = p_box
                p_w = max(10.0, px2 - px1)
                p_h = max(20.0, py2 - py1)

                # Check if hand is near person (expanded person box by 30%)
                pad_x = p_w * 0.30
                pad_y = p_h * 0.20
                if not (px1 - pad_x <= hx <= px2 + pad_x and py1 - pad_y <= hy <= py2 + pad_y):
                    continue

                # Check wrists if pose data exists
                rw = trk.pose_data.right_wrist if trk.pose_data else None
                lw = trk.pose_data.left_wrist if trk.pose_data else None

                # Test Right Wrist
                if rw and rw.get("conf", 0) > config.WRIST_CONF_THRESHOLD:
                    d_rw = math.hypot(hx - rw["x"], hy - rw["y"])
                    norm_d = d_rw / max(p_h * 0.4, 40.0)
                    if d_rw < self.max_dist_px and norm_d < min_score:
                        min_score = norm_d
                        best_match = {
                            "track_id": trk.track_id,
                            "side": "RIGHT",
                            "dist": d_rw,
                            "conf": max(0.2, min(1.0, 1.0 - (norm_d * 0.6)))
                        }

                # Test Left Wrist
                if lw and lw.get("conf", 0) > config.WRIST_CONF_THRESHOLD:
                    d_lw = math.hypot(hx - lw["x"], hy - lw["y"])
                    norm_d = d_lw / max(p_h * 0.4, 40.0)
                    if d_lw < self.max_dist_px and norm_d < min_score:
                        min_score = norm_d
                        best_match = {
                            "track_id": trk.track_id,
                            "side": "LEFT",
                            "dist": d_lw,
                            "conf": max(0.2, min(1.0, 1.0 - (norm_d * 0.6)))
                        }

                # Fallback: If wrists not detected, check torso/containment
                if best_match is None and (px1 <= hx <= px2 and py1 + p_h * 0.25 <= hy <= py2):
                    # Hand inside person lower half
                    side = "RIGHT" if hx >= (px1 + px2) / 2.0 else "LEFT"
                    best_match = {
                        "track_id": trk.track_id,
                        "side": side,
                        "dist": 150.0,
                        "conf": 0.45
                    }

            if best_match and best_match["conf"] >= self.min_assoc_conf:
                results.append(AssociatedHand(
                    hand=h,
                    track_id=best_match["track_id"],
                    side=best_match["side"],
                    association_confidence=best_match["conf"],
                    wrist_distance=best_match["dist"],
                    is_staff=True,
                    notes=f"Associated with Track {best_match['track_id']} ({best_match['side']})"
                ))
            else:
                results.append(AssociatedHand(
                    hand=h,
                    track_id=best_match["track_id"] if best_match else None,
                    side="UNCERTAIN",
                    association_confidence=best_match["conf"] if best_match else 0.0,
                    wrist_distance=best_match["dist"] if best_match else 999.0,
                    is_staff=False,
                    notes="Ambiguous or distant from tracked wrist"
                ))

        return results

