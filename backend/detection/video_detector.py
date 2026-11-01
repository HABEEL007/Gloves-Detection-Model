"""
Offline CCTV Video Processing Module.
Executes timestamp-based 3 FPS sampling on source video time,
implements the 6-state temporal compliance state machine,
handles alert deduplication, evidence snapshot capture,
and optional duration-preserving annotated video rendering with carried-forward overlay expiry.
"""
import os
import sys
import time
import math
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np

from ..model.config import config
from ..model.inference import HandDetector, DetectedHand, InferenceResult
from ..model.model_loader import get_model_manager
from .tracker import TrackingAndAssociationPipeline, AssociatedHand

logger = logging.getLogger("backend.video_detector")


class ComplianceState:
    UNKNOWN = "UNKNOWN"
    VERIFYING = "VERIFYING"
    COMPLIANT = "COMPLIANT"
    VIOLATION_PENDING = "VIOLATION_PENDING"
    VIOLATION_CONFIRMED = "VIOLATION_CONFIRMED"
    RECOVERY = "RECOVERY"


class TrackComplianceRecord:
    """Tracks state and violation lifecycle for an individual person track."""
    def __init__(self, track_id: int, initial_pts: float):
        self.track_id = track_id
        self.state = ComplianceState.UNKNOWN
        self.last_pts = initial_pts
        self.first_seen_pts = initial_pts
        
        # Observation counters (at 3 FPS cadence)
        self.consecutive_bare_observations = 0
        self.consecutive_gloved_observations = 0
        self.bare_start_pts: Optional[float] = None
        
        # Violation deduplication and cooldown
        self.current_violation_event_id: Optional[str] = None
        self.last_confirmed_violation_pts: float = -999.0
        self.violation_active: bool = False
        self.total_violations_in_track = 0


class ComplianceEngine:
    """
    6-State Temporal Compliance Verification Engine.
    Rules:
    - RIGHT_HAND_MANDATORY: Right hand must be gloved. Bare left hand is exempt.
    - EITHER_HAND_GLOVED: At least one observed hand must be gloved.
    """
    def __init__(
        self,
        rule_mode: str = config.DEFAULT_RULE_MODE,
        confirm_duration_sec: float = config.VIOLATION_CONFIRMATION_SECONDS,
        min_observations: int = config.MIN_OBSERVATIONS_FOR_CONFIRMATION,
        cooldown_sec: float = config.ALERT_COOLDOWN_SECONDS
    ):
        self.rule_mode = rule_mode
        self.confirm_duration_sec = confirm_duration_sec
        self.min_observations = min_observations
        self.cooldown_sec = cooldown_sec
        self.tracks: Dict[int, TrackComplianceRecord] = {}

    def reset(self):
        self.tracks.clear()

    def evaluate_person(
        self,
        track_id: int,
        associated_hands: List[AssociatedHand],
        current_pts: float
    ) -> Tuple[str, bool, Optional[str], Optional[AssociatedHand]]:
        """
        Evaluates compliance state for a tracked person.
        Returns:
            (current_state, is_newly_confirmed_violation, violation_reason, violating_hand)
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackComplianceRecord(track_id, current_pts)

        rec = self.tracks[track_id]
        rec.last_pts = current_pts

        # Filter hands associated with this track
        person_hands = [h for h in associated_hands if h.track_id == track_id]

        right_hand = next((h for h in person_hands if h.side == "RIGHT"), None)
        left_hand = next((h for h in person_hands if h.side == "LEFT"), None)

        is_bare = False
        is_gloved = False
        reason = ""
        violating_hand = None

        if self.rule_mode == "RIGHT_HAND_MANDATORY":
            # Right hand mandatory rule
            if right_hand is not None:
                if right_hand.class_id == 1:  # bare_hand
                    is_bare = True
                    reason = "Right hand observed bare without glove"
                    violating_hand = right_hand
                elif right_hand.class_id == 0:  # gloved_hand
                    is_gloved = True
            else:
                # Right hand missing / unobserved
                # A missing hand is NOT proof of bare hands or compliance
                pass

        elif self.rule_mode == "EITHER_HAND_GLOVED":
            # At least one hand must be gloved
            has_glove = any(h.class_id == 0 for h in person_hands)
            has_bare = any(h.class_id == 1 for h in person_hands)

            if has_glove:
                is_gloved = True
            elif has_bare:
                is_bare = True
                reason = "No gloved hand observed; bare hand detected"
                violating_hand = next((h for h in person_hands if h.class_id == 1), None)

        newly_confirmed = False

        if is_gloved:
            rec.consecutive_gloved_observations += 1
            rec.consecutive_bare_observations = 0
            rec.bare_start_pts = None

            if rec.violation_active:
                rec.state = ComplianceState.RECOVERY
                rec.violation_active = False
            else:
                rec.state = ComplianceState.COMPLIANT

        elif is_bare:
            rec.consecutive_bare_observations += 1
            rec.consecutive_gloved_observations = 0

            if rec.bare_start_pts is None:
                rec.bare_start_pts = current_pts

            elapsed_bare = current_pts - rec.bare_start_pts

            # Check if cooldown is active
            in_cooldown = (current_pts - rec.last_confirmed_violation_pts) < self.cooldown_sec

            if rec.consecutive_bare_observations >= self.min_observations and elapsed_bare >= self.confirm_duration_sec:
                if not rec.violation_active and not in_cooldown:
                    # New confirmed violation episode
                    rec.state = ComplianceState.VIOLATION_CONFIRMED
                    rec.violation_active = True
                    rec.last_confirmed_violation_pts = current_pts
                    rec.total_violations_in_track += 1
                    rec.current_violation_event_id = str(uuid.uuid4())[:8]
                    newly_confirmed = True
                else:
                    rec.state = ComplianceState.VIOLATION_CONFIRMED
            else:
                rec.state = ComplianceState.VIOLATION_PENDING

        else:
            # No hand observed or uncertain
            if rec.violation_active:
                # Keep active until recovery or timeout
                rec.state = ComplianceState.VIOLATION_CONFIRMED
            else:
                rec.state = ComplianceState.UNKNOWN if not person_hands else ComplianceState.VERIFYING

        return rec.state, newly_confirmed, reason, violating_hand


class EvidenceSnapshotRecorder:
    """Saves dual snapshots (original + annotated) for confirmed violations."""
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.EVIDENCE_DIR

    def save_violation_evidence(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        camera_id: str,
        track_id: Optional[int],
        video_pts: float,
        confidence: float,
        class_name: str,
        reason: str,
        model_name: str
    ) -> Dict[str, str]:
        """Saves both raw and annotated evidence snapshots with unique timestamped filenames."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pts_str = f"{video_pts:.2f}s".replace(".", "_")
        trk_str = f"trk{track_id}" if track_id is not None else "trkUnknown"
        event_uuid = str(uuid.uuid4())[:6]

        raw_name = f"violation_raw_{camera_id}_{trk_str}_pts{pts_str}_{event_uuid}.jpg"
        ann_name = f"violation_ann_{camera_id}_{trk_str}_pts{pts_str}_{event_uuid}.jpg"

        raw_path = self.output_dir / raw_name
        ann_path = self.output_dir / ann_name

        # Save raw frame
        cv2.imwrite(str(raw_path), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Enhance annotated frame with audit stamp
        h, w = annotated_frame.shape[:2]
        stamped = annotated_frame.copy()
        cv2.rectangle(stamped, (0, h - 50), (w, h), (15, 18, 25), -1)
        cv2.line(stamped, (0, h - 50), (w, h - 50), (0, 0, 240), 2)
        stamp_text = (
            f"VIOLATION EVIDENCE | CAM: {camera_id} | {trk_str.upper()} | "
            f"PTS: {video_pts:.2f}s | CONF: {confidence*100:.1f}% | {model_name}"
        )
        cv2.putText(stamped, stamp_text, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(ann_path), stamped, [cv2.IMWRITE_JPEG_QUALITY, 92])

        logger.info(f"Saved violation evidence: raw={raw_name}, ann={ann_name}")
        return {
            "original_snapshot_path": str(raw_path),
            "annotated_snapshot_path": str(ann_path)
        }


class OfflineVideoDetector:
    """
    Offline CCTV Video Detector operating at exactly 3 FPS of video time.
    """
    def __init__(
        self,
        hand_detector: Optional[HandDetector] = None,
        target_fps: float = config.OFFLINE_TARGET_FPS,
        rule_mode: str = config.DEFAULT_RULE_MODE,
        enable_pose_tracking: bool = True,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
        camera_id: str = "OFFLINE_CAM"
    ):
        self.hand_detector = hand_detector or HandDetector(
            conf_threshold=conf_threshold,
            imgsz=imgsz
        )
        self.target_fps = target_fps
        self.rule_mode = rule_mode
        self.camera_id = camera_id
        
        self.tracker_pipeline = TrackingAndAssociationPipeline(enable_pose=enable_pose_tracking)
        self.compliance_engine = ComplianceEngine(rule_mode=rule_mode)
        self.evidence_recorder = EvidenceSnapshotRecorder()

    def process_video(
        self,
        video_path: Union[str, Path],
        output_video_path: Optional[Union[str, Path]] = None,
        sampled_frames_only: bool = False,
        save_snapshots: bool = True
    ) -> Dict[str, Any]:
        """
        Executes offline video processing:
        - Samples exactly 3 inference frames per second of source-video time.
        - Preserves duration and original FPS when rendering output video.
        - Carries forward overlays with expiration.
        - Records confirmed violation episodes.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_source_frames / source_fps if total_source_frames > 0 else 0.0

        logger.info(
            f"Processing video: {video_path.name} ({source_width}x{source_height}, "
            f"{source_fps:.2f} fps, {duration_sec:.2f}s, {total_source_frames} frames)"
        )

        # Video Writer Setup
        writer = None
        if output_video_path:
            out_p = Path(output_video_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_fps = self.target_fps if sampled_frames_only else source_fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, out_fps, (source_width, source_height))

        # Scheduling Variables (timestamp-based)
        sample_interval = 1.0 / self.target_fps  # ~0.3333 seconds
        next_target_pts = 0.0

        # Telemetry & Results
        inference_latencies: List[float] = []
        processed_inference_frames = 0
        total_gloved_detections = 0
        total_bare_detections = 0
        confirmed_violations: List[Dict[str, Any]] = []

        # Carry-forward Overlay Cache
        last_inference_pts: float = -999.0
        cached_hands: List[DetectedHand] = []
        cached_associations: List[AssociatedHand] = []
        cached_tracks: List[Any] = []
        cached_states: Dict[int, str] = {}

        processing_start_wall = time.perf_counter()
        frame_idx = 0

        self.tracker_pipeline.reset()
        self.compliance_engine.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate PTS (Presentation Timestamp) in seconds
            pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_sec = (pts_msec / 1000.0) if pts_msec > 0 else (frame_idx / source_fps)

            # Check if this decoded frame aligns with the 3 FPS schedule
            is_inference_frame = False
            if pts_sec >= (next_target_pts - 0.005):
                is_inference_frame = True
                next_target_pts += sample_interval
                # Prevent drift if timestamps skip
                if next_target_pts < pts_sec:
                    next_target_pts = pts_sec + sample_interval

            if is_inference_frame:
                processed_inference_frames += 1

                # 1. Hand Inference
                t_inf_start = time.perf_counter()
                inf_res = self.hand_detector.detect(frame)
                t_inf_end = time.perf_counter()
                latency_ms = (t_inf_end - t_inf_start) * 1000.0
                inference_latencies.append(latency_ms)

                total_gloved_detections += len(inf_res.gloved_hands)
                total_bare_detections += len(inf_res.bare_hands)

                # 2. Tracking & Association
                assoc_hands, active_tracks = self.tracker_pipeline.process(
                    frame, inf_res.hands, frame_idx=frame_idx
                )

                # 3. Compliance Evaluation
                current_states: Dict[int, str] = {}
                evaluated_track_ids = set()

                for trk in active_tracks:
                    tid = trk.track_id
                    evaluated_track_ids.add(tid)
                    c_state, newly_violated, reason, v_hand = self.compliance_engine.evaluate_person(
                        track_id=tid,
                        associated_hands=assoc_hands,
                        current_pts=pts_sec
                    )
                    current_states[tid] = c_state

                    if newly_violated:
                        v_conf = v_hand.confidence if v_hand else 0.85
                        v_class = v_hand.class_name if v_hand else "bare_hand"

                        # Build annotated snapshot
                        annotated_snapshot = self._render_annotated_frame(
                            frame.copy(),
                            hands=inf_res.hands,
                            associations=assoc_hands,
                            tracks=active_tracks,
                            states=current_states,
                            pts=pts_sec,
                            is_fresh=True
                        )

                        paths = {}
                        if save_snapshots:
                            paths = self.evidence_recorder.save_violation_evidence(
                                raw_frame=frame,
                                annotated_frame=annotated_snapshot,
                                camera_id=self.camera_id,
                                track_id=tid,
                                video_pts=pts_sec,
                                confidence=v_conf,
                                class_name=v_class,
                                reason=reason,
                                model_name=self.hand_detector.manager.architecture
                            )

                        confirmed_violations.append({
                            "event_id": str(uuid.uuid4())[:8],
                            "track_id": tid,
                            "video_timestamp_sec": round(pts_sec, 2),
                            "frame_idx": frame_idx,
                            "rule_mode": self.rule_mode,
                            "reason": reason,
                            "confidence": round(v_conf, 3),
                            "evidence_snapshots": paths
                        })

                # If no tracked persons but hands detected (standalone compliance)
                if not active_tracks and inf_res.bare_hands:
                    # In hand-only mode, bare hands indicate potential violations
                    pass

                # Update carry-forward cache
                last_inference_pts = pts_sec
                cached_hands = inf_res.hands
                cached_associations = assoc_hands
                cached_tracks = active_tracks
                cached_states = current_states

            # Rendering Output Video Frame
            if writer:
                if sampled_frames_only:
                    if is_inference_frame:
                        annotated = self._render_annotated_frame(
                            frame, cached_hands, cached_associations, cached_tracks,
                            cached_states, pts_sec, is_fresh=True
                        )
                        writer.write(annotated)
                else:
                    # Full duration video: Carry forward overlays if not expired
                    overlay_age = pts_sec - last_inference_pts
                    overlay_expired = overlay_age > config.OVERLAY_EXPIRY_SECONDS
                    
                    render_hands = [] if overlay_expired else cached_hands
                    render_assoc = [] if overlay_expired else cached_associations
                    render_tracks = [] if overlay_expired else cached_tracks
                    render_states = {} if overlay_expired else cached_states

                    annotated = self._render_annotated_frame(
                        frame, render_hands, render_assoc, render_tracks,
                        render_states, pts_sec, is_fresh=is_inference_frame
                    )
                    writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        total_processing_wall_sec = time.perf_counter() - processing_start_wall
        processing_fps = frame_idx / total_processing_wall_sec if total_processing_wall_sec > 0 else 0.0

        # Latency statistics
        avg_latency = float(np.mean(inference_latencies)) if inference_latencies else 0.0
        p95_latency = float(np.percentile(inference_latencies, 95)) if inference_latencies else 0.0

        expected_samples = max(1, int(round(duration_sec * self.target_fps)))
        actual_sample_rate = processed_inference_frames / duration_sec if duration_sec > 0 else 0.0

        report = {
            "source_video": str(video_path),
            "source_duration_sec": round(duration_sec, 2),
            "source_fps": round(source_fps, 2),
            "total_source_frames": total_source_frames,
            "processed_inference_frames": processed_inference_frames,
            "expected_inference_samples": expected_samples,
            "actual_inference_sample_rate": round(actual_sample_rate, 2),
            "target_inference_fps": self.target_fps,
            "total_gloved_detections": total_gloved_detections,
            "total_bare_detections": total_bare_detections,
            "confirmed_violation_count": len(confirmed_violations),
            "confirmed_violations": confirmed_violations,
            "avg_inference_latency_ms": round(avg_latency, 2),
            "p95_inference_latency_ms": round(p95_latency, 2),
            "total_processing_duration_sec": round(total_processing_wall_sec, 2),
            "processing_fps": round(processing_fps, 2),
            "output_video_path": str(output_video_path) if output_video_path else None
        }

        logger.info(
            f"Offline processing complete: {processed_inference_frames} samples, "
            f"{len(confirmed_violations)} violations, "
            f"avg latency: {avg_latency:.1f}ms, P95: {p95_latency:.1f}ms"
        )
        return report

    def _render_annotated_frame(
        self,
        frame: np.ndarray,
        hands: List[DetectedHand],
        associations: List[AssociatedHand],
        tracks: List[Any],
        states: Dict[int, str],
        pts: float,
        is_fresh: bool
    ) -> np.ndarray:
        """
        Renders bounding boxes, tracks, status badge, and freshness indicator.
        Carried-forward overlays are rendered with dimmed/secondary styling.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw Person Tracks
        for trk in tracks:
            tid = trk.track_id
            bx = [int(v) for v in trk.box]
            p_state = states.get(tid, ComplianceState.UNKNOWN)

            if p_state == ComplianceState.VIOLATION_CONFIRMED:
                color = (0, 0, 240)  # Red
            elif p_state == ComplianceState.COMPLIANT:
                color = (0, 210, 0)  # Green
            elif p_state == ComplianceState.VIOLATION_PENDING:
                color = (0, 140, 255)  # Orange
            else:
                color = (200, 200, 200)  # Gray

            alpha = 1.0 if is_fresh else 0.65
            cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            lbl = f"Person #{tid} [{p_state}]"
            cv2.putText(out, lbl, (bx[0], max(20, bx[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Draw Hand Detections
        for hand in hands:
            bx = [int(v) for v in hand.box]
            is_bare = (hand.class_id == 1)

            if is_bare:
                box_color = (0, 0, 240) if is_fresh else (0, 0, 160)
            else:
                box_color = (0, 220, 0) if is_fresh else (0, 140, 0)

            thickness = 2 if is_fresh else 1
            cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), box_color, thickness)

            label = f"{hand.class_name} ({hand.confidence*100:.0f}%)"
            cv2.putText(out, label, (bx[0], max(15, bx[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        # 3. HUD Header (Freshness & Timestamp)
        hud_bg = np.zeros((40, w, 3), dtype=np.uint8)
        cv2.rectangle(out, (0, 0), (w, 36), (20, 22, 28), -1)

        fresh_tag = "[FRESH AI 3FPS]" if is_fresh else "[CARRIED-FORWARD OVERLAY]"
        fresh_color = (0, 255, 100) if is_fresh else (180, 180, 180)
        hud_text = f"PTS: {pts:.3f}s | Rule: {self.rule_mode} | {fresh_tag}"
        cv2.putText(out, hud_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fresh_color, 1, cv2.LINE_AA)

        return out

