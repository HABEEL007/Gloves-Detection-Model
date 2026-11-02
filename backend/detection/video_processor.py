"""
Low-Latency Online CCTV Video Processing Worker.
Decouples RTSP capture and AI inference via a bounded latest-frame buffer,
drops stale frames, tracks frame age and dropped frame counts,
reconnects gracefully on network interruption, and provides non-blocking annotated frames.
"""
import time
import queue
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import cv2
import numpy as np

from ..model.config import config
from ..model.inference import HandDetector, DetectedHand, InferenceResult
from ..model.model_loader import get_model_manager
from .tracker import TrackingAndAssociationPipeline, AssociatedHand
from .video_detector import ComplianceEngine, EvidenceSnapshotRecorder, ComplianceState

logger = logging.getLogger("backend.video_processor")


class CCTVStreamCapture:
    """
    Dedicated capture thread maintaining a strictly bounded latest-frame queue.
    Prevents latency accumulation on slow inference or high RTSP bitrates.
    """
    def __init__(
        self,
        source: str = config.RTSP_DEFAULT_URL,
        reconnect_delay: float = config.RTSP_RECONNECT_DELAY_SEC,
        max_buffer_size: int = config.RTSP_BUFFER_SIZE
    ):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.max_buffer_size = max_buffer_size

        self._queue: queue.Queue = queue.Queue(maxsize=self.max_buffer_size)
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self.total_frames_captured = 0
        self.total_frames_dropped = 0
        self.reconnect_count = 0
        self.is_connected = False
        self.last_frame_capture_time: float = 0.0

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"CCTV capture worker started for source: {self.source}")

    def stop(self):
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_cap()
        logger.info("CCTV capture worker stopped.")

    def _release_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.is_connected = False

    def _capture_loop(self):
        while self._is_running:
            # Connect or reconnect
            if self._cap is None or not self._cap.isOpened():
                self._release_cap()
                logger.info(f"Attempting connection to RTSP stream: {self.source}...")
                
                # FFMPEG low-latency environment flags
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if not self._cap.isOpened():
                    self.reconnect_count += 1
                    time.sleep(self.reconnect_delay)
                    continue
                else:
                    self.is_connected = True
                    logger.info("RTSP stream connected successfully.")

            ret, frame = self._cap.read()
            capture_time = time.time()

            if not ret or frame is None:
                self.is_connected = False
                self.reconnect_count += 1
                logger.warning("Frame grab failed. Stream may have dropped. Reconnecting...")
                self._release_cap()
                time.sleep(self.reconnect_delay)
                continue

            self.total_frames_captured += 1
            self.last_frame_capture_time = capture_time

            # Enforce bounded buffer: if full, drop oldest frame to maintain low latency
            if self._queue.full():
                try:
                    _ = self._queue.get_nowait()
                    self.total_frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait((frame, capture_time))
            except queue.Full:
                self.total_frames_dropped += 1

    def get_latest_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float]]:
        """Retrieves the newest frame and its capture timestamp."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class OnlineCCTVWorker:
    """
    Asynchronous Online CCTV Processing Pipeline:
    - Dedicated Capture thread -> Thread-safe latest-frame queue -> AI Inference worker
    - Frame age tracking (measures true latency from camera to detection)
    - Periodic ByteTrack and Compliance State updates
    - Non-blocking latest JPEG buffer for streaming to web clients
    """
    def __init__(
        self,
        source: str = config.RTSP_DEFAULT_URL,
        camera_id: str = "301",
        rule_mode: str = config.DEFAULT_RULE_MODE,
        target_inference_fps: float = 10.0,
        enable_pose: bool = True,
        conf_threshold: Optional[float] = None
    ):
        self.source = source
        self.camera_id = camera_id
        self.rule_mode = rule_mode
        self.target_inference_interval = 1.0 / max(1.0, float(target_inference_fps))

        # Core Components
        self.capture = CCTVStreamCapture(source=source)
        self.hand_detector = HandDetector(conf_threshold=conf_threshold)
        self.tracker_pipeline = TrackingAndAssociationPipeline(enable_pose=enable_pose)
        self.compliance_engine = ComplianceEngine(rule_mode=rule_mode)
        self.evidence_recorder = EvidenceSnapshotRecorder()

        # Telemetry and State
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_stats: Dict[str, Any] = {}
        self._active_violations_count = 0
        self._total_violations_today = 0

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self.capture.start()
        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker_thread.start()
        logger.info(f"Online CCTV Worker started for Camera {self.camera_id}")

    def stop(self):
        self._is_running = False
        self.capture.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info(f"Online CCTV Worker stopped for Camera {self.camera_id}")

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def get_stats(self) -> Dict[str, Any]:
        with self._frame_lock:
            return dict(self._latest_stats)

    def _inference_loop(self):
        last_inf_time = 0.0
        frame_count = 0
        fps_samples = []

        while self._is_running:
            item = self.capture.get_latest_frame(timeout=0.2)
            if item is None:
                continue

            frame, capture_timestamp = item
            now = time.time()
            frame_age_ms = (now - capture_timestamp) * 1000.0

            # Cadence throttler: do not run inference faster than target_inference_interval
            if (now - last_inf_time) < self.target_inference_interval:
                time.sleep(0.002)
                continue

            last_inf_time = now
            t0 = time.perf_counter()

            # 1. Person Detection & ByteTrack Tracking (Lightweight Stage)
            active_tracks = self.tracker_pipeline.detect_and_track_persons(frame)

            # 2. Person-Gated Hand Detection (Heavy RT-DETR-L Stage)
            # Glove detection service is ONLY activated when at least one person is detected in the frame.
            if len(active_tracks) > 0 or not self.tracker_pipeline.enable_pose:
                inf_res = self.hand_detector.detect(frame)
                current_hands = inf_res.hands
            else:
                inf_res = None
                current_hands = []

            # 3. Association
            assoc_hands = self.tracker_pipeline.associate_hands(frame, current_hands, active_tracks)

            # 3. Compliance Evaluation
            current_states = {}
            active_viol_count = 0

            for trk in active_tracks:
                tid = trk.track_id
                c_state, newly_violated, reason, v_hand = self.compliance_engine.evaluate_person(
                    track_id=tid,
                    associated_hands=assoc_hands,
                    current_pts=now
                )
                current_states[tid] = c_state
                if c_state == ComplianceState.VIOLATION_CONFIRMED:
                    active_viol_count += 1

                if newly_violated:
                    self._total_violations_today += 1
                    # Save dual snapshots asynchronously
                    annotated_frame = self._render_frame(
                        frame.copy(), inf_res.hands, active_tracks, current_states, frame_age_ms
                    )
                    threading.Thread(
                        target=self.evidence_recorder.save_violation_evidence,
                        kwargs={
                            "raw_frame": frame.copy(),
                            "annotated_frame": annotated_frame,
                            "camera_id": self.camera_id,
                            "track_id": tid,
                            "video_pts": now,
                            "confidence": v_hand.confidence if v_hand else 0.80,
                            "class_name": v_hand.class_name if v_hand else "bare_hand",
                            "reason": reason,
                            "model_name": self.hand_detector.manager.architecture
                        },
                        daemon=True
                    ).start()

            self._active_violations_count = active_viol_count

            # Measure inference latency and pipeline FPS
            dt = (time.perf_counter() - t0) * 1000.0
            fps_samples.append(1000.0 / max(dt, 1.0))
            if len(fps_samples) > 30:
                fps_samples.pop(0)
            avg_fps = float(np.mean(fps_samples))

            # 4. Render and Encode JPEG for streaming
            annotated_frame = self._render_frame(
                frame, inf_res.hands, active_tracks, current_states, frame_age_ms
            )

            ret, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self._frame_lock:
                    self._latest_jpeg = buffer.tobytes()
                    self._latest_stats = {
                        "camera_id": self.camera_id,
                        "is_connected": self.capture.is_connected,
                        "reconnect_count": self.capture.reconnect_count,
                        "total_frames_captured": self.capture.total_frames_captured,
                        "total_frames_dropped": self.capture.total_frames_dropped,
                        "frame_age_ms": round(frame_age_ms, 1),
                        "inference_latency_ms": round(dt, 1),
                        "fps": round(avg_fps, 1),
                        "active_violations": self._active_violations_count,
                        "total_violations_today": self._total_violations_today,
                        "gloved_hands_count": len(inf_res.gloved_hands) if inf_res else 0,
                        "bare_hands_count": len(inf_res.bare_hands) if inf_res else 0,
                        "tracked_people": len(active_tracks),
                        "model_architecture": self.hand_detector.manager.architecture,
                        "rule_mode": self.rule_mode
                    }

            frame_count += 1

    def _render_frame(
        self,
        frame: np.ndarray,
        hands: List[DetectedHand],
        tracks: List[Any],
        states: Dict[int, str],
        frame_age_ms: float
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw Person Tracks
        for trk in tracks:
            tid = trk.track_id
            bx = [int(v) for v in trk.box]
            p_state = states.get(tid, ComplianceState.UNKNOWN)

            color = (0, 0, 240) if p_state == ComplianceState.VIOLATION_CONFIRMED else (0, 220, 0)
            cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            cv2.putText(out, f"Staff #{tid} [{p_state}]", (bx[0], max(20, bx[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Hands
        for hand in hands:
            bx = [int(v) for v in hand.box]
            is_bare = (hand.class_id == 1)
            color = (0, 0, 240) if is_bare else (0, 220, 0)
            cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            cv2.putText(out, f"{hand.class_name} {hand.confidence*100:.0f}%",
                        (bx[0], max(15, bx[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Telemetry Overlay
        cv2.rectangle(out, (0, 0), (w, 36), (20, 22, 28), -1)
        status_text = (
            f"CAM-{self.camera_id} | Latency: {frame_age_ms:.0f}ms | "
            f"Violations: {self._active_violations_count} | Mode: {self.rule_mode}"
        )
        cv2.putText(out, status_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out

