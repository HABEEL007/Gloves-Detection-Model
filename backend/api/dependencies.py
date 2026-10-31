import threading
from typing import Optional
import cv2
import numpy as np

from ..model.config import config
from ..detection.video_processor import VideoProcessor
from ..detection.video_detector import VideoDetector

class BackendContext:
    """
    Central dependency holder managing the live video processor,
    detection engine, frame locks, and worker thread lifecycle.
    """
    def __init__(self):
        # Auto-detect local video if available
        default_source = config.DEFAULT_RTSP_URL
        default_cam_id = "301"

        videos_dir = config.PROJECT_ROOT / "uploads" / "videos"
        if videos_dir.exists():
            vids = list(videos_dir.glob("*.mp4"))
            if vids:
                vids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                default_source = str(vids[0])
                default_cam_id = vids[0].name

        self.processor = VideoProcessor(source=default_source, camera_id=default_cam_id)
        self.detector = VideoDetector()

        self.jpeg_lock = threading.Lock()
        self.new_jpeg_condition = threading.Condition(threading.Lock())
        self.latest_jpeg_bytes: Optional[bytes] = None
        self.latest_stats: dict = {}
        self.is_running = False
        self.consumer_thread: Optional[threading.Thread] = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.processor.start()
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True, name="BackendAPIConsumer")
        self.consumer_thread.start()

    def stop(self):
        self.is_running = False
        self.processor.stop()
        with self.new_jpeg_condition:
            self.new_jpeg_condition.notify_all()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.jpeg_lock:
            return self.latest_jpeg_bytes

    def _consumer_loop(self):
        while self.is_running:
            got_frame = self.processor.wait_for_frame(timeout=0.05)
            if not got_frame:
                continue

            frame, frame_time, frame_num = self.processor.get_latest_frame()
            if frame is None or frame.size == 0:
                continue

            try:
                annotated, stats = self.detector.process_frame(frame, frame_num)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                jpeg_bytes = buf.tobytes()

                with self.jpeg_lock:
                    self.latest_jpeg_bytes = jpeg_bytes
                    self.latest_stats = stats

                with self.new_jpeg_condition:
                    self.new_jpeg_condition.notify_all()
            except Exception as e:
                # Safe fallback to prevent black screen
                try:
                    h, w = frame.shape[:2]
                    disp = cv2.resize(frame, (1280, 720)) if (w != 1280 or h != 720) else frame.copy()
                    cv2.putText(disp, "[AI Syncing Feed]", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    with self.jpeg_lock:
                        self.latest_jpeg_bytes = buf.tobytes()
                    with self.new_jpeg_condition:
                        self.new_jpeg_condition.notify_all()
                except Exception:
                    pass

context = BackendContext()

def get_context() -> BackendContext:
    return context

