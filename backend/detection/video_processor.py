import os
import sys
import time
import urllib.parse
import threading
from pathlib import Path
from typing import Optional, Tuple, Any
import cv2
import numpy as np

# Zero-latency FFMPEG RTSP options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|threads;1|fflags;nobuffer|flags;low_delay|max_delay;0|probesize;32|analyzeduration;0|stimeout;2000000"
)

from ..model.config import config

class VideoProcessor:
    """
    High-performance video and RTSP stream ingest engine.
    Supports low-latency RTSP, offline video playback, seeking, and frame pacing.
    """
    def __init__(self, source: str = config.DEFAULT_RTSP_URL, camera_id: str = "301", loop_video: bool = True):
        self.raw_source = str(source)
        self.camera_id = camera_id
        self.loop_video = loop_video
        self.is_paused = False

        self.source, self.source_type = self._parse_source(self.raw_source)
        self.is_running = False
        self.is_connected = False
        self.frame_counter = 0

        # Video metadata
        self.video_fps = 25.0
        self.video_total_frames = 0
        self.video_current_frame = 0
        self.pending_seek_frame: Optional[int] = None

        if self.source_type == "video":
            self._probe_video_meta(self.raw_source)

        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_frame_time: float = 0.0
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.restart_source_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

    def _parse_source(self, src: str) -> Tuple[Any, str]:
        s = str(src).strip()
        if s.lower().startswith("rtsp://"):
            return s, "rtsp"
        elif s.isdigit():
            return int(s), "webcam"
        else:
            return s, "video"

    def _probe_video_meta(self, path: str):
        try:
            temp = cv2.VideoCapture(str(path))
            if temp.isOpened():
                tot = int(temp.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_val = temp.get(cv2.CAP_PROP_FPS)
                if tot > 0:
                    self.video_total_frames = tot
                if 0 < fps_val <= 120:
                    self.video_fps = fps_val
                temp.release()
        except Exception:
            pass

    def switch_source(self, new_source: str, camera_id: Optional[str] = None, loop_video: bool = True):
        with self.frame_lock:
            self.raw_source = str(new_source)
            self.source, self.source_type = self._parse_source(self.raw_source)
            self.camera_id = str(camera_id or Path(self.raw_source).name)
            self.loop_video = loop_video
            self.is_paused = False
            self.frame_counter = 0
            self.video_current_frame = 0
            self.pending_seek_frame = None
            if self.source_type == "video":
                self._probe_video_meta(self.raw_source)
            self.restart_source_event.set()

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        return self.is_paused

    def seek_frame(self, target_frame: int):
        with self.frame_lock:
            if self.source_type != "video":
                return
            target = max(0, min(max(0, self.video_total_frames - 1), int(target_frame)))
            self.pending_seek_frame = target
            self.new_frame_event.set()

    def get_playback_status(self) -> dict:
        with self.frame_lock:
            fps = self.video_fps or 25.0
            curr_f = self.video_current_frame
            tot_f = self.video_total_frames
            curr_sec = curr_f / fps if fps > 0 else 0.0
            tot_sec = tot_f / fps if fps > 0 else 0.0
            pct = (curr_f / max(1, tot_f) * 100.0) if tot_f > 0 else 0.0
            return {
                "source_type": self.source_type,
                "current_frame": curr_f,
                "total_frames": tot_f,
                "current_sec": round(curr_sec, 1),
                "total_sec": round(tot_sec, 1),
                "progress_pct": round(pct, 2),
                "is_paused": self.is_paused,
                "loop_video": self.loop_video,
                "video_fps": round(fps, 1),
                "current_time_str": f"{int(curr_sec)//60:02d}:{int(curr_sec)%60:02d}",
                "total_time_str": f"{int(tot_sec)//60:02d}:{int(tot_sec)%60:02d}",
            }

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True, name="BackendVideoWorker")
        self.worker_thread.start()

    def stop(self):
        self.is_running = False
        self.new_frame_event.set()
        self.restart_source_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self.frame_lock:
            if self.latest_raw_frame is None:
                return None, 0.0, self.frame_counter
            return self.latest_raw_frame.copy(), self.latest_frame_time, self.frame_counter

    def wait_for_frame(self, timeout: float = 0.05) -> bool:
        flag = self.new_frame_event.wait(timeout=timeout)
        self.new_frame_event.clear()
        return flag

    def _capture_loop(self):
        while self.is_running:
            self.restart_source_event.clear()
            curr_src = self.source
            curr_type = self.source_type

            if curr_type == "rtsp":
                cap = cv2.VideoCapture(curr_src, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif curr_type == "webcam":
                cap = cv2.VideoCapture(int(curr_src))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(str(curr_src))

            if not cap.isOpened():
                self.is_connected = False
                self.restart_source_event.wait(timeout=min(2.0, config.RTSP_RECONNECT_SEC))
                continue

            self.is_connected = True
            fps = cap.get(cv2.CAP_PROP_FPS) if curr_type == "video" else 25.0
            if fps <= 0 or fps > 120:
                fps = 25.0
            frame_interval = 1.0 / fps

            while self.is_running and cap.isOpened():
                if self.restart_source_event.is_set():
                    break

                if self.is_paused:
                    time.sleep(0.04)
                    continue

                if curr_type == "video":
                    with self.frame_lock:
                        seek_f = self.pending_seek_frame
                        self.pending_seek_frame = None
                    if seek_f is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_f)

                t_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    if curr_type == "video" and self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.02)
                        continue
                    else:
                        time.sleep(0.05)
                        break

                self.frame_counter += 1
                with self.frame_lock:
                    self.latest_raw_frame = frame
                    self.latest_frame_time = time.time()
                    if curr_type == "video":
                        self.video_current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.new_frame_event.set()

                if curr_type == "video":
                    elapsed = time.time() - t_start
                    delay = max(0.001, frame_interval - elapsed)
                    time.sleep(delay)

            cap.release()
            self.is_connected = False

