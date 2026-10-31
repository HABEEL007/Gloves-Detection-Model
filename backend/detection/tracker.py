from typing import List, Dict, Any, Optional
import time
import numpy as np

class PersonTrack:
    def __init__(self, track_id: int, bbox: List[float], conf: float, pose_result: Any):
        self.track_id = int(track_id)
        self.bbox = [float(c) for c in bbox]
        self.conf = float(conf)
        self.pose_result = pose_result
        self.last_seen = time.time()
        self.missed_frames = 0
        self.history: List[List[float]] = [self.bbox]

class PersonTracker:
    """
    ByteTrack-style IoU & spatial distance person tracker for maintaining continuous track IDs.
    """
    def __init__(self, max_missing_sec: float = 3.0, iou_thresh: float = 0.3):
        self.max_missing_sec = max_missing_sec
        self.iou_thresh = iou_thresh
        self.active_tracks: Dict[int, PersonTrack] = {}
        self.next_id = 1

    def _calc_iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, pose_results: List[Any]) -> List[PersonTrack]:
        now = time.time()
        # Expire stale tracks
        expired = [tid for tid, trk in self.active_tracks.items() if (now - trk.last_seen) > self.max_missing_sec]
        for tid in expired:
            del self.active_tracks[tid]

        if not pose_results:
            return list(self.active_tracks.values())

        unmatched_dets = list(range(len(pose_results)))
        matched_tracks = set()

        # Match by IoU
        for tid, trk in self.active_tracks.items():
            best_iou = 0.0
            best_det_idx = -1
            for d_idx in unmatched_dets:
                iou = self._calc_iou(trk.bbox, pose_results[d_idx].person_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx
            if best_iou >= self.iou_thresh and best_det_idx >= 0:
                det = pose_results[best_det_idx]
                trk.bbox = det.person_box
                trk.conf = det.person_conf
                trk.pose_result = det
                trk.last_seen = now
                trk.missed_frames = 0
                trk.history.append(det.person_box)
                if len(trk.history) > 30:
                    trk.history.pop(0)
                matched_tracks.add(tid)
                unmatched_dets.remove(best_det_idx)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = pose_results[d_idx]
            new_track = PersonTrack(
                track_id=self.next_id,
                bbox=det.person_box,
                conf=det.person_conf,
                pose_result=det
            )
            self.active_tracks[self.next_id] = new_track
            self.next_id += 1

        return list(self.active_tracks.values())

