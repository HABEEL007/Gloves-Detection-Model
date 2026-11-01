"""
Pydantic schemas for the Glove Compliance Detection API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "Glove Compliance Detection System"
    version: str = "2.0.0"
    model_loaded: bool = True
    device: str = "0"
    timestamp: str


class ModelInfoResponse(BaseModel):
    architecture: str
    weights_path: Optional[str]
    weights_filename: Optional[str]
    device: str
    classes: Dict[int, str]
    cuda_available: bool
    gpu_name: Optional[str]


class HandDetectionSchema(BaseModel):
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2] in pixels")
    class_id: int = Field(..., description="0 for gloved_hand, 1 for bare_hand")
    class_name: str
    confidence: float
    glove_color: str
    center: List[float]
    dimensions: List[float]


class TimingBreakdownSchema(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class ImageDetectResponse(BaseModel):
    total_hands: int
    gloved_count: int
    bare_count: int
    detections: List[HandDetectionSchema]
    timing_ms: TimingBreakdownSchema
    frame_shape: List[int]
    conf_threshold: float
    imgsz: int


class VideoJobCreateRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the input video file")
    output_video_path: Optional[str] = Field(None, description="Optional path to save annotated output video")
    sampled_frames_only: bool = Field(False, description="If True, output video contains only the 3 FPS sampled frames")
    rule_mode: str = Field("RIGHT_HAND_MANDATORY", description="Compliance rule: 'RIGHT_HAND_MANDATORY' or 'EITHER_HAND_GLOVED'")
    conf_threshold: Optional[float] = Field(0.20, description="Confidence threshold for hand detection")
    camera_id: Optional[str] = Field("OFFLINE_01", description="Camera or source identifier")


class VideoJobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="PENDING, PROCESSING, COMPLETED, FAILED")
    created_at: str
    completed_at: Optional[str] = None
    progress: float = Field(0.0, description="0.0 to 1.0")
    error: Optional[str] = None
    output_video_path: Optional[str] = None


class ViolationSnapshotSchema(BaseModel):
    original_snapshot_path: Optional[str] = None
    annotated_snapshot_path: Optional[str] = None


class ViolationEventSchema(BaseModel):
    event_id: str
    track_id: Optional[int]
    video_timestamp_sec: float
    frame_idx: int
    rule_mode: str
    reason: str
    confidence: float
    evidence_snapshots: Dict[str, str]


class VideoJobResultResponse(BaseModel):
    job_id: str
    status: str
    source_video: str
    source_duration_sec: float
    source_fps: float
    total_source_frames: int
    processed_inference_frames: int
    expected_inference_samples: int
    actual_inference_sample_rate: float
    target_inference_fps: float
    total_gloved_detections: int
    total_bare_detections: int
    confirmed_violation_count: int
    confirmed_violations: List[Dict[str, Any]]
    avg_inference_latency_ms: float
    p95_inference_latency_ms: float
    total_processing_duration_sec: float
    processing_fps: float
    output_video_path: Optional[str]

