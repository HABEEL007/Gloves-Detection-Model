from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HandDetectionSchema(BaseModel):
    box: List[float]
    conf: float
    color: str
    side: str
    track_id: Optional[int] = None

class FrameStatsSchema(BaseModel):
    status: str
    badge_type: str
    active_violations: int
    detected_gloves: List[HandDetectionSchema] = []
    gloves_count: int
    tracked_people: int
    latency_ms: float
    fps: float
    timestamp: str

class StreamControlSchema(BaseModel):
    action: str = Field(..., description="Action: pause, play, seek, restart")
    seconds: Optional[float] = 5.0
    percent: Optional[float] = None
    target_frame: Optional[int] = None

class SourceSwitchSchema(BaseModel):
    source: str = Field(..., description="RTSP URL, video file path, or webcam index")
    camera_id: Optional[str] = None
    loop: bool = True

class RuleConfigSchema(BaseModel):
    right_hand_only: bool = True

