"""
FastAPI Routes for Glove Compliance Detection System.
Includes image detection (raw binary, base64, or file path),
asynchronous offline video processing jobs, model info, and health diagnostics.
"""
import os
import uuid
import time
import base64
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from fastapi.responses import FileResponse

from ..model.config import config
from ..model.model_loader import get_model_manager, GloveModelManager
from ..model.inference import HandDetector, DetectedHand, InferenceResult
from ..detection.video_detector import OfflineVideoDetector
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    ImageDetectResponse,
    HandDetectionSchema,
    TimingBreakdownSchema,
    VideoJobCreateRequest,
    VideoJobStatusResponse,
    VideoJobResultResponse
)
from .dependencies import verify_api_key, get_detector

logger = logging.getLogger("backend.api.routes")
router = APIRouter(prefix="/api/v1", tags=["Detection & Compliance"])

# In-memory Job Store for Asynchronous Video Tasks
_jobs_lock = threading.Lock()
_jobs_store: Dict[str, Dict[str, Any]] = {}


def _execute_video_job(
    job_id: str,
    video_path: str,
    output_video_path: Optional[str],
    sampled_frames_only: bool,
    rule_mode: str,
    conf_threshold: float,
    camera_id: str
):
    """Worker function executed in background thread."""
    with _jobs_lock:
        if job_id in _jobs_store:
            _jobs_store[job_id]["status"] = "PROCESSING"

    try:
        detector = OfflineVideoDetector(
            target_fps=config.OFFLINE_TARGET_FPS,
            rule_mode=rule_mode,
            conf_threshold=conf_threshold,
            camera_id=camera_id
        )

        report = detector.process_video(
            video_path=video_path,
            output_video_path=output_video_path,
            sampled_frames_only=sampled_frames_only,
            save_snapshots=config.SNAPSHOT_ENABLED
        )

        with _jobs_lock:
            if job_id in _jobs_store:
                _jobs_store[job_id]["status"] = "COMPLETED"
                _jobs_store[job_id]["completed_at"] = datetime.now().isoformat()
                _jobs_store[job_id]["progress"] = 1.0
                _jobs_store[job_id]["report"] = report
                _jobs_store[job_id]["output_video_path"] = report.get("output_video_path")

    except Exception as e:
        logger.error(f"Video job {job_id} failed: {e}", exc_info=True)
        with _jobs_lock:
            if job_id in _jobs_store:
                _jobs_store[job_id]["status"] = "FAILED"
                _jobs_store[job_id]["completed_at"] = datetime.now().isoformat()
                _jobs_store[job_id]["error"] = str(e)


# ==========================================
# 1. System & Model Endpoints
# ==========================================

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Returns health status of the detection service."""
    mgr = get_model_manager()
    return HealthResponse(
        status="healthy",
        service="Glove Compliance Detection System",
        version=config.API_VERSION,
        model_loaded=(mgr.model is not None),
        device=mgr.device,
        timestamp=datetime.now().isoformat()
    )


@router.get("/model/info", response_model=ModelInfoResponse, dependencies=[Depends(verify_api_key)])
def get_model_info():
    """Returns metadata about the active detection model."""
    mgr = get_model_manager()
    info = mgr.get_model_info()
    return ModelInfoResponse(**info)


# ==========================================
# 2. Image Inference Endpoint
# ==========================================

@router.post("/detect/image", response_model=ImageDetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_image(
    request: Request,
    conf_threshold: Optional[float] = Query(None, ge=0.05, le=0.95, description="Confidence threshold"),
    imgsz: Optional[int] = Query(None, ge=320, le=1280, description="Image size"),
    detector: HandDetector = Depends(get_detector)
):
    """
    Executes hand detection on an input image.
    Supports:
    - Raw binary image upload (Content-Type: image/jpeg, image/png, application/octet-stream)
    - JSON payload with 'image_path': '/path/to/image.jpg'
    - JSON payload with 'image_base64': '...'
    """
    frame = None
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")

        if "image_path" in body:
            img_p = Path(body["image_path"]).resolve()
            if not img_p.exists():
                raise HTTPException(status_code=404, detail=f"Image file not found at: {img_p}")
            frame = cv2.imread(str(img_p))
        elif "image_base64" in body:
            try:
                b64_str = body["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")
        else:
            raise HTTPException(status_code=400, detail="JSON must contain 'image_path' or 'image_base64'.")

    else:
        # Raw binary body
        body_bytes = await request.body()
        if not body_bytes:
            raise HTTPException(status_code=400, detail="Empty request body. Send image bytes or JSON.")
        nparr = np.frombuffer(body_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. Ensure valid JPEG, PNG or BMP content."
        )

    res: InferenceResult = detector.detect(
        frame=frame,
        conf_threshold=conf_threshold,
        imgsz=imgsz
    )

    box_schemas = [
        HandDetectionSchema(
            box=h.box,
            class_id=h.class_id,
            class_name=h.class_name,
            confidence=h.confidence,
            glove_color=h.glove_color,
            center=[h.cx, h.cy],
            dimensions=[h.width, h.height]
        )
        for h in res.hands
    ]

    timing_schema = TimingBreakdownSchema(**res.timing)

    return ImageDetectResponse(
        total_hands=res.total_hands,
        gloved_count=len(res.gloved_hands),
        bare_count=len(res.bare_hands),
        detections=box_schemas,
        timing_ms=timing_schema,
        frame_shape=list(res.frame_shape),
        conf_threshold=res.conf_threshold,
        imgsz=res.imgsz
    )


# ==========================================
# 3. Asynchronous Offline Video Processing
# ==========================================

@router.post("/detect/video", response_model=VideoJobStatusResponse, dependencies=[Depends(verify_api_key)])
def detect_video(
    request: VideoJobCreateRequest
):
    """
    Creates an asynchronous offline video processing job.
    Processes exactly 3 AI inference frames per second of source-video time.
    Returns a unique job ID to track progress and retrieve results.
    """
    v_path = Path(request.video_path).resolve()
    if not v_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found at path: {request.video_path}"
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    created_at = datetime.now().isoformat()

    job_entry = {
        "job_id": job_id,
        "status": "PENDING",
        "created_at": created_at,
        "completed_at": None,
        "progress": 0.0,
        "error": None,
        "output_video_path": request.output_video_path,
        "report": None
    }

    with _jobs_lock:
        _jobs_store[job_id] = job_entry

    # Launch background thread
    worker_thread = threading.Thread(
        target=_execute_video_job,
        kwargs={
            "job_id": job_id,
            "video_path": str(v_path),
            "output_video_path": request.output_video_path,
            "sampled_frames_only": request.sampled_frames_only,
            "rule_mode": request.rule_mode,
            "conf_threshold": request.conf_threshold or config.HAND_CONF_THRESHOLD,
            "camera_id": request.camera_id or "OFFLINE_CAM"
        },
        daemon=True
    )
    worker_thread.start()

    logger.info(f"Enqueued offline video job: {job_id} for {v_path.name}")
    return VideoJobStatusResponse(
        job_id=job_id,
        status="PENDING",
        created_at=created_at,
        progress=0.0,
        output_video_path=request.output_video_path
    )


@router.get("/jobs/{job_id}", response_model=VideoJobStatusResponse, dependencies=[Depends(verify_api_key)])
def get_job_status(job_id: str):
    """Queries the current status of an offline video processing job."""
    with _jobs_lock:
        job = _jobs_store.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found."
        )

    return VideoJobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        progress=job.get("progress", 0.0),
        error=job.get("error"),
        output_video_path=job.get("output_video_path")
    )


@router.get("/jobs/{job_id}/result", response_model=VideoJobResultResponse, dependencies=[Depends(verify_api_key)])
def get_job_result(job_id: str):
    """Retrieves the full detection and compliance report for a completed video job."""
    with _jobs_lock:
        job = _jobs_store.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found."
        )

    if job["status"] == "FAILED":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job failed with error: {job.get('error')}"
        )

    if job["status"] != "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Job is still '{job['status']}'. Check back shortly."
        )

    report = job.get("report")
    if not report:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job marked COMPLETED but report data is missing."
        )

    return VideoJobResultResponse(
        job_id=job_id,
        status=job["status"],
        **report
    )


@router.get("/jobs/{job_id}/download", dependencies=[Depends(verify_api_key)])
def download_job_output(job_id: str):
    """Downloads the annotated output video file for a completed video job."""
    with _jobs_lock:
        job = _jobs_store.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found."
        )

    if job["status"] != "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed yet (current status: {job['status']})."
        )

    out_path = job.get("output_video_path")
    if not out_path or not Path(out_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Output video file does not exist at {out_path}."
        )

    return FileResponse(
        path=out_path,
        media_type="video/mp4",
        filename=Path(out_path).name
    )
