import time
from typing import Optional
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np

from .dependencies import get_context, BackendContext
from .schemas import StreamControlSchema, SourceSwitchSchema, RuleConfigSchema

router = APIRouter()

@router.get("/health")
def health(ctx: BackendContext = Depends(get_context)):
    return {
        "status": "healthy",
        "is_connected": ctx.processor.is_connected,
        "source": ctx.processor.camera_id,
        "source_type": ctx.processor.source_type
    }

@router.get("/api/video_feed")
def video_feed(ctx: BackendContext = Depends(get_context)):
    def generate():
        last_frame = None

        # Immediate first frame delivery to prevent browser black screen
        initial_frame = ctx.get_latest_jpeg()
        if initial_frame is None:
            raw, _, _ = ctx.processor.get_latest_frame()
            if raw is not None and raw.size > 0:
                try:
                    disp = cv2.resize(raw, (1280, 720)) if (raw.shape[1] != 1280 or raw.shape[0] != 720) else raw.copy()
                    cv2.putText(disp, "[Syncing AI Feed...]", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                    _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    initial_frame = buf.tobytes()
                except Exception:
                    pass
            if initial_frame is None:
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                canvas[:] = (20, 15, 10)
                cv2.putText(canvas, "AI Live CCTV System | Connecting Stream...", (320, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 220, 255), 2)
                _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                initial_frame = buf.tobytes()

        if initial_frame is not None:
            last_frame = initial_frame
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(initial_frame)).encode() + b"\r\n\r\n" +
                   initial_frame + b"\r\n")

        while True:
            with ctx.new_jpeg_condition:
                ctx.new_jpeg_condition.wait(timeout=0.08)
                frame = ctx.get_latest_jpeg()

            if frame is not None and frame != last_frame:
                last_frame = frame
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                       frame + b"\r\n")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@router.get("/api/live_stats")
def live_stats(ctx: BackendContext = Depends(get_context)):
    stats = dict(ctx.latest_stats)
    stats["source"] = ctx.processor.camera_id
    stats["source_type"] = ctx.processor.source_type
    stats["is_paused"] = ctx.processor.is_paused
    stats["playback"] = ctx.processor.get_playback_status() if ctx.processor.source_type == "video" else None
    return JSONResponse(content=stats)

@router.post("/api/stream/set_source")
def set_source(body: SourceSwitchSchema, ctx: BackendContext = Depends(get_context)):
    ctx.processor.switch_source(body.source, camera_id=body.camera_id, loop_video=body.loop)
    return {"success": True, "source": ctx.processor.camera_id, "type": ctx.processor.source_type}

@router.post("/api/stream/control")
def stream_control(body: StreamControlSchema, ctx: BackendContext = Depends(get_context)):
    if body.action == "pause":
        is_paused = ctx.processor.toggle_pause()
        return {"success": True, "action": "pause", "is_paused": is_paused}
    elif body.action == "restart":
        ctx.processor.seek_frame(0)
        return {"success": True, "action": "restart"}
    elif body.action == "seek" and body.target_frame is not None:
        ctx.processor.seek_frame(body.target_frame)
        return {"success": True, "action": "seek", "target_frame": body.target_frame}
    return {"success": False, "error": f"Unknown action: {body.action}"}

@router.post("/api/stream/set_rule")
def set_rule(body: RuleConfigSchema, ctx: BackendContext = Depends(get_context)):
    ctx.detector.set_rule_mode(body.right_hand_only)
    return {"success": True, "right_hand_only": ctx.detector.right_hand_only_rule}

