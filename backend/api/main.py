"""
FastAPI Application Entrypoint for Gloves Detection Module.
Exposes RESTful endpoints for health, model metadata, image inference,
and asynchronous offline 3 FPS video processing.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..model.config import config
from ..model.model_loader import get_model_manager
from .routes import router

app = FastAPI(
    title=config.API_TITLE,
    description="Production-Ready AI Detection, Tracking, Alerting and 3 FPS Offline Video Processing for Glove Compliance",
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Evidence Snapshots Directory
evidence_dir = Path(config.EVIDENCE_DIR).resolve()
evidence_dir.mkdir(parents=True, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(evidence_dir)), name="evidence")

# Include Detection API Router
app.include_router(router)


@app.on_event("startup")
def startup_event():
    """Warms up the model loader at application startup."""
    try:
        mgr = get_model_manager()
        print(f"[API Startup] Loaded {mgr.architecture} from {mgr.current_weights_path} on {mgr.device}")
    except Exception as e:
        print(f"[API Startup Warning] Could not warm up model manager: {e}")


@app.get("/")
def root():
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "online",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=False)

