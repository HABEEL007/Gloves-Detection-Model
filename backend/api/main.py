import os
import sys
import argparse
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.dependencies import get_context
from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = get_context()
    ctx.start()
    print("[+] Gloves Detection Backend Started Successfully")
    yield
    ctx.stop()
    print("[*] Gloves Detection Backend Stopped")

app = FastAPI(
    title="Gloves Compliance Detection API",
    description="Modular backend for real-time glove detection, tracking, and temporal compliance verification.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gloves Detection Backend Server")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on (default: 5001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)

