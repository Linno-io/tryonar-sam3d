import os
import shutil
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.engine import InferenceEngine

# Configuration
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Global Engine Instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager to initialize the model on startup.
    """
    global engine
    logger.info("Starting up API...")
    try:
        engine = InferenceEngine()
        # Pre-load model
        engine.load_model()
        logger.info("Engine initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        # We don't exit here so the app can still start and report unhealthy
    
    yield
    
    logger.info("Shutting down API...")
    # Cleanup logic if needed

app = FastAPI(lifespan=lifespan)

# Mount outputs for download
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

def cleanup_old_files():
    """Remove files older than 1 hour to prevent disk exhaustion."""
    now = time.time()
    retention_period = 3600  # 1 hour

    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in directory.iterdir():
            if f.is_file():
                if f.stat().st_mtime < now - retention_period:
                    try:
                        f.unlink()
                        logger.info(f"Deleted old file: {f}")
                    except Exception as e:
                        logger.error(f"Error deleting {f}: {e}")

@app.get("/health")
async def health_check():
    """
    Return 200 OK only if model is loaded and ready.
    """
    if engine is None or engine.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Optional: Check VRAM or other metrics if possible
    return {"status": "ready", "model": "loaded"}

@app.post("/api/inference")
async def inference(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Accept image, run inference, return link to GLB.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save Upload
    temp_input_path = UPLOAD_DIR / f"{int(time.time())}_{file.filename}"
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        # Run Inference
        start_time = time.time()
        
        # Determine strict job ID based on filename or random to avoid collision if desired
        # engine.process uses basename for job_id currently
        glb_path = engine.process(str(temp_input_path))
        
        duration = time.time() - start_time
        logger.info(f"Inference complete in {duration:.2f}s")
        
        # Construct Download Link
        # Assuming the service is behind a proxy or accessible via relative path
        filename = os.path.basename(glb_path)
        download_url = f"/outputs/{filename}"
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_old_files)
        
        return {
            "status": "success",
            "download_url": download_url,
            "inference_time": duration
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        # Cleanup input if failed
        if temp_input_path.exists():
            temp_input_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
