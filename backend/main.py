import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path so 'backend' imports work regardless of CWD
sys_path_root = str(Path(__file__).resolve().parent.parent)
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.cyclone import router as cyclone_router
from backend.api.prediction import router as prediction_router

app = FastAPI(
    title="Cyclone AI Backend Services & AI Prediction API",
    description="Multi-source resilient cyclone observation, tracking, detection models, and trajectory forecasting powered by INCOIS, EUMETSAT, IBTrACS, and Open-Meteo.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Registrations
app.include_router(cyclone_router)
app.include_router(prediction_router)

# Frontend Static File Mount
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def serve_frontend():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "system": "Cyclone AI Backend & ML Model API",
        "status": "operational"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
