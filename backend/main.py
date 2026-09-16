import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.cyclone import router as cyclone_router
from backend.api.prediction import router as prediction_router


app = FastAPI(
    title="Cyclone AI Backend Services & AI Prediction API",
    description=(
        "Multi-source resilient cyclone observation, tracking, "
        "detection models, and trajectory forecasting powered by "
        "INCOIS, EUMETSAT, IBTrACS, and Open-Meteo."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(cyclone_router)
app.include_router(prediction_router)


# ============================================================
# FRONTEND
# ============================================================

FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static"
    )


@app.get("/")
async def serve_frontend():

    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(str(index_file))

    return {
        "system": "Cyclone AI Backend & ML Model API",
        "status": "operational"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "service": "cyclone-ai-api"
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000
    )
