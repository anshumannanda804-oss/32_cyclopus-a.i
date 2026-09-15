import os

from fastapi import APIRouter, Query, HTTPException

from backend.services.data_orchestrator import DataOrchestrator
from models.detection_model import CycloneDetectionModel
from models.prediction_model import CyclonePredictionModel


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/v1/predict",
    tags=["AI Cyclone Prediction"]
)


# ============================================================
# SERVICES / MODELS
# ============================================================

orchestrator = DataOrchestrator()

detection_model = CycloneDetectionModel()


# ============================================================
# PYTORCH TRAJECTORY MODEL
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

TRAJECTORY_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cyclone_trajectory_lstm.pth"
)


prediction_model = CyclonePredictionModel(
    model_path=(
        TRAJECTORY_MODEL_PATH
        if os.path.exists(TRAJECTORY_MODEL_PATH)
        else None
    )
)


# ============================================================
# MODEL STATUS
# ============================================================

print("=" * 60)
print("CYCLONE AI PREDICTION MODEL")
print("=" * 60)

print(f"Project root:       {PROJECT_ROOT}")
print(f"Model path:         {TRAJECTORY_MODEL_PATH}")
print(
    f"Checkpoint exists:  "
    f"{os.path.exists(TRAJECTORY_MODEL_PATH)}"
)

print(
    f"Model loaded:       "
    f"{getattr(prediction_model, 'model_loaded', False)}"
)

print(
    f"Model mode:         "
    f"{getattr(prediction_model, 'model_mode', 'unknown')}"
)

print("=" * 60)


# ============================================================
# DETECTION
# ============================================================

@router.get("/detect")
async def run_detection(
    lat: float = Query(
        ...,
        description="Target Latitude"
    ),
    lon: float = Query(
        ...,
        description="Target Longitude"
    ),
    use_mosdac: bool = Query(
        False,
        description="Attempt MOSDAC primary before fallback"
    )
):
    """
    Run cyclone detection inference, Genesis Potential Index
    (GPI), and Dvorak T-number estimation on multi-source
    fused satellite and atmospheric data.
    """

    try:

        fused_data = await orchestrator.get_cyclone_data(
            lat,
            lon,
            use_mosdac_primary=use_mosdac
        )

        detection_results = detection_model.predict(
            fused_data
        )

        return {
            "success": True,
            "execution_mode": fused_data.get(
                "execution_mode"
            ),
            "detection": detection_results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# TRAJECTORY FORECAST
# ============================================================

@router.get("/trajectory")
async def run_trajectory_forecast(
    lat: float = Query(
        14.5,
        description="Current Cyclone Center Latitude"
    ),
    lon: float = Query(
        86.2,
        description="Current Cyclone Center Longitude"
    ),
    wind_knots: float = Query(
        55.0,
        description="Current Sustained Wind Speed (knots)"
    ),
    mslp_hpa: float = Query(
        988.0,
        description="Current MSLP (hPa)"
    )
):
    """
    Generate +6h to +72h cyclone trajectory,
    intensity forecast, confidence, and uncertainty cone.
    """

    try:

        forecast = prediction_model.predict_track_and_intensity(
            current_lat=lat,
            current_lon=lon,
            current_wind_knots=wind_knots,
            current_mslp_hpa=mslp_hpa
        )

        return {
            "success": True,
            "forecast": forecast
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# MODEL STATUS ENDPOINT
# ============================================================

@router.get("/model-status")
async def model_status():
    """
    Return the current PyTorch trajectory model status.
    """

    try:

        status_method = getattr(
            prediction_model,
            "checkpoint_status",
            None
        )

        if callable(status_method):
            status = status_method()

        else:
            status = {
                "model": "PyTorch LSTM",
                "model_loaded": getattr(
                    prediction_model,
                    "model_loaded",
                    False
                ),
                "model_mode": getattr(
                    prediction_model,
                    "model_mode",
                    "unknown"
                ),
                "model_path": TRAJECTORY_MODEL_PATH,
                "checkpoint_exists": os.path.exists(
                    TRAJECTORY_MODEL_PATH
                )
            }

        return {
            "success": True,
            "status": status
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )