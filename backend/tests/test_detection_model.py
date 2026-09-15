import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.detection_model import CycloneDetectionModel
from models.prediction_model import CyclonePredictionModel
from backend.services.data_orchestrator import DataOrchestrator

async def run_detection_tests():
    print("==================================================")
    print("   CYCLONE AI DETECTION & PREDICTION MODEL TESTS  ")
    print("==================================================")

    lat, lon = 14.5, 86.2

    # 1. Fetch Multi-Source Data
    print("\n[1/3] Fetching Fused Multi-Source Payload...")
    orchestrator = DataOrchestrator()
    fused_payload = await orchestrator.get_cyclone_data(lat, lon, use_mosdac_primary=False)
    assert fused_payload["status"] == "success"
    print("  [OK] Data Fusion Payload ready.")

    # 2. Test Detection Model Inference
    print("\n[2/3] Running Cyclone Detection Model Inference...")
    det_model = CycloneDetectionModel()
    results = det_model.predict(fused_payload)

    assert results["status"] == "success"
    gpi = results["cyclogenesis_assessment"]
    intensity = results["intensity_and_eye_detection"]

    print(f"  [OK] Overall Status: {results['overall_detection_status']}")
    print(f"  [OK] Genesis Potential Index (GPI): {gpi['genesis_potential_index']} ({gpi['cyclogenesis_risk']})")
    print(f"  [OK] Dvorak Intensity T-Number: T{intensity['dvorak_t_number']} (CI {intensity['dvorak_current_intensity_ci']})")
    print(f"  [OK] Eye Detected: {intensity['eye_detected']} (Radius: {intensity['eye_radius_km']} km)")
    print(f"  [OK] IMD Cyclone Category: {intensity['imd_cyclone_category']}")

    # 3. Test Prediction Model Trajectory Forecast
    print("\n[3/3] Running Cyclone Trajectory & Intensity Prediction Model...")
    pred_model = CyclonePredictionModel()
    forecast = pred_model.predict_track_and_intensity(
        current_lat=lat,
        current_lon=lon,
        current_wind_knots=55.0,
        current_mslp_hpa=988.0
    )

    assert forecast["status"] == "success"
    assert forecast["visualization"]["nasa_gibs"]["url"].startswith("https://gibs.earthdata.nasa.gov")
    print(f"  [OK] Forecast Track Points: {len(forecast['trajectory_points'])} timestamps (+6h to +72h)")
    p24 = forecast['trajectory_points'][3]
    print(f"  [OK] +24h Forecast: Lat {p24['latitude']}N, Lon {p24['longitude']}E | Wind: {p24['max_wind_knots']} kts ({p24['category']})")
    lf = forecast['landfall_forecast']
    print(f"  [OK] Landfall Region: {lf['estimated_landfall_region']} in ~{lf['estimated_time_to_landfall_hours']} hours")

    print("\n==================================================")
    print("   DETECTION & PREDICTION MODEL TESTS PASSED!    ")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_detection_tests())
