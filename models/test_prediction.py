import os
from prediction_model import CyclonePredictionModel

model_path = os.path.join(
    os.path.dirname(__file__), "cyclone_trajectory_lstm.pth"
)

model = CyclonePredictionModel(
    model_path=model_path if os.path.exists(model_path) else None
)

result = model.predict_track_and_intensity(
    current_lat=15.2,
    current_lon=86.4,
    current_wind_knots=60,
    current_mslp_hpa=978,
    steering_wind_speed_knots=12,
    steering_wind_dir_deg=315,
)

print("MODEL MODE:", result["model_mode"])
print("MODEL LOADED:", result["model_loaded"])
print("WARNING:", result["warning"])
print("\nFORECAST:")
for point in result["trajectory_points"]:
    print(point)
