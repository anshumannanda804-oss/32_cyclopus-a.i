import pickle
import os
import json
from detection_model import CycloneDetectionModel

def create_and_test_model():
    print("1. Instantiating CycloneDetectionModel...")
    model = CycloneDetectionModel()
    
    # Save the model to a .pkl file
    model_path = "cyclone_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"2. Model successfully saved to {model_path}")
    
    # Load the model from .pkl
    print("3. Loading model from .pkl to verify...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print("   Model loaded successfully.")
    
    # Test over Bay of Bengal
    # Coordinates: Latitude ~15°N, Longitude ~90°E
    bob_payload = {
        "data": {
            "target_location": {"latitude": 15.0, "longitude": 90.0},
            "atmospheric_state": {
                "vertical_wind_shear_knots": 10.0,
                "relative_humidity_pct": 85.0,
                "mslp_hpa": 980.0,
                "surface_wind_knots": 60.0
            },
            "oceanographic_state": {
                "sea_surface_temperature_celsius": 30.5
            },
            "satellite_structure": {
                "cloud_top_temperature_celsius": -75.0
            }
        }
    }
    
    print("\n=======================================================")
    print("TESTING OVER BAY OF BENGAL (BoB)")
    print("Location: Latitude 15.0°N, Longitude 90.0°E")
    print("Conditions: High SST (30.5°C), Low Shear, High Humidity")
    print("=======================================================")
    bob_results = loaded_model.predict(bob_payload)
    print(json.dumps(bob_results, indent=2))
    
    # Test over Arabian Sea
    # Coordinates: Latitude ~15°N, Longitude ~65°E
    as_payload = {
        "data": {
            "target_location": {"latitude": 15.0, "longitude": 65.0},
            "atmospheric_state": {
                "vertical_wind_shear_knots": 25.0,
                "relative_humidity_pct": 55.0,
                "mslp_hpa": 1004.0,
                "surface_wind_knots": 25.0
            },
            "oceanographic_state": {
                "sea_surface_temperature_celsius": 27.5
            },
            "satellite_structure": {
                "cloud_top_temperature_celsius": -50.0
            }
        }
    }
    
    print("\n=======================================================")
    print("TESTING OVER ARABIAN SEA (AS)")
    print("Location: Latitude 15.0°N, Longitude 65.0°E")
    print("Conditions: Moderate SST (27.5°C), High Shear, Lower Humidity")
    print("=======================================================")
    as_results = loaded_model.predict(as_payload)
    print(json.dumps(as_results, indent=2))

if __name__ == "__main__":
    create_and_test_model()
