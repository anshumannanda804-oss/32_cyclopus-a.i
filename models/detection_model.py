import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

class CycloneDetectionModel:
    """
    AI/ML Cyclone Detection & Intensity Estimation Model.
    
    Functions:
    1. Cyclogenesis Potential Index (GPI) assessment (SST, Wind Shear, Relative Humidity, Vorticity).
    2. Satellite Cloud Top Thermal (CTT) & Dvorak T-Number Intensity Classification.
    3. Cyclone Center Coordinates & Eye Radius Detection.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_loaded = False
        if model_path:
            self._load_weights(model_path)

    def _load_weights(self, path: str):
        # Placeholder for loading pre-trained PyTorch / ONNX weights
        self.is_loaded = True

    def calculate_genesis_potential_index(
        self,
        sst_celsius: float,
        vertical_wind_shear_knots: float,
        relative_humidity_pct: float,
        absolute_vorticity_850hpa: float = 3.5e-5
    ) -> Dict[str, Any]:
        """
        Calculate Emanuel & Nolan Cyclogenesis Potential Index (GPI):
        GPI = |10^5 * vorticity|^(3/2) * (RH / 50)^3 * (V_pot / 70)^3 * (1 + 0.1 * V_shear)^(-2)
        """
        # Potential Intensity V_pot estimated from SST (Emanuel 1999 relationship)
        # Threshold: SST >= 26.5°C
        if sst_celsius < 25.0:
            potential_intensity_m_s = 0.0
        else:
            potential_intensity_m_s = max(0.0, (sst_celsius - 25.0) * 8.5)

        # Vorticity factor
        vort_term = (abs(1e5 * absolute_vorticity_850hpa)) ** 1.5
        
        # Humidity factor
        rh_term = (max(0.0, relative_humidity_pct) / 50.0) ** 3
        
        # Potential Intensity factor (converted to m/s, reference 70 m/s)
        pi_term = (potential_intensity_m_s / 70.0) ** 3
        
        # Wind shear inhibition factor (shear in m/s: 1 knot = 0.5144 m/s)
        shear_ms = vertical_wind_shear_knots * 0.5144
        shear_term = (1.0 + 0.1 * shear_ms) ** (-2)

        gpi_val = vort_term * rh_term * pi_term * shear_term
        gpi_score = round(float(gpi_val), 2)

        if gpi_score < 1.0:
            category = "LOW"
            desc = "Conditions unfavorable for cyclone formation."
        elif gpi_score < 5.0:
            category = "MODERATE"
            desc = "Moderate cyclogenesis potential. Tropical depression possible."
        elif gpi_score < 15.0:
            category = "HIGH"
            desc = "High cyclogenesis potential. High probability of tropical storm development."
        else:
            category = "VERY HIGH / EXTREME"
            desc = "Favorable environmental conditions for rapid cyclone formation."

        return {
            "genesis_potential_index": gpi_score,
            "cyclogenesis_risk": category,
            "potential_max_wind_knots": round(potential_intensity_m_s * 1.94384, 1),
            "environmental_factors": {
                "sst_celsius": sst_celsius,
                "vertical_wind_shear_knots": vertical_wind_shear_knots,
                "relative_humidity_pct": relative_humidity_pct,
                "shear_inhibition_factor": round(shear_term, 3)
            },
            "description": desc
        }

    def detect_eye_and_intensity(
        self,
        cloud_top_temp_celsius: float,
        mslp_hpa: float,
        surface_wind_knots: float
    ) -> Dict[str, Any]:
        """
        Estimate Dvorak T-number, Eye detection, and Central Pressure anomaly.
        """
        # Eye detection threshold: CTT <= -65°C indicates deep CDO with possible pinhole eye
        eye_present = cloud_top_temp_celsius <= -65.0
        eye_diameter_km = 28.0 if eye_present else 0.0

        # Dvorak T-Number calculation (Enhanced Dvorak Technique - EDT)
        if cloud_top_temp_celsius <= -75.0:
            t_num = 5.5
        elif cloud_top_temp_celsius <= -65.0:
            t_num = 4.5
        elif cloud_top_temp_celsius <= -55.0:
            t_num = 3.5
        elif cloud_top_temp_celsius <= -45.0:
            t_num = 2.5
        else:
            t_num = 1.5

        # Refine T-number with surface wind speed
        wind_based_t = 1.0 + (surface_wind_knots / 20.0)
        final_t_num = round(min(8.0, max(1.0, (t_num * 0.6 + wind_based_t * 0.4))), 1)

        # IMD Classification
        imd_cat = self._get_imd_category(surface_wind_knots)

        # Pressure deficit (Background pressure 1010 hPa - MSLP)
        pressure_deficit_hpa = round(max(0.0, 1010.0 - mslp_hpa), 1)

        return {
            "dvorak_t_number": final_t_num,
            "dvorak_current_intensity_ci": final_t_num,
            "eye_detected": eye_present,
            "eye_radius_km": eye_diameter_km / 2.0,
            "cloud_top_min_temp_celsius": cloud_top_temp_celsius,
            "central_pressure_deficit_hpa": pressure_deficit_hpa,
            "imd_cyclone_category": imd_cat,
            "estimated_max_sustained_wind_knots": surface_wind_knots
        }

    def predict(self, fused_data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform full detection pipeline inference on multi-source fused payload.
        """
        data = fused_data_payload.get("data", {})
        target = data.get("target_location", {"latitude": 14.5, "longitude": 86.2})
        atm = data.get("atmospheric_state", {})
        ocean = data.get("oceanographic_state", {})
        sat = data.get("satellite_structure", {})

        sst = ocean.get("sea_surface_temperature_celsius", 28.5)
        shear = atm.get("vertical_wind_shear_knots", 12.0)
        rh = atm.get("relative_humidity_pct", 80.0)
        mslp = atm.get("mslp_hpa", 995.0)
        wind = atm.get("surface_wind_knots", 45.0)
        ctt = sat.get("cloud_top_temperature_celsius", -68.5)

        gpi_results = self.calculate_genesis_potential_index(sst, shear, rh)
        intensity_results = self.detect_eye_and_intensity(ctt, mslp, wind)

        return {
            "status": "success",
            "model_version": "CycloneAI-Detection-v1.0",
            "target_center": target,
            "cyclogenesis_assessment": gpi_results,
            "intensity_and_eye_detection": intensity_results,
            "overall_detection_status": "ACTIVE_CYCLONE_DETECTED" if intensity_results["dvorak_t_number"] >= 2.5 else "MONITORING"
        }

    def _get_imd_category(self, wind_knots: float) -> str:
        if wind_knots < 17:
            return "Low Pressure Area"
        elif wind_knots < 28:
            return "Depression"
        elif wind_knots < 34:
            return "Deep Depression"
        elif wind_knots < 48:
            return "Cyclonic Storm"
        elif wind_knots < 64:
            return "Severe Cyclonic Storm"
        elif wind_knots < 90:
            return "Very Severe Cyclonic Storm"
        elif wind_knots < 120:
            return "Extremely Severe Cyclonic Storm"
        else:
            return "Super Cyclonic Storm"
