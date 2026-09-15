import math
import httpx
from typing import Dict, Any, Optional

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

class OpenMeteoService:
    """
    Service for fetching atmospheric weather data from Open-Meteo API.
    Provides surface pressure, wind fields (10m, 850hPa, 200hPa), and vertical wind shear.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self.client = client or httpx.AsyncClient(timeout=15.0)

    async def get_cyclone_atmospheric_data(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 3
    ) -> Dict[str, Any]:
        """
        Fetch real-time & forecast atmospheric parameters relevant to cyclones.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "surface_pressure",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "wind_speed_850hPa",
                "wind_direction_850hPa",
                "wind_speed_200hPa",
                "wind_direction_200hPa",
                "relative_humidity_2m",
                "temperature_2m"
            ],
            "forecast_days": forecast_days,
            "timezone": "UTC"
        }

        try:
            response = await self.client.get(OPEN_METEO_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            processed = self._process_open_meteo_payload(data)
            return {
                "status": "success",
                "source": "Open-Meteo NWP",
                "coordinates": {"lat": lat, "lon": lon},
                "data": processed
            }
        except Exception as e:
            return {
                "status": "error",
                "source": "Open-Meteo NWP",
                "message": f"Failed to fetch Open-Meteo data: {str(e)}"
            }

    def _process_open_meteo_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            return {}

        current_idx = 0  # Latest current hour
        
        # Calculate u/v components for wind shear (850hPa vs 200hPa)
        ws850 = hourly.get("wind_speed_850hPa", [0])[current_idx] or 0
        wd850 = hourly.get("wind_direction_850hPa", [0])[current_idx] or 0
        ws200 = hourly.get("wind_speed_200hPa", [0])[current_idx] or 0
        wd200 = hourly.get("wind_direction_200hPa", [0])[current_idx] or 0

        u850 = -ws850 * math.sin(math.radians(wd850))
        v850 = -ws850 * math.cos(math.radians(wd850))
        u200 = -ws200 * math.sin(math.radians(wd200))
        v200 = -ws200 * math.cos(math.radians(wd200))

        wind_shear = math.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)

        return {
            "time_utc": times[current_idx],
            "mslp_hpa": hourly.get("pressure_msl", [None])[current_idx],
            "surface_pressure_hpa": hourly.get("surface_pressure", [None])[current_idx],
            "wind_speed_10m_knots": round((hourly.get("wind_speed_10m", [0])[current_idx] or 0) * 0.539957, 1),
            "wind_direction_10m_deg": hourly.get("wind_direction_10m", [None])[current_idx],
            "wind_gusts_10m_knots": round((hourly.get("wind_gusts_10m", [0])[current_idx] or 0) * 0.539957, 1),
            "vertical_wind_shear_knots": round(wind_shear * 0.539957, 1),
            "relative_humidity_pct": hourly.get("relative_humidity_2m", [None])[current_idx],
            "temperature_celsius": hourly.get("temperature_2m", [None])[current_idx],
            "hourly_timeline": {
                "times": times[:24],
                "mslp": hourly.get("pressure_msl", [])[:24],
                "wind_speed_10m": hourly.get("wind_speed_10m", [])[:24]
            }
        }
