import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.services.openmeteo_service import OpenMeteoService
from backend.services.ibtracs_service import IBTrACSService
from backend.services.incois_service import INCOISService
from backend.services.eumetsat_service import EUMETSATService
from backend.services.data_orchestrator import DataOrchestrator

async def run_tests():
    print("==================================================")
    print("   CYCLONE AI DATA STACK VERIFICATION TESTS       ")
    print("==================================================")

    lat, lon = 14.5, 86.2  # Bay of Bengal sample coordinates

    # 1. Test Open-Meteo
    print("\n[1/5] Testing Open-Meteo Atmospheric Service...")
    openmeteo = OpenMeteoService()
    om_res = await openmeteo.get_cyclone_atmospheric_data(lat, lon)
    assert om_res["status"] == "success", "Open-Meteo fetch failed"
    print(f"  [OK] Open-Meteo MSLP: {om_res['data'].get('mslp_hpa')} hPa")
    print(f"  [OK] 10m Wind Speed: {om_res['data'].get('wind_speed_10m_knots')} knots")
    print(f"  [OK] Vertical Wind Shear: {om_res['data'].get('vertical_wind_shear_knots')} knots")

    # 2. Test IBTrACS
    print("\n[2/5] Testing NOAA IBTrACS Best Track Service...")
    ibtracs = IBTrACSService()
    ib_res = await ibtracs.get_active_storms()
    assert ib_res["status"] == "success", "IBTrACS fetch failed"
    print(f"  [OK] Active Storms Count: {ib_res['count']}")
    if ib_res['storms']:
        s0 = ib_res['storms'][0]
        print(f"  [OK] Storm Name: {s0['name']} ({s0['category']})")
        print(f"  [OK] Max Winds: {s0['max_wind_knots']} knots, Min Pres: {s0['min_pressure_hpa']} hPa")

    # 3. Test INCOIS
    print("\n[3/5] Testing INCOIS Oceanographic Service...")
    incois = INCOISService()
    inc_res = await incois.get_ocean_state(lat, lon)
    assert inc_res["status"] == "success", "INCOIS fetch failed"
    print(f"  [OK] Sea Surface Temp (SST): {inc_res['sea_surface_temperature_celsius']} C")
    print(f"  [OK] Ocean Heat Content (OHC): {inc_res['ocean_heat_content_kj_cm2']} kJ/cm2")
    print(f"  [OK] Cyclogenesis Favorable: {inc_res['cyclogenesis_favorable']}")

    # 4. Test EUMETSAT
    print("\n[4/5] Testing EUMETSAT Meteosat Satellite Service...")
    eumetsat = EUMETSATService()
    eum_res = await eumetsat.get_satellite_imagery_metadata(lat, lon)
    assert eum_res["status"] == "success", "EUMETSAT fetch failed"
    print(f"  [OK] Cloud Top Temp: {eum_res['cloud_top_temperature_celsius']} C")
    print(f"  [OK] Estimated Dvorak T-number: T{eum_res['estimated_dvorak_t_number']}")
    print(f"  [OK] Cloud Pattern: {eum_res['cloud_pattern_type']}")

    # 5. Test Master Data Orchestrator
    print("\n[5/5] Testing Master Data Orchestrator (MOSDAC Failover Mode)...")
    orchestrator = DataOrchestrator()
    fused_res = await orchestrator.get_cyclone_data(lat, lon, use_mosdac_primary=False)
    assert fused_res["status"] == "success", "Data Orchestrator failed"
    print(f"  [OK] Execution Mode: {fused_res['execution_mode']}")
    print(f"  [OK] Activated Sources: {fused_res['data']['data_sources_activated']}")
    print(f"  [OK] Rapid Intensification Probability: {fused_res['data']['ai_risk_indicators']['rapid_intensification_probability_pct']}%")

    print("\n==================================================")
    print("   ALL 5 TESTS PASSED SUCCESSFULLY!              ")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_tests())
