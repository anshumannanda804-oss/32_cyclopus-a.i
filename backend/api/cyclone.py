from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from backend.services.data_orchestrator import DataOrchestrator

router = APIRouter(prefix="/api/v1/cyclone", tags=["Cyclone Monitoring"])

orchestrator = DataOrchestrator()

@router.get("/active-storms")
async def get_active_storms():
    """
    Get active tropical cyclones in the North Indian Ocean basin (Bay of Bengal & Arabian Sea).
    """
    try:
        data = await orchestrator.ibtracs.get_active_storms()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data-fusion")
async def get_cyclone_data_fusion(
    lat: float = Query(..., description="Latitude (e.g. 14.5)"),
    lon: float = Query(..., description="Longitude (e.g. 86.2)"),
    storm_id: Optional[str] = Query(None, description="Optional Storm ID"),
    use_mosdac: bool = Query(False, description="Attempt MOSDAC primary before fallback")
):
    """
    Fetch comprehensive atmospheric, oceanographic, and satellite data using
    MOSDAC (if primary enabled) or automatic failover stack (INCOIS + EUMETSAT + IBTrACS + Open-Meteo).
    """
    try:
        result = await orchestrator.get_cyclone_data(
            lat=lat,
            lon=lon,
            storm_id=storm_id,
            use_mosdac_primary=use_mosdac
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ocean-state")
async def get_ocean_state(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Fetch INCOIS Sea Surface Temperature, Ocean Heat Content, and Wave Height.
    """
    return await orchestrator.incois.get_ocean_state(lat, lon)

@router.get("/atmospheric-nwp")
async def get_atmospheric_nwp(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Fetch Open-Meteo MSLP, 10m Wind fields, and Vertical Wind Shear.
    """
    return await orchestrator.openmeteo.get_cyclone_atmospheric_data(lat, lon)

@router.get("/satellite-imagery")
async def get_satellite_imagery(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Fetch EUMETSAT Meteosat IODC IR Satellite Cloud Top Temperature & Dvorak T-number.
    """
    return await orchestrator.eumetsat.get_satellite_imagery_metadata(lat, lon)
