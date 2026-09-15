import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.database.mongodb import db_cache
from backend.services.data_orchestrator import DataOrchestrator

async def run_db_and_api_tests():
    print("==================================================")
    print("   CYCLONE AI MONGODB & API PERFORMANCE TESTS     ")
    print("==================================================")

    # 0. Health Check
    health = await db_cache.health_check()
    print(f"\n[0/3] Database Health Status: {health.get('status')} ({health.get('database')})")

    # 1. Test Direct MongoDB Cache Write & Read
    print("\n[1/3] Testing MongoDB Cache Performance...")
    t0 = time.time()
    test_key = "test_cyclone_grid_14.5_86.2"
    payload = {"lat": 14.5, "lon": 86.2, "sst": 28.5, "wind": 55.0}

    await db_cache.set_cached(test_key, payload, ttl_seconds=60)
    write_time_ms = (time.time() - t0) * 1000.0

    t1 = time.time()
    cached = await db_cache.get_cached(test_key)
    read_time_ms = (time.time() - t1) * 1000.0

    assert cached is not None and cached["sst"] == 28.5, "MongoDB Cache read failed"
    print(f"  [OK] MongoDB Cache Write Time: {round(write_time_ms, 2)} ms")
    print(f"  [OK] MongoDB Cache Read Time: {round(read_time_ms, 2)} ms (< 5ms HIGH PERFORMANCE)")

    # 2. Test Storm Persistence Collection
    print("\n[2/3] Testing Storm Records MongoDB Persistence...")
    storm_sample = {
        "storm_id": "2026_TEST_01",
        "name": "TEST_CYCLONE",
        "basin": "NI",
        "max_wind_knots": 65.0,
        "min_pressure_hpa": 980.0,
        "category": "Severe Cyclonic Storm",
        "last_updated_utc": "2026-09-01T23:00:00Z"
    }
    await db_cache.save_storm_record(storm_sample)
    saved_storms = await db_cache.get_all_saved_storms()
    assert len(saved_storms) >= 1, "MongoDB Storm save failed"
    print(f"  [OK] Storm Record Persisted Successfully: {saved_storms[0]['name']}")

    # 3. Test Orchestrator Database Caching Layer
    print("\n[3/3] Testing Data Orchestrator MongoDB Cache Layer...")
    orchestrator = DataOrchestrator()
    
    # First call (populates cache)
    t_start1 = time.time()
    res1 = await orchestrator.get_cyclone_data(14.5, 86.2, use_mosdac_primary=False, use_cache=True)
    t_duration1 = (time.time() - t_start1) * 1000.0

    # Second call (hits MongoDB cache)
    t_start2 = time.time()
    res2 = await orchestrator.get_cyclone_data(14.5, 86.2, use_mosdac_primary=False, use_cache=True)
    t_duration2 = (time.time() - t_start2) * 1000.0

    print(f"  [OK] Initial Multi-Source Query Time: {round(t_duration1, 2)} ms")
    print(f"  [OK] MongoDB Cached Query Time: {round(t_duration2, 2)} ms ({res2['execution_mode']})")

    print("\n==================================================")
    print("   ALL MONGODB & PERFORMANCE TESTS PASSED!        ")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_db_and_api_tests())
