import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("cyclone_ai.mongodb")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "cyclone_ai")

class MongoDatabaseManager:
    """
    High-Performance MongoDB Database & Caching Manager for Cyclone AI.
    Replaces SQLite with MongoDB using Motor (AsyncIO driver).
    Caches live data from INCOIS, EUMETSAT, IBTrACS, Open-Meteo, and AI predictions.
    Includes in-memory fallback mechanism if MongoDB daemon is offline.
    """

    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.use_fallback = False
        
        # Fallback storage if MongoDB server is unreachable
        self._in_memory_cache: Dict[str, Dict[str, Any]] = {}
        self._in_memory_storms: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = False

    def _init_client(self):
        """
        Lazy initialization of Motor client.
        """
        if self._initialized:
            return

        try:
            import motor.motor_asyncio
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=2000
            )
            self.db = self.client[self.db_name]
            self._initialized = True
        except Exception as e:
            logger.warning(f"MongoDB Client initialization error: {e}. Activating in-memory fallback.")
            self.use_fallback = True
            self._initialized = True

    async def _ensure_indexes(self):
        if self.use_fallback or not self.db:
            return
        try:
            # Create TTL index on api_cache collection if possible
            await self.db.api_cache.create_index("expires_at", expireAfterSeconds=0)
            await self.db.storm_records.create_index("max_wind_knots")
        except Exception as e:
            logger.warning(f"Failed to create MongoDB indexes: {e}")

    async def get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Async retrieval of cached payload from MongoDB api_cache collection.
        Returns None if expired or missing.
        """
        self._init_client()
        now = time.time()

        if not self.use_fallback and self.db is not None:
            try:
                doc = await self.db.api_cache.find_one({
                    "_id": cache_key,
                    "expires_at": {"$gt": now}
                })
                if doc:
                    return doc.get("payload")
            except Exception as e:
                logger.warning(f"MongoDB read error: {e}. Falling back to in-memory store.")
                self.use_fallback = True

        # Fallback in-memory read
        item = self._in_memory_cache.get(cache_key)
        if item:
            if item.get("expires_at", 0) > now:
                return item.get("payload")
            else:
                del self._in_memory_cache[cache_key]
        return None

    async def set_cached(self, cache_key: str, payload: Dict[str, Any], ttl_seconds: int = 900):
        """
        Async insertion of payload into MongoDB api_cache collection. Default TTL: 15 mins (900s).
        """
        self._init_client()
        now = time.time()
        expires_at = now + ttl_seconds

        doc = {
            "_id": cache_key,
            "payload": payload,
            "created_at": now,
            "expires_at": expires_at
        }

        if not self.use_fallback and self.db is not None:
            try:
                await self.db.api_cache.replace_one(
                    {"_id": cache_key},
                    doc,
                    upsert=True
                )
                return
            except Exception as e:
                logger.warning(f"MongoDB write error: {e}. Falling back to in-memory store.")
                self.use_fallback = True

        # Fallback in-memory set
        self._in_memory_cache[cache_key] = {
            "payload": payload,
            "expires_at": expires_at,
            "created_at": now
        }

    async def save_storm_record(self, storm_data: Dict[str, Any]):
        """
        Persist active tropical cyclone metadata to MongoDB storm_records collection.
        """
        self._init_client()
        sid = storm_data.get("storm_id", "UNKNOWN")
        name = storm_data.get("name", "UNNAMED")
        basin = storm_data.get("basin", "NI")
        wind = float(storm_data.get("max_wind_knots", 0.0))
        pres = float(storm_data.get("min_pressure_hpa", 1010.0))
        cat = storm_data.get("category", "Depression")
        updated = storm_data.get("last_updated_utc", str(time.time()))

        doc = {
            "_id": sid,
            "storm_id": sid,
            "name": name,
            "basin": basin,
            "max_wind_knots": wind,
            "min_pressure_hpa": pres,
            "category": cat,
            "last_updated_utc": updated,
            "details": storm_data
        }

        if not self.use_fallback and self.db is not None:
            try:
                await self.db.storm_records.replace_one(
                    {"_id": sid},
                    doc,
                    upsert=True
                )
                return
            except Exception as e:
                logger.warning(f"MongoDB save_storm error: {e}. Falling back to in-memory store.")
                self.use_fallback = True

        # Fallback in-memory store
        self._in_memory_storms[sid] = storm_data

    async def get_all_saved_storms(self) -> List[Dict[str, Any]]:
        """
        Retrieve all persisted storms from MongoDB storm_records collection.
        """
        self._init_client()

        if not self.use_fallback and self.db is not None:
            try:
                cursor = self.db.storm_records.find().sort("max_wind_knots", -1)
                results = []
                async for doc in cursor:
                    results.append(doc.get("details", doc))
                return results
            except Exception as e:
                logger.warning(f"MongoDB query storms error: {e}. Falling back to in-memory store.")
                self.use_fallback = True

        # Fallback in-memory get all
        storms = list(self._in_memory_storms.values())
        storms.sort(key=lambda x: x.get("max_wind_knots", 0.0), reverse=True)
        return storms

    async def health_check(self) -> Dict[str, Any]:
        """
        Check database connection status.
        """
        self._init_client()
        if not self.use_fallback and self.db is not None:
            try:
                await self.db.command("ping")
                return {
                    "database": "MongoDB",
                    "status": "connected",
                    "db_name": self.db_name,
                    "uri": self.uri
                }
            except Exception as e:
                return {
                    "database": "MongoDB",
                    "status": "unreachable",
                    "error": str(e),
                    "fallback_active": True
                }
        return {
            "database": "MongoDB",
            "status": "in_memory_fallback",
            "fallback_active": True
        }

# Global MongoDB Database Manager instance
db_cache = MongoDatabaseManager()
