"""Admin routes for monitoring and managing the backend."""
import asyncio
import subprocess

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from api.cache import SemanticCache
from api.dependencies import check_rate_limit, UserSession
from config.settings import settings

router = APIRouter()
_cache = SemanticCache()
_qdrant = AsyncQdrantClient(url=settings.qdrant_url)


class IngestResponse(BaseModel):
    message: str
    status: str


@router.get("/stats")
async def get_stats(user: UserSession = Depends(check_rate_limit)):
    if "admin" not in user.user_id:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    stats = {
        "qdrant": {
            "text_chunks": 0,
            "visual_pages": 0,
            "tables": 0
        },
        "redis": {
            "cache_size": 0
        },
        "system": {
            "status": "healthy"
        }
    }
    
    # Fetch Qdrant collection counts safely
    for col in [settings.text_collection, settings.visual_collection, settings.table_collection]:
        try:
            count_res = await _qdrant.count(collection_name=col)
            if col == settings.text_collection:
                stats["qdrant"]["text_chunks"] = count_res.count
            elif col == settings.visual_collection:
                stats["qdrant"]["visual_pages"] = count_res.count
            elif col == settings.table_collection:
                stats["qdrant"]["tables"] = count_res.count
        except Exception:
            pass  # Collection might not exist yet
            
    # Fetch Redis DB size
    try:
        stats["redis"]["cache_size"] = await _cache.redis.dbsize()
    except Exception:
        pass
        
    return stats


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(user: UserSession = Depends(check_rate_limit)):
    if "admin" not in user.user_id:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    try:
        # Run ingestion pipeline as a non-blocking subprocess
        process = await asyncio.create_subprocess_exec(
            "python", "-m", "ingestion.pipeline", "--source", "./data",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        return IngestResponse(
            message="Ingestion pipeline started in the background.",
            status="running"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
