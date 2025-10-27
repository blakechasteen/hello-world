"""
FastAPI bridge for VS Code extension.

Provides REST API endpoints for prompt management operations with caching.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import promptly modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

# Import Promptly core
try:
    # Try relative import first (when run as module)
    try:
        from .promptly import Promptly
    except ImportError:
        # Fall back to direct import (when run as script)
        from promptly import Promptly
    PROMPTLY_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Promptly core not available: {e}", file=sys.stderr)
    PROMPTLY_AVAILABLE = False


# Simple cache with TTL
class SimpleCache:
    """Simple in-memory cache with time-to-live."""

    def __init__(self, ttl_seconds: int = 30):
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())

    def invalidate(self, key: str = None):
        if key:
            self.cache.pop(key, None)
        else:
            self.cache.clear()


# Models
class PromptMetadata(BaseModel):
    name: str
    branch: str
    tags: List[str]
    created: str


class PromptData(BaseModel):
    content: str
    metadata: PromptMetadata


class PromptsListResponse(BaseModel):
    prompts: List[PromptMetadata]


# FastAPI app
app = FastAPI(
    title="Promptly VS Code Bridge",
    description="REST API bridge for Promptly VS Code extension",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize Promptly and cache
if PROMPTLY_AVAILABLE:
    try:
        p = Promptly()
        logger.info("Promptly initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Promptly: {e}")
        p = None
else:
    logger.warning("Promptly core not available")
    p = None

# Initialize cache (30 second TTL for prompt listings, 60s for individual prompts)
list_cache = SimpleCache(ttl_seconds=30)
prompt_cache = SimpleCache(ttl_seconds=60)
logger.info("Cache initialized (list: 30s TTL, prompts: 60s TTL)")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "promptly_available": PROMPTLY_AVAILABLE and p is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/prompts", response_model=PromptsListResponse)
async def list_prompts():
    """List all prompts with caching."""
    if not p:
        logger.error("Promptly not available for list_prompts")
        raise HTTPException(status_code=503, detail="Promptly not available")

    # Check cache first
    cached = list_cache.get("prompts_list")
    if cached:
        logger.debug(f"Cache hit for prompts_list ({len(cached)} prompts)")
        return PromptsListResponse(prompts=cached)

    try:
        # Get all prompts
        prompts = p.list_prompts()

        # Convert to metadata format
        metadata_list = []
        for prompt in prompts:
            try:
                # Get full prompt data to access metadata
                prompt_data = p.get(prompt['name'])
                if prompt_data:
                    metadata = prompt_data.get('metadata', {})
                    metadata_list.append(PromptMetadata(
                        name=prompt['name'],
                        branch=prompt_data.get('branch', 'main'),
                        tags=metadata.get('tags', []),
                        created=prompt_data.get('created_at', datetime.now().isoformat())
                    ))
            except Exception as e:
                print(f"WARNING: Failed to get metadata for {prompt['name']}: {e}", file=sys.stderr)
                continue

        # Cache the result
        list_cache.set("prompts_list", metadata_list)
        logger.info(f"Listed {len(metadata_list)} prompts (cached)")

        return PromptsListResponse(prompts=metadata_list)

    except Exception as e:
        logger.error(f"Failed to list prompts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {e}")


@app.get("/prompts/{prompt_name}", response_model=PromptData)
async def get_prompt(prompt_name: str):
    """Get a specific prompt by name with caching."""
    if not p:
        logger.error(f"Promptly not available for get_prompt: {prompt_name}")
        raise HTTPException(status_code=503, detail="Promptly not available")

    # Check cache first
    cache_key = f"prompt:{prompt_name}"
    cached = prompt_cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for prompt: {prompt_name}")
        return cached

    try:
        prompt_data = p.get(prompt_name)

        if not prompt_data:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")

        metadata = prompt_data.get('metadata', {})
        result = PromptData(
            content=prompt_data.get('content', ''),
            metadata=PromptMetadata(
                name=prompt_name,
                branch=prompt_data.get('branch', 'main'),
                tags=metadata.get('tags', []),
                created=prompt_data.get('created_at', datetime.now().isoformat())
            )
        )

        # Cache the result
        prompt_cache.set(cache_key, result)
        logger.info(f"Retrieved prompt: {prompt_name} (cached)")

        return result

    except KeyError:
        logger.warning(f"Prompt not found: {prompt_name}")
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
    except Exception as e:
        logger.error(f"Failed to get prompt {prompt_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get prompt: {e}")


def main():
    """Run the FastAPI server."""
    print("Starting Promptly VS Code Bridge on http://localhost:8765", file=sys.stderr)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="info"
    )


if __name__ == "__main__":
    main()