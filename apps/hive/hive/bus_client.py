"""HiveBusClient — API-first, LiteMemoryBus fallback.

Tries the HoloLoom API (port 8000) for bus operations.
Falls back to a local LiteMemoryBus with JSON snapshot persistence.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hololoom.core.memory.bus import MemoryItem, MemoryQuery, MemoryResult
from hololoom.core.memory.lite_bus import LiteMemoryBus, StoredItem

try:
    import httpx
    _HAVE_HTTPX = True
except ImportError:
    _HAVE_HTTPX = False

logger = logging.getLogger(__name__)


class HiveBusClient:
    """Bus access with HoloLoom API -> LiteMemoryBus fallback.

    Usage:
        async with HiveBusClient() as client:
            node_id = await client.store(item)
            result = await client.query(query)
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        snapshot_path: Optional[Path] = None,
    ):
        self.api_url = api_url or os.environ.get(
            "HOLOLOOM_API_URL", "http://127.0.0.1:8000"
        )
        self.snapshot_path = snapshot_path or Path(
            os.environ.get("HIVE_BUS_SNAPSHOT", str(Path.home() / ".hive" / "bus_snapshot.json"))
        )
        self._bus: Optional[LiteMemoryBus] = None
        self._http: Optional[Any] = None
        self.mode: Optional[str] = None  # "api" or "local"

    async def connect(self) -> str:
        """Connect to bus. Returns "api" or "local"."""
        if _HAVE_HTTPX:
            try:
                self._http = httpx.AsyncClient(base_url=self.api_url, timeout=5.0)
                resp = await self._http.get("/health")
                if resp.status_code == 200:
                    self.mode = "api"
                    logger.info("Connected to HoloLoom API at %s", self.api_url)
                    return self.mode
            except Exception as e:
                logger.debug("API unavailable (%s), falling back to local", e)
                if self._http:
                    await self._http.aclose()
                    self._http = None

        # Fallback: local LiteMemoryBus
        self._bus = LiteMemoryBus()
        await self._bus.initialize()
        if self.snapshot_path.exists():
            try:
                snapshot = json.loads(self.snapshot_path.read_text())
                self._bus.from_snapshot_dict(snapshot)
                logger.info("Loaded bus snapshot: %d items from %s",
                            len(self._bus._items), self.snapshot_path)
            except Exception as e:
                logger.warning("Failed to load snapshot: %s", e)
        self.mode = "local"
        logger.info("Using local LiteMemoryBus (snapshot: %s)", self.snapshot_path)
        return self.mode

    async def store(self, item: MemoryItem) -> str:
        """Store a memory item. Returns item ID."""
        if self.mode == "api" and self._http:
            resp = await self._http.post("/memory/store", json={
                "content": item.content,
                "memory_type": item.memory_type,
                "entity_ids": item.entity_ids or [],
                "properties": item.properties or {},
                "importance": item.importance,
            })
            resp.raise_for_status()
            return resp.json().get("node_id", "unknown")
        return await self._bus.store(item)

    async def query(self, q: MemoryQuery) -> MemoryResult:
        """Query the bus."""
        if self.mode == "api" and self._http:
            payload = {
                "intent": q.intent,
                "max_results": q.max_results,
                "min_importance": q.min_importance,
            }
            if q.memory_type:
                payload["memory_type"] = q.memory_type
            if q.properties_match:
                payload["properties_match"] = q.properties_match
            if q.entity_ids:
                payload["entity_ids"] = q.entity_ids
            resp = await self._http.post("/memory/query", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return MemoryResult(
                items=data.get("items", []),
                query=q,
                resolution_path=data.get("resolution_path", "api"),
                token_estimate=data.get("token_estimate", 0),
            )
        return await self._bus.query(q)

    async def store_edge(self, source_id: str, target_id: str, rel_type: str = "RELATED_TO"):
        """Store an edge between items."""
        if self.mode == "api" and self._http:
            await self._http.post("/memory/edge", json={
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
            })
            return
        await self._bus.store_edge(source_id, target_id, rel_type)

    def get_item(self, item_id: str) -> Optional[StoredItem]:
        """Get a stored item by ID (local mode only)."""
        if self._bus:
            return self._bus.get_item(item_id)
        return None

    async def close(self):
        """Cleanup. If local mode, save snapshot."""
        if self._http:
            await self._http.aclose()
            self._http = None
        if self._bus and self.mode == "local":
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self._bus.save_snapshot(str(self.snapshot_path))
            logger.info("Saved bus snapshot: %d items", len(self._bus._items))
            await self._bus.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
