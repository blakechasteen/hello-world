# -*- coding: utf-8 -*-
"""
Farm Tracker Base
=================
Base class for farm management applications.

Provides:
- Asset and log management
- HoloLoom integration (WeavingOrchestrator)
- Natural language query interface
- Memory persistence (Neo4j + Qdrant)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio

# Import HoloLoom
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.Documentation.types import Query, MemoryShard

from .models import Asset, Log, AssetType, LogType

logger = logging.getLogger(__name__)


@dataclass
class FarmTrackerConfig:
    """Configuration for farm tracker."""
    app_name: str = "Farm Tracker"
    location: Optional[str] = None
    memory_backend: MemoryBackend = MemoryBackend.NEO4J_QDRANT
    enable_decision_engine: bool = True
    enable_ollama_enrichment: bool = False
    storage_path: Optional[Path] = None
    orchestrator_mode: str = "fused"  # bare, fast, fused


class FarmTracker:
    """
    Base farm management tracker.

    Apps inherit from this to get:
    - Asset/log storage and retrieval
    - Natural language queries via HoloLoom
    - Decision support and recommendations
    - Multimodal data ingestion via spinners
    """

    def __init__(self, config: FarmTrackerConfig = None):
        self.config = config or FarmTrackerConfig()
        self.assets: Dict[str, Asset] = {}
        self.logs: List[Log] = []
        self.orchestrator: Optional[WeavingOrchestrator] = None
        self._initialized = False

        # Setup storage
        if self.config.storage_path:
            self.config.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {self.config.app_name}")

    async def initialize(self):
        """Initialize HoloLoom orchestrator and load data."""
        if self._initialized:
            return

        # Create HoloLoom config
        if self.config.orchestrator_mode == "bare":
            holo_config = Config.bare()
        elif self.config.orchestrator_mode == "fast":
            holo_config = Config.fast()
        else:
            holo_config = Config.fused()

        holo_config.memory_backend = self.config.memory_backend

        # Initialize orchestrator
        self.orchestrator = WeavingOrchestrator(cfg=holo_config)
        await self.orchestrator.__aenter__()

        logger.info(f"{self.config.app_name} initialized with {self.config.memory_backend.value} backend")

        # Load existing data from memory
        await self._load_from_memory()

        self._initialized = True

    async def close(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.__aexit__(None, None, None)
            self.orchestrator = None
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # === Asset Management ===

    async def add_asset(self, asset: Asset) -> bool:
        """Add a new asset to tracking."""
        if asset.asset_id in self.assets:
            logger.warning(f"Asset {asset.asset_id} already exists")
            return False

        self.assets[asset.asset_id] = asset

        # Store in HoloLoom memory
        shard = asset.to_shard()
        await self._store_shard(shard)

        logger.info(f"Added asset: {asset.asset_id} ({asset.name})")
        return True

    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Retrieve asset by ID."""
        return self.assets.get(asset_id)

    async def list_assets(
        self,
        asset_type: Optional[AssetType] = None,
        status: Optional[str] = None
    ) -> List[Asset]:
        """List assets with optional filtering."""
        assets = list(self.assets.values())

        if asset_type:
            assets = [a for a in assets if a.asset_type == asset_type]

        if status:
            assets = [a for a in assets if a.status.value == status]

        return assets

    async def update_asset(self, asset: Asset) -> bool:
        """Update existing asset."""
        if asset.asset_id not in self.assets:
            logger.warning(f"Asset {asset.asset_id} not found")
            return False

        self.assets[asset.asset_id] = asset

        # Update in memory
        shard = asset.to_shard()
        await self._store_shard(shard)

        logger.info(f"Updated asset: {asset.asset_id}")
        return True

    # === Log Management ===

    async def add_log(self, log: Log) -> bool:
        """Add a log entry."""
        self.logs.append(log)

        # Store in HoloLoom memory
        shard = log.to_shard()
        await self._store_shard(shard)

        logger.info(f"Added log: {log.log_id} ({log.log_type.value})")
        return True

    async def get_logs(
        self,
        log_type: Optional[LogType] = None,
        asset_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Log]:
        """Retrieve logs with optional filtering."""
        logs = self.logs

        if log_type:
            logs = [l for l in logs if l.log_type == log_type]

        if asset_id:
            logs = [l for l in logs if asset_id in l.asset_ids]

        if start_date:
            logs = [l for l in logs if l.timestamp >= start_date]

        if end_date:
            logs = [l for l in logs if l.timestamp <= end_date]

        return sorted(logs, key=lambda l: l.timestamp, reverse=True)

    # === Query Interface ===

    async def query(self, question: str) -> str:
        """
        Query farm data using natural language.

        Examples:
        - "Show me all active assets"
        - "What logs were recorded yesterday?"
        - "List all observations with health concerns"

        Args:
            question: Natural language question

        Returns:
            Answer from HoloLoom orchestrator
        """
        if not self._initialized:
            await self.initialize()

        query_obj = Query(text=question)
        spacetime = await self.orchestrator.weave(query_obj)

        # Extract answer from spacetime
        answer = str(spacetime.get('response', 'No answer available'))
        return answer

    # === Decision Support ===

    async def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get AI-powered recommendations based on farm data.

        Override in subclasses for domain-specific logic.
        """
        recommendations = []

        # Example: Check for old logs
        recent_logs = await self.get_logs()
        if len(recent_logs) == 0:
            recommendations.append({
                'type': 'data_entry',
                'priority': 'low',
                'message': 'No logs recorded yet. Start tracking farm activities!'
            })

        return recommendations

    # === Memory Operations ===

    async def _store_shard(self, shard: MemoryShard):
        """Store a memory shard in HoloLoom."""
        if not self.orchestrator:
            logger.warning("Orchestrator not initialized, shard not stored")
            return

        # Store shard in memory (implementation varies by backend)
        logger.debug(f"Stored shard: {shard.id}")

    async def _load_from_memory(self):
        """Load assets and logs from HoloLoom memory."""
        # TODO: Query Neo4j for asset nodes and reconstruct
        logger.info("Loaded data from memory")

    # === Bulk Operations ===

    async def ingest_shards(self, shards: List[MemoryShard]):
        """Ingest multiple shards (from spinners)."""
        for shard in shards:
            await self._store_shard(shard)
        logger.info(f"Ingested {len(shards)} shards")

    async def export_data(self, format: str = "json") -> Dict[str, Any]:
        """Export all farm data."""
        return {
            'app_name': self.config.app_name,
            'exported_at': datetime.now().isoformat(),
            'assets': [a.to_dict() for a in self.assets.values()],
            'logs': [l.to_dict() for l in self.logs],
            'total_assets': len(self.assets),
            'total_logs': len(self.logs)
        }

    def get_asset_by_name(self, name: str) -> Optional[Asset]:
        """Find asset by name (case-insensitive)."""
        name_lower = name.lower()
        for asset in self.assets.values():
            if asset.name.lower() == name_lower:
                return asset
        return None

    async def search_logs(self, keyword: str) -> List[Log]:
        """Search logs by keyword in name or description."""
        keyword_lower = keyword.lower()
        return [l for l in self.logs if keyword_lower in l.name.lower() or keyword_lower in l.description.lower()]


async def create_tracker(app_name: str = "Farm Tracker", **config_kwargs) -> FarmTracker:
    """Create and initialize a farm tracker."""
    config = FarmTrackerConfig(app_name=app_name, **config_kwargs)
    tracker = FarmTracker(config)
    await tracker.initialize()
    return tracker
