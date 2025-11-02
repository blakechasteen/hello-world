"""
Farm Management Core Framework
================================
Shared infrastructure for all farm management apps built on HoloLoom.

Inspired by farmOS data model:
- Assets: Things on the farm (hives, animals, equipment, fields)
- Logs: Events and observations (inspections, treatments, harvests)
- Memory: Neo4j + Qdrant for semantic search and graph relationships
- Query: Natural language interface via WeavingOrchestrator
"""

from .models import Asset, Log, AssetType, LogType

# Tracker import commented out to avoid HoloLoom dependency issues in demos
# from .tracker import FarmTracker, FarmTrackerConfig

__all__ = [
    'Asset',
    'Log',
    'AssetType',
    'LogType',
    # 'FarmTracker',
    # 'FarmTrackerConfig'
]
