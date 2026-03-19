"""
HoloLoom ML Integration Adapters

Adapters for integrating with HoloLoom's data ingestion and quality systems.

Created: 2025-12-31
"""

from hololoom.ml.integration.datapig_adapter import DataPigMLAdapter
from hololoom.ml.integration.spinningwheel_adapter import SpinningWheelMLAdapter

__all__ = [
    "SpinningWheelMLAdapter",
    "DataPigMLAdapter",
]
