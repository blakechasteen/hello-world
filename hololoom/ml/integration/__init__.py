"""
HoloLoom ML Integration Adapters

Adapters for integrating with HoloLoom's data ingestion and quality systems.

Created: 2025-12-31
"""

from HoloLoom.ml.integration.spinningwheel_adapter import SpinningWheelMLAdapter
from HoloLoom.ml.integration.datapig_adapter import DataPigMLAdapter

__all__ = [
    "SpinningWheelMLAdapter",
    "DataPigMLAdapter",
]
