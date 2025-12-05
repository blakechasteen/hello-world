"""
HoloLoom Bridge: Connects Portal to HoloLoom's Memory and Reasoning Systems

Provides async HTTP bridge to HoloLoom server for:
- Memory recall (semantic search)
- Experience storage (add to memory)
- Weaving cycles (full reasoning)
- System status monitoring

Clean public API with graceful fallback if HoloLoom unavailable.
"""

from .bridge import (
    HoloLoomBridge,
    LoomQuery,
    LoomResult,
)
from .config import BridgeConfig, load_bridge_config

__all__ = [
    "HoloLoomBridge",
    "BridgeConfig",
    "LoomQuery",
    "LoomResult",
    "load_bridge_config",
]

__version__ = "0.1.0"
