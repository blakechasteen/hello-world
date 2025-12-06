"""
Tapestry: Session Continuity for HoloLoom

A modular, extensible system for persisting task state across sessions,
enabling reliable long-running agent workflows.

Philosophy: Modular, extensible, elegant, tasteful.

Core Components:
- Tapestry: The woven record of work (.hololoom/tapestry.json)
- Thread: Single task in the tapestry
- Warper: Sets up warp threads before weaving
- LoomKeeper: Maintains loom state across sessions
- FabricInspector: Holistic verification (6 signals)

Usage:
    from HoloLoom.tapestry import LoomKeeper

    async with LoomKeeper() as keeper:
        async with keeper.session("Implement feature X") as ctx:
            while thread := ctx.next_thread:
                await ctx.weave(thread, my_executor)

Created: December 2025
"""

from HoloLoom.tapestry.protocol import (
    ThreadStatus,
    Thread,
    Tapestry,
    SignalResult,
    FabricCheckResult,
    TapestryBackend,
    FabricSignal,
)

__all__ = [
    # Enums
    "ThreadStatus",
    # Data classes
    "Thread",
    "Tapestry",
    "SignalResult",
    "FabricCheckResult",
    # Protocols
    "TapestryBackend",
    "FabricSignal",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "LoomKeeper":
        from HoloLoom.tapestry.keeper import LoomKeeper
        return LoomKeeper
    elif name == "Warper":
        from HoloLoom.tapestry.warper import Warper
        return Warper
    elif name == "FabricInspector":
        from HoloLoom.tapestry.inspector import FabricInspector
        return FabricInspector
    elif name == "SignalRegistry":
        from HoloLoom.tapestry.signals.registry import SignalRegistry
        return SignalRegistry
    elif name == "create_tapestry_backend":
        from HoloLoom.tapestry.factory import create_tapestry_backend
        return create_tapestry_backend
    elif name == "JsonTapestryBackend":
        from HoloLoom.tapestry.backends.json_backend import JsonTapestryBackend
        return JsonTapestryBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
