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
    from hololoom.tapestry import LoomKeeper

    async with LoomKeeper() as keeper:
        async with keeper.session("Implement feature X") as ctx:
            while thread := ctx.next_thread:
                await ctx.weave(thread, my_executor)

Created: December 2025
"""

from hololoom.tapestry.protocol import (
    FabricCheckResult,
    FabricSignal,
    SignalResult,
    Tapestry,
    TapestryBackend,
    Thread,
    ThreadStatus,
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
        from hololoom.tapestry.keeper import LoomKeeper
        return LoomKeeper
    elif name == "Warper":
        from hololoom.tapestry.warper import Warper
        return Warper
    elif name == "FabricInspector":
        from hololoom.tapestry.inspector import FabricInspector
        return FabricInspector
    elif name == "SignalRegistry":
        from hololoom.tapestry.signals.registry import SignalRegistry
        return SignalRegistry
    elif name == "create_tapestry_backend":
        from hololoom.tapestry.factory import create_tapestry_backend
        return create_tapestry_backend
    elif name == "JsonTapestryBackend":
        from hololoom.tapestry.backends.json_backend import JsonTapestryBackend
        return JsonTapestryBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
