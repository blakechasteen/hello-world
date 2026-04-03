#!/usr/bin/env python3
"""
HoloLoom Weaving Orchestrator - Compatibility Shim
====================================================

DEPRECATED: The canonical WeavingOrchestrator now lives in
hololoom.core.orchestrator.weaving_orchestrator.

New code should import from:
    from hololoom.orchestrator import WeavingOrchestrator

This shim re-exports everything for backward compatibility and will be
maintained through HoloLoom 1.x.
"""

# Re-export everything from the canonical location.
# No deprecation warning yet — too many importers. Warnings will be
# enabled once the import migration (Phase 5) is substantially complete.
from hololoom.core.orchestrator.weaving_orchestrator import *  # noqa: F401, F403
from hololoom.core.orchestrator.weaving_orchestrator import (  # noqa: F401
    CONSCIENCE_AVAILABLE,
    WeavingOrchestrator,
    _get_jenny_renderer_map,
)

__all__ = ["WeavingOrchestrator", "CONSCIENCE_AVAILABLE"]
