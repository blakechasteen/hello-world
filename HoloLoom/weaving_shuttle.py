#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Weaving Shuttle - Compatibility Shim
==============================================

DEPRECATED: This module is maintained for backward compatibility only.

The Shuttle architecture has been integrated into the canonical WeavingOrchestrator.
All new code should import from weaving_orchestrator instead:

    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

This compatibility shim will be maintained through mythRL 1.x for backward compatibility
but may be removed in 2.0.

Integration History:
- 2025-10-26: WeavingShuttle created with mythRL protocol enhancements
- 2025-10-27: Task 1.2 - Shuttle architecture integrated into WeavingOrchestrator
- 2025-10-27: weaving_shuttle.py converted to compatibility shim

See: SCOPE_AND_SEQUENCE.md Task 1.2 for integration details.
"""

import warnings

# Import the canonical implementation
from HoloLoom.weaving_orchestrator import (
    WeavingOrchestrator,
    ToolExecutor,
    YarnGraph
)

# Compatibility alias
WeavingShuttle = WeavingOrchestrator

# Emit deprecation warning on import
warnings.warn(
    "WeavingShuttle is deprecated. Use WeavingOrchestrator instead:\n"
    "  from HoloLoom.weaving_orchestrator import WeavingOrchestrator\n"
    "This compatibility shim will be removed in mythRL 2.0.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['WeavingShuttle', 'WeavingOrchestrator', 'ToolExecutor', 'YarnGraph']
