"""
types.py — DEPRECATED - Use HoloLoom.documentation.types instead
===================================================================

⚠️  DEPRECATION NOTICE ⚠️

This module is deprecated. All types have been moved to:
    HoloLoom.documentation.types

This compatibility shim will be removed in mythRL 2.0.

Migration Guide:
    Old: from HoloLoom.modules.Types import Query, Motif, Context, Response
    New: from HoloLoom.Documentation.types import Query, Motif, Context, Response
"""

import warnings

# Emit deprecation warning
warnings.warn(
    "HoloLoom.modules.Types is deprecated. "
    "Use 'from HoloLoom.Documentation.types import ...' instead. "
    "This module will be removed in mythRL 2.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from canonical location for backward compatibility
from HoloLoom.Documentation.types import (
    Query,
    Motif,
    Context,
    Response,
    MemoryShard,
    Features,
    ActionPlan,
    Spacetime,
    WeavingTrace,
)

__all__ = [
    'Query',
    'Motif',
    'Context',
    'Response',
    'MemoryShard',
    'Features',
    'ActionPlan',
    'Spacetime',
    'WeavingTrace',
]
