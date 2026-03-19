"""
HoloLoom Core Protocols - DEPRECATED
=====================================

**DEPRECATED**: This file has been split into:
- `memory_types.py` - Memory data types
- `memory_protocols.py` - Memory protocol definitions

Please import from:
    from hololoom.protocols import MemoryStore, MemoryNavigator, PatternDetector
    from hololoom.protocols import Memory, MemoryQuery, MemoryRetrievalResult

This file is kept for backward compatibility but will be removed in v2.0.

Author: HoloLoom Architecture Team
Date: 2025-01-12 (Phase 0, Task 7: Protocol Consolidation)
"""

import warnings

warnings.warn(
    "hololoom.protocols.core is deprecated. "
    "Import from hololoom.protocols instead:\n"
    "  from hololoom.protocols import MemoryStore, MemoryNavigator, PatternDetector\n"
    "  from hololoom.protocols import Memory, MemoryQuery, MemoryRetrievalResult",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility - re-export from new locations
from .memory_protocols import MemoryNavigator, MemoryStore, PatternDetector

__all__ = [
    'MemoryStore',
    'MemoryNavigator',
    'PatternDetector',
]
