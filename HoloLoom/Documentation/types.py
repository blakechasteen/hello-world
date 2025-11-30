"""
Backward compatibility module for HoloLoom.documentation.types

This module re-exports all types from HoloLoom.protocols.types
for backward compatibility with legacy imports.

New code should use: from HoloLoom.protocols.types import ...
"""

# Re-export everything from protocols.types
from HoloLoom.protocols.types import *

# Explicit re-exports for common types
from HoloLoom.protocols.types import (
    Query,
    Context,
    Features,
    MemoryShard,
    ComplexityLevel,
    BanditStrategy,
    ProvenanceTrace,
)

# Try to export additional types that may exist
try:
    from HoloLoom.protocols.types import Spacetime
except ImportError:
    pass

try:
    from HoloLoom.protocols.types import Vector
except ImportError:
    # Define Vector as fallback
    from typing import List
    Vector = List[float]
