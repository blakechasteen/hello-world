"""
Fabric - Woven Output
======================

The structured output from the weaving process.

Core Exports (spacetime.py):
- Spacetime: Complete woven fabric with lineage
- WeavingTrace: Computational trace
- FabricCollection: Batch analysis of fabrics

Multi-Loom Exports (fabric.py) - December 2025:
- Fabric: Extended Spacetime with perspective and epistemic metadata
- Tension: Productive disagreement between looms
- Resolution: Attempted resolution of a tension
- MultiFabricCollection: Collection of fabrics from multiple looms

Philosophy:
"The Fabric holds not just the answer, but the perspective that created it,
the blind spots it contains, and what evidence would change it."
"""

# Core Fabric (existing)
from .spacetime import Spacetime, WeavingTrace, FabricCollection

# Multi-Loom Fabric (December 2025)
from .fabric import (
    Fabric,
    Tension,
    Resolution,
    MultiFabricCollection,
)

__all__ = [
    # ===== Core Fabric =====
    "Spacetime",
    "WeavingTrace",
    "FabricCollection",

    # ===== Multi-Loom Fabric =====
    "Fabric",
    "Tension",
    "Resolution",
    "MultiFabricCollection",
]
