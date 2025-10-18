"""
Fabric - Woven Output
======================
The structured output from the weaving process.

Exports:
- Spacetime: Complete woven fabric with lineage
- WeavingTrace: Computational trace
- FabricCollection: Batch analysis of fabrics
"""

from .spacetime import Spacetime, WeavingTrace, FabricCollection

__all__ = ["Spacetime", "WeavingTrace", "FabricCollection"]
