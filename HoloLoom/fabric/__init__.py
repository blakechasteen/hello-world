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

# Synthesis (merged from HoloLoom.synthesis, 2026-02-27)
from .enriched_memory import EnrichedMemory, ReasoningType, MemoryEnricher
from .pattern_extractor import PatternExtractor, Pattern, PatternType
from .data_synthesizer import DataSynthesizer, TrainingExample, SynthesisConfig
from .synthesis_bridge import SynthesisBridge, SynthesisResult, create_synthesis_bridge

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

    # ===== Synthesis (merged from HoloLoom.synthesis) =====
    "EnrichedMemory",
    "ReasoningType",
    "MemoryEnricher",
    "PatternExtractor",
    "Pattern",
    "PatternType",
    "DataSynthesizer",
    "TrainingExample",
    "SynthesisConfig",
    "SynthesisBridge",
    "SynthesisResult",
    "create_synthesis_bridge",
]
