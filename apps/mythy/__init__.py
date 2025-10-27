"""
Mythy - Narrative Intelligence for mythRL
==========================================
A comprehensive narrative intelligence system built on the HoloLoom framework.

Features:
- Joseph Campbell's 17-stage Hero's Journey analysis
- 40+ universal character database (mythology, literature, history, fiction)
- 5-level Matryoshka depth analysis (Surface → Cosmic)
- Cross-domain narrative adaptation (business, science, personal, product, history)
- Real-time streaming analysis with progressive depth gating
- High-performance caching layer

Installation:
    pip install hololoom  # Framework (required)
    pip install mythy  # This package

Usage:
    from mythy import NarrativeIntelligence

    analyzer = NarrativeIntelligence()
    result = await analyzer.analyze(text)

    print(f"Campbell Stage: {result.narrative_arc.primary_arc}")
    print(f"Characters: {[c.name for c in result.detected_characters]}")
    print(f"Themes: {result.themes}")
"""

__version__ = "0.1.0"

# Core Intelligence
from mythy.intelligence import (
    NarrativeIntelligence,
    NarrativeIntelligenceResult,
    CampbellStage,
    ArchetypeType,
    NarrativeFunction,
)

# Depth Analysis
from mythy.matryoshka_depth import (
    MatryoshkaNarrativeDepth,
    MatryoshkaDepthResult,
    DepthLevel,
    MeaningLayer,
)

# Streaming Analysis
from mythy.streaming_depth import (
    StreamingNarrativeAnalyzer,
    StreamEvent,
)

# Cross-Domain Adaptation
from mythy.cross_domain_adapter import (
    CrossDomainAdapter,
    NarrativeDomain,
    DomainMapping,
)

# Loop Engine
from mythy.loop_engine import (
    NarrativeLoopEngine,
    LoopMode,
    Priority,
)

# Caching
from mythy.cache import (
    NarrativeCache,
    CachedMatryoshkaDepth,
)

__all__ = [
    # Core
    "NarrativeIntelligence",
    "NarrativeIntelligenceResult",
    "CampbellStage",
    "ArchetypeType",
    "NarrativeFunction",

    # Depth
    "MatryoshkaNarrativeDepth",
    "MatryoshkaDepthResult",
    "DepthLevel",
    "MeaningLayer",

    # Streaming
    "StreamingNarrativeAnalyzer",
    "StreamEvent",

    # Cross-Domain
    "CrossDomainAdapter",
    "NarrativeDomain",
    "DomainMapping",

    # Loop
    "NarrativeLoopEngine",
    "LoopMode",
    "Priority",

    # Cache
    "NarrativeCache",
    "CachedMatryoshkaDepth",
]
