# -*- coding: utf-8 -*-
"""
SpinningWheel Enrichment Submodules
====================================

Modular enrichment strategies for augmenting raw data before it enters
the HoloLoom orchestrator. Each enricher focuses on a specific aspect:

- **MetadataEnricher**: Extract structured metadata (tags, priority, categories)
- **SemanticEnricher**: Extract entities, motifs, sentiment (via Ollama/LLMs)
- **TemporalEnricher**: Extract dates, timelines, temporal relationships
- **GraphEnricher**: Lookup related entities in knowledge graphs (Neo4j)

Philosophy:
-----------
Enrichment is:
- **Optional**: Can be disabled for speed
- **Modular**: Each enricher is independent
- **Composable**: Can chain multiple enrichers
- **Lightweight**: Quick extraction, heavy processing in Orchestrator

Usage:
------
    >>> from SpinningWheel.enrichment import SemanticEnricher, TemporalEnricher
    >>> 
    >>> # Create enrichers
    >>> semantic = SemanticEnricher(model="llama3.2:3b")
    >>> temporal = TemporalEnricher()
    >>> 
    >>> # Enrich text
    >>> text = "On September 15th, I inspected hive Jodi and found mites."
    >>> semantic_data = await semantic.enrich(text)
    >>> temporal_data = await temporal.enrich(text)
    >>> 
    >>> # Results:
    >>> # semantic_data = {'entities': ['hive Jodi', 'mites'], 'motifs': ['INSPECTION']}
    >>> # temporal_data = {'dates': ['2025-09-15'], 'relative': 'past'}

Enrichment Pipeline:
--------------------
    Raw Text
        ↓
    MetadataEnricher → tags, priority, category
        ↓
    SemanticEnricher → entities, motifs, sentiment
        ↓
    TemporalEnricher → dates, timelines, sequences
        ↓
    GraphEnricher → related entities from Neo4j
        ↓
    Enriched MemoryShard

Version: 0.1.0
Date: 2025-10-13
Author: blakechasteen
"""

from .base import BaseEnricher, EnrichmentResult
from .metadata import MetadataEnricher
from .semantics import SemanticEnricher
from .temporal import TemporalEnricher

# Future enrichers
# from .graph import GraphEnricher

__all__ = [
    "BaseEnricher",
    "EnrichmentResult",
    "MetadataEnricher",
    "SemanticEnricher",
    "TemporalEnricher",
    # "GraphEnricher",
]

__version__ = "0.1.0"