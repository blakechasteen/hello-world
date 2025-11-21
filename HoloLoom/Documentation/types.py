"""
Legacy types for backward compatibility

This module provides simple dataclass versions of types used by older
HoloLoom modules (e.g., cache.py) for backward compatibility.

**Created: 2025-11-21**
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class Query:
    """Simple query wrapper with text attribute"""
    text: str


@dataclass
class Features:
    """Feature extraction results"""
    motifs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class Context:
    """
    Retrieved context for a query

    Attributes:
        hits: List of (shard, score) tuples
        kg_sub: Knowledge graph subgraph
        shard_texts: List of retrieved text snippets
        relevance: Overall relevance score
    """
    hits: List[Tuple[Any, float]] = field(default_factory=list)
    kg_sub: Any = None
    shard_texts: List[str] = field(default_factory=list)
    relevance: float = 0.0


# Type alias for vector embeddings
Vector = np.ndarray


__all__ = ["Query", "Context", "Features", "Vector"]
