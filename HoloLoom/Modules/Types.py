"""
types.py — Core Shared Types
============================

Defines the fundamental dataclasses used throughout HoloLoom.
Keep these pure: no heavy dependencies, no side effects, no imports
from other modules.

All downstream modules (motif, embedding, memory, policy, etc.)
should import from here.
"""

import uuid
import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ============================================================================
# Base Query Object
# ============================================================================

@dataclass
class Query:
    """
    The fundamental input unit of the HoloLoom pipeline.

    Attributes:
        text: The raw query or utterance.
        metadata: Arbitrary contextual or provenance info.
        id: Unique identifier for this query (auto-generated).
        timestamp: UNIX epoch timestamp when the query was created.
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict representation."""
        return asdict(self)

    @property
    def signature(self) -> str:
        """Lightweight hash for deduplication or tracing."""
        return hashlib.sha1(self.text.encode()).hexdigest()[:12]

    def to_json(self, indent: int = 2) -> str:
        """Return a stable JSON string of the Query."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def __repr__(self) -> str:
        return f"<Query {self.id[:8]}: {self.text[:40]!r}>"


# ============================================================================
# Optional Derived Types (extend later, but defined here for consistency)
# ============================================================================

@dataclass
class Motif:
    """A detected linguistic or semantic pattern in a query or context."""
    pattern: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Information about the provenance or retrieval context."""
    sources: List[str]
    confidence: Optional[float] = None


@dataclass
class Response:
    """
    Standardized model output type.

    Attributes:
        text: The generated or retrieved text.
        confidence: Confidence score (0.0–1.0).
        context: Optional context information.
        metadata: Additional result info.
    """
    text: str
    confidence: float = 1.0
    context: Optional[Context] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzedQuery(Query):
    """
    Query object augmented with downstream analysis outputs.
    Keeps base Query minimal while allowing pipeline extensions.
    """
    embeddings: Optional[List[float]] = None
    motifs: List[Motif] = field(default_factory=list)


# ============================================================================
# Export Control
# ============================================================================

__all__ = [
    "Query",
    "Motif",
    "Context",
    "Response",
    "AnalyzedQuery",
]

