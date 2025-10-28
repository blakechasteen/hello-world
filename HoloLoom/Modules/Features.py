"""
HoloLoom Feature Extraction
============================
Transforms raw queries into structured Features (Ψ + motifs).

This is the "threading" phase: we take the wool (Query) and prepare the warp threads
(Features) that the shuttle will weave across.

Architecture Notes:
- Does NOT import from motif/ or embedding/ modules directly
- Uses Protocol-based dependency injection instead
- Orchestrator passes in the concrete implementations
- This keeps modules independent while enabling composition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, List, Optional, runtime_checkable, Dict
import logging
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import shared types from the canonical location
from HoloLoom.documentation.types import Query, Features, Vector, Motif

# Import canonical protocols
from HoloLoom.protocols import MotifDetector as CanonicalMotifDetector
from HoloLoom.protocols import Embedder as CanonicalEmbedder


# ============================================================================
# Protocols - DEPRECATED (Import from HoloLoom.protocols instead)
# ============================================================================

@runtime_checkable
class _DeprecatedMotifDetector(Protocol):
    """
    DEPRECATED: This local protocol definition is deprecated.
    Use 'from HoloLoom.protocols import MotifDetector' instead.
    
    Protocol for motif detection implementations.
    """
    def detect(self, text: str) -> List[Motif]: ...
    # Optional async variant; implement if you like, otherwise the extractor will wrap in threadpool.
    async def detect_async(self, text: str) -> List[Motif]: ...  # type: ignore[override]


@runtime_checkable
class _DeprecatedEmbedder(Protocol):
    """
    DEPRECATED: This local protocol definition is deprecated.
    Use 'from HoloLoom.protocols import Embedder' instead.
    
    Protocol for embedding implementations.
    """
    def embed(self, text: str) -> Vector: ...
    # Optional: expose dimension for better fallback behavior
    @property
    def dim(self) -> int: ...  # type: ignore[override]
    # Optional async variant
    async def embed_async(self, text: str) -> Vector: ...  # type: ignore[override]


# Backward compatibility aliases
MotifDetector = CanonicalMotifDetector
Embedder = CanonicalEmbedder

# Emit deprecation warning on import
import warnings
warnings.warn(
    "Local protocol definitions in Features.py are deprecated. "
    "Import from HoloLoom.protocols instead: "
    "from HoloLoom.protocols import MotifDetector, Embedder",
    DeprecationWarning,
    stacklevel=2
)


# ============================================================================
# Feature Extractor - Composes motif + embedding
# ============================================================================

@dataclass(slots=True)
class FeatureExtractor:
    """
    Extracts Features from Queries by composing motif detection and embedding.
    
    This is the "warp preparation" - we take raw wool and create structured threads.

    Design Pattern: Dependency Injection
    - Takes Protocol-based components (not concrete implementations)
    - Orchestrator decides which implementations to use
    - This module stays independent and testable
    """
    motif_detector: MotifDetector
    embedder: Embedder
    logger: Optional[logging.Logger] = None
    default_fallback_dim: int = 384  # used only if embedder doesn't expose dim and embed() fails

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    def extract(self, query: Query) -> Features:
        """
        Extract Features from a Query.
        Pipeline:
        1. Detect motifs/patterns in query text
        2. Generate embedding vector (Ψ)
        3. Combine into Features object
        """
        self.logger.debug("Extracting features from query (len=%d)", len(query.text))

        motifs = self._detect_motifs(query)
        psi = self._generate_embedding(query, motifs)

        features = Features(
            psi=psi,
            motifs=motifs,
            metadata={
                "query_length": len(query.text),
                "num_motifs": len(motifs),
                "embedding_dim": len(psi),
                **query.metadata,  # propagate query metadata
            },
        )
        self.logger.debug("Extracted %d motifs; embedding_dim=%d", len(motifs), len(psi))
        return features

    def _detect_motifs(self, query: Query) -> List[Motif]:
        try:
            motifs = self.motif_detector.detect(query.text)
            self.logger.debug("Detected %d motifs", len(motifs))
            return motifs
        except Exception as e:
            self.logger.exception("Motif detection failed: %s", e)
            return []  # graceful degradation

    def _generate_embedding(self, query: Query, motifs: List[Motif]) -> Vector:
        try:
            psi = self.embedder.embed(query.text)
            return psi
        except Exception as e:
            self.logger.exception("Embedding generation failed: %s", e)
            # Prefer embedder.dim if present, else default
            dim = getattr(self.embedder, "dim", None)
            if not isinstance(dim, int) or dim <= 0:
                dim = self.default_fallback_dim
            return [0.0] * dim


# ============================================================================
# Optional: Advanced Feature Extractors
# ============================================================================

@dataclass(slots=True)
class ParallelFeatureExtractor(FeatureExtractor):
    """
    Runs motif detection and embedding in parallel.
    - Uses async if implementations provide it
    - Otherwise wraps sync calls in a threadpool
    """

    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2))

    async def extract_async(self, query: Query) -> Features:
        motifs, psi = await asyncio.gather(
            self._detect_motifs_async(query),
            self._generate_embedding_async(query),
        )

        return Features(
            psi=psi,
            motifs=motifs,
            metadata={
                "query_length": len(query.text),
                "num_motifs": len(motifs),
                "embedding_dim": len(psi),
                "extraction_mode": "parallel",
                **query.metadata,
            },
        )

    async def _detect_motifs_async(self, query: Query) -> List[Motif]:
        # If the implementation provides an async method, use it; else wrap sync in a threadpool.
        try:
            if hasattr(self.motif_detector, "detect_async"):
                return await self.motif_detector.detect_async(query.text)  # type: ignore[attr-defined]
        except TypeError:
            # detect_async exists but isn't awaitable; fall through to threadpool
            pass
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.motif_detector.detect, query.text)

    async def _generate_embedding_async(self, query: Query) -> Vector:
        try:
            if hasattr(self.embedder, "embed_async"):
                return await self.embedder.embed_async(query.text)  # type: ignore[attr-defined]
        except TypeError:
            pass
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self.embedder.embed, query.text)
        except Exception as e:
            self.logger.exception("Async embedding generation failed: %s", e)
            dim = getattr(self.embedder, "dim", None)
            if not isinstance(dim, int) or dim <= 0:
                dim = self.default_fallback_dim
            return [0.0] * dim


@dataclass(slots=True)
class CachedFeatureExtractor(FeatureExtractor):
    """
    Caches feature extraction results.
    Useful for:
    - Repeated queries
    - Interactive sessions
    - Development/debugging
    """
    _cache: Dict[str, Features] = field(default_factory=dict)

    def extract(self, query: Query) -> Features:
        key = self._make_cache_key(query)
        hit = key in self._cache
        if hit:
            self.logger.debug("Cache hit for key=%s", key[:12])
            return self._cache[key]

        features = super().extract(query)
        self._cache[key] = features
        return features

    def _make_cache_key(self, query: Query) -> str:
        """
        Deterministic cache key based on query text + stable metadata subset.
        """
        # Include only stable, serializable parts of metadata (ignore volatile fields if any)
        md = query.metadata or {}
        stable_md = {k: v for k, v in md.items() if isinstance(v, (str, int, float, bool))}
        material = f"{query.text}\n{stable_md}".encode("utf-8")
        return hashlib.sha1(material).hexdigest()


# ============================================================================
# Example Usage (for documentation/testing)
# ============================================================================

if __name__ == "__main__":
    # Mock implementations (in real code, these come from motif/ and embedding/)
    class MockMotifDetector:
        def detect(self, text: str) -> List[Motif]:
            import re
            patterns = re.findall(r"\b[A-Z]{2,}\b", text)  # Find acronyms
            return [Motif(pattern=p, score=0.8, metadata={"type": "acronym"}) for p in patterns]

    class MockEmbedder:
        def __init__(self, dim: int = 8):
            self._dim = dim
        @property
        def dim(self) -> int:
            return self._dim
        def embed(self, text: str) -> Vector:
            # Simple deterministic embedding
            h = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)
            base = (h % 1024) / 1024.0
            return [base + i * 0.001 for i in range(self._dim)]

    extractor = FeatureExtractor(
        motif_detector=MockMotifDetector(),
        embedder=MockEmbedder(dim=8),
    )

    query = Query(
        text="The NASA API returns JSON data about Mars rovers",
        metadata={"user_id": "123", "session": "abc"},
    )

    features = extractor.extract(query)

    print("Extracted Features:")
    print(f"  Ψ (embedding): {features.psi[:4]}... (dim={len(features.psi)})")
    print(f"  Motifs: {[m.pattern for m in features.motifs]}")
    print(f"  Metadata: {features.metadata}")
