"""
Drift Detector — Information-Theoretic Hallucination Detection
==============================================================

Detects when LLM output has drifted from the knowledge base context
using Jensen-Shannon divergence on embedding distributions.

High divergence = the response is saying things the context doesn't support.
This is Wire 2 of the structured AI integration roadmap.

Usage:
    detector = DriftDetector(threshold=0.4)
    result = detector.check(response_embedding, context_embeddings)
    if result.drifted:
        logger.warning(f"Response drifted: JSD={result.divergence:.3f}")

Author: Claude Code (Structured AI Wiring - Wire 2)
Date: 2026-03-18
"""

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol (swappable implementations — Design Principle #1)
# ============================================================================

@runtime_checkable
class DriftDetectorProtocol(Protocol):
    """Protocol for drift detection. Implementations are swappable."""

    def check(
        self,
        response_embedding: np.ndarray,
        context_embeddings: list[np.ndarray],
    ) -> 'DriftResult':
        ...


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class DriftResult:
    """Result of a drift check."""
    divergence: float       # JSD between response and context (0 = identical, 1 = maximally different)
    drifted: bool           # True if divergence exceeds threshold
    threshold: float        # The threshold used
    context_size: int       # Number of context embeddings used
    detail: str             # Human-readable explanation


# ============================================================================
# Drift Detector
# ============================================================================

class DriftDetector:
    """
    Detects semantic drift between LLM response and knowledge base context.

    Uses Jensen-Shannon divergence on softmax-normalized embeddings.
    JSD is symmetric, bounded [0,1], and the square root is a proper metric.

    The detector computes:
    1. Centroid of context embeddings (what the KG thinks is relevant)
    2. JSD between response embedding and context centroid
    3. Flags drift when JSD > threshold

    Threshold guidance:
    - 0.1: Very strict (only near-paraphrases pass)
    - 0.3: Moderate (allows creative synthesis)
    - 0.5: Permissive (only catches major hallucinations)
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: JSD threshold above which response is considered drifted.
        """
        self.threshold = threshold
        logger.info(f"DriftDetector initialized (threshold={threshold})")

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Convert embedding to probability distribution via softmax."""
        x = np.asarray(x, dtype=np.float64).flatten()
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _jsd(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence between two distributions."""
        m = 0.5 * (p + q)
        kl_pm = np.sum(np.where(p > 0, p * np.log2(p / m), 0.0))
        kl_qm = np.sum(np.where(q > 0, q * np.log2(q / m), 0.0))
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    def check(
        self,
        response_embedding: np.ndarray,
        context_embeddings: list[np.ndarray],
    ) -> DriftResult:
        """
        Check if response has drifted from context.

        Args:
            response_embedding: Embedding of the LLM response
            context_embeddings: List of embeddings from retrieved context

        Returns:
            DriftResult with divergence score and drift flag
        """
        if not context_embeddings:
            return DriftResult(
                divergence=0.0,
                drifted=False,
                threshold=self.threshold,
                context_size=0,
                detail="No context embeddings provided, skipping drift check",
            )

        response_embedding = np.asarray(response_embedding, dtype=np.float64).flatten()

        # Compute centroid of context embeddings
        context_matrix = np.array([
            np.asarray(e, dtype=np.float64).flatten() for e in context_embeddings
        ])

        # Ensure all embeddings have same dimensionality
        dims = set(len(e) for e in context_matrix)
        if len(dims) > 1 or len(response_embedding) != context_matrix.shape[1]:
            return DriftResult(
                divergence=0.0,
                drifted=False,
                threshold=self.threshold,
                context_size=len(context_embeddings),
                detail=f"Dimension mismatch (response={len(response_embedding)}, "
                       f"context dims={dims}), skipping drift check",
            )

        centroid = context_matrix.mean(axis=0)

        # Convert to probability distributions
        p = self._softmax(response_embedding)
        q = self._softmax(centroid)

        divergence = self._jsd(p, q)
        drifted = divergence > self.threshold

        detail = (
            f"JSD={divergence:.4f} {'>' if drifted else '<='} "
            f"threshold={self.threshold} "
            f"(context_size={len(context_embeddings)})"
        )

        if drifted:
            logger.warning(f"Drift detected: {detail}")
        else:
            logger.debug(f"No drift: {detail}")

        return DriftResult(
            divergence=divergence,
            drifted=drifted,
            threshold=self.threshold,
            context_size=len(context_embeddings),
            detail=detail,
        )


# ============================================================================
# Factory
# ============================================================================

def create_drift_detector(threshold: float = 0.3) -> DriftDetector:
    """Create a drift detector with the given threshold."""
    return DriftDetector(threshold=threshold)
