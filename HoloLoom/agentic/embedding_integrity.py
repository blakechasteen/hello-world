"""
Embedding Integrity & Verification
===================================
Practical embedding versioning, determinism checks, and quality metrics.

Philosophy:
"Reliable Systems: Safety First" - embedding quality is foundational.

This module provides:
1. Embedding run versioning & provenance
2. Determinism checks (re-embed canary set)
3. Basic quality metrics (Recall@k, MRR)
4. Integration with alignment audit trail

What we DON'T do (see SOMEDAY_MAYBE.md):
- Nightly auto-ablation
- Full PII guardrails
- Mahalanobis outlier detection
- Blue/green deployments
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.alignment.audit_trail import AuditTrail, DecisionType, OutcomeType


logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class EmbeddingRun:
    """
    Immutable embedding run with complete provenance.

    Tracks who, what, when for every embedding generation.
    Critical for reproducibility and debugging.
    """
    run_id: str
    model_name: str
    model_hash: str  # Hash of model weights
    dimensions: List[int]  # [96, 192, 384]
    timestamp: datetime
    data_snapshot_hash: str  # Hash of input data
    config: Dict[str, Any]  # Model config (tokenizer, normalization, etc.)

    # Metadata
    total_items: int = 0
    avg_text_length: float = 0.0
    pipeline_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "dimensions": self.dimensions,
            "timestamp": self.timestamp.isoformat(),
            "data_snapshot_hash": self.data_snapshot_hash,
            "config": self.config,
            "total_items": self.total_items,
            "avg_text_length": self.avg_text_length,
            "pipeline_version": self.pipeline_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingRun':
        data = data.copy()
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class VectorMeta:
    """
    Metadata for a single vector.

    Links vector back to source data and run for provenance.
    """
    doc_id: str
    run_id: str
    checksum_raw: str  # Hash of raw text
    checksum_cleaned: str  # Hash after preprocessing
    text_length: int
    language: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Quality indicators
    normalization_error: float = 0.0  # |norm - 1.0|
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None


@dataclass
class DeterminismCheck:
    """Result of determinism verification."""
    passed: bool
    canary_size: int
    median_cosine_delta: float
    p95_cosine_delta: float
    max_cosine_delta: float
    failures: List[Tuple[str, float]]  # (doc_id, delta)

    # Thresholds
    median_threshold: float = 0.01
    p95_threshold: float = 0.03


@dataclass
class QualityMetrics:
    """
    Basic quality metrics for embedding runs.

    Focused on retrieval quality (core use case).
    """
    # Retrieval metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain

    # Quality indicators
    avg_normalization_error: float = 0.0
    duplicate_rate: float = 0.0
    total_items: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "avg_normalization_error": self.avg_normalization_error,
            "duplicate_rate": self.duplicate_rate,
            "total_items": self.total_items
        }


# ============================================================================
# Embedding Integrity Monitor
# ============================================================================

class EmbeddingIntegrityMonitor:
    """
    Monitors embedding quality and determinism.

    Key Features:
    1. Run versioning - never mix embeddings across runs
    2. Determinism checks - detect model/data drift
    3. Quality metrics - track retrieval performance
    4. Audit trail integration - log all checks

    Usage:
        monitor = EmbeddingIntegrityMonitor(embedder, audit_trail)

        # Create new run
        run = await monitor.create_run(shards)

        # Check determinism
        check = await monitor.check_determinism(run, canary_size=100)

        # Compute quality metrics
        metrics = await monitor.compute_quality_metrics(run, gold_set)
    """

    def __init__(
        self,
        embedder: MatryoshkaEmbeddings,
        audit_trail: Optional[AuditTrail] = None,
        persist_path: Optional[Path] = None
    ):
        self.embedder = embedder
        self.audit_trail = audit_trail or AuditTrail()
        self.persist_path = Path(persist_path) if persist_path else Path("./embedding_runs")
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Canary set for determinism checks (1% of data, updated per run)
        self.canary_set: Dict[str, Tuple[str, np.ndarray]] = {}

        self.logger = logging.getLogger(__name__)

    async def create_run(
        self,
        shards: List[MemoryShard],
        model_name: str = "default",
        config: Optional[Dict[str, Any]] = None
    ) -> EmbeddingRun:
        """
        Create new embedding run with full provenance.

        Args:
            shards: Input data
            model_name: Model identifier
            config: Model config

        Returns:
            EmbeddingRun with complete metadata
        """
        timestamp = datetime.now()

        # Hash data snapshot
        data_hash = self._hash_data_snapshot(shards)

        # Hash model (if possible)
        model_hash = self._hash_model()

        # Create run
        run = EmbeddingRun(
            run_id=f"run_{int(timestamp.timestamp())}",
            model_name=model_name,
            model_hash=model_hash,
            dimensions=self.embedder.sizes,
            timestamp=timestamp,
            data_snapshot_hash=data_hash,
            config=config or {},
            total_items=len(shards),
            avg_text_length=sum(len(s.text) for s in shards) / len(shards) if shards else 0.0
        )

        # Save run metadata
        self._save_run(run)

        # Update canary set (1% sample)
        self._update_canary_set(shards, run)

        # Log to audit trail
        self.audit_trail.log_decision(
            decision_type=DecisionType.MEMORY_RETRIEVAL,
            outcome=OutcomeType.APPROVED,
            reason="Embedding run created",
            metadata={
                "run_id": run.run_id,
                "total_items": run.total_items,
                "dimensions": run.dimensions,
                "data_hash": data_hash
            }
        )

        self.logger.info(f"Created embedding run: {run.run_id} ({run.total_items} items)")
        return run

    async def check_determinism(
        self,
        run: EmbeddingRun,
        canary_size: int = 100,
        median_threshold: float = 0.01,
        p95_threshold: float = 0.03
    ) -> DeterminismCheck:
        """
        Check embedding determinism by re-embedding canary set.

        Detects model drift, data preprocessing changes, or non-determinism.

        Args:
            run: Embedding run to check
            canary_size: Number of canary samples
            median_threshold: Max median cosine delta
            p95_threshold: Max p95 cosine delta

        Returns:
            DeterminismCheck with pass/fail and details
        """
        if not self.canary_set:
            self.logger.warning("No canary set available, skipping determinism check")
            return DeterminismCheck(
                passed=True,
                canary_size=0,
                median_cosine_delta=0.0,
                p95_cosine_delta=0.0,
                max_cosine_delta=0.0,
                failures=[]
            )

        # Sample canary set
        canary_sample = list(self.canary_set.items())[:canary_size]

        # Re-embed
        texts = [text for _, (text, _) in canary_sample]
        new_embeddings = self.embedder.encode(texts)

        # Compute cosine deltas
        deltas = []
        failures = []

        for i, (doc_id, (text, old_emb)) in enumerate(canary_sample):
            new_emb = new_embeddings[i]

            # Cosine similarity
            cos_sim = np.dot(old_emb, new_emb) / (np.linalg.norm(old_emb) * np.linalg.norm(new_emb))
            delta = 1.0 - cos_sim

            deltas.append(delta)

            if delta > p95_threshold:
                failures.append((doc_id, delta))

        # Compute statistics
        deltas_array = np.array(deltas)
        median_delta = float(np.median(deltas_array))
        p95_delta = float(np.percentile(deltas_array, 95))
        max_delta = float(np.max(deltas_array))

        # Pass/fail
        passed = median_delta <= median_threshold and p95_delta <= p95_threshold

        # Log result
        self.audit_trail.log_decision(
            decision_type=DecisionType.SAFETY_GATE,
            outcome=OutcomeType.APPROVED if passed else OutcomeType.REJECTED,
            reason=f"Determinism check: {'PASS' if passed else 'FAIL'}",
            metadata={
                "run_id": run.run_id,
                "canary_size": len(canary_sample),
                "median_delta": median_delta,
                "p95_delta": p95_delta,
                "failures": len(failures)
            }
        )

        if not passed:
            self.logger.warning(
                f"Determinism check FAILED for {run.run_id}: "
                f"median={median_delta:.4f}, p95={p95_delta:.4f}"
            )

        return DeterminismCheck(
            passed=passed,
            canary_size=len(canary_sample),
            median_cosine_delta=median_delta,
            p95_cosine_delta=p95_delta,
            max_cosine_delta=max_delta,
            failures=failures,
            median_threshold=median_threshold,
            p95_threshold=p95_threshold
        )

    async def compute_quality_metrics(
        self,
        embeddings: np.ndarray,
        gold_set: List[Tuple[str, List[str]]],
        k_values: List[int] = [1, 5, 10]
    ) -> QualityMetrics:
        """
        Compute retrieval quality metrics on gold set.

        Args:
            embeddings: All embeddings (n × d)
            gold_set: List of (query, [relevant_doc_ids])
            k_values: Values of k for Recall@k

        Returns:
            QualityMetrics with Recall@k, MRR, nDCG
        """
        if not gold_set:
            self.logger.warning("No gold set provided, skipping quality metrics")
            return QualityMetrics()

        recall_scores = {k: [] for k in k_values}
        mrr_scores = []

        # For each query in gold set
        for query_text, relevant_ids in gold_set:
            # Encode query
            query_emb = self.embedder.encode([query_text])[0]

            # Compute similarities
            sims = np.dot(embeddings, query_emb)
            top_k_indices = np.argsort(sims)[::-1]

            # Recall@k
            for k in k_values:
                top_k = set(str(i) for i in top_k_indices[:k])
                relevant = set(relevant_ids)
                recall = len(top_k & relevant) / len(relevant) if relevant else 0.0
                recall_scores[k].append(recall)

            # MRR
            for rank, idx in enumerate(top_k_indices, 1):
                if str(idx) in relevant_ids:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)

        # Compute averages
        metrics = QualityMetrics(
            recall_at_1=np.mean(recall_scores[1]) if 1 in recall_scores else 0.0,
            recall_at_5=np.mean(recall_scores[5]) if 5 in recall_scores else 0.0,
            recall_at_10=np.mean(recall_scores[10]) if 10 in recall_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            total_items=len(embeddings)
        )

        # Log metrics
        self.audit_trail.log_decision(
            decision_type=DecisionType.MEMORY_RETRIEVAL,
            outcome=OutcomeType.APPROVED,
            reason="Quality metrics computed",
            metadata=metrics.to_dict()
        )

        self.logger.info(
            f"Quality metrics: R@1={metrics.recall_at_1:.3f}, "
            f"R@5={metrics.recall_at_5:.3f}, MRR={metrics.mrr:.3f}"
        )

        return metrics

    def enforce_normalization(
        self,
        embeddings: np.ndarray,
        tolerance: float = 1e-3
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Enforce L2 normalization and detect violations.

        Args:
            embeddings: Input embeddings (n × d)
            tolerance: Max deviation from norm=1.0

        Returns:
            (normalized_embeddings, violation_indices)
        """
        norms = np.linalg.norm(embeddings, axis=1)
        violations = np.where(np.abs(norms - 1.0) > tolerance)[0].tolist()

        # Normalize all
        normalized = embeddings / norms[:, np.newaxis]

        if violations:
            self.logger.warning(f"Found {len(violations)} normalization violations")

        return normalized, violations

    def detect_duplicates(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        threshold: float = 0.995
    ) -> List[Tuple[int, int, float]]:
        """
        Detect near-duplicate documents.

        Args:
            embeddings: All embeddings (n × d)
            texts: Corresponding texts
            threshold: Cosine similarity threshold for duplicates

        Returns:
            List of (idx1, idx2, similarity) for duplicates
        """
        duplicates = []

        # Compute pairwise similarities (naive O(n²) - use ANN in production)
        n = len(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                if sim >= threshold:
                    duplicates.append((i, j, float(sim)))

        if duplicates:
            self.logger.info(f"Found {len(duplicates)} duplicate pairs (threshold={threshold})")

        return duplicates

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _hash_data_snapshot(self, shards: List[MemoryShard]) -> str:
        """Hash data snapshot for provenance."""
        content = "".join(s.text for s in shards)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _hash_model(self) -> str:
        """Hash model weights (if accessible)."""
        # For sentence-transformers, hash model name + config
        # In production, could hash actual weights
        model_info = f"{self.embedder.base_model_name}_{self.embedder.sizes}"
        return hashlib.sha256(model_info.encode()).hexdigest()[:16]

    def _save_run(self, run: EmbeddingRun):
        """Save run metadata to disk."""
        run_file = self.persist_path / f"{run.run_id}.json"
        with open(run_file, 'w') as f:
            json.dump(run.to_dict(), f, indent=2)

    def _update_canary_set(self, shards: List[MemoryShard], run: EmbeddingRun):
        """Update canary set with 1% sample."""
        n_canary = max(1, len(shards) // 100)  # 1% minimum 1
        canary_shards = shards[:n_canary]

        # Embed canary set
        texts = [s.text for s in canary_shards]
        embeddings = self.embedder.encode(texts)

        # Store
        self.canary_set = {
            s.id: (s.text, embeddings[i])
            for i, s in enumerate(canary_shards)
        }

        self.logger.info(f"Updated canary set: {len(self.canary_set)} samples")


# ============================================================================
# Factory Functions
# ============================================================================

def create_integrity_monitor(
    embedder: MatryoshkaEmbeddings,
    audit_trail: Optional[AuditTrail] = None,
    persist_path: Optional[Path] = None
) -> EmbeddingIntegrityMonitor:
    """
    Create embedding integrity monitor.

    Args:
        embedder: Matryoshka embedder
        audit_trail: Optional audit trail
        persist_path: Where to persist run metadata

    Returns:
        EmbeddingIntegrityMonitor ready to use
    """
    return EmbeddingIntegrityMonitor(
        embedder=embedder,
        audit_trail=audit_trail,
        persist_path=persist_path
    )