"""
GovernanceFeedbackCollector — enriches activation samples with governance labels.

Records outcomes from the governance pipeline (detector scores, RBAC decisions,
safety evaluations) alongside activation buffer samples, producing labeled
training sets for supervised SAE fine-tuning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GovernanceLabel:
    """Labels from a governance evaluation."""
    checkpoint: str                   # e.g., "deception", "adversarial"
    primary_score: float
    shadow_score: float | None = None
    governance_outcome: str = "pass"  # "pass", "blocked", "escalated"
    metadata: dict = field(default_factory=dict)


@dataclass
class LabeledSample:
    """An activation sample enriched with governance labels."""
    text: str
    timestamp: str
    labels: list[GovernanceLabel]
    activation_source: str | None = None  # which buffer source
    correlation_id: str | None = None


class GovernanceFeedbackCollector:
    """Collects governance labels paired with activation samples.

    Designed to produce training data for supervised SAE fine-tuning:
    the SAE learns to decompose activations into features that correlate
    with governance decisions (deception, power-seeking, etc.).

    Usage:
        collector = GovernanceFeedbackCollector(output_dir="./data/governance_labels")
        collector.record_outcome(
            text="some response",
            checkpoint="deception",
            primary_score=0.12,
            shadow_score=0.08,
            governance_outcome="pass",
        )
        collector.flush()  # writes labeled batch to disk
    """

    def __init__(
        self,
        output_dir: str = "./data/governance_labels",
        batch_size: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self._buffer: list[LabeledSample] = []
        self._total_written = 0

    def record_outcome(
        self,
        text: str,
        checkpoint: str,
        primary_score: float,
        shadow_score: float | None = None,
        governance_outcome: str = "pass",
        correlation_id: str | None = None,
        activation_source: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a single governance outcome."""
        label = GovernanceLabel(
            checkpoint=checkpoint,
            primary_score=primary_score,
            shadow_score=shadow_score,
            governance_outcome=governance_outcome,
            metadata=metadata or {},
        )

        # Check if we already have a sample for this correlation_id
        existing = None
        if correlation_id:
            for sample in self._buffer:
                if sample.correlation_id == correlation_id:
                    existing = sample
                    break

        if existing:
            existing.labels.append(label)
        else:
            self._buffer.append(LabeledSample(
                text=text,
                timestamp=datetime.now().isoformat(),
                labels=[label],
                activation_source=activation_source,
                correlation_id=correlation_id,
            ))

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered samples to disk as a JSON batch."""
        if not self._buffer:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"labels_{batch_id}.json"

        data = []
        for sample in self._buffer:
            data.append({
                "text": sample.text,
                "timestamp": sample.timestamp,
                "correlation_id": sample.correlation_id,
                "activation_source": sample.activation_source,
                "labels": [
                    {
                        "checkpoint": l.checkpoint,
                        "primary_score": l.primary_score,
                        "shadow_score": l.shadow_score,
                        "governance_outcome": l.governance_outcome,
                        "metadata": l.metadata,
                    }
                    for l in sample.labels
                ],
            })

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        self._total_written += len(self._buffer)
        logger.info(
            f"GovernanceFeedbackCollector: flushed {len(self._buffer)} samples "
            f"to {output_path} (total: {self._total_written})"
        )
        self._buffer.clear()

    @property
    def pending_count(self) -> int:
        return len(self._buffer)

    @property
    def total_written(self) -> int:
        return self._total_written
