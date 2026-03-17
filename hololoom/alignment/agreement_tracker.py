"""
AgreementTracker — rolling statistics for primary/shadow detector comparison.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ComparisonSample:
    """Single comparison between primary and shadow detector."""
    primary_score: float
    shadow_score: float
    delta: float  # abs(primary - shadow)
    agreed: bool  # both above/below 0.5 threshold


@dataclass
class AgreementStats:
    """Aggregate statistics for a checkpoint."""
    total_samples: int
    agreement_rate: float
    mean_delta: float
    max_delta: float
    stddev_delta: float


class AgreementTracker:
    """Tracks agreement between primary and shadow detectors per checkpoint.

    Maintains a rolling window of comparison samples and computes
    statistics used by GraduationPolicy to decide promotion readiness.
    """

    def __init__(
        self,
        checkpoint: str,
        window_size: int = 1000,
        persist_dir: Optional[str] = None,
        agreement_threshold: float = 0.5,  # score threshold for agree/disagree
    ):
        self.checkpoint = checkpoint
        self.window_size = window_size
        self.agreement_threshold = agreement_threshold
        self._samples: deque[ComparisonSample] = deque(maxlen=window_size)

        self._persist_path: Optional[Path] = None
        if persist_dir:
            self._persist_path = Path(persist_dir) / f"{checkpoint}.json"
            self._load()

    def record(self, primary_score: float, shadow_score: float) -> ComparisonSample:
        """Record a comparison between primary and shadow scores."""
        delta = abs(primary_score - shadow_score)
        primary_positive = primary_score >= self.agreement_threshold
        shadow_positive = shadow_score >= self.agreement_threshold
        agreed = primary_positive == shadow_positive

        sample = ComparisonSample(
            primary_score=primary_score,
            shadow_score=shadow_score,
            delta=delta,
            agreed=agreed,
        )
        self._samples.append(sample)
        return sample

    def stats(self) -> Optional[AgreementStats]:
        """Compute aggregate statistics. Returns None if no samples."""
        if not self._samples:
            return None

        deltas = [s.delta for s in self._samples]
        return AgreementStats(
            total_samples=len(self._samples),
            agreement_rate=sum(1 for s in self._samples if s.agreed) / len(self._samples),
            mean_delta=statistics.mean(deltas),
            max_delta=max(deltas),
            stddev_delta=statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        )

    def save(self) -> None:
        """Persist samples to disk."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"primary": s.primary_score, "shadow": s.shadow_score,
             "delta": s.delta, "agreed": s.agreed}
            for s in self._samples
        ]
        with open(self._persist_path, "w") as f:
            json.dump(data, f)
        logger.debug(f"AgreementTracker[{self.checkpoint}] saved {len(data)} samples")

    def _load(self) -> None:
        """Load persisted samples."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            for item in data:
                self._samples.append(ComparisonSample(
                    primary_score=item["primary"],
                    shadow_score=item["shadow"],
                    delta=item["delta"],
                    agreed=item["agreed"],
                ))
            logger.debug(f"AgreementTracker[{self.checkpoint}] loaded {len(data)} samples")
        except Exception as e:
            logger.warning(f"AgreementTracker[{self.checkpoint}] load failed: {e}")
