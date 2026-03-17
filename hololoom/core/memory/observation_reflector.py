"""
ObservationReflector — Thompson Sampling over (mode, term_name) pairs.

Closes the learning loop: when scored observations resolve, the reflector
learns which scoring terms predict good outcomes in which modes. Feeds
learned weight multipliers back into the scoring pipeline.

Reuses BanditArm from ts_core/arm.py. Follows ModelBandit's save/load
pattern (atomic JSON persistence).

Usage:
    from hololoom.memory.observation_reflector import ObservationReflector

    reflector = ObservationReflector()
    reflector.load(Path("data/reflector.json"))

    # Before scoring: get learned adjustments
    adjustments = reflector.suggest_weight_adjustments("fall", terms)
    obs = scored_observation(..., mode_weights=adjustments)

    # After resolution: feed outcome back
    reflector.record_outcome("fall", obs.properties["breakdown"], success=True)
    reflector.save(Path("data/reflector.json"))
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hololoom.ts_core.arm import BanditArm

logger = logging.getLogger(__name__)


class ObservationReflector:
    """Thompson Sampling over (mode, term_name) pairs.

    Each pair gets a BanditArm that learns from resolved observations.
    Provides weight multipliers that feed back into score_terms() via
    the mode_weights parameter.
    """

    def __init__(self):
        self._arms: Dict[Tuple[str, str], BanditArm] = {}

    def _key(self, mode: str, term_name: str) -> Tuple[str, str]:
        return (mode, term_name)

    def _get_arm(self, mode: str, term_name: str) -> BanditArm:
        key = self._key(mode, term_name)
        if key not in self._arms:
            self._arms[key] = BanditArm(name=f"{mode}:{term_name}")
        return self._arms[key]

    # =========================================================================
    # Learning
    # =========================================================================

    def record_outcome(
        self,
        mode: str,
        breakdown: Dict[str, float],
        success: bool,
        weight: float = 1.0,
        firing_threshold: float = 0.1,
    ) -> None:
        """Update bandit arms for terms that fired in this observation.

        Terms that contributed more to the composite score get
        proportionally stronger updates (weighted by normalized contribution).

        Args:
            mode: The scoring mode that was active.
            breakdown: {term_name: weighted_contribution} from score_terms().
            success: Whether the outcome was positive (action worked).
            weight: Base update weight (default 1.0).
            firing_threshold: Minimum normalized contribution to count as
                            "fired" (0.0-1.0). Terms below this are skipped.
        """
        if not breakdown:
            return

        max_contribution = max(abs(v) for v in breakdown.values())
        if max_contribution == 0:
            return

        for term_name, contribution in breakdown.items():
            normalized = abs(contribution) / max_contribution
            if normalized >= firing_threshold:
                arm = self._get_arm(mode, term_name)
                arm.update(success=success, weight=weight * normalized)

    # =========================================================================
    # Weight Adjustment
    # =========================================================================

    def multiplier(self, mode: str, term_name: str) -> float:
        """Get the learned weight multiplier for a (mode, term) pair.

        Returns a value centered around 1.0:
            mean > 0.5 → multiplier > 1.0 (term predicts good outcomes)
            mean < 0.5 → multiplier < 1.0 (term is noisy/misleading)
            mean = 0.5 → multiplier = 1.0 (no evidence)

        Clamped to [0.5, 2.0] to prevent extreme adjustments.
        """
        key = self._key(mode, term_name)
        if key not in self._arms:
            return 1.0  # No data = no adjustment

        arm = self._arms[key]
        raw = 0.5 + arm.mean  # Map [0,1] → [0.5, 1.5]
        return max(0.5, min(2.0, raw))

    def suggest_weight_adjustments(
        self,
        mode: str,
        terms: List[Any],  # List[ScoringTerm] — duck-typed to avoid circular import
    ) -> Dict[str, float]:
        """Return learned multipliers for all terms in a scoring pipeline.

        Pass the result directly as mode_weights to score_terms():
            adjustments = reflector.suggest_weight_adjustments(mode, terms)
            composite, breakdown = score_terms(terms, ctx, mode_weights=adjustments)
        """
        return {
            term.name: self.multiplier(mode, term.name)
            for term in terms
        }

    # =========================================================================
    # Observability
    # =========================================================================

    def stats(self) -> Dict[str, dict]:
        """Return bandit statistics for all arms."""
        result = {}
        for (mode, term_name), arm in self._arms.items():
            key = f"{mode}:{term_name}"
            result[key] = {
                "alpha": arm.alpha,
                "beta": arm.beta,
                "mean": round(arm.mean, 3),
                "certainty": round(arm.geodesic_certainty(), 3),
                "total_pulls": arm.total_pulls,
                "multiplier": round(self.multiplier(mode, term_name), 3),
            }
        return result

    @property
    def arm_count(self) -> int:
        return len(self._arms)

    # =========================================================================
    # Persistence (follows ModelBandit pattern)
    # =========================================================================

    def save(self, path: Path) -> None:
        """Persist reflector state to JSON (atomic write)."""
        data = {}
        for (mode, term_name), arm in self._arms.items():
            key = f"{mode}:{term_name}"
            data[key] = {"a": arm.alpha, "b": arm.beta}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
        logger.debug("Reflector state saved: %d arms → %s", len(data), path)

    def load(self, path: Path) -> None:
        """Load reflector state from JSON. Missing/corrupt file = fresh start."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            loaded = 0
            for key, vals in data.items():
                parts = key.split(":", 1)
                if len(parts) != 2:
                    continue
                mode, term_name = parts
                arm = BanditArm(
                    name=key,
                    alpha=vals["a"],
                    beta=vals["b"],
                )
                self._arms[(mode, term_name)] = arm
                loaded += 1
            logger.info("Reflector state loaded: %d arms from %s", loaded, path)
        except Exception as e:
            logger.warning("Failed to load reflector state from %s: %s", path, e)
