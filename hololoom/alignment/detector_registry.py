"""
DetectorRegistry — configuration-driven detector binding with shadow mode.

Each checkpoint has a primary detector (enforced) and an optional shadow
detector (logged for comparison). The registry drives the AgreementTracker
and GraduationPolicy for each checkpoint that has a shadow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .feature_detector import FeatureDetector, DetectionResult
from .agreement_tracker import AgreementTracker
from .graduation_policy import GraduationPolicy, GraduationCriteria, GraduationVerdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointBinding:
    """A detector slot: primary + optional shadow."""
    primary: FeatureDetector
    shadow: Optional[FeatureDetector] = None
    tracker: Optional[AgreementTracker] = None
    graduation: Optional[GraduationPolicy] = None
    _previous_primary: Optional[FeatureDetector] = field(default=None, repr=False)


class DetectorRegistry:
    """Manages primary and shadow detectors per checkpoint.

    Supports registration, evaluation (primary + shadow), graduation
    checks, promotion of shadow to primary, and rollback.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._bindings: Dict[str, CheckpointBinding] = {}
        self._persist_dir = persist_dir

    def register(
        self,
        checkpoint: str,
        primary: FeatureDetector,
        shadow: Optional[FeatureDetector] = None,
        graduation_criteria: Optional[GraduationCriteria] = None,
    ) -> None:
        """Register a detector binding for a checkpoint."""
        tracker = None
        graduation = None
        if shadow is not None:
            tracker = AgreementTracker(
                checkpoint=checkpoint,
                persist_dir=self._persist_dir,
            )
            graduation = GraduationPolicy(
                graduation_criteria or GraduationCriteria()
            )

        self._bindings[checkpoint] = CheckpointBinding(
            primary=primary,
            shadow=shadow,
            tracker=tracker,
            graduation=graduation,
        )
        logger.info(
            f"DetectorRegistry: registered '{checkpoint}' "
            f"(primary={type(primary).__name__}, "
            f"shadow={type(shadow).__name__ if shadow else 'none'})"
        )

    def evaluate(
        self,
        checkpoint: str,
        text: str,
        context: Optional[dict] = None,
    ) -> Tuple[DetectionResult, Optional[DetectionResult]]:
        """Run primary (and shadow if present) detector for a checkpoint.

        Returns (primary_result, shadow_result_or_None).
        Also records comparison in AgreementTracker.
        """
        binding = self._bindings.get(checkpoint)
        if binding is None:
            raise KeyError(f"No detector registered for checkpoint '{checkpoint}'")

        primary_result = binding.primary.detect(text, context)
        shadow_result = None

        if binding.shadow is not None:
            try:
                shadow_result = binding.shadow.detect(text, context)
                if binding.tracker:
                    binding.tracker.record(primary_result.score, shadow_result.score)
            except Exception as e:
                logger.warning(f"Shadow detector for '{checkpoint}' failed: {e}")

        return primary_result, shadow_result

    def check_graduation(self, checkpoint: str) -> Optional[GraduationVerdict]:
        """Check if a shadow detector is ready for promotion."""
        binding = self._bindings.get(checkpoint)
        if binding is None or binding.graduation is None or binding.tracker is None:
            return None
        stats = binding.tracker.stats()
        return binding.graduation.evaluate(stats)

    def promote_shadow(self, checkpoint: str) -> bool:
        """Promote shadow to primary. Stores previous primary for rollback."""
        binding = self._bindings.get(checkpoint)
        if binding is None or binding.shadow is None:
            return False

        binding._previous_primary = binding.primary
        binding.primary = binding.shadow
        binding.shadow = None
        binding.tracker = None
        binding.graduation = None

        logger.info(f"DetectorRegistry: promoted shadow to primary for '{checkpoint}'")
        return True

    def rollback(self, checkpoint: str) -> bool:
        """Rollback to previous primary detector."""
        binding = self._bindings.get(checkpoint)
        if binding is None or binding._previous_primary is None:
            return False

        binding.primary = binding._previous_primary
        binding._previous_primary = None

        logger.info(f"DetectorRegistry: rolled back to previous primary for '{checkpoint}'")
        return True

    def save_all(self) -> None:
        """Persist all agreement trackers."""
        for binding in self._bindings.values():
            if binding.tracker:
                binding.tracker.save()

    @property
    def checkpoints(self) -> list[str]:
        """List all registered checkpoint names."""
        return list(self._bindings.keys())

    def get_binding(self, checkpoint: str) -> Optional[CheckpointBinding]:
        """Get the binding for a checkpoint, or None."""
        return self._bindings.get(checkpoint)
