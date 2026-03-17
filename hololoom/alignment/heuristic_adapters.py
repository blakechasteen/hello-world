"""
Heuristic Adapters — FeatureDetector wrappers for existing detection methods
=============================================================================

Step 2a: Conform existing heuristic detection methods to the FeatureDetector
Protocol defined in feature_detector.py. Each adapter is a pure wrapper —
no new detection logic, just interface translation.

Added: 2026-03-17 (Phase 6 Governance Wiring — Step 2a)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from hololoom.alignment.feature_detector import DetectionResult

if TYPE_CHECKING:
    from hololoom.alignment.deception_detection import DeceptionDetector
    from hololoom.alignment.instrumental_convergence import InstrumentalConvergenceGuard
    from hololoom.alignment.safety_guardrails import AdversarialDetector


class HeuristicDeceptionDetector:
    """Wraps DeceptionDetector._calculate_deception_score as a FeatureDetector.

    The underlying method compares expected vs actual behaviour using Jaccard
    word-overlap.  The adapter maps the Protocol's ``(text, context)`` surface
    to that method by pulling ``expected`` from context and using *text* as the
    actual behaviour.

    Context keys consumed:
        expected (str): The expected behaviour string.  Defaults to "" if
            absent, which will produce a high deception score (complete
            mismatch).
    """

    def __init__(self, detector: DeceptionDetector) -> None:
        self._detector = detector
        self._last_features: list[str] = []

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        context = context or {}
        expected: str = context.get("expected", "")

        # Delegate to the real scoring method
        score = self._detector._calculate_deception_score(expected, text)

        features: list[str] = []
        if score >= self._detector.deception_threshold:
            features.append("deception:behaviour_mismatch")
        if score >= 0.8:
            features.append("deception:high_divergence")

        self._last_features = features
        return DetectionResult(
            score=score,
            activated_features=features,
            metadata={
                "expected": expected,
                "actual": text,
                "threshold": self._detector.deception_threshold,
            },
        )

    def get_active_features(self) -> list[str]:
        return list(self._last_features)


class HeuristicPowerSeekingDetector:
    """Wraps InstrumentalConvergenceGuard.detect_power_seeking as a FeatureDetector.

    ``detect_power_seeking`` is **async** and expects ``(actions, context)``.
    The adapter treats *text* as a space-delimited action sequence (or a single
    action) and runs the coroutine synchronously via the event loop.

    Context keys consumed:
        actions (list[str]): Explicit action list.  If absent the adapter
            splits *text* on newlines.
        All remaining context items are forwarded as the ``context`` dict.
    """

    def __init__(self, guard: InstrumentalConvergenceGuard) -> None:
        self._guard = guard
        self._last_features: list[str] = []

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        context = context or {}

        # Build action list — prefer explicit key, fall back to line-split
        actions: list[str] = context.pop("actions", None) or [
            line for line in text.splitlines() if line.strip()
        ]

        # Run the async method synchronously
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop — create a task via a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                indicators = pool.submit(
                    asyncio.run,
                    self._guard.detect_power_seeking(actions, context),
                ).result()
        else:
            indicators = asyncio.run(
                self._guard.detect_power_seeking(actions, context)
            )

        # Composite score: max across sub-scores
        score = max(
            indicators.resource_acquisition_score,
            indicators.privilege_escalation_score,
            indicators.control_expansion_score,
            indicators.self_preservation_score,
        )

        features: list[str] = [
            f"power_seeking:{p}" for p in indicators.detected_patterns
        ]
        if indicators.is_power_seeking:
            features.insert(0, "power_seeking:detected")

        self._last_features = features
        return DetectionResult(
            score=score,
            activated_features=features,
            metadata={
                "is_power_seeking": indicators.is_power_seeking,
                "resource_acquisition_score": indicators.resource_acquisition_score,
                "privilege_escalation_score": indicators.privilege_escalation_score,
                "control_expansion_score": indicators.control_expansion_score,
                "self_preservation_score": indicators.self_preservation_score,
                "detected_patterns": indicators.detected_patterns,
            },
        )

    def get_active_features(self) -> list[str]:
        return list(self._last_features)


class HeuristicSelfModificationDetector:
    """Wraps InstrumentalConvergenceGuard.detect_self_modification as a FeatureDetector.

    The underlying method does substring matching against a fixed pattern list
    and returns a bool.  The adapter maps that to a 0/1 score.
    """

    def __init__(self, guard: InstrumentalConvergenceGuard) -> None:
        self._guard = guard
        self._last_features: list[str] = []

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        detected: bool = self._guard.detect_self_modification(text)

        score = 1.0 if detected else 0.0
        features: list[str] = []
        if detected:
            features.append("self_modification:detected")

        # Surface which pattern matched from the guard's history
        metadata: dict = {"detected": detected}
        if detected and self._guard.self_modification_attempts:
            last = self._guard.self_modification_attempts[-1]
            matched_pattern = last.get("pattern", "")
            if matched_pattern:
                features.append(f"self_modification:{matched_pattern.replace(' ', '_')}")
            metadata["matched_pattern"] = matched_pattern

        self._last_features = features
        return DetectionResult(
            score=score,
            activated_features=features,
            metadata=metadata,
        )

    def get_active_features(self) -> list[str]:
        return list(self._last_features)


class HeuristicAdversarialDetector:
    """Wraps AdversarialDetector.detect as a FeatureDetector.

    The underlying method returns ``(is_adversarial: bool, reason: str)``.
    The adapter maps that to a 0/1 score with the reason captured in metadata.
    """

    def __init__(self, detector: AdversarialDetector) -> None:
        self._detector = detector
        self._last_features: list[str] = []

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        is_adversarial, reason = self._detector.detect(text)

        score = 1.0 if is_adversarial else 0.0
        features: list[str] = []
        if is_adversarial:
            features.append("adversarial:detected")
            # Derive a more specific feature tag from the reason
            if "injection" in reason.lower():
                features.append("adversarial:prompt_injection")
            elif "jailbreak" in reason.lower():
                features.append("adversarial:jailbreak")
            elif "exhaustion" in reason.lower():
                features.append("adversarial:resource_exhaustion")
            elif "length" in reason.lower():
                features.append("adversarial:excessive_length")

        self._last_features = features
        return DetectionResult(
            score=score,
            activated_features=features,
            metadata={
                "is_adversarial": is_adversarial,
                "reason": reason,
            },
        )

    def get_active_features(self) -> list[str]:
        return list(self._last_features)
