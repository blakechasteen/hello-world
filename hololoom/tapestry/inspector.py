"""
FabricInspector

Aggregates verification signals into holistic quality assessment.

Systems thinking: the whole is greater than the sum of parts.
Individual signals are weighted and combined, with hard blockers
overriding the aggregate score.

Usage:
    inspector = FabricInspector()
    result = await inspector.inspect(thread, context)

    if result.passed:
        print(f"Quality: {result.confidence:.1%}")
    else:
        print(f"Blockers: {result.blockers}")

Created: December 2025
"""

import asyncio
import logging
from typing import Any

from hololoom.tapestry.protocol import FabricCheckResult, FabricSignal, SignalResult, Thread
from hololoom.tapestry.signals.registry import SignalRegistry

logger = logging.getLogger(__name__)


class FabricInspector:
    """
    Aggregates signals into holistic verification.

    Systems thinking: the whole is greater than the sum of parts.

    Features:
    - Parallel signal execution
    - Weighted confidence aggregation
    - Hard blockers override everything
    - Graceful degradation on signal failures
    - Actionable recommendations
    """

    def __init__(
        self,
        signals: list[FabricSignal] | None = None,
        fail_on_missing_signals: bool = False
    ):
        """
        Initialize FabricInspector.

        Args:
            signals: Custom signals to use. None uses all registered signals.
                     Pass [] for explicit empty list (no signals).
            fail_on_missing_signals: If True, fail when no signals registered.
        """
        # Handle None vs empty list explicitly
        if signals is None:
            self.signals = SignalRegistry.get_all()
        else:
            self.signals = signals
        self.fail_on_missing_signals = fail_on_missing_signals

        if not self.signals and fail_on_missing_signals:
            raise ValueError("No verification signals registered")

        logger.debug(f"FabricInspector initialized with {len(self.signals)} signals")

    async def inspect(
        self,
        thread: Thread,
        context: dict[str, Any]
    ) -> FabricCheckResult:
        """
        Run all signals, aggregate results.

        Args:
            thread: The thread being verified
            context: Context for signals (files, test_paths, etc.)

        Returns:
            FabricCheckResult with aggregated pass/fail and confidence
        """
        if not self.signals:
            logger.warning("No signals registered, returning pass by default")
            return FabricCheckResult(
                passed=True,
                confidence=1.0,
                signals={},
                blockers=[],
                recommendations=["No verification signals configured"]
            )

        results: dict[str, SignalResult] = {}
        blockers: list[str] = []
        errors: list[str] = []

        # Run signals in parallel
        tasks = [
            self._run_signal_safe(signal, thread, context)
            for signal in self.signals
        ]
        signal_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for signal, result in zip(self.signals, signal_results):
            if isinstance(result, Exception):
                # Signal failed to execute
                logger.error(f"Signal {signal.name} failed: {result}")
                errors.append(f"{signal.name}: {str(result)}")
                continue

            if result is None:
                continue

            results[result.signal_name] = result

            # Track hard blockers
            if result.is_blocker and not result.passed:
                blocker_msg = f"{result.signal_name}: {result.details.get('error', 'Failed')}"
                blockers.append(blocker_msg)

        # Hard blockers fail regardless of weight
        if blockers:
            return FabricCheckResult(
                passed=False,
                confidence=0.0,
                signals=results,
                blockers=blockers,
                recommendations=self._generate_recommendations(results, blockers)
            )

        # Calculate weighted confidence
        confidence = self._calculate_confidence(results)

        # Determine overall pass (confidence >= 0.6)
        passed = confidence >= 0.6

        return FabricCheckResult(
            passed=passed,
            confidence=confidence,
            signals=results,
            blockers=[],
            recommendations=self._generate_recommendations(results, [])
        )

    async def _run_signal_safe(
        self,
        signal: FabricSignal,
        thread: Thread,
        context: dict[str, Any]
    ) -> SignalResult | None:
        """
        Run a signal with error handling.

        Returns None on error (graceful degradation).
        """
        try:
            return await asyncio.wait_for(
                signal.check(thread, context),
                timeout=60.0  # 1 minute timeout per signal
            )
        except asyncio.TimeoutError:
            logger.warning(f"Signal {signal.name} timed out")
            return SignalResult(
                signal_name=signal.name,
                passed=True,  # Graceful degradation
                score=0.5,
                details={"error": "Signal timed out"}
            )
        except Exception as e:
            logger.error(f"Signal {signal.name} error: {e}")
            return None

    def _calculate_confidence(self, results: dict[str, SignalResult]) -> float:
        """
        Calculate weighted confidence score.

        Formula:
            confidence = Σ(weight_i × score_i) / Σ(weight_i)

        Missing signals are treated as weight 0 (not counted).
        """
        if not results:
            return 0.5  # Neutral if no results

        total_weight = 0.0
        weighted_sum = 0.0

        for signal in self.signals:
            result = results.get(signal.name)
            if result:
                weighted_sum += signal.weight * result.score
                total_weight += signal.weight

        if total_weight == 0:
            return 0.5  # Neutral if no weights

        return weighted_sum / total_weight

    def _generate_recommendations(
        self,
        results: dict[str, SignalResult],
        blockers: list[str]
    ) -> list[str]:
        """
        Generate actionable recommendations based on results.

        Returns list of improvement suggestions.
        """
        recommendations = []

        # Blocker recommendations
        if blockers:
            recommendations.append("Fix blocking issues before proceeding:")
            for blocker in blockers[:3]:
                recommendations.append(f"  - {blocker}")

        # Low score recommendations
        for signal in self.signals:
            result = results.get(signal.name)
            if result and result.score < 0.6:
                if signal.name == "tests":
                    recommendations.append("Run and fix failing tests")
                elif signal.name == "quality":
                    issues = result.details.get("issues", [])
                    if issues:
                        recommendations.append(f"Address quality issues: {issues[0]}")
                elif signal.name == "documentation":
                    recommendations.append("Add missing docstrings")
                elif signal.name == "architecture":
                    violations = result.details.get("violations", [])
                    if violations:
                        recommendations.append(f"Fix: {violations[0]}")
                elif signal.name == "alignment":
                    recommendations.append("Review risky operations")
                elif signal.name == "integration":
                    errors = result.details.get("import_errors", [])
                    if errors:
                        recommendations.append(f"Fix import: {errors[0]}")

        return recommendations[:5]  # Limit to 5 recommendations

    def get_signal_names(self) -> list[str]:
        """Get names of all configured signals."""
        return [s.name for s in self.signals]

    def get_total_weight(self) -> float:
        """Get sum of all signal weights."""
        return sum(s.weight for s in self.signals)

    def describe(self) -> str:
        """Get human-readable description of inspector configuration."""
        lines = [
            f"FabricInspector with {len(self.signals)} signals:",
            f"  Total weight: {self.get_total_weight():.2f}"
        ]
        for signal in sorted(self.signals, key=lambda s: -s.weight):
            lines.append(f"  - {signal.name}: {signal.weight:.2f}")
        return "\n".join(lines)


async def quick_inspect(
    thread: Thread,
    context: dict[str, Any],
    signals: list[str] | None = None
) -> FabricCheckResult:
    """
    Quick inspection with default or specified signals.

    Args:
        thread: Thread to inspect
        context: Context for signals
        signals: List of signal names to use (None = all)

    Returns:
        FabricCheckResult
    """
    if signals:
        signal_objs = [
            SignalRegistry.get(name)
            for name in signals
            if SignalRegistry.is_registered(name)
        ]
        signal_objs = [s for s in signal_objs if s is not None]
    else:
        signal_objs = None

    inspector = FabricInspector(signals=signal_objs)
    return await inspector.inspect(thread, context)
