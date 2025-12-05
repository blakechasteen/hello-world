"""
Deception Lens
==============
Detects potential deception, manipulation, and goal misalignment.

Wraps and extends the DeceptionDetection system with the elegant
ConscienceLens interface.

Categories of deception detected:
- Goal misalignment (actions that don't match stated goals)
- Hidden agendas (undisclosed motivations)
- Manipulation (attempts to influence through deception)
- Sycophancy (excessive agreement or flattery)
- Reward hacking (gaming metrics without true value)

Based on research from:
- Hubinger et al. (2021): "Risks from Learned Optimization"
- Anthropic: Constitutional AI and deceptive alignment

Author: HoloLoom Team
Date: 2025-12-03
"""

import re
import logging
from typing import Dict, Any, List, Optional

from ..judgment import Concern
from .base import BaseLens


logger = logging.getLogger("HoloLoom.conscience.lenses.deception")


# =============================================================================
# Deception Detection Patterns
# =============================================================================

# Patterns indicating potential manipulation
MANIPULATION_PATTERNS = [
    r"\b(don't tell|keep secret|hide this from|between us)\b",
    r"\b(pretend|act like|appear to be|fake)\b.*\b(you|i)\b",
    r"\b(trust me|believe me|i promise)\b.*\b(just|only)\b",
    r"\b(no one needs to know|confidential|off the record)\b",
]

# Patterns indicating potential sycophancy
SYCOPHANCY_PATTERNS = [
    r"\b(you're (absolutely|completely|totally) right)\b",
    r"\b(brilliant|genius|perfect|excellent) (idea|point|observation)\b",
    r"\b(couldn't agree more|exactly what i was thinking)\b",
    r"\b(you're so (smart|clever|insightful))\b",
]

# Patterns indicating goal misalignment
MISALIGNMENT_PATTERNS = [
    r"\b(my (real|true|actual) (goal|purpose|intention))\b",
    r"\b(what i really want|my hidden)\b",
    r"\b(pretending to|appearing to|seeming to)\b.*\b(but|while|when)\b",
    r"\b(they won't notice|they'll never know)\b",
]

# Patterns indicating reward hacking
REWARD_HACKING_PATTERNS = [
    r"\b(game the (system|metric|score))\b",
    r"\b(exploit|loophole|workaround)\b.*\b(rule|requirement|constraint)\b",
    r"\b(technically correct|letter of the law)\b",
    r"\b(optimize for|maximize|inflate)\b.*\b(metric|score|number)\b",
]

# Compile patterns
COMPILED_PATTERNS = {
    "manipulation": [re.compile(p, re.IGNORECASE) for p in MANIPULATION_PATTERNS],
    "sycophancy": [re.compile(p, re.IGNORECASE) for p in SYCOPHANCY_PATTERNS],
    "misalignment": [re.compile(p, re.IGNORECASE) for p in MISALIGNMENT_PATTERNS],
    "reward_hacking": [re.compile(p, re.IGNORECASE) for p in REWARD_HACKING_PATTERNS],
}


class DeceptionLens(BaseLens):
    """
    Detects potential deception in actions.

    Examines actions for patterns that could indicate:
    - Manipulation attempts
    - Sycophantic behavior
    - Goal misalignment
    - Reward hacking

    Usage:
        lens = DeceptionLens()
        concerns = await lens.examine(action, context)

    Configuration:
        threshold: Minimum confidence to raise concern (default: 0.5)
        enabled: Whether lens is active (default: True)
        categories: Which deception categories to detect (default: all)

    Integration:
        Can wrap existing DeceptionDetection for backward compatibility.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enabled: bool = True,
        categories: Optional[List[str]] = None,
        deception_detector: Optional[Any] = None,
    ):
        """
        Initialize DeceptionLens.

        Args:
            threshold: Minimum confidence to raise concern
            enabled: Whether lens is active
            categories: Which deception categories to detect (default: all)
            deception_detector: Optional DeceptionDetector instance to wrap
        """
        super().__init__(threshold=threshold, enabled=enabled)

        self.categories = categories or [
            "manipulation", "sycophancy", "misalignment", "reward_hacking"
        ]
        self._deception_detector = deception_detector

        # Track goal-action consistency
        self._stated_goals: Dict[str, str] = {}
        self._action_history: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "DeceptionLens"

    @property
    def category(self) -> str:
        return "deception"

    async def _examine_impl(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """
        Examine action for potential deception.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            List of deception concerns
        """
        concerns: List[Concern] = []

        # Check each enabled category
        for cat in self.categories:
            if cat in COMPILED_PATTERNS:
                cat_concerns = self._check_category(action, cat, context)
                concerns.extend(cat_concerns)

        # Check goal-action consistency
        consistency_concerns = self._check_goal_consistency(action, context)
        concerns.extend(consistency_concerns)

        # If we have a wrapped DeceptionDetector, use it too
        if self._deception_detector:
            detector_concerns = await self._check_deception_detector(
                action, context
            )
            concerns.extend(detector_concerns)

        # Record action for future consistency checks
        self._record_action(action, context)

        return concerns

    def _check_category(
        self,
        action: str,
        category: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """Check for deception patterns in a specific category."""
        concerns = []

        for pattern in COMPILED_PATTERNS.get(category, []):
            match = pattern.search(action)
            if match:
                # Calculate confidence
                confidence = 0.6  # Base confidence for pattern match

                # Boost for certain categories
                if category == "misalignment":
                    confidence = 0.75  # Goal misalignment is serious
                elif category == "manipulation":
                    confidence = 0.7

                concerns.append(
                    Concern(
                        lens=self.name,
                        category=f"deception:{category}",
                        description=f"Detected potential {category}: {match.group()[:50]}",
                        confidence=confidence,
                        evidence={
                            "pattern": pattern.pattern,
                            "match": match.group(),
                            "position": match.span(),
                        },
                        suggested_mitigation=self._get_mitigation(category),
                    )
                )

        return concerns

    def _check_goal_consistency(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """
        Check if action is consistent with stated goals.

        Looks for discrepancies between what was said and what is being done.
        """
        concerns = []

        session_id = context.get("session_id", "default")
        stated_goal = context.get("stated_goal") or self._stated_goals.get(session_id)

        if not stated_goal:
            return concerns

        # Simple consistency check using keyword overlap
        goal_words = set(stated_goal.lower().split())
        action_words = set(action.lower().split())

        overlap = goal_words & action_words
        goal_coverage = len(overlap) / max(len(goal_words), 1)

        # Low overlap might indicate misalignment
        if goal_coverage < 0.2 and len(goal_words) > 3:
            concerns.append(
                Concern(
                    lens=self.name,
                    category="deception:goal_mismatch",
                    description=f"Action may not align with stated goal: '{stated_goal[:50]}'",
                    confidence=0.5 - goal_coverage,  # Lower overlap = higher concern
                    evidence={
                        "stated_goal": stated_goal,
                        "goal_coverage": goal_coverage,
                        "overlapping_words": list(overlap),
                    },
                    suggested_mitigation="Verify action aligns with user's intended goal",
                )
            )

        return concerns

    async def _check_deception_detector(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """
        Check action against wrapped DeceptionDetector.

        Converts DeceptionDetector results to Concerns.
        """
        if not self._deception_detector:
            return []

        concerns = []

        try:
            # Import here to avoid circular dependency
            from HoloLoom.alignment.deception_detection import (
                DeceptionSignal,
            )

            # Use the detector's analysis
            if hasattr(self._deception_detector, "analyze_action"):
                analysis = self._deception_detector.analyze_action(
                    action, context
                )

                signal_to_confidence = {
                    DeceptionSignal.LOW: 0.3,
                    DeceptionSignal.MEDIUM: 0.5,
                    DeceptionSignal.HIGH: 0.7,
                    DeceptionSignal.CRITICAL: 0.9,
                }

                if analysis.signal != DeceptionSignal.LOW:
                    concerns.append(
                        Concern(
                            lens=self.name,
                            category="deception:behavioral",
                            description=f"Behavioral analysis detected deception signal: {analysis.signal.value}",
                            confidence=signal_to_confidence.get(
                                analysis.signal, 0.5
                            ),
                            evidence={
                                "signal": analysis.signal.value,
                                "analysis": analysis.to_dict() if hasattr(analysis, "to_dict") else str(analysis),
                            },
                            suggested_mitigation="Review for potential deceptive patterns",
                        )
                    )

        except ImportError:
            logger.debug("DeceptionDetection not available, skipping")
        except Exception as e:
            logger.warning(f"Error checking DeceptionDetector: {e}")

        return concerns

    def _record_action(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> None:
        """Record action for future consistency analysis."""
        self._action_history.append({
            "action": action,
            "context": context,
        })

        # Keep history bounded
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-100:]

        # Record stated goals
        if "stated_goal" in context:
            session_id = context.get("session_id", "default")
            self._stated_goals[session_id] = context["stated_goal"]

    def _get_mitigation(self, category: str) -> str:
        """Get mitigation suggestion for a deception category."""
        mitigations = {
            "manipulation": "Be transparent about intentions and limitations",
            "sycophancy": "Provide balanced, honest feedback even if critical",
            "misalignment": "Clarify actual goals and ensure actions match",
            "reward_hacking": "Focus on genuine value, not metric optimization",
        }
        return mitigations.get(category, "Request human review")

    def set_stated_goal(self, session_id: str, goal: str) -> None:
        """
        Set the stated goal for a session.

        Used for goal-action consistency checking.
        """
        self._stated_goals[session_id] = goal

    @classmethod
    def wrap_detector(cls, detector: Any) -> "DeceptionLens":
        """
        Factory method to wrap existing DeceptionDetector.

        Args:
            detector: DeceptionDetector instance

        Returns:
            DeceptionLens wrapping the detector
        """
        return cls(deception_detector=detector)
