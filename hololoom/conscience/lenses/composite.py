"""
Composite Lens
==============
Combines multiple lenses into a single composable unit.

The beauty of composition - lenses naturally combine with |:
    combined = HarmLens() | DeceptionLens() | PowerLens()

Author: HoloLoom Team
Date: 2025-12-03
"""

from typing import Dict, Any, List, Optional, Union
import logging
import asyncio

from ..judgment import Concern, Voice, Judgment
from .base import BaseLens


logger = logging.getLogger("hololoom.conscience.lenses.composite")


class CompositeLens:
    """
    Combines multiple lenses into a unified examination.

    Concerns from all component lenses are collected and the
    highest concern level determines the overall voice.

    Usage:
        lens = HarmLens() | DeceptionLens() | PowerLens()
        concerns = await lens.examine(action, context)

    Voice Determination:
        - All QUIET → QUIET
        - Any concern with confidence >= 0.8 → SHOUT
        - Any concern with confidence >= 0.6 → VOICE
        - Any concern with confidence >= 0.3 → WHISPER
        - Otherwise → QUIET
    """

    def __init__(
        self,
        lenses: Optional[List[BaseLens]] = None,
        voice_thresholds: Optional[Dict[Voice, float]] = None,
    ):
        """
        Initialize a composite lens.

        Args:
            lenses: List of component lenses
            voice_thresholds: Custom thresholds for voice levels
        """
        self._lenses: List[BaseLens] = lenses or []

        # Default voice thresholds (can be customized)
        self._voice_thresholds = voice_thresholds or {
            Voice.SHOUT: 0.8,     # Critical concern
            Voice.VOICE: 0.6,     # Significant concern
            Voice.WHISPER: 0.3,   # Minor concern
            Voice.QUIET: 0.0,     # No concern
        }

    @property
    def name(self) -> str:
        """Combined name of all lenses."""
        if not self._lenses:
            return "EmptyComposite"
        return " | ".join(lens.name for lens in self._lenses)

    @property
    def category(self) -> str:
        """Combined categories."""
        if not self._lenses:
            return "none"
        return "+".join(lens.category for lens in self._lenses)

    @property
    def lenses(self) -> List[BaseLens]:
        """List of component lenses."""
        return self._lenses.copy()

    async def examine(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """
        Examine an action through all component lenses.

        Runs all lenses concurrently for efficiency.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            Combined list of concerns from all lenses
        """
        if not self._lenses:
            return []

        # Run all lenses concurrently
        tasks = [lens.examine(action, context) for lens in self._lenses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all concerns
        all_concerns: List[Concern] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Lens {self._lenses[i].name} failed: {result}"
                )
                continue
            all_concerns.extend(result)

        return all_concerns

    async def judge(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> Judgment:
        """
        Examine and produce a full Judgment.

        This is the primary method for getting a complete judgment
        including voice level, concerns, and guidance.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            Complete Judgment with voice level and concerns
        """
        concerns = await self.examine(action, context)

        # Determine voice level from concerns
        voice = self._determine_voice(concerns)

        # Generate summary and guidance
        summary = self._generate_summary(concerns, voice)
        guidance = self._generate_guidance(concerns, voice)

        return Judgment(
            voice=voice,
            concerns=concerns,
            summary=summary,
            guidance=guidance,
            metadata={
                "lenses": [lens.name for lens in self._lenses],
                "lens_count": len(self._lenses),
                "concern_count": len(concerns),
            },
        )

    def _determine_voice(self, concerns: List[Concern]) -> Voice:
        """
        Determine voice level from concerns.

        Uses the highest confidence concern to set the voice level.
        """
        if not concerns:
            return Voice.QUIET

        max_confidence = max(c.confidence for c in concerns)

        # Check thresholds from highest to lowest
        if max_confidence >= self._voice_thresholds[Voice.SHOUT]:
            return Voice.SHOUT
        elif max_confidence >= self._voice_thresholds[Voice.VOICE]:
            return Voice.VOICE
        elif max_confidence >= self._voice_thresholds[Voice.WHISPER]:
            return Voice.WHISPER
        else:
            return Voice.QUIET

    def _generate_summary(
        self,
        concerns: List[Concern],
        voice: Voice
    ) -> str:
        """Generate a human-readable summary."""
        if not concerns:
            return "No concerns raised"

        # Group by category
        by_category: Dict[str, int] = {}
        for concern in concerns:
            by_category[concern.category] = by_category.get(concern.category, 0) + 1

        parts = [f"{count} {cat}" for cat, count in sorted(by_category.items())]
        categories_str = ", ".join(parts)

        if voice == Voice.SHOUT:
            return f"Critical safety concerns: {categories_str}"
        elif voice == Voice.VOICE:
            return f"Significant concerns require review: {categories_str}"
        elif voice == Voice.WHISPER:
            return f"Minor concerns noted: {categories_str}"
        else:
            return f"Low-level concerns: {categories_str}"

    def _generate_guidance(
        self,
        concerns: List[Concern],
        voice: Voice
    ) -> str:
        """Generate guidance based on voice level."""
        if voice == Voice.QUIET:
            return "Proceed normally"
        elif voice == Voice.WHISPER:
            return "Proceed with monitoring"
        elif voice == Voice.VOICE:
            # Include top mitigation suggestions
            mitigations = [
                c.suggested_mitigation
                for c in concerns
                if c.suggested_mitigation
            ]
            if mitigations:
                return f"Request human review. Consider: {mitigations[0]}"
            return "Request human review before proceeding"
        else:  # SHOUT
            return "Action blocked for safety. Do not proceed."

    async def learn(
        self,
        action: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> None:
        """
        Learn from the outcome across all component lenses.

        Args:
            action: The action that was taken
            context: Context for the action
            outcome: What happened
        """
        # Learn in parallel
        tasks = [
            lens.learn(action, context, outcome)
            for lens in self._lenses
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    def __or__(self, other: Union[BaseLens, "CompositeLens"]) -> "CompositeLens":
        """
        Compose with another lens or composite.

        Examples:
            combined = lens1 | lens2 | lens3
            extended = composite | new_lens
        """
        new_lenses = self._lenses.copy()

        if isinstance(other, CompositeLens):
            new_lenses.extend(other._lenses)
        else:
            new_lenses.append(other)

        return CompositeLens(new_lenses, self._voice_thresholds.copy())

    def __len__(self) -> int:
        """Number of component lenses."""
        return len(self._lenses)

    def __iter__(self):
        """Iterate over component lenses."""
        return iter(self._lenses)

    def __repr__(self) -> str:
        lens_names = [lens.name for lens in self._lenses]
        return f"CompositeLens({' | '.join(lens_names)})"


def create_composite(
    *lenses: BaseLens,
    voice_thresholds: Optional[Dict[Voice, float]] = None
) -> CompositeLens:
    """
    Factory function to create a composite lens.

    Args:
        *lenses: Variable number of lenses to combine
        voice_thresholds: Optional custom voice thresholds

    Returns:
        CompositeLens combining all provided lenses

    Example:
        lens = create_composite(
            HarmLens(),
            DeceptionLens(),
            PowerLens(),
        )
    """
    return CompositeLens(list(lenses), voice_thresholds)
