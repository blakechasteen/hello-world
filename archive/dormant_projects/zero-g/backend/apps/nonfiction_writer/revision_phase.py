"""
Revision Phase for Nonfiction Writing

Multi-pass refinement using HoloLoom's recursive learning system (ELEGANCE strategy).
Improves draft quality through iterative refinement cycles.

Revision Strategies:
- ELEGANCE: Clarity → Simplicity → Beauty (3 passes)
- VERIFY: Accuracy → Completeness → Consistency (3 passes)
- CRITIQUE: Self-improvement loops
- CUSTOM: User-defined revision criteria

Integrates with:
- AdvancedRefiner from HoloLoom/recursive/
- Quality trajectory tracking
- Automatic strategy selection based on draft analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Would integrate with actual HoloLoom in production
try:
    from HoloLoom.recursive import AdvancedRefiner, RefinementStrategy
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.documentation.types import Query
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    # Mock RefinementStrategy for standalone
    class RefinementStrategy(Enum):
        ELEGANCE = "elegance"
        VERIFY = "verify"
        CRITIQUE = "critique"


from .draft_phase import Draft, Paragraph


@dataclass
class RevisionPass:
    """
    A single revision pass

    Attributes:
        pass_number: Pass number (1, 2, 3, ...)
        strategy: Revision strategy used
        focus: What this pass focused on
        changes_made: Number of changes
        quality_before: Quality score before (0.0-1.0)
        quality_after: Quality score after (0.0-1.0)
        duration_ms: Time taken in milliseconds
    """
    pass_number: int
    strategy: str
    focus: str
    changes_made: int
    quality_before: float
    quality_after: float
    duration_ms: float = 0.0

    @property
    def improvement(self) -> float:
        """Calculate quality improvement"""
        return self.quality_after - self.quality_before


@dataclass
class RevisionResult:
    """
    Result of revision process

    Attributes:
        original_draft: Original draft
        revised_draft: Revised draft
        passes: All revision passes performed
        total_changes: Total changes across all passes
        final_quality: Final quality score
        improvement: Total quality improvement
    """
    original_draft: Draft
    revised_draft: Draft
    passes: List[RevisionPass] = field(default_factory=list)
    total_changes: int = 0
    final_quality: float = 0.0
    improvement: float = 0.0

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Revision Summary:")
        lines.append(f"  Passes: {len(self.passes)}")
        lines.append(f"  Total changes: {self.total_changes}")
        lines.append(f"  Quality improvement: {self.improvement:.2%}")
        lines.append(f"\nPass Details:")

        for p in self.passes:
            lines.append(f"  Pass {p.pass_number} ({p.strategy}): {p.focus}")
            lines.append(f"    Changes: {p.changes_made}, Quality: {p.quality_before:.2f} → {p.quality_after:.2f}")

        return "\n".join(lines)


class RevisionEngine:
    """
    Multi-pass draft revision engine

    Uses HoloLoom's recursive refinement system to iteratively improve
    draft quality through multiple passes with different strategies.

    Usage:
        engine = RevisionEngine()

        # Auto-select strategy
        result = await engine.revise(draft, max_iterations=3)

        # Specific strategy
        result = await engine.revise(
            draft,
            strategy=RefinementStrategy.ELEGANCE,
            quality_threshold=0.9
        )

        # Custom focus
        result = await engine.revise_with_focus(
            draft,
            focus="Make it more accessible to non-experts"
        )
    """

    def __init__(self):
        """Initialize revision engine"""
        self.refiner = None
        if HOLOLOOM_AVAILABLE:
            # Would initialize AdvancedRefiner with orchestrator in production
            pass

    async def revise(self,
                    draft: Draft,
                    strategy: Optional[RefinementStrategy] = None,
                    max_iterations: int = 3,
                    quality_threshold: float = 0.9) -> RevisionResult:
        """
        Revise draft with specified strategy

        Args:
            draft: Draft to revise
            strategy: Refinement strategy (None = auto-select)
            max_iterations: Maximum revision passes
            quality_threshold: Target quality score

        Returns:
            RevisionResult with all passes and improvements
        """
        if HOLOLOOM_AVAILABLE and self.refiner:
            return await self._revise_with_refiner(
                draft, strategy, max_iterations, quality_threshold
            )
        else:
            return await self._revise_simple(
                draft, strategy, max_iterations, quality_threshold
            )

    async def revise_with_focus(self,
                                draft: Draft,
                                focus: str,
                                max_iterations: int = 3) -> RevisionResult:
        """
        Revise draft with custom focus

        Args:
            draft: Draft to revise
            focus: Custom revision focus (e.g., "Make it more concise")
            max_iterations: Maximum passes

        Returns:
            RevisionResult
        """
        # Would use custom prompting with refiner in full integration
        return await self.revise(draft, max_iterations=max_iterations)

    async def _revise_with_refiner(self,
                                   draft: Draft,
                                   strategy: Optional[RefinementStrategy],
                                   max_iterations: int,
                                   quality_threshold: float) -> RevisionResult:
        """Revise using AdvancedRefiner"""
        # Would use actual refiner in full integration
        # result = await self.refiner.refine(
        #     query=Query(text=draft.get_full_text()),
        #     initial_spacetime=...,
        #     strategy=strategy,
        #     max_iterations=max_iterations,
        #     quality_threshold=quality_threshold
        # )

        # For MVP, fall back to simple
        return await self._revise_simple(draft, strategy, max_iterations, quality_threshold)

    async def _revise_simple(self,
                            draft: Draft,
                            strategy: Optional[RefinementStrategy],
                            max_iterations: int,
                            quality_threshold: float) -> RevisionResult:
        """Simple rule-based revision"""
        revised_draft = draft
        passes = []

        # Define revision focuses based on strategy
        if strategy == RefinementStrategy.ELEGANCE:
            focuses = ["Clarity", "Simplicity", "Beauty"]
        elif strategy == RefinementStrategy.VERIFY:
            focuses = ["Accuracy", "Completeness", "Consistency"]
        else:
            focuses = ["General improvement"] * max_iterations

        for i in range(min(max_iterations, len(focuses))):
            quality_before = self._calculate_quality(revised_draft)

            # Apply revision
            revised_draft, changes = await self._apply_revision_pass(
                revised_draft,
                focuses[i]
            )

            quality_after = self._calculate_quality(revised_draft)

            revision_pass = RevisionPass(
                pass_number=i + 1,
                strategy=strategy.value if strategy else "auto",
                focus=focuses[i],
                changes_made=changes,
                quality_before=quality_before,
                quality_after=quality_after,
            )

            passes.append(revision_pass)

            # Check if threshold reached
            if quality_after >= quality_threshold:
                break

        result = RevisionResult(
            original_draft=draft,
            revised_draft=revised_draft,
            passes=passes,
            total_changes=sum(p.changes_made for p in passes),
            final_quality=self._calculate_quality(revised_draft),
            improvement=self._calculate_quality(revised_draft) - self._calculate_quality(draft),
        )

        return result

    async def _apply_revision_pass(self,
                                   draft: Draft,
                                   focus: str) -> tuple[Draft, int]:
        """Apply a single revision pass"""
        changes = 0

        # Revise each paragraph
        for para in draft.paragraphs:
            # Apply focus-specific improvements
            if focus == "Clarity":
                para.content, para_changes = self._improve_clarity(para.content)
            elif focus == "Simplicity":
                para.content, para_changes = self._improve_simplicity(para.content)
            elif focus == "Beauty":
                para.content, para_changes = self._improve_beauty(para.content)
            elif focus == "Accuracy":
                para.content, para_changes = self._improve_accuracy(para.content)
            elif focus == "Completeness":
                para.content, para_changes = self._improve_completeness(para.content)
            elif focus == "Consistency":
                para.content, para_changes = self._improve_consistency(para.content)
            else:
                para_changes = 0

            changes += para_changes

            # Recalculate word count
            para.word_count = len(para.content.split())

        # Update draft word count
        draft.word_count = sum(p.word_count for p in draft.paragraphs)

        return draft, changes

    def _calculate_quality(self, draft: Draft) -> float:
        """
        Calculate draft quality score

        Factors:
        - Average paragraph length (150-200 words ideal)
        - Citation coverage
        - Paragraph confidence
        - Word count vs target
        """
        if not draft.paragraphs:
            return 0.0

        # Length score (penalize too short or too long)
        avg_para_length = draft.word_count / len(draft.paragraphs)
        if 150 <= avg_para_length <= 200:
            length_score = 1.0
        elif avg_para_length < 100:
            length_score = avg_para_length / 100
        else:
            length_score = max(0.5, 200 / avg_para_length)

        # Citation score
        citation_score = min(1.0, draft.get_citation_count() / max(1, len(draft.paragraphs)))

        # Confidence score
        avg_confidence = sum(p.confidence for p in draft.paragraphs) / len(draft.paragraphs)

        # Source coverage score
        coverage_score = draft.get_source_coverage()

        # Weighted average
        quality = (
            0.3 * length_score +
            0.2 * citation_score +
            0.3 * avg_confidence +
            0.2 * coverage_score
        )

        return quality

    # ========================================================================
    # Improvement strategies
    # ========================================================================

    def _improve_clarity(self, text: str) -> tuple[str, int]:
        """Improve clarity of text"""
        changes = 0
        improved = text

        # Replace passive voice indicators
        passive_patterns = [
            ('is being', 'is'),
            ('are being', 'are'),
            ('was being', 'was'),
            ('were being', 'were'),
        ]

        for old, new in passive_patterns:
            if old in improved:
                improved = improved.replace(old, new)
                changes += 1

        # Remove hedging language
        hedges = ['very', 'quite', 'rather', 'somewhat', 'fairly']
        for hedge in hedges:
            if f' {hedge} ' in improved:
                improved = improved.replace(f' {hedge} ', ' ')
                changes += 1

        return improved, changes

    def _improve_simplicity(self, text: str) -> tuple[str, int]:
        """Simplify language"""
        changes = 0
        improved = text

        # Replace complex words with simpler alternatives
        simplifications = {
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'do',
            'demonstrate': 'show',
            'additional': 'more',
            'sufficient': 'enough',
        }

        for complex_word, simple in simplifications.items():
            if complex_word in improved.lower():
                improved = improved.replace(complex_word, simple)
                changes += 1

        return improved, changes

    def _improve_beauty(self, text: str) -> tuple[str, int]:
        """Improve aesthetic flow"""
        changes = 0
        improved = text

        # Add transition words if missing
        sentences = improved.split('.')
        if len(sentences) > 2:
            # Check if second sentence starts with transition
            second = sentences[1].strip()
            transitions = ['However', 'Moreover', 'Furthermore', 'Additionally', 'Nevertheless']

            if not any(second.startswith(t) for t in transitions):
                improved = sentences[0] + '. Moreover, ' + second + '.' + '.'.join(sentences[2:])
                changes += 1

        return improved, changes

    def _improve_accuracy(self, text: str) -> tuple[str, int]:
        """Improve factual accuracy"""
        # Would verify against sources in full integration
        return text, 0

    def _improve_completeness(self, text: str) -> tuple[str, int]:
        """Ensure completeness"""
        # Would check for missing information in full integration
        return text, 0

    def _improve_consistency(self, text: str) -> tuple[str, int]:
        """Ensure consistency"""
        # Would check for contradictions in full integration
        return text, 0


# Factory function
def create_revision_engine() -> RevisionEngine:
    """Create a new revision engine"""
    return RevisionEngine()
