"""
Writing System Protocols
=========================

Protocol definitions for the HoloLoom writing system.
Defines interfaces for writers, composers, and refiners.
"""

from typing import Protocol, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import from hololoom types
try:
    from hololoom.protocols.types import MemoryShard
except ImportError:
    # Fallback for standalone usage
    @dataclass
    class MemoryShard:
        id: str
        text: str
        metadata: Dict[str, Any] = None


class WritingMode(Enum):
    """Writing modes for different content types."""
    NARRATIVE = "narrative"      # Story/explanation
    TECHNICAL = "technical"      # Documentation
    CREATIVE = "creative"        # Fiction/poetry
    ANALYSIS = "analysis"        # Report/analysis
    DIALOGUE = "dialogue"        # Conversational
    CODE_DOC = "code_doc"        # Code documentation
    AUTO = "auto"                # Auto-detect


class RefinementStrategy(Enum):
    """Refinement strategies for multi-pass improvement."""
    ELEGANCE = "elegance"        # Clarity → Simplicity → Beauty
    VERIFY = "verify"            # Accuracy → Completeness → Consistency
    TONE = "tone"                # Tone adjustment
    COHERENCE = "coherence"      # Coherence improvement
    NONE = "none"                # No refinement


class StyleGuide(Enum):
    """Style guides for content generation."""
    TUFTE = "tufte"              # Tufte-inspired clarity
    ACADEMIC = "academic"        # Academic writing
    CASUAL = "casual"            # Casual/conversational
    TECHNICAL = "technical"      # Technical documentation
    CREATIVE = "creative"        # Creative/expressive
    AUTO = "auto"                # Auto-detect


class OutputFormat(Enum):
    """Output format options."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"
    PLAIN = "plain"


@dataclass
class WritingContext:
    """Context for content generation."""
    query: str
    memories: List[MemoryShard]
    mode: WritingMode = WritingMode.AUTO
    style: StyleGuide = StyleGuide.AUTO
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RefinementPass:
    """Single refinement pass result."""
    pass_number: int
    strategy: str
    focus: str                    # What this pass focuses on
    input_text: str
    output_text: str
    quality_score: float          # 0.0-1.0
    improvements: List[str]       # List of improvements made
    duration_ms: float


@dataclass
class WritingResult:
    """Result of content generation."""
    content: str
    mode: WritingMode
    style: StyleGuide
    quality_score: float          # 0.0-1.0
    confidence: float             # 0.0-1.0
    refinement_passes: List[RefinementPass]
    metadata: Dict[str, Any]

    @property
    def quality_trajectory(self) -> List[float]:
        """Quality scores across refinement passes."""
        return [p.quality_score for p in self.refinement_passes]

    @property
    def total_passes(self) -> int:
        """Total number of refinement passes."""
        return len(self.refinement_passes)

    def summary(self) -> str:
        """Human-readable summary."""
        if not self.refinement_passes:
            return f"Generated {self.mode.value} content (quality: {self.quality_score:.2f})"

        initial = self.refinement_passes[0].quality_score
        final = self.refinement_passes[-1].quality_score
        improvement = final - initial

        return (
            f"Generated {self.mode.value} content with {self.total_passes} refinement passes\n"
            f"Quality: {initial:.2f} → {final:.2f} ({improvement:+.2f} improvement)\n"
            f"Style: {self.style.value}, Confidence: {self.confidence:.2f}"
        )


class WriterProtocol(Protocol):
    """
    Protocol for content writers.

    Writers transform memory context into prose.
    """

    async def write(
        self,
        context: WritingContext,
        refine: bool = True
    ) -> WritingResult:
        """
        Generate content from memory context.

        Args:
            context: Writing context with query and memories
            refine: Enable multi-pass refinement

        Returns:
            Writing result with content and metadata
        """
        ...

    def detect_mode(
        self,
        query: str,
        memories: List[MemoryShard]
    ) -> WritingMode:
        """
        Auto-detect appropriate writing mode.

        Args:
            query: User query
            memories: Memory context

        Returns:
            Detected writing mode
        """
        ...

    def detect_style(
        self,
        query: str,
        memories: List[MemoryShard],
        mode: WritingMode
    ) -> StyleGuide:
        """
        Auto-detect appropriate style guide.

        Args:
            query: User query
            memories: Memory context
            mode: Writing mode

        Returns:
            Detected style guide
        """
        ...


class ComposerProtocol(Protocol):
    """
    Protocol for multi-pass composers.

    Composers orchestrate refinement across multiple passes.
    """

    async def compose(
        self,
        initial_draft: str,
        context: WritingContext,
        strategy: RefinementStrategy = RefinementStrategy.ELEGANCE,
        max_passes: int = 3,
        quality_threshold: float = 0.9
    ) -> WritingResult:
        """
        Multi-pass composition with refinement.

        Args:
            initial_draft: Initial content draft
            context: Writing context
            strategy: Refinement strategy
            max_passes: Maximum refinement passes
            quality_threshold: Target quality score

        Returns:
            Writing result with refinement history
        """
        ...

    def score_quality(
        self,
        text: str,
        context: WritingContext
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score content quality.

        Args:
            text: Content to score
            context: Writing context

        Returns:
            (overall_quality, dimension_scores)
            dimension_scores: {clarity, completeness, coherence, etc.}
        """
        ...


class RefinerProtocol(Protocol):
    """
    Protocol for content refiners.

    Refiners improve content along specific dimensions.
    """

    async def refine(
        self,
        text: str,
        context: WritingContext,
        pass_number: int
    ) -> RefinementPass:
        """
        Refine content in single pass.

        Args:
            text: Content to refine
            context: Writing context
            pass_number: Current pass number

        Returns:
            Refinement pass result
        """
        ...

    @property
    def strategy_name(self) -> str:
        """Strategy name for this refiner."""
        ...

    @property
    def passes(self) -> List[str]:
        """List of focus areas for each pass."""
        ...


class ModeWriterProtocol(Protocol):
    """
    Protocol for mode-specific writers.

    Mode writers implement content generation for specific modes
    (narrative, technical, creative, etc.)
    """

    async def generate(
        self,
        context: WritingContext
    ) -> str:
        """
        Generate initial draft for this mode.

        Args:
            context: Writing context

        Returns:
            Initial draft content
        """
        ...

    @property
    def mode(self) -> WritingMode:
        """Writing mode this writer handles."""
        ...

    @property
    def default_style(self) -> StyleGuide:
        """Default style guide for this mode."""
        ...


# Quality scoring dimensions
QUALITY_DIMENSIONS = {
    'clarity': 'How clear and understandable is the content?',
    'completeness': 'How complete is the content?',
    'coherence': 'How coherent and well-structured?',
    'accuracy': 'How accurate is the information?',
    'conciseness': 'How concise without losing meaning?',
    'elegance': 'How graceful and beautiful?'
}
