"""
Loom Command - Pattern Card Selector
=====================================
The control system that selects which pattern card (execution template) to use.

Philosophy:
In traditional weaving, the loom "command" determines which warp threads to lift
and which pattern to follow. The Loom Command in HoloLoom selects the execution
template (Pattern Card) that dictates:
- Which scales to activate
- How much compute to use
- Which features to extract
- Threading density (BARE/FAST/FUSED)

The Pattern Card is the "DNA" of a weaving cycle - the blueprint that specifies
how all components should configure themselves.

Pattern Cards:
- BARE: Minimal threading (fastest, lowest quality)
- FAST: Balanced threading (good speed/quality)
- FUSED: Full threading (highest quality, slowest)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Pattern Cards
# ============================================================================

class PatternCard(Enum):
    """
    Pattern cards define execution templates.

    Each card specifies a complete configuration for the weaving process.
    """
    BARE = "bare"      # Minimal processing
    FAST = "fast"      # Balanced processing
    FUSED = "fused"    # Full processing


@dataclass
class PatternSpec:
    """
    Specification for a pattern card.

    Defines all parameters for executing a weaving cycle with this pattern.
    """
    # Pattern identity
    name: str
    card: PatternCard

    # Threading configuration
    scales: List[int]  # Embedding scales to activate
    fusion_weights: Dict[int, float]  # Weights for multi-scale fusion

    # Feature extraction
    enable_motifs: bool = True
    motif_mode: str = "regex"  # "regex" | "hybrid" | "nlp"
    enable_spectral: bool = False
    spectral_k_eigen: int = 4

    # Memory retrieval
    retrieval_mode: str = "fast"  # "fast" | "fused"
    retrieval_k: int = 6
    enable_bm25: bool = False
    bm25_weight: float = 0.15

    # Decision making
    n_transformer_layers: int = 1
    n_attention_heads: int = 2
    policy_complexity: str = "simple"  # "simple" | "full"

    # Temporal control
    pipeline_timeout: float = 5.0
    stage_timeouts: Dict[str, float] = None

    # Quality vs speed
    quality_target: float = 0.7  # 0-1 scale
    speed_priority: float = 0.5  # 0-1 scale (higher = faster)

    def __post_init__(self):
        """Set default stage timeouts if not provided."""
        if self.stage_timeouts is None:
            # Derive from pipeline timeout
            self.stage_timeouts = {
                'features': self.pipeline_timeout * 0.3,
                'retrieval': self.pipeline_timeout * 0.4,
                'decision': self.pipeline_timeout * 0.2,
                'execution': self.pipeline_timeout * 0.1
            }


# ============================================================================
# Pattern Card Templates
# ============================================================================

# BARE Pattern Card - Minimal processing
BARE_PATTERN = PatternSpec(
    name="Bare Threading",
    card=PatternCard.BARE,
    scales=[96],
    fusion_weights={96: 1.0},
    enable_motifs=True,
    motif_mode="regex",
    enable_spectral=False,
    retrieval_mode="fast",
    retrieval_k=3,
    enable_bm25=False,
    n_transformer_layers=1,
    n_attention_heads=2,
    policy_complexity="simple",
    pipeline_timeout=2.0,
    quality_target=0.6,
    speed_priority=0.9
)

# FAST Pattern Card - Balanced processing
FAST_PATTERN = PatternSpec(
    name="Fast Threading",
    card=PatternCard.FAST,
    scales=[96, 192],
    fusion_weights={96: 0.4, 192: 0.6},
    enable_motifs=True,
    motif_mode="hybrid",
    enable_spectral=True,
    spectral_k_eigen=2,
    retrieval_mode="fast",
    retrieval_k=6,
    enable_bm25=True,
    bm25_weight=0.1,
    n_transformer_layers=2,
    n_attention_heads=4,
    policy_complexity="full",
    pipeline_timeout=4.0,
    quality_target=0.75,
    speed_priority=0.5
)

# FUSED Pattern Card - Full processing
FUSED_PATTERN = PatternSpec(
    name="Fused Threading",
    card=PatternCard.FUSED,
    scales=[96, 192, 384],
    fusion_weights={96: 0.25, 192: 0.35, 384: 0.40},
    enable_motifs=True,
    motif_mode="hybrid",
    enable_spectral=True,
    spectral_k_eigen=4,
    retrieval_mode="fused",
    retrieval_k=10,
    enable_bm25=True,
    bm25_weight=0.15,
    n_transformer_layers=2,
    n_attention_heads=4,
    policy_complexity="full",
    pipeline_timeout=8.0,
    quality_target=0.9,
    speed_priority=0.2
)


# ============================================================================
# Loom Command
# ============================================================================

class LoomCommand:
    """
    Loom Command - Selects and manages pattern cards.

    The Loom Command determines which pattern card to use based on:
    - User preferences (if explicitly specified)
    - Query complexity (automatic selection)
    - Resource availability
    - Performance requirements

    It maintains the current pattern and can switch patterns dynamically.

    Usage:
        loom = LoomCommand(default_pattern=PatternCard.FAST)
        pattern_spec = loom.select_pattern(query_text, user_preference="fused")
    """

    def __init__(
        self,
        default_pattern: PatternCard = PatternCard.FAST,
        auto_select: bool = True
    ):
        """
        Initialize Loom Command.

        Args:
            default_pattern: Default pattern card if no preference
            auto_select: Enable automatic pattern selection based on query
        """
        self.default_pattern = default_pattern
        self.auto_select = auto_select

        # Pattern card registry
        self.patterns = {
            PatternCard.BARE: BARE_PATTERN,
            PatternCard.FAST: FAST_PATTERN,
            PatternCard.FUSED: FUSED_PATTERN
        }

        # Current pattern
        self.current_pattern: Optional[PatternSpec] = None

        # Selection history
        self.selection_history: List[Dict[str, Any]] = []

        logger.info(f"LoomCommand initialized (default={default_pattern.value}, auto_select={auto_select})")

    def select_pattern(
        self,
        query_text: Optional[str] = None,
        user_preference: Optional[str] = None,
        resource_constraints: Optional[Dict[str, Any]] = None
    ) -> PatternSpec:
        """
        Select pattern card for weaving.

        Selection priority:
        1. User explicit preference
        2. Resource constraints
        3. Automatic query-based selection (if enabled)
        4. Default pattern

        Args:
            query_text: Optional query text for auto-selection
            user_preference: Optional explicit pattern ("bare", "fast", "fused")
            resource_constraints: Optional constraints (e.g., {"max_timeout": 3.0})

        Returns:
            Selected PatternSpec
        """
        selected_card = None
        selection_reason = None

        # Priority 1: User preference
        if user_preference:
            try:
                selected_card = PatternCard(user_preference.lower())
                selection_reason = f"user_preference={user_preference}"
            except ValueError:
                logger.warning(f"Invalid user preference: {user_preference}, using default")

        # Priority 2: Resource constraints
        if not selected_card and resource_constraints:
            selected_card = self._select_by_constraints(resource_constraints)
            if selected_card:
                selection_reason = f"resource_constraints={resource_constraints}"

        # Priority 3: Automatic selection
        if not selected_card and self.auto_select and query_text:
            selected_card = self._auto_select(query_text)
            selection_reason = f"auto_select_from_query (len={len(query_text)})"

        # Priority 4: Default
        if not selected_card:
            selected_card = self.default_pattern
            selection_reason = "default_pattern"

        # Get pattern spec
        pattern_spec = self.patterns[selected_card]
        self.current_pattern = pattern_spec

        # Record selection
        self.selection_history.append({
            "pattern": selected_card.value,
            "reason": selection_reason,
            "query_length": len(query_text) if query_text else 0
        })

        logger.info(f"Selected pattern: {selected_card.value} ({selection_reason})")

        return pattern_spec

    def _select_by_constraints(self, constraints: Dict[str, Any]) -> Optional[PatternCard]:
        """
        Select pattern based on resource constraints.

        Args:
            constraints: Dict with constraints (e.g., max_timeout, max_memory)

        Returns:
            Selected PatternCard or None
        """
        max_timeout = constraints.get("max_timeout")

        if max_timeout:
            # Select pattern that fits within timeout
            if max_timeout <= 2.5:
                return PatternCard.BARE
            elif max_timeout <= 5.0:
                return PatternCard.FAST
            else:
                return PatternCard.FUSED

        return None

    def _auto_select(self, query_text: str) -> PatternCard:
        """
        Automatically select pattern based on query characteristics.

        Simple heuristic:
        - Short queries (<50 chars) → BARE or FAST
        - Medium queries (50-150 chars) → FAST
        - Long queries (>150 chars) → FAST or FUSED

        Args:
            query_text: Query text

        Returns:
            Selected PatternCard
        """
        query_len = len(query_text)

        if query_len < 50:
            # Short query: fast response
            return PatternCard.BARE

        elif query_len < 150:
            # Medium query: balanced
            return PatternCard.FAST

        else:
            # Long query: quality matters
            return PatternCard.FAST  # Could be FUSED if compute available

    def get_current_pattern(self) -> Optional[PatternSpec]:
        """Get currently active pattern spec."""
        return self.current_pattern

    def set_default(self, pattern: PatternCard) -> None:
        """
        Change default pattern.

        Args:
            pattern: New default pattern card
        """
        self.default_pattern = pattern
        logger.info(f"Default pattern changed to: {pattern.value}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get selection statistics.

        Returns:
            Dict with usage statistics
        """
        if not self.selection_history:
            return {"total_selections": 0}

        # Count by pattern
        pattern_counts = {}
        for selection in self.selection_history:
            pattern = selection["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Average query length by pattern
        pattern_query_lens = {}
        for selection in self.selection_history:
            pattern = selection["pattern"]
            if pattern not in pattern_query_lens:
                pattern_query_lens[pattern] = []
            pattern_query_lens[pattern].append(selection["query_length"])

        avg_query_lens = {
            pattern: sum(lens) / len(lens) if lens else 0
            for pattern, lens in pattern_query_lens.items()
        }

        return {
            "total_selections": len(self.selection_history),
            "pattern_distribution": pattern_counts,
            "avg_query_length_by_pattern": avg_query_lens,
            "current_pattern": self.current_pattern.name if self.current_pattern else None
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_loom_command(
    default: str = "fast",
    auto_select: bool = True
) -> LoomCommand:
    """
    Create Loom Command with default pattern.

    Args:
        default: Default pattern ("bare", "fast", "fused")
        auto_select: Enable auto-selection

    Returns:
        Configured LoomCommand
    """
    pattern = PatternCard(default.lower())
    return LoomCommand(default_pattern=pattern, auto_select=auto_select)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Loom Command Demo")
    print("="*80 + "\n")

    # Create loom command
    loom = LoomCommand(default_pattern=PatternCard.FAST, auto_select=True)

    # Test different selection modes
    test_cases = [
        ("Short query", None, None),
        ("This is a medium-length query that should trigger fast mode processing with balanced quality", None, None),
        ("Simple", "bare", None),
        ("Complex analysis query", "fused", None),
        ("Constrained query", None, {"max_timeout": 2.0})
    ]

    print("Pattern Selection Tests:\n")

    for query, preference, constraints in test_cases:
        pattern = loom.select_pattern(
            query_text=query,
            user_preference=preference,
            resource_constraints=constraints
        )

        print(f"Query: '{query[:50]}...'")
        print(f"  Preference: {preference or 'auto'}")
        print(f"  Constraints: {constraints or 'none'}")
        print(f"  → Selected: {pattern.card.value}")
        print(f"    Scales: {pattern.scales}")
        print(f"    Quality target: {pattern.quality_target:.1f}")
        print(f"    Speed priority: {pattern.speed_priority:.1f}")
        print(f"    Timeout: {pattern.pipeline_timeout:.1f}s")
        print()

    # Show statistics
    print("="*80)
    print("Selection Statistics:")
    print("="*80)
    stats = loom.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Demo complete!")
