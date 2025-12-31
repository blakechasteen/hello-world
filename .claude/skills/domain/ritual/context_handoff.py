"""
Ritual Context Handoff
======================

Created: December 30, 2025
Purpose: MI-aware context filtering between ritual phases

Mirrors HoloLoom/agentic/context_handoff.py pattern.
Ensures each phase receives:
- High-relevance context from previous phases
- Minimal redundancy
- Appropriate detail level for the target phase
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Handoff Strategies
# ============================================================================

class HandoffStrategy(Enum):
    """Strategies for context handoff between phases."""
    FULL = "full"               # Pass everything (no filtering)
    AGGRESSIVE = "aggressive"   # Keep only high-MI items (30%)
    BALANCED = "balanced"       # Keep moderate-MI items (50%)
    CONSERVATIVE = "conservative"  # Keep most items (70%)


# ============================================================================
# Context Data Structures
# ============================================================================

@dataclass
class ContextItem:
    """
    Single item in handoff context.

    Represents a piece of information from a previous phase
    that may be relevant to the next phase.
    """
    content: str
    source_phase: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5  # 0.0-1.0 base importance
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate importance range."""
        self.importance = max(0.0, min(1.0, self.importance))

    def token_estimate(self) -> int:
        """Estimate token count for this item (~4 chars per token)."""
        return len(self.content) // 4


@dataclass
class HandoffResult:
    """
    Result of context handoff filtering.

    Contains the selected items and metrics about the filtering process.
    """
    selected_items: List[ContextItem]
    original_count: int
    selected_count: int
    avg_mi: float
    redundancy_removed: float  # Percentage of redundant items removed
    strategy_used: HandoffStrategy

    def get_context_text(self) -> str:
        """Get filtered context as concatenated text."""
        return "\n\n".join([item.content for item in self.selected_items])

    def get_total_tokens(self) -> int:
        """Get total estimated tokens in filtered context."""
        return sum(item.token_estimate() for item in self.selected_items)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "selected_count": self.selected_count,
            "original_count": self.original_count,
            "avg_mi": self.avg_mi,
            "redundancy_removed": self.redundancy_removed,
            "strategy_used": self.strategy_used.value,
            "total_tokens": self.get_total_tokens(),
            "items": [
                {
                    "content": item.content[:100] + "..." if len(item.content) > 100 else item.content,
                    "source_phase": item.source_phase,
                    "importance": item.importance,
                }
                for item in self.selected_items
            ]
        }


# ============================================================================
# Phase-Specific Keywords for MI Calculation
# ============================================================================

PHASE_KEYWORDS = {
    "awaken": [
        "context", "previous", "yesterday", "last session", "working on",
        "recall", "remember", "history", "background", "before"
    ],
    "plan": [
        "requirements", "implement", "design", "architecture", "steps",
        "plan", "goal", "objective", "approach", "strategy", "todo"
    ],
    "implement": [
        "code", "function", "class", "method", "variable", "bug",
        "fix", "feature", "module", "import", "test", "error"
    ],
    "review": [
        "quality", "issue", "bug", "improve", "decision", "review",
        "check", "verify", "validate", "test", "coverage", "standard"
    ],
    "reflect": [
        "learned", "summary", "remember", "next session", "key",
        "insight", "pattern", "lesson", "takeaway", "note", "future"
    ],
}

# Capability-specific keywords for more targeted MI scoring
CAPABILITY_KEYWORDS = {
    "context_restoration": [
        "context", "restore", "previous", "session", "state", "history"
    ],
    "planning": [
        "plan", "design", "architecture", "approach", "strategy", "steps"
    ],
    "code_assistance": [
        "code", "implement", "function", "class", "bug", "feature"
    ],
    "quality_assurance": [
        "quality", "test", "review", "check", "validate", "standard"
    ],
    "knowledge_consolidation": [
        "learn", "summary", "insight", "pattern", "remember", "key"
    ],
}


# ============================================================================
# Context Handoff Implementation
# ============================================================================

class RitualContextHandoff:
    """
    MI-aware context filtering between ritual phases.

    Ensures each phase receives:
    - High-relevance context from previous phases
    - Minimal redundancy
    - Appropriate detail level for the target phase

    Uses Mutual Information (MI) scoring to determine relevance
    and Jaccard similarity to detect redundancy.

    Usage:
        handoff = RitualContextHandoff(strategy=HandoffStrategy.BALANCED)

        result = handoff.prepare_handoff(
            context_items=previous_phase_outputs,
            from_phase="awaken",
            to_phase="plan",
            target_capability="planning",
            token_budget=2000
        )

        context_text = result.get_context_text()
    """

    def __init__(self, strategy: HandoffStrategy = HandoffStrategy.BALANCED):
        """
        Initialize context handoff.

        Args:
            strategy: Default filtering strategy
        """
        self.strategy = strategy
        self._keep_ratios = {
            HandoffStrategy.FULL: 1.0,
            HandoffStrategy.AGGRESSIVE: 0.3,
            HandoffStrategy.BALANCED: 0.5,
            HandoffStrategy.CONSERVATIVE: 0.7,
        }

    def calculate_mi(
        self,
        item: ContextItem,
        target_phase: str,
        target_capability: str
    ) -> float:
        """
        Calculate mutual information between context item and target phase.

        I(Item; Target) = relevance to target phase/capability

        Uses keyword matching as a heuristic for MI.
        In production, could use embeddings for semantic similarity.

        Args:
            item: Context item to score
            target_phase: Target phase (awaken, plan, implement, review, reflect)
            target_capability: Target capability string

        Returns:
            MI score (0.0-1.0)
        """
        content_lower = item.content.lower()

        # Get phase-specific keywords
        phase_keywords = PHASE_KEYWORDS.get(target_phase, [])
        capability_keywords = CAPABILITY_KEYWORDS.get(target_capability, [])

        # Combine keywords with deduplication
        all_keywords = list(set(phase_keywords + capability_keywords))

        if not all_keywords:
            return 0.5  # Neutral score if no keywords defined

        # Count keyword matches
        matches = sum(1 for kw in all_keywords if kw in content_lower)
        keyword_score = min(matches / max(len(all_keywords), 1), 1.0)

        # Combine with item importance (60% keyword, 40% importance)
        mi = 0.6 * keyword_score + 0.4 * item.importance

        return mi

    def calculate_redundancy(
        self,
        item1: ContextItem,
        item2: ContextItem
    ) -> float:
        """
        Calculate redundancy between two context items.

        Uses Jaccard similarity on word sets.
        Higher score = more redundant.

        Args:
            item1: First context item
            item2: Second context item

        Returns:
            Redundancy score (0.0-1.0)
        """
        words1 = set(item1.content.lower().split())
        words2 = set(item2.content.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def prepare_handoff(
        self,
        context_items: List[ContextItem],
        from_phase: str,
        to_phase: str,
        target_capability: str,
        token_budget: Optional[int] = None
    ) -> HandoffResult:
        """
        Prepare context for handoff to target phase.

        1. Score items by MI with target
        2. Remove redundant items
        3. Keep top items based on strategy
        4. Respect token budget if provided

        Args:
            context_items: Items from previous phase(s)
            from_phase: Source phase
            to_phase: Target phase
            target_capability: Target phase capability
            token_budget: Optional max tokens (None = unlimited)

        Returns:
            HandoffResult with filtered items and metrics
        """
        if not context_items:
            return HandoffResult(
                selected_items=[],
                original_count=0,
                selected_count=0,
                avg_mi=0.0,
                redundancy_removed=0.0,
                strategy_used=self.strategy
            )

        # Calculate MI scores for all items
        scored_items: List[tuple[ContextItem, float]] = []
        for item in context_items:
            mi = self.calculate_mi(item, to_phase, target_capability)
            scored_items.append((item, mi))

        # Sort by MI (highest first)
        scored_items.sort(key=lambda x: -x[1])

        # Remove redundant items (keep first occurrence)
        filtered_items: List[tuple[ContextItem, float]] = []
        redundancy_threshold = 0.7  # Items with >70% similarity are redundant

        for item, mi in scored_items:
            is_redundant = False
            for existing, _ in filtered_items:
                if self.calculate_redundancy(item, existing) > redundancy_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_items.append((item, mi))

        redundancy_removed = 1.0 - len(filtered_items) / len(scored_items) if scored_items else 0.0

        # Apply strategy-based keep ratio
        keep_ratio = self._keep_ratios[self.strategy]
        keep_count = max(1, int(len(filtered_items) * keep_ratio))

        # Token budget constraint
        if token_budget is not None:
            cumulative_tokens = 0
            budget_limited: List[tuple[ContextItem, float]] = []

            for item, mi in filtered_items:
                item_tokens = item.token_estimate()
                if cumulative_tokens + item_tokens <= token_budget:
                    budget_limited.append((item, mi))
                    cumulative_tokens += item_tokens

            filtered_items = budget_limited
            keep_count = min(keep_count, len(filtered_items))

        # Select top items
        selected = filtered_items[:keep_count]

        # Calculate average MI
        avg_mi = sum(mi for _, mi in selected) / len(selected) if selected else 0.0

        logger.debug(
            f"Context handoff {from_phase} → {to_phase}: "
            f"{len(context_items)} → {len(selected)} items "
            f"(avg_mi={avg_mi:.2f}, redundancy_removed={redundancy_removed:.1%})"
        )

        return HandoffResult(
            selected_items=[item for item, _ in selected],
            original_count=len(context_items),
            selected_count=len(selected),
            avg_mi=avg_mi,
            redundancy_removed=redundancy_removed,
            strategy_used=self.strategy
        )

    def set_strategy(self, strategy: HandoffStrategy) -> None:
        """Update the handoff strategy."""
        self.strategy = strategy


# ============================================================================
# Factory Function
# ============================================================================

def create_context_handoff(
    strategy: HandoffStrategy = HandoffStrategy.BALANCED
) -> RitualContextHandoff:
    """
    Create a context handoff instance.

    Args:
        strategy: Filtering strategy (default: BALANCED)

    Returns:
        Configured RitualContextHandoff
    """
    return RitualContextHandoff(strategy=strategy)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'HandoffStrategy',

    # Data structures
    'ContextItem',
    'HandoffResult',

    # Main class
    'RitualContextHandoff',

    # Factory
    'create_context_handoff',

    # Constants
    'PHASE_KEYWORDS',
    'CAPABILITY_KEYWORDS',
]
