"""
Context Handoff - MI-Based Multi-Agent Context Passing
======================================================

Phase 6.3: Uses mutual information for intelligent agent-to-agent context
passing, minimizing redundant information and maximizing relevance.

Key Features:
- Score context by MI with target agent's capability
- Detect and remove redundant information
- Budget-aware context selection
- Integration with Phase 6.1 AdaptiveCompressionStrategy

Author: Claude Code
Date: 2025-12-09
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import re
import math
from collections import Counter


# ============================================================================
# Data Types
# ============================================================================

class HandoffStrategy(Enum):
    """Strategy for context handoff."""
    AGGRESSIVE = "aggressive"    # Keep minimal high-MI context (30%)
    BALANCED = "balanced"        # Keep moderate context (50%)
    CONSERVATIVE = "conservative"  # Keep most context (70%)
    FULL = "full"                # Keep all context (100%)


@dataclass
class ContextItem:
    """Single item of context for handoff."""
    content: str
    source_agent: str
    mi_score: float = 0.0
    relevance_score: float = 0.0
    redundancy_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffContext:
    """Result of context handoff preparation."""
    selected_items: List[ContextItem]
    dropped_items: List[ContextItem]

    # Metrics
    original_count: int
    selected_count: int
    redundant_count: int

    # Scores
    total_mi: float
    avg_mi: float
    redundancy_removed: float  # Percentage of redundancy removed

    # Strategy used
    strategy: HandoffStrategy
    from_agent: str
    to_agent: str
    target_capability: str

    # Budget
    token_budget: Optional[int] = None
    estimated_tokens: int = 0
    tokens_saved: int = 0

    def __str__(self) -> str:
        return (
            f"HandoffContext({self.from_agent} -> {self.to_agent}, "
            f"items: {self.selected_count}/{self.original_count}, "
            f"redundancy removed: {self.redundancy_removed:.1%}, "
            f"avg MI: {self.avg_mi:.3f})"
        )

    def get_context_text(self) -> str:
        """Get combined text of all selected context items."""
        return "\n\n".join(item.content for item in self.selected_items)


# ============================================================================
# Context Handoff Engine
# ============================================================================

class ContextHandoff:
    """
    MI-based context handoff for multi-agent coordination.

    Uses mutual information to select the most relevant context items
    for handoff between agents, minimizing redundancy and maximizing
    information density.

    Example:
        handoff = ContextHandoff()
        result = handoff.prepare_handoff(
            from_agent="researcher",
            to_agent="summarizer",
            full_context=["context1", "context2", ...],
            target_agent_capability="summarization"
        )
        print(result.get_context_text())
    """

    # MI threshold per strategy
    MI_THRESHOLD = {
        HandoffStrategy.AGGRESSIVE: 0.3,
        HandoffStrategy.BALANCED: 0.2,
        HandoffStrategy.CONSERVATIVE: 0.1,
        HandoffStrategy.FULL: 0.0,
    }

    # Keep ratio per strategy
    KEEP_RATIO = {
        HandoffStrategy.AGGRESSIVE: 0.30,
        HandoffStrategy.BALANCED: 0.50,
        HandoffStrategy.CONSERVATIVE: 0.70,
        HandoffStrategy.FULL: 1.00,
    }

    # Redundancy threshold (cosine similarity above this is redundant)
    REDUNDANCY_THRESHOLD = 0.85

    # Agent capability keywords for MI scoring
    CAPABILITY_KEYWORDS = {
        "summarization": {"summary", "summarize", "brief", "concise", "key points", "highlights"},
        "analysis": {"analyze", "analysis", "examine", "investigate", "evaluate", "assess"},
        "research": {"research", "explore", "investigate", "study", "comprehensive", "detailed"},
        "verification": {"verify", "check", "confirm", "validate", "fact-check", "accurate"},
        "synthesis": {"synthesize", "combine", "integrate", "merge", "consolidate"},
        "code_review": {"code", "review", "bug", "error", "fix", "refactor", "function", "class"},
        "planning": {"plan", "strategy", "approach", "steps", "roadmap", "outline"},
        "generation": {"generate", "create", "write", "produce", "compose"},
    }

    def __init__(
        self,
        redundancy_threshold: float = 0.85,
        min_items: int = 1,
        enable_adaptive_strategy: bool = True,
    ):
        """
        Initialize context handoff engine.

        Args:
            redundancy_threshold: Similarity threshold for redundancy detection
            min_items: Minimum items to keep regardless of scores
            enable_adaptive_strategy: Use Phase 6.1 adaptive strategy selection
        """
        self.redundancy_threshold = redundancy_threshold
        self.min_items = min_items
        self.enable_adaptive_strategy = enable_adaptive_strategy

        # Statistics
        self._total_handoffs = 0
        self._total_items_processed = 0
        self._total_redundancy_removed = 0.0

    def prepare_handoff(
        self,
        from_agent: str,
        to_agent: str,
        full_context: List[str],
        target_agent_capability: str,
        strategy: Optional[HandoffStrategy] = None,
        token_budget: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> HandoffContext:
        """
        Prepare context for handoff to target agent.

        Uses MI scoring to select the most relevant context items
        and removes redundant information.

        Args:
            from_agent: Source agent identifier
            to_agent: Target agent identifier
            full_context: List of context strings to filter
            target_agent_capability: Target agent's capability (e.g., "summarization")
            strategy: Handoff strategy (auto-selected if None)
            token_budget: Maximum tokens for handoff (optional)
            metadata: Optional metadata for each context item

        Returns:
            HandoffContext with selected and dropped items
        """
        if not full_context:
            return HandoffContext(
                selected_items=[],
                dropped_items=[],
                original_count=0,
                selected_count=0,
                redundant_count=0,
                total_mi=0.0,
                avg_mi=0.0,
                redundancy_removed=0.0,
                strategy=strategy or HandoffStrategy.BALANCED,
                from_agent=from_agent,
                to_agent=to_agent,
                target_capability=target_agent_capability,
            )

        # Auto-select strategy if not provided
        if strategy is None:
            strategy = self._select_strategy(target_agent_capability)

        # Create context items
        items = []
        for i, content in enumerate(full_context):
            item_metadata = metadata[i] if metadata and i < len(metadata) else {}
            items.append(ContextItem(
                content=content,
                source_agent=from_agent,
                metadata=item_metadata,
            ))

        # Step 1: Score items by MI with target capability
        self._score_items_by_mi(items, target_agent_capability)

        # Step 2: Detect and mark redundant items
        redundant_indices = self._detect_redundancy(items)
        for idx in redundant_indices:
            items[idx].redundancy_score = 1.0

        # Step 3: Select items based on strategy and budget
        selected, dropped = self._select_items(
            items=items,
            strategy=strategy,
            token_budget=token_budget,
            redundant_indices=redundant_indices,
        )

        # Calculate metrics
        total_mi = sum(item.mi_score for item in selected)
        avg_mi = total_mi / len(selected) if selected else 0.0
        redundancy_removed = len(redundant_indices) / len(items) if items else 0.0

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(selected)
        original_tokens = self._estimate_tokens(items)
        tokens_saved = original_tokens - estimated_tokens

        # Update statistics
        self._total_handoffs += 1
        self._total_items_processed += len(items)
        self._total_redundancy_removed += redundancy_removed

        return HandoffContext(
            selected_items=selected,
            dropped_items=dropped,
            original_count=len(items),
            selected_count=len(selected),
            redundant_count=len(redundant_indices),
            total_mi=total_mi,
            avg_mi=avg_mi,
            redundancy_removed=redundancy_removed,
            strategy=strategy,
            from_agent=from_agent,
            to_agent=to_agent,
            target_capability=target_agent_capability,
            token_budget=token_budget,
            estimated_tokens=estimated_tokens,
            tokens_saved=tokens_saved,
        )

    def _select_strategy(self, target_capability: str) -> HandoffStrategy:
        """
        Auto-select handoff strategy based on target capability.

        Research/analysis tasks -> CONSERVATIVE (need more context)
        Summarization/generation -> AGGRESSIVE (need less, focused context)
        Other -> BALANCED
        """
        capability = target_capability.lower()

        # Research-style capabilities need more context
        if capability in {"research", "analysis", "verification", "investigation"}:
            return HandoffStrategy.CONSERVATIVE

        # Generative/summarization needs focused context
        if capability in {"summarization", "generation", "planning"}:
            return HandoffStrategy.AGGRESSIVE

        # Default to balanced
        return HandoffStrategy.BALANCED

    def _score_items_by_mi(
        self,
        items: List[ContextItem],
        target_capability: str,
    ) -> None:
        """
        Score context items by MI with target agent capability.

        MI estimation based on:
        1. Keyword overlap with capability
        2. Content richness (word diversity)
        3. Length normalization
        """
        # Get capability keywords
        cap_keywords = self.CAPABILITY_KEYWORDS.get(
            target_capability.lower(),
            set()  # Empty if unknown capability
        )

        for item in items:
            content_lower = item.content.lower()
            words = set(re.findall(r'\w+', content_lower))

            # Component 1: Keyword overlap (direct relevance)
            if cap_keywords:
                keyword_overlap = len(words & cap_keywords) / len(cap_keywords)
            else:
                keyword_overlap = 0.5  # Default for unknown capability

            # Component 2: Content richness (normalized entropy)
            word_counts = Counter(re.findall(r'\w+', content_lower))
            total_words = sum(word_counts.values())
            if total_words > 0:
                probs = [c / total_words for c in word_counts.values()]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                max_entropy = math.log2(len(word_counts)) if word_counts else 1
                richness = entropy / max_entropy if max_entropy > 0 else 0
            else:
                richness = 0.0

            # Component 3: Length normalization (prefer moderate length)
            word_count = len(item.content.split())
            if word_count < 10:
                length_score = 0.3  # Too short
            elif word_count > 500:
                length_score = 0.5  # Too long
            else:
                length_score = 0.8  # Good length

            # Combined MI score (weighted combination)
            item.mi_score = (
                0.5 * keyword_overlap +
                0.3 * richness +
                0.2 * length_score
            )

            # Also set relevance score
            item.relevance_score = keyword_overlap

    def _detect_redundancy(self, items: List[ContextItem]) -> Set[int]:
        """
        Detect redundant items using word overlap similarity.

        Returns indices of items that are redundant (high similarity to earlier items).
        """
        redundant = set()

        # Convert items to word sets for comparison
        word_sets = []
        for item in items:
            words = set(re.findall(r'\w+', item.content.lower()))
            word_sets.append(words)

        # Pairwise comparison (O(n^2) but typically n is small)
        for i in range(len(items)):
            if i in redundant:
                continue

            for j in range(i + 1, len(items)):
                if j in redundant:
                    continue

                # Jaccard similarity
                set_i = word_sets[i]
                set_j = word_sets[j]

                if not set_i or not set_j:
                    continue

                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                similarity = intersection / union if union > 0 else 0

                if similarity >= self.redundancy_threshold:
                    # Mark lower MI item as redundant
                    if items[i].mi_score >= items[j].mi_score:
                        redundant.add(j)
                    else:
                        redundant.add(i)
                        break  # Stop checking i if it's redundant

        return redundant

    def _select_items(
        self,
        items: List[ContextItem],
        strategy: HandoffStrategy,
        token_budget: Optional[int],
        redundant_indices: Set[int],
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """
        Select items based on strategy, MI scores, and budget.

        Priority order:
        1. Remove redundant items
        2. Sort by MI score
        3. Apply strategy keep ratio
        4. Respect token budget
        5. Ensure minimum items
        """
        # Filter out redundant items
        non_redundant = [
            (i, item) for i, item in enumerate(items)
            if i not in redundant_indices
        ]

        # Sort by MI score (descending)
        non_redundant.sort(key=lambda x: x[1].mi_score, reverse=True)

        # Apply strategy keep ratio
        mi_threshold = self.MI_THRESHOLD[strategy]
        keep_ratio = self.KEEP_RATIO[strategy]
        max_items = max(self.min_items, int(len(items) * keep_ratio))

        selected = []
        dropped = []
        current_tokens = 0

        for _, item in non_redundant:
            # Check MI threshold (for non-FULL strategies)
            if strategy != HandoffStrategy.FULL and item.mi_score < mi_threshold:
                # Allow if we haven't reached minimum
                if len(selected) < self.min_items:
                    pass  # Continue with selection
                else:
                    dropped.append(item)
                    continue

            # Check max items
            if len(selected) >= max_items:
                dropped.append(item)
                continue

            # Check token budget
            item_tokens = self._estimate_tokens([item])
            if token_budget and current_tokens + item_tokens > token_budget:
                # Allow if we haven't reached minimum
                if len(selected) < self.min_items:
                    pass  # Continue with selection
                else:
                    dropped.append(item)
                    continue

            selected.append(item)
            current_tokens += item_tokens

        # Add redundant items to dropped
        for i in redundant_indices:
            dropped.append(items[i])

        return selected, dropped

    def _estimate_tokens(self, items: List[ContextItem]) -> int:
        """Estimate token count for items (rough: words * 1.3)."""
        total_words = sum(len(item.content.split()) for item in items)
        return int(total_words * 1.3)

    def get_statistics(self) -> Dict[str, Any]:
        """Get handoff statistics."""
        return {
            "total_handoffs": self._total_handoffs,
            "total_items_processed": self._total_items_processed,
            "avg_redundancy_removed": (
                self._total_redundancy_removed / self._total_handoffs
                if self._total_handoffs > 0 else 0.0
            ),
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def prepare_agent_handoff(
    from_agent: str,
    to_agent: str,
    context: List[str],
    capability: str,
    strategy: Optional[HandoffStrategy] = None,
    token_budget: Optional[int] = None,
) -> HandoffContext:
    """
    Convenience function for preparing agent context handoff.

    Args:
        from_agent: Source agent identifier
        to_agent: Target agent identifier
        context: List of context strings
        capability: Target agent's capability
        strategy: Handoff strategy (auto-selected if None)
        token_budget: Maximum tokens for handoff (optional)

    Returns:
        HandoffContext with selected context
    """
    handoff = ContextHandoff()
    return handoff.prepare_handoff(
        from_agent=from_agent,
        to_agent=to_agent,
        full_context=context,
        target_agent_capability=capability,
        strategy=strategy,
        token_budget=token_budget,
    )


def get_optimal_handoff_strategy(
    target_capability: str,
    context_size: int,
    urgency: str = "normal",
) -> HandoffStrategy:
    """
    Get optimal handoff strategy based on parameters.

    Args:
        target_capability: Target agent's capability
        context_size: Number of context items
        urgency: "low", "normal", or "high"

    Returns:
        Recommended HandoffStrategy
    """
    capability = target_capability.lower()

    # High urgency = more aggressive compression
    if urgency == "high":
        return HandoffStrategy.AGGRESSIVE

    # Low urgency = keep more context
    if urgency == "low":
        return HandoffStrategy.CONSERVATIVE

    # Large context = more aggressive
    if context_size > 20:
        return HandoffStrategy.BALANCED

    # Small context = keep most
    if context_size < 5:
        return HandoffStrategy.FULL

    # Research capabilities need more context
    if capability in {"research", "analysis", "verification"}:
        return HandoffStrategy.CONSERVATIVE

    # Generative tasks can use less
    if capability in {"summarization", "generation"}:
        return HandoffStrategy.AGGRESSIVE

    return HandoffStrategy.BALANCED
