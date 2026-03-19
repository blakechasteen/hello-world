"""
Multi-Resolution Rendering — FULL/COMPACT/STUB item rendering for token maximization.

Instead of full-or-nothing, renders items at three detail levels based on
an effective score (importance * 0.6 + keyword_match * 0.4). Fits 2-3x more
items into the same token budget.

Integration: optional `multi_resolution` parameter on LiteMemoryBus.__init__.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lite_bus import StoredItem

__all__ = [
    "RenderResolution",
    "ResolutionPolicy",
    "MultiResolutionRenderer",
]


class RenderResolution(Enum):
    FULL = "full"         # Full content + all properties (~item.token_estimate tokens)
    COMPACT = "compact"   # First sentence + type + importance (~30 tokens)
    STUB = "stub"         # One-line path + importance (~8 tokens)


@dataclass
class ResolutionPolicy:
    """Thresholds for assigning resolution levels."""
    full_threshold: float = 0.7
    compact_threshold: float = 0.3
    compact_max_tokens: int = 30
    stub_max_tokens: int = 8
    # Blending weights: importance vs keyword match
    importance_weight: float = 0.6
    keyword_weight: float = 0.4


class MultiResolutionRenderer:
    """Assigns and renders items at FULL/COMPACT/STUB resolution."""

    def __init__(self, policy: ResolutionPolicy | None = None):
        self.policy = policy or ResolutionPolicy()

    def assign_resolution(
        self,
        items: list["StoredItem"],
        query_keywords: list[str] | None = None,
    ) -> dict[str, RenderResolution]:
        """Assign resolution to each item based on effective score."""
        keywords = query_keywords or []
        assignments: dict[str, RenderResolution] = {}

        for item in items:
            if keywords:
                content_lower = item.content.lower()
                kw_score = sum(1 for kw in keywords if kw in content_lower) / len(keywords)
                effective = (
                    self.policy.importance_weight * item.importance
                    + self.policy.keyword_weight * kw_score
                )
            else:
                # No keywords → importance alone determines resolution
                effective = item.importance

            if effective >= self.policy.full_threshold:
                assignments[item.id] = RenderResolution.FULL
            elif effective >= self.policy.compact_threshold:
                assignments[item.id] = RenderResolution.COMPACT
            else:
                assignments[item.id] = RenderResolution.STUB

        return assignments

    def render_compact(self, item: "StoredItem") -> str:
        """First sentence + type + importance."""
        content = item.content
        # Take first sentence
        dot_pos = content.find(".")
        if dot_pos > 0 and dot_pos < 200:
            first_sentence = content[:dot_pos + 1]
        elif len(content) > 120:
            first_sentence = content[:117] + "..."
        else:
            first_sentence = content
        et = item.entity_type or "general"
        return f"[{item.memory_type}/{et}] {first_sentence} (imp={item.importance:.1f})"

    def render_stub(self, item: "StoredItem") -> str:
        """One-line path + importance."""
        et = item.entity_type or "general"
        return f"  - {item.memory_type}/{et}/{item.id[:8]} (imp={item.importance:.1f})"

    def token_cost(self, resolution: RenderResolution, item: "StoredItem") -> int:
        """Token cost for an item at a given resolution."""
        if resolution == RenderResolution.FULL:
            return item.token_estimate
        elif resolution == RenderResolution.COMPACT:
            return self.policy.compact_max_tokens
        else:
            return self.policy.stub_max_tokens

    def estimate_tokens_saved(
        self,
        items: list["StoredItem"],
        assignments: dict[str, RenderResolution],
    ) -> int:
        """Estimate token savings from using compact/stub instead of full."""
        saved = 0
        for item in items:
            res = assignments.get(item.id, RenderResolution.FULL)
            if res == RenderResolution.COMPACT:
                saved += max(0, item.token_estimate - self.policy.compact_max_tokens)
            elif res == RenderResolution.STUB:
                saved += max(0, item.token_estimate - self.policy.stub_max_tokens)
        return saved
