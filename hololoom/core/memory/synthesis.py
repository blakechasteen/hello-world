"""
Write-Time Synthesis — Proactive merging at store time (borrowed from SimpleMem).

When storing a new memory, checks for similar existing items and decides
whether to store new, merge into existing, link, or supersede.

Integration: optional `synthesis_fn` parameter on LiteMemoryBus.__init__.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lite_bus import StoredItem

__all__ = ["SynthesisResult", "default_synthesis_fn"]


@dataclass
class SynthesisResult:
    """What to do with a new memory item relative to existing items."""
    action: str  # "store_new" | "merge_into" | "link_to" | "supersedes"
    target_id: str | None = None         # Existing item to merge/link/supersede
    merged_content: str | None = None    # New content after merge
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    # [(source_id_placeholder, target_id, rel_type), ...]


def default_synthesis_fn(similarity_threshold: float = 0.85):
    """Factory: heuristic synthesis using substring matching.

    - Exact substring match → merge_into (keep longer content)
    - Otherwise if candidates exist → link_to (create SIMILAR_TO edge)
    - No candidates → store_new

    For full cosine similarity, pass a custom synthesis_fn that uses
    the embedding scores returned by semantic_fn.
    """
    async def _synthesize(
        content: str, candidates: list["StoredItem"],
    ) -> SynthesisResult:
        content_stripped = content.strip()
        content_lower = content_stripped.lower()

        for c in candidates:
            c_lower = c.content.strip().lower()
            # Exact dedup: one is substring of the other
            if content_lower in c_lower or c_lower in content_lower:
                merged = max(content_stripped, c.content.strip(), key=len)
                return SynthesisResult(
                    action="merge_into",
                    target_id=c.id,
                    merged_content=merged,
                )

        # No exact match but candidates exist → link to most relevant
        if candidates:
            return SynthesisResult(
                action="link_to",
                target_id=candidates[0].id,
                edges=[("__NEW__", candidates[0].id, "SIMILAR_TO")],
            )

        return SynthesisResult(action="store_new")

    return _synthesize
