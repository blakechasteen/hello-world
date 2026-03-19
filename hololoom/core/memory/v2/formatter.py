"""
Formatter — Fixed-budget context packer for the Matryoshka shell loop.

Each shell in the Matryoshka architecture gets a FIXED token budget.
The formatter's job is to pack the most valuable context into that budget.

Design principles:
  - Budget is per-shell, not per-query (the orchestrator allocates)
  - Items are packed greedily by PPR score (navigator output)
  - Each item gets a structural prefix: [entity|fact|episode] with confidence
  - Truncation happens at item boundaries (never mid-sentence)
  - The formatter produces a single string — the "context block" for the LLM

Token estimation uses the 4-chars-per-token heuristic (good enough for
English text with Llama/Qwen tokenizers). For production, plug in a
real tokenizer via the token_fn parameter.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ResolutionThresholds:
    """PPR score thresholds for multi-resolution formatting.

    Items with high PPR scores get full detail. Lower-scoring items
    get progressively compact formats, fitting more items into budget.
    """
    detail: float = 0.05     # Above this → full text (raw content)
    compact: float = 0.01    # Above this → key-value compact format
    # Below compact → summary only (if available via SUMMARIZES edges)


@dataclass
class FormatConfig:
    """Configuration for context formatting."""
    token_budget: int = 2048          # Max tokens for this shell's context block
    header_tokens: int = 20           # Reserved for the block header/separator
    item_overhead: int = 8            # Tokens per item for prefix/separator
    max_items: int = 30               # Hard cap on number of items
    include_scores: bool = False      # Include PPR scores in output (debug)
    include_provenance: bool = True   # Include source type prefix
    separator: str = "\n"             # Between items
    block_header: str = "--- CONTEXT ---"
    block_footer: str = "--- END CONTEXT ---"
    truncate_item_at: int = 512       # Max tokens per single item
    resolution: Optional["ResolutionThresholds"] = None  # Multi-resolution control


# =============================================================================
# Token Estimation
# =============================================================================

def default_token_estimate(text: str) -> int:
    """Estimate token count from text (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


# =============================================================================
# Formatted Item
# =============================================================================

@dataclass
class FormattedItem:
    """A single context item ready for packing."""
    node_id: str
    text: str
    score: float
    memory_type: str
    token_count: int
    confidence: float | None = None
    prefix: str = ""

    @property
    def formatted(self) -> str:
        """Full formatted string for this item."""
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        parts.append(self.text)
        return " ".join(parts)


# =============================================================================
# Formatter
# =============================================================================

class Formatter:
    """Fixed-budget context packer.

    Takes navigator output (ranked nodes with metadata) and packs them
    into a token-budgeted context block for the LLM.

    Usage:
        fmt = Formatter(config=FormatConfig(token_budget=2048))
        block = fmt.pack(
            ranked_nodes=navigator_result.ranked_nodes,
            node_data=navigator_result.node_data,
        )
        # block.text → "--- CONTEXT ---\n[entity] Thompson Sampling\n..."
        # block.token_count → 1847
        # block.items_packed → 12
    """

    def __init__(
        self,
        config: FormatConfig | None = None,
        token_fn: Callable[[str], int] | None = None,
    ):
        self.config = config or FormatConfig()
        self.token_fn = token_fn or default_token_estimate

    def pack(
        self,
        ranked_nodes: list[tuple[str, float]],
        node_data: dict[str, dict[str, Any]],
        confidence_scores: dict[str, float] | None = None,
    ) -> "ContextBlock":
        """Pack ranked nodes into a fixed-budget context block.

        Greedy packing: iterate through nodes in PPR-rank order,
        add each item if it fits within the remaining budget.

        Args:
            ranked_nodes: [(node_id, ppr_score)] from navigator
            node_data: {node_id: metadata_dict} from navigator
            confidence_scores: Optional {node_id: confidence} from confidence estimator

        Returns:
            ContextBlock with packed text and metadata
        """
        cfg = self.config
        confidence_scores = confidence_scores or {}

        available = cfg.token_budget - cfg.header_tokens
        packed: list[FormattedItem] = []
        total_tokens = cfg.header_tokens

        resolution = cfg.resolution  # May be None (no multi-resolution)

        for node_id, score in ranked_nodes:
            if len(packed) >= cfg.max_items:
                break

            data = node_data.get(node_id, {})
            content = data.get("content", data.get("text", ""))
            if not content:
                continue

            memory_type = data.get("memory_type", "fact")
            confidence = confidence_scores.get(node_id)

            # Multi-resolution: format based on PPR score
            full_text = self._apply_resolution(content, score, data, resolution)

            # Build prefix
            prefix = self._build_prefix(memory_type, score, confidence)

            # Estimate tokens
            item_tokens = self.token_fn(prefix + " " + full_text) + cfg.item_overhead

            # Truncate if single item exceeds per-item cap
            if item_tokens > cfg.truncate_item_at:
                char_limit = cfg.truncate_item_at * 4  # Reverse the 4-char heuristic
                full_text = full_text[:char_limit] + "..."
                item_tokens = self.token_fn(prefix + " " + full_text) + cfg.item_overhead

            # Check budget
            if total_tokens + item_tokens > cfg.token_budget:
                # Try to fit a truncated version
                remaining_chars = (cfg.token_budget - total_tokens - cfg.item_overhead) * 4
                if remaining_chars > 100:  # Only if meaningful space left
                    full_text = full_text[:remaining_chars] + "..."
                    item_tokens = self.token_fn(prefix + " " + full_text) + cfg.item_overhead
                else:
                    break  # No room

            item = FormattedItem(
                node_id=node_id,
                text=full_text,
                score=score,
                memory_type=memory_type,
                token_count=item_tokens,
                confidence=confidence,
                prefix=prefix,
            )
            packed.append(item)
            total_tokens += item_tokens

        # Assemble the context block
        lines = [cfg.block_header]
        for item in packed:
            lines.append(item.formatted)
        lines.append(cfg.block_footer)

        block_text = cfg.separator.join(lines)

        return ContextBlock(
            text=block_text,
            items=packed,
            token_count=total_tokens,
            token_budget=cfg.token_budget,
            items_packed=len(packed),
            items_offered=len(ranked_nodes),
            utilization=total_tokens / max(1, cfg.token_budget),
        )

    def _apply_resolution(
        self,
        content: str,
        score: float,
        data: dict[str, Any],
        resolution: ResolutionThresholds | None,
    ) -> str:
        """Apply multi-resolution formatting based on PPR score.

        High-scoring items → full text (detail)
        Mid-scoring items → compact key-value format
        Low-scoring items → summary only (if available)

        When resolution is None, returns full content (backward compatible).
        """
        if resolution is None:
            return content

        if score >= resolution.detail:
            # Full detail — raw content
            return content

        if score >= resolution.compact:
            # Compact format — extract key-value pairs
            return self._format_compact(content, data)

        # Summary level — use summary content if this is a summary node,
        # otherwise compact format as fallback
        memory_type = data.get("memory_type", "")
        if memory_type == "summary":
            return content  # Already a summary
        return self._format_compact(content, data)

    @staticmethod
    def _format_compact(content: str, data: dict[str, Any]) -> str:
        """Compact key-value format for mid-priority items.

        Extracts the first sentence and any structured metadata,
        reducing token usage while preserving key information.
        """
        # First sentence extraction
        first_sentence = content
        for delim in ".!?":
            idx = content.find(delim)
            if idx > 0 and idx < 200:
                first_sentence = content[:idx + 1]
                break
        else:
            if len(content) > 200:
                first_sentence = content[:200] + "..."

        # Add entity name if available
        entity = data.get("entity", data.get("name", ""))
        if entity and entity not in first_sentence:
            return f"{entity}: {first_sentence}"
        return first_sentence

    def _build_prefix(
        self,
        memory_type: str,
        score: float,
        confidence: float | None,
    ) -> str:
        """Build the structural prefix for an item."""
        if not self.config.include_provenance:
            return ""

        # Type tag
        type_map = {
            "entity": "entity",
            "factual": "fact",
            "episodic": "episode",
            "procedural": "proc",
            "summary": "summ",
        }
        tag = type_map.get(memory_type, memory_type[:6])

        parts = [f"[{tag}]"]

        if self.config.include_scores:
            parts.append(f"(r={score:.3f})")

        if confidence is not None:
            if confidence >= 0.8:
                parts.append("[HIGH]")
            elif confidence >= 0.5:
                parts.append("[MED]")
            else:
                parts.append("[LOW]")

        return " ".join(parts)


# =============================================================================
# Context Block (output)
# =============================================================================

@dataclass
class ContextBlock:
    """A packed context block ready for LLM consumption."""
    text: str
    items: list[FormattedItem]
    token_count: int
    token_budget: int
    items_packed: int
    items_offered: int
    utilization: float

    @property
    def is_empty(self) -> bool:
        return self.items_packed == 0

    @property
    def overflow(self) -> bool:
        return self.token_count > self.token_budget

    def summary(self) -> str:
        return (
            f"ContextBlock: {self.items_packed}/{self.items_offered} items, "
            f"{self.token_count}/{self.token_budget} tokens "
            f"({self.utilization:.0%} util)"
        )
