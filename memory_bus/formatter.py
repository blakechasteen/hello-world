"""
Token-budgeted context formatter — 2026-02-09

Formats MemoryItems into context strings with progressive detail degradation.
Invariant: never exceed token budget. Degrade detail level before truncating.
"""

from __future__ import annotations

from .models import DetailLevel, MemoryItem, PressureTier
from .token_estimator import estimate_tokens, trim_to_budget


# Per-item overhead: "---\n" separator + newlines
_SEPARATOR_TOKENS = 2
# Header line per item: "[entity] name\n" or "[episode] title (date)\n"
_HEADER_OVERHEAD = 8

# Detail level ordering: degrade in this order
_DETAIL_ORDERING = [DetailLevel.FULL, DetailLevel.COMPACT, DetailLevel.SUMMARY, DetailLevel.TITLE]


def _default_memory_budget(context_window: int, pressure_tier: PressureTier) -> int:
    """15% of context window, modulated by pressure tier."""
    base = int(context_window * 0.15)
    modifiers = {
        PressureTier.TIER0: 1.0,
        PressureTier.TIER1: 0.8,
        PressureTier.TIER2: 0.5,
        PressureTier.TIER3: 0.25,
    }
    return max(50, int(base * modifiers.get(pressure_tier, 1.0)))


def format_item(item: MemoryItem, detail: DetailLevel) -> str:
    """Format a single MemoryItem at the given detail level."""
    if detail == DetailLevel.TITLE:
        ts = f" ({item.timestamp.strftime('%Y-%m-%d')})" if item.timestamp else ""
        return f"[{item.kind}] {item.title}{ts}"

    if detail == DetailLevel.SUMMARY:
        ts = f" ({item.timestamp.strftime('%Y-%m-%d %H:%M')})" if item.timestamp else ""
        text = item.summary or item.title
        return f"[{item.kind}] {item.title}{ts}\n  {text}"

    if detail == DetailLevel.COMPACT:
        ts = f" ({item.timestamp.isoformat()})" if item.timestamp else ""
        lines = [f"[{item.kind}] {item.title}{ts}"]
        if item.summary and item.summary != item.title:
            lines.append(f"  Summary: {item.summary}")
        if item.related_entity_ids:
            lines.append(f"  Entities: {', '.join(item.related_entity_ids[:5])}")
        return "\n".join(lines)

    # FULL
    ts = f" ({item.timestamp.isoformat()})" if item.timestamp else ""
    lines = [f"[{item.kind}] {item.title}{ts}"]
    if item.summary:
        lines.append(f"  Summary: {item.summary}")
    if item.content:
        lines.append(f"  Content: {item.content}")
    if item.related_entity_ids:
        lines.append(f"  Entities: {', '.join(item.related_entity_ids)}")
    if item.properties:
        for k, v in list(item.properties.items())[:10]:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def format_items(
    items: list[MemoryItem],
    *,
    token_budget: int | None = None,
    detail_level: DetailLevel = DetailLevel.COMPACT,
    context_window: int = 128_000,
    pressure_tier: PressureTier = PressureTier.TIER0,
) -> tuple[str, int, DetailLevel]:
    """
    Format a list of MemoryItems into a context string that fits the token budget.

    Returns (formatted_text, tokens_used, actual_detail_level).

    Strategy:
    1. Try at requested detail level.
    2. If over budget, degrade detail level.
    3. If still over budget at TITLE level, truncate items.
    """
    if not items:
        return "", 0, detail_level

    if token_budget is None:
        token_budget = _default_memory_budget(context_window, pressure_tier)

    # Enforce per-query caps at high pressure
    if pressure_tier == PressureTier.TIER3:
        token_budget = min(token_budget, 500)

    # Find the index of the requested detail level in the ordering
    try:
        start_idx = _DETAIL_ORDERING.index(detail_level)
    except ValueError:
        start_idx = 0

    # Try each detail level from the requested down to TITLE
    for level_idx in range(start_idx, len(_DETAIL_ORDERING)):
        current_level = _DETAIL_ORDERING[level_idx]
        formatted_lines = []
        running_tokens = 0

        for item in items:
            line = format_item(item, current_level)
            line_tokens = estimate_tokens(line) + _SEPARATOR_TOKENS
            if running_tokens + line_tokens > token_budget:
                # At TITLE level, we truncate items (last resort)
                if current_level == DetailLevel.TITLE:
                    break
                # Otherwise try a lower detail level
                break
            formatted_lines.append(line)
            running_tokens += line_tokens
        else:
            # All items fit at this detail level
            text = "\n---\n".join(formatted_lines)
            return text, estimate_tokens(text), current_level

    # Last resort: TITLE level, truncate items to fit
    formatted_lines = []
    running_tokens = 0
    for item in items:
        line = format_item(item, DetailLevel.TITLE)
        line_tokens = estimate_tokens(line) + _SEPARATOR_TOKENS
        if running_tokens + line_tokens > token_budget:
            break
        formatted_lines.append(line)
        running_tokens += line_tokens

    text = "\n---\n".join(formatted_lines)
    actual_tokens = estimate_tokens(text)

    # Final safety: hard trim if somehow still over
    if actual_tokens > token_budget:
        text = trim_to_budget(text, token_budget)
        actual_tokens = token_budget

    return text, actual_tokens, DetailLevel.TITLE
