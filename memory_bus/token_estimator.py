"""
Simple token estimation heuristic — 2026-02-09

Marked as REPLACEABLE: swap for tiktoken or a proper tokenizer when needed.
Uses the standard ~4 chars per token approximation for English text.
"""

from __future__ import annotations

# Rough ratio: 1 token ≈ 4 characters for English text.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def fits_budget(text: str, budget: int) -> bool:
    """Check if text fits within a token budget."""
    return estimate_tokens(text) <= budget


def trim_to_budget(text: str, budget: int) -> str:
    """Trim text to fit within a token budget (character-level truncation)."""
    if budget <= 0:
        return ""
    max_chars = budget * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."
