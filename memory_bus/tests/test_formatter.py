"""
Unit tests for formatter — 2026-02-09

Invariant: formatter never exceeds token budget; degrades detail level correctly.
"""

from memory_bus.formatter import format_item, format_items
from memory_bus.models import DetailLevel, MemoryItem, PressureTier, ResolutionPath
from memory_bus.token_estimator import estimate_tokens


class TestFormatItem:
    def test_title_level(self, sample_items):
        item = sample_items[0]
        text = format_item(item, DetailLevel.TITLE)
        assert "[entity]" in text
        assert "Aurora" in text

    def test_summary_level(self, sample_items):
        item = sample_items[1]
        text = format_item(item, DetailLevel.SUMMARY)
        assert "[episode]" in text
        assert "Inspected" in text

    def test_compact_level(self, sample_items):
        item = sample_items[1]
        text = format_item(item, DetailLevel.COMPACT)
        assert "Summary:" in text
        assert "Entities:" in text

    def test_full_level(self, sample_items):
        item = sample_items[1]
        text = format_item(item, DetailLevel.FULL)
        assert "Content:" in text
        assert "Full inspection" in text


class TestFormatItems:
    def test_empty_list(self):
        text, tokens, level = format_items([])
        assert text == ""
        assert tokens == 0

    def test_fits_budget(self, sample_items):
        text, tokens, level = format_items(
            sample_items, token_budget=5000, detail_level=DetailLevel.FULL
        )
        assert tokens <= 5000
        assert len(text) > 0
        assert level == DetailLevel.FULL

    def test_budget_forces_detail_degradation(self, sample_items):
        # Very tight budget should force degradation
        text, tokens, level = format_items(
            sample_items, token_budget=30, detail_level=DetailLevel.FULL
        )
        assert tokens <= 30
        # Should have degraded from FULL
        assert level in (DetailLevel.TITLE, DetailLevel.SUMMARY, DetailLevel.COMPACT)

    def test_never_exceeds_budget(self, sample_items):
        for budget in [10, 20, 50, 100, 200, 500]:
            text, tokens, level = format_items(
                sample_items, token_budget=budget, detail_level=DetailLevel.FULL
            )
            assert tokens <= budget, f"Exceeded budget {budget}: got {tokens}"

    def test_tier3_enforces_500_cap(self, sample_items):
        text, tokens, level = format_items(
            sample_items,
            token_budget=10000,  # Large budget
            pressure_tier=PressureTier.TIER3,
        )
        # Tier3 enforces 500 token cap
        assert tokens <= 500

    def test_default_budget_scales_with_window(self, sample_items):
        # Small context window
        text1, tokens1, _ = format_items(
            sample_items, context_window=1000, pressure_tier=PressureTier.TIER0
        )
        # Large context window
        text2, tokens2, _ = format_items(
            sample_items, context_window=200_000, pressure_tier=PressureTier.TIER0
        )
        # Both should work, larger window allows more tokens
        assert tokens1 <= 150  # 15% of 1000
        assert tokens2 <= 30_000  # 15% of 200000

    def test_truncates_items_at_title_level(self):
        """Many items with tiny budget should truncate item count."""
        items = [
            MemoryItem(
                id=f"item_{i}",
                kind="episode",
                title=f"Episode number {i} with some description text",
                importance=0.5,
            )
            for i in range(50)
        ]
        text, tokens, level = format_items(
            items, token_budget=50, detail_level=DetailLevel.FULL
        )
        assert tokens <= 50
        # Should not contain all 50 items
        assert text.count("[episode]") < 50
