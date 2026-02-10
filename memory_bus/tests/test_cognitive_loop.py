"""
Tests for the cognitive loop — 2026-02-10

Validates wiring, tool dispatch, and result structure without requiring
a live database or API key.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_bus.cognitive_loop import (
    MEMORY_TOOLS,
    SYSTEM_PROMPT,
    LoopResult,
    execute_tool_call,
    extract_likely_entity_names,
    run_cognitive_loop,
)


# ---------------------------------------------------------------------------
# Entity name extraction
# ---------------------------------------------------------------------------


class TestExtractEntityNames:
    def test_extracts_capitalized_names(self):
        names = extract_likely_entity_names("What about Aurora and Nova?")
        assert "Aurora" in names
        assert "Nova" in names

    def test_filters_question_words(self):
        names = extract_likely_entity_names("What should I do about Aurora?")
        assert "What" not in names
        assert "Aurora" in names

    def test_multi_word_names(self):
        names = extract_likely_entity_names("Check on Hive Aurora tomorrow")
        assert "Hive Aurora" in names

    def test_deduplicates(self):
        names = extract_likely_entity_names("Aurora and aurora again")
        assert len([n for n in names if n.lower() == "aurora"]) == 1

    def test_empty_input(self):
        assert extract_likely_entity_names("") == []

    def test_no_entities(self):
        assert extract_likely_entity_names("what is 2 + 2?") == []


# ---------------------------------------------------------------------------
# Tool definitions structure
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_four_tools_defined(self):
        assert len(MEMORY_TOOLS) == 4

    def test_tool_names(self):
        names = {t["name"] for t in MEMORY_TOOLS}
        assert names == {
            "memory_query",
            "memory_store",
            "memory_resolve_entity",
            "memory_explain",
        }

    def test_all_have_input_schema(self):
        for tool in MEMORY_TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_resolve_requires_name(self):
        resolve = next(t for t in MEMORY_TOOLS if t["name"] == "memory_resolve_entity")
        assert "name" in resolve["input_schema"]["required"]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_mentions_memory_bus(self):
        assert "Memory Bus" in SYSTEM_PROMPT

    def test_mentions_confabulation(self):
        assert "confabulation" in SYSTEM_PROMPT.lower()

    def test_mentions_memory_query(self):
        assert "memory_query" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class TestExecuteToolCall:
    @pytest.fixture
    def mock_bus(self):
        bus = MagicMock()
        bus.memory_query = AsyncMock()
        bus.memory_store = AsyncMock()
        bus.memory_resolve_entity = AsyncMock()
        bus.memory_explain = AsyncMock()
        return bus

    @pytest.mark.asyncio
    async def test_routes_memory_query(self, mock_bus):
        mock_result = MagicMock()
        mock_result.items = []
        mock_result.resolution_path.value = "exact"
        mock_result.formatted_context = "context"
        mock_result.warnings = []
        mock_result.semantic_stub_reached = False
        mock_bus.memory_query.return_value = mock_result

        result = await execute_tool_call(
            mock_bus, "memory_query",
            {"entity_names": ["Aurora"]},
            loom_id="demo", loop_id="loop_1",
        )

        mock_bus.memory_query.assert_awaited_once()
        assert "items_count" in result
        assert "resolution_path" in result

    @pytest.mark.asyncio
    async def test_routes_memory_store(self, mock_bus):
        mock_bus.memory_store.return_value = {"entities": ["ent_1"]}

        result = await execute_tool_call(
            mock_bus, "memory_store",
            {"entities": [{"name": "Test"}]},
            loom_id="demo", loop_id="loop_1",
        )

        mock_bus.memory_store.assert_awaited_once()
        assert result == {"entities": ["ent_1"]}

    @pytest.mark.asyncio
    async def test_routes_memory_resolve(self, mock_bus):
        mock_result = MagicMock()
        mock_result.query_name = "Aurora"
        mock_result.candidates = []
        mock_result.best_match = None
        mock_bus.memory_resolve_entity.return_value = mock_result

        result = await execute_tool_call(
            mock_bus, "memory_resolve_entity",
            {"name": "Aurora"},
            loom_id="demo", loop_id="loop_1",
        )

        mock_bus.memory_resolve_entity.assert_awaited_once()
        assert result["query_name"] == "Aurora"
        assert result["best_match"] is None

    @pytest.mark.asyncio
    async def test_routes_memory_explain(self, mock_bus):
        mock_result = MagicMock()
        mock_result.item_id = "ent_1"
        mock_result.confidence = 0.9
        mock_result.status = "confirmed"
        mock_result.derived_from = []
        mock_bus.memory_explain.return_value = mock_result

        result = await execute_tool_call(
            mock_bus, "memory_explain",
            {"item_id": "ent_1"},
            loom_id="demo", loop_id="loop_1",
        )

        mock_bus.memory_explain.assert_awaited_once()
        assert result["item_id"] == "ent_1"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mock_bus):
        result = await execute_tool_call(
            mock_bus, "nonexistent_tool",
            {},
            loom_id="demo", loop_id="loop_1",
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# Dry-run loop
# ---------------------------------------------------------------------------


class TestDryRunLoop:
    @pytest.mark.asyncio
    async def test_dry_run_returns_loop_result(self):
        result = await run_cognitive_loop(
            "What treatment should I apply to Aurora?",
            dry_run=True,
        )

        assert isinstance(result, LoopResult)
        assert result.loop_id.startswith("loop_")
        assert result.question == "What treatment should I apply to Aurora?"
        assert "Aurora" in result.primed_entity_names
        assert "DRY RUN" in result.model_response
        assert result.primed_items_count == 0  # no bus connected
        assert result.verification == {}  # no bus connected
        assert result.total_ms > 0

    @pytest.mark.asyncio
    async def test_dry_run_extracts_entities(self):
        result = await run_cognitive_loop(
            "Tell me about Hive Nova and Hive Aurora",
            dry_run=True,
        )

        names_lower = [n.lower() for n in result.primed_entity_names]
        assert "hive nova" in names_lower or "nova" in names_lower
        assert "hive aurora" in names_lower or "aurora" in names_lower

    @pytest.mark.asyncio
    async def test_dry_run_no_entities(self):
        result = await run_cognitive_loop(
            "what is 2 plus 2?",
            dry_run=True,
        )

        assert result.primed_entity_names == []
        assert result.primed_items_count == 0


# ---------------------------------------------------------------------------
# Loop with mocked bus (no API key needed)
# ---------------------------------------------------------------------------


class TestLoopWithMockedBus:
    @pytest.fixture
    def mock_bus(self):
        bus = MagicMock()

        # memory_query returns primed context
        query_result = MagicMock()
        query_result.items = [
            MagicMock(id="ent_aurora", kind="entity"),
            MagicMock(id="ep_1", kind="episode"),
        ]
        query_result.formatted_context = "Aurora: 8 frames brood, varroa seen"
        query_result.warnings = []
        query_result.resolution_path.value = "structured"
        query_result.semantic_stub_reached = False
        bus.memory_query = AsyncMock(return_value=query_result)

        # memory_verify returns verification
        bus.memory_verify = AsyncMock(return_value={
            "total_mentions": 2,
            "grounded_count": 1,
            "ungrounded_count": 1,
            "confabulation_rate": 0.5,
            "grounding_rate": 0.5,
            "false_positive_count": 0,
            "grounded": [{"text": "Aurora", "grounded": True, "grounded_via": "primed_name", "entity_id": None}],
            "ungrounded": [{"text": "Oxalic Acid", "grounded": False, "grounded_via": "", "entity_id": None}],
        })

        return bus

    @pytest.mark.asyncio
    async def test_priming_uses_bus(self, mock_bus):
        """Verify the loop calls memory_query for priming."""
        result = await run_cognitive_loop(
            "What about Aurora?",
            bus=mock_bus,
            dry_run=True,  # dry_run skips Claude API but still primes
        )

        # Bus should have been called for priming
        mock_bus.memory_query.assert_awaited_once()
        call_kwargs = mock_bus.memory_query.call_args
        assert call_kwargs.kwargs.get("aggressive_prime") is True
        assert result.primed_items_count == 2
        assert "Aurora" in result.primed_context

    @pytest.mark.asyncio
    async def test_verification_called_on_response(self, mock_bus):
        """Verify the loop calls memory_verify on the model response."""
        result = await run_cognitive_loop(
            "What about Aurora?",
            bus=mock_bus,
            dry_run=True,
        )

        # Verify should have been called with the dry-run response
        mock_bus.memory_verify.assert_awaited_once()
        call_kwargs = mock_bus.memory_verify.call_args
        assert "Aurora" in call_kwargs.kwargs["primed_entity_names"]
        assert call_kwargs.kwargs["primed_entity_ids"] == ["ent_aurora"]

        # Result should contain verification data
        assert result.verification["total_mentions"] == 2
        assert result.verification["confabulation_rate"] == 0.5


# ---------------------------------------------------------------------------
# LoopResult structure
# ---------------------------------------------------------------------------


class TestLoopResult:
    def test_fields(self):
        r = LoopResult(
            loop_id="loop_test",
            question="test?",
            primed_entity_names=["A"],
            primed_items_count=1,
            primed_context="ctx",
            priming_ms=10.0,
            model_response="answer",
            tool_calls_made=[],
            model_ms=100.0,
            verification={"confabulation_rate": 0.0},
            verify_ms=5.0,
            total_ms=115.0,
        )
        assert r.loop_id == "loop_test"
        assert r.total_ms == 115.0
        assert r.verification["confabulation_rate"] == 0.0
