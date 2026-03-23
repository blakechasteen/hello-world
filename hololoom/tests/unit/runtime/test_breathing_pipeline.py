"""Tests for BreathingPipeline — orchestrator wraps with INHALE/EXHALE/REST."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestBreathingPipeline:
    """Test breathing pipeline composition."""

    def _make_ctx(self):
        """Minimal context-like object with breathing_phase."""
        ctx = MagicMock()
        ctx.breathing_phase = None
        return ctx

    def _make_executor(self, name: str):
        """Mock executor that records the breathing phase it sees."""
        ex = MagicMock()
        phases_seen = []

        async def execute(ctx):
            phases_seen.append(ctx.breathing_phase)
            return ctx

        ex.execute = execute
        ex.phases_seen = phases_seen
        ex.name = name
        return ex

    def _make_pipeline(self, n_executors=8):
        executors = [self._make_executor(f"stage_{i}") for i in range(n_executors)]
        pipeline = MagicMock()
        pipeline.executors = executors
        return pipeline, executors

    @pytest.mark.asyncio
    async def test_phases_set_on_context(self):
        from hololoom.core.orchestrator.breathing_pipeline import BreathingPipeline

        pipeline, executors = self._make_pipeline(8)
        chrono = MagicMock()
        chrono.current_phase = None

        bp = BreathingPipeline(pipeline, chrono=chrono, inhale_stages=4, exhale_stages=3)
        ctx = self._make_ctx()
        result = await bp.execute(ctx)

        # Stages 0-3 should see "inhale"
        for i in range(4):
            assert executors[i].phases_seen == ["inhale"], f"stage_{i}"

        # Stages 4-6 should see "exhale"
        for i in range(4, 7):
            assert executors[i].phases_seen == ["exhale"], f"stage_{i}"

        # Stage 7 should see "rest"
        assert executors[7].phases_seen == ["rest"]

        # Phase cleared after execution
        assert result.breathing_phase is None

    @pytest.mark.asyncio
    async def test_degrades_without_chrono(self):
        from hololoom.core.orchestrator.breathing_pipeline import BreathingPipeline

        pipeline = MagicMock()
        pipeline.execute = AsyncMock(return_value="result")

        bp = BreathingPipeline(pipeline, chrono=None)
        ctx = self._make_ctx()
        result = await bp.execute(ctx)

        # Should fall through to plain pipeline
        pipeline.execute.assert_called_once_with(ctx)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_breath_count_increments(self):
        from hololoom.core.orchestrator.breathing_pipeline import BreathingPipeline

        pipeline, _ = self._make_pipeline(8)
        chrono = MagicMock()
        chrono.current_phase = None

        bp = BreathingPipeline(pipeline, chrono=chrono)
        assert bp.breath_count == 0

        await bp.execute(self._make_ctx())
        assert bp.breath_count == 1

        await bp.execute(self._make_ctx())
        assert bp.breath_count == 2

    @pytest.mark.asyncio
    async def test_rest_calls_consolidation(self):
        from hololoom.core.orchestrator.breathing_pipeline import BreathingPipeline

        pipeline, _ = self._make_pipeline(8)
        chrono = MagicMock()
        chrono.current_phase = None

        consolidator = MagicMock()
        consolidator.run_cycle = AsyncMock(return_value={"promoted": 1})
        memory_layer = MagicMock()
        memory_layer._consolidator = consolidator

        bp = BreathingPipeline(pipeline, chrono=chrono, memory_layer=memory_layer)
        await bp.execute(self._make_ctx())

        consolidator.run_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidation_failure_non_critical(self):
        from hololoom.core.orchestrator.breathing_pipeline import BreathingPipeline

        pipeline, _ = self._make_pipeline(8)
        chrono = MagicMock()
        chrono.current_phase = None

        consolidator = MagicMock()
        consolidator.run_cycle = AsyncMock(side_effect=RuntimeError("boom"))
        memory_layer = MagicMock()
        memory_layer._consolidator = consolidator

        bp = BreathingPipeline(pipeline, chrono=chrono, memory_layer=memory_layer)
        # Should not raise
        await bp.execute(self._make_ctx())

    @pytest.mark.asyncio
    async def test_chrono_phase_set_and_cleared(self):
        from hololoom.core.orchestrator.breathing_pipeline import BreathingPipeline

        pipeline, _ = self._make_pipeline(8)
        chrono = MagicMock()
        chrono.current_phase = None

        bp = BreathingPipeline(pipeline, chrono=chrono)
        await bp.execute(self._make_ctx())

        # After execution, chrono phase should be cleared
        assert chrono.current_phase is None

    def test_pipeline_property(self):
        from hololoom.core.orchestrator.breathing_pipeline import BreathingPipeline

        pipeline = MagicMock()
        bp = BreathingPipeline(pipeline)
        assert bp.pipeline is pipeline
