"""Tests for ChronoTrigger decay and REST wiring (replacing stubs)."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class FakeConfig:
    pipeline_timeout = 5.0


class TestApplyDecay:
    @pytest.mark.asyncio
    async def test_decay_calls_memory_bus(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        bus = MagicMock()
        bus.decay = AsyncMock(return_value=5)

        ct = ChronoTrigger(FakeConfig(), memory_bus=bus)
        await ct._apply_decay()

        bus.decay.assert_called_once_with(max_age_seconds=86400, min_importance=0.1)

    @pytest.mark.asyncio
    async def test_decay_survives_missing_bus(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        ct = ChronoTrigger(FakeConfig(), memory_bus=None)
        # Should not raise
        await ct._apply_decay()

    @pytest.mark.asyncio
    async def test_decay_survives_bus_error(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        bus = MagicMock()
        bus.decay = AsyncMock(side_effect=RuntimeError("db locked"))

        ct = ChronoTrigger(FakeConfig(), memory_bus=bus)
        # Should not raise
        await ct._apply_decay()


class TestRestPhase:
    @pytest.mark.asyncio
    async def test_rest_calls_consolidator(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        consolidator = MagicMock()
        consolidator.run_cycle = AsyncMock(return_value={"promoted": 2})

        ct = ChronoTrigger(FakeConfig(), consolidator=consolidator)
        ct.breathing.rest_duration = 0.0  # Skip sleep for test speed
        metrics = await ct._rest()

        consolidator.run_cycle.assert_called_once()
        assert metrics["consolidation"]["promoted"] == 2

    @pytest.mark.asyncio
    async def test_rest_without_consolidator(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        ct = ChronoTrigger(FakeConfig())
        ct.breathing.rest_duration = 0.0
        metrics = await ct._rest()

        assert metrics["consolidation"] == {"skipped": True}

    @pytest.mark.asyncio
    async def test_rest_consolidation_failure_non_critical(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        consolidator = MagicMock()
        consolidator.run_cycle = AsyncMock(side_effect=RuntimeError("boom"))

        ct = ChronoTrigger(FakeConfig(), consolidator=consolidator)
        ct.breathing.rest_duration = 0.0
        # Should not raise
        metrics = await ct._rest()
        assert metrics["phase"] == "rest"

    @pytest.mark.asyncio
    async def test_rest_applies_mini_decay(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        bus = MagicMock()
        bus.decay = AsyncMock(return_value=0)

        ct = ChronoTrigger(FakeConfig(), memory_bus=bus)
        ct.breathing.rest_duration = 0.0
        await ct._rest()

        # Mini-decay should have been called with gentler params
        bus.decay.assert_called_once_with(max_age_seconds=172800, min_importance=0.05)


class TestInitParams:
    def test_memory_bus_param(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        bus = MagicMock()
        ct = ChronoTrigger(FakeConfig(), memory_bus=bus)
        assert ct._memory_bus is bus

    def test_consolidator_param(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        consolidator = MagicMock()
        ct = ChronoTrigger(FakeConfig(), consolidator=consolidator)
        assert ct._consolidator is consolidator

    def test_defaults_to_none(self):
        from hololoom.core.chrono.trigger import ChronoTrigger

        ct = ChronoTrigger(FakeConfig())
        assert ct._memory_bus is None
        assert ct._consolidator is None
