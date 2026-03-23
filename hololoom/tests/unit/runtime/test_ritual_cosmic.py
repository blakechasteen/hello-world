"""Tests for RitualLayer cosmic scheduling expansion."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hololoom.core.runtime.layer import LayerState
from hololoom.core.runtime.ritual_layer import RitualLayer


@pytest.fixture
def mock_runtime():
    rt = MagicMock()
    rt.bus = MagicMock()
    rt.bus.on = MagicMock()
    rt.journal = None
    return rt


class TestCosmicHandler:
    @pytest.mark.asyncio
    async def test_ignores_non_cosmic_signals(self, mock_runtime):
        layer = RitualLayer(register_defaults=False)
        await layer.start(mock_runtime)

        signal = MagicMock()
        signal.target = "memory_decay"  # Not cosmic

        # Should return without doing anything
        await layer._handle_cosmic(signal)
        assert layer._cosmic_events_fired == []

    @pytest.mark.asyncio
    async def test_handles_cosmic_check(self, mock_runtime):
        layer = RitualLayer(register_defaults=False)
        await layer.start(mock_runtime)

        signal = MagicMock()
        signal.target = "cosmic_check"

        # Mock the RitualScheduler
        mock_ritual = MagicMock()
        mock_ritual.ritual_type.value = "full_moon"

        with patch.dict("sys.modules", {
            "hololoom.domain_harness.strands.ritual_scheduler": MagicMock(
                RitualScheduler=MagicMock(return_value=MagicMock(
                    check_and_execute=MagicMock(return_value=[mock_ritual])
                ))
            )
        }):
            await layer._handle_cosmic(signal)

        assert "full_moon" in layer._cosmic_events_fired

    @pytest.mark.asyncio
    async def test_degrades_without_domain_harness(self, mock_runtime):
        layer = RitualLayer(register_defaults=False)
        await layer.start(mock_runtime)

        signal = MagicMock()
        signal.target = "cosmic_check"

        # ImportError should be caught silently
        with patch.dict("sys.modules", {
            "hololoom.domain_harness.strands.ritual_scheduler": None,
        }):
            await layer._handle_cosmic(signal)

        assert layer._cosmic_events_fired == []


class TestTensionModifiers:
    def test_full_moon_increases_breathing_rate(self):
        layer = RitualLayer(register_defaults=False)

        # Set up a mock runtime with chrono
        runtime = MagicMock()
        chrono = MagicMock()
        chrono.breathing.breathing_rate = 1.0
        runtime._chrono = chrono
        layer._runtime = runtime

        layer._apply_tension_modifier("full_moon")

        assert chrono.breathing.breathing_rate == 1.2

    def test_new_moon_decreases_breathing_rate(self):
        layer = RitualLayer(register_defaults=False)

        runtime = MagicMock()
        chrono = MagicMock()
        chrono.breathing.breathing_rate = 1.0
        runtime._chrono = chrono
        layer._runtime = runtime

        layer._apply_tension_modifier("new_moon")

        assert chrono.breathing.breathing_rate == pytest.approx(0.8)

    def test_no_modifier_for_unknown_event(self):
        layer = RitualLayer(register_defaults=False)

        runtime = MagicMock()
        chrono = MagicMock()
        chrono.breathing.breathing_rate = 1.0
        runtime._chrono = chrono
        layer._runtime = runtime

        layer._apply_tension_modifier("unknown_event")

        # Should be unchanged
        assert chrono.breathing.breathing_rate == 1.0

    def test_no_chrono_no_crash(self):
        layer = RitualLayer(register_defaults=False)
        layer._runtime = MagicMock()
        layer._runtime._chrono = None

        # Should not raise
        layer._apply_tension_modifier("full_moon")

    def test_no_runtime_no_crash(self):
        layer = RitualLayer(register_defaults=False)
        layer._runtime = None

        # Should not raise
        layer._apply_tension_modifier("full_moon")


class TestCosmicInStatus:
    @pytest.mark.asyncio
    async def test_status_includes_cosmic_events(self, mock_runtime):
        layer = RitualLayer(register_defaults=False)
        await layer.start(mock_runtime)

        layer._cosmic_events_fired.append("full_moon")
        status = layer.status()

        assert "cosmic_events" in status
        assert "full_moon" in status["cosmic_events"]

    @pytest.mark.asyncio
    async def test_cosmic_disabled(self, mock_runtime):
        layer = RitualLayer(register_defaults=False, enable_cosmic=False)
        await layer.start(mock_runtime)

        assert layer.state == LayerState.RUNNING
        # Cron subscription for cosmic should not be registered
        # (bus.on is called for RitualTrigger but not for cosmic)
