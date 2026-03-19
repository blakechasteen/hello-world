"""Tests for HoloLoomRuntime and RitualLayer."""

import pytest

from hololoom.core.runtime import HoloLoomRuntime, LayerState
from hololoom.core.runtime.ritual_layer import RitualLayer
from hololoom.core.ritual.registry import RitualRegistry
from hololoom.core.ritual.journal import RitualJournal
from hololoom.core.ritual.types import (
    Ritual, RitualBudget, RitualResult, HookBinding, Priority,
)
from hololoom.core.ritual.hooks import HookEvent
from hololoom.core.ritual.policies import FailurePolicy


def _make_ritual(ritual_id="test_ritual"):
    async def body(ctx):
        return RitualResult(success=True, produced={}, confidence_delta=0.0)

    return Ritual(
        id=ritual_id, version="1.0.0", domain="coding", intent="test",
        author="system", consumes=[], produces=[],
        failure_policy=FailurePolicy.SKIP,
        budget=RitualBudget(max_tokens=100, max_seconds=5.0, priority_class="standard"),
        body=body,
    )


class TestHoloLoomRuntime:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        rt = HoloLoomRuntime()
        await rt.start()
        assert rt.status()["started"] is True
        await rt.stop()
        assert rt.status()["started"] is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with HoloLoomRuntime() as rt:
            assert rt.status()["started"] is True
        # after exit, stopped

    @pytest.mark.asyncio
    async def test_register_and_start_layer(self):
        rt = HoloLoomRuntime()
        layer = RitualLayer(register_defaults=False)
        rt.register_layer(layer)
        await rt.start()

        status = rt.status()
        assert "ritual" in status["layers"]
        assert status["layers"]["ritual"]["state"] == "running"

        await rt.stop()

    @pytest.mark.asyncio
    async def test_failed_layer_degrades(self):
        class FailingLayer:
            name = "failing"
            async def start(self, runtime):
                raise RuntimeError("intentional failure")
            async def stop(self):
                pass
            def status(self):
                return {}
            def state_snapshot(self):
                return {}

        rt = HoloLoomRuntime()
        rt.register_layer(FailingLayer())
        await rt.start()

        status = rt.status()
        assert status["layers"]["failing"]["state"] == "degraded"
        await rt.stop()

    @pytest.mark.asyncio
    async def test_multiple_layers(self):
        class SimpleLayer:
            def __init__(self, n):
                self.name = f"layer_{n}"
                self._state = "init"
            async def start(self, runtime):
                self._state = "running"
            async def stop(self):
                self._state = "stopped"
            def status(self):
                return {"custom": self._state}
            def state_snapshot(self):
                return {}

        rt = HoloLoomRuntime()
        rt.register_layer(SimpleLayer(1))
        rt.register_layer(SimpleLayer(2))
        await rt.start()

        status = rt.status()
        assert len(status["layers"]) == 2
        await rt.stop()

    @pytest.mark.asyncio
    async def test_get_layer(self):
        rt = HoloLoomRuntime()
        layer = RitualLayer(register_defaults=False)
        rt.register_layer(layer)
        assert rt.get_layer("ritual") is layer
        assert rt.get_layer("nonexistent") is None


class TestRitualLayer:
    @pytest.mark.asyncio
    async def test_starts_running(self):
        layer = RitualLayer(register_defaults=False)
        rt = HoloLoomRuntime()
        await layer.start(rt)
        assert layer.state == LayerState.RUNNING

    @pytest.mark.asyncio
    async def test_registers_defaults(self):
        layer = RitualLayer(register_defaults=True)
        rt = HoloLoomRuntime()
        await layer.start(rt)
        # Should have at least pre_code_health_check, dawn, dusk, memory_guard, health_audit
        assert layer.status()["registered_rituals"] >= 4

    @pytest.mark.asyncio
    async def test_custom_bindings(self):
        ritual = _make_ritual("custom_ritual")
        binding = HookBinding(
            ritual=ritual, trigger=HookEvent.TASK_RECEIVED,
            condition=None, priority=Priority.BLOCKING,
        )
        layer = RitualLayer(
            register_defaults=False,
            bindings=[binding],
        )
        rt = HoloLoomRuntime()
        await layer.start(rt)
        assert layer.status()["registered_rituals"] == 1

    @pytest.mark.asyncio
    async def test_no_bus_still_works(self):
        """Without a bus, the layer runs but has no trigger."""
        layer = RitualLayer(register_defaults=False)
        rt = HoloLoomRuntime(bus=None)
        await layer.start(rt)
        assert layer.state == LayerState.RUNNING
        assert layer.status()["trigger_active"] is False

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        journal = RitualJournal(":memory:")
        layer = RitualLayer(register_defaults=False, journal=journal)
        rt = HoloLoomRuntime()
        await layer.start(rt)
        await layer.stop()
        assert layer.state == LayerState.STOPPED

    @pytest.mark.asyncio
    async def test_registry_accessible(self):
        layer = RitualLayer(register_defaults=False)
        rt = HoloLoomRuntime()
        await layer.start(rt)
        assert isinstance(layer.registry, RitualRegistry)
