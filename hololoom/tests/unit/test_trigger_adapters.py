"""
Tests for Trigger Adapters — Phase 4.

Tests HeartbeatTrigger and CronTrigger via manual pulse/tick methods
(no real timers needed).
"""

import asyncio
import pytest

from hololoom.core.bus import UnifiedBus, Signal, SignalKind
from hololoom.core.bus.adapters.trigger_adapters import (
    HeartbeatTrigger,
    CronTrigger,
)


class TestHeartbeatTrigger:
    @pytest.fixture
    def bus(self):
        return UnifiedBus()

    @pytest.mark.asyncio
    async def test_manual_pulse(self, bus):
        heartbeat = HeartbeatTrigger(bus)
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.HEARTBEAT, handler)

        await heartbeat.pulse()
        await asyncio.sleep(0.02)

        assert len(received) == 1
        assert received[0].payload["tick"] == 1
        assert received[0].payload["manual"] is True
        assert received[0].trigger_mode.value == "heartbeat"

    @pytest.mark.asyncio
    async def test_tick_increments(self, bus):
        heartbeat = HeartbeatTrigger(bus)

        await heartbeat.pulse()
        await heartbeat.pulse()
        await heartbeat.pulse()

        assert heartbeat.tick_count == 3

    @pytest.mark.asyncio
    async def test_health_fn_included(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.HEARTBEAT, handler)

        def health():
            return {"cpu": 0.3, "memory": 0.6}

        heartbeat = HeartbeatTrigger(bus, health_fn=health)
        await heartbeat.pulse()
        await asyncio.sleep(0.02)

        assert received[0].payload["health"]["cpu"] == 0.3

    @pytest.mark.asyncio
    async def test_health_fn_error_captured(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.HEARTBEAT, handler)

        def bad_health():
            raise RuntimeError("sensor failure")

        heartbeat = HeartbeatTrigger(bus, health_fn=bad_health)
        await heartbeat.pulse()
        await asyncio.sleep(0.02)

        assert "health_error" in received[0].payload

    @pytest.mark.asyncio
    async def test_start_stop(self, bus):
        heartbeat = HeartbeatTrigger(bus, interval=0.1)
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.HEARTBEAT, handler)

        await heartbeat.start()
        await asyncio.sleep(0.35)
        await heartbeat.stop()

        # Should have at least 2 heartbeats in 0.35s with 0.1s interval
        assert len(received) >= 2


class TestCronTrigger:
    @pytest.fixture
    def bus(self):
        return UnifiedBus()

    @pytest.mark.asyncio
    async def test_register_and_tick(self, bus):
        cron = CronTrigger(bus)
        cron.every(0, "test_job", {"action": "test"})  # interval=0 means always due

        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.CRON, handler)

        await cron.tick()
        await asyncio.sleep(0.02)

        assert len(received) == 1
        assert received[0].payload["job_name"] == "test_job"
        assert received[0].payload["action"] == "test"
        assert received[0].payload["run_count"] == 1
        assert received[0].trigger_mode.value == "cron"

    @pytest.mark.asyncio
    async def test_job_not_due(self, bus):
        cron = CronTrigger(bus)
        cron.every(9999, "slow_job", {})  # Very long interval

        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.CRON, handler)

        # First tick always fires (last_run=0)
        await cron.tick()
        await asyncio.sleep(0.02)

        count_after_first = len(received)

        # Second tick should NOT fire (interval not elapsed)
        await cron.tick()
        await asyncio.sleep(0.02)

        assert len(received) == count_after_first  # No new signals

    @pytest.mark.asyncio
    async def test_cron_target_routing(self, bus):
        """Cron signals have target=job_name for targeted routing."""
        cron = CronTrigger(bus)
        cron.every(0, "consolidate_memory", {"action": "consolidate"})

        targeted = []

        async def handler(s: Signal):
            targeted.append(s)
            return None

        bus.subscribe("consolidate_memory", handler)

        await cron.tick()
        await asyncio.sleep(0.02)

        assert len(targeted) == 1

    @pytest.mark.asyncio
    async def test_remove_job(self, bus):
        cron = CronTrigger(bus)
        cron.every(0, "temp_job", {})

        assert cron.remove("temp_job") is True
        assert cron.remove("nonexistent") is False

        jobs = cron.get_jobs()
        assert "temp_job" not in jobs

    @pytest.mark.asyncio
    async def test_get_jobs(self, bus):
        cron = CronTrigger(bus)
        cron.every(300, "memory_consolidation", {"action": "consolidate"})
        cron.every(3600, "mcts_prune", {"min_visits": 2})

        jobs = cron.get_jobs()
        assert "memory_consolidation" in jobs
        assert jobs["memory_consolidation"]["interval"] == 300
        assert "mcts_prune" in jobs

    @pytest.mark.asyncio
    async def test_multiple_jobs_fire(self, bus):
        cron = CronTrigger(bus)
        cron.every(0, "job_a", {"a": True})
        cron.every(0, "job_b", {"b": True})

        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.CRON, handler)

        await cron.tick()
        await asyncio.sleep(0.02)

        assert len(received) == 2
        names = {r.payload["job_name"] for r in received}
        assert names == {"job_a", "job_b"}
