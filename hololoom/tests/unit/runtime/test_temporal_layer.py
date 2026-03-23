"""Tests for TemporalLayer — the clock."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hololoom.core.runtime.layer import LayerState
from hololoom.core.runtime.temporal_layer import TemporalLayer


@pytest.fixture
def mock_bus():
    bus = MagicMock()
    bus.emit = AsyncMock()
    bus.on = MagicMock()
    return bus


@pytest.fixture
def mock_runtime(mock_bus):
    rt = MagicMock()
    rt.bus = mock_bus
    return rt


@pytest.fixture
def mock_runtime_no_bus():
    rt = MagicMock()
    rt.bus = None
    return rt


class TestTemporalLayerLifecycle:
    @pytest.mark.asyncio
    async def test_starts_with_bus(self, mock_runtime):
        layer = TemporalLayer(heartbeat_interval=60.0, cron_check_interval=10.0)
        await layer.start(mock_runtime)

        assert layer.state == LayerState.RUNNING
        assert layer.heartbeat is not None
        assert layer.cron is not None

    @pytest.mark.asyncio
    async def test_degrades_without_bus(self, mock_runtime_no_bus):
        layer = TemporalLayer()
        await layer.start(mock_runtime_no_bus)

        assert layer.state == LayerState.DEGRADED
        assert layer.heartbeat is None
        assert layer.cron is None

    @pytest.mark.asyncio
    async def test_stops_cleanly(self, mock_runtime):
        layer = TemporalLayer()
        await layer.start(mock_runtime)
        await layer.stop()

        assert layer.state == LayerState.STOPPED

    @pytest.mark.asyncio
    async def test_stops_without_start(self):
        layer = TemporalLayer()
        await layer.stop()
        assert layer.state == LayerState.STOPPED


class TestCronJobRegistration:
    @pytest.mark.asyncio
    async def test_standard_jobs_registered(self, mock_runtime):
        layer = TemporalLayer()
        await layer.start(mock_runtime)

        jobs = layer.cron.get_jobs()
        assert "memory_decay" in jobs
        assert "sleep_consolidation" in jobs
        assert "vault_promotion" in jobs
        assert "hot_pattern_decay" in jobs
        assert "cosmic_check" in jobs

    @pytest.mark.asyncio
    async def test_job_intervals(self, mock_runtime):
        layer = TemporalLayer()
        await layer.start(mock_runtime)

        jobs = layer.cron.get_jobs()
        assert jobs["memory_decay"]["interval"] == 300
        assert jobs["sleep_consolidation"]["interval"] == 1800
        assert jobs["vault_promotion"]["interval"] == 300
        assert jobs["cosmic_check"]["interval"] == 3600


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_reports_ticks(self, mock_runtime):
        layer = TemporalLayer()
        await layer.start(mock_runtime)

        status = layer.status()
        assert "ticks" in status
        assert "cron_jobs" in status
        assert status["state"] == "running"

    @pytest.mark.asyncio
    async def test_status_degraded(self, mock_runtime_no_bus):
        layer = TemporalLayer()
        await layer.start(mock_runtime_no_bus)

        status = layer.status()
        assert status["state"] == "degraded"

    @pytest.mark.asyncio
    async def test_state_snapshot(self, mock_runtime):
        layer = TemporalLayer()
        await layer.start(mock_runtime)

        snap = layer.state_snapshot()
        assert snap == {"state": "running"}

    def test_name(self):
        assert TemporalLayer.name == "temporal"
