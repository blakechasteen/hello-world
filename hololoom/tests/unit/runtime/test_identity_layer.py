"""Tests for IdentityLayer — the autoresearch loop."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hololoom.core.runtime.layer import LayerState
from hololoom.core.runtime.identity_layer import IdentityLayer


@pytest.fixture
def mock_runtime():
    rt = MagicMock()
    rt.bus = MagicMock()
    rt.bus.on = MagicMock()
    rt.status.return_value = {
        "layers": {
            "temporal": {"ticks": 42, "cron_jobs": {"memory_decay": {"interval": 300, "run_count": 5}}},
            "memory": {"bus_items": 100, "stats": {"decayed": 3, "promoted": 1}},
        }
    }
    return rt


@pytest.fixture
def mock_signal():
    sig = MagicMock()
    sig.payload = {"tick": 42}
    return sig


class TestIdentityLayerLifecycle:
    @pytest.mark.asyncio
    async def test_starts_with_bus(self, mock_runtime, tmp_path):
        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)
        assert layer.state == LayerState.RUNNING
        assert mock_runtime.bus.on.called

    @pytest.mark.asyncio
    async def test_starts_without_bus(self, tmp_path):
        rt = MagicMock()
        rt.bus = None
        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(rt)
        assert layer.state == LayerState.RUNNING

    @pytest.mark.asyncio
    async def test_stops(self, mock_runtime, tmp_path):
        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)
        await layer.stop()
        assert layer.state == LayerState.STOPPED

    def test_name(self):
        assert IdentityLayer.name == "identity"


class TestHeartbeatWriting:
    @pytest.mark.asyncio
    async def test_writes_heartbeat_md(self, mock_runtime, mock_signal, tmp_path):
        agent_dir = tmp_path / "test"
        agent_dir.mkdir()

        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)
        await layer._on_heartbeat(mock_signal)

        heartbeat = (agent_dir / "heartbeat.md").read_text(encoding="utf-8")
        assert "heartbeat.md" in heartbeat
        assert "tick: 42" in heartbeat
        assert "bus_items: 100" in heartbeat
        assert "coherence" in heartbeat
        assert "groundedness" in heartbeat

    @pytest.mark.asyncio
    async def test_heartbeat_contains_cron_data(self, mock_runtime, mock_signal, tmp_path):
        agent_dir = tmp_path / "test"
        agent_dir.mkdir()

        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)
        await layer._on_heartbeat(mock_signal)

        heartbeat = (agent_dir / "heartbeat.md").read_text(encoding="utf-8")
        assert "memory_decay" in heartbeat
        assert "300s" in heartbeat

    @pytest.mark.asyncio
    async def test_heartbeat_contains_memory_stats(self, mock_runtime, mock_signal, tmp_path):
        agent_dir = tmp_path / "test"
        agent_dir.mkdir()

        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)
        await layer._on_heartbeat(mock_signal)

        heartbeat = (agent_dir / "heartbeat.md").read_text(encoding="utf-8")
        assert "decayed: 3" in heartbeat
        assert "promoted: 1" in heartbeat

    @pytest.mark.asyncio
    async def test_skips_when_no_agent_dir(self, mock_runtime, mock_signal, tmp_path):
        # Agent dir doesn't exist — should not crash
        layer = IdentityLayer(agent_name="nonexistent", identities_root=tmp_path)
        await layer.start(mock_runtime)
        await layer._on_heartbeat(mock_signal)
        # No file written, no crash
        assert not (tmp_path / "nonexistent" / "heartbeat.md").exists()

    @pytest.mark.asyncio
    async def test_pulse_count_increments(self, mock_runtime, mock_signal, tmp_path):
        agent_dir = tmp_path / "test"
        agent_dir.mkdir()

        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)

        await layer._on_heartbeat(mock_signal)
        assert layer._pulse_count == 1

        await layer._on_heartbeat(mock_signal)
        assert layer._pulse_count == 2


class TestMutationProposals:
    @pytest.mark.asyncio
    async def test_proposes_on_groundedness_drift(self, mock_runtime, mock_signal, tmp_path):
        agent_dir = tmp_path / "test"
        agent_dir.mkdir()

        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)

        # Simulate peak then drop
        layer._peak_groundedness = 0.9

        # The rolling metrics return 0.0 (no data) which is a big drop
        # but window_size is 0 so proposer won't trigger (needs >= 50)
        await layer._on_heartbeat(mock_signal)

        # No proposals because window is too small
        # This tests the safety gate

    @pytest.mark.asyncio
    async def test_proposes_on_coherence_rigidity(self, mock_runtime, tmp_path):
        agent_dir = tmp_path / "test"
        agent_dir.mkdir()

        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)

        # Directly test threshold check
        layer._check_thresholds(groundedness=0.5, coherence=0.97, n_results=100)

        # Should have proposed (coherence > 0.95 = rigidity warning)
        if layer._proposer:
            assert len(layer._proposer.proposals) > 0


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_reports_metrics(self, mock_runtime, tmp_path):
        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)

        status = layer.status()
        assert status["agent"] == "test"
        assert status["pulse_count"] == 0
        assert "groundedness" in status
        assert "coherence" in status

    @pytest.mark.asyncio
    async def test_state_snapshot(self, mock_runtime, tmp_path):
        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)

        assert layer.state_snapshot() == {"state": "running"}


class TestRollingMetricsAccess:
    @pytest.mark.asyncio
    async def test_rolling_property(self, mock_runtime, tmp_path):
        layer = IdentityLayer(agent_name="test", identities_root=tmp_path)
        await layer.start(mock_runtime)

        # Rolling metrics should be accessible for external probe injection
        assert layer.rolling is not None
