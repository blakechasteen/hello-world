"""Tests for MemoryLayer — decay, consolidation, vault promotion."""

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hololoom.core.runtime.layer import LayerState
from hololoom.core.runtime.memory_layer import MemoryLayer


# ============================================================================
# Fixtures
# ============================================================================

@dataclass
class FakeStoredItem:
    id: str
    content: str
    importance: float = 0.5
    access_count: int = 0
    memory_type: str = "episodic"

    def is_expired(self):
        return False


class FakeMemoryBus:
    """Minimal mock of LiteMemoryBus for testing."""

    def __init__(self, items=None):
        self._items = {i.id: i for i in (items or [])}

    async def decay(self, max_age_seconds=86400, min_importance=0.1):
        return 0  # No items forgotten in tests by default


@pytest.fixture
def empty_bus():
    return FakeMemoryBus()


@pytest.fixture
def rich_bus():
    """Bus with items at various quality levels."""
    return FakeMemoryBus([
        FakeStoredItem(id="high1", content="A" * 60, importance=0.9, access_count=10),
        FakeStoredItem(id="high2", content="B" * 60, importance=0.8, access_count=7),
        FakeStoredItem(id="low1", content="C" * 60, importance=0.3, access_count=1),
        FakeStoredItem(id="short", content="Too short", importance=0.9, access_count=10),
    ])


@pytest.fixture
def mock_signal():
    sig = MagicMock()
    sig.target = ""
    sig.payload = {"tick": 1}
    return sig


@pytest.fixture
def mock_runtime(empty_bus):
    rt = MagicMock()
    rt.bus = MagicMock()
    rt.bus.on = MagicMock()
    rt.memory_bus = empty_bus
    return rt


@pytest.fixture
def mock_runtime_no_bus():
    rt = MagicMock()
    rt.bus = None
    rt.memory_bus = None
    return rt


# ============================================================================
# Lifecycle
# ============================================================================

class TestMemoryLayerLifecycle:
    @pytest.mark.asyncio
    async def test_starts_with_bus(self, mock_runtime, empty_bus):
        layer = MemoryLayer(memory_bus=empty_bus)
        await layer.start(mock_runtime)
        assert layer.state == LayerState.RUNNING

    @pytest.mark.asyncio
    async def test_degrades_without_bus(self, mock_runtime_no_bus):
        layer = MemoryLayer()
        await layer.start(mock_runtime_no_bus)
        assert layer.state == LayerState.DEGRADED

    @pytest.mark.asyncio
    async def test_late_wires_from_runtime(self, mock_runtime, empty_bus):
        layer = MemoryLayer()  # No bus provided
        mock_runtime.memory_bus = empty_bus
        await layer.start(mock_runtime)
        assert layer.state == LayerState.RUNNING

    @pytest.mark.asyncio
    async def test_subscribes_to_signals(self, mock_runtime, empty_bus):
        layer = MemoryLayer(memory_bus=empty_bus)
        await layer.start(mock_runtime)
        # Should subscribe to HEARTBEAT and CRON
        assert mock_runtime.bus.on.call_count == 2

    def test_name(self):
        assert MemoryLayer.name == "memory"


# ============================================================================
# Decay
# ============================================================================

class TestDecay:
    @pytest.mark.asyncio
    async def test_heartbeat_triggers_decay(self, mock_signal):
        bus = FakeMemoryBus()
        bus.decay = AsyncMock(return_value=3)
        layer = MemoryLayer(memory_bus=bus)
        layer._state = LayerState.RUNNING

        await layer._handle_heartbeat(mock_signal)

        bus.decay.assert_called_once_with(max_age_seconds=86400, min_importance=0.1)
        assert layer._stats["decayed"] == 3

    @pytest.mark.asyncio
    async def test_decay_survives_error(self, mock_signal):
        bus = FakeMemoryBus()
        bus.decay = AsyncMock(side_effect=RuntimeError("boom"))
        layer = MemoryLayer(memory_bus=bus)
        layer._state = LayerState.RUNNING

        # Should not raise
        await layer._handle_heartbeat(mock_signal)


# ============================================================================
# Consolidation
# ============================================================================

class TestConsolidation:
    @pytest.mark.asyncio
    async def test_cron_triggers_consolidation(self, mock_signal):
        mock_signal.target = "sleep_consolidation"
        consolidator = MagicMock()
        consolidator.run_cycle = AsyncMock(return_value={"promoted": 2})

        layer = MemoryLayer(memory_bus=FakeMemoryBus(), consolidator=consolidator)
        layer._state = LayerState.RUNNING

        await layer._handle_cron(mock_signal)

        consolidator.run_cycle.assert_called_once()
        assert layer._stats["consolidated"] == 2

    @pytest.mark.asyncio
    async def test_ignores_unrelated_cron(self, mock_signal):
        mock_signal.target = "cosmic_check"
        consolidator = MagicMock()
        consolidator.run_cycle = AsyncMock()

        layer = MemoryLayer(memory_bus=FakeMemoryBus(), consolidator=consolidator)
        layer._state = LayerState.RUNNING

        await layer._handle_cron(mock_signal)

        consolidator.run_cycle.assert_not_called()


# ============================================================================
# Vault Promotion
# ============================================================================

class TestVaultPromotion:
    @pytest.mark.asyncio
    async def test_promotes_high_value_items(self, mock_signal, rich_bus):
        mock_signal.target = "vault_promotion"
        layer = MemoryLayer(memory_bus=rich_bus)
        layer._state = LayerState.RUNNING

        with patch("hololoom.core.runtime.memory_layer.propose_note",
                   return_value={"status": "proposed"}, create=True):
            # Patch the lazy import inside _promote_to_vault
            with patch.dict("sys.modules", {
                "hololoom.apps.server.vault_bridge": MagicMock(
                    propose_note=MagicMock(return_value={"status": "proposed"})
                )
            }):
                await layer._handle_cron(mock_signal)

        # high1 and high2 qualify (importance >= 0.7, access >= 5, content >= 50)
        # short doesn't qualify (content < 50)
        # low1 doesn't qualify (importance < 0.7)
        assert layer._stats["promoted"] == 2
        assert "high1" in layer._promoted_ids
        assert "high2" in layer._promoted_ids

    @pytest.mark.asyncio
    async def test_dedup_prevents_double_promotion(self, mock_signal, rich_bus):
        mock_signal.target = "vault_promotion"
        layer = MemoryLayer(memory_bus=rich_bus)
        layer._state = LayerState.RUNNING
        layer._promoted_ids.add("high1")  # Already promoted

        with patch.dict("sys.modules", {
            "hololoom.apps.server.vault_bridge": MagicMock(
                propose_note=MagicMock(return_value={"status": "proposed"})
            )
        }):
            await layer._handle_cron(mock_signal)

        # Only high2 should be promoted (high1 already in set)
        assert layer._stats["promoted"] == 1
        assert "high2" in layer._promoted_ids

    @pytest.mark.asyncio
    async def test_respects_max_per_cycle(self, mock_signal):
        items = [
            FakeStoredItem(id=f"item{i}", content="X" * 60, importance=0.9, access_count=10)
            for i in range(10)
        ]
        bus = FakeMemoryBus(items)
        layer = MemoryLayer(memory_bus=bus, max_promotions_per_cycle=2)
        layer._state = LayerState.RUNNING

        mock_signal.target = "vault_promotion"
        with patch.dict("sys.modules", {
            "hololoom.apps.server.vault_bridge": MagicMock(
                propose_note=MagicMock(return_value={"status": "proposed"})
            )
        }):
            await layer._handle_cron(mock_signal)

        assert layer._stats["promoted"] == 2

    @pytest.mark.asyncio
    async def test_degrades_without_vault_bridge(self, mock_signal, rich_bus):
        mock_signal.target = "vault_promotion"
        layer = MemoryLayer(memory_bus=rich_bus)
        layer._state = LayerState.RUNNING

        # vault_bridge not importable
        with patch.dict("sys.modules", {"hololoom.apps.server.vault_bridge": None}):
            # Should not raise
            await layer._handle_cron(mock_signal)

        # No promotions — ImportError handled
        assert layer._stats["promoted"] == 0


# ============================================================================
# Status
# ============================================================================

class TestStatus:
    @pytest.mark.asyncio
    async def test_status_reports_bus_items(self):
        bus = FakeMemoryBus([FakeStoredItem(id="a", content="x")])
        layer = MemoryLayer(memory_bus=bus)
        layer._state = LayerState.RUNNING

        status = layer.status()
        assert status["bus_items"] == 1
        assert status["state"] == "running"

    @pytest.mark.asyncio
    async def test_status_without_bus(self):
        layer = MemoryLayer()
        layer._state = LayerState.DEGRADED

        status = layer.status()
        assert status["bus_items"] == 0
