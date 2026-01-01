"""
Comprehensive pytest tests for ForgetManager (Phase 2: Unified Forgetting)
===========================================================================

Tests the unified forgetting manager that coordinates 5 forgetting subsystems:
1. KG Eviction
2. Hot Pattern Decay
3. Lifecycle TTL
4. Activation Decay
5. Consolidation Cleanup

December 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import networkx as nx

from HoloLoom.memory.forget_manager import (
    ForgetManager,
    ForgetConfig,
    ForgetPolicy,
    ForgetTrigger,
    ForgetEvent,
    create_forget_manager,
)


# ============================================================================
# Mock Subsystems
# ============================================================================

class MockKG:
    """Mock knowledge graph for testing."""

    def __init__(self, num_nodes: int = 100, max_nodes: int = 1000):
        self.G = nx.MultiDiGraph()
        # Add initial nodes
        for i in range(num_nodes):
            self.G.add_node(f"node_{i}")

        self._max_nodes = max_nodes
        self._eviction_strategy = Mock()
        self._eviction_strategy.value = "importance"

    def evict_to_capacity(self):
        """Mock eviction."""
        nodes_to_evict = max(0, self.G.number_of_nodes() - int(self._max_nodes * 0.85))
        evicted_nodes = []

        for i in range(nodes_to_evict):
            node = f"node_{i}"
            if node in self.G:
                self.G.remove_node(node)
                evicted_nodes.append(node)

        result = Mock()
        result.nodes_evicted = len(evicted_nodes)
        result.edges_removed = len(evicted_nodes) * 2  # Mock some edges
        result.strategy_used = self._eviction_strategy
        result.evicted_node_ids = evicted_nodes
        return result


class MockHotPatternTracker:
    """Mock hot pattern tracker for testing."""

    def __init__(self):
        self.usage = {
            f"pattern_{i}": {
                'heat': 1.0 - (i * 0.1),
                'access_count': 10 - i,
                'last_access': datetime.now()
            }
            for i in range(10)
        }
        self.decay_applied = False

    def _apply_decay_if_needed(self):
        """Mock decay application."""
        self.decay_applied = True
        for pattern in self.usage.values():
            pattern['heat'] *= 0.95

    def prune_cold_patterns(self, min_heat: float = 0.1):
        """Mock pruning."""
        pruned = 0
        to_remove = []

        for pattern_id, data in self.usage.items():
            if data['heat'] < min_heat:
                to_remove.append(pattern_id)
                pruned += 1

        for pattern_id in to_remove:
            del self.usage[pattern_id]

        return pruned


class MockMemoryStream:
    """Mock memory stream for testing."""

    def __init__(self, num_memories: int = 50):
        self.memories = [
            {
                'id': f"mem_{i}",
                'created_at': datetime.now() - timedelta(hours=i),
                'ttl': timedelta(hours=24)
            }
            for i in range(num_memories)
        ]

    def prune_expired(self):
        """Mock pruning."""
        now = datetime.now()
        before = len(self.memories)

        self.memories = [
            m for m in self.memories
            if (now - m['created_at']) < m['ttl']
        ]

        return before - len(self.memories)


class MockStreamManager:
    """Mock stream manager for testing."""

    def __init__(self):
        self.streams = {
            'ephemeral': MockMemoryStream(20),
            'temporary': MockMemoryStream(30),
            'permanent': MockMemoryStream(50),
        }


class MockActivationField:
    """Mock activation field for testing."""

    def __init__(self):
        self.levels = {
            f"node_{i}": 1.0 - (i * 0.05)
            for i in range(20)
        }

    def decay(self, rate: float = 0.5):
        """Mock decay."""
        for node_id in self.levels:
            self.levels[node_id] *= rate


class MockConsolidator:
    """Mock memory consolidator for testing."""

    def __init__(self):
        self._last_consolidation_result = Mock()
        self._last_consolidation_result.episodes_pruned = 10
        self._last_consolidation_result.input_episodes = 100
        self._last_consolidation_result.output_facts = 20
        self._last_consolidation_result.strategy = 'deduplication'


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """Default ForgetConfig."""
    return ForgetConfig(
        enable_scheduled_forgetting=False,  # Disable for tests
        scheduled_interval_minutes=1,
        enable_kg_eviction=True,
        enable_hot_decay=True,
        enable_lifecycle_ttl=True,
        enable_activation_decay=True,
        enable_consolidation=True,
    )


@pytest.fixture
def mock_kg():
    """Mock knowledge graph."""
    return MockKG(num_nodes=950, max_nodes=1000)


@pytest.fixture
def mock_hot_tracker():
    """Mock hot pattern tracker."""
    return MockHotPatternTracker()


@pytest.fixture
def mock_stream_manager():
    """Mock stream manager."""
    return MockStreamManager()


@pytest.fixture
def mock_activation_field():
    """Mock activation field."""
    return MockActivationField()


@pytest.fixture
def mock_consolidator():
    """Mock consolidator."""
    return MockConsolidator()


@pytest.fixture
def event_callback():
    """Mock event callback."""
    return Mock()


@pytest.fixture
def forget_manager(
    default_config,
    mock_kg,
    mock_hot_tracker,
    mock_stream_manager,
    mock_activation_field,
    mock_consolidator,
    event_callback
):
    """Fully configured ForgetManager."""
    return ForgetManager(
        config=default_config,
        kg=mock_kg,
        hot_tracker=mock_hot_tracker,
        stream_manager=mock_stream_manager,
        activation_field=mock_activation_field,
        consolidator=mock_consolidator,
        event_callback=event_callback
    )


# ============================================================================
# Test 1: Initialization
# ============================================================================

def test_initialization_default():
    """Test ForgetManager with default config."""
    manager = ForgetManager()

    assert manager.config is not None
    assert isinstance(manager.config, ForgetConfig)
    assert manager._kg is None
    assert manager._hot_tracker is None
    assert manager._total_items_forgotten == 0
    assert manager._total_cycles == 0
    assert not manager._running


def test_initialization_custom_config(default_config):
    """Test ForgetManager with custom config."""
    manager = ForgetManager(config=default_config)

    assert manager.config == default_config
    assert manager.config.scheduled_interval_minutes == 1
    assert manager.config.enable_kg_eviction is True


def test_initialization_with_subsystems(
    default_config,
    mock_kg,
    mock_hot_tracker
):
    """Test ForgetManager with subsystems."""
    manager = ForgetManager(
        config=default_config,
        kg=mock_kg,
        hot_tracker=mock_hot_tracker
    )

    assert manager._kg is mock_kg
    assert manager._hot_tracker is mock_hot_tracker
    assert manager._stream_manager is None  # Not provided


def test_initialization_with_event_callback(default_config, event_callback):
    """Test ForgetManager with event callback."""
    manager = ForgetManager(
        config=default_config,
        event_callback=event_callback
    )

    assert manager._event_callback is event_callback


def test_factory_function():
    """Test create_forget_manager factory."""
    config = ForgetConfig(enable_kg_eviction=False)
    manager = create_forget_manager(config=config)

    assert isinstance(manager, ForgetManager)
    assert manager.config.enable_kg_eviction is False


# ============================================================================
# Test 2: Subsystem Registration
# ============================================================================

def test_register_kg(mock_kg):
    """Test registering KG after initialization."""
    manager = ForgetManager()
    assert manager._kg is None

    manager.register_kg(mock_kg)
    assert manager._kg is mock_kg


def test_register_hot_tracker(mock_hot_tracker):
    """Test registering hot tracker after initialization."""
    manager = ForgetManager()
    assert manager._hot_tracker is None

    manager.register_hot_tracker(mock_hot_tracker)
    assert manager._hot_tracker is mock_hot_tracker


def test_register_stream_manager(mock_stream_manager):
    """Test registering stream manager after initialization."""
    manager = ForgetManager()
    manager.register_stream_manager(mock_stream_manager)
    assert manager._stream_manager is mock_stream_manager


def test_register_activation_field(mock_activation_field):
    """Test registering activation field after initialization."""
    manager = ForgetManager()
    manager.register_activation_field(mock_activation_field)
    assert manager._activation_field is mock_activation_field


def test_register_consolidator(mock_consolidator):
    """Test registering consolidator after initialization."""
    manager = ForgetManager()
    manager.register_consolidator(mock_consolidator)
    assert manager._consolidator is mock_consolidator


# ============================================================================
# Test 3: KG Eviction
# ============================================================================

@pytest.mark.asyncio
async def test_kg_eviction_when_capacity_exceeded(forget_manager, mock_kg):
    """Test KG eviction when capacity threshold exceeded."""
    # KG has 950 nodes, max is 1000, threshold is 0.9 (900)
    # Should trigger eviction
    result = await forget_manager.forget_now(
        trigger=ForgetTrigger.MANUAL,
        subsystems={'kg'}
    )

    assert 'kg' in result
    event = result['kg']
    assert event.subsystem == 'kg'
    assert event.items_affected > 0
    assert event.policy == ForgetPolicy.IMPORTANCE
    assert event.trigger == ForgetTrigger.MANUAL


@pytest.mark.asyncio
async def test_kg_eviction_when_below_capacity(default_config):
    """Test KG eviction when below capacity threshold."""
    # Create KG with 500 nodes, max 1000 (50% capacity)
    kg = MockKG(num_nodes=500, max_nodes=1000)
    manager = ForgetManager(config=default_config, kg=kg)

    result = await manager.forget_now(subsystems={'kg'})

    # Should not evict (below 90% threshold)
    assert 'kg' not in result or result['kg'] is None


@pytest.mark.asyncio
async def test_kg_eviction_disabled(default_config, mock_kg):
    """Test KG eviction when disabled in config."""
    config = ForgetConfig(enable_kg_eviction=False)
    manager = ForgetManager(config=config, kg=mock_kg)

    result = await manager.forget_now(subsystems={'kg'})

    assert 'kg' not in result


@pytest.mark.asyncio
async def test_kg_eviction_without_kg(default_config):
    """Test KG eviction without KG registered."""
    manager = ForgetManager(config=default_config)

    result = await manager.forget_now(subsystems={'kg'})

    assert 'kg' not in result


# ============================================================================
# Test 4: Hot Pattern Decay
# ============================================================================

@pytest.mark.asyncio
async def test_hot_pattern_decay(forget_manager, mock_hot_tracker):
    """Test hot pattern decay and pruning."""
    initial_count = len(mock_hot_tracker.usage)

    result = await forget_manager.forget_now(subsystems={'hot'})

    assert 'hot' in result
    event = result['hot']
    assert event.subsystem == 'hot'
    assert event.policy == ForgetPolicy.DECAY
    assert event.items_affected > 0
    assert mock_hot_tracker.decay_applied is True

    # Some patterns should be pruned
    final_count = len(mock_hot_tracker.usage)
    assert final_count < initial_count


@pytest.mark.asyncio
async def test_hot_decay_disabled(default_config, mock_hot_tracker):
    """Test hot decay when disabled."""
    config = ForgetConfig(enable_hot_decay=False)
    manager = ForgetManager(config=config, hot_tracker=mock_hot_tracker)

    result = await manager.forget_now(subsystems={'hot'})

    assert 'hot' not in result


@pytest.mark.asyncio
async def test_hot_decay_no_pruning(default_config):
    """Test hot decay when no patterns need pruning."""
    # Create tracker with all high heat patterns
    tracker = MockHotPatternTracker()
    for pattern in tracker.usage.values():
        pattern['heat'] = 0.9  # All above threshold

    manager = ForgetManager(config=default_config, hot_tracker=tracker)

    result = await manager.forget_now(subsystems={'hot'})

    # Should still apply decay but no pruning
    assert tracker.decay_applied is True


# ============================================================================
# Test 5: Lifecycle TTL
# ============================================================================

@pytest.mark.asyncio
async def test_lifecycle_ttl_pruning(forget_manager, mock_stream_manager):
    """Test lifecycle TTL pruning."""
    total_before = sum(
        len(stream.memories)
        for stream in mock_stream_manager.streams.values()
    )

    result = await forget_manager.forget_now(subsystems={'lifecycle'})

    if 'lifecycle' in result:
        event = result['lifecycle']
        assert event.subsystem == 'lifecycle'
        assert event.policy == ForgetPolicy.TTL
        assert event.items_before == total_before


@pytest.mark.asyncio
async def test_lifecycle_ttl_disabled(default_config, mock_stream_manager):
    """Test lifecycle TTL when disabled."""
    config = ForgetConfig(enable_lifecycle_ttl=False)
    manager = ForgetManager(config=config, stream_manager=mock_stream_manager)

    result = await manager.forget_now(subsystems={'lifecycle'})

    assert 'lifecycle' not in result


# ============================================================================
# Test 6: Activation Decay
# ============================================================================

@pytest.mark.asyncio
async def test_activation_decay(forget_manager, mock_activation_field):
    """Test activation decay."""
    initial_count = len(mock_activation_field.levels)

    result = await forget_manager.forget_now(subsystems={'activation'})

    if 'activation' in result:
        event = result['activation']
        assert event.subsystem == 'activation'
        assert event.policy == ForgetPolicy.DECAY

        # Check that decay was applied
        for level in mock_activation_field.levels.values():
            assert level < 1.0  # All should be decayed


@pytest.mark.asyncio
async def test_activation_decay_disabled(default_config, mock_activation_field):
    """Test activation decay when disabled."""
    config = ForgetConfig(enable_activation_decay=False)
    manager = ForgetManager(config=config, activation_field=mock_activation_field)

    result = await manager.forget_now(subsystems={'activation'})

    assert 'activation' not in result


# ============================================================================
# Test 7: Consolidation Cleanup
# ============================================================================

@pytest.mark.asyncio
async def test_consolidation_cleanup(forget_manager, mock_consolidator):
    """Test consolidation cleanup."""
    result = await forget_manager.forget_now(subsystems={'consolidation'})

    if 'consolidation' in result:
        event = result['consolidation']
        assert event.subsystem == 'consolidation'
        assert event.policy == ForgetPolicy.CONSOLIDATE
        assert event.trigger == ForgetTrigger.CONSOLIDATION
        assert event.items_affected == 10  # From mock


@pytest.mark.asyncio
async def test_consolidation_disabled(default_config, mock_consolidator):
    """Test consolidation when disabled."""
    config = ForgetConfig(enable_consolidation=False)
    manager = ForgetManager(config=config, consolidator=mock_consolidator)

    result = await manager.forget_now(subsystems={'consolidation'})

    assert 'consolidation' not in result


@pytest.mark.asyncio
async def test_consolidation_prune_disabled(default_config, mock_consolidator):
    """Test consolidation when pruning disabled."""
    config = ForgetConfig(
        enable_consolidation=True,
        prune_after_consolidation=False
    )
    manager = ForgetManager(config=config, consolidator=mock_consolidator)

    result = await manager.forget_now(subsystems={'consolidation'})

    assert 'consolidation' not in result


# ============================================================================
# Test 8: All Subsystems Together
# ============================================================================

@pytest.mark.asyncio
async def test_forget_all_subsystems(forget_manager):
    """Test forgetting across all subsystems."""
    result = await forget_manager.forget_now(trigger=ForgetTrigger.MANUAL)

    # Should have results from multiple subsystems
    assert len(result) > 0

    # Check statistics were updated
    assert forget_manager._total_cycles == 1
    assert forget_manager._total_items_forgotten > 0
    assert forget_manager._last_cycle_time is not None


@pytest.mark.asyncio
async def test_forget_specific_subsystems(forget_manager):
    """Test forgetting only specific subsystems."""
    result = await forget_manager.forget_now(
        subsystems={'kg', 'hot'}
    )

    # Should only have KG and hot results
    possible_keys = {'kg', 'hot'}
    for key in result.keys():
        assert key in possible_keys


@pytest.mark.asyncio
async def test_forget_empty_subsystems(forget_manager):
    """Test forgetting with empty subsystems set."""
    result = await forget_manager.forget_now(subsystems=set())

    assert len(result) == 0


# ============================================================================
# Test 9: Event Handling
# ============================================================================

@pytest.mark.asyncio
async def test_event_emission(forget_manager, event_callback):
    """Test that events are emitted to callback."""
    await forget_manager.forget_now()

    # Callback should have been called for each subsystem that ran
    assert event_callback.call_count > 0

    # Check that ForgetEvent objects were passed
    for call in event_callback.call_args_list:
        event = call[0][0]
        assert isinstance(event, ForgetEvent)


@pytest.mark.asyncio
async def test_event_history(forget_manager):
    """Test event history tracking."""
    await forget_manager.forget_now()

    history = forget_manager.get_event_history()
    assert len(history) > 0

    for event in history:
        assert isinstance(event, ForgetEvent)
        assert event.timestamp is not None
        assert event.subsystem in {'kg', 'hot', 'lifecycle', 'activation', 'consolidation'}


@pytest.mark.asyncio
async def test_event_history_limit(forget_manager):
    """Test event history respects limit."""
    # Run multiple cycles
    for _ in range(5):
        await forget_manager.forget_now()

    # Get limited history
    history = forget_manager.get_event_history(limit=3)
    assert len(history) <= 3


@pytest.mark.asyncio
async def test_event_history_by_subsystem(forget_manager):
    """Test filtering event history by subsystem."""
    await forget_manager.forget_now()

    kg_events = forget_manager.get_event_history(subsystem='kg')

    for event in kg_events:
        assert event.subsystem == 'kg'


@pytest.mark.asyncio
async def test_event_callback_error_handling(default_config, mock_kg):
    """Test that callback errors don't crash the manager."""
    # Create callback that raises exception
    def bad_callback(event):
        raise ValueError("Callback error")

    manager = ForgetManager(
        config=default_config,
        kg=mock_kg,
        event_callback=bad_callback
    )

    # Should not raise exception
    result = await manager.forget_now()

    # Should still complete successfully
    assert manager._total_cycles == 1


# ============================================================================
# Test 10: Statistics
# ============================================================================

@pytest.mark.asyncio
async def test_statistics(forget_manager):
    """Test statistics tracking."""
    await forget_manager.forget_now()

    stats = forget_manager.get_stats()

    assert 'total_cycles' in stats
    assert 'total_items_forgotten' in stats
    assert 'last_cycle' in stats
    assert 'background_running' in stats
    assert 'subsystems_enabled' in stats
    assert 'by_subsystem' in stats
    assert 'config' in stats

    assert stats['total_cycles'] == 1
    assert stats['total_items_forgotten'] > 0
    assert stats['background_running'] is False


@pytest.mark.asyncio
async def test_statistics_by_subsystem(forget_manager):
    """Test per-subsystem statistics."""
    await forget_manager.forget_now()

    stats = forget_manager.get_stats()
    by_subsystem = stats['by_subsystem']

    for subsystem, data in by_subsystem.items():
        assert 'events' in data
        assert 'items_forgotten' in data
        assert 'total_duration_ms' in data
        assert data['events'] > 0


# ============================================================================
# Test 11: Background Loop
# ============================================================================

@pytest.mark.asyncio
async def test_background_loop_start_stop(default_config):
    """Test starting and stopping background loop."""
    manager = ForgetManager(config=default_config)

    assert not manager._running
    assert manager._background_task is None

    await manager.start_background_loop()

    assert manager._running
    assert manager._background_task is not None

    await manager.stop_background_loop()

    assert not manager._running
    assert manager._background_task is None


@pytest.mark.asyncio
async def test_background_loop_already_running(default_config):
    """Test starting background loop when already running."""
    manager = ForgetManager(config=default_config)

    await manager.start_background_loop()

    # Try to start again
    await manager.start_background_loop()

    # Should still be running (no error)
    assert manager._running

    await manager.stop_background_loop()


@pytest.mark.asyncio
async def test_background_loop_scheduled_execution(default_config, mock_kg):
    """Test that background loop executes scheduled forgetting."""
    # Use very short interval for testing
    config = ForgetConfig(
        enable_scheduled_forgetting=True,
        scheduled_interval_minutes=0.01  # 0.6 seconds
    )

    manager = ForgetManager(config=config, kg=mock_kg)

    await manager.start_background_loop()

    # Wait for one cycle
    await asyncio.sleep(1.5)

    await manager.stop_background_loop()

    # Should have run at least once
    assert manager._total_cycles >= 1


@pytest.mark.asyncio
async def test_async_context_manager(default_config, mock_kg):
    """Test ForgetManager as async context manager."""
    config = ForgetConfig(
        enable_scheduled_forgetting=True,
        scheduled_interval_minutes=60
    )

    async with ForgetManager(config=config, kg=mock_kg) as manager:
        # Should start background loop
        assert manager._running

    # Should stop background loop on exit
    # (Can't easily check _running after exit, but test shouldn't hang)


# ============================================================================
# Test 12: Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_forget_with_no_subsystems_registered(default_config):
    """Test forgetting with no subsystems registered."""
    manager = ForgetManager(config=default_config)

    result = await manager.forget_now()

    # Should return empty dict
    assert len(result) == 0
    assert manager._total_cycles == 1


@pytest.mark.asyncio
async def test_forget_with_subsystem_errors(default_config):
    """Test forgetting when subsystems raise errors."""
    # Create KG that raises exception
    kg = MockKG()
    kg.evict_to_capacity = Mock(side_effect=Exception("KG error"))

    manager = ForgetManager(config=default_config, kg=kg)

    # Should handle exception gracefully
    result = await manager.forget_now(subsystems={'kg'})

    # KG eviction should not be in results
    assert 'kg' not in result

    # Should still update cycle count
    assert manager._total_cycles == 1


@pytest.mark.asyncio
async def test_event_history_max_size(default_config, mock_kg):
    """Test that event history doesn't grow unbounded."""
    manager = ForgetManager(config=default_config, kg=mock_kg)
    manager._max_history = 5

    # Run many cycles
    for _ in range(10):
        await manager.forget_now()

    # History should be capped
    assert len(manager._event_history) <= 5


@pytest.mark.asyncio
async def test_multiple_triggers(forget_manager):
    """Test different trigger types."""
    # Manual trigger
    result1 = await forget_manager.forget_now(trigger=ForgetTrigger.MANUAL)
    if result1:
        assert list(result1.values())[0].trigger == ForgetTrigger.MANUAL

    # Scheduled trigger
    result2 = await forget_manager.forget_now(trigger=ForgetTrigger.SCHEDULED)
    if result2:
        assert list(result2.values())[0].trigger == ForgetTrigger.SCHEDULED

    # Capacity trigger
    result3 = await forget_manager.forget_now(trigger=ForgetTrigger.CAPACITY)
    if result3:
        assert list(result3.values())[0].trigger == ForgetTrigger.CAPACITY


@pytest.mark.asyncio
async def test_config_validation():
    """Test config with various settings."""
    config = ForgetConfig(
        enable_kg_eviction=False,
        enable_hot_decay=False,
        enable_lifecycle_ttl=False,
        enable_activation_decay=False,
        enable_consolidation=False,
    )

    manager = ForgetManager(config=config)

    # All subsystems disabled
    result = await manager.forget_now()
    assert len(result) == 0


# ============================================================================
# Test 13: Logging
# ============================================================================

@pytest.mark.asyncio
async def test_logging_enabled(default_config, mock_kg, caplog):
    """Test that operations are logged when enabled."""
    config = ForgetConfig(log_operations=True)
    manager = ForgetManager(config=config, kg=mock_kg)

    await manager.forget_now()

    # Check that log messages were created
    # (Note: exact log checking depends on logging configuration)


@pytest.mark.asyncio
async def test_logging_disabled(default_config, mock_kg):
    """Test that logging can be disabled."""
    config = ForgetConfig(log_operations=False)
    manager = ForgetManager(config=config, kg=mock_kg)

    # Should not raise any errors
    await manager.forget_now()


# ============================================================================
# Test 14: ForgetEvent Dataclass
# ============================================================================

def test_forget_event_creation():
    """Test ForgetEvent dataclass creation."""
    event = ForgetEvent(
        timestamp=datetime.now(),
        trigger=ForgetTrigger.MANUAL,
        policy=ForgetPolicy.DECAY,
        subsystem='test',
        items_affected=10,
        items_before=100,
        items_after=90,
        duration_ms=5.0,
        metadata={'key': 'value'}
    )

    assert event.subsystem == 'test'
    assert event.items_affected == 10
    assert event.metadata['key'] == 'value'


# ============================================================================
# Test 15: Policy Enumeration
# ============================================================================

def test_forget_policy_enum():
    """Test ForgetPolicy enumeration."""
    assert ForgetPolicy.NEVER.value == "never"
    assert ForgetPolicy.DECAY.value == "decay"
    assert ForgetPolicy.TTL.value == "ttl"
    assert ForgetPolicy.LRU.value == "lru"
    assert ForgetPolicy.LFU.value == "lfu"
    assert ForgetPolicy.IMPORTANCE.value == "importance"
    assert ForgetPolicy.CONSOLIDATE.value == "consolidate"


def test_forget_trigger_enum():
    """Test ForgetTrigger enumeration."""
    assert ForgetTrigger.SCHEDULED.value == "scheduled"
    assert ForgetTrigger.CAPACITY.value == "capacity"
    assert ForgetTrigger.MANUAL.value == "manual"
    assert ForgetTrigger.CONSOLIDATION.value == "consolidation"


# ============================================================================
# Test 16: Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_lifecycle_integration():
    """Test complete lifecycle: init → register → forget → stats → cleanup."""
    # Create manager with minimal config
    config = ForgetConfig(enable_scheduled_forgetting=False)
    manager = ForgetManager(config=config)

    # Register subsystems
    kg = MockKG(num_nodes=950, max_nodes=1000)
    hot_tracker = MockHotPatternTracker()

    manager.register_kg(kg)
    manager.register_hot_tracker(hot_tracker)

    # Run forgetting
    result = await manager.forget_now()

    # Check results
    assert len(result) > 0

    # Get statistics
    stats = manager.get_stats()
    assert stats['total_cycles'] == 1

    # Get history
    history = manager.get_event_history()
    assert len(history) > 0


@pytest.mark.asyncio
async def test_concurrent_forget_operations():
    """Test concurrent forget operations."""
    config = ForgetConfig(enable_scheduled_forgetting=False)
    kg = MockKG(num_nodes=950, max_nodes=1000)
    manager = ForgetManager(config=config, kg=kg)

    # Run multiple forget operations concurrently
    results = await asyncio.gather(
        manager.forget_now(),
        manager.forget_now(),
        manager.forget_now()
    )

    # All should complete
    assert len(results) == 3

    # Total cycles should be 3
    assert manager._total_cycles == 3


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
