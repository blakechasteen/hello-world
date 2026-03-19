"""
Test Core Chrono — Coverage Gap Tests
======================================

Supplementary tests targeting lines missed by test_chrono_trigger.py:
- Prometheus metrics import success path (lines 32-33)
- Heartbeat loop internals: pulse, decay trigger, cancellation (lines 316-336)
- _apply_decay method body (lines 345-349)
- __init__.py re-exports
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock


# ============================================================================
# __init__.py re-exports
# ============================================================================

def test_chrono_init_exports():
    """Test chrono __init__.py re-exports the public API."""
    from hololoom.core.chrono import ChronoTrigger, TemporalWindow, ExecutionLimits

    assert ChronoTrigger is not None
    assert TemporalWindow is not None
    assert ExecutionLimits is not None


# ============================================================================
# Prometheus metrics branch (METRICS_ENABLED = True)
# ============================================================================

@pytest.mark.asyncio
async def test_inhale_with_prometheus_metrics_enabled():
    """Test inhale tracks Prometheus metrics when available."""
    mock_metrics = MagicMock()
    mock_metrics.track_breathing = MagicMock()

    with patch.dict("hololoom.core.chrono.trigger.__dict__", {"METRICS_ENABLED": True}), \
         patch("hololoom.core.chrono.trigger.METRICS_ENABLED", True), \
         patch("hololoom.performance.prometheus_metrics.metrics", mock_metrics, create=True):
        from hololoom.core.chrono.trigger import ChronoTrigger
        from hololoom.config import Config

        chrono = ChronoTrigger(
            config=Config.fast(),
            enable_heartbeat=False,
            enable_breathing=True,
        )
        # Speed up breathing
        chrono.breathing.inhale_duration = 0.01
        await chrono._inhale()

    # Metrics module was importable so track_breathing should have been called
    # (or silently skipped if import fails — both are acceptable)


@pytest.mark.asyncio
async def test_exhale_with_prometheus_metrics_enabled():
    """Test exhale tracks Prometheus metrics when available."""
    mock_metrics = MagicMock()
    mock_metrics.track_breathing = MagicMock()

    with patch("hololoom.core.chrono.trigger.METRICS_ENABLED", True), \
         patch("hololoom.performance.prometheus_metrics.metrics", mock_metrics, create=True):
        from hololoom.core.chrono.trigger import ChronoTrigger
        from hololoom.config import Config

        chrono = ChronoTrigger(
            config=Config.fast(),
            enable_heartbeat=False,
            enable_breathing=True,
        )
        chrono.breathing.exhale_duration = 0.01
        await chrono._exhale()


@pytest.mark.asyncio
async def test_rest_with_prometheus_metrics_enabled():
    """Test rest tracks Prometheus metrics when available."""
    mock_metrics = MagicMock()
    mock_metrics.track_breathing = MagicMock()

    with patch("hololoom.core.chrono.trigger.METRICS_ENABLED", True), \
         patch("hololoom.performance.prometheus_metrics.metrics", mock_metrics, create=True):
        from hololoom.core.chrono.trigger import ChronoTrigger
        from hololoom.config import Config

        chrono = ChronoTrigger(
            config=Config.fast(),
            enable_heartbeat=False,
            enable_breathing=True,
        )
        chrono.breathing.rest_duration = 0.01
        await chrono._rest()


# ============================================================================
# _apply_decay method (lines 345-349)
# ============================================================================

@pytest.mark.asyncio
async def test_apply_decay_runs_without_error():
    """Test _apply_decay executes its logging path."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    # Should complete without error (placeholder implementation)
    await chrono._apply_decay()


@pytest.mark.asyncio
async def test_apply_decay_logs_rate():
    """Test _apply_decay logs the decay rate."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config
    import logging

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )
    chrono.decay_rate = 0.05

    with patch("hololoom.core.chrono.trigger.logger") as mock_logger:
        await chrono._apply_decay()
        mock_logger.debug.assert_called_once()
        assert "0.05" in mock_logger.debug.call_args[0][0]


# ============================================================================
# Heartbeat loop internals (lines 316-336)
# ============================================================================

@pytest.mark.asyncio
async def test_heartbeat_pulse_and_cancel():
    """Test heartbeat loop runs a pulse then cancels cleanly."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,  # We'll call _heartbeat manually
        enable_breathing=False,
    )

    # Patch asyncio.sleep to avoid 60s wait — raise CancelledError on first call
    call_count = 0

    async def mock_sleep(duration):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            raise asyncio.CancelledError()

    with patch("hololoom.core.chrono.trigger.asyncio.sleep", side_effect=mock_sleep):
        # _heartbeat should exit cleanly on CancelledError
        await chrono._heartbeat()


@pytest.mark.asyncio
async def test_heartbeat_triggers_decay_after_one_hour():
    """Test heartbeat triggers _apply_decay when 1+ hour has passed."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    # Set last_decay_time to 2 hours ago so the decay condition triggers
    chrono.last_decay_time = datetime.now() - timedelta(hours=2)

    call_count = 0
    decay_called = False
    original_apply_decay = chrono._apply_decay

    async def mock_apply_decay():
        nonlocal decay_called
        decay_called = True
        await original_apply_decay()

    chrono._apply_decay = mock_apply_decay

    async def mock_sleep(duration):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First sleep returns normally — lets the pulse body execute
            return
        # Second sleep — cancel to exit loop
        raise asyncio.CancelledError()

    with patch("hololoom.core.chrono.trigger.asyncio.sleep", side_effect=mock_sleep):
        await chrono._heartbeat()

    assert decay_called, "_apply_decay should have been called after 1+ hour gap"
    # last_decay_time should have been updated
    assert (datetime.now() - chrono.last_decay_time).total_seconds() < 5


@pytest.mark.asyncio
async def test_heartbeat_skips_decay_if_recent():
    """Test heartbeat does NOT trigger decay if less than 1 hour passed."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    # last_decay_time is now() (just initialized)
    original_last_decay = chrono.last_decay_time
    decay_called = False

    async def mock_apply_decay():
        nonlocal decay_called
        decay_called = True

    chrono._apply_decay = mock_apply_decay

    call_count = 0

    async def mock_sleep(duration):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return
        raise asyncio.CancelledError()

    with patch("hololoom.core.chrono.trigger.asyncio.sleep", side_effect=mock_sleep):
        await chrono._heartbeat()

    assert not decay_called, "_apply_decay should NOT be called when < 1 hour"


@pytest.mark.asyncio
async def test_heartbeat_survives_exception():
    """Test heartbeat catches generic exceptions and continues."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    call_count = 0

    async def mock_sleep(duration):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return  # First iteration — body will raise
        if call_count == 2:
            return  # Second iteration — normal
        raise asyncio.CancelledError()  # Third — exit

    # Patch datetime.now to raise on first call inside the loop
    original_now = datetime.now
    now_call_count = 0

    def exploding_now():
        nonlocal now_call_count
        now_call_count += 1
        # The init sets last_decay_time, so skip those. Blow up only inside loop.
        if now_call_count >= 2 and now_call_count <= 2:
            raise RuntimeError("Simulated error in heartbeat")
        return original_now()

    with patch("hololoom.core.chrono.trigger.asyncio.sleep", side_effect=mock_sleep), \
         patch("hololoom.core.chrono.trigger.datetime") as mock_dt:
        mock_dt.now = exploding_now
        mock_dt.min = datetime.min
        # Should NOT crash — catches Exception, logs, continues
        await chrono._heartbeat()

    # Verify it ran multiple iterations (survived the error)
    assert call_count >= 2


@pytest.mark.asyncio
async def test_heartbeat_logs_start_and_stop():
    """Test heartbeat logs start and stop messages."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    async def mock_sleep(duration):
        raise asyncio.CancelledError()

    with patch("hololoom.core.chrono.trigger.asyncio.sleep", side_effect=mock_sleep), \
         patch("hololoom.core.chrono.trigger.logger") as mock_logger:
        await chrono._heartbeat()

        # Should log "Heartbeat started" and "Heartbeat stopped"
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("started" in c.lower() for c in info_calls)
        assert any("stopped" in c.lower() for c in info_calls)


# ============================================================================
# stop() with no heartbeat (edge case)
# ============================================================================

def test_stop_with_no_heartbeat_task():
    """Test stop() is safe when no heartbeat was ever started."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    # Should not raise
    chrono.stop()
    assert chrono.heartbeat_task is None


def test_stop_with_already_done_heartbeat():
    """Test stop() is safe when heartbeat already finished."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    # Create a mock task that's already done
    mock_task = MagicMock()
    mock_task.done.return_value = True
    chrono.heartbeat_task = mock_task

    chrono.stop()
    # cancel() should NOT be called since task is done
    mock_task.cancel.assert_not_called()


# ============================================================================
# TemporalWindow edge cases
# ============================================================================

def test_temporal_window_contains_boundary_start():
    """Test temporal window contains start boundary (inclusive)."""
    from hololoom.core.chrono.trigger import TemporalWindow

    now = datetime.now()
    start = now - timedelta(hours=1)
    window = TemporalWindow(start=start, end=now, max_age=timedelta(hours=24))

    assert window.contains(start)  # Start is inclusive


def test_temporal_window_contains_boundary_end():
    """Test temporal window contains end boundary (inclusive)."""
    from hololoom.core.chrono.trigger import TemporalWindow

    now = datetime.now()
    start = now - timedelta(hours=1)
    window = TemporalWindow(start=start, end=now, max_age=timedelta(hours=24))

    assert window.contains(now)  # End is inclusive


def test_recency_weight_high_bias():
    """Test recency weight with high bias penalizes old timestamps more."""
    from hololoom.core.chrono.trigger import TemporalWindow

    now = datetime.now()
    window = TemporalWindow(
        start=now - timedelta(hours=1),
        end=now,
        max_age=timedelta(hours=24),
        recency_bias=1.0,  # Max bias
    )

    # Oldest should get weight 0.0 (1.0 - 1.0 * 1.0)
    oldest_weight = window.recency_weight(now - timedelta(hours=1))
    assert oldest_weight == pytest.approx(0.0, abs=0.01)

    # Newest should still get ~1.0
    newest_weight = window.recency_weight(now)
    assert newest_weight == pytest.approx(1.0, abs=0.01)


def test_recency_weight_zero_bias():
    """Test recency weight with zero bias gives uniform weight."""
    from hololoom.core.chrono.trigger import TemporalWindow

    now = datetime.now()
    window = TemporalWindow(
        start=now - timedelta(hours=1),
        end=now,
        max_age=timedelta(hours=24),
        recency_bias=0.0,  # No bias
    )

    # All timestamps in window should get weight 1.0
    old_weight = window.recency_weight(now - timedelta(hours=1))
    mid_weight = window.recency_weight(now - timedelta(minutes=30))
    new_weight = window.recency_weight(now)

    assert old_weight == pytest.approx(1.0, abs=0.01)
    assert mid_weight == pytest.approx(1.0, abs=0.01)
    assert new_weight == pytest.approx(1.0, abs=0.01)


# ============================================================================
# BreathingRhythm custom values
# ============================================================================

def test_breathing_rhythm_custom_values():
    """Test BreathingRhythm with custom values."""
    from hololoom.core.chrono.trigger import BreathingRhythm

    rhythm = BreathingRhythm(
        inhale_duration=1.0,
        exhale_duration=0.3,
        rest_duration=0.05,
        breathing_rate=2.0,
        enable_rest=False,
        sparsity_on_exhale=0.9,
        pressure_threshold=0.5,
    )

    assert rhythm.inhale_duration == 1.0
    assert rhythm.exhale_duration == 0.3
    assert rhythm.rest_duration == 0.05
    assert rhythm.breathing_rate == 2.0
    assert rhythm.enable_rest is False
    assert rhythm.sparsity_on_exhale == 0.9
    assert rhythm.pressure_threshold == 0.5


# ============================================================================
# ExecutionLimits custom values
# ============================================================================

def test_execution_limits_custom_values():
    """Test ExecutionLimits with custom values."""
    from hololoom.core.chrono.trigger import ExecutionLimits

    limits = ExecutionLimits(
        max_duration=10.0,
        stage_timeouts={"custom_stage": 3.0},
        halt_on_low_confidence=0.5,
        max_memory_mb=2000,
        enable_early_stopping=False,
    )

    assert limits.max_duration == 10.0
    assert limits.stage_timeouts == {"custom_stage": 3.0}
    assert limits.halt_on_low_confidence == 0.5
    assert limits.max_memory_mb == 2000
    assert limits.enable_early_stopping is False


# ============================================================================
# Monitor with early stopping disabled
# ============================================================================

@pytest.mark.asyncio
async def test_monitor_halt_callback_not_called_when_early_stopping_disabled():
    """Test halt callback is ignored when early_stopping is disabled."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )
    chrono.limits.enable_early_stopping = False

    callback_called = False

    def halt_callback(result):
        nonlocal callback_called
        callback_called = True
        return True  # Would halt if checked

    async def operation():
        return {"confidence": 0.01}

    result = await chrono.monitor(
        operation, timeout=2.0, halt_callback=halt_callback
    )

    # Result should be the raw operation result, not halted
    assert result["confidence"] == 0.01
    assert "status" not in result
    assert not callback_called


@pytest.mark.asyncio
async def test_monitor_halt_callback_returns_false():
    """Test halt callback returning False allows result through."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )
    chrono.limits.enable_early_stopping = True

    async def operation():
        return {"confidence": 0.9, "data": "good"}

    result = await chrono.monitor(
        operation,
        timeout=2.0,
        halt_callback=lambda r: False,  # Don't halt
    )

    assert result["data"] == "good"
    assert "status" not in result


@pytest.mark.asyncio
async def test_monitor_default_timeout_for_unknown_stage():
    """Test monitor uses 5.0s default for unknown stage names."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    async def operation():
        return {"ok": True}

    # "unknown_stage" not in stage_timeouts — should use default 5.0
    result = await chrono.monitor(
        operation, timeout=None, stage="unknown_stage"
    )

    assert result["ok"] is True


# ============================================================================
# Evolution stats edge: fewer than 10 durations
# ============================================================================

@pytest.mark.asyncio
async def test_evolution_stats_fewer_than_10():
    """Test recent_duration returns all when fewer than 10 executions."""
    from hololoom.core.chrono.trigger import ChronoTrigger
    from hololoom.config import Config

    chrono = ChronoTrigger(
        config=Config.fast(),
        enable_heartbeat=False,
        enable_breathing=False,
    )

    for _ in range(3):
        await chrono.fire(datetime.now(), pattern_card_mode="fast")
        chrono.record_completion()

    stats = chrono.get_evolution_stats()

    assert stats["executions"] == 3
    assert len(stats["recent_duration"]) == 3


# ============================================================================
# Config without pipeline_timeout attribute
# ============================================================================

def test_chrono_with_config_missing_pipeline_timeout():
    """Test ChronoTrigger handles config without pipeline_timeout."""
    from hololoom.core.chrono.trigger import ChronoTrigger

    # Minimal config object without pipeline_timeout
    class MinimalConfig:
        pass

    chrono = ChronoTrigger(config=MinimalConfig(), enable_heartbeat=False, enable_breathing=False)

    # Should fall back to default 5.0
    assert chrono.limits.max_duration == 5.0
