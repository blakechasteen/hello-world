"""Tests for ContextBudget — partitioned allocation with pressure tracking."""

import pytest

from hololoom.core.ritual.budget import (
    ContextBudget,
    ContextAllocation,
    check_admission,
    MODE_RESEARCH,
    MODE_FUSED,
    MODE_FAST,
    MODE_BARE,
)
from hololoom.core.ritual.types import RitualBudget, TokenBudget


class TestContextBudgetBasics:
    def test_initial_state(self):
        b = ContextBudget(total=5000)
        assert b.remaining() == 5000
        assert b.used == 0
        assert b.pressure() == 0.0

    def test_allocate_reduces_remaining(self):
        b = ContextBudget(total=1000)
        alloc = b.allocate("ritual", 300)
        assert alloc.allocated == 300
        assert b.remaining() == 700
        assert b.used == 300

    def test_allocate_multiple_partitions(self):
        b = ContextBudget(total=1000)
        b.allocate("ritual", 200)
        b.allocate("memory", 300)
        b.allocate("generation", 100)
        assert b.used == 600
        assert b.remaining() == 400

    def test_allocate_caps_at_available(self):
        b = ContextBudget(total=100)
        alloc = b.allocate("ritual", 500)
        assert alloc.allocated == 100  # capped
        assert b.remaining() == 0

    def test_partition_usage(self):
        b = ContextBudget(total=1000)
        b.allocate("ritual", 200)
        b.allocate("memory", 300)
        usage = b.partition_usage()
        assert usage["ritual"] == 200
        assert usage["memory"] == 300


class TestContextBudgetPressure:
    def test_pressure_zero_when_empty(self):
        b = ContextBudget(total=1000)
        assert b.pressure() == 0.0

    def test_pressure_half(self):
        b = ContextBudget(total=1000)
        b.allocate("x", 500)
        assert b.pressure() == pytest.approx(0.5)

    def test_pressure_full(self):
        b = ContextBudget(total=1000)
        b.allocate("x", 1000)
        assert b.pressure() == 1.0

    def test_pressure_zero_total(self):
        b = ContextBudget(total=0)
        assert b.pressure() == 1.0


class TestContextBudgetMode:
    def test_mode_fused_at_low_pressure(self):
        b = ContextBudget(total=1000, _base_mode=MODE_FUSED)
        assert b.mode() == MODE_FUSED

    def test_mode_fast_at_70_percent(self):
        b = ContextBudget(total=1000)
        b.allocate("x", 700)
        assert b.mode() == MODE_FAST

    def test_mode_bare_at_90_percent(self):
        b = ContextBudget(total=1000)
        b.allocate("x", 900)
        assert b.mode() == MODE_BARE

    def test_mode_bare_at_full(self):
        b = ContextBudget(total=1000)
        b.allocate("x", 1000)
        assert b.mode() == MODE_BARE

    def test_custom_base_mode(self):
        b = ContextBudget(total=1000, _base_mode=MODE_RESEARCH)
        assert b.mode() == MODE_RESEARCH

    def test_mode_switches_dynamically(self):
        b = ContextBudget(total=1000)
        assert b.mode() == MODE_FUSED
        b.allocate("x", 700)
        assert b.mode() == MODE_FAST
        b.allocate("y", 200)
        assert b.mode() == MODE_BARE


class TestContextAllocation:
    @pytest.mark.asyncio
    async def test_context_manager_releases(self):
        b = ContextBudget(total=1000)
        async with b.allocate("ritual", 300) as alloc:
            alloc.used = 100
        # 200 unused tokens should be released
        assert b.remaining() == 900

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        b = ContextBudget(total=1000)
        try:
            async with b.allocate("ritual", 300) as alloc:
                alloc.used = 50
                raise ValueError("boom")
        except ValueError:
            pass
        assert b.remaining() == 950

    @pytest.mark.asyncio
    async def test_full_usage_no_release(self):
        b = ContextBudget(total=1000)
        async with b.allocate("ritual", 300) as alloc:
            alloc.used = 300
        assert b.remaining() == 700

    def test_manual_release(self):
        b = ContextBudget(total=1000)
        alloc = b.allocate("ritual", 300)
        alloc.used = 100
        released = alloc.release()
        assert released == 200
        assert b.remaining() == 900

    def test_double_release_is_noop(self):
        b = ContextBudget(total=1000)
        alloc = b.allocate("ritual", 300)
        alloc.used = 100
        alloc.release()
        released = alloc.release()  # second release
        assert released == 0


class TestCheckAdmission:
    def test_with_token_budget(self):
        rb = RitualBudget(max_tokens=200, max_seconds=10.0, priority_class="standard")
        tb = TokenBudget(total=1000, used=500)
        assert check_admission(rb, tb) is True

    def test_with_context_budget(self):
        rb = RitualBudget(max_tokens=200, max_seconds=10.0, priority_class="standard")
        cb = ContextBudget(total=1000)
        assert check_admission(rb, cb) is True

    def test_context_budget_insufficient(self):
        rb = RitualBudget(max_tokens=600, max_seconds=10.0, priority_class="standard")
        cb = ContextBudget(total=1000)
        cb.allocate("other", 500)
        assert check_admission(rb, cb) is False

    def test_critical_always_admitted(self):
        rb = RitualBudget(max_tokens=9999, max_seconds=10.0, priority_class="critical")
        cb = ContextBudget(total=100)
        cb.allocate("x", 99)
        assert check_admission(rb, cb) is True
