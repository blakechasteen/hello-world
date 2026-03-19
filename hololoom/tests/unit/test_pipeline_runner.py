"""
Unit tests for PipelineRunner — the generic parallel level executor.

Tests:
- Sequential stages execute in order
- Parallel stages execute concurrently (wall-clock < sum)
- Policy hooks fire correctly
- Stage failure + abort semantics
- Stage caching (hit/miss)
- explain() and explain_stage() output
"""

import asyncio
import time
import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

from hololoom.pipeline.runner import PipelineRunner
from hololoom.pipeline.types import DefaultPolicy
from hololoom.pipeline.cache import StageCache


# =========================================================================
# Test context and stage stubs
# =========================================================================

@dataclass
class TestContext:
    """Minimal context for runner tests."""
    query: str = "test"
    features: Optional[str] = None
    result: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    blocked_by: Optional[str] = None
    timings: dict = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)

    @property
    def is_blocked(self) -> bool:
        return self.blocked_by is not None

    def record_timing(self, name: str, ms: float):
        self.timings[name] = ms


class AsyncStage:
    """Stage stub with configurable behavior."""
    def __init__(self, reads, writes, delay=0, error=None, cacheable=True):
        self.reads = reads
        self.writes = writes
        self.cacheable = cacheable
        self._delay = delay
        self._error = error
        self.call_count = 0

    async def execute(self, ctx):
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        # Write a marker to each write field
        for f in self.writes:
            setattr(ctx, f, f"produced_by_{id(self)}")


class TrackingPolicy:
    """Policy that records all lifecycle calls."""
    def __init__(self, skip_stages=None, abort_stages=None):
        self.skip_stages = skip_stages or set()
        self.abort_stages = abort_stages or set()
        self.started = []
        self.completed = []

    def should_skip(self, name, ctx):
        return name in self.skip_stages

    def should_abort_on_failure(self, name):
        return name in self.abort_stages

    def on_level_complete(self, level, ctx):
        return None

    def on_stage_start(self, name, ctx):
        self.started.append(name)

    def on_stage_complete(self, name, ctx, duration_ms, error):
        self.completed.append((name, error))


# =========================================================================
# Basic execution tests
# =========================================================================

class TestRunnerExecution:
    """Tests for basic pipeline execution."""

    @pytest.mark.asyncio
    async def test_sequential_stages(self):
        """Stages with dependencies execute in order."""
        stages = {
            'a': AsyncStage(frozenset({'query'}), frozenset({'features'})),
            'b': AsyncStage(frozenset({'features'}), frozenset({'result'})),
        }
        runner = PipelineRunner(stages, frozenset({'query'}))
        ctx = TestContext()
        result = await runner.execute(ctx)
        assert result.features is not None
        assert result.result is not None

    @pytest.mark.asyncio
    async def test_parallel_stages_faster_than_sequential(self):
        """Parallel stages run concurrently — wall time < sum of delays."""
        stages = {
            'a': AsyncStage(frozenset(), frozenset({'x'}), delay=0.05),
            'b': AsyncStage(frozenset(), frozenset({'y'}), delay=0.05),
        }
        runner = PipelineRunner(stages)
        ctx = TestContext()
        t0 = time.perf_counter()
        await runner.execute(ctx)
        elapsed = time.perf_counter() - t0
        # If truly parallel, ~50ms. If sequential, ~100ms.
        assert elapsed < 0.09, f"Expected parallel execution, took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Empty stage dict runs without error."""
        runner = PipelineRunner({})
        ctx = TestContext()
        result = await runner.execute(ctx)
        assert result is ctx


# =========================================================================
# Policy tests
# =========================================================================

class TestRunnerPolicy:
    """Tests for RunPolicy integration."""

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_fire(self):
        """on_stage_start and on_stage_complete called for each stage."""
        stages = {
            'a': AsyncStage(frozenset(), frozenset({'x'})),
            'b': AsyncStage(frozenset({'x'}), frozenset({'y'})),
        }
        policy = TrackingPolicy()
        runner = PipelineRunner(stages, policy=policy)
        await runner.execute(TestContext())
        assert policy.started == ['a', 'b']
        assert len(policy.completed) == 2
        assert all(err is None for _, err in policy.completed)

    @pytest.mark.asyncio
    async def test_skip_stage(self):
        """Skipped stages don't execute."""
        stage_a = AsyncStage(frozenset(), frozenset({'x'}))
        stages = {
            'a': stage_a,
            'b': AsyncStage(frozenset({'x'}), frozenset({'y'})),
        }
        policy = TrackingPolicy(skip_stages={'a'})
        runner = PipelineRunner(stages, policy=policy)
        await runner.execute(TestContext())
        assert stage_a.call_count == 0
        assert 'a' not in policy.started

    @pytest.mark.asyncio
    async def test_abort_on_failure(self):
        """When policy says abort, pipeline stops."""
        stages = {
            'a': AsyncStage(frozenset(), frozenset({'x'}), error=ValueError("boom")),
            'b': AsyncStage(frozenset({'x'}), frozenset({'y'})),
        }
        stage_b = stages['b']
        policy = TrackingPolicy(abort_stages={'a'})
        runner = PipelineRunner(stages, policy=policy)
        ctx = TestContext()
        await runner.execute(ctx)
        assert ctx.blocked_by == 'a'
        assert stage_b.call_count == 0
        assert len(ctx.errors) == 1
        assert 'boom' in ctx.errors[0]

    @pytest.mark.asyncio
    async def test_non_abort_failure_continues(self):
        """When policy says don't abort, pipeline continues past failure."""
        stages = {
            'a': AsyncStage(frozenset(), frozenset({'x'}), error=ValueError("minor")),
            'b': AsyncStage(frozenset(), frozenset({'y'})),
        }
        stage_b = stages['b']
        # a and b at same level, a fails, but policy doesn't abort on 'a'
        policy = TrackingPolicy(abort_stages=set())  # never abort
        runner = PipelineRunner(stages, policy=policy)
        ctx = TestContext()
        await runner.execute(ctx)
        assert ctx.blocked_by is None  # not blocked
        assert len(ctx.errors) == 1


# =========================================================================
# Caching tests
# =========================================================================

class TestRunnerCaching:
    """Tests for stage-level caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_execution(self):
        """Cached stage skips execute() on second run."""
        stage = AsyncStage(frozenset({'query'}), frozenset({'features'}))
        stages = {'a': stage}
        cache = StageCache(maxsize=10)
        runner = PipelineRunner(stages, frozenset({'query'}), cache=cache)

        # First run
        ctx1 = TestContext(query="hello")
        await runner.execute(ctx1)
        assert stage.call_count == 1

        # Second run with same inputs
        ctx2 = TestContext(query="hello")
        stage.call_count = 0
        await runner.execute(ctx2)
        assert stage.call_count == 0  # cache hit

    @pytest.mark.asyncio
    async def test_uncacheable_stage_always_executes(self):
        """Stages with cacheable=False always execute."""
        stage = AsyncStage(frozenset({'query'}), frozenset({'features'}), cacheable=False)
        stages = {'a': stage}
        cache = StageCache(maxsize=10)
        runner = PipelineRunner(stages, frozenset({'query'}), cache=cache)

        await runner.execute(TestContext(query="hello"))
        await runner.execute(TestContext(query="hello"))
        assert stage.call_count == 2


# =========================================================================
# Explain tests
# =========================================================================

class TestRunnerExplain:
    """Tests for pipeline introspection methods."""

    def test_explain_output(self):
        """explain() returns human-readable level listing."""
        stages = {
            'a': AsyncStage(frozenset(), frozenset({'x'})),
            'b': AsyncStage(frozenset(), frozenset({'y'})),
            'c': AsyncStage(frozenset({'x', 'y'}), frozenset({'z'})),
        }
        runner = PipelineRunner(stages)
        output = runner.explain()
        assert 'Level 0' in output
        assert 'PARALLEL' in output
        assert 'Level 1' in output

    def test_explain_stage(self):
        """explain_stage() shows reads, writes, level, and peers."""
        stages = {
            'a': AsyncStage(frozenset(), frozenset({'x'})),
            'b': AsyncStage(frozenset(), frozenset({'y'})),
        }
        runner = PipelineRunner(stages)
        output = runner.explain_stage('a')
        assert 'a' in output
        assert 'writes' in output
        assert 'level' in output
