"""
Unit tests for pipeline resolver — level-aware topological sort.

Tests:
- Linear dependency chains resolve to sequential levels
- Independent stages land at the same level (parallel)
- Cycles are detected with clear error messages
- Disjoint-writes validation catches write conflicts
- Completeness validation warns about unresolvable reads
- Field name validation catches typos
"""

import pytest
from dataclasses import dataclass
from typing import ClassVar

from hololoom.pipeline.resolver import (
    resolve_levels,
    validate_completeness,
    validate_disjoint_writes,
    validate_stage_fields,
)
from hololoom.pipeline.exceptions import CyclicDependencyError, WriteConflictError


# =========================================================================
# Minimal stage stubs for testing (no async, just reads/writes/cacheable)
# =========================================================================

class StageStub:
    """Minimal stage with reads/writes for resolver tests."""
    def __init__(self, reads, writes, cacheable=True):
        self.reads = reads
        self.writes = writes
        self.cacheable = cacheable

    async def execute(self, ctx):
        pass


# =========================================================================
# resolve_levels tests
# =========================================================================

class TestResolveLevels:
    """Tests for the core resolve_levels algorithm."""

    def test_single_stage_no_deps(self):
        """A single stage with no reads resolves to one level."""
        stages = {'a': StageStub(frozenset(), frozenset({'x'}))}
        levels = resolve_levels(stages)
        assert levels == [['a']]

    def test_linear_chain(self):
        """A → B → C produces three sequential levels."""
        stages = {
            'a': StageStub(frozenset(), frozenset({'x'})),
            'b': StageStub(frozenset({'x'}), frozenset({'y'})),
            'c': StageStub(frozenset({'y'}), frozenset({'z'})),
        }
        levels = resolve_levels(stages)
        assert levels == [['a'], ['b'], ['c']]

    def test_parallel_stages(self):
        """Independent stages land at the same level."""
        stages = {
            'a': StageStub(frozenset(), frozenset({'x'})),
            'b': StageStub(frozenset(), frozenset({'y'})),
        }
        levels = resolve_levels(stages)
        assert levels == [['a', 'b']]

    def test_diamond_dependency(self):
        """Diamond: root → {left, right} → join."""
        stages = {
            'root': StageStub(frozenset(), frozenset({'x'})),
            'left': StageStub(frozenset({'x'}), frozenset({'a'})),
            'right': StageStub(frozenset({'x'}), frozenset({'b'})),
            'join': StageStub(frozenset({'a', 'b'}), frozenset({'z'})),
        }
        levels = resolve_levels(stages)
        assert levels[0] == ['root']
        assert sorted(levels[1]) == ['left', 'right']
        assert levels[2] == ['join']

    def test_pre_satisfied_fields(self):
        """Pre-satisfied fields let dependent stages run at level 0."""
        stages = {
            'a': StageStub(frozenset({'query'}), frozenset({'features'})),
        }
        levels = resolve_levels(stages, pre_satisfied=frozenset({'query'}))
        assert levels == [['a']]

    def test_weaving_parallel_pattern(self):
        """Mimics the weaving steps 4-6 parallel group."""
        stages = {
            'pattern_sel': StageStub(frozenset({'query'}), frozenset({'pattern_spec'})),
            'temporal': StageStub(frozenset({'query', 'pattern_spec'}), frozenset({'temporal_window'})),
            'thread_sel': StageStub(frozenset({'pattern_spec', 'temporal_window'}), frozenset({'threads'})),
            'feature_ext': StageStub(frozenset({'query', 'pattern_spec'}), frozenset({'features'})),
            'warp_tens': StageStub(frozenset({'threads'}), frozenset({'warp_ops'})),
            'memory_ret': StageStub(frozenset({'query', 'pattern_spec'}), frozenset({'shards'})),
        }
        pre = frozenset({'query'})
        levels = resolve_levels(stages, pre)

        # pattern_sel at level 0
        assert levels[0] == ['pattern_sel']
        # feature_ext, temporal at level 1 (both read query + pattern_spec)
        assert 'feature_ext' in levels[1]
        assert 'temporal' in levels[1]
        # thread_sel at level 2 (needs temporal_window)
        # memory_ret could be level 1 (reads query + pattern_spec only)
        assert 'memory_ret' in levels[1]
        # warp_tens needs threads → must come after thread_sel
        thread_level = next(i for i, l in enumerate(levels) if 'thread_sel' in l)
        warp_level = next(i for i, l in enumerate(levels) if 'warp_tens' in l)
        assert warp_level > thread_level

    def test_deterministic_ordering(self):
        """Same input always produces same level order (sorted names)."""
        stages = {
            'c': StageStub(frozenset(), frozenset({'x'})),
            'a': StageStub(frozenset(), frozenset({'y'})),
            'b': StageStub(frozenset(), frozenset({'z'})),
        }
        for _ in range(10):
            levels = resolve_levels(stages)
            assert levels == [['a', 'b', 'c']]

    def test_empty_stages(self):
        """Empty stage dict produces empty levels."""
        assert resolve_levels({}) == []

    def test_side_effect_stages_last(self):
        """Stages with empty writes but reads land at the end."""
        stages = {
            'producer': StageStub(frozenset(), frozenset({'data'})),
            'metrics': StageStub(frozenset({'data'}), frozenset()),
        }
        levels = resolve_levels(stages)
        assert levels == [['producer'], ['metrics']]


class TestCycleDetection:
    """Tests for cyclic dependency detection."""

    def test_simple_cycle(self):
        """A reads B's output, B reads A's output → cycle."""
        stages = {
            'a': StageStub(frozenset({'y'}), frozenset({'x'})),
            'b': StageStub(frozenset({'x'}), frozenset({'y'})),
        }
        with pytest.raises(CyclicDependencyError) as exc_info:
            resolve_levels(stages)
        assert 'a' in str(exc_info.value) or 'b' in str(exc_info.value)

    def test_self_cycle(self):
        """A reads its own output → cycle."""
        stages = {
            'a': StageStub(frozenset({'x'}), frozenset({'x'})),
        }
        with pytest.raises(CyclicDependencyError):
            resolve_levels(stages)

    def test_transitive_cycle(self):
        """A → B → C → A → cycle."""
        stages = {
            'a': StageStub(frozenset({'z'}), frozenset({'x'})),
            'b': StageStub(frozenset({'x'}), frozenset({'y'})),
            'c': StageStub(frozenset({'y'}), frozenset({'z'})),
        }
        with pytest.raises(CyclicDependencyError):
            resolve_levels(stages)


# =========================================================================
# validate_disjoint_writes tests
# =========================================================================

class TestDisjointWritesValidation:
    """Tests for concurrent write-conflict detection."""

    def test_disjoint_writes_pass(self):
        """Parallel stages with disjoint writes are OK."""
        stages = {
            'a': StageStub(frozenset(), frozenset({'x'})),
            'b': StageStub(frozenset(), frozenset({'y'})),
        }
        levels = [['a', 'b']]
        validate_disjoint_writes(levels, stages)  # should not raise

    def test_overlapping_writes_fail(self):
        """Two stages at the same level writing the same field → WriteConflictError."""
        stages = {
            'a': StageStub(frozenset(), frozenset({'x', 'shared'})),
            'b': StageStub(frozenset(), frozenset({'y', 'shared'})),
        }
        levels = [['a', 'b']]
        with pytest.raises(WriteConflictError) as exc_info:
            validate_disjoint_writes(levels, stages)
        assert 'shared' in str(exc_info.value)

    def test_overlapping_writes_different_levels_ok(self):
        """Same field written at different levels is fine (sequential)."""
        stages = {
            'a': StageStub(frozenset(), frozenset({'x'})),
            'b': StageStub(frozenset({'x'}), frozenset({'x'})),  # overwrites x
        }
        levels = [['a'], ['b']]
        validate_disjoint_writes(levels, stages)  # should not raise


# =========================================================================
# validate_completeness tests
# =========================================================================

class TestCompletenessValidation:
    """Tests for missing-dependency detection."""

    def test_complete_pipeline(self):
        """All reads satisfied by writes or pre_satisfied → no warnings."""
        stages = {
            'a': StageStub(frozenset({'query'}), frozenset({'x'})),
            'b': StageStub(frozenset({'x'}), frozenset({'y'})),
        }
        warnings = validate_completeness(stages, frozenset({'query'}))
        assert warnings == []

    def test_missing_dependency(self):
        """A reads field nobody writes and isn't pre_satisfied → warning."""
        stages = {
            'a': StageStub(frozenset({'missing'}), frozenset({'x'})),
        }
        warnings = validate_completeness(stages)
        assert len(warnings) == 1
        assert 'missing' in warnings[0]


# =========================================================================
# validate_stage_fields tests
# =========================================================================

class TestFieldValidation:
    """Tests for typo detection in field names."""

    def test_valid_fields(self):
        """All declared fields exist on context → no error."""
        @dataclass
        class Ctx:
            query: str = ""
            features: str = ""

        stages = {
            'a': StageStub(frozenset({'query'}), frozenset({'features'})),
        }
        validate_stage_fields(stages, Ctx, frozenset({'query'}))

    def test_typo_detection(self):
        """Misspelled field name raises ValueError with suggestion."""
        @dataclass
        class Ctx:
            query: str = ""
            features: str = ""

        stages = {
            'a': StageStub(frozenset({'qurey'}), frozenset({'featuers'})),
        }
        with pytest.raises(ValueError) as exc_info:
            validate_stage_fields(stages, Ctx, frozenset({'qurey'}))
        error_msg = str(exc_info.value)
        assert 'qurey' in error_msg or 'featuers' in error_msg
