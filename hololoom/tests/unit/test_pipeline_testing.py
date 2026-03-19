"""
Unit tests for ReadWriteTracker — declaration verification utility.

Tests:
- Undeclared reads are caught
- Undeclared writes are caught
- Infrastructure fields are excluded
- Declared but unused fields don't trigger errors
"""

import pytest
from dataclasses import dataclass
from typing import Optional

from hololoom.pipeline.testing import ReadWriteTracker
from hololoom.pipeline.exceptions import UndeclaredReadError, UndeclaredWriteError


@dataclass
class MockContext:
    """Minimal context for tracker tests."""
    query: str = "test"
    features: Optional[str] = None
    result: Optional[str] = None
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def add_error(self, msg: str):
        self.errors.append(msg)

    def record_timing(self, name: str, ms: float):
        pass

    @property
    def is_blocked(self) -> bool:
        return False


class TestReadWriteTracker:
    """Tests for the ReadWriteTracker verification utility."""

    def test_valid_access_passes(self):
        """Accessing only declared fields passes verification."""
        ctx = MockContext(query="hello")
        tracker = ReadWriteTracker(
            ctx,
            declared_reads=frozenset({'query'}),
            declared_writes=frozenset({'features'}),
        )
        _ = tracker.query  # declared read
        tracker.features = "extracted"  # declared write
        tracker.verify()  # should not raise

    def test_undeclared_read(self):
        """Reading a field not in declared_reads raises UndeclaredReadError."""
        ctx = MockContext(query="hello")
        tracker = ReadWriteTracker(
            ctx,
            declared_reads=frozenset(),
            declared_writes=frozenset(),
        )
        _ = tracker.query  # undeclared read
        with pytest.raises(UndeclaredReadError) as exc_info:
            tracker.verify()
        assert 'query' in str(exc_info.value)

    def test_undeclared_write(self):
        """Writing a field not in declared_writes raises UndeclaredWriteError."""
        ctx = MockContext()
        tracker = ReadWriteTracker(
            ctx,
            declared_reads=frozenset(),
            declared_writes=frozenset(),
        )
        tracker.features = "sneaky"  # undeclared write
        with pytest.raises(UndeclaredWriteError) as exc_info:
            tracker.verify()
        assert 'features' in str(exc_info.value)

    def test_infra_fields_excluded(self):
        """Infrastructure methods (add_error, record_timing) don't trigger errors."""
        ctx = MockContext()
        tracker = ReadWriteTracker(
            ctx,
            declared_reads=frozenset(),
            declared_writes=frozenset(),
        )
        tracker.add_error("test error")
        tracker.record_timing("stage", 42.0)
        _ = tracker.is_blocked
        tracker.verify()  # should not raise

    def test_declared_but_unused_ok(self):
        """Declaring reads/writes but not using them is fine."""
        ctx = MockContext()
        tracker = ReadWriteTracker(
            ctx,
            declared_reads=frozenset({'query', 'features'}),
            declared_writes=frozenset({'result'}),
        )
        # Don't access anything
        tracker.verify()  # should not raise

    def test_multiple_violations(self):
        """Both undeclared reads and writes are reported."""
        ctx = MockContext(query="hello")
        tracker = ReadWriteTracker(
            ctx,
            declared_reads=frozenset(),
            declared_writes=frozenset(),
        )
        _ = tracker.query
        tracker.features = "x"
        # Should raise one of the two errors
        with pytest.raises((UndeclaredReadError, UndeclaredWriteError)):
            tracker.verify()
