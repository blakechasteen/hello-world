"""
Pipeline Testing — ReadWriteTracker for CI verification.

Wraps a context dataclass to track actual field access and verify
it matches the stage's declared reads/writes.

Usage:
    tracker = ReadWriteTracker(ctx, stage.reads, stage.writes)
    await stage.execute(tracker)
    tracker.verify()  # raises if undeclared access

Known limitation: catches direct field access (ctx.field) but not
access through helper methods or properties. Stages should access
data fields directly. Infrastructure methods (add_error, record_timing,
is_blocked) are in the infra exclusion set.
"""

from hololoom.pipeline.exceptions import UndeclaredReadError, UndeclaredWriteError

# Infrastructure methods/properties that all stages may access
# without declaring in reads/writes.
INFRA_FIELDS = frozenset({
    'add_error',
    'add_warning',
    'record_timing',
    'record_timing_since',
    'start_timer',
    'is_blocked',
    'blocked_by',
    'has_errors',
    'has_warnings',
    'add_warp_operation',
})


class ReadWriteTracker:
    """
    Wraps a context to verify reads/writes declarations match actual access.

    Pass this instead of the real context to stage.execute() in tests.
    Call verify() afterward to check for undeclared field access.
    """

    def __init__(
        self,
        ctx,
        declared_reads: frozenset[str],
        declared_writes: frozenset[str],
    ):
        object.__setattr__(self, '_ctx', ctx)
        object.__setattr__(self, '_actual_reads', set())
        object.__setattr__(self, '_actual_writes', set())
        object.__setattr__(self, '_declared_reads', declared_reads)
        object.__setattr__(self, '_declared_writes', declared_writes)

    def __getattr__(self, name: str):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        self._actual_reads.add(name)
        return getattr(self._ctx, name)

    def __setattr__(self, name: str, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        self._actual_writes.add(name)
        setattr(self._ctx, name, value)

    def verify(self) -> None:
        """
        Assert declared reads/writes match actual field access.

        Raises:
            UndeclaredReadError: if stage read fields not in declared reads.
            UndeclaredWriteError: if stage wrote fields not in declared writes.
        """
        undeclared_reads = self._actual_reads - self._declared_reads - INFRA_FIELDS
        undeclared_writes = self._actual_writes - self._declared_writes - INFRA_FIELDS

        if undeclared_reads:
            raise UndeclaredReadError(
                f"Stage accessed {undeclared_reads} without declaring in reads. "
                f"Declared reads: {self._declared_reads}"
            )
        if undeclared_writes:
            raise UndeclaredWriteError(
                f"Stage wrote to {undeclared_writes} without declaring in writes. "
                f"Declared writes: {self._declared_writes}"
            )

    @property
    def actual_reads(self) -> frozenset[str]:
        """Fields actually read (for debugging)."""
        return frozenset(self._actual_reads)

    @property
    def actual_writes(self) -> frozenset[str]:
        """Fields actually written (for debugging)."""
        return frozenset(self._actual_writes)
