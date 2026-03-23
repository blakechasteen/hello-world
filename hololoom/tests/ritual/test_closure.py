"""Tests for BindingObligation — mandatory closure gate."""

import pytest
from datetime import datetime, timezone

from hololoom.core.ritual.closure import OpenBinding, ClosureGate


def _make_binding(opener="dawn_prime", closer="dusk_consolidate", **overrides):
    defaults = dict(
        opener_ritual_id=opener,
        closer_ritual_id=closer,
        opened_at=datetime.now(timezone.utc).isoformat(),
        session_id="sess-1",
        deadline_seconds=0.0,
        priority="mandatory",
    )
    defaults.update(overrides)
    return OpenBinding(**defaults)


class TestClosureGate:
    def test_empty_gate_not_gated(self):
        gate = ClosureGate()
        assert not gate.is_gated("any_ritual")
        assert not gate.has_mandatory_open()

    def test_open_gates_other_rituals(self):
        gate = ClosureGate()
        gate.open(_make_binding())
        assert gate.is_gated("pre_code_health_check")  # blocked
        assert not gate.is_gated("dusk_consolidate")    # closer passes through
        assert gate.has_mandatory_open()

    def test_close_unblocks(self):
        gate = ClosureGate()
        gate.open(_make_binding())
        assert gate.is_gated("some_ritual")

        closed = gate.close("dusk_consolidate")
        assert closed is True
        assert not gate.is_gated("some_ritual")
        assert not gate.has_mandatory_open()

    def test_close_unknown_is_noop(self):
        gate = ClosureGate()
        closed = gate.close("nonexistent")
        assert closed is False

    def test_multiple_openers(self):
        gate = ClosureGate()
        gate.open(_make_binding(opener="dawn_a", closer="dusk"))
        gate.open(_make_binding(opener="dawn_b", closer="dusk"))

        assert gate.is_gated("other")
        assert len(gate.pending()) == 2

        gate.close("dusk")
        assert not gate.has_mandatory_open()
        assert len(gate.pending()) == 0

    def test_advisory_binding_does_not_gate(self):
        gate = ClosureGate()
        gate.open(_make_binding(priority="advisory"))
        assert not gate.is_gated("any_ritual")  # advisory doesn't gate

    def test_force_close_all(self):
        gate = ClosureGate()
        gate.open(_make_binding(opener="a"))
        gate.open(_make_binding(opener="b"))
        orphaned = gate.force_close_all()
        assert len(orphaned) == 2
        assert not gate.has_mandatory_open()

    def test_pending_returns_all(self):
        gate = ClosureGate()
        gate.open(_make_binding(opener="a", closer="x"))
        gate.open(_make_binding(opener="b", closer="y"))
        pending = gate.pending()
        assert len(pending) == 2

    def test_overdue_empty_when_no_deadline(self):
        gate = ClosureGate()
        gate.open(_make_binding(deadline_seconds=0))
        assert gate.overdue() == []

    def test_overdue_detected(self):
        gate = ClosureGate()
        # Open with a deadline of 0.001 seconds (already expired)
        gate.open(_make_binding(
            deadline_seconds=0.001,
            opened_at="2020-01-01T00:00:00+00:00",
        ))
        overdue = gate.overdue()
        assert len(overdue) == 1


class TestOpenBinding:
    def test_binding_fields(self):
        b = _make_binding()
        assert b.opener_ritual_id == "dawn_prime"
        assert b.closer_ritual_id == "dusk_consolidate"
        assert b.priority == "mandatory"
