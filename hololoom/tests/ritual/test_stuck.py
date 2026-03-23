"""Tests for SnagWatcher — stuck pattern detection."""

import pytest

from hololoom.core.ritual.stuck import (
    StuckSignal, StuckDetection, SlidingWindowDetector,
)
from hololoom.core.ritual.journal import (
    RitualJournal, DISPATCH, BODY_COMPLETE, BODY_FAILED, POLICY_INVOKED, SKIP,
)
from hololoom.core.ritual.types import RitualContext, TokenBudget


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="agent-1", agent_role="coder", domain="coding",
        task_id="t-1", task_type="code_modify", task_intent="fix bug",
        task_complexity="standard", yarn=None, warp=None,
        confidence=0.8, confidence_history=[0.8],
        ritual_outputs={}, ritual_log=[],
        token_budget=TokenBudget(total=5000, used=0),
        time_budget=60.0, escalation_path=[], session_id="sess-1",
    )
    defaults.update(overrides)
    return RitualContext(**defaults)


class TestSlidingWindowDetector:
    def test_no_events_returns_none(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        detector = SlidingWindowDetector()
        ctx = _make_ctx()
        result = detector.check(journal, ctx)
        assert result is None
        journal.close()

    def test_no_journal_returns_none(self):
        detector = SlidingWindowDetector()
        ctx = _make_ctx()
        result = detector.check(None, ctx)
        assert result is None

    def test_runaway_detection(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        detector = SlidingWindowDetector(max_session_events=10)
        for i in range(15):
            journal.record(f"r{i}", "1.0.0", "sess-1", None, DISPATCH)

        ctx = _make_ctx()
        result = detector.check(journal, ctx)
        assert result is not None
        assert result.signal == StuckSignal.RUNAWAY
        assert result.severity == 1.0
        assert result.recommendation == "halt"
        journal.close()

    def test_repetition_detection(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        detector = SlidingWindowDetector(repetition_threshold=3)

        # Same ritual fails 3 times
        for _ in range(3):
            journal.record("failing_ritual", "1.0.0", "sess-1", None, BODY_FAILED, {"error": "boom"})

        ctx = _make_ctx()
        result = detector.check(journal, ctx)
        assert result is not None
        assert result.signal == StuckSignal.REPETITION
        assert "failing_ritual" in result.details
        journal.close()

    def test_paralysis_detection(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        detector = SlidingWindowDetector(paralysis_threshold=3)

        # 3 BODY_COMPLETE with empty produced
        for i in range(3):
            journal.record(f"r{i}", "1.0.0", "sess-1", None, BODY_COMPLETE, {"produced": {}})

        ctx = _make_ctx()
        result = detector.check(journal, ctx)
        assert result is not None
        assert result.signal == StuckSignal.PARALYSIS
        journal.close()

    def test_oscillation_detection(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        detector = SlidingWindowDetector(window_size=8)

        # A→B→A→B pattern
        for _ in range(2):
            journal.record("ritual_a", "1.0.0", "sess-1", None, BODY_FAILED)
            journal.record("ritual_b", "1.0.0", "sess-1", None, BODY_COMPLETE)

        ctx = _make_ctx()
        result = detector.check(journal, ctx)
        assert result is not None
        assert result.signal == StuckSignal.OSCILLATION
        journal.close()

    def test_healthy_session_returns_none(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        detector = SlidingWindowDetector()

        # Normal healthy sequence
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"produced": {"X": 1}})
        journal.record("r2", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r2", "1.0.0", "sess-1", None, BODY_COMPLETE, {"produced": {"Y": 2}})

        ctx = _make_ctx()
        result = detector.check(journal, ctx)
        assert result is None
        journal.close()

    def test_paralysis_not_triggered_when_producing(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        detector = SlidingWindowDetector(paralysis_threshold=3)

        # 2 empty + 1 producing = not stuck
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"produced": {}})
        journal.record("r2", "1.0.0", "sess-1", None, BODY_COMPLETE, {"produced": {}})
        journal.record("r3", "1.0.0", "sess-1", None, BODY_COMPLETE, {"produced": {"X": 1}})

        ctx = _make_ctx()
        result = detector.check(journal, ctx)
        assert result is None
        journal.close()


class TestStuckDetection:
    def test_detection_fields(self):
        d = StuckDetection(
            signal=StuckSignal.OSCILLATION,
            severity=0.7,
            evidence=["e1", "e2"],
            recommendation="switch_mode",
            details="A ↔ B",
        )
        assert d.signal == StuckSignal.OSCILLATION
        assert d.severity == 0.7
        assert len(d.evidence) == 2
