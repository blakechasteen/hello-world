"""Tests for ThreadPrimer — pre-inlined context snapshots."""

import pytest

from hololoom.core.ritual.snapshot import ContextSnapshot, JournalSnapshotBuilder
from hololoom.core.ritual.journal import RitualJournal, BODY_COMPLETE, DISPATCH
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


class TestContextSnapshot:
    def test_default_empty(self):
        s = ContextSnapshot()
        assert s.recent_outcomes == []
        assert s.session_history == []
        assert s.skill_hint is None
        assert s.token_cost == 0

    def test_with_data(self):
        s = ContextSnapshot(
            recent_outcomes=[{"ritual_id": "r1", "success": True}],
            skill_hint={"skill_id": "s1", "steps": []},
            token_cost=100,
        )
        assert len(s.recent_outcomes) == 1
        assert s.skill_hint["skill_id"] == "s1"


class TestJournalSnapshotBuilder:
    def test_build_empty_journal(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        builder = JournalSnapshotBuilder(journal=journal)
        ctx = _make_ctx()
        snapshot = builder.build(ctx, active_rules=[], budget_tokens=200)
        assert isinstance(snapshot, ContextSnapshot)
        assert snapshot.session_history == []
        journal.close()

    def test_build_with_journal_events(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"produced": ["X"]})

        builder = JournalSnapshotBuilder(journal=journal, max_recent=5)
        ctx = _make_ctx()
        snapshot = builder.build(ctx, active_rules=[], budget_tokens=200)

        assert len(snapshot.session_history) == 2
        assert snapshot.session_history[0]["event_type"] == DISPATCH
        assert snapshot.token_cost > 0
        journal.close()

    def test_build_no_journal(self):
        builder = JournalSnapshotBuilder(journal=None)
        ctx = _make_ctx()
        snapshot = builder.build(ctx, budget_tokens=200)
        assert isinstance(snapshot, ContextSnapshot)

    def test_build_includes_domain_priors(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        builder = JournalSnapshotBuilder(journal=journal)
        ctx = _make_ctx(confidence=0.65, domain="beekeeping")
        snapshot = builder.build(ctx, budget_tokens=200)
        assert "beekeeping" in snapshot.domain_priors
        assert snapshot.domain_priors["beekeeping"] == 0.65
        journal.close()

    def test_build_respects_max_recent(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        for i in range(10):
            journal.record(f"r{i}", "1.0.0", "sess-1", None, DISPATCH)

        builder = JournalSnapshotBuilder(journal=journal, max_recent=3)
        ctx = _make_ctx()
        snapshot = builder.build(ctx, budget_tokens=500)
        assert len(snapshot.session_history) == 3
        journal.close()
