"""Tests for the event-sourced RitualJournal."""

import pytest

from hololoom.core.ritual.journal import (
    RitualJournal,
    RitualJournalEvent,
    DISPATCH,
    CONDITION_PASS,
    BODY_START,
    BODY_COMPLETE,
    BODY_FAILED,
    BODY_HEARTBEAT,
    SKIP,
    POLICY_INVOKED,
    OUTCOME_WRITTEN,
)


@pytest.fixture
def journal(tmp_path):
    j = RitualJournal(str(tmp_path / "test_journal.db"))
    yield j
    j.close()


class TestJournalRecord:
    def test_record_returns_event_id(self, journal):
        eid = journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        assert isinstance(eid, str)
        assert len(eid) == 12

    def test_record_with_payload(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"score": 9.8})
        events = journal.replay_ritual("r1", "sess-1")
        assert len(events) == 1
        assert events[0].payload["score"] == 9.8

    def test_record_multiple_events(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, CONDITION_PASS)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_START)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE)
        journal.record("r1", "1.0.0", "sess-1", None, OUTCOME_WRITTEN)
        events = journal.replay_ritual("r1", "sess-1")
        assert len(events) == 5
        types = [e.event_type for e in events]
        assert types == [DISPATCH, CONDITION_PASS, BODY_START, BODY_COMPLETE, OUTCOME_WRITTEN]


class TestJournalReplay:
    def test_replay_filters_by_ritual_and_session(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r2", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-2", None, DISPATCH)

        events = journal.replay_ritual("r1", "sess-1")
        assert len(events) == 1
        assert events[0].ritual_id == "r1"

    def test_replay_session_returns_all(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r2", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r3", "1.0.0", "sess-1", None, SKIP)

        events = journal.replay_session("sess-1")
        assert len(events) == 3

    def test_replay_from_correlation(self, journal):
        journal.record("r1", "1.0.0", "sess-1", "corr-42", DISPATCH)
        journal.record("r2", "1.0.0", "sess-1", "corr-42", DISPATCH)
        journal.record("r3", "1.0.0", "sess-1", "corr-99", DISPATCH)

        events = journal.replay_from_correlation("corr-42")
        assert len(events) == 2
        assert all(e.correlation_id == "corr-42" for e in events)

    def test_replay_empty_returns_empty(self, journal):
        events = journal.replay_ritual("nonexistent", "sess-1")
        assert events == []

    def test_replay_ordered_by_time(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_START)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE)

        events = journal.replay_ritual("r1", "sess-1")
        types = [e.event_type for e in events]
        assert types == [DISPATCH, BODY_START, BODY_COMPLETE]


class TestJournalCount:
    def test_count_all(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE)
        assert journal.count_events() == 2

    def test_count_by_ritual(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r2", "1.0.0", "sess-1", None, DISPATCH)
        assert journal.count_events(ritual_id="r1") == 1

    def test_count_by_event_type(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, SKIP)
        journal.record("r2", "1.0.0", "sess-1", None, SKIP)
        assert journal.count_events(event_type=SKIP) == 2


class TestJournalEvent:
    def test_event_to_dict(self, journal):
        journal.record("r1", "1.0.0", "sess-1", "corr-1", DISPATCH, {"key": "val"})
        events = journal.replay_ritual("r1", "sess-1")
        d = events[0].to_dict()
        assert d["ritual_id"] == "r1"
        assert d["event_type"] == DISPATCH
        assert d["payload"]["key"] == "val"

    def test_in_memory_journal(self):
        """In-memory journal for testing."""
        j = RitualJournal(":memory:")
        j.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        assert j.count_events() == 1
        j.close()
