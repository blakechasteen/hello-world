"""Tests for wired stubs — journal in registry, Yarn, Warp, deferred dispatch."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from hololoom.core.ritual.journal import RitualJournal, DISPATCH, CONDITION_PASS, BODY_START, BODY_COMPLETE, SKIP, ADMISSION_PASS, POLICY_INVOKED, OUTCOME_WRITTEN
from hololoom.core.ritual.registry import RitualRegistry
from hololoom.core.ritual.types import (
    Ritual, RitualBudget, RitualResult, RitualContext,
    HookBinding, Priority, TokenBudget,
)
from hololoom.core.ritual.hooks import HookEvent
from hololoom.core.ritual.policies import FailurePolicy
from hololoom.core.ritual.memory import (
    write_outcome, _write_to_yarn, _embed_to_warp,
    _hash_embedding, _text_to_qdrant_id,
)
from hololoom.core.ritual.types import RitualOutcome
from datetime import datetime, timezone


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="agent-1", agent_role="coder", domain="coding",
        task_id="task-1", task_type="code_modify", task_intent="fix bug",
        task_complexity="standard", yarn=None, warp=None,
        confidence=0.8, confidence_history=[0.8], ritual_outputs={},
        ritual_log=[], token_budget=TokenBudget(total=5000, used=0),
        time_budget=60.0, escalation_path=[], session_id="sess-1",
    )
    defaults.update(overrides)
    return RitualContext(**defaults)


def _make_ritual(ritual_id="test_ritual", body=None, skip_condition=None, failure_policy=FailurePolicy.SKIP):
    async def default_body(ctx):
        return RitualResult(success=True, produced={"X": 1}, confidence_delta=0.01)
    return Ritual(
        id=ritual_id, version="1.0.0", domain="coding", intent="test",
        author="system", consumes=[], produces=[],
        failure_policy=failure_policy,
        budget=RitualBudget(max_tokens=100, max_seconds=5.0, priority_class="standard"),
        body=body or default_body, skip_condition=skip_condition,
    )


def _bind(ritual, trigger=HookEvent.TASK_RECEIVED, priority=Priority.BLOCKING, condition=None):
    return HookBinding(ritual=ritual, trigger=trigger, condition=condition, priority=priority)


# =============================================================================
# Phase 1: Journal integration in registry
# =============================================================================

class TestJournalInRegistry:
    @pytest.mark.asyncio
    async def test_success_produces_full_event_sequence(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        reg = RitualRegistry(journal=journal)
        reg.register(_bind(_make_ritual()))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)

        events = journal.replay_session("sess-1")
        types = [e.event_type for e in events]
        assert types == [DISPATCH, CONDITION_PASS, ADMISSION_PASS, BODY_START, BODY_COMPLETE, OUTCOME_WRITTEN]
        journal.close()

    @pytest.mark.asyncio
    async def test_skip_produces_dispatch_skip(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        reg = RitualRegistry(journal=journal)
        ritual = _make_ritual(skip_condition=lambda ctx: True)
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)

        events = journal.replay_session("sess-1")
        types = [e.event_type for e in events]
        assert DISPATCH in types
        assert SKIP in types
        journal.close()

    @pytest.mark.asyncio
    async def test_condition_fail_produces_dispatch_condition_fail(self, tmp_path):
        journal = RitualJournal(str(tmp_path / "j.db"))
        reg = RitualRegistry(journal=journal)
        ritual = _make_ritual()
        reg.register(_bind(ritual, condition=lambda ctx: False))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)

        events = journal.replay_session("sess-1")
        types = [e.event_type for e in events]
        assert types == [DISPATCH, "CONDITION_FAIL"]
        journal.close()

    @pytest.mark.asyncio
    async def test_failure_produces_policy_invoked(self, tmp_path):
        async def failing(ctx):
            raise ValueError("boom")

        journal = RitualJournal(str(tmp_path / "j.db"))
        reg = RitualRegistry(journal=journal)
        reg.register(_bind(_make_ritual(body=failing)))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)

        events = journal.replay_session("sess-1")
        types = [e.event_type for e in events]
        assert "BODY_FAILED" in types
        assert POLICY_INVOKED in types
        journal.close()

    @pytest.mark.asyncio
    async def test_no_journal_still_works(self):
        """Registry without journal dispatches normally."""
        reg = RitualRegistry(journal=None)
        reg.register(_bind(_make_ritual()))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "test_ritual" in results

    @pytest.mark.asyncio
    async def test_journal_failure_doesnt_block(self, tmp_path):
        """If journal.record() raises, dispatch continues."""
        journal = MagicMock()
        journal.record = MagicMock(side_effect=RuntimeError("db locked"))
        reg = RitualRegistry(journal=journal)
        reg.register(_bind(_make_ritual()))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "test_ritual" in results


# =============================================================================
# Phase 2: Yarn write (Neo4j)
# =============================================================================

class TestYarnWrite:
    @pytest.mark.asyncio
    async def test_writes_to_neo4j_session(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        outcome = RitualOutcome(
            ritual_id="r1", ritual_version="1.0.0", session_id="s1",
            agent_id="a1", task_context="fix bug",
            fired_at=datetime.now(timezone.utc), duration_ms=100,
            tokens_used=42, produced={}, downstream_task_success=None,
            confidence_delta=0.03, was_skipped=False,
            skip_reason=None, failure_policy_invoked=None,
        )
        ctx = _make_ctx(yarn=mock_driver)
        await _write_to_yarn(outcome, ctx)

        assert mock_session.run.call_count == 3  # node + agent edge + session edge

    @pytest.mark.asyncio
    async def test_yarn_none_skips(self):
        outcome = RitualOutcome(
            ritual_id="r1", ritual_version="1.0.0", session_id="s1",
            agent_id="a1", task_context="fix bug",
            fired_at=datetime.now(timezone.utc), duration_ms=100,
            tokens_used=0, produced={}, downstream_task_success=None,
            confidence_delta=0.0, was_skipped=False,
            skip_reason=None, failure_policy_invoked=None,
        )
        ctx = _make_ctx(yarn=None)
        await _write_to_yarn(outcome, ctx)  # no crash

    @pytest.mark.asyncio
    async def test_yarn_failure_doesnt_crash(self):
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("connection refused")

        outcome = RitualOutcome(
            ritual_id="r1", ritual_version="1.0.0", session_id="s1",
            agent_id="a1", task_context="fix",
            fired_at=datetime.now(timezone.utc), duration_ms=0,
            tokens_used=0, produced={}, downstream_task_success=None,
            confidence_delta=0.0, was_skipped=False,
            skip_reason=None, failure_policy_invoked=None,
        )
        ctx = _make_ctx(yarn=mock_driver)
        await _write_to_yarn(outcome, ctx)  # logs warning, no crash


# =============================================================================
# Phase 3: Warp embed (Qdrant)
# =============================================================================

class TestWarpEmbed:
    def test_hash_embedding_dimensions(self):
        emb = _hash_embedding("test text", dim=384)
        assert len(emb) == 384

    def test_hash_embedding_normalized(self):
        emb = _hash_embedding("test text")
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_text_to_qdrant_id(self):
        id1 = _text_to_qdrant_id("abc")
        id2 = _text_to_qdrant_id("abc")
        id3 = _text_to_qdrant_id("def")
        assert id1 == id2  # deterministic
        assert id1 != id3

    @pytest.mark.asyncio
    async def test_embeds_to_qdrant_client(self):
        mock_warp = MagicMock()
        outcome = RitualOutcome(
            ritual_id="r1", ritual_version="1.0.0", session_id="s1",
            agent_id="a1", task_context="fix bug",
            fired_at=datetime.now(timezone.utc), duration_ms=100,
            tokens_used=42, produced={}, downstream_task_success=None,
            confidence_delta=0.03, was_skipped=False,
            skip_reason=None, failure_policy_invoked=None,
        )
        ctx = _make_ctx(warp=mock_warp)
        await _embed_to_warp(outcome, ctx)

        mock_warp.upsert.assert_called_once()
        call_kwargs = mock_warp.upsert.call_args
        assert call_kwargs[1]["collection_name"] == "ritual_outcomes"
        point = call_kwargs[1]["points"][0]
        assert point["payload"]["ritual_id"] == "r1"
        assert len(point["vector"]) == 384

    @pytest.mark.asyncio
    async def test_warp_none_skips(self):
        outcome = RitualOutcome(
            ritual_id="r1", ritual_version="1.0.0", session_id="s1",
            agent_id="a1", task_context="fix",
            fired_at=datetime.now(timezone.utc), duration_ms=0,
            tokens_used=0, produced={}, downstream_task_success=None,
            confidence_delta=0.0, was_skipped=False,
            skip_reason=None, failure_policy_invoked=None,
        )
        ctx = _make_ctx(warp=None)
        await _embed_to_warp(outcome, ctx)  # no crash


# =============================================================================
# Phase 6: Deferred dispatch
# =============================================================================

class TestDeferredDispatch:
    @pytest.mark.asyncio
    async def test_deferred_calls_queue(self):
        queue_calls = []
        def mock_queue(binding, ctx):
            queue_calls.append(binding.ritual.id)

        reg = RitualRegistry(deferred_queue=mock_queue)
        ritual = _make_ritual(ritual_id="deferred_r")
        reg.register(_bind(ritual, priority=Priority.DEFERRED))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)

        assert queue_calls == ["deferred_r"]

    @pytest.mark.asyncio
    async def test_no_queue_drops_silently(self):
        reg = RitualRegistry(deferred_queue=None)
        ritual = _make_ritual(ritual_id="deferred_r")
        reg.register(_bind(ritual, priority=Priority.DEFERRED))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)  # no crash

    @pytest.mark.asyncio
    async def test_queue_failure_doesnt_crash(self):
        def broken_queue(binding, ctx):
            raise RuntimeError("queue full")

        reg = RitualRegistry(deferred_queue=broken_queue)
        ritual = _make_ritual(ritual_id="deferred_r")
        reg.register(_bind(ritual, priority=Priority.DEFERRED))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)  # logs warning, no crash
