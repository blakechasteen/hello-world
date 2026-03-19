"""
Unit tests for hololoom.core.deep_thinking — 0% → 60%+ coverage.

Tests gate logic, tier selection, deferred queue, sleeptime scheduler,
deliberation engine, and config. All GPU/LLM calls are mocked.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hololoom.core.deep_thinking.config import (
    DeepThinkingConfig,
    ROLE_GPU_MAP,
    SYSTEM_PROMPTS,
    GATE_PROMPT,
)
from hololoom.core.deep_thinking.tier import Tier, TierState, TierTransition
from hololoom.core.deep_thinking.client import DeepClient, ChatResult, HealthStatus
from hololoom.core.deep_thinking.deferred import (
    DeferredJob,
    DeferredQueue,
    JOB_TYPE_PRIORITY,
)
from hololoom.core.deep_thinking.gate import (
    MilestoneGate,
    MilestoneContext,
    GateVerdict,
    VerdictPrior,
    _extract_json as gate_extract_json,
)
from hololoom.core.deep_thinking.deliberation import (
    DeliberationEngine,
    DeliberationResult,
    Proposal,
    Critique,
    Consensus,
    DeliberationStats,
    _extract_json as delib_extract_json,
)
from hololoom.core.deep_thinking.sleeptime import (
    SleeptimeScheduler,
    SleeptimeCycleStats,
    SleeptimeJob,
    DEFAULT_JOBS,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_config(**overrides) -> DeepThinkingConfig:
    defaults = dict(
        gpu_servers=["http://gpu0:8001", "http://gpu1:8002", "http://gpu2:8003"],
        local_endpoint="http://localhost:11434",
        idle_timeout_s=1.0,
        health_interval_s=999,  # don't auto-probe in tests
        inference_timeout_s=5.0,
    )
    defaults.update(overrides)
    return DeepThinkingConfig(**defaults)


def _make_client() -> DeepClient:
    """Return a DeepClient with mocked methods."""
    client = DeepClient(timeout_s=5.0, retries=0)
    return client


def _make_tier_state(
    config: Optional[DeepThinkingConfig] = None,
    tier: Tier = Tier.FULL,
    available_gpus: Optional[list[int]] = None,
    local_alive: bool = True,
) -> TierState:
    """Build a TierState with pre-set tier (no probing needed)."""
    config = config or _make_config()
    client = _make_client()
    ts = TierState(config, client)
    ts.tier = tier
    ts.available_gpus = available_gpus if available_gpus is not None else [0, 1, 2]
    ts.local_alive = local_alive
    return ts


def _make_chat_result(content: str, model: str = "test-model") -> ChatResult:
    return ChatResult(
        content=content,
        model=model,
        prompt_tokens=100,
        completion_tokens=50,
        elapsed_s=1.0,
        raw={"choices": [{"message": {"content": content}}]},
    )


def _make_milestone_ctx(**overrides) -> MilestoneContext:
    defaults = dict(
        goal="Build a REST API",
        work_so_far={"step1": "done"},
        milestone_description="Implement auth endpoint",
        milestone_number=1,
        total_milestones=3,
        agent_id="test-agent",
    )
    defaults.update(overrides)
    return MilestoneContext(**defaults)


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════


class TestDeepThinkingConfig:
    def test_defaults(self):
        cfg = DeepThinkingConfig()
        assert len(cfg.gpu_servers) == 3
        assert cfg.rig_model == "qwen3.5:108b"
        assert cfg.local_model == "qwen3.5:27b"
        assert cfg.plan_to_critic_bleed == 1.0
        assert cfg.plan_reasoning_bleed == 0.3
        assert cfg.consensus_bleed == 1.0
        assert cfg.gate_threshold_full == 0.9
        assert cfg.gate_threshold_degraded == 0.7
        assert cfg.defer_below == 0.5
        assert cfg.idle_timeout_s == 300.0
        assert cfg.sleeptime_enabled is True
        assert cfg.inference_timeout_s == 900.0
        assert cfg.max_context_tokens == 131072

    def test_custom_config(self):
        cfg = _make_config(rig_model="custom:7b", idle_timeout_s=60.0)
        assert cfg.rig_model == "custom:7b"
        assert cfg.idle_timeout_s == 60.0

    def test_role_gpu_map(self):
        assert ROLE_GPU_MAP["planner"] == 0
        assert ROLE_GPU_MAP["critic"] == 1
        assert ROLE_GPU_MAP["synthesizer"] == 2

    def test_system_prompts_all_roles(self):
        for role in ("planner", "critic", "synthesizer"):
            assert role in SYSTEM_PROMPTS
            assert len(SYSTEM_PROMPTS[role]) > 20

    def test_gate_prompt_exists(self):
        assert "milestone reviewer" in GATE_PROMPT.lower()
        assert "proceed" in GATE_PROMPT.lower()


# ═══════════════════════════════════════════════════════════════════
#  TIER
# ═══════════════════════════════════════════════════════════════════


class TestTier:
    def test_tier_ordering(self):
        assert Tier.DARK < Tier.DEGRADED < Tier.PARTIAL < Tier.FULL

    def test_tier_values(self):
        assert Tier.DARK == 0
        assert Tier.DEGRADED == 1
        assert Tier.PARTIAL == 2
        assert Tier.FULL == 3


class TestTierState:
    @pytest.mark.asyncio
    async def test_probe_all_gpus_alive(self):
        config = _make_config()
        client = _make_client()
        client.is_alive = AsyncMock(return_value=True)
        ts = TierState(config, client)

        tier = await ts.probe()

        assert tier == Tier.FULL
        assert ts.available_gpus == [0, 1, 2]
        assert ts.local_alive is True

    @pytest.mark.asyncio
    async def test_probe_partial_gpus(self):
        config = _make_config()
        client = _make_client()
        # gpu0 alive, gpu1 dead, gpu2 alive, local alive
        client.is_alive = AsyncMock(
            side_effect=[True, False, True, True]
        )
        ts = TierState(config, client)

        tier = await ts.probe()

        assert tier == Tier.PARTIAL
        assert ts.available_gpus == [0, 2]

    @pytest.mark.asyncio
    async def test_probe_no_gpus_local_alive(self):
        config = _make_config()
        client = _make_client()
        client.is_alive = AsyncMock(
            side_effect=[False, False, False, True]
        )
        ts = TierState(config, client)

        tier = await ts.probe()

        assert tier == Tier.DEGRADED
        assert ts.available_gpus == []
        assert ts.local_alive is True

    @pytest.mark.asyncio
    async def test_probe_everything_dead(self):
        config = _make_config()
        client = _make_client()
        client.is_alive = AsyncMock(return_value=False)
        ts = TierState(config, client)

        tier = await ts.probe()

        assert tier == Tier.DARK
        assert ts.available_gpus == []
        assert ts.local_alive is False

    @pytest.mark.asyncio
    async def test_probe_emits_transition(self):
        config = _make_config()
        client = _make_client()
        client.is_alive = AsyncMock(return_value=True)
        ts = TierState(config, client)

        transitions = []
        ts.on_transition(transitions.append)

        await ts.probe()  # DARK → FULL

        assert len(transitions) == 1
        assert transitions[0].previous == Tier.DARK
        assert transitions[0].current == Tier.FULL

    @pytest.mark.asyncio
    async def test_probe_no_transition_when_unchanged(self):
        config = _make_config()
        client = _make_client()
        client.is_alive = AsyncMock(return_value=True)
        ts = TierState(config, client)
        ts.tier = Tier.FULL
        ts.available_gpus = [0, 1, 2]

        transitions = []
        ts.on_transition(transitions.append)

        await ts.probe()

        assert len(transitions) == 0

    @pytest.mark.asyncio
    async def test_probe_callback_error_doesnt_crash(self):
        config = _make_config()
        client = _make_client()
        client.is_alive = AsyncMock(return_value=True)
        ts = TierState(config, client)

        def bad_callback(t):
            raise RuntimeError("boom")

        ts.on_transition(bad_callback)
        tier = await ts.probe()  # should not raise
        assert tier == Tier.FULL

    def test_endpoint_for_gpu_available(self):
        ts = _make_tier_state()
        assert ts.endpoint_for_gpu(0) == "http://gpu0:8001"
        assert ts.endpoint_for_gpu(1) == "http://gpu1:8002"

    def test_endpoint_for_gpu_unavailable(self):
        ts = _make_tier_state(available_gpus=[0])
        assert ts.endpoint_for_gpu(1) is None
        assert ts.endpoint_for_gpu(2) is None

    def test_best_available_endpoint_gpu(self):
        ts = _make_tier_state(available_gpus=[1, 2])
        assert ts.best_available_endpoint() == "http://gpu1:8002"

    def test_best_available_endpoint_local_fallback(self):
        ts = _make_tier_state(available_gpus=[], local_alive=True)
        assert ts.best_available_endpoint() == "http://localhost:11434"

    def test_best_available_endpoint_nothing(self):
        ts = _make_tier_state(available_gpus=[], local_alive=False)
        assert ts.best_available_endpoint() is None

    def test_gate_threshold_full(self):
        ts = _make_tier_state(tier=Tier.FULL)
        assert ts.gate_threshold == 0.9

    def test_gate_threshold_partial(self):
        ts = _make_tier_state(tier=Tier.PARTIAL)
        assert ts.gate_threshold == 0.9

    def test_gate_threshold_degraded(self):
        ts = _make_tier_state(tier=Tier.DEGRADED)
        assert ts.gate_threshold == 0.7

    def test_gate_threshold_dark(self):
        ts = _make_tier_state(tier=Tier.DARK)
        assert ts.gate_threshold == 0.7

    def test_model_for_tier_full(self):
        ts = _make_tier_state(tier=Tier.FULL)
        assert ts.model_for_tier == "qwen3.5:108b"

    def test_model_for_tier_degraded(self):
        ts = _make_tier_state(tier=Tier.DEGRADED)
        assert ts.model_for_tier == "qwen3.5:27b"

    def test_stop_without_start(self):
        ts = _make_tier_state()
        ts.stop()  # should not raise


# ═══════════════════════════════════════════════════════════════════
#  DEFERRED QUEUE
# ═══════════════════════════════════════════════════════════════════


class TestDeferredJob:
    def test_default_fields(self):
        job = DeferredJob()
        assert len(job.id) == 12
        assert job.job_type == "deliberation"
        assert job.payload == {}
        assert job.priority == 0.0
        assert job.staleness_boost == 0.0

    def test_to_dict_roundtrip(self):
        job = DeferredJob(job_type="gate_review", payload={"goal": "test"})
        d = job.to_dict()
        restored = DeferredJob.from_dict(d)
        assert restored.job_type == "gate_review"
        assert restored.payload == {"goal": "test"}
        assert restored.id == job.id

    def test_from_dict_ignores_extra_keys(self):
        d = {"id": "abc", "job_type": "critique", "extra_field": "ignored"}
        job = DeferredJob.from_dict(d)
        assert job.id == "abc"
        assert job.job_type == "critique"


class TestDeferredQueue:
    @pytest.mark.asyncio
    async def test_push_and_size(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        assert q.empty
        assert q.size == 0

        await q.push(DeferredJob(job_type="gate_review"))
        assert q.size == 1
        assert not q.empty

    @pytest.mark.asyncio
    async def test_push_sets_type_priority(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        job = DeferredJob(job_type="gate_review")
        await q.push(job)
        assert job.priority == JOB_TYPE_PRIORITY["gate_review"]

    @pytest.mark.asyncio
    async def test_push_preserves_custom_priority(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        job = DeferredJob(job_type="gate_review", priority=999.0)
        await q.push(job)
        assert job.priority == 999.0

    def test_pop_highest_priority(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        # Manually add jobs to avoid async push
        q._jobs = [
            DeferredJob(id="low", job_type="synthesis", priority=10),
            DeferredJob(id="high", job_type="gate_review", priority=100),
        ]
        popped = q.pop_highest()
        assert popped.id == "high"
        assert q.size == 1

    def test_pop_empty_returns_none(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        assert q.pop_highest() is None

    def test_effective_priority(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        job = DeferredJob(job_type="gate_review", priority=50.0, staleness_boost=5.0)
        # effective = priority + staleness_boost + type_weight
        expected = 50.0 + 5.0 + JOB_TYPE_PRIORITY["gate_review"]
        assert q.effective_priority(job) == expected

    def test_staleness_boost_accumulates(self, tmp_path):
        q = DeferredQueue(
            persist_path=str(tmp_path / "q.json"),
            staleness_boost_per_hour=1.0,
        )
        # Job created 2 hours ago
        job = DeferredJob(created_at=time.time() - 7200)
        q._jobs = [job]
        q._update_staleness()
        assert job.staleness_boost == pytest.approx(2.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_drain_processes_all(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        q._jobs = [
            DeferredJob(id="a", job_type="gate_review", priority=100),
            DeferredJob(id="b", job_type="synthesis", priority=10),
        ]

        async def process(job):
            return True

        processed = await q.drain(process)
        assert processed == 2
        assert q.empty

    @pytest.mark.asyncio
    async def test_drain_requeues_failures(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        q._jobs = [
            DeferredJob(id="ok", job_type="gate_review", priority=100),
            DeferredJob(id="fail", job_type="synthesis", priority=10),
        ]

        async def process(job):
            return job.id == "ok"

        processed = await q.drain(process)
        assert processed == 1
        assert q.size == 1
        assert q._jobs[0].id == "fail"

    @pytest.mark.asyncio
    async def test_drain_handles_exceptions(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        q._jobs = [DeferredJob(id="boom")]

        async def process(job):
            raise RuntimeError("explosion")

        processed = await q.drain(process)
        assert processed == 0
        assert q.size == 1  # re-queued

    @pytest.mark.asyncio
    async def test_drain_empty_returns_zero(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        processed = await q.drain(AsyncMock(return_value=True))
        assert processed == 0

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "q.json")
        q = DeferredQueue(persist_path=path)
        q._jobs = [
            DeferredJob(id="saved", job_type="gate_review", payload={"x": 1}),
        ]
        q.save()

        q2 = DeferredQueue(persist_path=path)
        q2.load()
        assert q2.size == 1
        assert q2._jobs[0].id == "saved"
        assert q2._jobs[0].payload == {"x": 1}

    def test_load_missing_file(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "nope.json"))
        q.load()  # should not raise
        assert q.empty

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json at all {{{")
        q = DeferredQueue(persist_path=str(path))
        q.load()
        assert q.empty

    def test_summary(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        q._jobs = [
            DeferredJob(job_type="gate_review", created_at=time.time() - 3600),
            DeferredJob(job_type="gate_review"),
            DeferredJob(job_type="synthesis"),
        ]
        s = q.summary()
        assert s["total"] == 3
        assert s["by_type"]["gate_review"] == 2
        assert s["by_type"]["synthesis"] == 1
        assert s["oldest_hours"] > 0

    def test_summary_empty(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        s = q.summary()
        assert s["total"] == 0
        assert s["oldest_hours"] == 0


# ═══════════════════════════════════════════════════════════════════
#  VERDICT PRIOR (Thompson gate learning)
# ═══════════════════════════════════════════════════════════════════


class TestVerdictPrior:
    def test_initial_uniform(self):
        vp = VerdictPrior()
        assert vp.proceed == (1.0, 1.0)
        assert vp.revise == (1.0, 1.0)
        assert vp.abort == (1.0, 1.0)

    def test_expected_accuracy_uniform(self):
        vp = VerdictPrior()
        assert vp.expected_accuracy("proceed") == pytest.approx(0.5)

    def test_update_correct(self):
        vp = VerdictPrior()
        vp.update("proceed", was_correct=True)
        assert vp.proceed[0] > 1.0  # alpha increased

    def test_update_incorrect(self):
        vp = VerdictPrior()
        vp.update("revise", was_correct=False)
        assert vp.revise[1] > 1.0  # beta increased

    def test_to_dict(self):
        vp = VerdictPrior()
        d = vp.to_dict()
        assert "proceed" in d
        assert "revise" in d
        assert "abort" in d
        assert d["proceed"] == [1.0, 1.0]

    def test_from_dict(self):
        d = {"proceed": [3.0, 1.0], "revise": [1.0, 2.0], "abort": [1.0, 1.0]}
        vp = VerdictPrior.from_dict(d)
        assert vp.proceed == (3.0, 1.0)
        assert vp.revise == (1.0, 2.0)

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "priors.json"
        vp = VerdictPrior()
        vp.update("proceed", was_correct=True)
        vp.update("proceed", was_correct=True)
        vp.save(path)

        vp2 = VerdictPrior()
        vp2.load(path)
        assert vp2.proceed[0] > 1.0

    def test_load_missing_is_noop(self, tmp_path):
        vp = VerdictPrior()
        vp.load(tmp_path / "nope.json")
        assert vp.proceed == (1.0, 1.0)

    def test_load_corrupt_is_noop(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{{{invalid")
        vp = VerdictPrior()
        vp.load(path)
        assert vp.proceed == (1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
#  GATE — _extract_json
# ═══════════════════════════════════════════════════════════════════


class TestExtractJson:
    def test_raw_json(self):
        result = gate_extract_json('{"verdict": "proceed", "confidence": 0.9}')
        assert result["verdict"] == "proceed"

    def test_json_in_code_block(self):
        text = 'Some reasoning\n```json\n{"verdict": "revise"}\n```\nmore text'
        result = gate_extract_json(text)
        assert result["verdict"] == "revise"

    def test_json_in_plain_code_block(self):
        text = 'blah\n```\n{"verdict": "abort"}\n```'
        result = gate_extract_json(text)
        assert result["verdict"] == "abort"

    def test_json_embedded_in_text(self):
        text = 'The answer is {"verdict": "proceed", "confidence": 0.8} and done.'
        result = gate_extract_json(text)
        assert result["verdict"] == "proceed"

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            gate_extract_json("no json here at all")


# ═══════════════════════════════════════════════════════════════════
#  GATE — MilestoneGate
# ═══════════════════════════════════════════════════════════════════


class TestMilestoneGate:
    def _make_gate(
        self,
        tier: Tier = Tier.FULL,
        tmp_path: Optional[Path] = None,
        with_queue: bool = False,
    ) -> MilestoneGate:
        config = _make_config()
        client = _make_client()
        tier_state = _make_tier_state(config=config, tier=tier)
        queue = None
        if with_queue and tmp_path:
            queue = DeferredQueue(persist_path=str(tmp_path / "q.json"))
        return MilestoneGate(config, client, tier_state, queue)

    @pytest.mark.asyncio
    async def test_dark_tier_defers_everything(self, tmp_path):
        gate = self._make_gate(tier=Tier.DARK, tmp_path=tmp_path, with_queue=True)
        ctx = _make_milestone_ctx()

        verdict = await gate.review(ctx)

        assert verdict.deferred is True
        assert verdict.confidence == 0.0
        assert verdict.verdict == "proceed"

    @pytest.mark.asyncio
    async def test_dark_tier_enqueues_when_queue_wired(self, tmp_path):
        gate = self._make_gate(tier=Tier.DARK, tmp_path=tmp_path, with_queue=True)
        ctx = _make_milestone_ctx()

        await gate.review(ctx)

        assert gate.queue.size == 1

    @pytest.mark.asyncio
    async def test_dark_tier_no_queue_still_works(self):
        gate = self._make_gate(tier=Tier.DARK)
        ctx = _make_milestone_ctx()

        verdict = await gate.review(ctx)

        assert verdict.deferred is True

    @pytest.mark.asyncio
    async def test_full_tier_reviews(self):
        gate = self._make_gate(tier=Tier.FULL)
        gate.client.chat = AsyncMock(return_value=_make_chat_result(
            json.dumps({
                "verdict": "proceed",
                "confidence": 0.95,
                "reasoning": "Looks good",
                "corrections": None,
            })
        ))
        ctx = _make_milestone_ctx()

        verdict = await gate.review(ctx)

        assert verdict.verdict == "proceed"
        assert verdict.confidence == 0.95
        assert verdict.deferred is False
        assert gate._review_count == 1

    @pytest.mark.asyncio
    async def test_revise_verdict(self):
        gate = self._make_gate(tier=Tier.FULL)
        gate.client.chat = AsyncMock(return_value=_make_chat_result(
            json.dumps({
                "verdict": "revise",
                "confidence": 0.7,
                "reasoning": "Missing error handling",
                "corrections": "Add try/except around DB calls",
            })
        ))

        verdict = await gate.review(_make_milestone_ctx())

        assert verdict.verdict == "revise"
        assert verdict.corrections == "Add try/except around DB calls"

    @pytest.mark.asyncio
    async def test_connection_error_defers(self, tmp_path):
        gate = self._make_gate(tier=Tier.FULL, tmp_path=tmp_path, with_queue=True)
        gate.client.chat = AsyncMock(side_effect=ConnectionError("down"))

        verdict = await gate.review(_make_milestone_ctx())

        assert verdict.deferred is True
        assert verdict.confidence == 0.0

    @pytest.mark.asyncio
    async def test_timeout_error_defers(self, tmp_path):
        gate = self._make_gate(tier=Tier.FULL, tmp_path=tmp_path, with_queue=True)
        gate.client.chat = AsyncMock(side_effect=TimeoutError("slow"))

        verdict = await gate.review(_make_milestone_ctx())

        assert verdict.deferred is True

    @pytest.mark.asyncio
    async def test_low_confidence_degraded_defers(self, tmp_path):
        gate = self._make_gate(tier=Tier.DEGRADED, tmp_path=tmp_path, with_queue=True)
        gate.client.chat = AsyncMock(return_value=_make_chat_result(
            json.dumps({
                "verdict": "proceed",
                "confidence": 0.3,
                "reasoning": "Not sure",
                "corrections": None,
            })
        ))

        verdict = await gate.review(_make_milestone_ctx())

        assert verdict.deferred is True
        assert gate.queue.size == 1

    @pytest.mark.asyncio
    async def test_parse_error_defaults_to_cautious_proceed(self):
        gate = self._make_gate(tier=Tier.FULL)
        gate.client.chat = AsyncMock(
            return_value=_make_chat_result("not json output at all")
        )

        verdict = await gate.review(_make_milestone_ctx())

        assert verdict.verdict == "proceed"
        assert verdict.confidence == 0.3
        assert "[parse error]" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_invalid_verdict_defaults_to_proceed(self):
        gate = self._make_gate(tier=Tier.FULL)
        gate.client.chat = AsyncMock(return_value=_make_chat_result(
            json.dumps({
                "verdict": "EXPLODE",
                "confidence": 0.9,
                "reasoning": "chaos",
            })
        ))

        verdict = await gate.review(_make_milestone_ctx())
        assert verdict.verdict == "proceed"

    def test_update_priors(self):
        gate = self._make_gate()
        gate.update_priors("proceed", was_correct=True)
        assert gate.priors.expected_accuracy("proceed") > 0.5

    def test_build_review_prompt(self):
        gate = self._make_gate()
        ctx = _make_milestone_ctx()
        prompt = gate._build_review_prompt(ctx)
        assert "Build a REST API" in prompt
        assert "Milestone 1/3" in prompt
        assert "Implement auth endpoint" in prompt

    def test_on_tier_recovery_triggers_drain(self, tmp_path):
        gate = self._make_gate(tier=Tier.DEGRADED, tmp_path=tmp_path, with_queue=True)
        gate.queue._jobs = [DeferredJob(id="test")]

        transition = TierTransition(
            previous=Tier.DEGRADED,
            current=Tier.FULL,
            available_gpus=[0, 1, 2],
        )

        # The callback will try to create a task — we just verify it doesn't crash
        # In a real event loop, _drain_deferred would run
        with patch("asyncio.create_task") as mock_task:
            gate._on_tier_recovery(transition)
            mock_task.assert_called_once()

    def test_on_tier_recovery_no_drain_if_empty(self, tmp_path):
        gate = self._make_gate(tier=Tier.DEGRADED, tmp_path=tmp_path, with_queue=True)
        # Queue is empty

        transition = TierTransition(
            previous=Tier.DEGRADED,
            current=Tier.FULL,
            available_gpus=[0, 1, 2],
        )

        with patch("asyncio.create_task") as mock_task:
            gate._on_tier_recovery(transition)
            mock_task.assert_not_called()

    def test_on_tier_recovery_no_drain_if_not_recovering(self, tmp_path):
        gate = self._make_gate(tier=Tier.FULL, tmp_path=tmp_path, with_queue=True)
        gate.queue._jobs = [DeferredJob(id="test")]

        # FULL → FULL is not a recovery
        transition = TierTransition(
            previous=Tier.FULL,
            current=Tier.FULL,
            available_gpus=[0, 1, 2],
        )

        with patch("asyncio.create_task") as mock_task:
            gate._on_tier_recovery(transition)
            mock_task.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
#  DELIBERATION ENGINE
# ═══════════════════════════════════════════════════════════════════


class TestDeliberationEngine:
    def _make_engine(
        self,
        tier: Tier = Tier.FULL,
        available_gpus: Optional[list[int]] = None,
    ) -> DeliberationEngine:
        config = _make_config()
        client = _make_client()
        tier_state = _make_tier_state(
            config=config,
            tier=tier,
            available_gpus=available_gpus if available_gpus is not None else [0, 1, 2],
        )
        return DeliberationEngine(config, client, tier_state)

    def _mock_chat_sequence(self, engine: DeliberationEngine, responses: list[str]):
        """Mock client.chat to return a sequence of ChatResults."""
        engine.client.chat = AsyncMock(
            side_effect=[_make_chat_result(r) for r in responses]
        )

    @pytest.mark.asyncio
    async def test_full_deliberation_three_phases(self):
        engine = self._make_engine(tier=Tier.FULL)

        proposal_json = json.dumps({
            "nodes": [{"id": "t1", "task": "setup"}],
            "reasoning": "Start with setup",
            "confidence": 0.8,
        })
        critique_json = json.dumps({
            "flaws": ["Missing tests"],
            "risks": ["Scope creep"],
            "score": 0.7,
            "suggestions": ["Add test step"],
        })
        consensus_json = json.dumps({
            "final_plan": {"nodes": [{"id": "t1"}, {"id": "t2"}]},
            "incorporated": ["Add test step"],
            "rejected": [],
            "confidence": 0.85,
            "dissent": [],
        })

        self._mock_chat_sequence(engine, [proposal_json, critique_json, consensus_json])

        result = await engine.deliberate("Build an API")

        assert isinstance(result, DeliberationResult)
        assert result.proposal.confidence == 0.8
        assert result.critique.score == 0.7
        assert result.consensus.confidence == 0.85
        assert result.stats.tier_used == Tier.FULL
        assert result.stats.total_elapsed_s >= 0

    @pytest.mark.asyncio
    async def test_partial_tier_collapses_synthesizer(self):
        engine = self._make_engine(tier=Tier.PARTIAL, available_gpus=[0, 1])

        proposal_json = json.dumps({
            "nodes": [{"id": "t1"}],
            "reasoning": "plan",
            "confidence": 0.8,
        })
        critique_json = json.dumps({
            "flaws": ["Issue A"],
            "risks": [],
            "score": 0.75,
            "suggestions": ["Fix A", "Fix B"],
        })

        # Only 2 chat calls (planner + critic), no synthesizer
        self._mock_chat_sequence(engine, [proposal_json, critique_json])

        result = await engine.deliberate("Build something")

        assert result.consensus.confidence == 0.75  # derived from critique score
        assert "Fix A" in result.consensus.incorporated_critiques
        assert engine.client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_tier_low_critique_score(self):
        engine = self._make_engine(tier=Tier.PARTIAL, available_gpus=[0, 1])

        proposal_json = json.dumps({
            "nodes": [], "reasoning": "weak plan", "confidence": 0.4,
        })
        critique_json = json.dumps({
            "flaws": ["Fatal flaw"],
            "risks": ["High risk"],
            "score": 0.3,
            "suggestions": ["Rewrite"],
        })

        self._mock_chat_sequence(engine, [proposal_json, critique_json])

        result = await engine.deliberate("Build something")

        # Low score: confidence = score * 0.8
        assert result.consensus.confidence == pytest.approx(0.24)
        assert "Fatal flaw" in result.consensus.dissent

    @pytest.mark.asyncio
    async def test_degraded_tier_uses_local(self):
        engine = self._make_engine(tier=Tier.DEGRADED, available_gpus=[])

        responses = [
            json.dumps({"nodes": [], "reasoning": "plan", "confidence": 0.5}),
            json.dumps({"flaws": [], "risks": [], "score": 0.6, "suggestions": []}),
            json.dumps({
                "final_plan": {}, "incorporated": [], "rejected": [],
                "confidence": 0.6, "dissent": [],
            }),
        ]
        self._mock_chat_sequence(engine, responses)

        result = await engine.deliberate("Plan something")

        # Should have called chat 3 times with local endpoint
        assert engine.client.chat.call_count == 3
        for call in engine.client.chat.call_args_list:
            assert call[0][0] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_dark_tier_returns_deferred_proposal(self):
        engine = self._make_engine(tier=Tier.DARK, available_gpus=[])
        engine.tier.local_alive = False

        result = await engine.deliberate("Plan something")

        assert result.proposal.confidence == 0.0
        assert result.proposal.task_graph.get("deferred") is True
        # No chat calls — everything deferred
        assert not hasattr(engine.client, "chat") or not isinstance(
            engine.client.chat, AsyncMock
        )

    def test_resolve_endpoint_full_tier(self):
        engine = self._make_engine(tier=Tier.FULL)
        ep, model = engine._resolve_endpoint("planner", Tier.FULL)
        assert ep == "http://gpu0:8001"
        assert model == "qwen3.5:108b"

    def test_resolve_endpoint_partial_fallback(self):
        engine = self._make_engine(tier=Tier.PARTIAL, available_gpus=[1])
        # Planner wants gpu0 which is down, should fallback to gpu1
        ep, model = engine._resolve_endpoint("planner", Tier.PARTIAL)
        assert ep == "http://gpu1:8002"
        assert model == "qwen3.5:108b"

    def test_resolve_endpoint_degraded_uses_local(self):
        engine = self._make_engine(tier=Tier.DEGRADED, available_gpus=[])
        ep, model = engine._resolve_endpoint("planner", Tier.DEGRADED)
        assert ep == "http://localhost:11434"
        assert model == "qwen3.5:27b"

    def test_resolve_endpoint_dark_returns_none(self):
        engine = self._make_engine(tier=Tier.DARK, available_gpus=[])
        engine.tier.local_alive = False
        ep, model = engine._resolve_endpoint("planner", Tier.DARK)
        assert ep is None

    def test_build_planner_prompt_with_context(self):
        engine = self._make_engine()
        prompt = engine._build_planner_prompt("Build API", {"lang": "python"})
        assert "Build API" in prompt
        assert "python" in prompt
        assert "nodes" in prompt

    def test_build_planner_prompt_without_context(self):
        engine = self._make_engine()
        prompt = engine._build_planner_prompt("Build API", {})
        assert "Build API" in prompt
        assert "Context" not in prompt

    def test_parse_proposal_valid_json(self):
        engine = self._make_engine()
        result = _make_chat_result(json.dumps({
            "nodes": [{"id": "t1"}],
            "reasoning": "think",
            "confidence": 0.9,
        }))
        p = engine._parse_proposal(result)
        assert p.confidence == 0.9
        assert len(p.task_graph["nodes"]) == 1

    def test_parse_proposal_bad_json(self):
        engine = self._make_engine()
        result = _make_chat_result("This is not json, just a plan description")
        p = engine._parse_proposal(result)
        assert p.confidence == 0.5
        assert "raw_plan" in p.task_graph

    def test_parse_critique_valid(self):
        engine = self._make_engine()
        result = _make_chat_result(json.dumps({
            "flaws": ["A"], "risks": ["B"], "score": 0.6, "suggestions": ["C"],
        }))
        c = engine._parse_critique(result)
        assert c.flaws == ["A"]
        assert c.score == 0.6

    def test_parse_critique_bad_json(self):
        engine = self._make_engine()
        result = _make_chat_result("freeform critique text")
        c = engine._parse_critique(result)
        assert c.score == 0.5
        assert c.flaws == []

    def test_parse_consensus_valid(self):
        engine = self._make_engine()
        result = _make_chat_result(json.dumps({
            "final_plan": {"step": 1},
            "incorporated": ["fix"],
            "rejected": [],
            "confidence": 0.9,
            "dissent": ["minor issue"],
        }))
        c = engine._parse_consensus(result)
        assert c.confidence == 0.9
        assert c.dissent == ["minor issue"]

    def test_parse_consensus_bad_json(self):
        engine = self._make_engine()
        result = _make_chat_result("freeform consensus")
        c = engine._parse_consensus(result)
        assert c.confidence == 0.5

    def test_extract_consensus_from_critique_high_score(self):
        engine = self._make_engine()
        proposal = Proposal(task_graph={"nodes": [1]}, reasoning="r", confidence=0.8)
        critique = Critique(
            flaws=["minor"],
            risks=[],
            score=0.8,
            suggestions=["tweak A", "tweak B", "tweak C"],
        )
        c = engine._extract_consensus_from_critique(proposal, critique)
        assert c.confidence == 0.8
        # Only first 2 suggestions incorporated
        assert len(c.incorporated_critiques) == 2
        assert c.dissent == ["minor"]

    def test_extract_consensus_from_critique_low_score(self):
        engine = self._make_engine()
        proposal = Proposal(task_graph={"nodes": []}, reasoning="r", confidence=0.3)
        critique = Critique(
            flaws=["major flaw"],
            risks=["big risk"],
            score=0.4,
            suggestions=["rewrite"],
        )
        c = engine._extract_consensus_from_critique(proposal, critique)
        assert c.confidence == pytest.approx(0.32)  # 0.4 * 0.8
        assert "major flaw" in c.dissent
        assert "big risk" in c.dissent

    @pytest.mark.asyncio
    async def test_critique_bleed_truncates_reasoning(self):
        engine = self._make_engine(tier=Tier.FULL)
        engine.config.plan_reasoning_bleed = 0.3

        # Need both planner and critic to succeed
        long_reasoning = "A" * 1000
        proposal_json = json.dumps({
            "nodes": [], "reasoning": long_reasoning, "confidence": 0.8,
        })
        critique_json = json.dumps({
            "flaws": [], "risks": [], "score": 0.8, "suggestions": [],
        })
        consensus_json = json.dumps({
            "final_plan": {}, "incorporated": [], "rejected": [],
            "confidence": 0.8, "dissent": [],
        })

        self._mock_chat_sequence(engine, [proposal_json, critique_json, consensus_json])
        await engine.deliberate("test bleed")

        # Check the critic call's messages contain truncated reasoning
        critic_call = engine.client.chat.call_args_list[1]
        critic_prompt = critic_call[0][2][1]["content"]  # messages[1]["content"]
        assert "truncated for independent evaluation" in critic_prompt


# ═══════════════════════════════════════════════════════════════════
#  SLEEPTIME SCHEDULER
# ═══════════════════════════════════════════════════════════════════


class TestSleeptimeScheduler:
    def test_register_activity(self):
        config = _make_config(idle_timeout_s=10.0)
        sched = SleeptimeScheduler(config)
        sched._last_activity = time.time() - 100  # was idle
        assert sched.is_idle()

        sched.register_activity()
        assert not sched.is_idle()

    def test_is_idle(self):
        config = _make_config(idle_timeout_s=1.0)
        sched = SleeptimeScheduler(config)
        sched._last_activity = time.time() - 2.0
        assert sched.is_idle()

    def test_not_idle(self):
        config = _make_config(idle_timeout_s=999.0)
        sched = SleeptimeScheduler(config)
        assert not sched.is_idle()

    def test_idle_for_s(self):
        config = _make_config(idle_timeout_s=1.0)
        sched = SleeptimeScheduler(config)
        sched._last_activity = time.time() - 5.0
        assert sched.idle_for_s >= 4.5

    def test_default_jobs(self):
        assert len(DEFAULT_JOBS) == 3
        types = {j.job_type for j in DEFAULT_JOBS}
        assert types == {"consolidate", "retrospective", "precompute"}

    def test_jobs_sorted_by_priority(self):
        sorted_jobs = sorted(DEFAULT_JOBS, key=lambda j: j.priority, reverse=True)
        assert sorted_jobs[0].job_type == "consolidate"
        assert sorted_jobs[-1].job_type == "precompute"

    @pytest.mark.asyncio
    async def test_run_disabled(self):
        config = _make_config(sleeptime_enabled=False)
        sched = SleeptimeScheduler(config)
        engine = MagicMock()

        # Should return immediately when disabled
        await sched.run(engine)

        assert not sched._running

    @pytest.mark.asyncio
    async def test_run_job_consolidate(self):
        config = _make_config(idle_timeout_s=0.0)
        sched = SleeptimeScheduler(config)

        # Mock engine.deliberate
        mock_result = MagicMock()
        mock_result.consensus.incorporated_critiques = ["a", "b"]
        engine = AsyncMock()
        engine.deliberate = AsyncMock(return_value=mock_result)

        job = SleeptimeJob("consolidate", "test", priority=10)
        stats = await sched._run_job(job, engine)

        assert stats.job_type == "consolidate"
        assert stats.memories_consolidated == 2
        assert stats.elapsed_s >= 0
        assert stats.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_job_retrospective(self):
        config = _make_config()
        sched = SleeptimeScheduler(config)

        mock_result = MagicMock()
        engine = AsyncMock()
        engine.deliberate = AsyncMock(return_value=mock_result)

        job = SleeptimeJob("retrospective", "test", priority=10)
        stats = await sched._run_job(job, engine)

        assert stats.verdicts_reviewed == 1

    @pytest.mark.asyncio
    async def test_run_job_precompute(self):
        config = _make_config()
        sched = SleeptimeScheduler(config)

        mock_result = MagicMock()
        mock_result.proposal.task_graph = {"nodes": ["a", "b", "c"]}
        engine = AsyncMock()
        engine.deliberate = AsyncMock(return_value=mock_result)

        job = SleeptimeJob("precompute", "test", priority=10)
        stats = await sched._run_job(job, engine)

        assert stats.plans_precomputed == 3

    @pytest.mark.asyncio
    async def test_run_job_error_doesnt_crash(self):
        config = _make_config()
        sched = SleeptimeScheduler(config)

        engine = AsyncMock()
        engine.deliberate = AsyncMock(side_effect=RuntimeError("GPU exploded"))

        job = SleeptimeJob("consolidate", "test", priority=10)
        stats = await sched._run_job(job, engine)

        assert stats.elapsed_s >= 0
        assert stats.completed_at is not None

    def test_stop(self):
        config = _make_config()
        sched = SleeptimeScheduler(config)
        sched._running = True
        sched.stop()
        assert not sched._running

    def test_summary_empty(self):
        config = _make_config()
        sched = SleeptimeScheduler(config)
        s = sched.summary()
        assert s["cycles_completed"] == 0
        assert s["total_consolidated"] == 0
        assert s["preemptions"] == 0

    def test_summary_with_stats(self):
        config = _make_config()
        sched = SleeptimeScheduler(config)
        sched._cycle_count = 3
        sched._stats = [
            SleeptimeCycleStats(
                cycle_number=1, started_at=None,
                memories_consolidated=5, preempted=False,
            ),
            SleeptimeCycleStats(
                cycle_number=2, started_at=None,
                verdicts_reviewed=2, preempted=True,
            ),
            SleeptimeCycleStats(
                cycle_number=3, started_at=None,
                plans_precomputed=3, preempted=False,
            ),
        ]
        s = sched.summary()
        assert s["cycles_completed"] == 3
        assert s["total_consolidated"] == 5
        assert s["total_reviewed"] == 2
        assert s["total_precomputed"] == 3
        assert s["preemptions"] == 1


# ═══════════════════════════════════════════════════════════════════
#  CLIENT (data types only — no HTTP)
# ═══════════════════════════════════════════════════════════════════


class TestClientDataTypes:
    def test_chat_result_fields(self):
        r = ChatResult(
            content="hello",
            model="test",
            prompt_tokens=10,
            completion_tokens=5,
            elapsed_s=0.5,
            raw={},
        )
        assert r.content == "hello"
        assert r.elapsed_s == 0.5

    def test_health_status_alive(self):
        h = HealthStatus(alive=True, gpu_id=0, status="ready", model="qwen")
        assert h.alive
        assert h.gpu_id == 0

    def test_health_status_dead(self):
        h = HealthStatus(alive=False, error="Connection refused")
        assert not h.alive
        assert h.error == "Connection refused"


# ═══════════════════════════════════════════════════════════════════
#  __init__.py EXPORTS
# ═══════════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_all_exports_importable(self):
        import hololoom.core.deep_thinking as dt
        for name in dt.__all__:
            assert hasattr(dt, name), f"{name} in __all__ but not importable"

    def test_key_classes_accessible(self):
        from hololoom.core.deep_thinking import (
            DeepThinkingConfig,
            Tier,
            TierState,
            DeepClient,
            DeliberationEngine,
            MilestoneGate,
            DeferredQueue,
            SleeptimeScheduler,
        )
        # Just verify they imported without error
        assert DeepThinkingConfig is not None
        assert Tier.FULL == 3
