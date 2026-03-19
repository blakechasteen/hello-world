"""
Unit tests for the Deep Thinking module (hololoom/core/deep_thinking/).

Covers: config, tier state, gate, client, deliberation engine, deferred queue,
and sleeptime scheduler. All GPU/network calls are mocked.

Design Principle #2 (Graceful Degradation) is tested explicitly:
    FULL → PARTIAL → DEGRADED → DARK at every layer.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
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
from hololoom.core.deep_thinking.deliberation import (
    DeliberationEngine,
    DeliberationResult,
    Proposal,
    Critique,
    Consensus,
    _extract_json as delib_extract_json,
)
from hololoom.core.deep_thinking.gate import (
    MilestoneGate,
    MilestoneContext,
    GateVerdict,
    VerdictPrior,
    _extract_json as gate_extract_json,
)
from hololoom.core.deep_thinking.deferred import (
    DeferredQueue,
    DeferredJob,
    JOB_TYPE_PRIORITY,
)
from hololoom.core.deep_thinking.sleeptime import (
    SleeptimeScheduler,
    SleeptimeCycleStats,
    SleeptimeJob,
    DEFAULT_JOBS,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    """Default deep thinking config with localhost endpoints."""
    return DeepThinkingConfig(
        gpu_servers=[
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8002",
            "http://127.0.0.1:8003",
        ],
    )


@pytest.fixture
def mock_client():
    """DeepClient with mocked HTTP calls."""
    client = DeepClient()
    client.chat = AsyncMock()
    client.health = AsyncMock()
    client.is_alive = AsyncMock()
    return client


@pytest.fixture
def tier_state(config, mock_client):
    """TierState wired to a mock client."""
    return TierState(config, mock_client)


@pytest.fixture
def milestone_ctx():
    """Standard milestone context for gate tests."""
    return MilestoneContext(
        goal="Build a REST API",
        work_so_far={"step_1": "defined routes"},
        milestone_description="Route definitions complete",
        milestone_number=1,
        total_milestones=5,
        agent_id="test-agent",
    )


@pytest.fixture
def deferred_queue(tmp_path):
    """DeferredQueue using tmp_path (SOP-11, AP-3)."""
    return DeferredQueue(
        persist_path=str(tmp_path / "deferred.json"),
        staleness_boost_per_hour=0.1,
    )


# ═══════════════════════════════════════════════════════════════════
# config.py
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
        assert cfg.inference_timeout_s == 900.0
        assert cfg.max_context_tokens == 131072

    def test_gpu_servers_from_env(self, monkeypatch):
        monkeypatch.setenv("MINING_RIG_IP", "10.0.0.99")
        cfg = DeepThinkingConfig()
        for server in cfg.gpu_servers:
            assert "10.0.0.99" in server

    def test_gpu_servers_default_localhost(self, monkeypatch):
        monkeypatch.delenv("MINING_RIG_IP", raising=False)
        cfg = DeepThinkingConfig()
        for server in cfg.gpu_servers:
            assert "127.0.0.1" in server

    def test_bleed_rates_in_valid_range(self):
        cfg = DeepThinkingConfig()
        assert 0.0 <= cfg.plan_to_critic_bleed <= 1.0
        assert 0.0 <= cfg.plan_reasoning_bleed <= 1.0
        assert 0.0 <= cfg.consensus_bleed <= 1.0

    def test_gate_thresholds_ordered(self):
        cfg = DeepThinkingConfig()
        assert cfg.gate_threshold_full > cfg.gate_threshold_degraded
        assert cfg.gate_threshold_degraded > cfg.defer_below

    def test_role_gpu_map_covers_three_roles(self):
        assert "planner" in ROLE_GPU_MAP
        assert "critic" in ROLE_GPU_MAP
        assert "synthesizer" in ROLE_GPU_MAP
        assert set(ROLE_GPU_MAP.values()) == {0, 1, 2}

    def test_system_prompts_per_role(self):
        for role in ("planner", "critic", "synthesizer"):
            assert role in SYSTEM_PROMPTS
            assert len(SYSTEM_PROMPTS[role]) > 20

    def test_gate_prompt_exists(self):
        assert "proceed" in GATE_PROMPT.lower()
        assert "revise" in GATE_PROMPT.lower()
        assert "abort" in GATE_PROMPT.lower()

    def test_deferred_path_default(self):
        cfg = DeepThinkingConfig()
        assert cfg.deferred_path == "./deep_thinking_deferred.json"


# ═══════════════════════════════════════════════════════════════════
# tier.py
# ═══════════════════════════════════════════════════════════════════

class TestTier:

    def test_tier_ordering(self):
        assert Tier.DARK < Tier.DEGRADED < Tier.PARTIAL < Tier.FULL

    def test_tier_int_values(self):
        assert int(Tier.DARK) == 0
        assert int(Tier.DEGRADED) == 1
        assert int(Tier.PARTIAL) == 2
        assert int(Tier.FULL) == 3


class TestTierState:

    async def test_all_gpus_alive_means_full(self, config, mock_client):
        mock_client.is_alive.return_value = True
        ts = TierState(config, mock_client)
        tier = await ts.probe()
        assert tier == Tier.FULL
        assert len(ts.available_gpus) == 3

    async def test_two_gpus_alive_means_partial(self, config, mock_client):
        mock_client.is_alive.side_effect = [True, True, False, True]  # gpu0, gpu1, gpu2, local
        ts = TierState(config, mock_client)
        tier = await ts.probe()
        assert tier == Tier.PARTIAL
        assert ts.available_gpus == [0, 1]

    async def test_one_gpu_alive_means_partial(self, config, mock_client):
        mock_client.is_alive.side_effect = [False, True, False, True]
        ts = TierState(config, mock_client)
        tier = await ts.probe()
        assert tier == Tier.PARTIAL
        assert ts.available_gpus == [1]

    async def test_no_gpus_local_alive_means_degraded(self, config, mock_client):
        mock_client.is_alive.side_effect = [False, False, False, True]
        ts = TierState(config, mock_client)
        tier = await ts.probe()
        assert tier == Tier.DEGRADED
        assert ts.available_gpus == []
        assert ts.local_alive is True

    async def test_nothing_alive_means_dark(self, config, mock_client):
        mock_client.is_alive.return_value = False
        ts = TierState(config, mock_client)
        tier = await ts.probe()
        assert tier == Tier.DARK
        assert ts.available_gpus == []
        assert ts.local_alive is False

    async def test_transition_callback_fires(self, config, mock_client):
        mock_client.is_alive.return_value = False
        ts = TierState(config, mock_client)
        transitions = []
        ts.on_transition(lambda t: transitions.append(t))

        # DARK (initial) stays DARK — no transition
        await ts.probe()
        assert len(transitions) == 0

        # Now all GPUs come online → DARK → FULL
        mock_client.is_alive.return_value = True
        await ts.probe()
        assert len(transitions) == 1
        assert transitions[0].previous == Tier.DARK
        assert transitions[0].current == Tier.FULL

    async def test_no_transition_when_tier_unchanged(self, config, mock_client):
        mock_client.is_alive.return_value = True
        ts = TierState(config, mock_client)
        transitions = []
        ts.on_transition(lambda t: transitions.append(t))

        # First probe: DARK → FULL (transition)
        await ts.probe()
        # Second probe: FULL → FULL (no transition)
        await ts.probe()
        assert len(transitions) == 1

    async def test_callback_error_does_not_crash(self, config, mock_client):
        mock_client.is_alive.return_value = True
        ts = TierState(config, mock_client)

        def bad_callback(t):
            raise RuntimeError("callback boom")

        ts.on_transition(bad_callback)
        # Should not raise
        await ts.probe()
        assert ts.tier == Tier.FULL

    def test_endpoint_for_gpu_available(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.available_gpus = [0, 2]
        assert ts.endpoint_for_gpu(0) == "http://127.0.0.1:8001"
        assert ts.endpoint_for_gpu(2) == "http://127.0.0.1:8003"

    def test_endpoint_for_gpu_unavailable(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.available_gpus = [0]
        assert ts.endpoint_for_gpu(1) is None

    def test_best_available_endpoint_prefers_gpu(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.available_gpus = [1, 2]
        ts.local_alive = True
        assert ts.best_available_endpoint() == "http://127.0.0.1:8002"

    def test_best_available_endpoint_falls_to_local(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.available_gpus = []
        ts.local_alive = True
        assert ts.best_available_endpoint() == config.local_endpoint

    def test_best_available_endpoint_none_when_dark(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.available_gpus = []
        ts.local_alive = False
        assert ts.best_available_endpoint() is None

    def test_gate_threshold_full_tier(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.tier = Tier.FULL
        assert ts.gate_threshold == config.gate_threshold_full

    def test_gate_threshold_partial_tier(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.tier = Tier.PARTIAL
        assert ts.gate_threshold == config.gate_threshold_full

    def test_gate_threshold_degraded_tier(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.tier = Tier.DEGRADED
        assert ts.gate_threshold == config.gate_threshold_degraded

    def test_model_for_full_tier(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.tier = Tier.FULL
        assert ts.model_for_tier == config.rig_model

    def test_model_for_degraded_tier(self, config, mock_client):
        ts = TierState(config, mock_client)
        ts.tier = Tier.DEGRADED
        assert ts.model_for_tier == config.local_model

    async def test_probe_handles_exceptions_in_gpu_results(self, config, mock_client):
        """Exceptions from individual probes should be treated as not-alive."""
        mock_client.is_alive.side_effect = [
            ConnectionError("boom"), True, True, True
        ]
        ts = TierState(config, mock_client)
        tier = await ts.probe()
        assert tier == Tier.PARTIAL
        assert 0 not in ts.available_gpus


# ═══════════════════════════════════════════════════════════════════
# client.py
# ═══════════════════════════════════════════════════════════════════

class TestDeepClient:

    async def test_chat_success(self):
        """Mock a successful chat completion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            client = DeepClient(timeout_s=10.0, retries=0)
            result = await client.chat(
                "http://localhost:8001", "test-model",
                [{"role": "user", "content": "hi"}],
            )

            assert result.content == "hello world"
            assert result.model == "test-model"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 5
            assert result.elapsed_s >= 0

    async def test_chat_connect_error_retries(self):
        """Connection errors should retry up to retries count."""
        import httpx

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("refused")

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            client = DeepClient(timeout_s=5.0, retries=2)
            with pytest.raises(ConnectionError, match="Cannot reach"):
                await client.chat(
                    "http://localhost:8001", "model",
                    [{"role": "user", "content": "hi"}],
                )

            assert call_count == 3  # initial + 2 retries

    async def test_chat_timeout_no_retry(self):
        """Timeouts should not retry (AirLLM is slow, retrying won't help)."""
        import httpx

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("timed out")

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            client = DeepClient(timeout_s=5.0, retries=2)
            with pytest.raises(TimeoutError, match="timed out"):
                await client.chat(
                    "http://localhost:8001", "model",
                    [{"role": "user", "content": "hi"}],
                )

            assert call_count == 1  # no retry on timeout

    async def test_chat_429_raises_runtime_error(self):
        """GPU busy (429) should raise RuntimeError."""
        import httpx

        resp = MagicMock()
        resp.status_code = 429

        async def mock_post(*args, **kwargs):
            raise httpx.HTTPStatusError("busy", request=MagicMock(), response=resp)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            client = DeepClient(timeout_s=5.0, retries=0)
            with pytest.raises(RuntimeError, match="GPU busy"):
                await client.chat(
                    "http://localhost:8001", "model",
                    [{"role": "user", "content": "hi"}],
                )

    async def test_health_alive(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "gpu_id": 0, "status": "ready", "model": "qwen3.5:108b",
            "uptime_s": 3600.0,
        }

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            client = DeepClient()
            status = await client.health("http://localhost:8001")
            assert status.alive is True
            assert status.gpu_id == 0
            assert status.status == "ready"

    async def test_health_dead(self):
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.side_effect = Exception("refused")
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            client = DeepClient()
            status = await client.health("http://localhost:8001")
            assert status.alive is False
            assert status.error is not None

    async def test_is_alive_checks_status(self):
        """is_alive requires alive=True AND status in (ready, busy)."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"status": "loading"}
            mock_http.get.return_value = mock_resp
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_http

            client = DeepClient()
            alive = await client.is_alive("http://localhost:8001")
            # status is "loading", not "ready" or "busy"
            assert alive is False


# ═══════════════════════════════════════════════════════════════════
# deferred.py
# ═══════════════════════════════════════════════════════════════════

class TestDeferredJob:

    def test_default_fields(self):
        job = DeferredJob()
        assert len(job.id) == 12
        assert job.job_type == "deliberation"
        assert job.priority == 0.0

    def test_roundtrip_dict(self):
        job = DeferredJob(job_type="gate_review", payload={"goal": "test"})
        d = job.to_dict()
        restored = DeferredJob.from_dict(d)
        assert restored.id == job.id
        assert restored.job_type == "gate_review"
        assert restored.payload == {"goal": "test"}

    def test_from_dict_ignores_extra_keys(self):
        d = {"id": "abc", "job_type": "critique", "payload": {}, "extra": "ignored",
             "created_at": 100.0, "priority": 0.0, "staleness_boost": 0.0}
        job = DeferredJob.from_dict(d)
        assert job.id == "abc"


class TestDeferredQueue:

    async def test_push_and_size(self, deferred_queue):
        assert deferred_queue.empty
        await deferred_queue.push(DeferredJob(job_type="gate_review"))
        assert deferred_queue.size == 1
        assert not deferred_queue.empty

    async def test_push_sets_type_priority(self, deferred_queue):
        job = DeferredJob(job_type="gate_review")
        await deferred_queue.push(job)
        assert job.priority == JOB_TYPE_PRIORITY["gate_review"]

    async def test_push_respects_existing_priority(self, deferred_queue):
        job = DeferredJob(job_type="gate_review", priority=999.0)
        await deferred_queue.push(job)
        assert job.priority == 999.0

    def test_pop_highest_by_effective_priority(self, deferred_queue):
        # Manually add jobs with different types
        deferred_queue._jobs = [
            DeferredJob(job_type="synthesis", priority=10, created_at=time.time()),
            DeferredJob(job_type="gate_review", priority=100, created_at=time.time()),
        ]
        popped = deferred_queue.pop_highest()
        # gate_review has type_weight 100 + priority 100 = 200 effective
        assert popped.job_type == "gate_review"

    def test_pop_highest_empty(self, deferred_queue):
        assert deferred_queue.pop_highest() is None

    def test_staleness_boost(self, deferred_queue):
        old_job = DeferredJob(
            job_type="synthesis",
            priority=10,
            created_at=time.time() - 7200,  # 2 hours ago
        )
        deferred_queue._jobs = [old_job]
        deferred_queue._update_staleness()
        # 2 hours * 0.1 per hour = ~0.2
        assert old_job.staleness_boost >= 0.19

    async def test_drain_processes_in_priority_order(self, deferred_queue):
        processed_order = []

        async def process_fn(job):
            processed_order.append(job.job_type)
            return True

        deferred_queue._jobs = [
            DeferredJob(job_type="synthesis", priority=10, created_at=time.time()),
            DeferredJob(job_type="gate_review", priority=100, created_at=time.time()),
            DeferredJob(job_type="deliberation", priority=50, created_at=time.time()),
        ]
        count = await deferred_queue.drain(process_fn)
        assert count == 3
        assert processed_order[0] == "gate_review"
        assert deferred_queue.empty

    async def test_drain_requeues_failures(self, deferred_queue):
        async def process_fn(job):
            return job.job_type != "synthesis"  # synthesis fails

        deferred_queue._jobs = [
            DeferredJob(job_type="gate_review", priority=100, created_at=time.time()),
            DeferredJob(job_type="synthesis", priority=10, created_at=time.time()),
        ]
        count = await deferred_queue.drain(process_fn)
        assert count == 1
        assert deferred_queue.size == 1
        assert deferred_queue._jobs[0].job_type == "synthesis"

    async def test_drain_requeues_on_exception(self, deferred_queue):
        async def process_fn(job):
            raise RuntimeError("boom")

        deferred_queue._jobs = [
            DeferredJob(job_type="gate_review", priority=100, created_at=time.time()),
        ]
        count = await deferred_queue.drain(process_fn)
        assert count == 0
        assert deferred_queue.size == 1

    async def test_drain_empty_queue(self, deferred_queue):
        count = await deferred_queue.drain(AsyncMock(return_value=True))
        assert count == 0

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "q.json")
        q = DeferredQueue(persist_path=path)
        q._jobs = [
            DeferredJob(id="aaa", job_type="gate_review", payload={"x": 1},
                        created_at=1000.0, priority=100.0, staleness_boost=0.0),
        ]
        q.save()

        q2 = DeferredQueue(persist_path=path)
        q2.load()
        assert q2.size == 1
        assert q2._jobs[0].id == "aaa"
        assert q2._jobs[0].payload == {"x": 1}

    def test_load_missing_file(self, tmp_path):
        q = DeferredQueue(persist_path=str(tmp_path / "no_exist.json"))
        q.load()
        assert q.empty

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        q = DeferredQueue(persist_path=str(path))
        q.load()
        assert q.empty

    def test_summary(self, deferred_queue):
        deferred_queue._jobs = [
            DeferredJob(job_type="gate_review", created_at=time.time() - 3600),
            DeferredJob(job_type="gate_review", created_at=time.time()),
            DeferredJob(job_type="synthesis", created_at=time.time()),
        ]
        s = deferred_queue.summary()
        assert s["total"] == 3
        assert s["by_type"]["gate_review"] == 2
        assert s["by_type"]["synthesis"] == 1
        assert s["oldest_hours"] >= 0.9


# ═══════════════════════════════════════════════════════════════════
# gate.py — VerdictPrior (Thompson Sampling)
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

    def test_update_shifts_prior(self):
        vp = VerdictPrior()
        vp.update("proceed", was_correct=True)
        # alpha went from 1 to 2, beta stayed at 1
        assert vp.proceed[0] == 2.0
        assert vp.proceed[1] == 1.0
        assert vp.expected_accuracy("proceed") == pytest.approx(2.0 / 3.0)

    def test_update_failure_shifts_beta(self):
        vp = VerdictPrior()
        vp.update("revise", was_correct=False)
        assert vp.revise[0] == 1.0
        assert vp.revise[1] == 2.0

    def test_to_dict_from_dict_roundtrip(self):
        vp = VerdictPrior()
        vp.update("proceed", was_correct=True)
        vp.update("abort", was_correct=False)

        d = vp.to_dict()
        vp2 = VerdictPrior.from_dict(d)
        assert vp2.proceed == vp.proceed
        assert vp2.abort == vp.abort

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "priors.json"
        vp = VerdictPrior()
        vp.update("proceed", was_correct=True)
        vp.save(path)

        vp2 = VerdictPrior()
        vp2.load(path)
        assert vp2.proceed == vp.proceed

    def test_load_missing_file_stays_uniform(self, tmp_path):
        vp = VerdictPrior()
        vp.load(tmp_path / "no_exist.json")
        assert vp.proceed == (1.0, 1.0)

    def test_load_corrupt_file_stays_uniform(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        vp = VerdictPrior()
        vp.load(path)
        assert vp.proceed == (1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# gate.py — JSON extraction
# ═══════════════════════════════════════════════════════════════════

class TestGateExtractJson:

    def test_raw_json(self):
        data = gate_extract_json('{"verdict": "proceed", "confidence": 0.9}')
        assert data["verdict"] == "proceed"

    def test_json_in_code_block(self):
        text = 'Some preamble\n```json\n{"verdict": "revise"}\n```\nExtra'
        data = gate_extract_json(text)
        assert data["verdict"] == "revise"

    def test_json_in_plain_code_block(self):
        text = '```\n{"verdict": "abort"}\n```'
        data = gate_extract_json(text)
        assert data["verdict"] == "abort"

    def test_json_embedded_in_text(self):
        text = 'Here is my analysis: {"verdict": "proceed", "confidence": 0.8} end'
        data = gate_extract_json(text)
        assert data["verdict"] == "proceed"

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            gate_extract_json("no json here at all")


# ═══════════════════════════════════════════════════════════════════
# gate.py — MilestoneGate
# ═══════════════════════════════════════════════════════════════════

class TestMilestoneGate:

    def _make_gate(self, config, mock_client, tier_state, deferred_queue=None):
        return MilestoneGate(config, mock_client, tier_state, deferred_queue)

    async def test_dark_tier_defers_everything(
        self, config, mock_client, tier_state, milestone_ctx, deferred_queue
    ):
        tier_state.tier = Tier.DARK
        gate = self._make_gate(config, mock_client, tier_state, deferred_queue)

        verdict = await gate.review(milestone_ctx)
        assert verdict.deferred is True
        assert verdict.confidence == 0.0
        assert verdict.verdict == "proceed"
        assert deferred_queue.size == 1

    async def test_review_parses_model_response(
        self, config, mock_client, tier_state, milestone_ctx
    ):
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        mock_client.chat.return_value = ChatResult(
            content='{"verdict": "revise", "confidence": 0.85, '
                    '"reasoning": "Missing error handling", '
                    '"corrections": "Add try/except around DB calls"}',
            model="qwen3.5:108b",
            prompt_tokens=100,
            completion_tokens=50,
            elapsed_s=2.0,
            raw={},
        )

        gate = self._make_gate(config, mock_client, tier_state)
        verdict = await gate.review(milestone_ctx)
        assert verdict.verdict == "revise"
        assert verdict.confidence == 0.85
        assert verdict.corrections == "Add try/except around DB calls"
        assert verdict.deferred is False

    async def test_review_parse_failure_defaults_cautious(
        self, config, mock_client, tier_state, milestone_ctx
    ):
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        mock_client.chat.return_value = ChatResult(
            content="I'm not sure what to say, this is unparseable",
            model="qwen3.5:108b",
            prompt_tokens=100,
            completion_tokens=50,
            elapsed_s=2.0,
            raw={},
        )

        gate = self._make_gate(config, mock_client, tier_state)
        verdict = await gate.review(milestone_ctx)
        assert verdict.verdict == "proceed"
        assert verdict.confidence == 0.3
        assert "[parse error]" in verdict.reasoning

    async def test_review_invalid_verdict_normalized(
        self, config, mock_client, tier_state, milestone_ctx
    ):
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        mock_client.chat.return_value = ChatResult(
            content='{"verdict": "maybe", "confidence": 0.6, "reasoning": "hm"}',
            model="qwen3.5:108b",
            prompt_tokens=100,
            completion_tokens=50,
            elapsed_s=1.0,
            raw={},
        )

        gate = self._make_gate(config, mock_client, tier_state)
        verdict = await gate.review(milestone_ctx)
        assert verdict.verdict == "proceed"  # invalid → default proceed

    async def test_review_connection_error_defers(
        self, config, mock_client, tier_state, milestone_ctx, deferred_queue
    ):
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        mock_client.chat.side_effect = ConnectionError("rig down")

        gate = self._make_gate(config, mock_client, tier_state, deferred_queue)
        verdict = await gate.review(milestone_ctx)
        assert verdict.deferred is True
        assert deferred_queue.size == 1

    async def test_low_confidence_degraded_defers(
        self, config, mock_client, tier_state, milestone_ctx, deferred_queue
    ):
        """Confidence below defer_below (0.5) in DEGRADED tier → deferred."""
        tier_state.tier = Tier.DEGRADED
        tier_state.available_gpus = []
        tier_state.local_alive = True

        mock_client.chat.return_value = ChatResult(
            content='{"verdict": "proceed", "confidence": 0.3, "reasoning": "unsure"}',
            model="qwen3.5:27b",
            prompt_tokens=50,
            completion_tokens=20,
            elapsed_s=1.0,
            raw={},
        )

        gate = self._make_gate(config, mock_client, tier_state, deferred_queue)
        verdict = await gate.review(milestone_ctx)
        assert verdict.deferred is True
        assert deferred_queue.size == 1

    async def test_no_deferral_without_queue(
        self, config, mock_client, tier_state, milestone_ctx
    ):
        """If no deferred_queue is wired, deferral is a no-op."""
        tier_state.tier = Tier.DARK
        gate = self._make_gate(config, mock_client, tier_state, None)
        verdict = await gate.review(milestone_ctx)
        assert verdict.deferred is True
        # No crash, but nothing queued (queue is None)

    def test_update_priors(self, config, mock_client, tier_state):
        gate = self._make_gate(config, mock_client, tier_state)
        gate.update_priors("proceed", was_correct=True)
        assert gate.priors.expected_accuracy("proceed") > 0.5

    def test_build_review_prompt(self, config, mock_client, tier_state, milestone_ctx):
        gate = self._make_gate(config, mock_client, tier_state)
        prompt = gate._build_review_prompt(milestone_ctx)
        assert "Build a REST API" in prompt
        assert "1/5" in prompt


# ═══════════════════════════════════════════════════════════════════
# deliberation.py — JSON extraction
# ═══════════════════════════════════════════════════════════════════

class TestDelibExtractJson:

    def test_raw_json(self):
        data = delib_extract_json('{"nodes": []}')
        assert data["nodes"] == []

    def test_json_in_markdown_block(self):
        text = '```json\n{"nodes": [{"id": 1}]}\n```'
        data = delib_extract_json(text)
        assert len(data["nodes"]) == 1

    def test_embedded_json(self):
        text = 'I propose: {"nodes": []} and that is all.'
        data = delib_extract_json(text)
        assert "nodes" in data


# ═══════════════════════════════════════════════════════════════════
# deliberation.py — DeliberationEngine
# ═══════════════════════════════════════════════════════════════════

class TestDeliberationEngine:

    def _make_engine(self, config, mock_client, tier_state):
        return DeliberationEngine(config, mock_client, tier_state)

    def _chat_result(self, content: str) -> ChatResult:
        return ChatResult(
            content=content,
            model="test",
            prompt_tokens=10,
            completion_tokens=5,
            elapsed_s=0.1,
            raw={},
        )

    async def test_full_tier_three_phase_pipeline(
        self, config, mock_client, tier_state
    ):
        """FULL tier: Planner → Critic → Synthesizer (3 calls)."""
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        mock_client.chat.side_effect = [
            # Planner
            self._chat_result(json.dumps({
                "nodes": [{"id": 1, "task": "design API"}],
                "reasoning": "Start with the API spec",
                "confidence": 0.8,
            })),
            # Critic
            self._chat_result(json.dumps({
                "flaws": ["No auth mentioned"],
                "risks": ["Scalability"],
                "score": 0.6,
                "suggestions": ["Add JWT auth"],
            })),
            # Synthesizer
            self._chat_result(json.dumps({
                "final_plan": {"nodes": [{"id": 1, "task": "design API with JWT"}]},
                "incorporated": ["Add JWT auth"],
                "rejected": [],
                "confidence": 0.85,
                "dissent": [],
            })),
        ]

        engine = self._make_engine(config, mock_client, tier_state)
        result = await engine.deliberate("Build a REST API")

        assert mock_client.chat.call_count == 3
        assert result.proposal.confidence == 0.8
        assert result.critique.score == 0.6
        assert result.consensus.confidence == 0.85
        assert result.stats.tier_used == Tier.FULL

    async def test_partial_tier_collapses_synthesizer(
        self, config, mock_client, tier_state
    ):
        """PARTIAL tier: Planner → Critic (2 calls), consensus from critique."""
        tier_state.tier = Tier.PARTIAL
        tier_state.available_gpus = [0, 1]
        tier_state.local_alive = True

        mock_client.chat.side_effect = [
            # Planner
            self._chat_result(json.dumps({
                "nodes": [{"id": 1, "task": "design"}],
                "reasoning": "Basic plan",
                "confidence": 0.7,
            })),
            # Critic
            self._chat_result(json.dumps({
                "flaws": ["Missing tests"],
                "risks": [],
                "score": 0.75,
                "suggestions": ["Add tests"],
            })),
        ]

        engine = self._make_engine(config, mock_client, tier_state)
        result = await engine.deliberate("Build something")

        assert mock_client.chat.call_count == 2
        # Consensus derived from critique (score >= 0.7)
        assert result.consensus.confidence == 0.75
        assert "Add tests" in result.consensus.incorporated_critiques

    async def test_degraded_tier_uses_local_model(
        self, config, mock_client, tier_state
    ):
        """DEGRADED tier: all roles on local 27B."""
        tier_state.tier = Tier.DEGRADED
        tier_state.available_gpus = []
        tier_state.local_alive = True

        mock_client.chat.side_effect = [
            self._chat_result(json.dumps({
                "nodes": [], "reasoning": "local plan", "confidence": 0.5,
            })),
            self._chat_result(json.dumps({
                "flaws": [], "risks": [], "score": 0.5, "suggestions": [],
            })),
            self._chat_result(json.dumps({
                "final_plan": {}, "incorporated": [], "rejected": [],
                "confidence": 0.5, "dissent": [],
            })),
        ]

        engine = self._make_engine(config, mock_client, tier_state)
        result = await engine.deliberate("Plan something")

        # All calls should use local endpoint
        for call in mock_client.chat.call_args_list:
            assert call[0][0] == config.local_endpoint
            assert call[0][1] == config.local_model

    async def test_dark_tier_returns_deferred_result(
        self, config, mock_client, tier_state
    ):
        """DARK tier: no inference available, synthetic deferred results."""
        tier_state.tier = Tier.DARK
        tier_state.available_gpus = []
        tier_state.local_alive = False

        engine = self._make_engine(config, mock_client, tier_state)
        result = await engine.deliberate("Plan something")

        # No LLM calls
        mock_client.chat.assert_not_called()
        assert result.proposal.confidence == 0.0
        assert "deferred" in result.proposal.task_graph

    async def test_bleed_control_truncates_reasoning(
        self, config, mock_client, tier_state
    ):
        """Critic sees only 30% of planner reasoning (anti-groupthink)."""
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        long_reasoning = "A" * 1000  # 1000 chars
        mock_client.chat.side_effect = [
            self._chat_result(json.dumps({
                "nodes": [], "reasoning": long_reasoning, "confidence": 0.8,
            })),
            self._chat_result(json.dumps({
                "flaws": [], "risks": [], "score": 0.8, "suggestions": [],
            })),
            self._chat_result(json.dumps({
                "final_plan": {}, "incorporated": [], "rejected": [],
                "confidence": 0.8, "dissent": [],
            })),
        ]

        engine = self._make_engine(config, mock_client, tier_state)
        await engine.deliberate("test bleed")

        # Check the critic's prompt — reasoning should be truncated
        critic_call = mock_client.chat.call_args_list[1]
        critic_messages = critic_call[0][2]  # messages arg
        user_content = critic_messages[1]["content"]
        # 30% of 1000 = 300 A's, plus truncation notice
        assert "truncated for independent evaluation" in user_content
        # Should not contain the full 1000 chars of reasoning
        assert user_content.count("A") < 400

    async def test_parse_proposal_fallback(self, config, mock_client, tier_state):
        """Unparseable planner output → graceful fallback."""
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        mock_client.chat.side_effect = [
            self._chat_result("This is just plain text, not JSON"),
            self._chat_result(json.dumps({
                "flaws": [], "risks": [], "score": 0.5, "suggestions": [],
            })),
            self._chat_result(json.dumps({
                "final_plan": {}, "incorporated": [], "rejected": [],
                "confidence": 0.5, "dissent": [],
            })),
        ]

        engine = self._make_engine(config, mock_client, tier_state)
        result = await engine.deliberate("test")

        # Should use plain text as reasoning, 0.5 confidence
        assert result.proposal.confidence == 0.5
        assert "raw_plan" in result.proposal.task_graph

    def test_extract_consensus_from_critique_high_score(
        self, config, mock_client, tier_state
    ):
        """When critique score >= 0.7, plan is mostly accepted."""
        engine = self._make_engine(config, mock_client, tier_state)
        proposal = Proposal(
            task_graph={"nodes": [1, 2]},
            reasoning="plan",
            confidence=0.8,
        )
        critique = Critique(
            flaws=["minor issue"],
            risks=[],
            score=0.75,
            suggestions=["fix minor"],
        )
        consensus = engine._extract_consensus_from_critique(proposal, critique)
        assert consensus.confidence == 0.75
        assert consensus.final_plan == proposal.task_graph
        assert "fix minor" in consensus.incorporated_critiques

    def test_extract_consensus_from_critique_low_score(
        self, config, mock_client, tier_state
    ):
        """When critique score < 0.7, confidence is reduced."""
        engine = self._make_engine(config, mock_client, tier_state)
        proposal = Proposal(
            task_graph={"nodes": []},
            reasoning="plan",
            confidence=0.5,
        )
        critique = Critique(
            flaws=["major flaw"],
            risks=["big risk"],
            score=0.3,
            suggestions=["redo everything"],
        )
        consensus = engine._extract_consensus_from_critique(proposal, critique)
        assert consensus.confidence == pytest.approx(0.3 * 0.8)
        assert "major flaw" in consensus.dissent
        assert "big risk" in consensus.dissent

    def test_resolve_endpoint_full_tier(self, config, mock_client, tier_state):
        tier_state.tier = Tier.FULL
        tier_state.available_gpus = [0, 1, 2]
        tier_state.local_alive = True

        engine = self._make_engine(config, mock_client, tier_state)
        ep, model = engine._resolve_endpoint("planner", Tier.FULL)
        assert ep == config.gpu_servers[ROLE_GPU_MAP["planner"]]
        assert model == config.rig_model

    def test_resolve_endpoint_fallback_gpu(self, config, mock_client, tier_state):
        """If planner's GPU (0) is down but others alive → use fallback GPU."""
        tier_state.tier = Tier.PARTIAL
        tier_state.available_gpus = [1, 2]
        tier_state.local_alive = True

        engine = self._make_engine(config, mock_client, tier_state)
        ep, model = engine._resolve_endpoint("planner", Tier.PARTIAL)
        # planner's gpu 0 not available, falls to best_available (gpu 1)
        assert ep == config.gpu_servers[1]
        assert model == config.rig_model

    def test_resolve_endpoint_dark_returns_none(self, config, mock_client, tier_state):
        tier_state.tier = Tier.DARK
        tier_state.available_gpus = []
        tier_state.local_alive = False

        engine = self._make_engine(config, mock_client, tier_state)
        ep, model = engine._resolve_endpoint("planner", Tier.DARK)
        assert ep is None


# ═══════════════════════════════════════════════════════════════════
# sleeptime.py
# ═══════════════════════════════════════════════════════════════════

class TestSleeptimeScheduler:

    def test_register_activity_resets_idle(self):
        cfg = DeepThinkingConfig(idle_timeout_s=10.0)
        sched = SleeptimeScheduler(cfg)
        sched._last_activity = time.time() - 20  # 20s ago
        assert sched.is_idle()
        sched.register_activity()
        assert not sched.is_idle()

    def test_is_idle_respects_timeout(self):
        cfg = DeepThinkingConfig(idle_timeout_s=5.0)
        sched = SleeptimeScheduler(cfg)
        sched._last_activity = time.time() - 6
        assert sched.is_idle()

    def test_not_idle_within_timeout(self):
        cfg = DeepThinkingConfig(idle_timeout_s=5.0)
        sched = SleeptimeScheduler(cfg)
        sched._last_activity = time.time()
        assert not sched.is_idle()

    def test_idle_for_s(self):
        cfg = DeepThinkingConfig(idle_timeout_s=5.0)
        sched = SleeptimeScheduler(cfg)
        sched._last_activity = time.time() - 10
        assert sched.idle_for_s >= 9.5

    def test_default_jobs_exist(self):
        assert len(DEFAULT_JOBS) == 3
        types = {j.job_type for j in DEFAULT_JOBS}
        assert types == {"consolidate", "retrospective", "precompute"}

    def test_default_jobs_priority_order(self):
        sorted_jobs = sorted(DEFAULT_JOBS, key=lambda j: j.priority, reverse=True)
        assert sorted_jobs[0].job_type == "consolidate"
        assert sorted_jobs[-1].job_type == "precompute"

    async def test_run_returns_immediately_when_disabled(self):
        cfg = DeepThinkingConfig(sleeptime_enabled=False)
        sched = SleeptimeScheduler(cfg)
        engine = MagicMock()
        # Should return immediately without entering the loop
        await sched.run(engine)
        engine.deliberate.assert_not_called()

    async def test_run_job_records_stats(self):
        cfg = DeepThinkingConfig(idle_timeout_s=1.0)
        sched = SleeptimeScheduler(cfg)

        mock_engine = AsyncMock()
        mock_engine.deliberate.return_value = DeliberationResult(
            proposal=Proposal(task_graph={"nodes": []}, reasoning="", confidence=0.5),
            critique=Critique(flaws=[], risks=[], score=0.5, suggestions=[]),
            consensus=Consensus(
                final_plan={}, incorporated_critiques=["a", "b"],
                rejected_critiques=[], confidence=0.5, dissent=[],
            ),
            stats=MagicMock(),
        )

        job = SleeptimeJob("consolidate", "test consolidation", priority=30)
        stats = await sched._run_job(job, mock_engine)
        assert stats.job_type == "consolidate"
        assert stats.cycle_number == 1
        assert stats.elapsed_s >= 0
        assert stats.memories_consolidated == 2  # len(incorporated_critiques)

    def test_summary(self):
        cfg = DeepThinkingConfig()
        sched = SleeptimeScheduler(cfg)
        s = sched.summary()
        assert s["cycles_completed"] == 0
        assert s["total_consolidated"] == 0
        assert "idle_now" in s

    def test_stop_sets_running_false(self):
        cfg = DeepThinkingConfig()
        sched = SleeptimeScheduler(cfg)
        sched._running = True
        sched.stop()
        assert sched._running is False


# ═══════════════════════════════════════════════════════════════════
# Integration: gate + deferred + tier recovery
# ═══════════════════════════════════════════════════════════════════

class TestGateTierRecovery:

    async def test_tier_recovery_triggers_deferred_drain(
        self, config, mock_client, deferred_queue
    ):
        """When rig recovers (DEGRADED → FULL), deferred queue should drain."""
        tier_state = TierState(config, mock_client)
        tier_state.tier = Tier.DEGRADED

        gate = MilestoneGate(config, mock_client, tier_state, deferred_queue)

        # Queue a job manually
        await deferred_queue.push(DeferredJob(
            job_type="gate_review",
            payload={"goal": "test"},
        ))
        assert deferred_queue.size == 1

        # Simulate tier recovery: DEGRADED → FULL
        transition = TierTransition(
            previous=Tier.DEGRADED,
            current=Tier.FULL,
            available_gpus=[0, 1, 2],
        )

        # _on_tier_recovery will try to create an asyncio task;
        # patch asyncio.create_task to capture the coroutine
        with patch("asyncio.create_task") as mock_create_task:
            gate._on_tier_recovery(transition)
            # Should have spawned a drain task
            mock_create_task.assert_called_once()

    async def test_no_drain_on_lateral_transition(
        self, config, mock_client, deferred_queue
    ):
        """FULL → PARTIAL doesn't drain (not a recovery)."""
        tier_state = TierState(config, mock_client)
        gate = MilestoneGate(config, mock_client, tier_state, deferred_queue)

        await deferred_queue.push(DeferredJob(job_type="gate_review", payload={}))

        transition = TierTransition(
            previous=Tier.FULL,
            current=Tier.PARTIAL,
            available_gpus=[0, 1],
        )

        with patch("asyncio.create_task") as mock_create_task:
            gate._on_tier_recovery(transition)
            mock_create_task.assert_not_called()

    async def test_no_drain_when_queue_empty(
        self, config, mock_client, deferred_queue
    ):
        """Recovery with empty queue is a no-op."""
        tier_state = TierState(config, mock_client)
        gate = MilestoneGate(config, mock_client, tier_state, deferred_queue)

        transition = TierTransition(
            previous=Tier.DARK,
            current=Tier.FULL,
            available_gpus=[0, 1, 2],
        )

        with patch("asyncio.create_task") as mock_create_task:
            gate._on_tier_recovery(transition)
            mock_create_task.assert_not_called()
