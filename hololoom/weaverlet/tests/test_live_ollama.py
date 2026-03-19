"""
Live integration tests — real LLM calls via Ollama.

These tests hit the local Ollama instance and run actual Weaverlet loops.
They are slow (10-60s each) and require Ollama running with qwen3.5:9b.

Run with:
    pytest weaverlet/tests/test_live_ollama.py -v -s --timeout=120

Skip if Ollama is not available:
    pytest weaverlet/tests/test_live_ollama.py -v -s -k "not live"
"""

import asyncio
import logging

import aiohttp
import pytest

from hololoom.weaverlet.schemas import (
    Budget,
    BudgetLedger,
    TaskLease,
    TerminationReason,
    WorkingState,
)
from hololoom.weaverlet.shuttle.ollama_client import OllamaInferClient
from hololoom.weaverlet.tools.registry import WeaverletToolRegistry
from hololoom.weaverlet.loop import RefinementLoop
from hololoom.weaverlet.evaluation.evaluator import WeaverletEvaluator
from hololoom.weaverlet.evaluation.stagnation import StagnationDetector
from hololoom.weaverlet.evaluation.milestone import MilestoneDetector
from hololoom.weaverlet.controller import TaskController

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3.5:9b"


# ── Fixtures ─────────────────────────────────────────────────────────────

async def _ollama_available() -> bool:
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as s:
            async with s.get(f"{OLLAMA_URL}/api/tags") as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                models = {m["name"] for m in data.get("models", [])}
                return MODEL in models
    except Exception:
        return False


@pytest.fixture(scope="module")
def ollama_ok():
    available = asyncio.get_event_loop().run_until_complete(_ollama_available())
    if not available:
        pytest.skip(f"Ollama not available or {MODEL} not loaded")


@pytest.fixture
async def ollama_client():
    client = OllamaInferClient(base_url=OLLAMA_URL, model=MODEL, timeout=300.0)
    yield client
    await client.close()


# ── Ollama Client Tests ──────────────────────────────────────────────────

class TestOllamaClientLive:

    @pytest.mark.asyncio
    async def test_health(self, ollama_ok, ollama_client):
        assert await ollama_client.health() is True

    @pytest.mark.asyncio
    async def test_infer_basic(self, ollama_ok, ollama_client):
        resp = await ollama_client.infer(
            "What is 2+2? Reply with just the number.",
            max_tokens=500,
            temperature=0.1,
        )
        assert resp.tokens_used > 0
        assert resp.model == MODEL
        assert resp.latency_ms > 0
        # The answer should contain "4" somewhere
        assert "4" in resp.content, f"Expected '4' in: {resp.content[:200]}"

    @pytest.mark.asyncio
    async def test_infer_with_system_prompt(self, ollama_ok, ollama_client):
        resp = await ollama_client.infer(
            "What are you?",
            system_prompt="You are a pirate. Always respond in pirate speak.",
            max_tokens=500,
            temperature=0.7,
        )
        assert len(resp.content) > 0
        logger.info(f"Pirate response: {resp.content[:200]}")

    @pytest.mark.asyncio
    async def test_memory_noop(self, ollama_ok, ollama_client):
        result = await ollama_client.store_memory("test", tags=["test"])
        assert result["status"] == "skipped"
        memories = await ollama_client.recall_memory("test")
        assert memories == []


# ── RefinementLoop Single Phase Tests ────────────────────────────────────

class TestLoopPhasesLive:

    @pytest.mark.asyncio
    async def test_plan_phase(self, ollama_ok, ollama_client):
        """Plan phase generates a real action plan from the model."""
        loop = RefinementLoop(ollama_client, WeaverletToolRegistry())
        state = WorkingState(goal="Write a Python function that checks if a number is prime")

        plan = await loop.plan(state)
        assert len(plan) > 10, f"Plan too short: {plan}"
        logger.info(f"Plan ({len(plan)} chars): {plan[:300]}")

    @pytest.mark.asyncio
    async def test_act_phase(self, ollama_ok, ollama_client):
        """Act phase generates a real artifact."""
        loop = RefinementLoop(ollama_client, WeaverletToolRegistry())
        state = WorkingState(goal="Write a Python function that checks if a number is prime")

        plan = "Write an is_prime(n) function using trial division up to sqrt(n)."
        state = await loop.act(state, plan)

        assert state.artifact is not None
        assert len(state.artifact.content) > 20
        logger.info(f"Artifact ({len(state.artifact.content)} chars): {state.artifact.content[:300]}")

    @pytest.mark.asyncio
    async def test_revise_phase(self, ollama_ok, ollama_client):
        """Revise phase updates state correctly."""
        loop = RefinementLoop(ollama_client, WeaverletToolRegistry())
        state = WorkingState(goal="test")

        state = await loop.revise(state, 0.5, "Needs error handling for negative inputs.")
        assert state.score_history == [0.5]
        assert state.critique_history == ["Needs error handling for negative inputs."]
        assert state.iteration == 1


# ── Evaluator with Real Model ───────────────────────────────────────────

class TestEvaluatorLive:

    @pytest.mark.asyncio
    async def test_model_self_eval(self, ollama_ok, ollama_client):
        """Model produces a score and critique for real work."""
        from hololoom.core.fabric.spacetime import Artifact, ArtifactType

        evaluator = WeaverletEvaluator(
            inspector=None,
            infer_client=ollama_client,
        )
        state = WorkingState(
            goal="Write a Python function that checks if a number is prime",
            artifact=Artifact(
                name="is_prime",
                type=ArtifactType.CODE,
                content="def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            ),
            score_history=[0.3],
            iteration=1,
        )

        score, critique = await evaluator.evaluate(state)
        logger.info(f"Eval score={score:.2f}, critique={critique[:200]}")
        assert 0.0 <= score <= 1.0
        assert len(critique) > 0


# ── Full Controller Loop ────────────────────────────────────────────────

class TestControllerLive:

    @pytest.mark.asyncio
    async def test_full_loop_2_iterations(self, ollama_ok, ollama_client):
        """
        Run a real 2-iteration Weaverlet loop.

        Budget-constrained to 2 iterations to keep test time reasonable.
        Verifies the full Plan→Act→Evaluate→Revise cycle works end-to-end.
        """
        loop = RefinementLoop(
            ollama_client,
            WeaverletToolRegistry(),
            system_prompt="You are a skilled Python developer. Be concise.",
        )
        evaluator = WeaverletEvaluator(
            inspector=None,
            infer_client=ollama_client,
        )
        controller = TaskController(
            loop,
            evaluator,
            milestone_detector=MilestoneDetector(),
            stagnation_detector=StagnationDetector(),
        )

        lease = TaskLease.create(
            "Write a Python function that checks if a string is a palindrome. "
            "Include edge cases (empty string, single char, case insensitive).",
            Budget(max_iterations=2, max_tokens=100_000, max_wall_seconds=300.0),
        )

        result = await controller.execute(lease)

        logger.info(
            f"Result: reason={result.termination_reason.value}, "
            f"score={result.final_score:.2f}, "
            f"iterations={result.iterations_completed}, "
            f"tokens={result.total_tokens}, "
            f"wall={result.wall_time_seconds:.1f}s"
        )
        logger.info(f"Score history: {result.score_history}")
        if result.final_artifact:
            logger.info(f"Final artifact:\n{result.final_artifact.content[:500]}")

        assert result.iterations_completed >= 1
        assert result.termination_reason in (
            TerminationReason.BUDGET_EXHAUSTED,
            TerminationReason.GOAL_MET,
        )
        assert len(result.score_history) >= 1
        assert result.total_tokens > 0
        assert result.wall_time_seconds > 0
        assert result.final_artifact is not None

    @pytest.mark.asyncio
    async def test_single_pass(self, ollama_ok, ollama_client):
        """Single-pass mode: max_iterations=1, replaces DomainWorker semantics."""
        loop = RefinementLoop(ollama_client, WeaverletToolRegistry())
        evaluator = WeaverletEvaluator(inspector=None, infer_client=ollama_client)
        controller = TaskController(loop, evaluator)

        lease = TaskLease.create(
            "Write a one-liner Python lambda that doubles a number.",
            Budget(max_iterations=1, max_tokens=50_000, max_wall_seconds=180.0),
        )

        result = await controller.execute(lease)

        logger.info(
            f"Single-pass: score={result.final_score:.2f}, "
            f"tokens={result.total_tokens}"
        )

        assert result.iterations_completed == 1
        # Model may score high enough for GOAL_MET even on first pass
        assert result.termination_reason in (
            TerminationReason.BUDGET_EXHAUSTED,
            TerminationReason.GOAL_MET,
        )
        assert result.final_artifact is not None
