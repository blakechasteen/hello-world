"""Tests for weaverlet.evaluation.evaluator — WeaverletEvaluator."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from typing import List, Optional

from hololoom.core.fabric.spacetime import Artifact, ArtifactType
from hololoom.weaverlet.schemas import Budget, BudgetLedger, WorkingState
from hololoom.weaverlet.shuttle.http_client import InferResponse
from hololoom.weaverlet.evaluation.evaluator import WeaverletEvaluator


# ── Mock FabricInspector ─────────────────────────────────────────────────

@dataclass
class MockFabricCheckResult:
    confidence: float
    passed: bool
    blockers: List[str]
    recommendations: List[str]
    signal_results: dict


class MockInspector:
    def __init__(self, confidence=0.8, blockers=None, recommendations=None):
        self.confidence = confidence
        self.blockers = blockers or []
        self.recommendations = recommendations or []

    async def inspect(self, thread, context):
        return MockFabricCheckResult(
            confidence=self.confidence,
            passed=self.confidence >= 0.6,
            blockers=self.blockers,
            recommendations=self.recommendations,
            signal_results={},
        )


class MockInferClient:
    def __init__(self, response_text="SCORE: 0.7\nCRITIQUE: Looking good.", tokens=50):
        self.response_text = response_text
        self.tokens = tokens

    async def infer(self, prompt, **kwargs):
        return InferResponse(
            content=self.response_text,
            tokens_used=self.tokens,
            model="mock",
            latency_ms=10.0,
        )


# ── Tests ────────────────────────────────────────────────────────────────

class TestWeaverletEvaluator:

    @pytest.mark.asyncio
    async def test_both_sources(self):
        """Blended score: 0.6 * 0.8 (signal) + 0.4 * 0.7 (model) = 0.76."""
        inspector = MockInspector(confidence=0.8)
        infer_client = MockInferClient("SCORE: 0.7\nCRITIQUE: Good progress.")
        evaluator = WeaverletEvaluator(
            inspector=inspector,
            infer_client=infer_client,
            signal_weight=0.6,
            self_eval_weight=0.4,
        )
        state = WorkingState(goal="test task")
        score, critique = await evaluator.evaluate(state)
        assert abs(score - 0.76) < 0.01
        assert "Good progress" in critique

    @pytest.mark.asyncio
    async def test_signal_only(self):
        """No infer client → signal score gets full weight."""
        inspector = MockInspector(confidence=0.65)
        evaluator = WeaverletEvaluator(inspector=inspector, infer_client=None)
        state = WorkingState(goal="test")
        score, critique = await evaluator.evaluate(state)
        assert abs(score - 0.65) < 0.01

    @pytest.mark.asyncio
    async def test_model_only(self):
        """No inspector → model score gets full weight."""
        infer_client = MockInferClient("SCORE: 0.85\nCRITIQUE: Nearly done.")
        evaluator = WeaverletEvaluator(inspector=None, infer_client=infer_client)
        state = WorkingState(goal="test")
        score, critique = await evaluator.evaluate(state)
        assert abs(score - 0.85) < 0.01
        assert "Nearly done" in critique

    @pytest.mark.asyncio
    async def test_neither_source(self):
        """No inspector, no infer client → (0.5, fallback critique)."""
        evaluator = WeaverletEvaluator(inspector=None, infer_client=None)
        state = WorkingState(goal="test")
        score, critique = await evaluator.evaluate(state)
        assert score == 0.5
        assert "No evaluation available" in critique

    @pytest.mark.asyncio
    async def test_signal_failure_falls_back(self):
        """If inspector raises, falls back to model-only."""

        class FailingInspector:
            async def inspect(self, thread, context):
                raise RuntimeError("signal pipeline error")

        infer_client = MockInferClient("SCORE: 0.6\nCRITIQUE: ok")
        evaluator = WeaverletEvaluator(
            inspector=FailingInspector(), infer_client=infer_client
        )
        state = WorkingState(goal="test")
        score, critique = await evaluator.evaluate(state)
        assert abs(score - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_model_failure_falls_back(self):
        """If infer client raises, falls back to signal-only."""

        class FailingClient:
            async def infer(self, prompt, **kwargs):
                raise ConnectionError("daemon down")

        inspector = MockInspector(confidence=0.7)
        evaluator = WeaverletEvaluator(
            inspector=inspector, infer_client=FailingClient()
        )
        state = WorkingState(goal="test")
        score, critique = await evaluator.evaluate(state)
        assert abs(score - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_tokens_charged_to_ledger(self):
        infer_client = MockInferClient("SCORE: 0.5\nCRITIQUE: ok", tokens=100)
        evaluator = WeaverletEvaluator(inspector=None, infer_client=infer_client)
        ledger = BudgetLedger(budget=Budget(max_tokens=10000))
        state = WorkingState(goal="test", ledger=ledger)
        await evaluator.evaluate(state)
        assert ledger.tokens_used == 100

    @pytest.mark.asyncio
    async def test_blockers_in_critique(self):
        inspector = MockInspector(
            confidence=0.3, blockers=["missing tests", "no docstring"]
        )
        evaluator = WeaverletEvaluator(inspector=inspector, infer_client=None)
        state = WorkingState(goal="test")
        score, critique = await evaluator.evaluate(state)
        assert "missing tests" in critique
        assert "no docstring" in critique

    @pytest.mark.asyncio
    async def test_recommendations_in_critique(self):
        inspector = MockInspector(
            confidence=0.7, recommendations=["Add error handling", "Write tests"]
        )
        evaluator = WeaverletEvaluator(inspector=inspector, infer_client=None)
        state = WorkingState(goal="test")
        _, critique = await evaluator.evaluate(state)
        assert "Add error handling" in critique
        assert "Write tests" in critique

    @pytest.mark.asyncio
    async def test_artifact_in_model_eval(self):
        """Model eval includes artifact content in prompt."""
        captured_prompts = []

        class CapturingClient:
            async def infer(self, prompt, **kwargs):
                captured_prompts.append(prompt)
                return InferResponse(
                    content="SCORE: 0.5\nCRITIQUE: ok",
                    tokens_used=10,
                    model="mock",
                    latency_ms=5.0,
                )

        evaluator = WeaverletEvaluator(inspector=None, infer_client=CapturingClient())
        state = WorkingState(
            goal="build thing",
            artifact=Artifact(name="code", type=ArtifactType.DOCUMENT, content="def hello(): pass"),
        )
        await evaluator.evaluate(state)
        assert "def hello(): pass" in captured_prompts[0]


class TestParseEvalResponse:
    def test_standard_format(self):
        evaluator = WeaverletEvaluator()
        score, critique = evaluator._parse_eval_response(
            "SCORE: 0.85\nCRITIQUE: Good structure but needs tests."
        )
        assert abs(score - 0.85) < 0.01
        assert "Good structure" in critique

    def test_missing_score(self):
        evaluator = WeaverletEvaluator()
        score, _ = evaluator._parse_eval_response("No score here, just text.")
        assert score == 0.5  # default

    def test_score_clamped_high(self):
        evaluator = WeaverletEvaluator()
        score, _ = evaluator._parse_eval_response("SCORE: 1.5\nCRITIQUE: over")
        assert score == 1.0

    def test_score_clamped_low(self):
        evaluator = WeaverletEvaluator()
        score, _ = evaluator._parse_eval_response("SCORE: -0.3\nCRITIQUE: under")
        assert score == 0.0


class TestBlend:
    def test_both(self):
        ev = WeaverletEvaluator(signal_weight=0.6, self_eval_weight=0.4)
        assert abs(ev._blend(0.8, 0.5) - 0.68) < 0.01

    def test_signal_only(self):
        ev = WeaverletEvaluator()
        assert ev._blend(0.7, None) == 0.7

    def test_model_only(self):
        ev = WeaverletEvaluator()
        assert ev._blend(None, 0.6) == 0.6

    def test_neither(self):
        ev = WeaverletEvaluator()
        assert ev._blend(None, None) == 0.5
