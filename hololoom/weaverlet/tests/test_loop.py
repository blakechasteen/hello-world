"""Tests for weaverlet.loop — RefinementLoop (Plan→Act→Revise)."""

import pytest

from hololoom.core.fabric.spacetime import Artifact, ArtifactType
from hololoom.weaverlet.schemas import Budget, BudgetLedger, WorkingState
from hololoom.weaverlet.shuttle.http_client import InferResponse
from hololoom.weaverlet.loop import RefinementLoop


# ── Mock infrastructure ──────────────────────────────────────────────────

class MockInferClient:
    """Records calls, returns configurable responses."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses or [])
        self._default = InferResponse(
            content="default response",
            tokens_used=50,
            model="mock",
            latency_ms=10.0,
        )

    async def infer(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self._responses:
            return self._responses.pop(0)
        return self._default


class MockToolRegistry:
    """Records tool calls, returns configurable results."""

    def __init__(self, results=None):
        self.calls = []
        self._results = results or {}

    async def execute(self, tool_name, **kwargs):
        self.calls.append({"tool": tool_name, "args": kwargs})
        return self._results.get(tool_name, {"success": True})

    def list_tools(self):
        return list(self._results.keys())


# ── Plan phase ───────────────────────────────────────────────────────────

class TestPlan:

    @pytest.mark.asyncio
    async def test_plan_returns_string(self):
        client = MockInferClient([
            InferResponse(content="Step 1: do X\nStep 2: do Y", tokens_used=30, model="m", latency_ms=5)
        ])
        loop = RefinementLoop(client, MockToolRegistry())
        state = WorkingState(goal="build a thing")

        plan = await loop.plan(state)
        assert "Step 1" in plan
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_plan_charges_tokens(self):
        client = MockInferClient([
            InferResponse(content="plan", tokens_used=100, model="m", latency_ms=5)
        ])
        loop = RefinementLoop(client, MockToolRegistry())
        ledger = BudgetLedger(budget=Budget(max_tokens=10000))
        state = WorkingState(goal="test", ledger=ledger)

        await loop.plan(state)
        assert ledger.tokens_used == 100

    @pytest.mark.asyncio
    async def test_plan_no_ledger(self):
        client = MockInferClient()
        loop = RefinementLoop(client, MockToolRegistry())
        state = WorkingState(goal="test", ledger=None)

        plan = await loop.plan(state)
        assert isinstance(plan, str)

    @pytest.mark.asyncio
    async def test_plan_includes_goal(self):
        client = MockInferClient()
        loop = RefinementLoop(client, MockToolRegistry())
        state = WorkingState(goal="implement authentication")

        await loop.plan(state)
        assert "implement authentication" in client.calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_plan_includes_critique(self):
        client = MockInferClient()
        loop = RefinementLoop(client, MockToolRegistry())
        state = WorkingState(
            goal="test",
            critique_history=["Needs better error handling"],
        )

        await loop.plan(state)
        assert "better error handling" in client.calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_plan_includes_score_trend(self):
        client = MockInferClient()
        loop = RefinementLoop(client, MockToolRegistry())
        state = WorkingState(goal="test", score_history=[0.2, 0.4, 0.6])

        await loop.plan(state)
        assert "0.20" in client.calls[0]["prompt"]
        assert "0.60" in client.calls[0]["prompt"]


# ── Act phase ────────────────────────────────────────────────────────────

class TestAct:

    @pytest.mark.asyncio
    async def test_act_updates_artifact(self):
        client = MockInferClient([
            InferResponse(content="new artifact content", tokens_used=80, model="m", latency_ms=5)
        ])
        loop = RefinementLoop(client, MockToolRegistry())
        state = WorkingState(goal="test")

        state = await loop.act(state, "just generate output")
        assert state.artifact is not None
        assert state.artifact.content == "new artifact content"

    @pytest.mark.asyncio
    async def test_act_charges_tokens(self):
        client = MockInferClient([
            InferResponse(content="out", tokens_used=200, model="m", latency_ms=5)
        ])
        loop = RefinementLoop(client, MockToolRegistry())
        ledger = BudgetLedger(budget=Budget(max_tokens=10000))
        state = WorkingState(goal="test", ledger=ledger)

        await loop.act(state, "plan")
        assert ledger.tokens_used == 200

    @pytest.mark.asyncio
    async def test_act_extracts_tool_calls(self):
        client = MockInferClient()
        registry = MockToolRegistry({"code_execute": {"success": True, "stdout": "hello"}})
        loop = RefinementLoop(client, registry)
        state = WorkingState(goal="test")

        plan = 'TOOL: code_execute(code="print(42)")'
        await loop.act(state, plan)
        assert len(registry.calls) == 1
        assert registry.calls[0]["tool"] == "code_execute"
        assert registry.calls[0]["args"]["code"] == "print(42)"

    @pytest.mark.asyncio
    async def test_act_appends_evidence(self):
        client = MockInferClient()
        registry = MockToolRegistry({"code_execute": {"success": True}})
        loop = RefinementLoop(client, registry)
        state = WorkingState(goal="test")

        plan = 'TOOL: code_execute(code="x")'
        await loop.act(state, plan)
        assert len(state.evidence) == 1
        assert state.evidence[0]["tool"] == "code_execute"

    @pytest.mark.asyncio
    async def test_act_no_tool_calls(self):
        client = MockInferClient()
        registry = MockToolRegistry()
        loop = RefinementLoop(client, registry)
        state = WorkingState(goal="test")

        await loop.act(state, "Just improve the text.")
        assert len(registry.calls) == 0


# ── Revise phase ─────────────────────────────────────────────────────────

class TestRevise:

    @pytest.mark.asyncio
    async def test_revise_appends_score(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        state = WorkingState(goal="test")

        state = await loop.revise(state, 0.5, "needs work")
        assert state.score_history == [0.5]
        assert state.critique_history == ["needs work"]

    @pytest.mark.asyncio
    async def test_revise_increments_iteration(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        state = WorkingState(goal="test", iteration=2)

        state = await loop.revise(state, 0.6, "ok")
        assert state.iteration == 3

    @pytest.mark.asyncio
    async def test_revise_charges_iteration(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        ledger = BudgetLedger(budget=Budget(max_iterations=10))
        state = WorkingState(goal="test", ledger=ledger)

        await loop.revise(state, 0.5, "ok")
        assert ledger.iterations_used == 1

    @pytest.mark.asyncio
    async def test_revise_extracts_questions(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        state = WorkingState(goal="test")

        await loop.revise(state, 0.5, "Good start. Should we add logging? What about auth?")
        assert len(state.open_questions) > 0

    @pytest.mark.asyncio
    async def test_revise_no_questions(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        state = WorkingState(goal="test", open_questions=["old question"])

        await loop.revise(state, 0.5, "Looks perfect, no issues found.")
        # Old questions preserved when no new ones extracted
        # (only replaced if questions found)


# ── Build iteration output ───────────────────────────────────────────────

class TestBuildIterationOutput:

    def test_basic(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        state = WorkingState(
            goal="test",
            score_history=[0.3, 0.5],
            critique_history=["meh", "better"],
            iteration=2,
        )

        out = loop.build_iteration_output(state, "next plan")
        assert out.progress_score == 0.5
        assert out.critique == "better"
        assert out.next_action == "next plan"
        assert len(out.iteration_id) == 12

    def test_with_ledger(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        ledger = BudgetLedger(budget=Budget())
        ledger.tokens_used = 500
        ledger.tool_calls_used = 3
        state = WorkingState(goal="test", score_history=[0.4], critique_history=["ok"], ledger=ledger)

        out = loop.build_iteration_output(state, "plan")
        assert out.tokens_used == 500
        assert out.tool_calls_used == 3


# ── Tool extraction helpers ──────────────────────────────────────────────

class TestExtractToolCalls:

    def test_single_tool(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        calls = loop._extract_tool_calls('TOOL: code_execute(code="print(1)")')
        assert len(calls) == 1
        assert calls[0]["tool"] == "code_execute"
        assert calls[0]["args"]["code"] == "print(1)"

    def test_multiple_tools(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        plan = (
            'Step 1: TOOL: code_execute(code="x=1")\n'
            'Step 2: TOOL: memory_query_yarn(cypher="MATCH (n) RETURN n")'
        )
        calls = loop._extract_tool_calls(plan)
        assert len(calls) == 2

    def test_no_tools(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        assert loop._extract_tool_calls("Just do it manually.") == []

    def test_case_insensitive(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        calls = loop._extract_tool_calls('tool: code_execute(code="hi")')
        assert len(calls) == 1

    def test_single_quotes(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        calls = loop._extract_tool_calls("TOOL: code_execute(code='hello')")
        assert calls[0]["args"]["code"] == "hello"


class TestExtractQuestions:

    def test_basic(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        questions = loop._extract_questions("Good work. Should we add tests? What about docs?")
        assert len(questions) >= 1

    def test_no_questions(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        questions = loop._extract_questions("Everything looks perfect.")
        assert questions == []

    def test_max_five(self):
        loop = RefinementLoop(MockInferClient(), MockToolRegistry())
        text = "Q1? Q2? Q3? Q4? Q5? Q6? Q7?"
        questions = loop._extract_questions(text)
        assert len(questions) <= 5
