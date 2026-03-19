"""
Refinement Loop
================

The 4-phase iterative cycle: Plan -> Act -> Evaluate -> Revise.

Each phase is a composable async method. The loop itself is stateless;
all state lives in WorkingState.

Replaces DomainWorker._act() with an iterative refinement pattern.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from hololoom.core.fabric.spacetime import Artifact, ArtifactType
from hololoom.weaverlet.schemas import IterationOutput, WorkingState
from hololoom.weaverlet.shuttle.http_client import DaemonInferClient
from hololoom.weaverlet.tools.registry import WeaverletToolRegistry

logger = logging.getLogger(__name__)

# ── Prompt Templates ──────────────────────────────────────────────────────

PLAN_PROMPT = """You are an iterative worker refining a solution.

## Goal
{goal}

## Current State
- Iteration: {iteration}
- Score trend: {score_trend}
- Latest critique: {latest_critique}
- Open questions: {open_questions}

## Current Artifact
{artifact_summary}

## Instructions
Produce a concrete action plan for the next iteration. Focus on the highest-impact
improvement based on the critique. Be specific about what tools to use and what
changes to make.

Respond with a clear, step-by-step plan (3-5 steps max).
"""

ACT_PROMPT = """Execute this plan and produce an updated artifact.

## Plan
{plan}

## Current Artifact
{artifact_summary}

## Goal
{goal}

Produce the updated artifact content. If this is code, output the full updated code.
If this is a document, output the full updated document.
"""


class RefinementLoop:

    def __init__(
        self,
        infer_client: DaemonInferClient,
        tool_registry: WeaverletToolRegistry,
        system_prompt: str = "",
    ):
        self.infer_client = infer_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt

    async def plan(self, state: WorkingState) -> str:
        """
        Phase 1: Analyze current state, produce an action plan.

        Sends state summary to daemon /infer, returns plan string.
        Charges tokens to ledger.
        """
        artifact_summary = self._summarize_artifact(state)
        latest_critique = state.critique_history[-1] if state.critique_history else "None yet"
        score_trend = self._format_score_trend(state.score_history)

        prompt = PLAN_PROMPT.format(
            goal=state.goal,
            iteration=state.iteration,
            score_trend=score_trend,
            latest_critique=latest_critique,
            open_questions=", ".join(state.open_questions[:5]) or "None",
            artifact_summary=artifact_summary,
        )

        resp = await self.infer_client.infer(
            prompt,
            system_prompt=self.system_prompt,
            max_tokens=4096,
            temperature=0.7,
        )

        if state.ledger:
            state.ledger.charge_tokens(resp.tokens_used)

        logger.debug(f"Plan generated ({resp.tokens_used} tokens)")
        return resp.content

    async def act(self, state: WorkingState, plan: str) -> WorkingState:
        """
        Phase 2: Execute the plan using tools and model.

        May invoke tools from the registry. Updates state.artifact and
        state.evidence. Charges tool_calls to ledger.
        """
        # Extract tool calls from plan (if any)
        tool_calls = self._extract_tool_calls(plan)

        evidence: list[dict[str, Any]] = []

        for call in tool_calls:
            result = await self.tool_registry.execute(
                call["tool"], **call.get("args", {})
            )
            evidence.append({
                "tool": call["tool"],
                "args": call.get("args", {}),
                "result": result,
            })

        # Generate updated artifact via model
        artifact_summary = self._summarize_artifact(state)
        prompt = ACT_PROMPT.format(
            plan=plan,
            artifact_summary=artifact_summary,
            goal=state.goal,
        )

        resp = await self.infer_client.infer(
            prompt,
            system_prompt=self.system_prompt,
            max_tokens=4096,
            temperature=0.5,
        )

        if state.ledger:
            state.ledger.charge_tokens(resp.tokens_used)

        # Update artifact
        state.artifact = Artifact(
            name=state.artifact.name if state.artifact else "artifact",
            type=state.artifact.type if state.artifact else ArtifactType.DOCUMENT,
            content=resp.content,
        )

        state.evidence.extend(evidence)

        logger.debug(
            f"Act completed: {len(tool_calls)} tool calls, "
            f"{resp.tokens_used} tokens"
        )
        return state

    async def revise(self, state: WorkingState, score: float, critique: str) -> WorkingState:
        """
        Phase 4: Update state based on evaluation results.

        Appends to score and critique histories, extracts open questions
        from critique text.
        """
        state.score_history.append(score)
        state.critique_history.append(critique)

        # Extract open questions from critique
        questions = self._extract_questions(critique)
        if questions:
            state.open_questions = questions

        state.iteration += 1
        if state.ledger:
            state.ledger.charge_iteration()

        return state

    def build_iteration_output(self, state: WorkingState, plan: str) -> IterationOutput:
        """Build a sealed IterationOutput record from current state."""
        return IterationOutput(
            iteration_id=uuid.uuid4().hex[:12],
            progress_score=state.latest_score,
            updated_artifact=state.artifact,
            critique=state.critique_history[-1] if state.critique_history else "",
            next_action=plan,
            evidence_refs=[
                e.get("tool", "unknown") for e in state.evidence[-5:]
            ],
            blockers=[q for q in state.open_questions if "block" in q.lower()],
            tokens_used=state.ledger.tokens_used if state.ledger else 0,
            tool_calls_used=state.ledger.tool_calls_used if state.ledger else 0,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _summarize_artifact(self, state: WorkingState) -> str:
        if state.artifact is None:
            return "(no artifact yet)"
        content = state.artifact.content
        if isinstance(content, str):
            return content[:3000]
        if content is not None:
            return str(content)[:3000]
        return f"[{state.artifact.type.value}: {state.artifact.name}]"

    def _format_score_trend(self, history: list[float]) -> str:
        if not history:
            return "No scores yet"
        recent = history[-5:]
        return " -> ".join(f"{s:.2f}" for s in recent)

    def _extract_tool_calls(self, plan: str) -> list[dict[str, Any]]:
        """
        Extract tool calls from plan text.

        Looks for patterns like:
            TOOL: code_execute(code="print('hello')")
            TOOL: memory_query_yarn(cypher="MATCH ...")
        """
        calls: list[dict[str, Any]] = []
        for match in re.finditer(
            r"TOOL:\s*(\w+)\(([^)]*)\)", plan, re.IGNORECASE
        ):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            args: dict[str, Any] = {}
            if args_str:
                # Simple key=value parsing
                for pair in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
                    args[pair.group(1)] = pair.group(2)
                for pair in re.finditer(r"(\w+)\s*=\s*'([^']*)'", args_str):
                    args[pair.group(1)] = pair.group(2)
            calls.append({"tool": tool_name, "args": args})
        return calls

    def _extract_questions(self, critique: str) -> list[str]:
        """Extract sentences ending with ? from critique text."""
        return [
            s.strip()
            for s in re.split(r"[.!?\n]", critique)
            if s.strip().endswith("?") or "?" in s
        ][:5]
