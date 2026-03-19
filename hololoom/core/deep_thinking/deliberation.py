"""
Deliberation Engine — Propose → Challenge → Resolve
=====================================================

Three-phase slow reasoning modeled on dreaming.py's collective
consolidation. Each phase runs on a dedicated GPU via AirLLM.

Bleed rates control information flow between phases:
    Planner → Critic:       100% of plan, 30% of reasoning
    Critic → Synthesizer:   100% of critique + full plan
    Synthesizer → out:      100% consensus propagates

Anti-groupthink (from dreaming.py):
    - Critic sees the plan but not full reasoning (forces independent eval)
    - Dissent is surfaced in Consensus.dissent, never averaged away
    - Each worker tracks independent Thompson priors

Adapts to tier:
    FULL:     3 GPUs → sequential pipeline, each phase on dedicated hardware
    PARTIAL:  1-2 GPUs → collapse synthesizer into critic, or skip critic
    DEGRADED: local 27B → all three roles serialized on one model, tighter prompts
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from hololoom.core.deep_thinking.client import ChatResult, DeepClient
from hololoom.core.deep_thinking.config import (
    ROLE_GPU_MAP,
    SYSTEM_PROMPTS,
    DeepThinkingConfig,
)
from hololoom.core.deep_thinking.tier import Tier, TierState

logger = logging.getLogger(__name__)


# ── Deliberation data types ───────────────────────────────────

@dataclass
class Proposal:
    """Output of the Planner phase."""
    task_graph: dict           # structured plan with nodes/edges
    reasoning: str             # full chain of thought
    confidence: float          # self-assessed 0-1
    raw: str = ""              # raw model output


@dataclass
class Critique:
    """Output of the Critic phase."""
    flaws: list[str]           # logical flaws found
    risks: list[str]           # identified risks
    score: float               # 0-1 soundness score
    suggestions: list[str]     # specific corrections
    raw: str = ""


@dataclass
class Consensus:
    """Output of the Synthesizer — the resolved deliberation."""
    final_plan: dict
    incorporated_critiques: list[str]
    rejected_critiques: list[str]    # with reasons
    confidence: float
    dissent: list[str]               # unresolved tensions (surfaced, not averaged)
    raw: str = ""


@dataclass
class DeliberationStats:
    """Timing and metadata for a deliberation cycle."""
    tier_used: Tier
    propose_elapsed_s: float = 0.0
    critique_elapsed_s: float = 0.0
    synthesize_elapsed_s: float = 0.0
    total_elapsed_s: float = 0.0
    bleed_rates: dict = field(default_factory=dict)


@dataclass
class DeliberationResult:
    """Complete output of one deliberation cycle."""
    proposal: Proposal
    critique: Critique
    consensus: Consensus
    stats: DeliberationStats


# ── Engine ─────────────────────────────────────────────────────

class DeliberationEngine:
    """
    Three-phase deliberation: Propose → Challenge → Resolve.

    Usage:
        engine = DeliberationEngine(config, client, tier_state)
        result = await engine.deliberate("Build a REST API for ...", context={...})
    """

    def __init__(
        self,
        config: DeepThinkingConfig,
        client: DeepClient,
        tier_state: TierState,
    ):
        self.config = config
        self.client = client
        self.tier = tier_state

    async def deliberate(
        self,
        goal: str,
        context: dict | None = None,
    ) -> DeliberationResult:
        """
        Run a full deliberation cycle.

        Adapts to current tier:
            FULL:     Planner(gpu0) → Critic(gpu1) → Synthesizer(gpu2)
            PARTIAL:  Planner(gpu_a) → Critic(gpu_b), synth collapsed into critic
            DEGRADED: All three roles on local 27B, serialized
        """
        context = context or {}
        t0 = time.time()
        tier = self.tier.tier

        # Phase 1: Propose
        t_propose = time.perf_counter()
        proposal = await self._propose(goal, context, tier)
        propose_elapsed = time.perf_counter() - t_propose

        # Phase 2: Challenge (with bleed control)
        critique = await self._critique(proposal, context, tier)

        # Phase 3: Resolve
        if tier >= Tier.FULL:
            # Full pipeline — dedicated synthesizer
            consensus = await self._synthesize(proposal, critique, context, tier)
        elif tier >= Tier.PARTIAL:
            # Partial — synthesizer collapsed into critic response
            consensus = self._extract_consensus_from_critique(proposal, critique)
        else:
            # Degraded — explicit synthesis on local model
            consensus = await self._synthesize(proposal, critique, context, tier)

        stats = DeliberationStats(
            tier_used=tier,
            propose_elapsed_s=propose_elapsed,
            total_elapsed_s=time.time() - t0,
            bleed_rates={
                "plan_to_critic": self.config.plan_to_critic_bleed,
                "plan_reasoning": self.config.plan_reasoning_bleed,
                "consensus": self.config.consensus_bleed,
            },
        )

        logger.info(
            f"Deliberation complete: tier={tier.name}, "
            f"confidence={consensus.confidence:.2f}, "
            f"dissent={len(consensus.dissent)}, "
            f"elapsed={stats.total_elapsed_s:.1f}s"
        )

        return DeliberationResult(
            proposal=proposal,
            critique=critique,
            consensus=consensus,
            stats=stats,
        )

    # ── Phase 1: Propose ──────────────────────────────────────

    async def _propose(
        self, goal: str, context: dict, tier: Tier
    ) -> Proposal:
        """Planner generates a structured task graph."""
        endpoint, model = self._resolve_endpoint("planner", tier)

        if endpoint is None:
            logger.info("No inference available for planner — returning deferred proposal")
            return Proposal(
                task_graph={"nodes": [], "deferred": True},
                reasoning="No inference available — deferred for retroactive review",
                confidence=0.0,
                raw="",
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["planner"]},
            {"role": "user", "content": self._build_planner_prompt(goal, context)},
        ]

        result = await self.client.chat(
            endpoint, model, messages,
            temperature=0.7,
            timeout_s=self.config.inference_timeout_s,
        )

        return self._parse_proposal(result)

    # ── Phase 2: Challenge ────────────────────────────────────

    async def _critique(
        self, proposal: Proposal, context: dict, tier: Tier
    ) -> Critique:
        """
        Critic evaluates the proposal.

        Bleed control: Critic sees the full plan but only a fraction
        of the Planner's reasoning (config.plan_to_critic_bleed for plan,
        config.plan_reasoning_bleed applied as reasoning truncation).
        This forces independent evaluation — anti-groupthink.
        """
        endpoint, model = self._resolve_endpoint("critic", tier)

        if endpoint is None:
            logger.info("No inference available for critic — returning deferred critique")
            return Critique(
                flaws=[],
                risks=[],
                score=0.0,
                suggestions=[],
                raw="No inference available — deferred for retroactive review",
            )

        # Apply bleed: plan flows fully, reasoning is truncated
        bleed = self.config.plan_reasoning_bleed  # 0.3
        reasoning_chars = int(len(proposal.reasoning) * bleed)
        bled_reasoning = proposal.reasoning[:reasoning_chars]
        if bled_reasoning and reasoning_chars < len(proposal.reasoning):
            bled_reasoning += "\n[... reasoning truncated for independent evaluation]"

        critic_context = (
            f"## Plan to evaluate\n"
            f"```json\n{json.dumps(proposal.task_graph, indent=2)}\n```\n\n"
            f"## Planner's reasoning (partial)\n{bled_reasoning}\n\n"
            f"## Planner's self-assessed confidence: {proposal.confidence:.2f}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["critic"]},
            {"role": "user", "content": critic_context},
        ]

        result = await self.client.chat(
            endpoint, model, messages,
            temperature=0.4,  # critic should be more deterministic
            timeout_s=self.config.inference_timeout_s,
        )

        return self._parse_critique(result)

    # ── Phase 3: Resolve ──────────────────────────────────────

    async def _synthesize(
        self,
        proposal: Proposal,
        critique: Critique,
        context: dict,
        tier: Tier,
    ) -> Consensus:
        """
        Synthesizer resolves proposal + critique into consensus.

        Full bleed: Synthesizer sees everything (100% both ways).
        Its job is to resolve tensions, not evaluate independently.
        """
        endpoint, model = self._resolve_endpoint("synthesizer", tier)

        if endpoint is None:
            logger.info("No inference available for synthesizer — returning deferred consensus")
            return Consensus(
                final_plan=proposal.task_graph,
                incorporated_critiques=[],
                rejected_critiques=[],
                confidence=0.0,
                dissent=[],
                raw="No inference available — deferred for retroactive review",
            )

        synth_context = (
            f"## Original Plan\n"
            f"```json\n{json.dumps(proposal.task_graph, indent=2)}\n```\n\n"
            f"## Planner's Reasoning\n{proposal.reasoning}\n\n"
            f"## Critic's Assessment (score: {critique.score:.2f})\n"
            f"### Flaws\n" + "\n".join(f"- {f}" for f in critique.flaws) + "\n\n"
            "### Risks\n" + "\n".join(f"- {r}" for r in critique.risks) + "\n\n"
            "### Suggestions\n" + "\n".join(f"- {s}" for s in critique.suggestions)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["synthesizer"]},
            {"role": "user", "content": synth_context},
        ]

        result = await self.client.chat(
            endpoint, model, messages,
            temperature=0.5,
            timeout_s=self.config.inference_timeout_s,
        )

        return self._parse_consensus(result)

    # ── Endpoint resolution ───────────────────────────────────

    def _resolve_endpoint(self, role: str, tier: Tier) -> tuple[str | None, str]:
        """Resolve which endpoint and model to use for a role at current tier.

        Returns (None, model) when no endpoint is available (DARK tier),
        signalling callers to return a synthetic deferred result.
        """
        if tier >= Tier.PARTIAL:
            gpu_idx = ROLE_GPU_MAP[role]
            endpoint = self.tier.endpoint_for_gpu(gpu_idx)
            if endpoint:
                return endpoint, self.config.rig_model

            # GPU for this role is down — use any available GPU
            fallback = self.tier.best_available_endpoint()
            if fallback and fallback != self.config.local_endpoint:
                return fallback, self.config.rig_model

        # DARK tier: no inference available — signal callers to defer
        if self.tier.best_available_endpoint() is None:
            return None, self.config.local_model

        # Fall through to local
        return self.config.local_endpoint, self.config.local_model

    # ── Prompt building ───────────────────────────────────────

    def _build_planner_prompt(self, goal: str, context: dict) -> str:
        parts = [f"## Goal\n{goal}"]
        if context:
            parts.append(f"## Context\n```json\n{json.dumps(context, indent=2)}\n```")
        parts.append(
            "## Instructions\n"
            "Produce a JSON task graph with:\n"
            "- \"nodes\": list of {\"id\", \"task\", \"depends_on\": [ids], \"output\"}\n"
            "- \"reasoning\": your step-by-step thinking\n"
            "- \"confidence\": 0.0-1.0 self-assessment\n"
            "- \"risks\": list of identified risks\n"
        )
        return "\n\n".join(parts)

    # ── Response parsing ──────────────────────────────────────

    def _parse_proposal(self, result: ChatResult) -> Proposal:
        """Parse planner output into Proposal."""
        try:
            data = _extract_json(result.content)
            return Proposal(
                task_graph={"nodes": data.get("nodes", [])},
                reasoning=data.get("reasoning", result.content),
                confidence=float(data.get("confidence", 0.5)),
                raw=result.content,
            )
        except (json.JSONDecodeError, KeyError):
            # Graceful fallback: treat entire output as reasoning
            return Proposal(
                task_graph={"nodes": [], "raw_plan": result.content},
                reasoning=result.content,
                confidence=0.5,
                raw=result.content,
            )

    def _parse_critique(self, result: ChatResult) -> Critique:
        """Parse critic output into Critique."""
        try:
            data = _extract_json(result.content)
            return Critique(
                flaws=data.get("flaws", []),
                risks=data.get("risks", []),
                score=float(data.get("score", 0.5)),
                suggestions=data.get("suggestions", []),
                raw=result.content,
            )
        except (json.JSONDecodeError, KeyError):
            return Critique(
                flaws=[],
                risks=[],
                score=0.5,
                suggestions=[],
                raw=result.content,
            )

    def _parse_consensus(self, result: ChatResult) -> Consensus:
        """Parse synthesizer output into Consensus."""
        try:
            data = _extract_json(result.content)
            return Consensus(
                final_plan=data.get("final_plan", data),
                incorporated_critiques=data.get("incorporated", []),
                rejected_critiques=data.get("rejected", []),
                confidence=float(data.get("confidence", 0.5)),
                dissent=data.get("dissent", []),
                raw=result.content,
            )
        except (json.JSONDecodeError, KeyError):
            return Consensus(
                final_plan={"raw": result.content},
                incorporated_critiques=[],
                rejected_critiques=[],
                confidence=0.5,
                dissent=[],
                raw=result.content,
            )

    def _extract_consensus_from_critique(
        self, proposal: Proposal, critique: Critique
    ) -> Consensus:
        """When synthesizer is unavailable, derive consensus from critique."""
        if critique.score >= 0.7:
            # Plan is mostly sound — use it with minor tweaks
            return Consensus(
                final_plan=proposal.task_graph,
                incorporated_critiques=critique.suggestions[:2],
                rejected_critiques=[],
                confidence=critique.score,
                dissent=critique.flaws,
                raw=f"[auto-consensus from critique score {critique.score:.2f}]",
            )
        else:
            # Plan needs significant revision
            return Consensus(
                final_plan=proposal.task_graph,
                incorporated_critiques=critique.suggestions,
                rejected_critiques=[],
                confidence=critique.score * 0.8,
                dissent=critique.flaws + critique.risks,
                raw=f"[auto-consensus: low critique score {critique.score:.2f}, needs revision]",
            )


# ── Utilities ──────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON from model output, handling markdown code blocks."""
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try finding first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    raise json.JSONDecodeError("No JSON found", text, 0)
