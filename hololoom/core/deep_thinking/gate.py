"""
Milestone Gate — Agent Checkpoint Reviewer
==========================================

At each milestone in agent orchestration, work-so-far is submitted
to the 108B model for deep review. Returns a verdict: proceed,
revise, or abort.

Thompson Sampling priors (reusing policy/thompson_sampling.py's
TSBandit pattern) track which verdict types at which confidence
levels were ultimately correct. Over time, the gate learns the
right auto-approve threshold.

Tier-aware:
    FULL/PARTIAL: 108B reviews, auto-approve > 0.9
    DEGRADED: 27B reviews, auto-approve > 0.7, defer below 0.5
    DARK: everything deferred
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from hololoom.core.deep_thinking.client import DeepClient
from hololoom.core.deep_thinking.config import GATE_PROMPT, DeepThinkingConfig
from hololoom.core.deep_thinking.deferred import DeferredJob, DeferredQueue
from hololoom.core.deep_thinking.tier import Tier, TierState

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────

@dataclass
class MilestoneContext:
    """Everything the gate needs to evaluate a milestone."""
    goal: str
    work_so_far: dict
    milestone_description: str
    milestone_number: int
    total_milestones: int
    agent_id: str | None = None


@dataclass
class GateVerdict:
    """The gate's decision on a milestone."""
    verdict: Literal["proceed", "revise", "abort"]
    confidence: float          # 0-1
    corrections: str | None # specific fixes if verdict == "revise"
    reasoning: str
    deferred: bool = False     # True if queued for retroactive 108B review
    model_used: str = ""
    elapsed_s: float = 0.0


# ── Thompson priors for verdict learning ──────────────────────

@dataclass
class VerdictPrior:
    """
    Beta distribution tracking whether a verdict type was correct.

    Mirrors policy/thompson_sampling.py's TSBandit but simplified
    for three arms (proceed/revise/abort).
    """
    # Per-verdict BetaArm (updated 2026-03-12, Governance Phase 4)
    _proceed: BetaArm = field(default=None)  # type: ignore[assignment]
    _revise:  BetaArm = field(default=None)  # type: ignore[assignment]
    _abort:   BetaArm = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        from hololoom.bandits.beta_arm import BetaArm
        if self._proceed is None:
            self._proceed = BetaArm()
        if self._revise is None:
            self._revise = BetaArm()
        if self._abort is None:
            self._abort = BetaArm()

    def _arm(self, verdict: str):
        return getattr(self, f"_{verdict}")

    # Backward-compatible tuple properties
    @property
    def proceed(self) -> tuple[float, float]:
        return (self._proceed.alpha, self._proceed.beta)

    @property
    def revise(self) -> tuple[float, float]:
        return (self._revise.alpha, self._revise.beta)

    @property
    def abort(self) -> tuple[float, float]:
        return (self._abort.alpha, self._abort.beta)

    def expected_accuracy(self, verdict: str) -> float:
        """E[X] = alpha / (alpha + beta) for this verdict type."""
        return self._arm(verdict).expected_value()

    def update(self, verdict: str, was_correct: bool) -> None:
        """Update prior after ground truth feedback."""
        self._arm(verdict).update(success=was_correct)

    def to_dict(self) -> dict:
        return {
            "proceed": list(self.proceed),
            "revise": list(self.revise),
            "abort": list(self.abort),
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerdictPrior:
        from hololoom.bandits.beta_arm import BetaArm
        p = d.get("proceed", [1.0, 1.0])
        r = d.get("revise", [1.0, 1.0])
        a = d.get("abort", [1.0, 1.0])
        vp = cls()
        vp._proceed = BetaArm(alpha=p[0], beta=p[1])
        vp._revise  = BetaArm(alpha=r[0], beta=r[1])
        vp._abort   = BetaArm(alpha=a[0], beta=a[1])
        return vp

    def save(self, path) -> None:
        """Persist verdict priors to JSON (atomic write)."""
        import json as _json
        from pathlib import Path as _Path
        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    def load(self, path) -> None:
        """Load verdict priors from JSON. Missing/corrupt = fresh start."""
        import json as _json
        from pathlib import Path as _Path
        path = _Path(path)
        if not path.exists():
            return
        try:
            d = _json.loads(path.read_text())
            from hololoom.bandits.beta_arm import BetaArm
            if "proceed" in d:
                self._proceed = BetaArm(alpha=d["proceed"][0], beta=d["proceed"][1])
            if "revise" in d:
                self._revise = BetaArm(alpha=d["revise"][0], beta=d["revise"][1])
            if "abort" in d:
                self._abort = BetaArm(alpha=d["abort"][0], beta=d["abort"][1])
        except Exception:
            pass  # Corrupt file = fresh start


# ── Gate ───────────────────────────────────────────────────────

class MilestoneGate:
    """
    Reviews agent work at checkpoints.

    Score threshold matches heartbeat.py's require_approval pattern:
    - Tier 1 (FULL/PARTIAL): auto-approve if confidence > 0.9
    - Tier 2 (DEGRADED): auto-approve if confidence > 0.7
    - Below defer_below (0.5): deferred for retroactive 108B review
    """

    def __init__(
        self,
        config: DeepThinkingConfig,
        client: DeepClient,
        tier_state: TierState,
        deferred_queue: DeferredQueue | None = None,
    ):
        self.config = config
        self.client = client
        self.tier = tier_state
        self.queue = deferred_queue
        self.priors = VerdictPrior()
        self._review_count = 0

        # Wire tier recovery → deferred drain
        if self.queue is not None:
            self.tier.on_transition(self._on_tier_recovery)

    async def review(self, ctx: MilestoneContext) -> GateVerdict:
        """
        Review a milestone. Returns verdict with confidence.

        In DARK tier, returns a deferred verdict immediately.
        """
        t0 = time.time()
        tier = self.tier.tier

        # DARK: can't review at all — defer everything
        if tier == Tier.DARK:
            verdict = GateVerdict(
                verdict="proceed",
                confidence=0.0,
                corrections=None,
                reasoning="No inference available — deferred for retroactive review",
                deferred=True,
                elapsed_s=0.0,
            )
            await self._enqueue_deferred(ctx, verdict)
            return verdict

        # Build review prompt
        endpoint = self.tier.best_available_endpoint()
        model = self.tier.model_for_tier

        prompt = self._build_review_prompt(ctx)
        messages = [
            {"role": "system", "content": GATE_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self.client.chat(
                endpoint, model, messages,
                temperature=0.3,  # gate should be deterministic
                timeout_s=self.config.inference_timeout_s,
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Gate review failed ({e}), deferring")
            verdict = GateVerdict(
                verdict="proceed",
                confidence=0.0,
                corrections=None,
                reasoning=f"Review failed: {e} — deferred",
                deferred=True,
                elapsed_s=time.time() - t0,
            )
            await self._enqueue_deferred(ctx, verdict)
            return verdict

        verdict = self._parse_verdict(result.content, model, time.time() - t0)

        # Apply tier-based deferral logic
        if verdict.confidence < self.config.defer_below and tier <= Tier.DEGRADED:
            verdict.deferred = True
            logger.info(
                f"Gate: confidence {verdict.confidence:.2f} < defer threshold "
                f"{self.config.defer_below} — deferring for 108B review"
            )
            await self._enqueue_deferred(ctx, verdict)

        self._review_count += 1
        logger.info(
            f"Gate #{self._review_count}: {verdict.verdict} "
            f"(conf={verdict.confidence:.2f}, model={model}, "
            f"deferred={verdict.deferred}, {verdict.elapsed_s:.1f}s)"
        )

        return verdict

    def update_priors(self, verdict: str, was_correct: bool) -> None:
        """Feed ground truth back to Thompson priors."""
        self.priors.update(verdict, was_correct)
        accuracy = self.priors.expected_accuracy(verdict)
        logger.debug(
            f"Gate prior updated: {verdict} {'correct' if was_correct else 'wrong'} "
            f"→ expected accuracy {accuracy:.2f}"
        )

    # ── Deferred queue integration (Phase 3) ─────────────────
    # Added 2026-03-12: Complete deferred lifecycle wiring.
    # DARK tier defers → rig comes back → deferred work auto-processes.

    def _on_tier_recovery(self, transition) -> None:
        """
        Callback fired on every tier transition.

        When moving from DARK/DEGRADED → PARTIAL/FULL, kick off a
        background drain of the deferred queue so queued gate reviews
        get retroactively processed by the now-available 108B endpoint.
        """
        recovering = (
            transition.previous <= Tier.DEGRADED
            and transition.current >= Tier.PARTIAL
        )
        if not recovering or self.queue is None or self.queue.empty:
            return

        logger.info(
            f"Tier recovery ({transition.previous.name} → "
            f"{transition.current.name}): draining {self.queue.size} "
            f"deferred jobs"
        )
        asyncio.create_task(self._drain_deferred())

    async def _drain_deferred(self) -> None:
        """Drain the deferred queue using the now-available endpoint."""
        processed = await self.queue.drain(self._resubmit_gate_review)
        logger.info(f"Tier recovery drain complete: {processed} jobs processed")

    async def _resubmit_gate_review(self, job: DeferredJob) -> bool:
        """
        Re-submit a single deferred gate review to the live endpoint.

        Reconstructs MilestoneContext from the job payload, runs a fresh
        review (which will now hit the 108B model), and returns True on
        success.
        """
        payload = job.payload
        ctx = MilestoneContext(
            goal=payload.get("goal", ""),
            work_so_far=payload.get("work_so_far", {}),
            milestone_description=payload.get("milestone_description", ""),
            milestone_number=payload.get("milestone_number", 0),
            total_milestones=payload.get("total_milestones", 0),
            agent_id=payload.get("agent_id"),
        )
        try:
            verdict = await self.review(ctx)
            logger.info(
                f"Deferred gate review resubmitted (job={job.id}): "
                f"{verdict.verdict} conf={verdict.confidence:.2f} "
                f"deferred={verdict.deferred}"
            )
            return not verdict.deferred  # success if it wasn't deferred again
        except Exception as e:
            logger.error(f"Deferred gate review failed (job={job.id}): {e}")
            return False

    async def _enqueue_deferred(
        self, ctx: MilestoneContext, verdict: GateVerdict
    ) -> None:
        """Push a deferred gate review onto the queue (if queue wired)."""
        if self.queue is None:
            return
        job = DeferredJob(
            job_type="gate_review",
            payload={
                "goal": ctx.goal,
                "work_so_far": ctx.work_so_far,
                "milestone_description": ctx.milestone_description,
                "milestone_number": ctx.milestone_number,
                "total_milestones": ctx.total_milestones,
                "agent_id": ctx.agent_id,
                "original_verdict": verdict.verdict,
                "original_confidence": verdict.confidence,
                "original_reasoning": verdict.reasoning,
            },
        )
        await self.queue.push(job)

    # ── Prompt building ───────────────────────────────────────

    def _build_review_prompt(self, ctx: MilestoneContext) -> str:
        return (
            f"## Goal\n{ctx.goal}\n\n"
            f"## Milestone {ctx.milestone_number}/{ctx.total_milestones}\n"
            f"{ctx.milestone_description}\n\n"
            f"## Work So Far\n```json\n{json.dumps(ctx.work_so_far, indent=2)}\n```\n\n"
            f"Evaluate this milestone. Output JSON with verdict, confidence, "
            f"reasoning, and corrections (if revise)."
        )

    # ── Response parsing ──────────────────────────────────────

    def _parse_verdict(
        self, content: str, model: str, elapsed_s: float
    ) -> GateVerdict:
        """Parse model output into GateVerdict."""
        try:
            data = _extract_json(content)
            verdict = data.get("verdict", "proceed").lower()
            if verdict not in ("proceed", "revise", "abort"):
                verdict = "proceed"

            return GateVerdict(
                verdict=verdict,
                confidence=float(data.get("confidence", 0.5)),
                corrections=data.get("corrections"),
                reasoning=data.get("reasoning", content),
                model_used=model,
                elapsed_s=elapsed_s,
            )
        except Exception:
            # Parse failure — default to cautious proceed
            return GateVerdict(
                verdict="proceed",
                confidence=0.3,
                corrections=None,
                reasoning=f"[parse error] Raw: {content[:500]}",
                model_used=model,
                elapsed_s=elapsed_s,
            )


def _extract_json(text: str) -> dict:
    """Extract JSON from model output."""
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON found", text, 0)
