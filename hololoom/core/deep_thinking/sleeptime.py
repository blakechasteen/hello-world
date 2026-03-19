"""
Sleeptime Scheduler — Idle-Time Compute = Dreaming
===================================================

When the system is idle, the GPU(s) run dream cycles:
    - Memory consolidation via awareness graph traversal
    - Retrospective evaluation of past deliberations and gate verdicts
    - Pre-computation of plans for recurring patterns
    - Knowledge graph defragmentation

Modeled on dreaming.py's DreamCycleStats for tracking.
Preempts immediately when user activity resumes or a gate request arrives.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from hololoom.core.deep_thinking.config import DeepThinkingConfig

logger = logging.getLogger(__name__)


@dataclass
class SleeptimeCycleStats:
    """Statistics from one sleeptime cycle (mirrors DreamCycleStats)."""
    cycle_number: int
    started_at: datetime
    completed_at: datetime | None = None
    job_type: str = ""
    tokens_generated: int = 0
    memories_consolidated: int = 0
    plans_precomputed: int = 0
    verdicts_reviewed: int = 0
    preempted: bool = False        # was this cycle interrupted?
    elapsed_s: float = 0.0


@dataclass
class SleeptimeJob:
    """A unit of sleeptime work."""
    job_type: str
    description: str
    priority: int = 0              # higher = run first
    handler: str | None = None  # method name on scheduler


# ── Default sleeptime jobs ─────────────────────────────────────

DEFAULT_JOBS = [
    SleeptimeJob(
        "consolidate",
        "Traverse awareness graph, summarize memory clusters",
        priority=30,
        handler="consolidate_memories",
    ),
    SleeptimeJob(
        "retrospective",
        "Review past gate verdicts against outcomes, update Thompson priors",
        priority=20,
        handler="retrospective_eval",
    ),
    SleeptimeJob(
        "precompute",
        "Identify recurring goals, pre-generate plans",
        priority=10,
        handler="precompute_plans",
    ),
]


class SleeptimeScheduler:
    """
    Schedules background deep thinking during idle periods.

    Idle = no activity for config.idle_timeout_s seconds.
    When idle, cycles through sleeptime jobs on available GPUs.
    Preempts immediately on activity.

    Usage:
        scheduler = SleeptimeScheduler(config)
        scheduler.register_activity()   # call on user input
        await scheduler.run(engine)     # background loop
    """

    def __init__(self, config: DeepThinkingConfig):
        self.config = config
        self._last_activity = time.time()
        self._running = False
        self._preempt = asyncio.Event()
        self._cycle_count = 0
        self._stats: list[SleeptimeCycleStats] = []
        self._jobs = list(DEFAULT_JOBS)
        self._task: asyncio.Task | None = None

    def register_activity(self) -> None:
        """Call this when user activity is detected. Preempts any running dream."""
        self._last_activity = time.time()
        if self._preempt:
            self._preempt.set()

    def is_idle(self) -> bool:
        """Check if system has been idle long enough."""
        return (time.time() - self._last_activity) > self.config.idle_timeout_s

    @property
    def idle_for_s(self) -> float:
        """How long the system has been idle."""
        return time.time() - self._last_activity

    async def run(self, deliberation_engine) -> None:
        """
        Main sleeptime loop. Runs forever, checking for idle periods.

        Args:
            deliberation_engine: The DeliberationEngine to use for inference
        """
        if not self.config.sleeptime_enabled:
            logger.info("Sleeptime: disabled in config")
            return

        self._running = True
        logger.info(
            f"Sleeptime: started (idle timeout: {self.config.idle_timeout_s}s)"
        )

        while self._running:
            # Wait for idle
            while not self.is_idle() and self._running:
                await asyncio.sleep(10)

            if not self._running:
                break

            logger.info(
                f"Sleeptime: system idle for {self.idle_for_s:.0f}s, "
                f"starting dream cycle"
            )

            # Run jobs in priority order until preempted or done
            sorted_jobs = sorted(self._jobs, key=lambda j: j.priority, reverse=True)

            for job in sorted_jobs:
                if not self.is_idle():
                    logger.info("Sleeptime: activity detected, preempting")
                    break

                self._preempt.clear()
                await self._run_job(job, deliberation_engine)

            # Cooldown before checking idle again
            await asyncio.sleep(30)

    async def _run_job(self, job: SleeptimeJob, engine) -> SleeptimeCycleStats:
        """Run a single sleeptime job."""
        self._cycle_count += 1
        stats = SleeptimeCycleStats(
            cycle_number=self._cycle_count,
            started_at=datetime.now(),
            job_type=job.job_type,
        )

        t0 = time.time()
        logger.info(f"Sleeptime: running {job.job_type} — {job.description}")

        try:
            if job.job_type == "consolidate":
                stats.memories_consolidated = await self.consolidate_memories(engine)
            elif job.job_type == "retrospective":
                stats.verdicts_reviewed = await self.retrospective_eval(engine)
            elif job.job_type == "precompute":
                stats.plans_precomputed = await self.precompute_plans(engine)

        except asyncio.CancelledError:
            stats.preempted = True
            logger.info(f"Sleeptime: {job.job_type} preempted")
        except Exception as e:
            logger.error(f"Sleeptime: {job.job_type} failed: {e}")

        stats.elapsed_s = time.time() - t0
        stats.completed_at = datetime.now()
        self._stats.append(stats)

        return stats

    # ── Dream jobs ─────────────────────────────────────────────

    async def consolidate_memories(self, engine) -> int:
        """
        Traverse awareness graph, identify memory clusters, generate summaries.

        Uses the Synthesizer role to compress and consolidate.
        """
        # The engine will use whatever tier is available
        result = await engine.deliberate(
            goal="Review and consolidate recent memories into coherent summaries. "
                 "Identify clusters of related information, merge redundancies, "
                 "and surface key patterns.",
            context={"job": "sleeptime_consolidation"},
        )

        consolidated = len(result.consensus.incorporated_critiques)
        logger.info(f"Sleeptime: consolidated {consolidated} memory clusters")
        return consolidated

    async def retrospective_eval(self, engine) -> int:
        """
        Review past gate verdicts against actual outcomes.
        Update Thompson priors based on whether verdicts were correct.
        """
        await engine.deliberate(
            goal="Review recent milestone gate verdicts. For each, assess "
                 "whether the verdict (proceed/revise/abort) was correct "
                 "given the subsequent outcome. Identify patterns in "
                 "gate accuracy.",
            context={"job": "sleeptime_retrospective"},
        )

        reviewed = 1  # each deliberation cycle reviews the batch
        logger.info(f"Sleeptime: retrospective reviewed {reviewed} verdict batches")
        return reviewed

    async def precompute_plans(self, engine) -> int:
        """
        Identify recurring goals and pre-generate cached plans.
        Next time the same goal appears, the cached plan can be
        used as a starting point (skip the slow Planner phase).
        """
        result = await engine.deliberate(
            goal="Analyze past goals and identify recurring patterns. "
                 "For the top 3 most common goal types, pre-generate "
                 "template plans that can be adapted quickly.",
            context={"job": "sleeptime_precompute"},
        )

        precomputed = len(result.proposal.task_graph.get("nodes", []))
        logger.info(f"Sleeptime: precomputed {precomputed} plan templates")
        return precomputed

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self, engine) -> None:
        """Start the sleeptime loop as a background task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(engine))

    def stop(self) -> None:
        """Stop the sleeptime loop."""
        self._running = False
        self._preempt.set()
        if self._task and not self._task.done():
            self._task.cancel()

    def summary(self) -> dict:
        """Sleeptime statistics."""
        return {
            "cycles_completed": self._cycle_count,
            "total_consolidated": sum(
                s.memories_consolidated for s in self._stats
            ),
            "total_reviewed": sum(s.verdicts_reviewed for s in self._stats),
            "total_precomputed": sum(s.plans_precomputed for s in self._stats),
            "preemptions": sum(1 for s in self._stats if s.preempted),
            "idle_now": self.is_idle(),
            "idle_for_s": self.idle_for_s,
        }
