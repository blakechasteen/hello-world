"""
Deferred Queue — Activation-Weighted Priority Queue
=====================================================

When the rig is down, work that needs 108B review gets queued.
When the rig comes back, the queue drains in priority order.

Priority uses the activation field concept:
    - Gate reviews always outrank background work
    - Hot items (recently accessed, active goals) = high priority
    - Stale items get a time-based boost (don't let things rot)

Persists to disk (JSON, same pattern as lite_bus.py snapshots).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Job types (drain order: gate first, synthesis last) ────────

JOB_TYPE_PRIORITY = {
    "gate_review": 100,       # agents may be waiting
    "deliberation": 50,       # strategic planning
    "critique": 40,           # review existing plans
    "synthesis": 10,          # consolidation (least urgent)
}


@dataclass
class DeferredJob:
    """A unit of work waiting for the rig."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_type: str = "deliberation"
    payload: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    priority: float = 0.0           # base priority (activation weight)
    staleness_boost: float = 0.0    # accumulated time-based boost

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DeferredJob:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DeferredQueue:
    """
    Persisted priority queue for rig-down periods.

    Usage:
        q = DeferredQueue(config.deferred_path, config.staleness_boost_per_hour)
        q.load()
        await q.push(DeferredJob(job_type="gate_review", payload={...}))
        # ... rig comes back ...
        await q.drain(engine)  # processes in priority order
    """

    def __init__(
        self,
        persist_path: str = "./deep_thinking_deferred.json",
        staleness_boost_per_hour: float = 0.1,
    ):
        self._path = Path(persist_path)
        self._staleness_rate = staleness_boost_per_hour
        self._jobs: list[DeferredJob] = []

    @property
    def size(self) -> int:
        return len(self._jobs)

    @property
    def empty(self) -> bool:
        return len(self._jobs) == 0

    async def push(self, job: DeferredJob) -> None:
        """Add a job to the queue. Auto-persists."""
        # Set base priority from job type if not already set
        if job.priority == 0.0:
            job.priority = JOB_TYPE_PRIORITY.get(job.job_type, 20)

        self._jobs.append(job)
        self.save()

        logger.info(
            f"Deferred: queued {job.job_type} (id={job.id}, "
            f"priority={job.priority:.1f}, queue_size={self.size})"
        )

    def pop_highest(self) -> DeferredJob | None:
        """Pop the highest effective-priority job."""
        if not self._jobs:
            return None

        self._update_staleness()
        self._jobs.sort(key=lambda j: self.effective_priority(j), reverse=True)

        job = self._jobs.pop(0)
        self.save()
        return job

    def effective_priority(self, job: DeferredJob) -> float:
        """Priority + staleness boost + job-type weight."""
        type_weight = JOB_TYPE_PRIORITY.get(job.job_type, 20)
        return job.priority + job.staleness_boost + type_weight

    def _update_staleness(self) -> None:
        """Boost priority of stale items."""
        now = time.time()
        for job in self._jobs:
            hours_waiting = (now - job.created_at) / 3600
            job.staleness_boost = hours_waiting * self._staleness_rate

    async def drain(self, process_fn) -> int:
        """
        Drain queue through a processing function.

        Args:
            process_fn: async callable(DeferredJob) -> bool (success)

        Returns:
            Number of successfully processed jobs
        """
        if self.empty:
            logger.info("Deferred: queue empty, nothing to drain")
            return 0

        self._update_staleness()
        self._jobs.sort(key=lambda j: self.effective_priority(j), reverse=True)

        processed = 0
        failed = []

        logger.info(f"Deferred: draining {self.size} jobs...")

        while self._jobs:
            job = self._jobs.pop(0)
            try:
                success = await process_fn(job)
                if success:
                    processed += 1
                    logger.info(
                        f"Deferred: processed {job.job_type} (id={job.id})"
                    )
                else:
                    failed.append(job)
            except Exception as e:
                logger.error(f"Deferred: failed {job.id}: {e}")
                failed.append(job)

        # Re-queue failures
        self._jobs = failed
        self.save()

        logger.info(
            f"Deferred: drained {processed} jobs, "
            f"{len(failed)} failed (re-queued)"
        )
        return processed

    # ── Persistence (lite_bus.py snapshot pattern) ─────────────

    def save(self) -> None:
        """Persist queue to disk as JSON."""
        try:
            data = {
                "version": 1,
                "saved_at": datetime.now().isoformat(),
                "jobs": [j.to_dict() for j in self._jobs],
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Deferred: save failed: {e}")

    def load(self) -> None:
        """Load queue from disk."""
        if not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text())
            self._jobs = [DeferredJob.from_dict(j) for j in data.get("jobs", [])]
            logger.info(f"Deferred: loaded {len(self._jobs)} jobs from {self._path}")
        except Exception as e:
            logger.error(f"Deferred: load failed: {e}")
            self._jobs = []

    def summary(self) -> dict:
        """Queue summary for monitoring."""
        by_type: dict[str, int] = {}
        for j in self._jobs:
            by_type[j.job_type] = by_type.get(j.job_type, 0) + 1

        return {
            "total": self.size,
            "by_type": by_type,
            "oldest_hours": (
                (time.time() - min(j.created_at for j in self._jobs)) / 3600
                if self._jobs else 0
            ),
        }
