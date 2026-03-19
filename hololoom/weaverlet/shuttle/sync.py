"""
Shuttle Sync
=============

Comms layer between Weaverlet and HoloLoom daemon.

Handles:
- Milestone emission (fire-and-forget)
- Working state deltas
- Final artifact transmission
- Context recall from daemon memory
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hololoom.weaverlet.schemas import MilestoneDelta, TaskResult, WorkingState
from hololoom.weaverlet.shuttle.http_client import DaemonInferClient

logger = logging.getLogger(__name__)


class ShuttleSync:

    def __init__(
        self,
        http_client: DaemonInferClient,
    ):
        self.client = http_client

    async def send_milestone(
        self, delta: MilestoneDelta, state: WorkingState
    ) -> None:
        """Store milestone as a memory item on the daemon. Fire-and-forget."""
        content = json.dumps({
            "type": "weaverlet_milestone",
            "milestone_type": delta.milestone_type,
            "score_before": delta.score_before,
            "score_after": delta.score_after,
            "iteration": delta.iteration,
            "goal": state.goal,
            "timestamp": delta.timestamp.isoformat(),
        })
        await self.client.store_memory(
            content=content,
            tags=["weaverlet", "milestone", delta.milestone_type],
        )
        logger.info(
            f"Milestone emitted: {delta.milestone_type} "
            f"(score {delta.score_before:.2f} -> {delta.score_after:.2f})"
        )

    async def send_delta(self, state: WorkingState) -> None:
        """Push incremental state update to daemon."""
        content = json.dumps({
            "type": "weaverlet_delta",
            "iteration": state.iteration,
            "score": state.latest_score,
            "goal": state.goal,
            "open_questions": state.open_questions[:5],
        })
        await self.client.store_memory(
            content=content,
            tags=["weaverlet", "delta"],
        )

    async def send_final(self, result: TaskResult) -> None:
        """Push final result to daemon memory."""
        content = json.dumps({
            "type": "weaverlet_result",
            "lease_id": result.lease_id,
            "final_score": result.final_score,
            "iterations_completed": result.iterations_completed,
            "termination_reason": result.termination_reason.value,
            "wall_time_seconds": result.wall_time_seconds,
            "total_tokens": result.total_tokens,
            "milestone_count": len(result.milestones),
        })
        await self.client.store_memory(
            content=content,
            tags=["weaverlet", "result", result.termination_reason.value],
        )
        logger.info(
            f"Final result emitted: {result.termination_reason.value} "
            f"(score={result.final_score:.2f}, iters={result.iterations_completed})"
        )

    async def recall_context(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Query daemon memory for relevant prior context."""
        return await self.client.recall_memory(query, limit=limit)
