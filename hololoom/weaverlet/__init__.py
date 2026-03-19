"""
Weaverlet — HoloLoom Iterative Worker
======================================

Replaces DomainWorker with an iterative Plan->Act->Evaluate->Revise loop.

Quick start:

    from hololoom.weaverlet import TaskController, TaskLease, Budget, RefinementLoop
    from hololoom.weaverlet.shuttle.http_client import DaemonInferClient
    from hololoom.weaverlet.tools.registry import WeaverletToolRegistry
    from hololoom.weaverlet.evaluation.evaluator import WeaverletEvaluator

    async with DaemonInferClient("http://localhost:4243") as client:
        loop = RefinementLoop(client, WeaverletToolRegistry())
        evaluator = WeaverletEvaluator(infer_client=client)
        controller = TaskController(loop, evaluator)

        result = await controller.execute(
            TaskLease.create("Implement hello world", Budget(max_iterations=5))
        )
        print(result.termination_reason, result.final_score)

Single-pass mode (replaces DomainWorker):

    result = await controller.execute(
        TaskLease.create("Single task", Budget(max_iterations=1))
    )
"""

from hololoom.weaverlet.controller import TaskController
from hololoom.weaverlet.loop import RefinementLoop
from hololoom.weaverlet.schemas import (
    Budget,
    BudgetExhaustedError,
    BudgetLedger,
    IterationOutput,
    MilestoneDelta,
    TaskLease,
    TaskResult,
    TerminationReason,
    WorkingState,
)

__all__ = [
    "Budget",
    "BudgetExhaustedError",
    "BudgetLedger",
    "IterationOutput",
    "MilestoneDelta",
    "RefinementLoop",
    "TaskController",
    "TaskLease",
    "TaskResult",
    "TerminationReason",
    "WorkingState",
]

__version__ = "0.1.0"
