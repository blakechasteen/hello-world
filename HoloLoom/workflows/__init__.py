"""
HoloLoom Workflows - Agentic Workflow System
=============================================

Visual workflow builder for creating complex agentic workflows without code.

Features:
- JSON/YAML workflow definition
- Visual drag-and-drop builder
- State management with checkpointing
- Parallel execution support
- Error handling and retries
- Conditional branching and loops
- Integration with prompt chains and recursive reasoner
- Workflow templates and marketplace

Quick Start:
    from HoloLoom.workflows import WorkflowExecutor, WorkflowDefinition

    # Load workflow from JSON
    workflow = WorkflowDefinition.from_json(json_str)

    # Execute workflow
    executor = WorkflowExecutor(workflow)
    result = await executor.execute({"query": "What is Thompson Sampling?"})

    # Result contains outputs from all nodes
    print(result.outputs)
    print(result.trace)

Author: HoloLoom Team
Date: November 2025
"""

from HoloLoom.workflows.schema import (
    NodeType,
    WorkflowNode,
    WorkflowDefinition,
    WorkflowResult,
    ExecutionTrace,
)
from HoloLoom.workflows.executor import WorkflowExecutor
from HoloLoom.workflows.state import (
    StateBackend,
    InMemoryState,
    SQLiteState,
    RedisState,
    CheckpointManager,
)
from HoloLoom.workflows.templates import WorkflowTemplates

__all__ = [
    "NodeType",
    "WorkflowNode",
    "WorkflowDefinition",
    "WorkflowResult",
    "ExecutionTrace",
    "WorkflowExecutor",
    "StateBackend",
    "InMemoryState",
    "SQLiteState",
    "RedisState",
    "CheckpointManager",
    "WorkflowTemplates",
]

__version__ = "1.0.0"
