"""
Workflow Schema Definitions
============================

Defines the JSON/YAML schema for agentic workflows.

Core Concepts:
- **WorkflowNode**: Single node in workflow (query, process, decision, etc.)
- **WorkflowDefinition**: Complete workflow with nodes and connections
- **NodeType**: Enum of supported node types (18+ types)
- **ExecutionTrace**: Complete trace of workflow execution with timing

Author: HoloLoom Team
Date: November 2025
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import yaml


class NodeType(Enum):
    """Workflow node types."""
    # Query Agents
    QUERY = "query"  # RAG query
    SEARCH = "search"  # Memory search
    MULTIQUERY = "multi_query"  # Break into sub-queries

    # Processing Agents
    VERIFY = "verify"  # Verification
    REFINE = "refine"  # Refinement
    SYNTHESIZE = "synthesize"  # Entity/motif extraction
    EMBED = "embed"  # Generate embeddings

    # Memory Agents
    STORE = "store"  # Store in memory
    RETRIEVE = "retrieve"  # Retrieve context
    FUSION = "fusion"  # Multi-hop traversal

    # Decision Agents
    THOMPSON = "thompson"  # Thompson Sampling
    CONVERGENCE = "convergence"  # Decision collapse
    SAFETY = "safety"  # Safety guardrails

    # Chain Agents
    CHAIN = "chain"  # Execute prompt chain
    RECURSIVE = "recursive"  # Recursive reasoning

    # Control Flow
    CONDITION = "condition"  # If/else branch
    LOOP = "loop"  # While loop
    PARALLEL = "parallel"  # Parallel execution
    MERGE = "merge"  # Merge parallel results

    # Output Agents
    RESPONSE = "response"  # Generate response
    FORMAT = "format"  # Format output

    # External Tools
    HUMAN_IN_LOOP = "human_in_loop"  # Wait for human approval
    TOOL = "tool"  # External tool call
    API = "api"  # HTTP API call


@dataclass
class RetryPolicy:
    """Retry policy for node execution."""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    retry_on_errors: list[str] = field(default_factory=lambda: ["timeout", "connection_error"])


@dataclass
class WorkflowNode:
    """
    Single node in workflow.

    Attributes:
        id: Unique node identifier
        type: Node type (NodeType enum)
        name: Human-readable name
        params: Node-specific parameters
        next: List of next node IDs (for sequential flow)
        on_error: Error handler node ID (optional)
        retry_policy: Retry configuration (optional)
        timeout_seconds: Execution timeout (optional)
        description: Node description for documentation
    """
    id: str
    type: NodeType
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    next: list[str] | None = None
    on_error: str | None = None
    retry_policy: RetryPolicy | None = None
    timeout_seconds: int | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "params": self.params,
        }

        if self.next:
            data["next"] = self.next
        if self.on_error:
            data["on_error"] = self.on_error
        if self.retry_policy:
            data["retry_policy"] = asdict(self.retry_policy)
        if self.timeout_seconds:
            data["timeout_seconds"] = self.timeout_seconds
        if self.description:
            data["description"] = self.description

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorkflowNode':
        """Create from dictionary."""
        node_type = NodeType(data["type"])

        retry_policy = None
        if "retry_policy" in data:
            retry_policy = RetryPolicy(**data["retry_policy"])

        return cls(
            id=data["id"],
            type=node_type,
            name=data["name"],
            params=data.get("params", {}),
            next=data.get("next"),
            on_error=data.get("on_error"),
            retry_policy=retry_policy,
            timeout_seconds=data.get("timeout_seconds"),
            description=data.get("description", ""),
        )


@dataclass
class WorkflowDefinition:
    """
    Complete workflow definition.

    Attributes:
        name: Workflow name
        version: Version string (e.g., "1.0.0")
        nodes: Dictionary of node_id -> WorkflowNode
        entry_point: ID of entry node
        metadata: Additional workflow metadata
    """
    name: str
    version: str
    nodes: dict[str, WorkflowNode]
    entry_point: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'WorkflowDefinition':
        """Load workflow from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'WorkflowDefinition':
        """Load workflow from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorkflowDefinition':
        """Create from dictionary."""
        nodes = {
            node_id: WorkflowNode.from_dict(node_data)
            for node_id, node_data in data["nodes"].items()
        }

        return cls(
            name=data["name"],
            version=data["version"],
            nodes=nodes,
            entry_point=data["entry_point"],
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        data = self.to_dict()
        return json.dumps(data, indent=indent)

    def to_yaml(self) -> str:
        """Export to YAML string."""
        data = self.to_dict()
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            },
            "entry_point": self.entry_point,
            "metadata": self.metadata,
        }

    def validate(self) -> list[str]:
        """
        Validate workflow structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check entry point exists
        if self.entry_point not in self.nodes:
            errors.append(f"Entry point '{self.entry_point}' not found in nodes")

        # Check all node references exist
        for node_id, node in self.nodes.items():
            if node.next:
                for next_id in node.next:
                    if next_id not in self.nodes:
                        errors.append(f"Node '{node_id}' references non-existent node '{next_id}'")

            if node.on_error and node.on_error not in self.nodes:
                errors.append(f"Node '{node_id}' references non-existent error handler '{node.on_error}'")

        # Check for cycles (DFS)
        if self._has_cycles():
            errors.append("Workflow contains cycles (infinite loops)")

        # Check for unreachable nodes
        reachable = self._find_reachable_nodes()
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            errors.append(f"Unreachable nodes: {sorted(unreachable)}")

        return errors

    def _has_cycles(self) -> bool:
        """Check for cycles using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            node = self.nodes[node_id]
            if node.next:
                for next_id in node.next:
                    if next_id not in visited:
                        if dfs(next_id):
                            return True
                    elif next_id in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def _find_reachable_nodes(self) -> set:
        """Find all reachable nodes from entry point."""
        reachable = set()
        queue = [self.entry_point]

        while queue:
            node_id = queue.pop(0)
            if node_id in reachable:
                continue

            reachable.add(node_id)
            node = self.nodes[node_id]

            if node.next:
                queue.extend(node.next)
            if node.on_error:
                queue.append(node.on_error)

        return reachable


@dataclass
class ExecutionTrace:
    """
    Trace of single node execution.

    Attributes:
        node_id: Node identifier
        start_time: Execution start timestamp
        end_time: Execution end timestamp
        duration_ms: Execution duration in milliseconds
        status: Execution status ("success", "error", "timeout", "skipped")
        inputs: Input data to node
        outputs: Output data from node
        error: Error message (if failed)
        retry_count: Number of retries attempted
    """
    node_id: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    error: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkflowResult:
    """
    Complete workflow execution result.

    Attributes:
        execution_id: Unique execution identifier
        workflow_name: Name of executed workflow
        status: Overall status ("success", "failed", "partial")
        outputs: Final outputs from workflow
        execution_time_ms: Total execution time
        nodes_executed: Number of nodes executed
        errors: List of errors encountered
        trace: List of execution traces for each node
        checkpoint_id: Checkpoint ID (if saved)
    """
    execution_id: str
    workflow_name: str
    status: str
    outputs: dict[str, Any]
    execution_time_ms: float
    nodes_executed: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    trace: list[ExecutionTrace] = field(default_factory=list)
    checkpoint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "outputs": self.outputs,
            "execution_time_ms": self.execution_time_ms,
            "nodes_executed": self.nodes_executed,
            "errors": self.errors,
            "trace": [t.to_dict() for t in self.trace],
            "checkpoint_id": self.checkpoint_id,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Execution ID: {self.execution_id}",
            f"Workflow: {self.workflow_name}",
            f"Status: {self.status}",
            f"Nodes Executed: {self.nodes_executed}",
            f"Total Time: {self.execution_time_ms:.1f}ms",
        ]

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for error in self.errors:
                lines.append(f"  - {error.get('node_id')}: {error.get('error')}")

        return "\n".join(lines)
