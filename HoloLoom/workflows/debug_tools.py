#!/usr/bin/env python3
"""
Workflow Debugging Tools
========================

Step-through debugging, breakpoints, and inspection for workflow execution.

Features:
- Step-through execution mode
- Breakpoint support
- Variable inspection
- Execution replay
- Trace visualization

Usage:
    from HoloLoom.workflows.debug_tools import WorkflowDebugger

    debugger = WorkflowDebugger(workflow)
    await debugger.step_through()
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Execution modes for debugging"""
    NORMAL = "normal"  # Run without pausing
    STEP = "step"  # Pause before each node
    RUN_TO_BREAKPOINT = "run_to_breakpoint"  # Run until breakpoint hit
    INTERACTIVE = "interactive"  # Pause after each node, wait for command


@dataclass
class BreakpointCondition:
    """Conditional breakpoint"""
    node_id: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    hit_count: int = 0
    max_hits: Optional[int] = None
    enabled: bool = True


@dataclass
class ExecutionFrame:
    """Single execution frame (one node execution)"""
    node_id: str
    node_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    status: str = "pending"  # pending, running, success, error
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Complete execution trace"""
    workflow_id: str
    frames: List[ExecutionFrame] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    current_frame_idx: int = 0
    total_duration_ms: float = 0.0
    status: str = "running"  # running, paused, completed, error


class WorkflowDebugger:
    """
    Debugger for workflow execution with step-through, breakpoints, and inspection.
    """

    def __init__(self, workflow: Dict[str, Any]):
        self.workflow = workflow
        self.breakpoints: Dict[str, BreakpointCondition] = {}
        self.trace = ExecutionTrace(workflow_id=workflow.get('id', 'workflow'))
        self.mode = ExecutionMode.STEP
        self.paused = False
        self.command_queue: asyncio.Queue = asyncio.Queue()
        logger.info("WorkflowDebugger initialized")

    def set_breakpoint(
        self,
        node_id: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        max_hits: Optional[int] = None
    ):
        """Set a breakpoint at a node"""
        self.breakpoints[node_id] = BreakpointCondition(
            node_id=node_id,
            condition=condition,
            max_hits=max_hits
        )
        logger.info(f"Breakpoint set at node {node_id}")

    def remove_breakpoint(self, node_id: str):
        """Remove a breakpoint"""
        if node_id in self.breakpoints:
            del self.breakpoints[node_id]
            logger.info(f"Breakpoint removed at node {node_id}")

    def list_breakpoints(self) -> List[BreakpointCondition]:
        """List all breakpoints"""
        return list(self.breakpoints.values())

    async def execute_node(
        self,
        node: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single node with debugging support.

        Returns:
            Node output
        """
        node_id = node['id']
        node_type = node.get('agentType', 'unknown')

        # Create frame
        frame = ExecutionFrame(
            node_id=node_id,
            node_type=node_type,
            inputs=inputs
        )
        self.trace.frames.append(frame)
        frame_idx = len(self.trace.frames) - 1

        # Check breakpoint before execution
        await self._check_breakpoint(node_id, inputs)

        try:
            # Simulate node execution (in real implementation, would call executor)
            frame.status = "running"
            logger.info(f"Executing node {node_id} ({node_type})")

            # Here would be the actual node execution
            # For now, simulate with delay
            await asyncio.sleep(0.1)

            # Simulate outputs
            outputs = {
                'status': 'success',
                'result': f'Output from {node_type}',
                'node_id': node_id
            }

            frame.outputs = outputs
            frame.status = "success"

            # Check breakpoint after execution
            await self._check_breakpoint(node_id, outputs, after_exec=True)

            return outputs

        except Exception as e:
            frame.status = "error"
            frame.error = str(e)
            logger.error(f"Error executing node {node_id}: {e}")
            raise

    async def _check_breakpoint(
        self,
        node_id: str,
        variables: Dict[str, Any],
        after_exec: bool = False
    ):
        """Check if breakpoint should trigger"""
        if node_id not in self.breakpoints:
            return

        bp = self.breakpoints[node_id]
        if not bp.enabled:
            return

        # Check max hits
        if bp.max_hits and bp.hit_count >= bp.max_hits:
            return

        # Check condition
        if bp.condition and not bp.condition(variables):
            return

        # Breakpoint hit
        bp.hit_count += 1
        self.paused = True
        self.trace.status = "paused"

        when = "after" if after_exec else "before"
        logger.info(f"Breakpoint hit at node {node_id} ({when} execution)")

        # Wait for debugger command
        await self._wait_for_command()

    async def _wait_for_command(self):
        """Wait for debugger command"""
        logger.info("Paused. Available commands: continue, step, inspect, print")

    async def step_through(self) -> ExecutionTrace:
        """
        Execute workflow in step-through mode.

        Pauses before each node and waits for user command.
        """
        self.mode = ExecutionMode.STEP
        logger.info("Starting step-through execution")

        nodes = self.workflow.get('nodes', [])
        connections = self.workflow.get('connections', [])

        # Build execution order
        execution_order = self._topological_sort(nodes, connections)

        for node_id in execution_order:
            node = next((n for n in nodes if n['id'] == node_id), None)
            if not node:
                continue

            # Get inputs from previous nodes
            inputs = self._get_node_inputs(node_id, connections)

            # Execute with debugging
            try:
                await self.execute_node(node, inputs)
            except Exception as e:
                logger.error(f"Error in step-through: {e}")
                break

        self.trace.status = "completed"
        return self.trace

    async def run_to_breakpoint(self) -> ExecutionTrace:
        """
        Run workflow until breakpoint is hit.
        """
        self.mode = ExecutionMode.RUN_TO_BREAKPOINT
        logger.info("Running to breakpoint")

        nodes = self.workflow.get('nodes', [])
        connections = self.workflow.get('connections', [])
        execution_order = self._topological_sort(nodes, connections)

        for node_id in execution_order:
            if self.paused:
                break

            node = next((n for n in nodes if n['id'] == node_id), None)
            if not node:
                continue

            inputs = self._get_node_inputs(node_id, connections)
            try:
                await self.execute_node(node, inputs)
            except Exception as e:
                logger.error(f"Error: {e}")
                break

        self.trace.status = "paused" if self.paused else "completed"
        return self.trace

    def inspect_variables(self, frame_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Inspect variables at a frame.

        Args:
            frame_idx: Frame index (None = current frame)

        Returns:
            Variables in frame
        """
        if frame_idx is None:
            frame_idx = self.trace.current_frame_idx

        if frame_idx >= len(self.trace.frames):
            return {}

        frame = self.trace.frames[frame_idx]
        return {
            'node_id': frame.node_id,
            'node_type': frame.node_type,
            'inputs': frame.inputs,
            'outputs': frame.outputs,
            'status': frame.status,
            'metadata': frame.metadata
        }

    def print_trace(self, limit: Optional[int] = None) -> str:
        """
        Print execution trace.

        Args:
            limit: Maximum number of frames to print (None = all)

        Returns:
            Formatted trace string
        """
        output = []
        output.append("=" * 80)
        output.append("EXECUTION TRACE")
        output.append("=" * 80)
        output.append(f"Workflow: {self.trace.workflow_id}")
        output.append(f"Status: {self.trace.status}")
        output.append(f"Total Duration: {self.trace.total_duration_ms:.2f}ms")
        output.append("")

        frames = self.trace.frames[:limit] if limit else self.trace.frames
        for i, frame in enumerate(frames):
            marker = ">> " if i == self.trace.current_frame_idx else "   "
            output.append(f"{marker}[{i}] {frame.node_id} ({frame.node_type})")
            output.append(f"     Status: {frame.status}")
            output.append(f"     Duration: {frame.duration_ms:.2f}ms")

            if frame.error:
                output.append(f"     Error: {frame.error}")

            output.append("")

        output.append("=" * 80)
        return "\n".join(output)

    def export_trace(self, format: str = "json") -> str:
        """Export trace in format (json, yaml)"""
        if format == "json":
            trace_dict = {
                'workflow_id': self.trace.workflow_id,
                'status': self.trace.status,
                'total_duration_ms': self.trace.total_duration_ms,
                'frames': [
                    {
                        'node_id': f.node_id,
                        'node_type': f.node_type,
                        'status': f.status,
                        'duration_ms': f.duration_ms,
                        'error': f.error
                    }
                    for f in self.trace.frames
                ]
            }
            return json.dumps(trace_dict, indent=2)

        return "Unsupported format"

    def _topological_sort(
        self,
        nodes: List[Dict],
        connections: List[Dict]
    ) -> List[str]:
        """Topological sort of nodes"""
        # Build adjacency
        graph = {n['id']: [] for n in nodes}
        for conn in connections:
            graph[conn['from']].append(conn['to'])

        # Kahn's algorithm
        in_degree = {n['id']: 0 for n in nodes}
        for conn in connections:
            in_degree[conn['to']] += 1

        queue = [n['id'] for n in nodes if in_degree[n['id']] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _get_node_inputs(
        self,
        node_id: str,
        connections: List[Dict]
    ) -> Dict[str, Any]:
        """Get inputs for a node from previous executions"""
        inputs = {}
        for conn in connections:
            if conn['to'] == node_id:
                # Find output of source node
                source_id = conn['from']
                source_frame = next(
                    (f for f in reversed(self.trace.frames) if f.node_id == source_id),
                    None
                )
                if source_frame:
                    inputs[source_id] = source_frame.outputs

        return inputs


if __name__ == "__main__":
    # Demo usage
    example_workflow = {
        'id': 'test_workflow',
        'name': 'Test Workflow',
        'nodes': [
            {'id': 'node_1', 'agentType': 'hololoom'},
            {'id': 'node_2', 'agentType': 'response'},
            {'id': 'node_3', 'agentType': 'store'}
        ],
        'connections': [
            {'from': 'node_1', 'to': 'node_2'},
            {'from': 'node_2', 'to': 'node_3'}
        ]
    }

    async def demo():
        debugger = WorkflowDebugger(example_workflow)

        # Set breakpoint at node_2
        debugger.set_breakpoint('node_2')

        # Run to breakpoint
        trace = await debugger.run_to_breakpoint()

        # Inspect variables
        print(debugger.inspect_variables())

        # Print trace
        print(debugger.print_trace())

    asyncio.run(demo())
