#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Workflow Executor
===========================

Backend server for executing visual workflows created in the workflow builder.

Features:
- Execute multi-agent workflows
- Real-time execution status via WebSocket
- Workflow validation and optimization
- Integration with all HoloLoom agents

Usage:
    python workflow_executor.py

    Then open workflow_builder.html in browser
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn websockets")
    raise

try:
    from HoloLoom.unified_api import HoloLoom
    from HoloLoom.config import Config
    from HoloLoom.alignment.safety_guardrails import SafetyGuardrails, RiskLevel
    from HoloLoom.recursive.full_learning_loop import FullLearningEngine
except ImportError as e:
    print(f"HoloLoom imports failed: {e}")
    print("Make sure PYTHONPATH is set to repository root")
    raise

# Import LLM agent executor
try:
    from HoloLoom.web_dashboard.llm_executor import execute_llm_agent
    LLM_AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLM agents not available: {e}")
    logger.warning("LLM agents will not work. Install with: pip install openai anthropic")
    LLM_AGENTS_AVAILABLE = False

    # Create fallback function
    async def execute_llm_agent(agent_type, config, inputs):
        return {
            'error': 'LLM agents not available',
            'message': 'Install dependencies: pip install openai anthropic',
            'agent_type': agent_type
        }

# Import optimization engine
try:
    from HoloLoom.web_dashboard.optimization_engine import ThompsonOptimizer
    optimizer = ThompsonOptimizer()
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimization engine not available: {e}")
    OPTIMIZATION_AVAILABLE = False
    optimizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HoloLoom Workflow Executor")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class WorkflowNode(BaseModel):
    id: str
    agentType: str
    x: float
    y: float
    config: Dict[str, Any]

class WorkflowConnection(BaseModel):
    id: str
    from_node: str = None  # Field name aliasing
    to: str

    class Config:
        fields = {'from_node': 'from'}

class Workflow(BaseModel):
    version: str
    name: str
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]

class ExecutionRequest(BaseModel):
    workflow: Workflow
    input_data: Optional[Dict[str, Any]] = None

class SaveVersionRequest(BaseModel):
    workflow: Workflow
    message: str
    description: Optional[str] = None
    branch: str = 'main'
    timestamp: str

class CreateBranchRequest(BaseModel):
    branch_name: str
    from_branch: str = 'main'
    from_version: int = 1

class URLIngestRequest(BaseModel):
    url: str
    options: Dict[str, Any] = {}

class FileIngestResponse(BaseModel):
    job_id: str
    shards_created: int
    filename: str
    content_type: Optional[str] = None
    file_size: int = 0
    warning: Optional[str] = None

class URLIngestResponse(BaseModel):
    job_id: str
    shards_created: int
    url: str
    options_applied: Dict[str, Any] = {}
    warning: Optional[str] = None

class EntityRelationship(BaseModel):
    target: str
    relation_type: str
    weight: float = 1.0

class EntityResponse(BaseModel):
    entity_id: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = {}
    relationships: List[EntityRelationship] = []
    relationship_count: int = 0

# Collaboration Models
class CollaborationSession(BaseModel):
    session_id: str
    workflow_id: str
    participants: List[Dict[str, Any]] = []
    created_at: float

class CursorUpdate(BaseModel):
    user_id: str
    user_name: str
    x: float
    y: float
    color: str = "#3b82f6"

class PresenceUpdate(BaseModel):
    user_id: str
    user_name: str
    status: str  # "active", "idle", "editing"
    selected_node: Optional[str] = None

class NodeLockRequest(BaseModel):
    user_id: str
    user_name: str
    node_id: str

class JoinSessionRequest(BaseModel):
    user_id: str
    user_name: str
    color: Optional[str] = None

# Optimization Models
class OptimizationSuggestion(BaseModel):
    suggestion_id: str
    workflow_id: str
    suggestion_type: str  # "parallelize", "cache", "reorder", "substitute"
    description: str
    expected_improvement: float  # percentage
    confidence: float
    affected_nodes: List[str]

class PerformanceProfile(BaseModel):
    node_id: str
    node_type: str
    avg_latency_ms: float
    p95_latency_ms: float
    success_rate: float
    throughput: float  # calls per minute
    bottleneck_score: float  # 0-1, higher = more likely bottleneck

# Thompson Sampling state for workflow optimization
node_performance_stats = {}  # node_type -> {alpha, beta, latency_samples, success_count, failure_count}
optimization_history = []  # List of optimization suggestions and their outcomes
workflow_pattern_history = []  # List of executed workflow patterns for pattern mining
execution_metrics = {}  # node_type -> {samples: [...], success_count, failure_count}


def update_execution_metrics(node_type: str, latency_ms: float, success: bool, confidence: float = 0.5):
    """
    Update real-time execution metrics for a node type.

    This data feeds into:
    - Performance profiling endpoints
    - Thompson Sampling optimization
    - Pattern miner for suggestions
    """
    global execution_metrics, node_performance_stats

    if node_type not in execution_metrics:
        execution_metrics[node_type] = {
            'samples': [],
            'success_count': 0,
            'failure_count': 0,
            'confidence_sum': 0.0,
            'last_updated': None
        }

    metrics = execution_metrics[node_type]

    # Keep last 100 samples for rolling statistics
    metrics['samples'].append({
        'latency_ms': latency_ms,
        'success': success,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    })
    if len(metrics['samples']) > 100:
        metrics['samples'] = metrics['samples'][-100:]

    # Update counters
    if success:
        metrics['success_count'] += 1
    else:
        metrics['failure_count'] += 1

    metrics['confidence_sum'] += confidence
    metrics['last_updated'] = datetime.now().isoformat()

    # Update Thompson Sampling priors in node_performance_stats
    if node_type not in node_performance_stats:
        node_performance_stats[node_type] = {
            'alpha': 1.0,  # Beta distribution prior
            'beta': 1.0,
            'latency_samples': [],
            'success_count': 0,
            'failure_count': 0
        }

    stats = node_performance_stats[node_type]
    stats['latency_samples'].append(latency_ms)
    if len(stats['latency_samples']) > 100:
        stats['latency_samples'] = stats['latency_samples'][-100:]

    # Thompson Sampling update: success boosts alpha, failure boosts beta
    if success and confidence >= 0.5:
        stats['alpha'] += confidence
        stats['success_count'] += 1
    else:
        stats['beta'] += (1 - confidence)
        stats['failure_count'] += 1


def get_real_performance_stats(node_type: str) -> dict:
    """Get computed performance statistics for a node type."""
    if node_type not in execution_metrics:
        return {
            'avg_latency_ms': 100.0,  # Default estimate
            'p95_latency_ms': 200.0,
            'success_rate': 0.9,
            'throughput': 0.0,
            'call_count': 0,
            'confidence_avg': 0.5
        }

    metrics = execution_metrics[node_type]
    samples = metrics['samples']

    if not samples:
        return {
            'avg_latency_ms': 100.0,
            'p95_latency_ms': 200.0,
            'success_rate': 0.9,
            'throughput': 0.0,
            'call_count': 0,
            'confidence_avg': 0.5
        }

    latencies = [s['latency_ms'] for s in samples]
    total_calls = metrics['success_count'] + metrics['failure_count']

    # Calculate p95
    sorted_latencies = sorted(latencies)
    p95_idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)

    return {
        'avg_latency_ms': sum(latencies) / len(latencies),
        'p95_latency_ms': sorted_latencies[p95_idx],
        'success_rate': metrics['success_count'] / max(total_calls, 1),
        'throughput': total_calls / max(1, 60),  # Approximation per minute
        'call_count': total_calls,
        'confidence_avg': metrics['confidence_sum'] / max(total_calls, 1)
    }


def record_workflow_pattern(workflow_name: str, node_sequence: List[str], success: bool, total_latency_ms: float):
    """Record a workflow execution pattern for pattern mining."""
    global workflow_pattern_history

    pattern = {
        'workflow_name': workflow_name,
        'node_sequence': node_sequence,
        'success': success,
        'total_latency_ms': total_latency_ms,
        'timestamp': datetime.now().isoformat()
    }

    workflow_pattern_history.append(pattern)

    # Keep last 500 patterns
    if len(workflow_pattern_history) > 500:
        workflow_pattern_history = workflow_pattern_history[-500:]

# Global state
active_workflows: Dict[str, "WorkflowExecutor"] = {}
ws_connections: List[WebSocket] = []
version_store: Dict[int, Dict[str, Any]] = {}
version_counter = 0
branch_store: Dict[str, Dict[str, Any]] = {
    'main': {'versions': [], 'head': None}
}

# Execution logs storage (Wave 5.6)
execution_logs: List[Dict[str, Any]] = []
MAX_EXECUTION_LOGS = 1000

# WebSocket execution state (Wave 5.6)
active_ws_executions: Dict[str, Dict[str, Any]] = {}  # execution_id -> {executor, status, progress, cancelled}


async def broadcast_log(level: str, message: str, node_id: Optional[str] = None, execution_id: Optional[str] = None):
    """
    Broadcast a log message to all WebSocket clients.

    Args:
        level: Log level (INFO, WARN, ERROR, DEBUG)
        message: Log message text
        node_id: Optional node ID this log relates to
        execution_id: Optional execution ID for filtering
    """
    global execution_logs

    log_entry = {
        'type': 'log',
        'level': level,
        'message': message,
        'node_id': node_id,
        'execution_id': execution_id,
        'timestamp': datetime.now().isoformat()
    }

    # Store in execution logs (ring buffer)
    execution_logs.append(log_entry)
    if len(execution_logs) > MAX_EXECUTION_LOGS:
        execution_logs = execution_logs[-MAX_EXECUTION_LOGS:]

    # Broadcast to all connected clients
    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_json(log_entry)
        except:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in ws_connections:
            ws_connections.remove(ws)


async def broadcast_workflow_status(
    execution_id: str,
    status: str,
    progress: float = 0.0,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None
):
    """
    Broadcast workflow execution status to all WebSocket clients.

    Args:
        execution_id: Unique execution identifier
        status: Status string (started, running, completed, error, cancelled)
        progress: Progress percentage (0.0-100.0)
        error: Optional error message
        result: Optional result data on completion
    """
    message = {
        'type': f'workflow_{status}' if status in ['start', 'complete', 'error'] else 'workflow_status',
        'execution_id': execution_id,
        'status': status,
        'progress': progress,
        'timestamp': datetime.now().isoformat()
    }

    if error:
        message['error'] = error
    if result:
        message['result'] = result

    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in ws_connections:
            ws_connections.remove(ws)


async def broadcast_node_event(
    execution_id: str,
    node_id: str,
    event: str,  # 'start', 'complete', 'error'
    output: Optional[Any] = None,
    error: Optional[str] = None,
    elapsed_ms: Optional[float] = None
):
    """
    Broadcast node execution event to all WebSocket clients.

    Args:
        execution_id: Unique execution identifier
        node_id: Node being executed
        event: Event type (start, complete, error)
        output: Node output on completion
        error: Error message on error
        elapsed_ms: Execution time in milliseconds
    """
    message = {
        'type': f'node_{event}',
        'execution_id': execution_id,
        'node_id': node_id,
        'timestamp': datetime.now().isoformat()
    }

    if output is not None:
        message['output'] = output
    if error:
        message['error'] = error
    if elapsed_ms is not None:
        message['elapsed_ms'] = elapsed_ms

    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in ws_connections:
            ws_connections.remove(ws)


async def execute_workflow_with_streaming(
    executor: "WorkflowExecutor",
    execution_id: str,
    websocket: WebSocket
) -> Dict[str, Any]:
    """
    Execute a workflow with real-time streaming updates via WebSocket.

    Broadcasts node start/complete/error events and progress updates.
    Checks for cancellation between nodes.

    Args:
        executor: WorkflowExecutor instance
        execution_id: Unique execution identifier
        websocket: WebSocket connection for direct responses

    Returns:
        Workflow execution result
    """
    import time

    # Validate workflow first
    try:
        executor.validate_workflow()
    except Exception as e:
        await broadcast_log('ERROR', f'Workflow validation failed: {e}', execution_id=execution_id)
        raise

    # Get execution order
    try:
        execution_order = executor.topological_sort()
    except Exception as e:
        await broadcast_log('ERROR', f'Workflow has cycles: {e}', execution_id=execution_id)
        raise

    total_nodes = len(execution_order)
    node_outputs = {}
    result = {'status': 'completed', 'node_results': {}, 'execution_order': execution_order}

    await broadcast_log('INFO', f'Executing {total_nodes} nodes', execution_id=execution_id)

    for i, node_id in enumerate(execution_order):
        # Check for cancellation
        if active_ws_executions.get(execution_id, {}).get('cancelled'):
            result['status'] = 'cancelled'
            result['cancelled_at'] = node_id
            return result

        node = executor.node_map.get(node_id)
        if not node:
            continue

        # Calculate progress
        progress = ((i) / total_nodes) * 100
        active_ws_executions[execution_id]['progress'] = progress
        await broadcast_workflow_status(execution_id, 'running', progress=progress)

        # Broadcast node start
        await broadcast_node_event(execution_id, node_id, 'start')
        await broadcast_log('INFO', f'Starting node: {node.config.get("label", node_id)}', node_id=node_id, execution_id=execution_id)

        start_time = time.time()

        try:
            # Gather inputs from connected nodes
            inputs = {}
            for conn_id, conn in executor.connection_map.items():
                if conn.to == node_id:
                    from_output = node_outputs.get(conn.from_node)
                    if from_output:
                        inputs[conn.from_node] = from_output

            # Execute the node
            output = await executor.execute_agent(node.agentType, node.config, inputs)
            node_outputs[node_id] = output
            result['node_results'][node_id] = output

            elapsed_ms = (time.time() - start_time) * 1000

            # Broadcast node complete
            await broadcast_node_event(execution_id, node_id, 'complete', output=output, elapsed_ms=elapsed_ms)
            await broadcast_log('INFO', f'Node completed in {elapsed_ms:.0f}ms', node_id=node_id, execution_id=execution_id)

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            # Broadcast node error
            await broadcast_node_event(execution_id, node_id, 'error', error=str(e), elapsed_ms=elapsed_ms)
            await broadcast_log('ERROR', f'Node failed: {e}', node_id=node_id, execution_id=execution_id)

            result['status'] = 'failed'
            result['failed_at'] = node_id
            result['error'] = str(e)
            raise

    # Update final progress
    active_ws_executions[execution_id]['progress'] = 100.0
    await broadcast_workflow_status(execution_id, 'running', progress=100.0)

    return result


# Collaboration state
collaboration_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> {workflow_id, participants, cursor_positions, last_activity}
presence_subscriptions: Dict[WebSocket, str] = {}  # websocket -> session_id


class WorkflowCRDT:
    """
    Simple Conflict-Free Replicated Data Type for workflow state.

    Uses last-writer-wins strategy with vector clocks for conflict resolution.
    This ensures eventual consistency across all collaborators.
    """

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.state = {"nodes": {}, "connections": {}}
        self.vector_clock: Dict[str, float] = {}  # user_id -> timestamp
        self.node_locks: Dict[str, Dict[str, Any]] = {}  # node_id -> {user_id, locked_at}

    def apply_operation(self, user_id: str, operation: dict) -> dict:
        """
        Apply an operation to the workflow state.

        Args:
            user_id: User performing the operation
            operation: {
                'type': 'add_node' | 'update_node' | 'delete_node' | 'add_connection' | 'delete_connection',
                'data': {...},
                'timestamp': float
            }

        Returns:
            Merged state after applying operation
        """
        op_type = operation.get('type')
        data = operation.get('data', {})
        timestamp = operation.get('timestamp', datetime.now().timestamp())

        # Update vector clock
        if user_id not in self.vector_clock or timestamp > self.vector_clock[user_id]:
            self.vector_clock[user_id] = timestamp
        else:
            # Reject stale operation
            logger.warning(f"Rejected stale operation from {user_id}: {op_type}")
            return self.state

        # Apply operation based on type
        if op_type == 'add_node' or op_type == 'update_node':
            node_id = data.get('id')
            if node_id:
                self.state['nodes'][node_id] = {
                    **data,
                    'last_modified_by': user_id,
                    'last_modified_at': timestamp
                }
                logger.info(f"Applied {op_type}: {node_id} by {user_id}")

        elif op_type == 'delete_node':
            node_id = data.get('id')
            if node_id and node_id in self.state['nodes']:
                del self.state['nodes'][node_id]
                # Also remove connections referencing this node
                self.state['connections'] = {
                    conn_id: conn for conn_id, conn in self.state['connections'].items()
                    if conn.get('from') != node_id and conn.get('to') != node_id
                }
                logger.info(f"Applied delete_node: {node_id} by {user_id}")

        elif op_type == 'add_connection':
            conn_id = data.get('id')
            if conn_id:
                self.state['connections'][conn_id] = {
                    **data,
                    'last_modified_by': user_id,
                    'last_modified_at': timestamp
                }
                logger.info(f"Applied add_connection: {conn_id} by {user_id}")

        elif op_type == 'delete_connection':
            conn_id = data.get('id')
            if conn_id and conn_id in self.state['connections']:
                del self.state['connections'][conn_id]
                logger.info(f"Applied delete_connection: {conn_id} by {user_id}")

        return self.state

    def try_lock_node(self, node_id: str, user_id: str) -> bool:
        """
        Try to acquire exclusive lock on a node for editing.

        Returns:
            True if lock acquired, False if already locked by another user
        """
        if node_id in self.node_locks:
            lock = self.node_locks[node_id]
            # Check if lock is stale (>30 seconds)
            if datetime.now().timestamp() - lock['locked_at'] > 30:
                # Release stale lock
                del self.node_locks[node_id]
            elif lock['user_id'] != user_id:
                # Node locked by another user
                return False

        # Acquire lock
        self.node_locks[node_id] = {
            'user_id': user_id,
            'locked_at': datetime.now().timestamp()
        }
        logger.info(f"Lock acquired: node {node_id} by {user_id}")
        return True

    def unlock_node(self, node_id: str, user_id: str) -> bool:
        """
        Release lock on a node.

        Returns:
            True if lock released, False if not locked or locked by another user
        """
        if node_id in self.node_locks:
            lock = self.node_locks[node_id]
            if lock['user_id'] == user_id:
                del self.node_locks[node_id]
                logger.info(f"Lock released: node {node_id} by {user_id}")
                return True
        return False

    def get_lock_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get current lock status for a node."""
        return self.node_locks.get(node_id)


class WorkflowExecutor:
    """
    Executes a workflow graph with topological ordering and dependency resolution.
    """

    def __init__(self, workflow: Workflow, input_data: Optional[Dict] = None):
        self.workflow = workflow
        self.input_data = input_data or {}
        self.results: Dict[str, Any] = {}
        self.executed_nodes: set = set()
        self.hololoom: Optional[HoloLoom] = None
        self.safety_guardrails = SafetyGuardrails(enable_human_in_loop=False)

    async def initialize(self):
        """Initialize HoloLoom instance."""
        self.hololoom = await HoloLoom.create(
            pattern="fast",
            memory_backend="simple",
            enable_synthesis=True
        )
        logger.info("WorkflowExecutor initialized")

    async def close(self):
        """Cleanup resources."""
        if self.hololoom:
            await self.hololoom.close()

    async def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow in topological order.

        Returns:
            Dict with execution results for each node
        """
        import time
        workflow_start_time = time.time()
        node_sequence = []  # Track execution order for pattern mining

        try:
            await self.initialize()

            # Validate workflow
            self.validate_workflow()

            # Find starting nodes (no incoming connections)
            start_nodes = self.find_start_nodes()
            if not start_nodes:
                raise ValueError("No starting nodes found - all nodes have inputs")

            # Execute in topological order
            queue = [self.get_node_by_id(nid) for nid in start_nodes]
            executed_count = 0

            while queue:
                node = queue.pop(0)

                if node.id in self.executed_nodes:
                    continue

                # Check dependencies
                dependencies = self.get_node_dependencies(node.id)
                if dependencies and not all(dep in self.executed_nodes for dep in dependencies):
                    queue.append(node)  # Re-queue
                    continue

                # Execute node
                await self.execute_node(node)
                self.executed_nodes.add(node.id)
                executed_count += 1
                node_sequence.append(node.agentType)  # Track for pattern mining

                # Broadcast progress
                await self.broadcast_progress(node.id, 'completed')

                # Add dependent nodes to queue
                dependent_nodes = self.get_dependent_nodes(node.id)
                queue.extend([self.get_node_by_id(nid) for nid in dependent_nodes])

            logger.info(f"Workflow execution complete: {executed_count} nodes executed")

            # Record workflow pattern for mining
            total_latency_ms = (time.time() - workflow_start_time) * 1000
            record_workflow_pattern(
                workflow_name=self.workflow.name,
                node_sequence=node_sequence,
                success=True,
                total_latency_ms=total_latency_ms
            )

            return {
                'status': 'success',
                'nodes_executed': executed_count,
                'results': self.results,
                'timestamp': datetime.now().isoformat(),
                'total_latency_ms': total_latency_ms
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Record failed workflow pattern
            total_latency_ms = (time.time() - workflow_start_time) * 1000
            record_workflow_pattern(
                workflow_name=self.workflow.name if hasattr(self, 'workflow') else 'unknown',
                node_sequence=node_sequence,
                success=False,
                total_latency_ms=total_latency_ms
            )
            await self.broadcast_error(str(e))
            raise

        finally:
            await self.close()

    def validate_workflow(self):
        """Validate workflow structure."""
        # Check for cycles
        if self.has_cycles():
            raise ValueError("Workflow contains cycles")

        # Validate all nodes have valid agent types
        valid_types = {
            'hololoom', 'search', 'multiquery',
            'embedder', 'synthesizer', 'refiner',
            'store', 'retrieve', 'fusion',
            'thompson', 'convergence', 'safety',
            'response', 'format',
            'conditional', 'loop', 'parallel'
        }

        for node in self.workflow.nodes:
            if node.agentType not in valid_types:
                raise ValueError(f"Invalid agent type: {node.agentType}")

    def has_cycles(self) -> bool:
        """Check for cycles using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)

            for conn in self.workflow.connections:
                if conn.from_node == node_id:
                    if conn.to not in visited:
                        if dfs(conn.to):
                            return True
                    elif conn.to in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node in self.workflow.nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True

        return False

    def find_start_nodes(self) -> List[str]:
        """Find nodes with no incoming connections."""
        all_nodes = {n.id for n in self.workflow.nodes}
        nodes_with_inputs = {c.to for c in self.workflow.connections}
        return list(all_nodes - nodes_with_inputs)

    def get_node_by_id(self, node_id: str) -> WorkflowNode:
        """Get node by ID."""
        for node in self.workflow.nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"Node not found: {node_id}")

    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get IDs of nodes that must execute before this node."""
        return [c.from_node for c in self.workflow.connections if c.to == node_id]

    def get_dependent_nodes(self, node_id: str) -> List[str]:
        """Get IDs of nodes that depend on this node."""
        return [c.to for c in self.workflow.connections if c.from_node == node_id]

    async def execute_node(self, node: WorkflowNode):
        """Execute a single node with performance tracking."""
        logger.info(f"Executing node: {node.id} ({node.agentType})")

        await self.broadcast_progress(node.id, 'running')

        # Gather inputs from dependencies
        inputs = {}
        for dep_id in self.get_node_dependencies(node.id):
            if dep_id in self.results:
                inputs[dep_id] = self.results[dep_id]

        # Track execution time
        import time
        start_time = time.time()

        # Execute based on agent type
        result = await self.execute_agent(node, inputs)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Determine success and confidence
        success = 'error' not in result
        confidence = result.get('confidence', 0.8 if success else 0.2)

        # Update real-time performance metrics
        update_execution_metrics(
            node_type=node.agentType,
            latency_ms=latency_ms,
            success=success,
            confidence=confidence
        )

        # Store result with performance data
        self.results[node.id] = {
            'node_id': node.id,
            'agent_type': node.agentType,
            'timestamp': datetime.now().isoformat(),
            'config': node.config,
            'output': result,
            'performance': {
                'latency_ms': latency_ms,
                'success': success,
                'confidence': confidence
            }
        }

        logger.info(f"Node {node.id} completed in {latency_ms:.2f}ms (success={success})")

    async def execute_agent(self, node: WorkflowNode, inputs: Dict) -> Any:
        """
        Execute specific agent type.

        This is where the actual HoloLoom agents are invoked.
        """
        agent_type = node.agentType
        config = node.config

        try:
            # Query Agents
            if agent_type == 'hololoom':
                query = inputs.get('query', self.input_data.get('query', 'Default query'))
                if isinstance(query, dict) and 'output' in query:
                    query = query['output'].get('response', str(query))

                result = await self.hololoom.query(
                    query,
                    pattern=config.get('pattern', 'fast'),
                    return_trace=config.get('return_trace', True)
                )
                return {'response': result.response, 'confidence': result.confidence}

            elif agent_type == 'search':
                query = inputs.get('query', self.input_data.get('query', ''))
                # Implement memory search
                return {'memories': [], 'count': 0}

            elif agent_type == 'multiquery':
                query = inputs.get('query', self.input_data.get('query', ''))
                max_subqueries = config.get('max_subqueries', 5)
                # Break query into sub-queries
                subqueries = [f"{query} - aspect {i+1}" for i in range(max_subqueries)]
                return {'subqueries': subqueries}

            # Processing Agents
            elif agent_type == 'embedder':
                text = str(inputs.get('text', self.input_data.get('text', '')))
                # Generate embeddings
                return {'embeddings': [0.1, 0.2, 0.3], 'dimensions': 384}

            elif agent_type == 'synthesizer':
                text = str(inputs.get('text', ''))
                # Extract entities and motifs
                return {
                    'entities': ['entity1', 'entity2'],
                    'motifs': ['motif1'],
                    'reasoning_type': 'analytical'
                }

            elif agent_type == 'refiner':
                # Recursive refinement
                strategy = config.get('strategy', 'refine')
                max_iterations = config.get('max_iterations', 3)
                return {'refined': True, 'iterations': max_iterations, 'quality': 0.95}

            # Memory Agents
            elif agent_type == 'store':
                data = inputs.get('data', {})
                backend = config.get('backend', 'inmemory')
                return {'stored': True, 'backend': backend}

            elif agent_type == 'retrieve':
                query = inputs.get('query', '')
                k = config.get('k', 5)
                return {'context': [], 'count': k}

            elif agent_type == 'fusion':
                query = inputs.get('query', '')
                max_depth = config.get('max_depth', 2)
                return {'expanded': [], 'depth': max_depth}

            # Decision Agents
            elif agent_type == 'thompson':
                options = inputs.get('options', [])
                return {'selected': 'option_1', 'confidence': 0.85}

            elif agent_type == 'convergence':
                features = inputs.get('features', {})
                strategy = config.get('strategy', 'epsilon_greedy')
                return {'decision': 'action_1', 'strategy': strategy}

            elif agent_type == 'safety':
                action = inputs.get('action', {})
                risk_threshold = config.get('risk_threshold', 'MEDIUM')

                gate_result = await self.safety_guardrails.gate_action(
                    action=str(action),
                    context={'workflow_node': node.id}
                )

                return {
                    'allowed': gate_result.allowed,
                    'safety_score': gate_result.safety_score,
                    'reason': gate_result.reason
                }

            # Output Agents
            elif agent_type == 'response':
                data = inputs.get('data', {})
                format_type = config.get('format', 'text')
                return {'response': str(data), 'format': format_type}

            elif agent_type == 'format':
                data = inputs.get('data', {})
                output_format = config.get('output_format', 'json')
                if output_format == 'json':
                    return {'formatted': json.dumps(data, indent=2)}
                elif output_format == 'markdown':
                    return {'formatted': f"# Result\n\n{data}"}
                else:
                    return {'formatted': str(data)}

            # Control Flow
            elif agent_type == 'conditional':
                condition = inputs.get('condition', {})
                condition_type = config.get('condition_type', 'confidence')
                threshold = config.get('threshold', 0.75)

                if condition_type == 'confidence':
                    confidence = condition.get('confidence', 0.0)
                    branch = 'true' if confidence >= threshold else 'false'
                else:
                    branch = 'true'

                return {'branch': branch, 'threshold': threshold}

            elif agent_type == 'loop':
                data = inputs.get('data', {})
                max_iterations = config.get('max_iterations', 10)
                return {'iteration': 1, 'max': max_iterations, 'continue': True}

            elif agent_type == 'parallel':
                tasks = inputs.get('tasks', [])
                max_concurrent = config.get('max_concurrent', 5)
                # Execute tasks in parallel
                results = await asyncio.gather(*[
                    self.execute_parallel_task(task)
                    for task in tasks[:max_concurrent]
                ])
                return {'results': results, 'count': len(results)}

            # LLM Agents (NEW!)
            elif agent_type in ['llm_prompt', 'structured_llm', 'prompt_chain',
                                'few_shot', 'llm_consensus', 'rag_prompt']:
                if not LLM_AGENTS_AVAILABLE:
                    logger.warning(f"LLM agent {agent_type} called but LLM dependencies not installed")
                    return {
                        'error': 'LLM agents not available',
                        'message': 'Install: pip install openai anthropic',
                        'agent_type': agent_type
                    }

                logger.info(f"Executing LLM agent: {agent_type}")
                result = await execute_llm_agent(agent_type, config, inputs)
                return result

            else:
                logger.warning(f"Unknown agent type: {agent_type}")
                return {'status': 'unknown_agent_type'}

        except Exception as e:
            logger.error(f"Agent execution failed ({agent_type}): {e}")
            return {'error': str(e), 'status': 'failed'}

    async def execute_parallel_task(self, task):
        """Execute a task in parallel."""
        await asyncio.sleep(0.1)  # Simulate work
        return {'task': task, 'status': 'completed'}

    async def broadcast_progress(self, node_id: str, status: str):
        """Broadcast execution progress to WebSocket clients."""
        message = {
            'type': 'node_status',
            'node_id': node_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

        disconnected = []
        for ws in ws_connections:
            try:
                await ws.send_json(message)
            except:
                disconnected.append(ws)

        # Remove disconnected clients
        for ws in disconnected:
            ws_connections.remove(ws)

    async def broadcast_error(self, error: str):
        """Broadcast error to WebSocket clients."""
        message = {
            'type': 'error',
            'error': error,
            'timestamp': datetime.now().isoformat()
        }

        for ws in ws_connections:
            try:
                await ws.send_json(message)
            except:
                pass


# API Endpoints
@app.post("/api/workflow/execute")
async def execute_workflow(request: ExecutionRequest):
    """Execute a workflow."""
    try:
        executor = WorkflowExecutor(request.workflow, request.input_data)
        result = await executor.execute()
        return result

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflow/validate")
async def validate_workflow(workflow: Workflow):
    """Validate a workflow without executing it."""
    try:
        executor = WorkflowExecutor(workflow)
        executor.validate_workflow()

        return {
            'valid': True,
            'nodes': len(workflow.nodes),
            'connections': len(workflow.connections),
            'start_nodes': executor.find_start_nodes()
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time execution updates and collaboration.

    Supported message types:
    - cursor_move: Broadcast cursor position to all participants
    - presence_update: Update user presence status
    - node_lock: Request exclusive edit lock on a node
    - node_unlock: Release edit lock on a node
    - workflow_operation: Apply CRDT operation to workflow state
    """
    await websocket.accept()
    ws_connections.append(websocket)
    logger.info("WebSocket client connected")

    session_id = None
    user_id = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get('type')

            logger.info(f"Received WebSocket message: {msg_type}")

            # Handle collaboration messages
            if msg_type == 'subscribe':
                # Subscribe to a collaboration session
                session_id = message.get('session_id')
                user_id = message.get('user_id')

                if session_id and session_id in collaboration_sessions:
                    presence_subscriptions[websocket] = session_id
                    logger.info(f"WebSocket subscribed to session {session_id}")

                    await websocket.send_json({
                        'type': 'subscribed',
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    await websocket.send_json({
                        'type': 'error',
                        'error': 'Session not found',
                        'session_id': session_id
                    })

            elif msg_type == 'cursor_move':
                # Broadcast cursor position to session participants
                if session_id and session_id in collaboration_sessions:
                    session = collaboration_sessions[session_id]
                    cursor_data = message.get('data', {})

                    # Update cursor position in session
                    session['cursor_positions'][user_id] = {
                        'x': cursor_data.get('x', 0),
                        'y': cursor_data.get('y', 0),
                        'user_name': cursor_data.get('user_name', 'Unknown'),
                        'color': cursor_data.get('color', '#3b82f6'),
                        'timestamp': datetime.now().timestamp()
                    }

                    # Broadcast to all session participants
                    await broadcast_to_session(session_id, {
                        'type': 'cursor_update',
                        'user_id': user_id,
                        'data': session['cursor_positions'][user_id]
                    }, exclude_ws=websocket)

            elif msg_type == 'presence_update':
                # Update user presence status
                if session_id and session_id in collaboration_sessions:
                    session = collaboration_sessions[session_id]
                    presence_data = message.get('data', {})

                    # Update participant status
                    for participant in session['participants']:
                        if participant['user_id'] == user_id:
                            participant['status'] = presence_data.get('status', 'active')
                            participant['selected_node'] = presence_data.get('selected_node')
                            break

                    # Broadcast presence update
                    await broadcast_to_session(session_id, {
                        'type': 'presence_update',
                        'user_id': user_id,
                        'data': presence_data
                    }, exclude_ws=websocket)

            elif msg_type == 'node_lock':
                # Request exclusive lock on a node
                if session_id and session_id in collaboration_sessions:
                    session = collaboration_sessions[session_id]
                    crdt = session['crdt']
                    node_id = message.get('node_id')

                    if node_id:
                        lock_acquired = crdt.try_lock_node(node_id, user_id)

                        await websocket.send_json({
                            'type': 'node_lock_response',
                            'node_id': node_id,
                            'locked': lock_acquired,
                            'user_id': user_id
                        })

                        if lock_acquired:
                            # Broadcast lock to other participants
                            await broadcast_to_session(session_id, {
                                'type': 'node_locked',
                                'node_id': node_id,
                                'user_id': user_id,
                                'user_name': message.get('user_name', 'Unknown')
                            }, exclude_ws=websocket)

            elif msg_type == 'node_unlock':
                # Release lock on a node
                if session_id and session_id in collaboration_sessions:
                    session = collaboration_sessions[session_id]
                    crdt = session['crdt']
                    node_id = message.get('node_id')

                    if node_id:
                        lock_released = crdt.unlock_node(node_id, user_id)

                        await websocket.send_json({
                            'type': 'node_unlock_response',
                            'node_id': node_id,
                            'unlocked': lock_released
                        })

                        if lock_released:
                            # Broadcast unlock to other participants
                            await broadcast_to_session(session_id, {
                                'type': 'node_unlocked',
                                'node_id': node_id,
                                'user_id': user_id
                            }, exclude_ws=websocket)

            elif msg_type == 'workflow_operation':
                # Apply CRDT operation to workflow state
                if session_id and session_id in collaboration_sessions:
                    session = collaboration_sessions[session_id]
                    crdt = session['crdt']
                    operation = message.get('operation', {})

                    # Apply operation
                    new_state = crdt.apply_operation(user_id, operation)

                    # Broadcast operation to all participants
                    await broadcast_to_session(session_id, {
                        'type': 'workflow_state_update',
                        'operation': operation,
                        'user_id': user_id,
                        'state': new_state,
                        'timestamp': datetime.now().isoformat()
                    }, exclude_ws=websocket)

                    # Acknowledge to sender
                    await websocket.send_json({
                        'type': 'operation_applied',
                        'operation': operation,
                        'state': new_state
                    })

            elif msg_type == 'execute_workflow':
                # Execute workflow via WebSocket (Wave 5.6)
                workflow_data = message.get('workflow')
                input_data = message.get('input_data', {})

                if not workflow_data:
                    await websocket.send_json({
                        'type': 'error',
                        'error': 'No workflow provided',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue

                # Generate execution ID
                execution_id = f"ws-exec-{uuid.uuid4().hex[:8]}"

                try:
                    # Parse workflow
                    workflow = Workflow(**workflow_data)

                    # Create executor
                    executor = WorkflowExecutor(workflow, input_data)
                    executor.execution_id = execution_id

                    # Register in active executions
                    active_ws_executions[execution_id] = {
                        'executor': executor,
                        'status': 'running',
                        'progress': 0.0,
                        'cancelled': False,
                        'websocket': websocket
                    }

                    # Acknowledge start
                    await websocket.send_json({
                        'type': 'execution_started',
                        'execution_id': execution_id,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Broadcast workflow start
                    await broadcast_workflow_status(execution_id, 'start', progress=0.0)
                    await broadcast_log('INFO', f'Starting workflow execution: {workflow.name}', execution_id=execution_id)

                    # Execute workflow with streaming updates
                    result = await execute_workflow_with_streaming(
                        executor, execution_id, websocket
                    )

                    # Check if cancelled
                    if active_ws_executions.get(execution_id, {}).get('cancelled'):
                        await broadcast_workflow_status(execution_id, 'cancelled')
                        await broadcast_log('WARN', 'Workflow execution cancelled', execution_id=execution_id)
                    else:
                        # Broadcast completion
                        await broadcast_workflow_status(execution_id, 'complete', progress=100.0, result=result)
                        await broadcast_log('INFO', f'Workflow completed successfully', execution_id=execution_id)

                except Exception as e:
                    logger.error(f"WebSocket workflow execution failed: {e}")
                    await broadcast_workflow_status(execution_id, 'error', error=str(e))
                    await broadcast_log('ERROR', f'Workflow execution failed: {e}', execution_id=execution_id)

                finally:
                    # Clean up
                    if execution_id in active_ws_executions:
                        del active_ws_executions[execution_id]

            elif msg_type == 'stop_workflow':
                # Stop workflow execution (Wave 5.6)
                execution_id = message.get('execution_id')

                if not execution_id:
                    await websocket.send_json({
                        'type': 'error',
                        'error': 'No execution_id provided',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue

                if execution_id in active_ws_executions:
                    # Mark as cancelled
                    active_ws_executions[execution_id]['cancelled'] = True
                    active_ws_executions[execution_id]['status'] = 'cancelling'

                    await websocket.send_json({
                        'type': 'stop_acknowledged',
                        'execution_id': execution_id,
                        'timestamp': datetime.now().isoformat()
                    })

                    await broadcast_log('WARN', 'Workflow cancellation requested', execution_id=execution_id)
                else:
                    await websocket.send_json({
                        'type': 'error',
                        'error': f'Execution {execution_id} not found or already completed',
                        'timestamp': datetime.now().isoformat()
                    })

            elif msg_type == 'get_logs':
                # Get execution logs with optional filtering (Wave 5.6)
                filter_level = message.get('level')  # Optional: INFO, WARN, ERROR
                filter_node = message.get('node_id')  # Optional: filter by node
                filter_execution = message.get('execution_id')  # Optional: filter by execution
                limit = message.get('limit', 100)

                filtered_logs = execution_logs.copy()

                if filter_level:
                    filtered_logs = [l for l in filtered_logs if l.get('level') == filter_level]
                if filter_node:
                    filtered_logs = [l for l in filtered_logs if l.get('node_id') == filter_node]
                if filter_execution:
                    filtered_logs = [l for l in filtered_logs if l.get('execution_id') == filter_execution]

                # Return latest logs (up to limit)
                await websocket.send_json({
                    'type': 'logs_response',
                    'logs': filtered_logs[-limit:],
                    'total': len(filtered_logs),
                    'timestamp': datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        ws_connections.remove(websocket)

        # Clean up collaboration state
        if session_id and session_id in collaboration_sessions:
            session = collaboration_sessions[session_id]

            # Remove from presence subscriptions
            if websocket in presence_subscriptions:
                del presence_subscriptions[websocket]

            # Release all locks held by this user
            if user_id:
                crdt = session['crdt']
                for node_id in list(crdt.node_locks.keys()):
                    lock = crdt.node_locks[node_id]
                    if lock['user_id'] == user_id:
                        crdt.unlock_node(node_id, user_id)

                # Broadcast user disconnection
                await broadcast_to_session(session_id, {
                    'type': 'user_disconnected',
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat()
                })

        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_connections.remove(websocket)


async def broadcast_to_session(session_id: str, message: dict, exclude_ws: Optional[WebSocket] = None):
    """
    Broadcast a message to all participants in a collaboration session.

    Args:
        session_id: ID of the session
        message: Message to broadcast
        exclude_ws: Optional WebSocket to exclude from broadcast (sender)
    """
    if session_id not in collaboration_sessions:
        return

    disconnected = []

    for ws, ws_session_id in presence_subscriptions.items():
        if ws_session_id == session_id and ws != exclude_ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to WebSocket: {e}")
                disconnected.append(ws)

    # Clean up disconnected WebSockets
    for ws in disconnected:
        if ws in presence_subscriptions:
            del presence_subscriptions[ws]
        if ws in ws_connections:
            ws_connections.remove(ws)


@app.get("/api/agents")
async def get_available_agents():
    """Get list of available agent types."""
    return {
        'query': ['hololoom', 'search', 'multiquery'],
        'process': ['embedder', 'synthesizer', 'refiner'],
        'memory': ['store', 'retrieve', 'fusion'],
        'decision': ['thompson', 'convergence', 'safety'],
        'output': ['response', 'format'],
        'control': ['conditional', 'loop', 'parallel']
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'active_workflows': len(active_workflows),
        'ws_connections': len(ws_connections)
    }


# Version Control Endpoints
@app.post("/api/workflow/save")
async def save_workflow_version(request: SaveVersionRequest):
    """Save a workflow version with commit message."""
    global version_counter
    version_counter += 1

    version_data = {
        'version': version_counter,
        'message': request.message,
        'description': request.description or '',
        'branch': request.branch,
        'timestamp': request.timestamp,
        'workflow': request.workflow.dict(),
        'workflow_name': request.workflow.name
    }

    version_store[version_counter] = version_data

    # Add to branch
    if request.branch not in branch_store:
        branch_store[request.branch] = {'versions': [], 'head': None}

    branch_store[request.branch]['versions'].append(version_counter)
    branch_store[request.branch]['head'] = version_counter

    logger.info(f"Saved workflow version {version_counter} on branch '{request.branch}'")

    return {
        'version': version_counter,
        'message': request.message,
        'branch': request.branch,
        'timestamp': request.timestamp
    }


@app.post("/api/workflow/branch")
async def create_branch(request: CreateBranchRequest):
    """Create a new branch from an existing version."""
    if request.branch_name in branch_store:
        raise HTTPException(status_code=400, detail=f"Branch '{request.branch_name}' already exists")

    if request.from_branch not in branch_store:
        raise HTTPException(status_code=404, detail=f"Source branch '{request.from_branch}' not found")

    # Create new branch
    branch_store[request.branch_name] = {
        'versions': [request.from_version],
        'head': request.from_version
    }

    logger.info(f"Created branch '{request.branch_name}' from '{request.from_branch}' at version {request.from_version}")

    return {
        'branch': request.branch_name,
        'created_from': request.from_branch,
        'created_from_version': request.from_version,
        'branches': list(branch_store.keys())
    }


@app.get("/api/workflow/versions")
async def get_versions(branch: Optional[str] = None):
    """Get version history, optionally filtered by branch."""
    versions = []

    for version_id, version_data in sorted(version_store.items()):
        if branch is None or version_data['branch'] == branch:
            versions.append({
                'version': version_data['version'],
                'message': version_data['message'],
                'description': version_data['description'],
                'branch': version_data['branch'],
                'timestamp': version_data['timestamp'],
                'workflow': version_data['workflow'],
                'workflow_name': version_data['workflow_name']
            })

    return {'versions': versions, 'total': len(versions)}


@app.get("/api/workflow/diff")
async def get_diff(from_version: int, to_version: int):
    """Compare two workflow versions and return differences."""
    if from_version not in version_store:
        raise HTTPException(status_code=404, detail=f"Version {from_version} not found")

    if to_version not in version_store:
        raise HTTPException(status_code=404, detail=f"Version {to_version} not found")

    from_workflow = version_store[from_version]['workflow']
    to_workflow = version_store[to_version]['workflow']

    # Compare nodes
    from_nodes = {n['id']: n for n in from_workflow['nodes']}
    to_nodes = {n['id']: n for n in to_workflow['nodes']}

    nodes_added = []
    nodes_removed = []
    nodes_modified = []

    # Find added and modified nodes
    for node_id, node in to_nodes.items():
        if node_id not in from_nodes:
            nodes_added.append({
                'id': node['id'],
                'agentType': node['agentType']
            })
        elif from_nodes[node_id] != node:
            nodes_modified.append({
                'id': node['id'],
                'agentType': node['agentType']
            })

    # Find removed nodes
    for node_id, node in from_nodes.items():
        if node_id not in to_nodes:
            nodes_removed.append({
                'id': node['id'],
                'agentType': node['agentType']
            })

    # Compare connections
    from_conns = set(f"{c['from']}->{c['to']}" for c in from_workflow['connections'])
    to_conns = set(f"{c['from']}->{c['to']}" for c in to_workflow['connections'])

    connections_added = sorted(to_conns - from_conns)
    connections_removed = sorted(from_conns - to_conns)

    return {
        'from_version': from_version,
        'to_version': to_version,
        'nodes_added': nodes_added,
        'nodes_removed': nodes_removed,
        'nodes_modified': nodes_modified,
        'connections_added': connections_added,
        'connections_removed': connections_removed,
        'connections_changed': connections_added + connections_removed
    }


@app.get("/api/workflow/branches")
async def list_branches():
    """List all available branches."""
    branches = []
    for branch_name, branch_data in branch_store.items():
        branches.append({
            'name': branch_name,
            'head': branch_data['head'],
            'version_count': len(branch_data['versions'])
        })

    return {'branches': branches}


# Ingestion Endpoints (Wave 1.5) - Enhanced with proper models
@app.post("/api/ingest/file", response_model=FileIngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a file using SpinningWheel auto-format detection.

    Supports:
    - PDFs, DOCX, PPTX, Excel files
    - Code files (Python, JavaScript, TypeScript, etc.)
    - Markdown, LaTeX, plain text
    - JSON, YAML, CSV, XML
    - Audio transcripts, video metadata

    Returns job_id for tracking ingestion progress.
    """
    try:
        job_id = str(uuid.uuid4())
        content = await file.read()

        # Try to use SpinningWheel for auto-format detection
        shards_created = 0
        warning = None

        try:
            from HoloLoom.spinningWheel import spin
            shards = await spin(content, filename=file.filename)
            shards_created = len(shards) if shards else 0
            logger.info(f"File {file.filename} ingested: {shards_created} shards created (job: {job_id})")
        except ImportError:
            logger.warning("SpinningWheel not available, returning file metadata only")
            warning = "SpinningWheel not available - install with: pip install -e ."
        except Exception as e:
            logger.warning(f"SpinningWheel processing failed: {e}, returning file metadata")
            warning = str(e)

        return FileIngestResponse(
            job_id=job_id,
            shards_created=shards_created,
            filename=file.filename,
            content_type=file.content_type,
            file_size=len(content),
            warning=warning
        )
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/url", response_model=URLIngestResponse)
async def ingest_url(request: URLIngestRequest):
    """Ingest content from a URL using SpinningWheel.

    Supports:
    - Web pages (HTML, Markdown conversion)
    - RSS feeds
    - API responses (JSON, XML)
    - PDF/Document URLs
    - Code repository URLs (GitHub, GitLab)

    Request body:
    {
        "url": "https://example.com/article",
        "options": {
            "chunk_size": 512,
            "include_metadata": true
        }
    }
    """
    try:
        job_id = str(uuid.uuid4())
        shards_created = 0
        warning = None

        # Try to use SpinningWheel for URL processing
        try:
            from HoloLoom.spinningWheel import spin
            shards = await spin(request.url, **request.options)
            shards_created = len(shards) if shards else 0
            logger.info(f"URL {request.url} ingested: {shards_created} shards created (job: {job_id})")
        except ImportError:
            logger.warning("SpinningWheel not available, returning URL metadata only")
            warning = "SpinningWheel not available - install with: pip install -e ."
        except Exception as e:
            logger.warning(f"SpinningWheel URL processing failed: {e}")
            warning = str(e)

        return URLIngestResponse(
            job_id=job_id,
            shards_created=shards_created,
            url=request.url,
            options_applied=request.options,
            warning=warning
        )
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/entity/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: str):
    """Get entity details and relationships from knowledge graph.

    Returns:
    - Entity content and metadata
    - All incoming and outgoing relationships
    - Relationship types (IS_A, USES, MENTIONS, LEADS_TO, PART_OF, IN_TIME, OCCURRED_AT)
    - Relationship weights for importance ranking

    Example response:
    {
        "entity_id": "thompson_sampling",
        "content": "Thompson Sampling is a Bayesian approach...",
        "metadata": {
            "type": "algorithm",
            "domain": "machine_learning",
            "confidence": 0.92
        },
        "relationships": [
            {
                "target": "bayesian_methods",
                "relation_type": "IS_A",
                "weight": 1.0
            },
            {
                "target": "exploration_exploitation",
                "relation_type": "USES",
                "weight": 0.85
            }
        ],
        "relationship_count": 2
    }
    """
    try:
        if not entity_id or not entity_id.strip():
            raise HTTPException(status_code=400, detail="Entity ID cannot be empty")

        # Initialize empty response
        entity_content = None
        metadata = {}
        relationships = []

        # Try to access any active executor's memory backend
        # This is a graceful implementation that works without persistent executor state
        global active_workflows

        hololoom_instance = None
        for executor in active_workflows.values():
            if executor.hololoom:
                hololoom_instance = executor.hololoom
                break

        # If we found a HoloLoom instance with memory backend, try to use it
        if hololoom_instance:
            try:
                # Try to access the memory graph
                if hasattr(hololoom_instance, '_memory') and hasattr(hololoom_instance._memory, '_backend'):
                    graph = hololoom_instance._memory._backend.graph

                    # Check if entity exists in graph
                    if hasattr(graph, 'G') and entity_id in graph.G.nodes:
                        node_data = graph.G.nodes[entity_id]
                        entity_content = node_data.get('content')
                        metadata = {k: v for k, v in node_data.items() if k != 'content'}

                        # Get relationships (edges)
                        for _, target, edge_data in graph.G.out_edges(entity_id, data=True):
                            relationships.append(EntityRelationship(
                                target=target,
                                relation_type=edge_data.get('relation', 'RELATED'),
                                weight=float(edge_data.get('weight', 1.0))
                            ))

                        # Also get incoming relationships
                        for source, _, edge_data in graph.G.in_edges(entity_id, data=True):
                            relationships.append(EntityRelationship(
                                target=source,
                                relation_type=f"{edge_data.get('relation', 'RELATED')}_FROM",
                                weight=float(edge_data.get('weight', 1.0))
                            ))

                        logger.info(f"Entity query: {entity_id} - found with {len(relationships)} relationships")
                    else:
                        logger.info(f"Entity query: {entity_id} - not found in graph")
            except Exception as e:
                logger.warning(f"Error accessing memory backend for entity {entity_id}: {e}")
        else:
            logger.info(f"Entity query: {entity_id} - no active executors with HoloLoom instance")

        return EntityResponse(
            entity_id=entity_id,
            content=entity_content,
            metadata=metadata,
            relationships=relationships,
            relationship_count=len(relationships)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity lookup failed for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Collaboration Endpoints
@app.post("/api/collaboration/create")
async def create_collaboration_session(workflow_id: str):
    """
    Create a new collaboration session for a workflow.

    Args:
        workflow_id: ID of the workflow to collaborate on

    Returns:
        {
            'session_id': str,
            'workflow_id': str,
            'created_at': float
        }
    """
    session_id = str(uuid.uuid4())
    now = datetime.now().timestamp()

    collaboration_sessions[session_id] = {
        'session_id': session_id,
        'workflow_id': workflow_id,
        'participants': [],
        'cursor_positions': {},
        'last_activity': now,
        'created_at': now,
        'crdt': WorkflowCRDT(workflow_id)
    }

    logger.info(f"Created collaboration session {session_id} for workflow {workflow_id}")

    return {
        'session_id': session_id,
        'workflow_id': workflow_id,
        'created_at': now
    }


@app.post("/api/collaboration/join/{session_id}")
async def join_collaboration_session(session_id: str, request: JoinSessionRequest):
    """
    Join an existing collaboration session.

    Args:
        session_id: ID of the session to join
        request: User info (user_id, user_name, optional color)

    Returns:
        Session state with current participants
    """
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")

    session = collaboration_sessions[session_id]

    # Check if user already in session
    existing_participant = next(
        (p for p in session['participants'] if p['user_id'] == request.user_id),
        None
    )

    if not existing_participant:
        # Add new participant
        participant = {
            'user_id': request.user_id,
            'user_name': request.user_name,
            'color': request.color or f"#{hash(request.user_id) % 0xFFFFFF:06x}",
            'joined_at': datetime.now().timestamp(),
            'status': 'active'
        }
        session['participants'].append(participant)
        logger.info(f"User {request.user_name} joined session {session_id}")
    else:
        # Update existing participant
        existing_participant['status'] = 'active'
        logger.info(f"User {request.user_name} rejoined session {session_id}")

    session['last_activity'] = datetime.now().timestamp()

    return {
        'session_id': session_id,
        'workflow_id': session['workflow_id'],
        'participants': session['participants'],
        'created_at': session['created_at']
    }


@app.get("/api/collaboration/session/{session_id}")
async def get_collaboration_session(session_id: str):
    """
    Get current state of a collaboration session.

    Returns:
        Complete session state including participants, cursor positions, locks
    """
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")

    session = collaboration_sessions[session_id]
    crdt = session['crdt']

    return {
        'session_id': session_id,
        'workflow_id': session['workflow_id'],
        'participants': session['participants'],
        'cursor_positions': session.get('cursor_positions', {}),
        'node_locks': crdt.node_locks,
        'workflow_state': crdt.state,
        'created_at': session['created_at'],
        'last_activity': session['last_activity']
    }


@app.post("/api/collaboration/leave/{session_id}")
async def leave_collaboration_session(session_id: str, user_id: str):
    """
    Leave a collaboration session.

    Args:
        session_id: ID of the session to leave
        user_id: ID of the user leaving

    Returns:
        Updated session state
    """
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")

    session = collaboration_sessions[session_id]
    crdt = session['crdt']

    # Remove participant
    session['participants'] = [
        p for p in session['participants']
        if p['user_id'] != user_id
    ]

    # Release all node locks held by this user
    for node_id in list(crdt.node_locks.keys()):
        lock = crdt.node_locks[node_id]
        if lock['user_id'] == user_id:
            crdt.unlock_node(node_id, user_id)

    # Remove cursor position
    if user_id in session.get('cursor_positions', {}):
        del session['cursor_positions'][user_id]

    session['last_activity'] = datetime.now().timestamp()

    logger.info(f"User {user_id} left session {session_id}")

    # If no participants left, clean up session after 5 minutes
    if not session['participants']:
        logger.info(f"Session {session_id} now empty, will clean up after timeout")

    return {
        'session_id': session_id,
        'participants': session['participants'],
        'left': True
    }


# ============================================================================
# Auto-Optimization Endpoints (Wave 2)
# ============================================================================

@app.get("/api/workflow/{workflow_id}/profile")
async def get_workflow_performance_profile(workflow_id: str):
    """
    Get performance profile for a workflow.

    Returns performance metrics for all nodes including:
    - Average and p95 latency
    - Success rate
    - Throughput (calls per minute)
    - Bottleneck score (0-1, higher = more likely bottleneck)

    Example response:
    {
        "workflow_id": "my_workflow",
        "profiles": [
            {
                "node_id": "node_1",
                "node_type": "hololoom",
                "avg_latency_ms": 150.0,
                "p95_latency_ms": 250.0,
                "success_rate": 0.95,
                "throughput": 10.0,
                "bottleneck_score": 0.45
            }
        ],
        "total_nodes": 5
    }
    """
    if not OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Optimization engine not available"
        )

    # Mock performance data (in production, this would come from execution history)
    # You would track this in node_performance_stats during execution
    mock_workflow = {
        'name': workflow_id,
        'nodes': [
            {'id': 'node_1', 'agentType': 'hololoom'},
            {'id': 'node_2', 'agentType': 'embedder'},
            {'id': 'node_3', 'agentType': 'synthesizer'}
        ]
    }

    mock_performance = {
        'node_1': {'avg_latency_ms': 150, 'p95_latency_ms': 250, 'success_rate': 0.95, 'call_count': 10},
        'node_2': {'avg_latency_ms': 50, 'p95_latency_ms': 75, 'success_rate': 0.98, 'call_count': 15},
        'node_3': {'avg_latency_ms': 200, 'p95_latency_ms': 400, 'success_rate': 0.85, 'call_count': 8}
    }

    profiles = optimizer.get_performance_profile(mock_workflow, mock_performance)

    return {
        'workflow_id': workflow_id,
        'profiles': profiles,
        'total_nodes': len(profiles)
    }


@app.post("/api/workflow/{workflow_id}/optimize")
async def get_optimization_suggestions(workflow_id: str, workflow: Workflow):
    """
    Get Thompson Sampling-based optimization suggestions for a workflow.

    Request body should contain the full workflow definition.

    Returns suggestions with:
    - Suggestion type (parallelize, cache, reorder, substitute)
    - Description of optimization
    - Expected improvement percentage
    - Confidence level from Thompson Sampling
    - List of affected nodes

    Example response:
    {
        "workflow_id": "my_workflow",
        "suggestion": {
            "suggestion_id": "abc-123",
            "workflow_id": "my_workflow",
            "suggestion_type": "parallelize",
            "description": "Parallelize nodes node_2, node_3 - no data dependencies",
            "expected_improvement": 35.5,
            "confidence": 0.72,
            "affected_nodes": ["node_2", "node_3"]
        },
        "timestamp": "2025-12-09T10:30:00"
    }
    """
    if not OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Optimization engine not available"
        )

    # Mock performance data (in production, from execution history)
    mock_performance = {
        node.id: {
            'avg_latency_ms': 100 + (hash(node.id) % 100),
            'p95_latency_ms': 150 + (hash(node.id) % 150),
            'success_rate': 0.85 + (hash(node.id) % 15) / 100,
            'call_count': 5 + (hash(node.id) % 10)
        }
        for node in workflow.nodes
    }

    # Get suggestion using Thompson Sampling
    workflow_dict = workflow.dict()
    suggestion = optimizer.suggest_optimization(workflow_dict, mock_performance)

    # Store in optimization history
    optimization_history.append({
        **suggestion,
        'timestamp': datetime.now().isoformat(),
        'status': 'suggested'
    })

    logger.info(f"Generated optimization suggestion for {workflow_id}: {suggestion['suggestion_type']}")

    return {
        'workflow_id': workflow_id,
        'suggestion': suggestion,
        'timestamp': datetime.now().isoformat()
    }


@app.post("/api/workflow/{workflow_id}/apply-optimization")
async def apply_optimization(workflow_id: str, suggestion_id: str, success: bool = True, improvement: float = 0.0):
    """
    Apply an optimization suggestion and update Thompson Sampling beliefs.

    Query parameters:
    - suggestion_id: ID of the suggestion to apply
    - success: Whether the optimization was successful (default: True)
    - improvement: Actual improvement percentage achieved (default: 0.0)

    This updates the Thompson Sampling priors (alpha/beta) based on outcome,
    allowing the system to learn which optimization strategies work best.

    Example:
    POST /api/workflow/my_workflow/apply-optimization?suggestion_id=abc-123&success=true&improvement=32.5

    Response:
    {
        "workflow_id": "my_workflow",
        "suggestion_id": "abc-123",
        "applied": true,
        "outcome": {
            "success": true,
            "improvement": 32.5
        },
        "updated_priors": {
            "parallelize": {"alpha": 2.0, "beta": 1.0}
        }
    }
    """
    if not OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Optimization engine not available"
        )

    outcome = {
        'success': success,
        'improvement': improvement
    }

    # Update Thompson Sampling beliefs
    optimizer.update_belief(suggestion_id, outcome, optimization_history)

    # Update suggestion status in history
    for item in optimization_history:
        if item.get('suggestion_id') == suggestion_id:
            item['status'] = 'applied'
            item['outcome'] = outcome
            item['applied_at'] = datetime.now().isoformat()
            break

    logger.info(f"Applied optimization {suggestion_id} for {workflow_id}: success={success}, improvement={improvement}%")

    return {
        'workflow_id': workflow_id,
        'suggestion_id': suggestion_id,
        'applied': True,
        'outcome': outcome,
        'updated_priors': optimizer.priors,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/api/optimization/history")
async def get_optimization_history(limit: int = 10):
    """
    Get history of optimization suggestions and their outcomes.

    Query parameters:
    - limit: Maximum number of history entries to return (default: 10)

    Returns chronological list of suggestions with their status and outcomes.

    Example response:
    {
        "history": [
            {
                "suggestion_id": "abc-123",
                "workflow_id": "my_workflow",
                "suggestion_type": "parallelize",
                "timestamp": "2025-12-09T10:30:00",
                "status": "applied",
                "outcome": {
                    "success": true,
                    "improvement": 32.5
                }
            }
        ],
        "total": 1,
        "thompson_priors": {
            "parallelize": {"alpha": 2.0, "beta": 1.0},
            "cache": {"alpha": 1.0, "beta": 1.0},
            "reorder": {"alpha": 1.0, "beta": 1.0},
            "substitute": {"alpha": 1.0, "beta": 1.0}
        }
    }
    """
    if not OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Optimization engine not available"
        )

    # Return most recent entries
    recent_history = optimization_history[-limit:] if limit > 0 else optimization_history

    return {
        'history': recent_history,
        'total': len(optimization_history),
        'thompson_priors': optimizer.priors,
        'timestamp': datetime.now().isoformat()
    }


# ==============================================================================
# AI Suggestion Endpoints (Wave 3.2)
# ==============================================================================

@app.get("/api/suggestions/next-nodes")
async def get_next_node_suggestions(current: str = "", k: int = 5):
    """
    Get AI-powered suggestions for the next node to add to a workflow.

    Uses Thompson Sampling to recommend nodes based on historical patterns
    of successful workflow executions.

    Query parameters:
    - current: Comma-separated list of current node types in execution order
    - k: Number of suggestions to return (default: 5)

    Example: /api/suggestions/next-nodes?current=hololoom,search&k=3

    Returns suggestions with confidence scores and pattern sources.
    """
    from pattern_miner import suggest_next_nodes, get_pattern_miner

    # Parse current nodes
    current_nodes = [n.strip() for n in current.split(',') if n.strip()]

    # Get suggestions
    suggestions = suggest_next_nodes(current_nodes, k=k)

    # Also return node performance stats for context
    miner = get_pattern_miner()
    rankings = miner.get_node_performance_ranking()

    return {
        'suggestions': suggestions,
        'current_context': current_nodes,
        'node_rankings': rankings[:10],  # Top 10 performing nodes
        'patterns_available': len(miner.pair_patterns) + len(miner.triple_patterns),
        'timestamp': datetime.now().isoformat()
    }


@app.get("/api/suggestions/patterns")
async def get_workflow_patterns(min_frequency: int = 2, pattern_type: str = "all"):
    """
    Get common workflow patterns for template suggestions.

    Query parameters:
    - min_frequency: Minimum occurrences to include pattern (default: 2)
    - pattern_type: Filter by pattern type ("pair", "triple", or "all")

    Returns list of common node sequences with success rates.
    """
    from pattern_miner import get_common_patterns, get_pattern_miner

    patterns = get_common_patterns(min_frequency=min_frequency)

    # Filter by type if specified
    if pattern_type != "all":
        patterns = [p for p in patterns if p['type'] == pattern_type]

    return {
        'patterns': patterns,
        'total': len(patterns),
        'min_frequency': min_frequency,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/api/patterns/frequency")
async def get_pattern_frequency():
    """
    Get node type frequency and success statistics.

    Returns ranking of all node types by Thompson Sampling score,
    useful for ordering the agent palette by relevance.
    """
    from pattern_miner import get_pattern_miner

    miner = get_pattern_miner()
    rankings = miner.get_node_performance_ranking()

    # Also include overall stats
    total_executions = sum(miner.node_frequency.values())
    total_success = sum(miner.node_success.values())

    return {
        'rankings': rankings,
        'total_executions': total_executions,
        'overall_success_rate': total_success / total_executions if total_executions > 0 else 0.5,
        'unique_patterns': {
            'pairs': len(miner.pair_patterns),
            'triples': len(miner.triple_patterns)
        },
        'timestamp': datetime.now().isoformat()
    }


@app.post("/api/patterns/learn")
async def learn_from_workflow_history():
    """
    Trigger pattern learning from workflow execution history.

    Extracts patterns from the global workflow_pattern_history
    and updates the pattern miner's knowledge base.
    """
    global workflow_pattern_history

    from pattern_miner import get_pattern_miner

    miner = get_pattern_miner()

    # Learn from history
    if workflow_pattern_history:
        miner.extract_patterns(workflow_pattern_history)
        miner.save_patterns()

        return {
            'status': 'success',
            'patterns_learned': {
                'pairs': len(miner.pair_patterns),
                'triples': len(miner.triple_patterns)
            },
            'workflows_processed': len(workflow_pattern_history),
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'status': 'no_data',
            'message': 'No workflow history to learn from',
            'timestamp': datetime.now().isoformat()
        }


@app.get("/api/suggestions/performance")
async def get_performance_suggestions():
    """
    Get performance-based suggestions for workflow optimization.

    Returns real-time performance statistics including:
    - Node type latencies
    - Bottleneck detection
    - Thompson Sampling priors
    """
    global execution_metrics, node_performance_stats

    performance_data = []
    for node_type in execution_metrics.keys():
        stats = get_real_performance_stats(node_type)
        performance_data.append({
            'node_type': node_type,
            **stats,
            'is_bottleneck': stats.get('avg_latency_ms', 0) > 400,
            'thompson_alpha': node_performance_stats.get(node_type, {}).get('alpha', 1.0),
            'thompson_beta': node_performance_stats.get(node_type, {}).get('beta', 1.0)
        })

    # Sort by latency (bottlenecks first)
    performance_data.sort(key=lambda x: x.get('avg_latency_ms', 0), reverse=True)

    return {
        'performance': performance_data,
        'bottlenecks': [p for p in performance_data if p.get('is_bottleneck', False)],
        'recommendations': _generate_performance_recommendations(performance_data),
        'timestamp': datetime.now().isoformat()
    }


def _generate_performance_recommendations(performance_data: list) -> list:
    """Generate actionable performance recommendations."""
    recommendations = []

    for node in performance_data:
        node_type = node.get('node_type', '')
        avg_latency = node.get('avg_latency_ms', 0)
        success_rate = node.get('success_rate', 1.0)

        # High latency recommendation
        if avg_latency > 500:
            recommendations.append({
                'node_type': node_type,
                'issue': 'high_latency',
                'severity': 'high' if avg_latency > 1000 else 'medium',
                'recommendation': f"Consider caching results or using a lighter alternative for {node_type}",
                'metric': f"{avg_latency:.0f}ms avg latency"
            })

        # Low success rate recommendation
        if success_rate < 0.7:
            recommendations.append({
                'node_type': node_type,
                'issue': 'low_success_rate',
                'severity': 'high' if success_rate < 0.5 else 'medium',
                'recommendation': f"Review {node_type} configuration - success rate is below threshold",
                'metric': f"{success_rate:.0%} success rate"
            })

    return recommendations


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("HoloLoom Workflow Executor")
    print("=" * 80)
    print("\nStarting server...")
    print("Open workflow_builder.html in your browser to design workflows")
    print("\nAPI documentation: http://localhost:8001/docs")
    print("=" * 80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
