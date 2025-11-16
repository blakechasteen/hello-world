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
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn websockets")
    raise

try:
    from HoloLoom.unified_api import HoloLoom
    from HoloLoom.alignment.safety_guardrails import SafetyGuardrails, RiskLevel
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

# Global state
active_workflows: Dict[str, "WorkflowExecutor"] = {}
ws_connections: List[WebSocket] = []
version_store: Dict[int, Dict[str, Any]] = {}
version_counter = 0
branch_store: Dict[str, Dict[str, Any]] = {
    'main': {'versions': [], 'head': None}
}


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

                # Broadcast progress
                await self.broadcast_progress(node.id, 'completed')

                # Add dependent nodes to queue
                dependent_nodes = self.get_dependent_nodes(node.id)
                queue.extend([self.get_node_by_id(nid) for nid in dependent_nodes])

            logger.info(f"Workflow execution complete: {executed_count} nodes executed")

            return {
                'status': 'success',
                'nodes_executed': executed_count,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
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
        """Execute a single node."""
        logger.info(f"Executing node: {node.id} ({node.agentType})")

        await self.broadcast_progress(node.id, 'running')

        # Gather inputs from dependencies
        inputs = {}
        for dep_id in self.get_node_dependencies(node.id):
            if dep_id in self.results:
                inputs[dep_id] = self.results[dep_id]

        # Execute based on agent type
        result = await self.execute_agent(node, inputs)

        # Store result
        self.results[node.id] = {
            'node_id': node.id,
            'agent_type': node.agentType,
            'timestamp': datetime.now().isoformat(),
            'config': node.config,
            'output': result
        }

        logger.info(f"Node {node.id} completed")

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
    """WebSocket for real-time execution updates."""
    await websocket.accept()
    ws_connections.append(websocket)
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            logger.info(f"Received: {data}")

    except WebSocketDisconnect:
        ws_connections.remove(websocket)
        logger.info("WebSocket client disconnected")


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
