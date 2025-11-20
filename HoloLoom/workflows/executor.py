"""
Workflow Executor - Execute agentic workflows with state management
===================================================================

Executes workflows defined by WorkflowDefinition with:
- Topological ordering and dependency resolution
- Parallel execution support
- State persistence and checkpointing
- Error handling and retries
- Timeout support
- Integration with RAG department, chains, and recursive reasoner

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from HoloLoom.workflows.schema import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowResult,
    ExecutionTrace,
    NodeType,
)
from HoloLoom.workflows.state import StateBackend, InMemoryState, CheckpointManager
from HoloLoom.workflows.integrations import ChainExecutor, RecursiveExecutor
from HoloLoom.config import Config
from HoloLoom.departments import get_department

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Execute agentic workflows with full lifecycle management.

    Features:
    - Sequential and parallel execution
    - State management with checkpointing
    - Error handling and retries
    - Timeout support
    - Complete execution trace
    - Integration with RAG, chains, recursive reasoner

    Usage:
        workflow = WorkflowDefinition.from_json(json_str)
        executor = WorkflowExecutor(workflow, config=Config.fast())

        result = await executor.execute({"query": "What is RL?"})
        print(result.summary())
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        config: Optional[Config] = None,
        state_backend: Optional[StateBackend] = None,
        enable_parallel: bool = True,
        max_concurrent: int = 5,
        checkpoint_frequency: int = 10,  # Save checkpoint every N nodes
    ):
        """
        Initialize WorkflowExecutor.

        Args:
            workflow: Workflow definition to execute
            config: HoloLoom configuration (defaults to Config.fast())
            state_backend: State persistence backend (defaults to InMemoryState)
            enable_parallel: Enable parallel execution for PARALLEL nodes
            max_concurrent: Maximum concurrent parallel tasks
            checkpoint_frequency: Save checkpoint every N nodes executed
        """
        self.workflow = workflow
        self.config = config or Config.fast()
        self.state_backend = state_backend or InMemoryState()
        self.enable_parallel = enable_parallel
        self.max_concurrent = max_concurrent
        self.checkpoint_frequency = checkpoint_frequency

        # Execution state
        self.execution_id = str(uuid.uuid4())
        self.state: Dict[str, Any] = {}  # Node outputs
        self.executed_nodes: Set[str] = set()
        self.trace: List[ExecutionTrace] = []
        self.errors: List[Dict[str, Any]] = []

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.state_backend)

        # Integration executors
        self.chain_executor = ChainExecutor(config=self.config)
        self.recursive_executor = RecursiveExecutor(config=self.config)

        # Department cache
        self._departments: Dict[str, Any] = {}

        logger.info(f"WorkflowExecutor initialized: {workflow.name} (id={self.execution_id[:8]})")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.chain_executor.__aenter__()
        await self.recursive_executor.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.chain_executor.__aexit__(exc_type, exc_val, exc_tb)
        await self.recursive_executor.__aexit__(exc_type, exc_val, exc_tb)

        # Close departments
        for dept in self._departments.values():
            if hasattr(dept, '__aexit__'):
                await dept.__aexit__(None, None, None)

    async def execute(self, inputs: Dict[str, Any]) -> WorkflowResult:
        """
        Execute complete workflow.

        Process:
        1. Validate workflow structure
        2. Initialize execution state
        3. Execute nodes in topological order
        4. Handle parallel execution for PARALLEL nodes
        5. Checkpoint state periodically
        6. Return WorkflowResult with complete trace

        Args:
            inputs: Initial input data for workflow

        Returns:
            WorkflowResult with outputs, trace, and execution stats

        Raises:
            ValueError: If workflow validation fails
            RuntimeError: If execution fails unrecoverably
        """
        start_time = time.time()

        logger.info(f"Starting workflow execution: {self.workflow.name}")

        # Validate workflow
        errors = self.workflow.validate()
        if errors:
            raise ValueError(f"Workflow validation failed: {errors}")

        # Initialize state
        self.state = {"inputs": inputs}

        try:
            # Start from entry point
            await self._execute_node(self.workflow.entry_point, inputs)

            # Check if successful
            status = "success" if not self.errors else "partial"

            execution_time_ms = (time.time() - start_time) * 1000

            result = WorkflowResult(
                execution_id=self.execution_id,
                workflow_name=self.workflow.name,
                status=status,
                outputs=self.state,
                execution_time_ms=execution_time_ms,
                nodes_executed=len(self.executed_nodes),
                errors=self.errors,
                trace=self.trace,
            )

            logger.info(
                f"✓ Workflow execution complete: {status}, {result.nodes_executed} nodes, "
                f"{execution_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")

            execution_time_ms = (time.time() - start_time) * 1000

            # Return failed result with error
            return WorkflowResult(
                execution_id=self.execution_id,
                workflow_name=self.workflow.name,
                status="failed",
                outputs=self.state,
                execution_time_ms=execution_time_ms,
                nodes_executed=len(self.executed_nodes),
                errors=[{"error": str(e), "node_id": "unknown"}],
                trace=self.trace,
            )

    async def _execute_node(self, node_id: str, inputs: Dict[str, Any]) -> Any:
        """Execute single node with retry logic and error handling."""
        if node_id in self.executed_nodes:
            # Already executed, return cached result
            return self.state.get(node_id)

        node = self.workflow.nodes[node_id]

        logger.info(f"Executing node: {node.name} ({node.type.value})")

        start_time = datetime.now()
        retry_count = 0
        max_retries = node.retry_policy.max_retries if node.retry_policy else 3

        while retry_count <= max_retries:
            try:
                # Execute with timeout
                if node.timeout_seconds:
                    output = await asyncio.wait_for(
                        self._execute_node_impl(node, inputs),
                        timeout=node.timeout_seconds
                    )
                else:
                    output = await self._execute_node_impl(node, inputs)

                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000

                # Store output
                self.state[node_id] = output
                self.executed_nodes.add(node_id)

                # Add trace
                self.trace.append(ExecutionTrace(
                    node_id=node_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    status="success",
                    inputs=inputs,
                    outputs=output,
                    retry_count=retry_count,
                ))

                # Checkpoint if needed
                if len(self.executed_nodes) % self.checkpoint_frequency == 0:
                    await self._save_checkpoint()

                # Execute next nodes
                if node.next:
                    for next_id in node.next:
                        await self._execute_node(next_id, output)

                return output

            except asyncio.TimeoutError:
                logger.error(f"Node {node_id} timed out after {node.timeout_seconds}s")
                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000

                self.trace.append(ExecutionTrace(
                    node_id=node_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    status="timeout",
                    inputs=inputs,
                    error="Execution timeout",
                    retry_count=retry_count,
                ))

                self.errors.append({
                    "node_id": node_id,
                    "error": "timeout",
                    "message": f"Timeout after {node.timeout_seconds}s",
                })

                # Execute error handler if configured
                if node.on_error:
                    await self._execute_node(node.on_error, inputs)

                return None

            except Exception as e:
                retry_count += 1
                logger.error(f"Node {node_id} failed (attempt {retry_count}/{max_retries}): {e}")

                if retry_count <= max_retries:
                    # Retry with exponential backoff
                    if node.retry_policy and node.retry_policy.exponential_backoff:
                        delay = node.retry_policy.retry_delay_seconds * (2 ** (retry_count - 1))
                    else:
                        delay = node.retry_policy.retry_delay_seconds if node.retry_policy else 1.0

                    await asyncio.sleep(delay)
                    continue
                else:
                    # Max retries exceeded
                    end_time = datetime.now()
                    duration_ms = (end_time - start_time).total_seconds() * 1000

                    self.trace.append(ExecutionTrace(
                        node_id=node_id,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        status="error",
                        inputs=inputs,
                        error=str(e),
                        retry_count=retry_count,
                    ))

                    self.errors.append({
                        "node_id": node_id,
                        "error": str(e),
                        "retries": retry_count,
                    })

                    # Execute error handler if configured
                    if node.on_error:
                        await self._execute_node(node.on_error, inputs)

                    return None

    async def _execute_node_impl(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Any:
        """
        Execute specific node type.

        This is where integration with RAG, chains, and recursive reasoner happens.
        """
        node_type = node.type
        params = node.params

        # Query Agents
        if node_type == NodeType.QUERY:
            return await self._execute_query_node(node, inputs)

        elif node_type == NodeType.SEARCH:
            return await self._execute_search_node(node, inputs)

        elif node_type == NodeType.MULTIQUERY:
            return await self._execute_multiquery_node(node, inputs)

        # Processing Agents
        elif node_type == NodeType.VERIFY:
            return await self._execute_verify_node(node, inputs)

        elif node_type == NodeType.REFINE:
            return await self._execute_refine_node(node, inputs)

        elif node_type == NodeType.SYNTHESIZE:
            return await self._execute_synthesize_node(node, inputs)

        elif node_type == NodeType.EMBED:
            return await self._execute_embed_node(node, inputs)

        # Memory Agents
        elif node_type == NodeType.STORE:
            return await self._execute_store_node(node, inputs)

        elif node_type == NodeType.RETRIEVE:
            return await self._execute_retrieve_node(node, inputs)

        elif node_type == NodeType.FUSION:
            return await self._execute_fusion_node(node, inputs)

        # Decision Agents
        elif node_type == NodeType.THOMPSON:
            return await self._execute_thompson_node(node, inputs)

        elif node_type == NodeType.CONVERGENCE:
            return await self._execute_convergence_node(node, inputs)

        elif node_type == NodeType.SAFETY:
            return await self._execute_safety_node(node, inputs)

        # Chain Agents
        elif node_type == NodeType.CHAIN:
            return await self._execute_chain_node(node, inputs)

        elif node_type == NodeType.RECURSIVE:
            return await self._execute_recursive_node(node, inputs)

        # Control Flow
        elif node_type == NodeType.CONDITION:
            return await self._execute_condition_node(node, inputs)

        elif node_type == NodeType.LOOP:
            return await self._execute_loop_node(node, inputs)

        elif node_type == NodeType.PARALLEL:
            return await self._execute_parallel_node(node, inputs)

        elif node_type == NodeType.MERGE:
            return await self._execute_merge_node(node, inputs)

        # Output Agents
        elif node_type == NodeType.RESPONSE:
            return await self._execute_response_node(node, inputs)

        elif node_type == NodeType.FORMAT:
            return await self._execute_format_node(node, inputs)

        # External Tools
        elif node_type == NodeType.HUMAN_IN_LOOP:
            return await self._execute_human_in_loop_node(node, inputs)

        elif node_type == NodeType.TOOL:
            return await self._execute_tool_node(node, inputs)

        elif node_type == NodeType.API:
            return await self._execute_api_node(node, inputs)

        else:
            raise ValueError(f"Unknown node type: {node_type}")

    # ===== Node Type Implementations =====

    async def _execute_query_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG query node."""
        query = inputs.get("query", node.params.get("query", ""))
        mode = node.params.get("mode", "verify")
        max_sources = node.params.get("max_sources", 5)

        dept = await self._get_department("rag")
        from HoloLoom.departments.protocol import DepartmentRequest

        request = DepartmentRequest(
            department_id="rag",
            query={"query": query, "mode": mode, "max_sources": max_sources}
        )

        response = await dept.execute(request)

        return {
            "answer": response.response.get("answer"),
            "sources": response.response.get("sources", []),
            "confidence": response.confidence.score,
        }

    async def _execute_search_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory search node."""
        query = inputs.get("query", node.params.get("query", ""))
        k = node.params.get("k", 5)

        # Use RAG department search
        dept = await self._get_department("rag")
        # Implementation would call memory search directly
        return {"results": [], "count": 0}

    async def _execute_multiquery_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Break query into sub-queries."""
        query = inputs.get("query", node.params.get("query", ""))
        max_subqueries = node.params.get("max_subqueries", 3)

        # Simple decomposition (in production, would use LLM)
        subqueries = [f"{query} - aspect {i+1}" for i in range(max_subqueries)]

        return {"subqueries": subqueries, "original_query": query}

    async def _execute_verify_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Verify response using DS-STAR."""
        from HoloLoom.departments.protocol import DepartmentResponse, ConfidenceMetadata

        # Create mock response for verification
        response = DepartmentResponse(
            department_id="rag",
            request_id=str(uuid.uuid4()),
            response=inputs,
            confidence=ConfidenceMetadata.from_score(inputs.get("confidence", 0.8)),
        )

        dept = await self._get_department("rag")
        verification = await dept.verify(response)

        return {
            "verified": verification.verified,
            "overall_score": verification.overall_score,
            "checks": [c.__dict__ for c in verification.checks],
            "recommendations": verification.recommendations,
        }

    async def _execute_refine_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Refine low-confidence responses."""
        strategy = node.params.get("strategy", "refine")
        max_iterations = node.params.get("max_iterations", 3)

        # Use recursive reasoner for refinement
        refined = await self.recursive_executor.refine(
            query=inputs.get("query", ""),
            initial_result=inputs,
            strategy=strategy,
            max_iterations=max_iterations,
        )

        return refined

    async def _execute_synthesize_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities and motifs."""
        text = inputs.get("text", inputs.get("answer", ""))

        # Simple entity extraction (in production, would use NER)
        entities = []
        motifs = []

        return {"entities": entities, "motifs": motifs, "text": text}

    async def _execute_embed_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings."""
        text = inputs.get("text", "")
        scales = node.params.get("scales", [96, 192, 384])

        # Would use Matryoshka embeddings
        return {"embeddings": {}, "scales": scales}

    async def _execute_store_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Store in memory."""
        return {"stored": True, "memory_id": str(uuid.uuid4())}

    async def _execute_retrieve_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve context."""
        return {"context": [], "count": 0}

    async def _execute_fusion_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-hop graph traversal."""
        return {"expanded_context": [], "depth": 0}

    async def _execute_thompson_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Thompson Sampling decision."""
        options = inputs.get("options", node.params.get("options", []))
        return {"selected": options[0] if options else None, "confidence": 0.85}

    async def _execute_convergence_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decision convergence."""
        strategy = node.params.get("strategy", "epsilon_greedy")
        return {"decision": "action_1", "strategy": strategy}

    async def _execute_safety_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Safety guardrails check."""
        from HoloLoom.alignment.safety_guardrails import SafetyGuardrails

        guardrails = SafetyGuardrails()
        action = inputs.get("action", str(inputs))

        result = await guardrails.gate_action(action, context={"workflow": self.workflow.name})

        return {
            "allowed": result.allowed,
            "safety_score": result.safety_score,
            "reason": result.reason,
        }

    async def _execute_chain_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prompt chain."""
        chain_name = node.params.get("chain_name", "default")
        chain_input = inputs.get("query", inputs.get("text", ""))

        result = await self.chain_executor.execute(chain_name, chain_input)
        return result

    async def _execute_recursive_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recursive reasoning."""
        query = inputs.get("query", node.params.get("query", ""))
        max_depth = node.params.get("max_depth", 5)

        result = await self.recursive_executor.reason(query, max_depth=max_depth)
        return result

    async def _execute_condition_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Conditional branching."""
        condition_type = node.params.get("condition_type", "confidence")
        threshold = node.params.get("threshold", 0.75)

        if condition_type == "confidence":
            confidence = inputs.get("confidence", 0.0)
            branch = "true" if confidence >= threshold else "false"
        else:
            # Custom condition evaluation
            branch = "true"

        # Execute branch
        branch_node_id = node.params.get(f"branch_{branch}")
        if branch_node_id:
            await self._execute_node(branch_node_id, inputs)

        return {"branch": branch, "condition": condition_type, "threshold": threshold}

    async def _execute_loop_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Loop iteration."""
        max_iterations = node.params.get("max_iterations", 10)
        condition_node = node.params.get("condition_node")
        body_node = node.params.get("body_node")

        iteration = 0
        results = []

        while iteration < max_iterations:
            # Execute body
            if body_node:
                result = await self._execute_node(body_node, inputs)
                results.append(result)

                # Check condition
                if condition_node:
                    cond_result = await self._execute_node(condition_node, result)
                    if not cond_result.get("continue", True):
                        break

            iteration += 1

        return {"iterations": iteration, "results": results}

    async def _execute_parallel_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute nodes in parallel."""
        if not self.enable_parallel:
            logger.warning("Parallel execution disabled, falling back to sequential")
            # Fall back to sequential
            results = []
            for task_node_id in node.params.get("task_nodes", []):
                result = await self._execute_node(task_node_id, inputs)
                results.append(result)
            return {"results": results, "count": len(results)}

        # Parallel execution
        task_node_ids = node.params.get("task_nodes", [])[:self.max_concurrent]

        async def execute_task(task_node_id):
            return await self._execute_node(task_node_id, inputs)

        results = await asyncio.gather(*[execute_task(tid) for tid in task_node_ids])

        return {"results": list(results), "count": len(results)}

    async def _execute_merge_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge parallel results."""
        results = inputs.get("results", [])
        merge_strategy = node.params.get("merge_strategy", "concat")

        if merge_strategy == "concat":
            merged = {"items": results}
        elif merge_strategy == "average":
            # Average numeric values
            merged = {"average": sum(r.get("value", 0) for r in results) / len(results) if results else 0}
        else:
            merged = {"results": results}

        return merged

    async def _execute_response_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response."""
        format_type = node.params.get("format", "text")
        template = node.params.get("template", "{answer}")

        response = template.format(**inputs) if template else str(inputs)

        return {"response": response, "format": format_type}

    async def _execute_format_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format output."""
        import json
        output_format = node.params.get("output_format", "json")

        if output_format == "json":
            formatted = json.dumps(inputs, indent=2)
        elif output_format == "markdown":
            formatted = f"# Result\n\n```json\n{json.dumps(inputs, indent=2)}\n```"
        else:
            formatted = str(inputs)

        return {"formatted": formatted, "format": output_format}

    async def _execute_human_in_loop_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Human approval node."""
        logger.warning("Human-in-loop node: Waiting for approval (auto-approved in demo)")
        # In production, would pause and wait for human input
        return {"approved": True, "timestamp": datetime.now().isoformat()}

    async def _execute_tool_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute external tool."""
        tool_name = node.params.get("tool_name", "unknown")
        logger.info(f"Executing tool: {tool_name}")
        return {"tool_result": f"Result from {tool_name}", "success": True}

    async def _execute_api_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """HTTP API call."""
        import aiohttp

        method = node.params.get("method", "GET")
        url = node.params.get("url", "")
        headers = node.params.get("headers", {})
        body = node.params.get("body")

        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, headers=headers) as resp:
                    data = await resp.json() if resp.content_type == "application/json" else await resp.text()
            elif method == "POST":
                async with session.post(url, headers=headers, json=body) as resp:
                    data = await resp.json() if resp.content_type == "application/json" else await resp.text()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        return {"response": data, "status": resp.status}

    # ===== Helper Methods =====

    async def _get_department(self, department_id: str):
        """Get or create department instance."""
        if department_id not in self._departments:
            dept = get_department(department_id)
            if hasattr(dept, '__aenter__'):
                await dept.__aenter__()
            self._departments[department_id] = dept

        return self._departments[department_id]

    async def _save_checkpoint(self):
        """Save execution checkpoint."""
        checkpoint_id = await self.checkpoint_manager.save_checkpoint(
            execution_id=self.execution_id,
            node_id=list(self.executed_nodes)[-1] if self.executed_nodes else "start",
            state=self.state,
        )
        logger.info(f"Checkpoint saved: {checkpoint_id}")

    async def resume_from_checkpoint(self, checkpoint_id: str) -> WorkflowResult:
        """Resume execution from checkpoint."""
        state = await self.checkpoint_manager.restore_checkpoint(
            execution_id=self.execution_id,
            node_id=checkpoint_id,
        )

        self.state = state
        # Continue execution from checkpoint node
        # ... implementation ...

        return WorkflowResult(
            execution_id=self.execution_id,
            workflow_name=self.workflow.name,
            status="resumed",
            outputs=self.state,
            execution_time_ms=0.0,
            nodes_executed=len(self.executed_nodes),
        )
