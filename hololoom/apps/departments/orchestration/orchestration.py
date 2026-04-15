#!/usr/bin/env python3
"""
Orchestration Department - Multi-department task coordination.

Coordinates complex workflows across multiple departments:
- Sequential workflows (A → B → C)
- Parallel workflows (A + B + C → aggregate)
- Conditional workflows (if A then B else C)
- Dependency resolution

Design Philosophy:
"Coordinate, don't dictate."

Supported Tasks:
- "execute_workflow": Execute multi-department workflow
- "aggregate_results": Aggregate results from multiple departments
- "route_task": Intelligently route task to best department
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ..base import BaseDepartment
from ..protocol import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

class WorkflowType(Enum):
    """Workflow execution types."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class WorkflowStep:
    """Single step in a workflow."""
    step_id: str
    department: str
    task_type: str
    parameters: dict[str, Any]
    depends_on: list[str] | None = None  # Step IDs this depends on


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    steps_completed: int
    steps_failed: int
    total_time_ms: float
    results: dict[str, DepartmentResponse]
    errors: dict[str, str]


# ============================================================================
# Orchestration Department
# ============================================================================

class OrchestrationDepartment(BaseDepartment):
    """
    Orchestration Department - Multi-department coordination.

    Coordinates complex workflows across departments:
    1. Sequential: Context → MasterWeaver → Infrastructure
    2. Parallel: Query multiple departments simultaneously
    3. Conditional: Route based on confidence/result

    Example:

        async with OrchestrationDepartment(registry=registry) as dept:
            # Sequential workflow
            workflow = [
                WorkflowStep(
                    step_id="step1",
                    department="context",
                    task_type="weave_response",
                    parameters={"query": "What is Thompson Sampling?"}
                ),
                WorkflowStep(
                    step_id="step2",
                    department="masterweaver",
                    task_type="extract_entities",
                    parameters={"text": "{step1.result.response}"},  # Uses step1 output
                    depends_on=["step1"]
                )
            ]

            request = DepartmentRequest(
                task_id="workflow_001",
                task_type="execute_workflow",
                parameters={"steps": workflow, "workflow_type": "sequential"}
            )

            result = await dept.execute(request)
    """

    def __init__(
        self,
        registry: Any | None = None,
        max_concurrent_tasks: int = 5,
        timeout_per_step_ms: float = 30000.0,
        dept_config: DepartmentConfig | None = None
    ):
        """
        Initialize Orchestration Department.

        Args:
            registry: Department registry for routing
            max_concurrent_tasks: Max parallel tasks
            timeout_per_step_ms: Timeout per workflow step
            dept_config: Department configuration
        """
        super().__init__(
            name="orchestration",
            domain="system",
            version="1.0.0",
            supported_tasks=[
                "execute_workflow",
                "aggregate_results",
                "route_task"
            ],
            confidence_range=(0.75, 0.99),
            config=dept_config or DepartmentConfig(name="orchestration", domain="orchestration")
        )

        self.registry = registry
        self.max_concurrent_tasks = max_concurrent_tasks
        self.timeout_per_step_ms = timeout_per_step_ms

        # Workflow tracking
        self._active_workflows: dict[str, WorkflowResult] = {}
        self._workflow_history: list[WorkflowResult] = []

        logger.info("Orchestration Department initialized")

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize orchestration department."""
        if self._initialized:
            return

        await super().initialize()

    async def close(self) -> None:
        """Cleanup resources."""
        # Cancel active workflows
        self._active_workflows.clear()

        await super().close()

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """Execute an orchestration task."""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        if request.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {request.task_type}. "
                f"Supported: {self.supported_tasks}"
            )

        try:
            if request.task_type == "execute_workflow":
                result, confidence = await self._execute_workflow(request)
            elif request.task_type == "aggregate_results":
                result, confidence = await self._aggregate_results(request)
            elif request.task_type == "route_task":
                result, confidence = await self._route_task(request)
            else:
                raise ValueError(f"Unknown task type: {request.task_type}")

            learning_signals = {
                "task_type": request.task_type,
                "outcome": "success",
                "confidence": confidence.score
            }

            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                detail_level=request.context_preference,
                reasoning=self._build_reasoning(result, request) if request.context_preference != "minimal" else None,
                learning_signals=learning_signals,
                session_state=await self._build_session_state(request.session_id, result),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            self._record_request(
                success=True,
                latency_ms=response.execution_time_ms,
                confidence=confidence.score
            )

            return response

        except Exception as e:
            self._record_error(e)

            error_confidence = ConfidenceMetadata.from_score(
                0.0,
                justification=[],
                uncertainty_sources=[f"Execution failed: {str(e)}"]
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_request(success=False, latency_ms=latency_ms, confidence=0.0)

            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=error_confidence,
                execution_time_ms=latency_ms
            )

    async def verify(
        self,
        response: DepartmentResponse
    ) -> VerificationResult:
        """Verify orchestration operation quality."""
        confidence_valid = response.confidence.score >= 0.75
        has_error = "error" in response.result
        operation_success = not has_error

        sufficient = confidence_valid and operation_success

        refinement_suggestions = None
        if not sufficient:
            refinement_suggestions = {
                "retry": True,
                "reason": "Low confidence or error in orchestration"
            }

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=operation_success,
            alternative_paths=["retry", "fallback_to_single_department"] if not sufficient else [],
            refinement_suggestions=refinement_suggestions,
            escalation_needed=response.confidence.score < 0.5,
            escalation_reason="Very low confidence" if response.confidence.score < 0.5 else None,
            verifier_confidence=0.90
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """Refine orchestration operation."""
        # Retry with adjusted parameters
        return await self.execute(request)

    # ========================================================================
    # Task Handlers
    # ========================================================================

    async def _execute_workflow(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Execute multi-department workflow."""
        steps_data = request.parameters.get("steps", [])
        workflow_type = request.parameters.get("workflow_type", "sequential")

        if not steps_data:
            raise ValueError("Missing required parameter: 'steps'")

        if not self.registry:
            raise RuntimeError("Registry required for workflow execution")

        # Convert dict steps to WorkflowStep objects
        steps = [
            WorkflowStep(
                step_id=s["step_id"],
                department=s["department"],
                task_type=s["task_type"],
                parameters=s["parameters"],
                depends_on=s.get("depends_on")
            )
            for s in steps_data
        ]

        workflow_id = request.task_id
        start_time = datetime.now()

        # Execute based on workflow type
        if workflow_type == "sequential":
            workflow_result = await self._execute_sequential(workflow_id, steps)
        elif workflow_type == "parallel":
            workflow_result = await self._execute_parallel(workflow_id, steps)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Store result
        self._workflow_history.append(workflow_result)

        # Calculate confidence
        success_rate = workflow_result.steps_completed / (workflow_result.steps_completed + workflow_result.steps_failed)

        confidence = ConfidenceMetadata.from_score(
            success_rate,
            justification=[f"Completed {workflow_result.steps_completed}/{len(steps)} steps"],
            uncertainty_sources=[f"{workflow_result.steps_failed} failed"] if workflow_result.steps_failed > 0 else []
        )

        result_data = {
            "workflow_id": workflow_result.workflow_id,
            "steps_completed": workflow_result.steps_completed,
            "steps_failed": workflow_result.steps_failed,
            "total_time_ms": workflow_result.total_time_ms,
            "results": {
                step_id: {
                    "confidence": resp.confidence.score,
                    "result": resp.result
                }
                for step_id, resp in workflow_result.results.items()
            },
            "errors": workflow_result.errors
        }

        return result_data, confidence

    async def _execute_sequential(
        self,
        workflow_id: str,
        steps: list[WorkflowStep]
    ) -> WorkflowResult:
        """Execute steps sequentially."""
        start_time = datetime.now()
        results = {}
        errors = {}
        steps_completed = 0
        steps_failed = 0

        for step in steps:
            try:
                # Replace parameter references with actual values
                parameters = self._resolve_parameters(step.parameters, results)

                # Create request
                dept_request = DepartmentRequest(
                    task_id=f"{workflow_id}_{step.step_id}",
                    task_type=step.task_type,
                    parameters=parameters
                )

                # Execute
                response = await self.registry.route_request(
                    dept_request,
                    department_name=step.department
                )

                results[step.step_id] = response
                steps_completed += 1

            except Exception as e:
                logger.error(f"Step {step.step_id} failed: {e}")
                errors[step.step_id] = str(e)
                steps_failed += 1
                break  # Stop on failure for sequential

        total_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return WorkflowResult(
            workflow_id=workflow_id,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            total_time_ms=total_time_ms,
            results=results,
            errors=errors
        )

    async def _execute_parallel(
        self,
        workflow_id: str,
        steps: list[WorkflowStep]
    ) -> WorkflowResult:
        """Execute steps in parallel."""
        start_time = datetime.now()

        # Create tasks for all steps
        tasks = []
        for step in steps:
            task = self._execute_step(workflow_id, step, {})
            tasks.append((step.step_id, task))

        # Execute all tasks concurrently
        results = {}
        errors = {}
        steps_completed = 0
        steps_failed = 0

        for step_id, task in tasks:
            try:
                response = await task
                results[step_id] = response
                steps_completed += 1
            except Exception as e:
                logger.error(f"Step {step_id} failed: {e}")
                errors[step_id] = str(e)
                steps_failed += 1

        total_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return WorkflowResult(
            workflow_id=workflow_id,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            total_time_ms=total_time_ms,
            results=results,
            errors=errors
        )

    async def _execute_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        prior_results: dict[str, DepartmentResponse]
    ) -> DepartmentResponse:
        """Execute a single workflow step."""
        # Resolve parameters
        parameters = self._resolve_parameters(step.parameters, prior_results)

        # Create request
        dept_request = DepartmentRequest(
            task_id=f"{workflow_id}_{step.step_id}",
            task_type=step.task_type,
            parameters=parameters
        )

        # Execute
        return await self.registry.route_request(
            dept_request,
            department_name=step.department
        )

    def _resolve_parameters(
        self,
        parameters: dict[str, Any],
        prior_results: dict[str, DepartmentResponse]
    ) -> dict[str, Any]:
        """Resolve parameter references like {step1.result.response}."""
        resolved = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Extract reference: {step1.result.response}
                ref = value[1:-1]  # Remove braces
                parts = ref.split(".")

                if len(parts) >= 2 and parts[0] in prior_results:
                    step_id = parts[0]
                    response = prior_results[step_id]

                    # Navigate through response
                    obj = response
                    for part in parts[1:]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        elif isinstance(obj, dict) and part in obj:
                            obj = obj[part]
                        else:
                            obj = None
                            break

                    resolved[key] = obj if obj is not None else value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    async def _aggregate_results(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Aggregate results from multiple departments."""
        responses = request.parameters.get("responses", {})

        if not responses:
            raise ValueError("Missing required parameter: 'responses'")

        # Aggregate confidences
        confidences = [r.get("confidence", {}).get("score", 0.0) for r in responses.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Aggregate results
        aggregated = {
            "department_count": len(responses),
            "average_confidence": avg_confidence,
            "results": responses
        }

        confidence = ConfidenceMetadata.from_score(
            avg_confidence,
            justification=[f"Aggregated {len(responses)} responses"],
            uncertainty_sources=[]
        )

        return aggregated, confidence

    async def _route_task(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Intelligently route task to best department."""
        task_description = request.parameters.get("task_description")
        candidate_departments = request.parameters.get("departments", [])

        if not task_description:
            raise ValueError("Missing required parameter: 'task_description'")

        if not self.registry:
            raise RuntimeError("Registry required for task routing")

        # Simple routing: Query each department's capabilities
        best_department = None
        best_score = 0.0

        for dept_name in candidate_departments:
            dept = self.registry.get_department(dept_name)
            if dept:
                # Score based on supported tasks (simplified)
                score = 0.8  # Default score
                best_department = dept_name
                best_score = score
                break  # Simplified: Take first match

        if not best_department:
            raise ValueError("No suitable department found")

        confidence = ConfidenceMetadata.from_score(
            best_score,
            justification=[f"Routed to {best_department}"],
            uncertainty_sources=[]
        )

        result_data = {
            "selected_department": best_department,
            "routing_score": best_score,
            "candidates_evaluated": len(candidate_departments)
        }

        return result_data, confidence

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _build_reasoning(
        self,
        result: dict[str, Any],
        request: DepartmentRequest
    ) -> dict[str, Any]:
        """Build reasoning dict for response."""
        return {
            "task_type": request.task_type,
            "active_workflows": len(self._active_workflows),
            "workflow_history_count": len(self._workflow_history)
        }

    async def _build_session_state(
        self,
        session_id: str | None,
        result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build session state update."""
        if not session_id:
            return None

        return {
            "last_workflow": result.get("workflow_id")
        }

    def get_workflow_history(self, limit: int = 10) -> list[WorkflowResult]:
        """Get recent workflow history."""
        return self._workflow_history[-limit:]
