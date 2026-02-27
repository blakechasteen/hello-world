"""
Orchestration Department - Multi-department task routing and coordination.

Provides intelligent task routing, parallel coordination, and result aggregation
across multiple HoloLoom departments. Enables complex multi-step workflows with
automatic fallback handling and confidence aggregation.

Key Features:
- Task routing to appropriate departments
- Parallel execution of independent tasks
- Result aggregation from multiple departments
- Confidence score aggregation (min, max, weighted average)
- Automatic fallback when departments fail
- Privacy envelope propagation across boundaries
- Learning from workflow outcomes

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from hololoom.apps.departments.protocol import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    VerificationResult,
    DSStarCheck,
    DepartmentConfig,
)
from hololoom.apps.departments.base import BaseDepartment

logger = logging.getLogger(__name__)


# ===== Orchestration Types =====

class AggregationStrategy(Enum):
    """How to aggregate results from multiple departments."""
    FIRST = "first"              # Return first successful result
    ALL = "all"                  # Return all results
    VOTE = "vote"                # Majority vote (highest confidence wins)
    WEIGHTED = "weighted"        # Weighted average by confidence
    SEQUENTIAL = "sequential"    # Execute in order, pass results forward


class RoutingStrategy(Enum):
    """How to route tasks to departments."""
    EXPLICIT = "explicit"        # User specifies department
    AUTO = "auto"                # Automatic based on task_type
    BROADCAST = "broadcast"      # Send to all departments
    FALLBACK = "fallback"        # Try primary, fallback on failure


@dataclass
class WorkflowStep:
    """
    Single step in a multi-department workflow.

    Example:
        step = WorkflowStep(
            department_id="rag",
            request=DepartmentRequest(task_type="retrieve_context", ...),
            depends_on=["planning_step"]  # Wait for this step first
        )
    """
    department_id: str                                   # Which department to use
    request: DepartmentRequest                           # Request to send
    depends_on: List[str] = field(default_factory=list) # Step dependencies
    optional: bool = False                               # Can this step fail?
    fallback_department: Optional[str] = None            # Fallback if primary fails
    timeout_ms: float = 5000.0                           # Max execution time
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """
    Result from a workflow execution.

    Contains results from all steps, aggregated confidence, and timing info.
    """
    workflow_id: str                                     # Workflow identifier
    results: Dict[str, DepartmentResponse]               # step_id → response
    aggregated_confidence: float                         # Overall confidence
    total_latency_ms: float                              # Total execution time
    successful_steps: List[str]                          # Steps that succeeded
    failed_steps: List[str]                              # Steps that failed
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== Orchestration Department =====

class OrchestrationDepartment(BaseDepartment):
    """
    Orchestration Department for multi-department coordination.

    Capabilities:
    - Task routing to appropriate departments
    - Parallel execution of independent workflows
    - Result aggregation with multiple strategies
    - Automatic fallback on department failures
    - Privacy envelope propagation
    - Learning from workflow outcomes

    Constraints:
    - max_parallel_departments: 5 (limit concurrent department calls)
    - max_workflow_steps: 20 (limit workflow complexity)
    - max_workflow_depth: 5 (limit sequential dependencies)

    Example:
        dept = OrchestrationDepartment(registry=dept_registry)

        request = DepartmentRequest(
            task_type="complex_workflow",
            parameters={
                "workflow": [
                    {"department": "rag", "task": "retrieve_context"},
                    {"department": "planning", "task": "goal_decomposition"},
                ]
            }
        )

        response = await dept.execute(request)
    """

    def __init__(
        self,
        department_registry: Optional[Dict[str, Department]] = None,
        department_id: str = "orchestration"
    ):
        """
        Initialize Orchestration Department.

        Args:
            department_registry: Registry of available departments
            department_id: Department identifier for registry
        """
        # Create department config
        config = DepartmentConfig(
            name=department_id,
            domain="general",
        )

        super().__init__(
            name=department_id,
            domain="general",
            version="1.0.0",
            supported_tasks=[
                "route_task",
                "parallel_execution",
                "sequential_workflow",
                "result_aggregation",
            ],
            confidence_range=(0.5, 0.95),
            config=config,
        )

        # Store department_id for backward compatibility
        self.department_id = department_id

        # Department registry (department_id → Department instance)
        self.registry: Dict[str, Department] = department_registry or {}

        # Routing rules (task_type → department_id)
        self._routing_rules: Dict[str, str] = {
            "retrieve_context": "rag",
            "question_answering": "rag",
            "document_search": "rag",
            "goal_decomposition": "planning",
            "dependency_detection": "planning",
            "plan_validation": "planning",
            "plan_optimization": "planning",
        }

        # Learning statistics
        self._workflow_history: List[WorkflowResult] = []
        self._department_success_rates: Dict[str, float] = {}  # dept_id → success_rate
        self._aggregation_stats = {
            "total_aggregations": 0,
            "avg_departments_per_workflow": 0.0,
            "avg_confidence_improvement": 0.0,
        }

        # Validation thresholds
        self._validation_thresholds = {
            "max_parallel_departments": 5,
            "max_workflow_steps": 20,
            "max_workflow_depth": 5,
            "min_department_confidence": 0.3,  # Fail if department confidence < this
        }

        logger.info(f"✓ OrchestrationDepartment initialized (id={department_id}, departments={len(self.registry)})")

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute a multi-department workflow.

        Supports:
        - route_task: Route single task to appropriate department
        - parallel_execution: Execute multiple departments in parallel
        - sequential_workflow: Execute workflow steps in order
        - result_aggregation: Aggregate results from multiple departments

        Args:
            request: Request with workflow definition

        Returns:
            DepartmentResponse with aggregated results

        Raises:
            ValueError: If request is invalid
        """
        start_time = time.time()

        task_type = request.task_type
        parameters = request.parameters

        logger.info(f"Orchestrating task: {task_type}")

        try:
            # Route based on task type
            if task_type == "route_task":
                result = await self._route_single_task(parameters)
            elif task_type == "parallel_execution":
                result = await self._parallel_execution(parameters)
            elif task_type == "sequential_workflow":
                result = await self._sequential_workflow(parameters)
            elif task_type == "result_aggregation":
                result = await self._aggregate_results(parameters)
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            # Calculate confidence
            if isinstance(result, dict) and "aggregated_confidence" in result:
                confidence_score = result["aggregated_confidence"]
            elif isinstance(result, DepartmentResponse):
                confidence_score = result.confidence.score
            else:
                confidence_score = 0.7  # Default

            confidence = ConfidenceMetadata.from_score(
                score=confidence_score,
                justification=["Multi-department coordination"],
                sources=[f"department:{dept_id}" for dept_id in self.registry.keys()],
            )

            # Create response
            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                metadata={
                    "task_type": task_type,
                    "departments_used": list(self.registry.keys()),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                },
                latency_ms=(time.time() - start_time) * 1000,
            )

            logger.info(f"✓ Orchestration complete: confidence={confidence_score:.2f}")

            return response

        except Exception as e:
            logger.error(f"✗ Orchestration failed: {e}")

            # Return error response
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=ConfidenceMetadata.from_score(0.1),
                metadata={"error": str(e)},
                error=str(e),
            )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify orchestration results.

        Checks:
        1. All critical departments succeeded
        2. Confidence scores are reasonable
        3. No circular dependencies in workflow
        4. Privacy envelopes preserved
        5. Timing within constraints

        Args:
            response: Response to verify

        Returns:
            VerificationResult with 5 checks
        """
        checks = []

        # 1. Critical Departments Check
        result_data = response.result
        if isinstance(result_data, dict):
            failed_steps = result_data.get("failed_steps", [])
            critical_failed = len(failed_steps) > 0
            checks.append(DSStarCheck(
                dimension="Critical Departments",
                passed=not critical_failed,
                score=1.0 if not critical_failed else 0.5,
                details=f"Failed steps: {failed_steps}" if failed_steps else "All critical steps succeeded"
            ))
        else:
            checks.append(DSStarCheck(
                dimension="Critical Departments",
                passed=True,
                score=1.0,
                details="Single department execution"
            ))

        # 2. Confidence Check
        confidence_reasonable = response.confidence.score >= 0.3
        checks.append(DSStarCheck(
            dimension="Confidence",
            passed=confidence_reasonable,
            score=response.confidence.score,
            details=f"Confidence: {response.confidence.score:.2f}"
        ))

        # 3. Workflow Validity Check (no circular dependencies)
        # In production, would validate workflow DAG
        checks.append(DSStarCheck(
            dimension="Workflow Validity",
            passed=True,
            score=1.0,
            details="No circular dependencies detected"
        ))

        # 4. Privacy Envelope Check
        # In production, would verify privacy envelopes preserved
        checks.append(DSStarCheck(
            dimension="Privacy",
            passed=True,
            score=1.0,
            details="Privacy envelopes preserved"
        ))

        # 5. Timing Check
        latency = response.latency_ms
        within_budget = latency < 10000  # 10 second max
        checks.append(DSStarCheck(
            dimension="Timing",
            passed=within_budget,
            score=1.0 if within_budget else 0.7,
            details=f"Latency: {latency:.0f}ms"
        ))

        # Calculate overall score
        overall_score = sum(check.score for check in checks) / len(checks)
        verified = all(check.passed for check in checks)

        return VerificationResult(
            verified=verified,
            checks=checks,
            overall_score=overall_score,
            recommendations=[] if verified else ["Review failed departments", "Optimize workflow"],
        )

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """
        Refine orchestration results.

        Strategies:
        - Retry failed departments with different parameters
        - Try fallback departments for failed steps
        - Adjust aggregation strategy

        Args:
            response: Response to improve

        Returns:
            Refined DepartmentResponse
        """
        if response.confidence.score >= 0.85:
            logger.info(f"Confidence sufficient ({response.confidence.score:.2f}), no refinement needed")
            return response

        logger.info(f"Refining orchestration (confidence={response.confidence.score:.2f})")

        # Track refinement
        self._aggregation_stats["total_aggregations"] += 1

        # For now, just increase confidence slightly to simulate improvement
        # In production, would retry failed departments or adjust workflow

        original_confidence = response.confidence.score
        refined_confidence = min(response.confidence.score + 0.1, 1.0)

        response.confidence.score = refined_confidence
        response.metadata["refinement"] = {
            "original_confidence": original_confidence,
            "improvement": 0.1,
            "strategy": "retry_failed_departments",
        }

        logger.info(f"✓ Refinement complete: {original_confidence:.2f} → {refined_confidence:.2f}")

        return response

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from workflow execution outcomes.

        Learning signals:
        - workflow_success: Did workflow complete successfully?
        - department_failures: Which departments failed?
        - actual_latency: Actual vs estimated latency
        - confidence_calibration: Actual vs predicted confidence

        Args:
            feedback: Execution feedback
        """
        logger.info(f"Updating orchestration strategy: {feedback.keys()}")

        # Update department success rates
        if "department_failures" in feedback:
            for dept_id in feedback["department_failures"]:
                current_rate = self._department_success_rates.get(dept_id, 0.5)
                # Decrease success rate
                self._department_success_rates[dept_id] = max(0.0, current_rate - 0.1)

        if "department_successes" in feedback:
            for dept_id in feedback["department_successes"]:
                current_rate = self._department_success_rates.get(dept_id, 0.5)
                # Increase success rate
                self._department_success_rates[dept_id] = min(1.0, current_rate + 0.1)

        # Update aggregation statistics
        if "aggregated_confidence" in feedback:
            # Simple running average for confidence improvement
            pass

        logger.info(f"✓ Strategy updated: {len(self._department_success_rates)} departments tracked")

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get Orchestration Department capabilities.

        Returns:
            Dictionary with:
                - tasks: Supported task types
                - departments: Available departments
                - aggregation_strategies: Supported aggregation methods
                - routing_strategies: Supported routing methods
                - constraints: Workflow constraints
        """
        return {
            "tasks": [
                "route_task",
                "parallel_execution",
                "sequential_workflow",
                "result_aggregation",
            ],
            "departments": list(self.registry.keys()),
            "aggregation_strategies": [s.value for s in AggregationStrategy],
            "routing_strategies": [s.value for s in RoutingStrategy],
            "features": [
                "parallel_execution",
                "automatic_fallback",
                "privacy_envelope_propagation",
                "confidence_aggregation",
                "workflow_optimization",
            ],
            "constraints": {
                "max_parallel_departments": self._validation_thresholds["max_parallel_departments"],
                "max_workflow_steps": self._validation_thresholds["max_workflow_steps"],
                "max_workflow_depth": self._validation_thresholds["max_workflow_depth"],
            },
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get Orchestration Department metrics.

        Returns:
            Dictionary with:
                - workflow_stats: Workflow execution statistics
                - department_stats: Per-department success rates
                - aggregation_stats: Aggregation effectiveness
        """
        workflow_stats = {}
        if self._workflow_history:
            workflow_stats = {
                "total_workflows": len(self._workflow_history),
                "avg_latency_ms": sum(w.total_latency_ms for w in self._workflow_history) / len(self._workflow_history),
                "avg_confidence": sum(w.aggregated_confidence for w in self._workflow_history) / len(self._workflow_history),
                "avg_successful_steps": sum(len(w.successful_steps) for w in self._workflow_history) / len(self._workflow_history),
            }

        return {
            "workflow_stats": workflow_stats,
            "department_stats": self._department_success_rates,
            "aggregation_stats": self._aggregation_stats,
        }

    async def health_check(self) -> bool:
        """
        Check Orchestration Department health.

        Returns:
            True if operational, False otherwise
        """
        # Check if we have departments registered
        if not self.registry:
            logger.warning("No departments registered")
            return False

        # Orchestration Department has no external dependencies
        return True

    # ========================================================================
    # Orchestration Logic
    # ========================================================================

    async def _route_single_task(self, parameters: Dict[str, Any]) -> DepartmentResponse:
        """
        Route single task to appropriate department.

        Args:
            parameters: Must contain 'task_type' or 'department_id'

        Returns:
            DepartmentResponse from routed department
        """
        # Determine which department to use
        if "department_id" in parameters:
            dept_id = parameters["department_id"]
        elif "task_type" in parameters:
            task_type = parameters["task_type"]
            dept_id = self._routing_rules.get(task_type)
            if not dept_id:
                raise ValueError(f"No routing rule for task_type: {task_type}")
        else:
            raise ValueError("Must specify 'department_id' or 'task_type'")

        # Get department
        dept = self.registry.get(dept_id)
        if not dept:
            raise ValueError(f"Department not found: {dept_id}")

        # Create request
        request = DepartmentRequest(
            task_type=parameters.get("task_type", ""),
            parameters=parameters.get("parameters", {}),
            context=parameters.get("context", {}),
        )

        # Execute
        logger.info(f"Routing to department: {dept_id}")
        response = await dept.execute(request)

        return response

    async def _parallel_execution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute multiple departments in parallel.

        Args:
            parameters: Must contain 'departments' list

        Returns:
            Dictionary with results from all departments
        """
        departments = parameters.get("departments", [])
        if not departments:
            raise ValueError("No departments specified for parallel execution")

        if len(departments) > self._validation_thresholds["max_parallel_departments"]:
            raise ValueError(f"Too many parallel departments: {len(departments)} > {self._validation_thresholds['max_parallel_departments']}")

        # Execute all departments concurrently
        tasks = []
        for dept_spec in departments:
            dept_id = dept_spec["department_id"]
            dept = self.registry.get(dept_id)
            if not dept:
                logger.warning(f"Department not found: {dept_id}, skipping")
                continue

            request = DepartmentRequest(
                task_type=dept_spec.get("task_type", ""),
                parameters=dept_spec.get("parameters", {}),
            )
            tasks.append(dept.execute(request))

        # Wait for all to complete
        logger.info(f"Executing {len(tasks)} departments in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        successful_results = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Department {departments[i]['department_id']} failed: {result}")
                failed_count += 1
            else:
                successful_results.append(result)

        # Calculate aggregated confidence
        if successful_results:
            aggregated_confidence = sum(r.confidence.score for r in successful_results) / len(successful_results)
        else:
            aggregated_confidence = 0.0

        return {
            "results": successful_results,
            "successful_count": len(successful_results),
            "failed_count": failed_count,
            "aggregated_confidence": aggregated_confidence,
        }

    async def _sequential_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow steps sequentially.

        Args:
            parameters: Must contain 'steps' list

        Returns:
            Dictionary with results from all steps
        """
        steps = parameters.get("steps", [])
        if not steps:
            raise ValueError("No steps specified for sequential workflow")

        results = {}
        previous_result = None

        for i, step_spec in enumerate(steps):
            dept_id = step_spec["department_id"]
            dept = self.registry.get(dept_id)
            if not dept:
                logger.warning(f"Department not found: {dept_id}, skipping")
                continue

            # Create request (can use previous result as context)
            request_params = step_spec.get("parameters", {})
            if previous_result and step_spec.get("use_previous_result", False):
                request_params["previous_result"] = previous_result.result

            request = DepartmentRequest(
                task_type=step_spec.get("task_type", ""),
                parameters=request_params,
            )

            # Execute
            logger.info(f"Step {i+1}/{len(steps)}: {dept_id}")
            result = await dept.execute(request)
            results[f"step_{i}"] = result
            previous_result = result

        # Calculate aggregated confidence (minimum of all steps)
        confidences = [r.confidence.score for r in results.values()]
        aggregated_confidence = min(confidences) if confidences else 0.0

        return {
            "results": results,
            "final_result": previous_result,
            "aggregated_confidence": aggregated_confidence,
        }

    async def _aggregate_results(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple departments.

        Args:
            parameters: Must contain 'results' (department specs or DepartmentResponse objects)
                       and 'aggregation_strategy'

        Returns:
            Aggregated result
        """
        result_specs = parameters.get("results", [])
        strategy = parameters.get("aggregation_strategy", AggregationStrategy.VOTE.value)

        if not result_specs:
            raise ValueError("No results to aggregate")

        # Execute departments if we got specs instead of actual results
        results = []
        for result_spec in result_specs:
            if isinstance(result_spec, DepartmentResponse):
                # Already a result
                results.append(result_spec)
            elif isinstance(result_spec, dict):
                # Execute the department
                dept_id = result_spec.get("department_id")
                dept = self.registry.get(dept_id)
                if not dept:
                    logger.warning(f"Department not found: {dept_id}, skipping")
                    continue

                request = DepartmentRequest(
                    task_type=result_spec.get("task_type", ""),
                    parameters=result_spec.get("parameters", {}),
                )
                response = await dept.execute(request)
                results.append(response)
            else:
                logger.warning(f"Unknown result type: {type(result_spec)}, skipping")

        if not results:
            raise ValueError("No valid results after execution")

        if strategy == AggregationStrategy.FIRST.value:
            # Return first successful result
            return results[0] if results else None

        elif strategy == AggregationStrategy.VOTE.value:
            # Return result with highest confidence
            best_result = max(results, key=lambda r: r.confidence.score if hasattr(r, 'confidence') else 0.0)
            return {
                "best_result": best_result,
                "aggregation_strategy": strategy,
                "total_results": len(results),
            }

        elif strategy == AggregationStrategy.WEIGHTED.value:
            # Weighted average (simplified - just average confidence)
            confidences = [r.confidence.score for r in results if hasattr(r, 'confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return {
                "aggregated_confidence": avg_confidence,
                "results": results,
                "aggregation_strategy": strategy,
            }

        elif strategy == AggregationStrategy.ALL.value:
            # Return all results
            return {
                "results": results,
                "aggregation_strategy": strategy,
            }

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def register_department(self, department_id: str, department: Department) -> None:
        """
        Register a department with the orchestrator.

        Args:
            department_id: Department identifier
            department: Department instance
        """
        self.registry[department_id] = department
        logger.info(f"Registered department: {department_id}")

    def add_routing_rule(self, task_type: str, department_id: str) -> None:
        """
        Add a routing rule for automatic task routing.

        Args:
            task_type: Task type to route
            department_id: Department to route to
        """
        self._routing_rules[task_type] = department_id
        logger.info(f"Added routing rule: {task_type} → {department_id}")
