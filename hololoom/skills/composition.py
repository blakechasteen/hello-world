"""Skill Composition - Chain skills into complex workflows"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from hololoom.skills.base import SkillOutput, SkillStatus
from hololoom.skills.executor import SkillExecutor, ExecutionResult, BatchResult

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Workflow step types."""
    EXECUTE = "execute"        # Execute single skill
    PARALLEL = "parallel"      # Execute multiple skills in parallel
    CONDITIONAL = "conditional"  # Conditional branch
    LOOP = "loop"             # Repeat steps
    TRANSFORM = "transform"    # Transform data between steps


@dataclass
class WorkflowStep:
    """Single step in workflow."""
    name: str
    step_type: StepType

    # For EXECUTE steps
    skill_name: Optional[str] = None
    operation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    # For PARALLEL steps
    parallel_steps: Optional[List['WorkflowStep']] = None

    # For CONDITIONAL steps
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    if_true: Optional[List['WorkflowStep']] = None
    if_false: Optional[List['WorkflowStep']] = None

    # For LOOP steps
    loop_condition: Optional[Callable[[Dict[str, Any], int], bool]] = None
    loop_steps: Optional[List['WorkflowStep']] = None
    max_iterations: int = 10

    # For TRANSFORM steps
    transform_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    # Metadata
    timeout: Optional[float] = None
    continue_on_failure: bool = False


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    success: bool
    steps_executed: int
    total_time_ms: float
    outputs: Dict[str, Any]  # Named outputs from steps
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Detailed execution trace
    step_results: List[Dict[str, Any]] = field(default_factory=list)


class SkillWorkflow:
    """
    Skill workflow composer - chain multiple skills into complex pipelines.

    Features:
    - Sequential and parallel execution
    - Conditional branching
    - Loops and iterations
    - Data transformation between steps
    - Error handling with partial results
    - Execution tracing
    """

    def __init__(
        self,
        name: str,
        steps: List[WorkflowStep],
        executor: Optional[SkillExecutor] = None
    ):
        """
        Initialize workflow.

        Args:
            name: Workflow name
            steps: List of workflow steps
            executor: Optional SkillExecutor (creates default if None)
        """
        self.name = name
        self.steps = steps
        self.executor = executor or SkillExecutor()

        # Workflow context - shared state across steps
        self._context: Dict[str, Any] = {}

    async def execute(
        self,
        initial_context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> WorkflowResult:
        """
        Execute workflow.

        Args:
            initial_context: Initial context data
            timeout: Overall workflow timeout

        Returns:
            WorkflowResult with outputs and metadata
        """
        start_time = time.time()

        # Initialize context
        self._context = initial_context.copy() if initial_context else {}
        self._context["_workflow_start"] = start_time

        # Execute steps
        steps_executed = 0
        step_results = []
        errors = []
        warnings = []

        try:
            for step in self.steps:
                # Check timeout
                if timeout:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        errors.append(f"Workflow timeout after {elapsed:.2f}s")
                        break

                # Execute step
                step_result = await self._execute_step(step)
                steps_executed += 1
                step_results.append(step_result)

                # Check for errors
                if not step_result.get("success", False):
                    error_msg = step_result.get("error", "Unknown error")
                    errors.append(f"Step '{step.name}' failed: {error_msg}")

                    # Stop if not configured to continue
                    if not step.continue_on_failure:
                        break

                # Collect warnings
                if "warnings" in step_result:
                    warnings.extend(step_result["warnings"])

        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            errors.append(f"Workflow execution error: {e}")

        # Compute final result
        total_time = (time.time() - start_time) * 1000
        success = len(errors) == 0

        # Extract named outputs
        outputs = {
            key: value for key, value in self._context.items()
            if not key.startswith("_")  # Exclude internal keys
        }

        return WorkflowResult(
            success=success,
            steps_executed=steps_executed,
            total_time_ms=total_time,
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            step_results=step_results
        )

    async def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """
        Execute a single workflow step.

        Args:
            step: Step to execute

        Returns:
            Dict with step result
        """
        step_start = time.time()

        try:
            if step.step_type == StepType.EXECUTE:
                result = await self._execute_skill_step(step)

            elif step.step_type == StepType.PARALLEL:
                result = await self._execute_parallel_step(step)

            elif step.step_type == StepType.CONDITIONAL:
                result = await self._execute_conditional_step(step)

            elif step.step_type == StepType.LOOP:
                result = await self._execute_loop_step(step)

            elif step.step_type == StepType.TRANSFORM:
                result = await self._execute_transform_step(step)

            else:
                raise ValueError(f"Unknown step type: {step.step_type}")

            step_time = (time.time() - step_start) * 1000

            return {
                "step_name": step.name,
                "step_type": step.step_type.value,
                "success": result.get("success", True),
                "execution_time_ms": step_time,
                **result
            }

        except Exception as e:
            logger.error(f"Step '{step.name}' error: {e}")
            step_time = (time.time() - step_start) * 1000

            return {
                "step_name": step.name,
                "step_type": step.step_type.value,
                "success": False,
                "error": str(e),
                "execution_time_ms": step_time
            }

    async def _execute_skill_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a skill execution step."""
        # Resolve parameters from context
        parameters = self._resolve_parameters(step.parameters)

        # Execute skill
        result = await self.executor.execute_single(
            skill_name=step.skill_name,
            operation=step.operation,
            parameters=parameters,
            timeout=step.timeout
        )

        # Store result in context
        self._context[step.name] = result.output.result

        return {
            "success": result.output.status == SkillStatus.SUCCESS,
            "output": result.output.result,
            "cache_hit": result.cache_hit,
            "execution_time_ms": result.output.execution_time_ms
        }

    async def _execute_parallel_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute multiple steps in parallel."""
        # Build execution specs from parallel steps
        executions = []

        for sub_step in step.parallel_steps:
            if sub_step.step_type != StepType.EXECUTE:
                raise ValueError(f"Parallel step must contain EXECUTE steps, got {sub_step.step_type}")

            parameters = self._resolve_parameters(sub_step.parameters)
            executions.append({
                "skill_name": sub_step.skill_name,
                "operation": sub_step.operation,
                "parameters": parameters,
                "timeout": sub_step.timeout
            })

        # Execute in parallel
        batch_result = await self.executor.execute_parallel(executions)

        # Store results in context
        for i, sub_step in enumerate(step.parallel_steps):
            exec_result = batch_result.results[i]
            self._context[sub_step.name] = exec_result.output.result

        return {
            "success": batch_result.success_count == len(batch_result.results),
            "parallel_count": len(batch_result.results),
            "success_count": batch_result.success_count,
            "failure_count": batch_result.failure_count,
            "cache_hit_rate": batch_result.cache_hit_rate
        }

    async def _execute_conditional_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute conditional branch."""
        # Evaluate condition
        condition_result = step.condition(self._context)

        # Execute appropriate branch
        if condition_result:
            branch_steps = step.if_true or []
            branch_name = "if_true"
        else:
            branch_steps = step.if_false or []
            branch_name = "if_false"

        # Execute branch steps
        for branch_step in branch_steps:
            await self._execute_step(branch_step)

        return {
            "success": True,
            "condition_result": condition_result,
            "branch_taken": branch_name,
            "steps_executed": len(branch_steps)
        }

    async def _execute_loop_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute loop iteration."""
        iteration = 0
        iterations_executed = 0

        while iteration < step.max_iterations:
            # Check loop condition
            if not step.loop_condition(self._context, iteration):
                break

            # Execute loop steps
            for loop_step in step.loop_steps:
                await self._execute_step(loop_step)

            iteration += 1
            iterations_executed += 1

        return {
            "success": True,
            "iterations_executed": iterations_executed,
            "max_iterations": step.max_iterations
        }

    async def _execute_transform_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute data transformation."""
        # Apply transformation function
        transformed = step.transform_func(self._context)

        # Update context
        self._context.update(transformed)

        return {
            "success": True,
            "keys_added": len(transformed)
        }

    def _resolve_parameters(self, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve parameters from context.

        Supports variable substitution:
        - "${step_name}" -> value from context["step_name"]
        - {"$ref": "step_name"} -> value from context["step_name"]

        Args:
            parameters: Raw parameters

        Returns:
            Resolved parameters
        """
        if parameters is None:
            return {}

        resolved = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # String variable substitution
                var_name = value[2:-1]
                resolved[key] = self._context.get(var_name)

            elif isinstance(value, dict) and "$ref" in value:
                # Reference substitution
                ref_name = value["$ref"]
                resolved[key] = self._context.get(ref_name)

            else:
                # Direct value
                resolved[key] = value

        return resolved


# Workflow builder helpers

def execute_step(
    name: str,
    skill_name: str,
    operation: str,
    parameters: Dict[str, Any],
    **kwargs
) -> WorkflowStep:
    """
    Create an EXECUTE step.

    Args:
        name: Step name
        skill_name: Skill to execute
        operation: Operation name
        parameters: Operation parameters
        **kwargs: Additional step options

    Returns:
        WorkflowStep
    """
    return WorkflowStep(
        name=name,
        step_type=StepType.EXECUTE,
        skill_name=skill_name,
        operation=operation,
        parameters=parameters,
        **kwargs
    )


def parallel_step(
    name: str,
    steps: List[WorkflowStep],
    **kwargs
) -> WorkflowStep:
    """
    Create a PARALLEL step.

    Args:
        name: Step name
        steps: Steps to execute in parallel
        **kwargs: Additional step options

    Returns:
        WorkflowStep
    """
    return WorkflowStep(
        name=name,
        step_type=StepType.PARALLEL,
        parallel_steps=steps,
        **kwargs
    )


def conditional_step(
    name: str,
    condition: Callable[[Dict[str, Any]], bool],
    if_true: List[WorkflowStep],
    if_false: Optional[List[WorkflowStep]] = None,
    **kwargs
) -> WorkflowStep:
    """
    Create a CONDITIONAL step.

    Args:
        name: Step name
        condition: Condition function
        if_true: Steps if condition is True
        if_false: Steps if condition is False
        **kwargs: Additional step options

    Returns:
        WorkflowStep
    """
    return WorkflowStep(
        name=name,
        step_type=StepType.CONDITIONAL,
        condition=condition,
        if_true=if_true,
        if_false=if_false,
        **kwargs
    )


def transform_step(
    name: str,
    transform_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    **kwargs
) -> WorkflowStep:
    """
    Create a TRANSFORM step.

    Args:
        name: Step name
        transform_func: Transformation function
        **kwargs: Additional step options

    Returns:
        WorkflowStep
    """
    return WorkflowStep(
        name=name,
        step_type=StepType.TRANSFORM,
        transform_func=transform_func,
        **kwargs
    )


__all__ = [
    "SkillWorkflow",
    "WorkflowStep",
    "WorkflowResult",
    "StepType",
    "execute_step",
    "parallel_step",
    "conditional_step",
    "transform_step"
]
