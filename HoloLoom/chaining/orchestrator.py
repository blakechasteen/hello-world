"""
Chain Orchestrator - Execute declarative chains

Executes Chain objects with:
- Sequential step processing
- Context passing between steps
- Conditional branching
- Loop support
- Error handling and rollback
- Complete execution tracing

Author: HoloLoom Architecture Team
Date: November 2025
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

from HoloLoom.protocols.department import DepartmentRequest, DepartmentResponse
from .chain import Chain, ChainStep, StepType
from .types import (
    StepResult,
    StepStatus,
    ExecutionContext,
    RollbackPoint,
    ChainExecutionStats,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTrace:
    """Trace of chain execution for debugging."""
    step_results: List[StepResult]
    final_context: Dict[str, Any]
    total_duration_ms: float
    steps_executed: int
    errors: List[str]

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Execution Trace")
        lines.append(f"{'='*60}")
        lines.append(f"Steps: {self.steps_executed} (errors: {len(self.errors)})")
        lines.append(f"Duration: {self.total_duration_ms:.1f}ms")
        lines.append("")

        for result in self.step_results:
            status_icon = {
                StepStatus.SUCCESS: "✓",
                StepStatus.FAILED: "✗",
                StepStatus.SKIPPED: "○",
                StepStatus.CONDITIONAL_BRANCH: "→",
            }.get(result.status, "?")

            lines.append(
                f"{status_icon} {result.step_id:20s} [{result.step_type:15s}] "
                f"{result.duration_ms:6.1f}ms"
            )

            if result.error:
                lines.append(f"  Error: {result.error}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


@dataclass
class ChainResult:
    """Result of chain execution."""
    success: bool
    final_response: Optional[Any] = None
    confidence: Optional[float] = None
    trace: Optional[ExecutionTrace] = None
    error: Optional[str] = None
    stats: Optional[ChainExecutionStats] = None

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ChainResult(success={self.success}, "
            f"confidence={self.confidence:.2f if self.confidence else None}, "
            f"error={self.error})"
        )


class ChainOrchestrator:
    """Execute declarative chains."""

    def __init__(self, department, enable_tracing: bool = True, enable_rollback: bool = False):
        """
        Initialize orchestrator.

        Args:
            department: Department implementing DepartmentProtocol
            enable_tracing: Track execution trace
            enable_rollback: Enable rollback on failure
        """
        self.department = department
        self.enable_tracing = enable_tracing
        self.enable_rollback = enable_rollback
        self.logger = logger

    async def execute_chain(
        self,
        chain: Chain,
        initial_input: Any,
        max_total_steps: int = 100,
    ) -> ChainResult:
        """
        Execute a complete chain.

        Args:
            chain: Chain to execute
            initial_input: Input to first step
            max_total_steps: Safety limit on total steps

        Returns:
            ChainResult with final response and execution trace
        """
        # Validate chain
        validation_errors = chain.validate()
        if validation_errors:
            error_msg = f"Chain validation failed: {validation_errors}"
            self.logger.error(error_msg)
            return ChainResult(
                success=False,
                error=error_msg,
                trace=None,
            )

        # Initialize execution context
        context = ExecutionContext(initial_input=initial_input)
        step_results: List[StepResult] = []
        rollback_stack: List[RollbackPoint] = []
        stats = ChainExecutionStats()

        start_time = time.time()

        try:
            current_step_id = chain.entry_point
            steps_executed = 0

            while current_step_id and steps_executed < max_total_steps:
                if current_step_id not in chain.steps:
                    error = f"Step '{current_step_id}' not found in chain"
                    self.logger.error(error)
                    return ChainResult(
                        success=False,
                        error=error,
                        trace=self._build_trace(step_results, context, start_time),
                        stats=stats,
                    )

                step = chain.steps[current_step_id]
                stats.total_steps += 1

                # Check skip condition
                if step.skip_condition and step.skip_condition(context.shared_state):
                    self.logger.info(f"Skipping step '{current_step_id}' (skip_condition=true)")
                    result = StepResult(
                        step_id=current_step_id,
                        step_type=step.step_type.value,
                        status=StepStatus.SKIPPED,
                    )
                    step_results.append(result)
                    stats.skipped_steps += 1

                    # Move to next step
                    if step.next_step:
                        current_step_id = step.next_step
                    elif step.on_success:
                        current_step_id = step.on_success
                    else:
                        current_step_id = None
                    continue

                # Save rollback point before executing
                if self.enable_rollback:
                    rollback_stack.append(RollbackPoint(
                        step_id=current_step_id,
                        context_snapshot=context.shared_state.copy(),
                    ))

                # Execute step with retries
                result = await self._execute_step_with_retries(
                    step_id=current_step_id,
                    step=step,
                    context=context,
                    retry_count=step.retry_count,
                )

                step_results.append(result)
                stats.completed_steps += 1 if result.status == StepStatus.SUCCESS else 0
                stats.failed_steps += 1 if result.status == StepStatus.FAILED else 0

                # Store step output
                if result.status == StepStatus.SUCCESS:
                    context.set_step_output(current_step_id, result.output)

                # Determine next step
                next_step_id = self._get_next_step(
                    step=step,
                    result=result,
                    context=context,
                )

                if next_step_id:
                    self.logger.info(f"✓ Step '{current_step_id}' complete → '{next_step_id}'")
                else:
                    self.logger.info(f"✓ Step '{current_step_id}' complete → [end]")

                current_step_id = next_step_id
                steps_executed += 1

            # Extract final response and confidence
            final_response = context.get_step_output(chain.entry_point)
            if not final_response:
                # Try to get from last execute/verify step
                for result in reversed(step_results):
                    if result.step_type in ["execute", "verify"]:
                        final_response = result.output
                        break

            confidence = None
            if isinstance(final_response, DepartmentResponse):
                confidence = final_response.confidence.score
            elif isinstance(final_response, dict) and "confidence" in final_response:
                confidence = final_response.get("confidence", {}).get("score") if isinstance(final_response["confidence"], dict) else final_response["confidence"]

            stats.end_time = datetime.utcnow()
            stats.total_duration_ms = (time.time() - start_time) * 1000

            return ChainResult(
                success=True,
                final_response=final_response,
                confidence=confidence,
                trace=self._build_trace(step_results, context, start_time) if self.enable_tracing else None,
                stats=stats,
            )

        except Exception as e:
            self.logger.error(f"Chain execution failed: {e}", exc_info=True)

            stats.end_time = datetime.utcnow()
            stats.total_duration_ms = (time.time() - start_time) * 1000

            return ChainResult(
                success=False,
                error=str(e),
                trace=self._build_trace(step_results, context, start_time) if self.enable_tracing else None,
                stats=stats,
            )

    async def _execute_step_with_retries(
        self,
        step_id: str,
        step: ChainStep,
        context: ExecutionContext,
        retry_count: int = 0,
    ) -> StepResult:
        """Execute a step with automatic retries."""
        for attempt in range(max(1, retry_count + 1)):
            try:
                result = await self._execute_step(
                    step_id=step_id,
                    step=step,
                    context=context,
                )
                return result
            except Exception as e:
                if attempt < retry_count:
                    self.logger.warning(
                        f"Step '{step_id}' attempt {attempt + 1} failed: {e}. "
                        f"Retrying ({retry_count - attempt} remaining)..."
                    )
                    await asyncio.sleep(0.5)  # Brief delay before retry
                else:
                    # Final attempt failed
                    context.errors.append(str(e))
                    return StepResult(
                        step_id=step_id,
                        step_type=step.step_type.value,
                        status=StepStatus.FAILED,
                        error=str(e),
                    )

    async def _execute_step(
        self,
        step_id: str,
        step: ChainStep,
        context: ExecutionContext,
    ) -> StepResult:
        """Execute a single step."""
        start_time = time.time()

        # Apply timeout if configured
        if step.timeout_seconds:
            execute_func = asyncio.wait_for(
                self._execute_step_internal(step_id, step, context),
                timeout=step.timeout_seconds,
            )
        else:
            execute_func = self._execute_step_internal(step_id, step, context)

        try:
            output = await execute_func

            return StepResult(
                step_id=step_id,
                step_type=step.step_type.value,
                status=StepStatus.SUCCESS,
                output=output,
                duration_ms=(time.time() - start_time) * 1000,
            )

        except asyncio.TimeoutError:
            error = f"Step '{step_id}' exceeded timeout ({step.timeout_seconds}s)"
            self.logger.error(error)
            return StepResult(
                step_id=step_id,
                step_type=step.step_type.value,
                status=StepStatus.FAILED,
                error=error,
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_step_internal(
        self,
        step_id: str,
        step: ChainStep,
        context: ExecutionContext,
    ) -> Any:
        """Internal step execution logic."""
        self.logger.info(f"→ Executing step '{step_id}' ({step.step_type.value})")

        if step.step_type == StepType.EXECUTE:
            return await self._execute_query(step, context)

        elif step.step_type == StepType.VERIFY:
            return await self._verify_response(step, context)

        elif step.step_type == StepType.REFINE:
            return await self._refine_response(step, context)

        elif step.step_type == StepType.UPDATE_STRATEGY:
            return await self._update_strategy(step, context)

        elif step.step_type == StepType.CONDITION:
            return await self._handle_condition(step, context)

        elif step.step_type == StepType.LOOP:
            return await self._handle_loop(step, context)

        elif step.step_type == StepType.CUSTOM:
            # Custom steps should have a handler
            handler = step.params.get("handler")
            if not handler:
                raise ValueError(f"CUSTOM step '{step_id}' missing 'handler' parameter")
            return await handler(context)

        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

    async def _execute_query(self, step: ChainStep, context: ExecutionContext) -> DepartmentResponse:
        """Execute RAG query."""
        query_text = context.initial_input if isinstance(context.initial_input, str) else ""
        mode = step.params.get("mode", "verify")
        max_sources = step.params.get("max_sources", 5)

        request = DepartmentRequest(
            task_type="query",
            parameters={
                "query": query_text,
                "mode": mode,
                "max_sources": max_sources,
            },
        )

        response = await self.department.execute(request)

        # Update shared context
        response_data = response.metadata.get("response_data", {})
        context.shared_state["response"] = response_data
        context.shared_state["confidence"] = response.confidence.score
        context.shared_state["sources"] = response_data.get("sources", [])
        context.shared_state["reasoning_mode"] = response_data.get("reasoning_mode", "")
        context.shared_state["original_query"] = query_text

        return response

    async def _verify_response(self, step: ChainStep, context: ExecutionContext) -> dict:
        """Verify using department verification."""
        response = context.get_step_output("execute")
        if not response:
            raise ValueError("No prior response to verify (expected 'execute' step)")

        if not isinstance(response, DepartmentResponse):
            raise ValueError("Expected DepartmentResponse from execute step")

        verification_result = await self.department.verify(response)

        # Update shared context
        context.shared_state["verification_result"] = verification_result
        context.shared_state["verification_score"] = verification_result.overall_score
        context.shared_state["verification_checks"] = [
            {"dimension": c.dimension, "passed": c.passed, "score": c.score}
            for c in verification_result.checks
        ]

        return {
            "verified": verification_result.verified,
            "score": verification_result.overall_score,
            "checks": [c.dimension for c in verification_result.checks if c.passed],
            "recommendations": verification_result.recommendations,
        }

    async def _refine_response(self, step: ChainStep, context: ExecutionContext) -> DepartmentResponse:
        """Refine low-confidence response."""
        response = context.get_step_output("execute")
        if not response:
            raise ValueError("No prior response to refine (expected 'execute' step)")

        if not isinstance(response, DepartmentResponse):
            raise ValueError("Expected DepartmentResponse from execute step")

        # Add original query to metadata for refine
        query_text = context.initial_input if isinstance(context.initial_input, str) else ""
        response.metadata["original_query"] = query_text

        refined_response = await self.department.refine(response)

        # Update shared context
        context.shared_state["response"] = refined_response.response
        context.shared_state["confidence"] = refined_response.confidence.score
        context.shared_state["sources"] = refined_response.response.get("sources", [])

        return refined_response

    async def _update_strategy(self, step: ChainStep, context: ExecutionContext) -> dict:
        """Update learning strategy with feedback."""
        feedback = step.params.get("feedback", {})

        # Add context to feedback
        feedback.update({
            "confidence": context.shared_state.get("confidence", 0.0),
            "sources_count": len(context.shared_state.get("sources", [])),
        })

        await self.department.update_strategy(feedback)

        return {"status": "strategy_updated"}

    async def _handle_condition(self, step: ChainStep, context: ExecutionContext) -> dict:
        """Handle conditional branching."""
        if not step.condition:
            raise ValueError("Condition step missing 'condition' function")

        condition_result = step.condition(context.shared_state)

        return {
            "condition_result": condition_result,
            "branch": "true" if condition_result else "false",
        }

    async def _handle_loop(self, step: ChainStep, context: ExecutionContext) -> dict:
        """Handle loop step."""
        loop_id = id(step)  # Use step object id as loop identifier
        iteration = context.increment_loop_counter(loop_id)

        if iteration > step.max_iterations:
            context.reset_loop_counter(loop_id)
            return {"loop_complete": True, "iterations": iteration - 1}

        return {"loop_iteration": iteration}

    def _get_next_step(
        self,
        step: ChainStep,
        result: StepResult,
        context: ExecutionContext,
    ) -> Optional[str]:
        """Determine next step to execute."""
        # Conditional branching
        if step.condition:
            condition_met = step.condition(context.shared_state)
            if condition_met and step.on_success:
                return step.on_success
            elif not condition_met and step.on_failure:
                return step.on_failure

        # Success/failure branching
        if result.status == StepStatus.SUCCESS:
            if step.on_success:
                return step.on_success
        elif result.status == StepStatus.FAILED:
            if step.on_failure:
                return step.on_failure

        # Default: sequential next step
        return step.next_step

    def _build_trace(
        self,
        step_results: List[StepResult],
        context: ExecutionContext,
        start_time: float,
    ) -> ExecutionTrace:
        """Build execution trace."""
        return ExecutionTrace(
            step_results=step_results,
            final_context=context.shared_state.copy(),
            total_duration_ms=(time.time() - start_time) * 1000,
            steps_executed=len(step_results),
            errors=context.errors,
        )
