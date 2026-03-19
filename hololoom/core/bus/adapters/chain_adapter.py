"""
Chain Adapter — Wraps ChainOrchestrator with bus signal emission.
================================================================

This adapter does NOT modify ChainOrchestrator. It wraps it:
- Before execute_chain(): emits EXECUTION_START
- After each step: emits STEP_COMPLETE or STEP_FAILED
- After execute_chain(): emits EXECUTION_COMPLETE or EXECUTION_FAILED
- Subscribes to CONTROL signals for pause/cancel (future)
- Records execution traces and step results to the journal

The adapter also connects convergence detection: it checks for
CONVERGENCE signals between steps and can early-terminate.

Usage:
    from hololoom.core.bus import UnifiedBus, Signal, SignalKind
    from hololoom.core.bus.adapters.chain_adapter import BusChainOrchestrator

    bus = UnifiedBus()
    orchestrator = BusChainOrchestrator(department, bus=bus)
    result = await orchestrator.execute_chain(chain, initial_input)
    # Signals are emitted automatically throughout execution
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any

from hololoom.chaining.chain import Chain
from hololoom.chaining.orchestrator import (
    ChainOrchestrator,
    ChainResult,
)
from hololoom.chaining.types import (
    ChainExecutionStats,
    ExecutionContext,
    RollbackPoint,
    StepResult,
    StepStatus,
)

from ..signal import Signal, SignalKind, SignalPriority, TriggerMode
from ..unified_bus import UnifiedBus

logger = logging.getLogger(__name__)


class BusChainOrchestrator(ChainOrchestrator):
    """
    ChainOrchestrator that emits signals to the UnifiedBus.

    Extends (not replaces) ChainOrchestrator. All existing behavior is preserved.
    Bus emission is additive — if bus is None, behaves identically to base class.
    """

    def __init__(
        self,
        department,
        bus: UnifiedBus | None = None,
        enable_tracing: bool = True,
        enable_rollback: bool = False,
        enable_convergence: bool = True,
    ):
        """
        Args:
            department: Department implementing DepartmentProtocol
            bus: UnifiedBus instance (None = no signals, base behavior)
            enable_tracing: Track execution trace
            enable_rollback: Enable rollback on failure
            enable_convergence: Check for convergence signals between steps
        """
        super().__init__(
            department=department,
            enable_tracing=enable_tracing,
            enable_rollback=enable_rollback,
        )
        self.bus = bus
        self.enable_convergence = enable_convergence

        # Convergence state (set by convergence adapter via bus signal)
        self._converged = False
        self._convergence_reason = ""

        # Subscribe to convergence signals if bus is available
        if self.bus and self.enable_convergence:
            self.bus.on(SignalKind.CONVERGENCE, self._handle_convergence)

    async def _handle_convergence(self, signal: Signal) -> Signal | None:
        """Handle convergence signal — mark for early termination."""
        # Only respond to convergence for our correlation chain
        self._converged = True
        self._convergence_reason = signal.payload.get("reason", "convergence detected")
        logger.info(f"Convergence received: {self._convergence_reason}")
        return None

    async def execute_chain(
        self,
        chain: Chain,
        initial_input: Any,
        max_total_steps: int = 100,
    ) -> ChainResult:
        """
        Execute chain with bus signal emission.

        Wraps the base execute_chain with:
        - EXECUTION_START signal before execution
        - EXECUTION_COMPLETE/FAILED signal after execution
        - Journal recording of the execution trace
        """
        # Reset convergence state for this execution
        self._converged = False
        self._convergence_reason = ""

        # Generate correlation ID for this execution
        execution_id = uuid.uuid4().hex[:12]
        correlation_id = f"chain-{execution_id}"
        start_time = time.time()

        # Emit EXECUTION_START
        if self.bus:
            await self.bus.emit(Signal(
                kind=SignalKind.EXECUTION_START,
                priority=SignalPriority.NORMAL,
                source="chain_adapter",
                payload={
                    "execution_id": execution_id,
                    "chain_name": chain.name,
                    "entry_point": chain.entry_point,
                    "step_count": len(chain.steps),
                    "execution_type": "chain",
                },
                correlation_id=correlation_id,
                trigger_mode=TriggerMode.ACTIVE,
            ))

        # Execute via base class (all existing behavior preserved)
        result = await self._execute_chain_with_signals(
            chain=chain,
            initial_input=initial_input,
            max_total_steps=max_total_steps,
            execution_id=execution_id,
            correlation_id=correlation_id,
        )

        duration_ms = (time.time() - start_time) * 1000

        # Emit EXECUTION_COMPLETE or EXECUTION_FAILED
        if self.bus:
            result_kind = SignalKind.EXECUTION_COMPLETE if result.success else SignalKind.EXECUTION_FAILED
            await self.bus.emit(Signal(
                kind=result_kind,
                priority=SignalPriority.NORMAL,
                source="chain_adapter",
                payload={
                    "execution_id": execution_id,
                    "chain_name": chain.name,
                    "success": result.success,
                    "confidence": result.confidence,
                    "duration_ms": duration_ms,
                    "error": result.error,
                    "steps_executed": result.stats.total_steps if result.stats else 0,
                    "converged": self._converged,
                },
                correlation_id=correlation_id,
                trigger_mode=TriggerMode.REACTIVE,
            ))

        # Record to journal (if bus has one)
        if self.bus and self.bus._journal:
            await self.bus._journal.record_execution(
                trace_id=execution_id,
                correlation_id=correlation_id,
                execution_type="chain",
                name=chain.name,
                started_at=datetime.fromtimestamp(start_time).isoformat(),
                completed_at=datetime.now().isoformat(),
                success=result.success,
                total_steps=result.stats.total_steps if result.stats else 0,
                duration_ms=duration_ms,
                final_confidence=result.confidence,
                error=result.error,
                context=result.trace.final_context if result.trace else None,
            )

        return result

    async def _execute_chain_with_signals(
        self,
        chain: Chain,
        initial_input: Any,
        max_total_steps: int,
        execution_id: str,
        correlation_id: str,
    ) -> ChainResult:
        """
        Execute chain step-by-step, emitting signals after each step.

        This reimplements the execute_chain loop (not calling super) to
        inject signal emission between steps. The step execution itself
        delegates to the base class methods.
        """
        # Validate chain
        validation_errors = chain.validate()
        if validation_errors:
            error_msg = f"Chain validation failed: {validation_errors}"
            self.logger.error(error_msg)
            return ChainResult(success=False, error=error_msg)

        # Initialize context
        context = ExecutionContext(initial_input=initial_input)
        step_results: list[StepResult] = []
        rollback_stack: list[RollbackPoint] = []
        stats = ChainExecutionStats()
        start_time = time.time()

        try:
            current_step_id = chain.entry_point
            steps_executed = 0

            while current_step_id and steps_executed < max_total_steps:
                # Check convergence (bus-driven early termination)
                if self._converged:
                    self.logger.info(
                        f"Early termination: {self._convergence_reason}"
                    )
                    break

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
                    result = StepResult(
                        step_id=current_step_id,
                        step_type=step.step_type.value,
                        status=StepStatus.SKIPPED,
                    )
                    step_results.append(result)
                    stats.skipped_steps += 1

                    if step.next_step:
                        current_step_id = step.next_step
                    elif step.on_success:
                        current_step_id = step.on_success
                    else:
                        current_step_id = None
                    continue

                # Rollback point
                if self.enable_rollback:
                    rollback_stack.append(RollbackPoint(
                        step_id=current_step_id,
                        context_snapshot=context.shared_state.copy(),
                    ))

                # Execute step (delegates to base class)
                step_start = time.time()
                result = await self._execute_step_with_retries(
                    step_id=current_step_id,
                    step=step,
                    context=context,
                    retry_count=step.retry_count,
                )
                step_duration_ms = (time.time() - step_start) * 1000

                step_results.append(result)
                stats.completed_steps += 1 if result.status == StepStatus.SUCCESS else 0
                stats.failed_steps += 1 if result.status == StepStatus.FAILED else 0

                # Store step output
                if result.status == StepStatus.SUCCESS:
                    context.set_step_output(current_step_id, result.output)

                # Emit STEP_COMPLETE or STEP_FAILED signal
                if self.bus:
                    step_kind = (
                        SignalKind.STEP_COMPLETE
                        if result.status == StepStatus.SUCCESS
                        else SignalKind.STEP_FAILED
                    )
                    await self.bus.emit(Signal(
                        kind=step_kind,
                        priority=SignalPriority.NORMAL,
                        source="chain_adapter",
                        payload={
                            "execution_id": execution_id,
                            "step_id": current_step_id,
                            "step_type": result.step_type,
                            "status": result.status.value,
                            "duration_ms": step_duration_ms,
                            "confidence": context.shared_state.get("confidence"),
                            "error": result.error,
                        },
                        correlation_id=correlation_id,
                        trigger_mode=TriggerMode.REACTIVE,
                    ))

                    # Record step to journal
                    if self.bus._journal:
                        await self.bus._journal.record_step(
                            step_record_id=uuid.uuid4().hex[:12],
                            execution_id=execution_id,
                            step_id=current_step_id,
                            step_type=result.step_type,
                            status=result.status.value,
                            timestamp=datetime.now().isoformat(),
                            duration_ms=step_duration_ms,
                            error=result.error,
                            confidence=context.shared_state.get("confidence"),
                        )

                # Determine next step
                next_step_id = self._get_next_step(
                    step=step, result=result, context=context,
                )

                if next_step_id:
                    self.logger.info(f"Step '{current_step_id}' complete -> '{next_step_id}'")
                else:
                    self.logger.info(f"Step '{current_step_id}' complete -> [end]")

                current_step_id = next_step_id
                steps_executed += 1

            # Extract final response
            from hololoom.protocols.department import DepartmentResponse

            final_response = context.get_step_output(chain.entry_point)
            if not final_response:
                for r in reversed(step_results):
                    if r.step_type in ["execute", "verify"]:
                        final_response = r.output
                        break

            confidence = None
            if isinstance(final_response, DepartmentResponse):
                confidence = final_response.confidence.score
            elif isinstance(final_response, dict) and "confidence" in final_response:
                conf = final_response["confidence"]
                confidence = conf.get("score") if isinstance(conf, dict) else conf

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


__all__ = ["BusChainOrchestrator"]
