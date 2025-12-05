"""
Workflow Coordinator - Multi-Step Event Chains
===============================================

**Created**: December 3, 2025
**Purpose**: Coordinate multi-step skill workflows with compensation on failure

## Features

- Declarative workflow definitions
- State machine execution
- Automatic correlation ID tracking
- Saga pattern for distributed transactions
- Compensation handlers for rollback
- Progress tracking and callbacks

## Use Cases

1. **Multi-Skill Pipelines**: Chain skills (analyze → enhance → validate)
2. **Saga Transactions**: Coordinate distributed operations with rollback
3. **Long-Running Processes**: Track progress across multiple events
4. **Conditional Branching**: Route based on intermediate results
"""

import asyncio
import uuid
from typing import Optional, List, Dict, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from skills.protocol import SkillEvent, EventType

logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Definitions
# ============================================================================

class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """Status of a workflow."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    CANCELLED = "cancelled"


# Type aliases
StepHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
CompensationHandler = Callable[[Dict[str, Any]], Awaitable[None]]
ConditionFunc = Callable[[Dict[str, Any]], bool]


@dataclass
class WorkflowStep:
    """Definition of a single workflow step."""
    name: str
    handler: StepHandler
    compensation: Optional[CompensationHandler] = None
    condition: Optional[ConditionFunc] = None  # Skip if returns False
    timeout_seconds: float = 60.0
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow."""
    name: str
    steps: List[WorkflowStep]
    description: str = ""
    version: str = "1.0.0"
    enable_compensation: bool = True
    timeout_seconds: float = 300.0  # Overall workflow timeout
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    step_name: str
    status: StepStatus
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    retry_count: int = 0


@dataclass
class WorkflowInstance:
    """A running instance of a workflow."""
    id: str
    definition: WorkflowDefinition
    correlation_id: str
    status: WorkflowStatus
    current_step: int
    input_data: Dict[str, Any]
    context: Dict[str, Any]  # Accumulated step outputs
    step_results: List[StepResult]
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Workflow Coordinator
# ============================================================================

class WorkflowCoordinator:
    """
    Coordinates multi-step workflow execution.

    Provides:
    - Declarative workflow definitions
    - State machine execution
    - Automatic compensation on failure
    - Event emission for each step
    """

    def __init__(
        self,
        broker: Optional[Any] = None,  # EventBroker
        event_store: Optional[Any] = None  # EventStore
    ):
        """
        Initialize workflow coordinator.

        Args:
            broker: Optional EventBroker for event emission
            event_store: Optional EventStore for persistence
        """
        self.broker = broker
        self.event_store = event_store

        # Registered workflow definitions
        self._definitions: Dict[str, WorkflowDefinition] = {}

        # Active workflow instances
        self._instances: Dict[str, WorkflowInstance] = {}

        # Callbacks
        self._on_step_complete: List[Callable[[WorkflowInstance, StepResult], Awaitable[None]]] = []
        self._on_workflow_complete: List[Callable[[WorkflowInstance], Awaitable[None]]] = []

        # Statistics
        self.stats = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'workflows_compensated': 0,
            'total_steps_executed': 0
        }

        logger.info("WorkflowCoordinator created")

    def register_workflow(self, definition: WorkflowDefinition) -> None:
        """
        Register a workflow definition.

        Args:
            definition: Workflow definition to register
        """
        self._definitions[definition.name] = definition
        logger.info(f"Registered workflow: {definition.name} ({len(definition.steps)} steps)")

    def get_workflow_definition(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a registered workflow definition."""
        return self._definitions.get(name)

    async def start_workflow(
        self,
        workflow_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowInstance:
        """
        Start a new workflow instance.

        Args:
            workflow_name: Name of registered workflow
            input_data: Initial input data
            correlation_id: Optional correlation ID (auto-generated if not provided)
            metadata: Optional metadata

        Returns:
            WorkflowInstance for tracking
        """
        definition = self._definitions.get(workflow_name)
        if not definition:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Create instance
        instance = WorkflowInstance(
            id=str(uuid.uuid4()),
            definition=definition,
            correlation_id=correlation_id or str(uuid.uuid4()),
            status=WorkflowStatus.CREATED,
            current_step=0,
            input_data=input_data or {},
            context={},
            step_results=[],
            started_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        self._instances[instance.id] = instance
        self.stats['workflows_started'] += 1

        # Emit start event
        await self._emit_workflow_event(instance, EventType.SKILL_STARTED)

        logger.info(f"Started workflow {workflow_name} (id={instance.id})")

        # Execute workflow
        asyncio.create_task(self._execute_workflow(instance))

        return instance

    async def _execute_workflow(self, instance: WorkflowInstance) -> None:
        """Execute a workflow instance."""
        instance.status = WorkflowStatus.RUNNING

        try:
            # Execute each step
            for i, step in enumerate(instance.definition.steps):
                instance.current_step = i

                # Check condition
                if step.condition:
                    try:
                        if not step.condition(instance.context):
                            result = StepResult(
                                step_name=step.name,
                                status=StepStatus.SKIPPED,
                                started_at=datetime.now().isoformat(),
                                completed_at=datetime.now().isoformat()
                            )
                            instance.step_results.append(result)
                            logger.info(f"Skipped step {step.name} (condition not met)")
                            continue
                    except Exception as e:
                        logger.warning(f"Condition check failed for {step.name}: {e}")

                # Execute step
                result = await self._execute_step(instance, step)
                instance.step_results.append(result)

                # Notify callbacks
                for callback in self._on_step_complete:
                    try:
                        await callback(instance, result)
                    except Exception as e:
                        logger.warning(f"Step callback error: {e}")

                if result.status == StepStatus.FAILED:
                    # Step failed - initiate compensation
                    instance.status = WorkflowStatus.FAILED
                    instance.error = result.error

                    if instance.definition.enable_compensation:
                        await self._compensate_workflow(instance)

                    break

                # Add output to context
                instance.context[step.name] = result.output

            else:
                # All steps completed successfully
                instance.status = WorkflowStatus.COMPLETED
                self.stats['workflows_completed'] += 1

        except asyncio.TimeoutError:
            instance.status = WorkflowStatus.FAILED
            instance.error = "Workflow timeout"
            self.stats['workflows_failed'] += 1

            if instance.definition.enable_compensation:
                await self._compensate_workflow(instance)

        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            self.stats['workflows_failed'] += 1
            logger.error(f"Workflow {instance.id} failed: {e}")

            if instance.definition.enable_compensation:
                await self._compensate_workflow(instance)

        finally:
            instance.completed_at = datetime.now().isoformat()

            # Emit completion event
            event_type = EventType.SKILL_COMPLETED if instance.status == WorkflowStatus.COMPLETED else EventType.SKILL_FAILED
            await self._emit_workflow_event(instance, event_type)

            # Notify completion callbacks
            for callback in self._on_workflow_complete:
                try:
                    await callback(instance)
                except Exception as e:
                    logger.warning(f"Workflow callback error: {e}")

    async def _execute_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> StepResult:
        """Execute a single workflow step."""
        result = StepResult(
            step_name=step.name,
            status=StepStatus.RUNNING,
            started_at=datetime.now().isoformat()
        )

        # Emit step start event
        await self._emit_step_event(instance, step, EventType.SKILL_STARTED)

        # Prepare input (merge workflow input + context)
        step_input = {**instance.input_data, **instance.context}

        retry = 0
        last_error = None

        while retry <= step.retry_count:
            try:
                # Execute with timeout
                output = await asyncio.wait_for(
                    step.handler(step_input),
                    timeout=step.timeout_seconds
                )

                result.status = StepStatus.COMPLETED
                result.output = output or {}
                result.retry_count = retry

                self.stats['total_steps_executed'] += 1

                # Emit step complete event
                await self._emit_step_event(instance, step, EventType.SKILL_COMPLETED, output)

                logger.info(f"Step {step.name} completed (workflow={instance.id})")
                break

            except asyncio.TimeoutError:
                last_error = f"Step timeout after {step.timeout_seconds}s"
                logger.warning(f"Step {step.name} timeout (retry {retry}/{step.retry_count})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step {step.name} failed: {e} (retry {retry}/{step.retry_count})")

            retry += 1
            if retry <= step.retry_count:
                await asyncio.sleep(step.retry_delay_seconds)

        else:
            # All retries exhausted
            result.status = StepStatus.FAILED
            result.error = last_error
            result.retry_count = retry - 1

            # Emit step failed event
            await self._emit_step_event(instance, step, EventType.SKILL_FAILED, error=last_error)

            logger.error(f"Step {step.name} failed after {step.retry_count} retries")

        result.completed_at = datetime.now().isoformat()
        if result.started_at:
            start = datetime.fromisoformat(result.started_at)
            end = datetime.fromisoformat(result.completed_at)
            result.duration_ms = (end - start).total_seconds() * 1000

        return result

    async def _compensate_workflow(self, instance: WorkflowInstance) -> None:
        """Execute compensation handlers in reverse order."""
        instance.status = WorkflowStatus.COMPENSATING
        logger.info(f"Starting compensation for workflow {instance.id}")

        # Get completed steps in reverse order
        completed_steps = [
            (i, step) for i, step in enumerate(instance.definition.steps)
            if i < len(instance.step_results)
            and instance.step_results[i].status == StepStatus.COMPLETED
            and step.compensation is not None
        ]

        for i, step in reversed(completed_steps):
            try:
                # Mark as compensating
                instance.step_results[i].status = StepStatus.COMPENSATING

                # Execute compensation with step's output
                await step.compensation(instance.step_results[i].output)

                instance.step_results[i].status = StepStatus.COMPENSATED
                logger.info(f"Compensated step {step.name}")

            except Exception as e:
                logger.error(f"Compensation failed for step {step.name}: {e}")
                # Continue with other compensations

        instance.status = WorkflowStatus.COMPENSATED
        self.stats['workflows_compensated'] += 1
        logger.info(f"Compensation complete for workflow {instance.id}")

    async def _emit_workflow_event(
        self,
        instance: WorkflowInstance,
        event_type: EventType
    ) -> None:
        """Emit a workflow-level event."""
        if not self.broker:
            return

        event = SkillEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            skill_name=f"workflow:{instance.definition.name}",
            topic=f"workflow.{event_type.value}.{instance.definition.name}",
            timestamp=datetime.now().isoformat(),
            correlation_id=instance.correlation_id,
            sequence=0,
            payload={
                'workflow_id': instance.id,
                'workflow_name': instance.definition.name,
                'status': instance.status.value,
                'current_step': instance.current_step,
                'error': instance.error
            },
            metadata=instance.metadata
        )

        try:
            await self.broker.emit(event)

            if self.event_store:
                await self.event_store.store(event)
        except Exception as e:
            logger.warning(f"Failed to emit workflow event: {e}")

    async def _emit_step_event(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep,
        event_type: EventType,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Emit a step-level event."""
        if not self.broker:
            return

        event = SkillEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            skill_name=f"workflow:{instance.definition.name}:{step.name}",
            topic=f"workflow.step.{event_type.value}.{instance.definition.name}.{step.name}",
            timestamp=datetime.now().isoformat(),
            correlation_id=instance.correlation_id,
            causation_id=instance.id,  # Link to workflow
            sequence=instance.current_step,
            payload={
                'workflow_id': instance.id,
                'step_name': step.name,
                'output': output,
                'error': error
            },
            metadata=step.metadata
        )

        try:
            await self.broker.emit(event)

            if self.event_store:
                await self.event_store.store(event)
        except Exception as e:
            logger.warning(f"Failed to emit step event: {e}")

    def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get a workflow instance by ID."""
        return self._instances.get(instance_id)

    def get_instances_by_correlation(self, correlation_id: str) -> List[WorkflowInstance]:
        """Get all workflow instances with a correlation ID."""
        return [
            instance for instance in self._instances.values()
            if instance.correlation_id == correlation_id
        ]

    async def cancel_workflow(self, instance_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            instance_id: Workflow instance ID

        Returns:
            True if cancelled, False if not found or already complete
        """
        instance = self._instances.get(instance_id)
        if not instance:
            return False

        if instance.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
            return False

        instance.status = WorkflowStatus.CANCELLED
        instance.completed_at = datetime.now().isoformat()

        # Emit cancellation event
        await self._emit_workflow_event(instance, EventType.SKILL_FAILED)

        logger.info(f"Cancelled workflow {instance_id}")
        return True

    def on_step_complete(
        self,
        callback: Callable[[WorkflowInstance, StepResult], Awaitable[None]]
    ) -> None:
        """Register callback for step completion."""
        self._on_step_complete.append(callback)

    def on_workflow_complete(
        self,
        callback: Callable[[WorkflowInstance], Awaitable[None]]
    ) -> None:
        """Register callback for workflow completion."""
        self._on_workflow_complete.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        active = sum(
            1 for i in self._instances.values()
            if i.status in (WorkflowStatus.CREATED, WorkflowStatus.RUNNING)
        )

        return {
            **self.stats,
            'registered_workflows': len(self._definitions),
            'active_instances': active,
            'total_instances': len(self._instances)
        }


# ============================================================================
# Builder Functions
# ============================================================================

def workflow(name: str, description: str = "", **kwargs) -> WorkflowDefinition:
    """
    Create a workflow definition.

    Args:
        name: Workflow name
        description: Workflow description
        **kwargs: Additional workflow options

    Returns:
        WorkflowDefinition (add steps with .step())
    """
    return WorkflowDefinition(
        name=name,
        steps=[],
        description=description,
        **kwargs
    )


def step(
    name: str,
    handler: StepHandler,
    compensation: Optional[CompensationHandler] = None,
    condition: Optional[ConditionFunc] = None,
    **kwargs
) -> WorkflowStep:
    """
    Create a workflow step.

    Args:
        name: Step name
        handler: Async handler function
        compensation: Optional compensation handler
        condition: Optional condition function
        **kwargs: Additional step options

    Returns:
        WorkflowStep
    """
    return WorkflowStep(
        name=name,
        handler=handler,
        compensation=compensation,
        condition=condition,
        **kwargs
    )


# ============================================================================
# Factory Function
# ============================================================================

def create_workflow_coordinator(
    broker: Optional[Any] = None,
    event_store: Optional[Any] = None
) -> WorkflowCoordinator:
    """
    Create a workflow coordinator.

    Args:
        broker: Optional EventBroker
        event_store: Optional EventStore

    Returns:
        WorkflowCoordinator instance
    """
    return WorkflowCoordinator(broker, event_store)
