#!/usr/bin/env python3
"""
Multi-Step Workflow Engine for Promptly Matrix Bot

Enables chaining multiple operations together with:
- Sequential and parallel execution
- Progress tracking per step
- Conditional branching
- Error recovery and rollback
- State persistence
- Async notifications

Features:
- Define workflows as YAML or Python dicts
- Built-in step types (optimize, run, code-review, approval, etc.)
- Custom step functions
- Context passing between steps
- Retry logic with exponential backoff
- Workflow templates
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Workflow step status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """Overall workflow status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Single step in a workflow"""
    name: str
    type: str  # optimize, run, code-review, approval, custom
    params: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None  # Python expression to evaluate
    on_error: str = "fail"  # fail, skip, retry, rollback
    retry_count: int = 0
    retry_delay: int = 5  # seconds
    depends_on: List[str] = field(default_factory=list)  # Step dependencies
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started: Optional[datetime] = None
    completed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.type,
            "params": self.params,
            "condition": self.condition,
            "on_error": self.on_error,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started": self.started.isoformat() if self.started else None,
            "completed": self.completed.isoformat() if self.completed else None
        }


@dataclass
class Workflow:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    context: Dict[str, Any] = field(default_factory=dict)  # Shared state
    status: WorkflowStatus = WorkflowStatus.PENDING
    created: datetime = field(default_factory=datetime.now)
    started: Optional[datetime] = None
    completed: Optional[datetime] = None
    room_id: Optional[str] = None
    initiator: Optional[str] = None

    def get_step(self, name: str) -> Optional[WorkflowStep]:
        """Get step by name"""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_pending_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to run"""
        pending = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue

            # Check if dependencies are met
            deps_met = all(
                self.get_step(dep).status == StepStatus.SUCCESS
                for dep in step.depends_on
            )

            if deps_met:
                pending.append(step)

        return pending

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
            "status": self.status.value,
            "created": self.created.isoformat(),
            "started": self.started.isoformat() if self.started else None,
            "completed": self.completed.isoformat() if self.completed else None,
            "room_id": self.room_id,
            "initiator": self.initiator
        }


class WorkflowEngine:
    """
    Executes multi-step workflows with progress tracking.

    Usage:
        engine = WorkflowEngine(bot_client, state_manager)

        # Define workflow
        workflow = Workflow(
            id="deploy_001",
            name="Deploy Prompt",
            description="Deploy prompt to production",
            steps=[
                WorkflowStep(name="optimize", type="optimize", params={...}),
                WorkflowStep(name="review", type="code-review", depends_on=["optimize"]),
                WorkflowStep(name="approve", type="approval", depends_on=["review"]),
                WorkflowStep(name="deploy", type="custom", depends_on=["approve"])
            ]
        )

        # Execute
        result = await engine.execute(workflow)
    """

    def __init__(self, bot_client=None, state_manager=None, promptly_core=None):
        """
        Initialize workflow engine.

        Args:
            bot_client: Matrix client for notifications
            state_manager: State manager for persistence
            promptly_core: Promptly Core for AI operations
        """
        self.client = bot_client
        self.state = state_manager
        self.promptly_core = promptly_core

        # Running workflows
        self.workflows: Dict[str, Workflow] = {}

        # Custom step handlers
        self.step_handlers: Dict[str, Callable] = {}

        # Register built-in handlers
        self._register_builtin_handlers()

    def _register_builtin_handlers(self):
        """Register built-in step handlers"""
        self.step_handlers['optimize'] = self._handle_optimize
        self.step_handlers['run'] = self._handle_run
        self.step_handlers['code-review'] = self._handle_code_review
        self.step_handlers['approval'] = self._handle_approval
        self.step_handlers['wait'] = self._handle_wait
        self.step_handlers['condition'] = self._handle_condition

    def register_handler(self, step_type: str, handler: Callable[[WorkflowStep, Workflow], Awaitable[Any]]):
        """Register custom step handler"""
        self.step_handlers[step_type] = handler

    async def execute(self, workflow: Workflow) -> Workflow:
        """
        Execute workflow.

        Args:
            workflow: Workflow to execute

        Returns:
            Completed workflow with results
        """
        self.workflows[workflow.id] = workflow
        workflow.status = WorkflowStatus.RUNNING
        workflow.started = datetime.now()

        logger.info(f"Starting workflow '{workflow.name}' ({workflow.id})")

        # Notify start
        if self.client and workflow.room_id:
            await self._notify_start(workflow)

        try:
            # Execute steps
            while True:
                # Get pending steps (dependencies met)
                pending = workflow.get_pending_steps()

                if not pending:
                    # Check if all steps complete
                    all_done = all(s.status in [StepStatus.SUCCESS, StepStatus.SKIPPED, StepStatus.FAILED] for s in workflow.steps)
                    if all_done:
                        break

                    # Deadlock - some steps waiting on failed dependencies
                    logger.warning(f"Workflow {workflow.id} deadlocked")
                    break

                # Execute pending steps (can be parallel)
                tasks = [self._execute_step(step, workflow) for step in pending]
                await asyncio.gather(*tasks)

            # Determine final status
            if any(s.status == StepStatus.FAILED for s in workflow.steps):
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.SUCCESS

        except Exception as e:
            logger.error(f"Workflow {workflow.id} failed: {e}")
            workflow.status = WorkflowStatus.FAILED

        workflow.completed = datetime.now()

        # Notify completion
        if self.client and workflow.room_id:
            await self._notify_complete(workflow)

        # Save to state
        if self.state:
            self._save_workflow(workflow)

        logger.info(f"Workflow '{workflow.name}' completed with status: {workflow.status.value}")

        return workflow

    async def _execute_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """
        Execute a single workflow step.

        Args:
            step: Step to execute
            workflow: Parent workflow

        Returns:
            Step result
        """
        step.status = StepStatus.RUNNING
        step.started = datetime.now()

        logger.info(f"Executing step '{step.name}' (type: {step.type})")

        # Notify step start
        if self.client and workflow.room_id:
            await self._notify_step_start(step, workflow)

        # Check condition
        if step.condition:
            condition_met = self._evaluate_condition(step.condition, workflow.context)
            if not condition_met:
                step.status = StepStatus.SKIPPED
                step.completed = datetime.now()
                logger.info(f"Step '{step.name}' skipped (condition not met)")
                return None

        # Execute with retry logic
        retries = 0
        max_retries = step.retry_count if step.on_error == "retry" else 0

        while retries <= max_retries:
            try:
                # Get handler
                handler = self.step_handlers.get(step.type)
                if not handler:
                    raise ValueError(f"Unknown step type: {step.type}")

                # Execute handler
                result = await handler(step, workflow)

                # Success
                step.status = StepStatus.SUCCESS
                step.result = result
                step.completed = datetime.now()

                # Update context
                if result is not None:
                    workflow.context[step.name] = result

                logger.info(f"Step '{step.name}' completed successfully")

                # Notify step completion
                if self.client and workflow.room_id:
                    await self._notify_step_complete(step, workflow)

                return result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Step '{step.name}' failed: {error_msg}")

                if retries < max_retries:
                    # Retry
                    retries += 1
                    delay = step.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                    logger.info(f"Retrying step '{step.name}' in {delay}s (attempt {retries}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

                # Handle error based on strategy
                if step.on_error == "skip":
                    step.status = StepStatus.SKIPPED
                    logger.info(f"Step '{step.name}' skipped after error")
                elif step.on_error == "rollback":
                    step.status = StepStatus.FAILED
                    await self._rollback_workflow(workflow, step)
                else:  # fail
                    step.status = StepStatus.FAILED

                step.error = error_msg
                step.completed = datetime.now()

                # Notify step error
                if self.client and workflow.room_id:
                    await self._notify_step_error(step, workflow)

                return None

    async def _handle_optimize(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """Handle optimize step"""
        if not self.promptly_core:
            raise ValueError("Promptly Core not available")

        task = step.params.get('task')
        examples = step.params.get('examples', [])

        result = await self.promptly_core.optimize_prompt(task, examples)
        return result

    async def _handle_run(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """Handle run step"""
        if not self.promptly_core:
            raise ValueError("Promptly Core not available")

        workflow_name = step.params.get('workflow')
        input_data = step.params.get('input', '')

        result = await self.promptly_core.run_workflow(workflow_name, input_data)
        return result

    async def _handle_code_review(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """Handle code-review step"""
        # Import here to avoid circular dependency
        from .code_reviewer import get_code_reviewer

        reviewer = get_code_reviewer()

        code = step.params.get('code', '')
        language = step.params.get('language')

        if not code:
            # Try to get code from previous step
            code_step = step.params.get('code_from_step')
            if code_step and code_step in workflow.context:
                code = workflow.context[code_step].get('code', '')

        result = reviewer.review(code, language)

        # Return as dict
        return {
            'language': result.language,
            'risk_score': result.risk_score,
            'issues': [
                {
                    'type': i.type.value,
                    'severity': i.severity.value,
                    'title': i.title,
                    'description': i.description,
                    'line': i.line
                }
                for i in result.issues[:5]  # Top 5
            ]
        }

    async def _handle_approval(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """Handle approval step"""
        # Import here to avoid circular dependency
        from .approval_workflow import get_approval_manager, ActionRisk

        if not workflow.room_id or not workflow.initiator:
            raise ValueError("Approval requires room_id and initiator")

        manager = get_approval_manager(self.state, self.client)

        action = step.params.get('action', step.name)
        context = step.params.get('context', {})
        risk_level = ActionRisk(step.params.get('risk_level', 'medium'))

        # Request approval
        request = await manager.request_approval(
            room_id=workflow.room_id,
            initiator=workflow.initiator,
            action=action,
            context=context,
            risk_level=risk_level
        )

        # Wait for approval (poll status)
        max_wait = 3600  # 1 hour
        waited = 0
        while waited < max_wait:
            await asyncio.sleep(5)
            waited += 5

            # Check status
            updated = await manager.get_request(request.request_id)
            if updated and updated.status.value == "approved":
                return {"approved": True, "approvers": updated.approvals}
            elif updated and updated.status.value in ["rejected", "expired"]:
                raise Exception(f"Approval {updated.status.value}")

        raise Exception("Approval timeout")

    async def _handle_wait(self, step: WorkflowStep, workflow: Workflow) -> None:
        """Handle wait step"""
        duration = step.params.get('duration', 0)  # seconds
        await asyncio.sleep(duration)
        return None

    async def _handle_condition(self, step: WorkflowStep, workflow: Workflow) -> bool:
        """Handle conditional branching"""
        condition = step.params.get('condition')
        if not condition:
            raise ValueError("Condition step requires 'condition' parameter")

        result = self._evaluate_condition(condition, workflow.context)
        return result

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition expression.

        Args:
            condition: Python expression (e.g., "risk_score > 5.0")
            context: Workflow context

        Returns:
            Boolean result
        """
        try:
            # Safe eval with limited context
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False

    async def _rollback_workflow(self, workflow: Workflow, failed_step: WorkflowStep):
        """Rollback workflow after failure"""
        logger.info(f"Rolling back workflow {workflow.id}")

        # Execute rollback in reverse order
        for step in reversed(workflow.steps):
            if step.status != StepStatus.SUCCESS:
                continue

            if step == failed_step:
                break

            # Execute rollback handler if exists
            rollback_handler = self.step_handlers.get(f"{step.type}_rollback")
            if rollback_handler:
                try:
                    await rollback_handler(step, workflow)
                    logger.info(f"Rolled back step '{step.name}'")
                except Exception as e:
                    logger.error(f"Rollback failed for step '{step.name}': {e}")

    async def _notify_start(self, workflow: Workflow):
        """Notify workflow start"""
        if not self.client:
            return

        message = f"🚀 **Workflow Started**\n\n**Name:** {workflow.name}\n**Steps:** {len(workflow.steps)}"

        await self.client.room_send(
            workflow.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.notice", "body": message}
        )

    async def _notify_complete(self, workflow: Workflow):
        """Notify workflow completion"""
        if not self.client:
            return

        success_count = sum(1 for s in workflow.steps if s.status == StepStatus.SUCCESS)
        duration = (workflow.completed - workflow.started).total_seconds()

        emoji = "✅" if workflow.status == WorkflowStatus.SUCCESS else "❌"
        message = f"{emoji} **Workflow Complete**\n\n"
        message += f"**Name:** {workflow.name}\n"
        message += f"**Status:** {workflow.status.value}\n"
        message += f"**Steps Completed:** {success_count}/{len(workflow.steps)}\n"
        message += f"**Duration:** {duration:.1f}s"

        await self.client.room_send(
            workflow.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.notice", "body": message}
        )

    async def _notify_step_start(self, step: WorkflowStep, workflow: Workflow):
        """Notify step start"""
        if not self.client:
            return

        message = f"⏳ Step '{step.name}' starting..."

        await self.client.room_send(
            workflow.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.notice", "body": message}
        )

    async def _notify_step_complete(self, step: WorkflowStep, workflow: Workflow):
        """Notify step completion"""
        if not self.client:
            return

        duration = (step.completed - step.started).total_seconds()
        message = f"✅ Step '{step.name}' completed ({duration:.1f}s)"

        await self.client.room_send(
            workflow.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.notice", "body": message}
        )

    async def _notify_step_error(self, step: WorkflowStep, workflow: Workflow):
        """Notify step error"""
        if not self.client:
            return

        message = f"❌ Step '{step.name}' failed: {step.error}"

        await self.client.room_send(
            workflow.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.notice", "body": message}
        )

    def _save_workflow(self, workflow: Workflow):
        """Save workflow to state manager"""
        if not self.state:
            return

        self.state.set_workflow_state(
            room_id=workflow.room_id or "global",
            workflow_id=workflow.id,
            state=workflow.to_dict()
        )


# Singleton
_workflow_engine = None

def get_workflow_engine(bot_client=None, state_manager=None, promptly_core=None) -> WorkflowEngine:
    """Get singleton workflow engine"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine(bot_client, state_manager, promptly_core)
    return _workflow_engine


# Test
if __name__ == "__main__":
    async def test_workflow():
        engine = WorkflowEngine()

        # Define simple workflow
        workflow = Workflow(
            id="test_001",
            name="Test Workflow",
            description="Simple test workflow",
            steps=[
                WorkflowStep(
                    name="step1",
                    type="wait",
                    params={"duration": 0.1}
                ),
                WorkflowStep(
                    name="step2",
                    type="condition",
                    params={"condition": "True"},
                    depends_on=["step1"]
                ),
                WorkflowStep(
                    name="step3",
                    type="wait",
                    params={"duration": 0.1},
                    depends_on=["step2"]
                )
            ]
        )

        # Execute
        result = await engine.execute(workflow)

        print(f"✅ Workflow completed: {result.status.value}")
        print(f"   Steps: {len([s for s in result.steps if s.status == StepStatus.SUCCESS])}/{len(result.steps)} successful")

        assert result.status == WorkflowStatus.SUCCESS
        assert all(s.status == StepStatus.SUCCESS for s in result.steps)

        print("\n✅ All workflow engine tests passed!")

    asyncio.run(test_workflow())
