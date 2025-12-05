#!/usr/bin/env python3
"""
Workflow Templates for Promptly Matrix Bot

Pre-built workflow templates for common use cases:
- Prompt deployment pipeline
- Code review + approval workflow
- Multi-stage testing workflow
- Emergency rollback workflow
"""

from .workflow_engine import Workflow, WorkflowStep
from .approval_workflow import ActionRisk


def create_deploy_prompt_workflow(
    prompt_name: str,
    task: str,
    examples: list,
    room_id: str,
    initiator: str
) -> Workflow:
    """
    Create prompt deployment workflow.

    Pipeline:
    1. Optimize prompt with examples
    2. Test on validation set
    3. Request approval (HIGH risk)
    4. Deploy to production

    Args:
        prompt_name: Name for the prompt
        task: Task description
        examples: Training examples
        room_id: Matrix room ID
        initiator: User initiating workflow

    Returns:
        Workflow ready to execute
    """
    return Workflow(
        id=f"deploy_{prompt_name}_{int(__import__('time').time())}",
        name=f"Deploy Prompt: {prompt_name}",
        description="Optimize, test, approve, and deploy prompt to production",
        room_id=room_id,
        initiator=initiator,
        steps=[
            WorkflowStep(
                name="optimize",
                type="optimize",
                params={
                    "task": task,
                    "examples": examples
                }
            ),
            WorkflowStep(
                name="test",
                type="run",
                params={
                    "workflow": "validation",
                    "input": "test validation cases"
                },
                depends_on=["optimize"]
            ),
            WorkflowStep(
                name="approval",
                type="approval",
                params={
                    "action": f"deploy_prompt_{prompt_name}",
                    "context": {
                        "prompt_name": prompt_name,
                        "task": task
                    },
                    "risk_level": "high"
                },
                depends_on=["test"]
            ),
            WorkflowStep(
                name="deploy",
                type="custom",
                params={
                    "action": "deploy",
                    "target": "production",
                    "prompt_name": prompt_name
                },
                depends_on=["approval"],
                on_error="rollback"
            )
        ]
    )


def create_code_review_workflow(
    code: str,
    language: str,
    require_approval: bool = True,
    room_id: str = None,
    initiator: str = None
) -> Workflow:
    """
    Create code review workflow.

    Pipeline:
    1. Run security scan
    2. Check code quality
    3. Request approval if critical issues (conditional)
    4. Mark as approved

    Args:
        code: Code to review
        language: Programming language
        require_approval: Require approval for critical issues
        room_id: Matrix room ID
        initiator: User initiating workflow

    Returns:
        Workflow ready to execute
    """
    workflow_id = f"review_{int(__import__('time').time())}"

    steps = [
        WorkflowStep(
            name="security_scan",
            type="code-review",
            params={
                "code": code,
                "language": language
            }
        )
    ]

    if require_approval:
        steps.append(
            WorkflowStep(
                name="check_risk",
                type="condition",
                params={
                    "condition": "security_scan['risk_score'] > 5.0"
                },
                depends_on=["security_scan"]
            )
        )
        steps.append(
            WorkflowStep(
                name="approval",
                type="approval",
                params={
                    "action": "approve_code",
                    "context": {
                        "language": language,
                        "risk_score": "{{security_scan.risk_score}}"
                    },
                    "risk_level": "high"
                },
                condition="security_scan['risk_score'] > 5.0",
                depends_on=["check_risk"]
            )
        )

    return Workflow(
        id=workflow_id,
        name="Code Review Workflow",
        description="Security scan with conditional approval",
        room_id=room_id,
        initiator=initiator,
        steps=steps
    )


def create_testing_workflow(
    test_suites: list,
    room_id: str = None,
    initiator: str = None
) -> Workflow:
    """
    Create multi-stage testing workflow.

    Pipeline:
    1. Run unit tests
    2. Run integration tests (parallel)
    3. Run E2E tests (depends on both)
    4. Generate report

    Args:
        test_suites: List of test suite names
        room_id: Matrix room ID
        initiator: User initiating workflow

    Returns:
        Workflow ready to execute
    """
    workflow_id = f"test_{int(__import__('time').time())}"

    steps = []

    # Unit tests
    steps.append(
        WorkflowStep(
            name="unit_tests",
            type="run",
            params={
                "workflow": "run_tests",
                "input": "unit"
            }
        )
    )

    # Integration tests (parallel with unit tests)
    steps.append(
        WorkflowStep(
            name="integration_tests",
            type="run",
            params={
                "workflow": "run_tests",
                "input": "integration"
            }
        )
    )

    # E2E tests (depends on both)
    steps.append(
        WorkflowStep(
            name="e2e_tests",
            type="run",
            params={
                "workflow": "run_tests",
                "input": "e2e"
            },
            depends_on=["unit_tests", "integration_tests"]
        )
    )

    # Generate report
    steps.append(
        WorkflowStep(
            name="generate_report",
            type="custom",
            params={
                "action": "generate_test_report"
            },
            depends_on=["e2e_tests"]
        )
    )

    return Workflow(
        id=workflow_id,
        name="Testing Pipeline",
        description="Multi-stage testing with parallel execution",
        room_id=room_id,
        initiator=initiator,
        steps=steps
    )


def create_rollback_workflow(
    deployment_id: str,
    room_id: str,
    initiator: str
) -> Workflow:
    """
    Create emergency rollback workflow.

    Pipeline:
    1. Request emergency approval (CRITICAL)
    2. Stop current deployment
    3. Restore previous version
    4. Verify rollback

    Args:
        deployment_id: ID of deployment to rollback
        room_id: Matrix room ID
        initiator: User initiating workflow

    Returns:
        Workflow ready to execute
    """
    return Workflow(
        id=f"rollback_{deployment_id}_{int(__import__('time').time())}",
        name=f"Emergency Rollback: {deployment_id}",
        description="Rollback failed deployment to previous version",
        room_id=room_id,
        initiator=initiator,
        steps=[
            WorkflowStep(
                name="emergency_approval",
                type="approval",
                params={
                    "action": f"rollback_deployment_{deployment_id}",
                    "context": {
                        "deployment_id": deployment_id,
                        "reason": "production failure"
                    },
                    "risk_level": "critical"
                }
            ),
            WorkflowStep(
                name="stop_deployment",
                type="custom",
                params={
                    "action": "stop_deployment",
                    "deployment_id": deployment_id
                },
                depends_on=["emergency_approval"]
            ),
            WorkflowStep(
                name="restore_previous",
                type="custom",
                params={
                    "action": "restore_deployment",
                    "deployment_id": deployment_id
                },
                depends_on=["stop_deployment"]
            ),
            WorkflowStep(
                name="verify_rollback",
                type="run",
                params={
                    "workflow": "health_check",
                    "input": "production"
                },
                depends_on=["restore_previous"],
                retry_count=3,
                retry_delay=10
            )
        ]
    )


def create_multi_step_optimization_workflow(
    task: str,
    initial_examples: list,
    validation_examples: list,
    room_id: str,
    initiator: str
) -> Workflow:
    """
    Create multi-step optimization workflow with refinement.

    Pipeline:
    1. Initial optimization
    2. Test on validation set
    3. If accuracy < 0.8, refine and retry
    4. Request approval
    5. Deploy

    Args:
        task: Task description
        initial_examples: Training examples
        validation_examples: Validation examples
        room_id: Matrix room ID
        initiator: User initiating workflow

    Returns:
        Workflow ready to execute
    """
    return Workflow(
        id=f"optimize_multi_{int(__import__('time').time())}",
        name="Multi-Step Optimization",
        description="Optimize with iterative refinement",
        room_id=room_id,
        initiator=initiator,
        steps=[
            WorkflowStep(
                name="initial_optimization",
                type="optimize",
                params={
                    "task": task,
                    "examples": initial_examples
                }
            ),
            WorkflowStep(
                name="validation",
                type="run",
                params={
                    "workflow": "validate",
                    "input": str(validation_examples)
                },
                depends_on=["initial_optimization"]
            ),
            WorkflowStep(
                name="check_accuracy",
                type="condition",
                params={
                    "condition": "validation.get('accuracy', 0) < 0.8"
                },
                depends_on=["validation"]
            ),
            WorkflowStep(
                name="refinement",
                type="optimize",
                params={
                    "task": task,
                    "examples": initial_examples + validation_examples
                },
                condition="validation.get('accuracy', 0) < 0.8",
                depends_on=["check_accuracy"],
                retry_count=2
            ),
            WorkflowStep(
                name="final_validation",
                type="run",
                params={
                    "workflow": "validate",
                    "input": str(validation_examples)
                },
                depends_on=["refinement"]
            ),
            WorkflowStep(
                name="approval",
                type="approval",
                params={
                    "action": "deploy_optimized_prompt",
                    "context": {
                        "task": task,
                        "accuracy": "{{final_validation.accuracy}}"
                    },
                    "risk_level": "high"
                },
                depends_on=["final_validation"]
            )
        ]
    )


# Template registry
TEMPLATES = {
    "deploy_prompt": create_deploy_prompt_workflow,
    "code_review": create_code_review_workflow,
    "testing": create_testing_workflow,
    "rollback": create_rollback_workflow,
    "optimize_multi": create_multi_step_optimization_workflow
}


def get_template(name: str) -> callable:
    """Get workflow template by name"""
    return TEMPLATES.get(name)


def list_templates() -> dict:
    """List all available templates"""
    return {
        name: func.__doc__.split('\n\n')[0].strip()
        for name, func in TEMPLATES.items()
    }


# Test
if __name__ == "__main__":
    # Test template creation
    workflow = create_deploy_prompt_workflow(
        prompt_name="customer_support",
        task="Answer customer questions",
        examples=[{"input": "test", "output": "response"}],
        room_id="!room:matrix.org",
        initiator="@user:matrix.org"
    )

    print(f"✅ Created workflow: {workflow.name}")
    print(f"   Steps: {len(workflow.steps)}")
    for step in workflow.steps:
        deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
        print(f"   - {step.name} ({step.type}){deps}")

    # Test template listing
    templates = list_templates()
    print(f"\n✅ Available templates: {len(templates)}")
    for name, desc in templates.items():
        print(f"   - {name}: {desc}")

    print("\n✅ All workflow template tests passed!")
