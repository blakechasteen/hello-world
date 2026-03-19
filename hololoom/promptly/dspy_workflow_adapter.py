#!/usr/bin/env python3
"""
DSPy-Promptly Workflow Adapter

Integrates DSPy programs into Promptly's workflow system, enabling:
- DSPy signatures as workflow steps
- Automatic optimization workflows
- Chained DSPy programs
- Prompt A/B testing with DSPy

Usage:
    from hololoom.promptly.dspy_workflow_adapter import DSPyWorkflowAdapter

    adapter = DSPyWorkflowAdapter(bridge)

    # Register DSPy step
    adapter.register_dspy_step("answer_question", question_sig)

    # Create workflow with DSPy steps
    workflow = adapter.create_workflow([
        {"step": "answer_question", "inputs": {"question": "{query}"}},
        {"step": "verify_answer", "inputs": {"answer": "{answer_question.answer}"}}
    ])

    # Execute
    result = await adapter.execute_workflow(workflow, {"query": "What is DSPy?"})
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hololoom.promptly.dspy_bridge import DSPyHoloLoom, DSPyProgram, DSPySignature, create_signature
from hololoom.promptly.workflow_store import WorkflowStore


@dataclass
class DSPyWorkflowStep:
    """A single DSPy step in a workflow"""
    step_id: str
    signature_name: str
    inputs: dict[str, str]  # Field name → value or reference
    outputs: list[str]
    optimize: bool = False
    optimization_query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DSPyWorkflow:
    """Complete DSPy workflow"""
    name: str
    description: str
    steps: list[DSPyWorkflowStep]
    metadata: dict[str, Any] = field(default_factory=dict)


class DSPyWorkflowAdapter:
    """
    Adapter between DSPy and Promptly workflows.

    Enables using DSPy programs as workflow steps with:
    - Automatic input/output mapping
    - Chaining between steps
    - Optimization workflows
    - Error handling and retries
    """

    def __init__(
        self,
        bridge: DSPyHoloLoom,
        workflow_store: WorkflowStore | None = None
    ):
        """
        Initialize adapter.

        Args:
            bridge: DSPyHoloLoom bridge
            workflow_store: Optional workflow persistence
        """
        self.bridge = bridge
        self.workflow_store = workflow_store

        # Registry of available signatures
        self.signatures: dict[str, DSPySignature] = {}

        # Cache of optimized programs
        self.programs: dict[str, DSPyProgram] = {}

        # Workflow execution history
        self.execution_history: list[dict[str, Any]] = []

        logging.info("DSPyWorkflowAdapter initialized")

    def register_signature(self, signature: DSPySignature):
        """Register a DSPy signature for use in workflows"""
        self.signatures[signature.name] = signature
        logging.info(f"Registered signature: {signature.name}")

    def create_workflow(
        self,
        name: str,
        description: str,
        steps: list[dict[str, Any]]
    ) -> DSPyWorkflow:
        """
        Create a DSPy workflow.

        Args:
            name: Workflow name
            description: Human-readable description
            steps: List of step definitions

        Returns:
            DSPyWorkflow instance

        Example step format:
            {
                "step_id": "answer_q1",
                "signature": "QuestionAnswering",
                "inputs": {
                    "question": "{query}",
                    "context": "{retrieved_context}"
                },
                "outputs": ["answer", "confidence"],
                "optimize": True,
                "optimization_query": "Q&A examples"
            }
        """
        workflow_steps = []

        for step_def in steps:
            sig_name = step_def["signature"]

            if sig_name not in self.signatures:
                raise ValueError(f"Signature '{sig_name}' not registered")

            step = DSPyWorkflowStep(
                step_id=step_def["step_id"],
                signature_name=sig_name,
                inputs=step_def["inputs"],
                outputs=step_def.get("outputs", []),
                optimize=step_def.get("optimize", False),
                optimization_query=step_def.get("optimization_query"),
                metadata=step_def.get("metadata", {})
            )

            workflow_steps.append(step)

        workflow = DSPyWorkflow(
            name=name,
            description=description,
            steps=workflow_steps
        )

        logging.info(f"Created workflow: {name} with {len(workflow_steps)} steps")

        return workflow

    async def execute_workflow(
        self,
        workflow: DSPyWorkflow,
        initial_inputs: dict[str, Any],
        optimize_on_the_fly: bool = False
    ) -> dict[str, Any]:
        """
        Execute a DSPy workflow.

        Args:
            workflow: Workflow to execute
            initial_inputs: Initial input values
            optimize_on_the_fly: Optimize programs during execution if needed

        Returns:
            Final outputs and execution trace
        """
        logging.info(f"Executing workflow: {workflow.name}")

        # Execution context (accumulates outputs)
        context = initial_inputs.copy()

        # Execution trace
        trace = []

        # Execute each step
        for i, step in enumerate(workflow.steps):
            logging.info(f"  Step {i+1}/{len(workflow.steps)}: {step.step_id}")

            try:
                # Get or optimize program
                program = await self._get_or_optimize_program(
                    step,
                    optimize_on_the_fly
                )

                # Resolve inputs from context
                resolved_inputs = self._resolve_inputs(step.inputs, context)

                # Execute program
                step_start = asyncio.get_event_loop().time()

                outputs = await self.bridge.execute(
                    program,
                    use_hololoom_context=False,  # Already resolved
                    **resolved_inputs
                )

                step_duration = asyncio.get_event_loop().time() - step_start

                # Update context with outputs
                for output_key, output_value in outputs.items():
                    context[f"{step.step_id}.{output_key}"] = output_value

                # Add to trace
                trace.append({
                    "step_id": step.step_id,
                    "signature": step.signature_name,
                    "inputs": resolved_inputs,
                    "outputs": outputs,
                    "duration_ms": step_duration * 1000,
                    "success": True
                })

                logging.info(f"    ✓ Completed in {step_duration*1000:.1f}ms")

            except Exception as e:
                logging.error(f"    ✗ Failed: {e}")

                # Add error to trace
                trace.append({
                    "step_id": step.step_id,
                    "signature": step.signature_name,
                    "error": str(e),
                    "success": False
                })

                # Stop execution on error
                break

        # Save to execution history
        execution_record = {
            "workflow": workflow.name,
            "initial_inputs": initial_inputs,
            "final_context": context,
            "trace": trace,
            "timestamp": asyncio.get_event_loop().time()
        }

        self.execution_history.append(execution_record)

        logging.info(f"Workflow complete: {len(trace)} steps executed")

        return {
            "success": all(t["success"] for t in trace),
            "context": context,
            "trace": trace
        }

    async def _get_or_optimize_program(
        self,
        step: DSPyWorkflowStep,
        optimize_on_the_fly: bool
    ) -> DSPyProgram:
        """Get cached program or optimize if needed"""

        cache_key = f"{step.signature_name}_{step.step_id}"

        # Check cache
        if cache_key in self.programs:
            return self.programs[cache_key]

        # Check bridge cache
        if step.signature_name in self.bridge.program_cache:
            return self.bridge.program_cache[step.signature_name]

        # Optimize if requested
        if step.optimize and step.optimization_query:
            if optimize_on_the_fly:
                logging.info(f"    Optimizing {step.signature_name}...")

                signature = self.signatures[step.signature_name]

                program = await self.bridge.optimize_from_memory(
                    signature=signature,
                    memory_query=step.optimization_query
                )

                self.programs[cache_key] = program

                return program
            else:
                logging.warning(f"    Step {step.step_id} requires optimization but optimize_on_the_fly=False")

        # Create unoptimized program
        import dspy

        signature = self.signatures[step.signature_name]
        dspy_sig = signature.to_dspy_signature()

        base_program = dspy.Predict(dspy_sig)

        program = DSPyProgram(
            signature=signature,
            program=base_program,
            optimizer="none",
            training_examples=0,
            validation_score=0.0,
            created_at=str(asyncio.get_event_loop().time())
        )

        self.programs[cache_key] = program

        return program

    def _resolve_inputs(
        self,
        input_spec: dict[str, str],
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Resolve input values from context.

        Supports:
        - Literal values: "hello"
        - Context references: "{query}"
        - Step output references: "{step1.answer}"
        """
        resolved = {}

        for field_name, value_spec in input_spec.items():
            if isinstance(value_spec, str) and value_spec.startswith("{") and value_spec.endswith("}"):
                # Reference to context
                ref_key = value_spec[1:-1]  # Remove { }

                if ref_key in context:
                    resolved[field_name] = context[ref_key]
                else:
                    logging.warning(f"Context key '{ref_key}' not found, using empty string")
                    resolved[field_name] = ""
            else:
                # Literal value
                resolved[field_name] = value_spec

        return resolved

    async def save_workflow(self, workflow: DSPyWorkflow, path: Path):
        """Save workflow to YAML"""
        workflow_dict = {
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "step_id": step.step_id,
                    "signature": step.signature_name,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                    "optimize": step.optimize,
                    "optimization_query": step.optimization_query,
                    "metadata": step.metadata
                }
                for step in workflow.steps
            ],
            "metadata": workflow.metadata
        }

        with open(path, 'w') as f:
            yaml.dump(workflow_dict, f, default_flow_style=False, indent=2)

        logging.info(f"Saved workflow to {path}")

    async def load_workflow(self, path: Path) -> DSPyWorkflow:
        """Load workflow from YAML"""
        with open(path) as f:
            workflow_dict = yaml.safe_load(f)

        workflow = self.create_workflow(
            name=workflow_dict["name"],
            description=workflow_dict["description"],
            steps=workflow_dict["steps"]
        )

        workflow.metadata = workflow_dict.get("metadata", {})

        logging.info(f"Loaded workflow from {path}")

        return workflow

    async def optimize_workflow(
        self,
        workflow: DSPyWorkflow,
        force_reoptimize: bool = False
    ):
        """Pre-optimize all steps in a workflow"""
        logging.info(f"Optimizing workflow: {workflow.name}")

        for step in workflow.steps:
            if step.optimize and step.optimization_query:
                cache_key = f"{step.signature_name}_{step.step_id}"

                if cache_key in self.programs and not force_reoptimize:
                    logging.info(f"  Step {step.step_id}: already optimized (cached)")
                    continue

                logging.info(f"  Step {step.step_id}: optimizing...")

                signature = self.signatures[step.signature_name]

                try:
                    program = await self.bridge.optimize_from_memory(
                        signature=signature,
                        memory_query=step.optimization_query
                    )

                    self.programs[cache_key] = program

                    logging.info(f"    ✓ Score: {program.validation_score:.3f}")

                except Exception as e:
                    logging.error(f"    ✗ Failed: {e}")

        logging.info("Workflow optimization complete")

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get workflow execution statistics"""
        if not self.execution_history:
            return {"total_executions": 0}

        total = len(self.execution_history)
        successful = sum(1 for ex in self.execution_history if all(t["success"] for t in ex["trace"]))

        total_steps = sum(len(ex["trace"]) for ex in self.execution_history)
        successful_steps = sum(
            sum(1 for t in ex["trace"] if t["success"])
            for ex in self.execution_history
        )

        avg_duration = sum(
            sum(t.get("duration_ms", 0) for t in ex["trace"])
            for ex in self.execution_history
        ) / total if total > 0 else 0

        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "step_success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "avg_duration_ms": avg_duration
        }


# Common workflow patterns

async def create_qa_workflow(adapter: DSPyWorkflowAdapter) -> DSPyWorkflow:
    """Create a question-answering workflow"""

    # Register signatures
    adapter.register_signature(create_signature(
        "Retrieve relevant context",
        inputs=["question"],
        outputs=["context"],
        name="ContextRetrieval"
    ))

    adapter.register_signature(create_signature(
        "Answer question with context",
        inputs=["question", "context"],
        outputs=["answer", "confidence"],
        name="QuestionAnswering"
    ))

    adapter.register_signature(create_signature(
        "Verify answer accuracy",
        inputs=["question", "answer"],
        outputs=["verification", "is_accurate"],
        name="AnswerVerification"
    ))

    # Create workflow
    workflow = adapter.create_workflow(
        name="QA_Pipeline",
        description="Question answering with retrieval and verification",
        steps=[
            {
                "step_id": "retrieve",
                "signature": "ContextRetrieval",
                "inputs": {"question": "{query}"},
                "outputs": ["context"],
                "optimize": True,
                "optimization_query": "context retrieval examples"
            },
            {
                "step_id": "answer",
                "signature": "QuestionAnswering",
                "inputs": {
                    "question": "{query}",
                    "context": "{retrieve.context}"
                },
                "outputs": ["answer", "confidence"],
                "optimize": True,
                "optimization_query": "question answering examples"
            },
            {
                "step_id": "verify",
                "signature": "AnswerVerification",
                "inputs": {
                    "question": "{query}",
                    "answer": "{answer.answer}"
                },
                "outputs": ["verification", "is_accurate"]
            }
        ]
    )

    return workflow


async def create_research_workflow(adapter: DSPyWorkflowAdapter) -> DSPyWorkflow:
    """Create a multi-query research workflow"""

    adapter.register_signature(create_signature(
        "Break down research question",
        inputs=["research_question"],
        outputs=["sub_questions"],
        name="QuestionDecomposition"
    ))

    adapter.register_signature(create_signature(
        "Answer single question",
        inputs=["question", "context"],
        outputs=["answer"],
        name="SimpleQA"
    ))

    adapter.register_signature(create_signature(
        "Synthesize multiple answers",
        inputs=["research_question", "answers"],
        outputs=["synthesis", "sources"],
        name="AnswerSynthesis"
    ))

    workflow = adapter.create_workflow(
        name="Research_Pipeline",
        description="Multi-query research with synthesis",
        steps=[
            {
                "step_id": "decompose",
                "signature": "QuestionDecomposition",
                "inputs": {"research_question": "{query}"},
                "outputs": ["sub_questions"]
            },
            {
                "step_id": "answer_subq",
                "signature": "SimpleQA",
                "inputs": {
                    "question": "{decompose.sub_questions}",
                    "context": ""
                },
                "outputs": ["answer"]
            },
            {
                "step_id": "synthesize",
                "signature": "AnswerSynthesis",
                "inputs": {
                    "research_question": "{query}",
                    "answers": "{answer_subq.answer}"
                },
                "outputs": ["synthesis", "sources"]
            }
        ]
    )

    return workflow


# Demo
async def demo_workflow_adapter():
    """Demonstrate DSPy workflow adapter"""

    print("🎯 DSPy Workflow Adapter Demo\n")

    from hololoom.config import Config

    # Initialize
    bridge = DSPyHoloLoom(config=Config.fused(), lm_model="openai/gpt-4o-mini")
    adapter = DSPyWorkflowAdapter(bridge)

    # Create QA workflow
    print("1️⃣  Creating QA workflow...")
    workflow = await create_qa_workflow(adapter)
    print(f"   Created: {workflow.name} with {len(workflow.steps)} steps\n")

    # Execute workflow
    print("2️⃣  Executing workflow...")
    result = await adapter.execute_workflow(
        workflow,
        initial_inputs={"query": "What is Thompson Sampling?"},
        optimize_on_the_fly=False
    )

    print(f"   Success: {result['success']}")
    print(f"   Steps: {len(result['trace'])}")
    print()

    # Show trace
    print("3️⃣  Execution trace:")
    for step in result["trace"]:
        status = "✓" if step["success"] else "✗"
        duration = step.get("duration_ms", 0)
        print(f"   {status} {step['step_id']}: {duration:.1f}ms")
    print()

    # Statistics
    print("4️⃣  Statistics:")
    stats = adapter.get_execution_statistics()
    print(f"   Executions: {stats['total_executions']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Avg duration: {stats['avg_duration_ms']:.1f}ms")
    print()

    print("✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_workflow_adapter())
