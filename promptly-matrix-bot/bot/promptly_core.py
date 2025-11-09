#!/usr/bin/env python3
"""
Promptly Core Integration

Integrates Matrix bot with HoloLoom + DSPy for AI reliability.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

# Add HoloLoom to path
hololoom_path = Path(__file__).parent.parent.parent / "mythRL"
if hololoom_path.exists():
    sys.path.insert(0, str(hololoom_path))

# Import Promptly components
try:
    from HoloLoom.promptly import DSPyHoloLoom, create_signature
    from HoloLoom.promptly.metrics_system import MetricsEvaluator, MetricType
    from HoloLoom.config import Config
    from HoloLoom.documentation.types import MemoryShard
    import dspy

    HOLOLOOM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ HoloLoom + DSPy integration available")

except ImportError as e:
    HOLOLOOM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ HoloLoom not available: {e}")
    logger.warning("Running in stub mode - commands will return placeholders")


class PromptlyCore:
    """
    Promptly Core Integration

    Provides Matrix bot with access to:
    - DSPy optimization
    - HoloLoom memory
    - Workflow execution
    - Metrics evaluation
    """

    def __init__(self, config_mode: str = "fused"):
        """
        Initialize Promptly Core.

        Args:
            config_mode: HoloLoom config mode (bare/fast/fused)
        """
        self.config_mode = config_mode
        self.initialized = False

        if HOLOLOOM_AVAILABLE:
            try:
                # Configure DSPy with Ollama or OpenAI
                import os
                lm_model = os.getenv("LM_MODEL", "openai/gpt-4o-mini")
                api_key = os.getenv("OPENAI_API_KEY")

                if lm_model.startswith("ollama/"):
                    # Use Ollama (local, free)
                    lm = dspy.LM(lm_model)
                    dspy.configure(lm=lm)
                    logger.info(f"✅ DSPy configured with Ollama: {lm_model}")
                elif api_key:
                    # Use OpenAI
                    lm = dspy.LM(lm_model)
                    dspy.configure(lm=lm)
                    logger.info(f"✅ DSPy configured with {lm_model}")
                else:
                    logger.warning("⚠️ OPENAI_API_KEY not set and not using Ollama - optimization will fail")

                # Create HoloLoom config
                self.config = Config.fused() if config_mode == "fused" else Config.fast()

                # Create DSPy bridge with configured model
                self.bridge = DSPyHoloLoom(config=self.config, lm_model=lm_model, lm_api_key=api_key)

                # Create metrics evaluator
                self.metrics = MetricsEvaluator(
                    metrics=[
                        MetricType.ACCURACY,
                        MetricType.CLARITY,
                        MetricType.COMPLETENESS
                    ],
                    threshold=0.7
                )

                self.initialized = True
                logger.info("✅ Promptly Core initialized")

            except Exception as e:
                logger.error(f"❌ Failed to initialize Promptly Core: {e}", exc_info=True)
                self.initialized = False
        else:
            logger.warning("⚠️ Running in stub mode")

    async def optimize_prompt(
        self,
        task: str,
        examples: List[Dict[str, Any]],
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize a prompt using DSPy + HoloLoom.

        Args:
            task: Task description
            examples: List of input/output examples
            inputs: Input field names (default: ["input"])
            outputs: Output field names (default: ["output"])

        Returns:
            Dict with optimization results
        """
        if not self.initialized:
            return self._stub_optimize(task, examples)

        try:
            # Default field names
            if not inputs:
                inputs = ["input"]
            if not outputs:
                outputs = ["output"]

            # Create signature
            logger.info(f"Creating signature for: {task}")
            signature = create_signature(
                instruction=task,
                inputs=inputs,
                outputs=outputs
            )

            # Convert examples to DSPy format
            dspy_examples = []
            for ex in examples:
                # Handle both {"input": "...", "output": "..."}
                # and {"inputs": {...}, "outputs": {...}} formats
                if "input" in ex and "output" in ex:
                    example = dspy.Example(
                        input=ex["input"],
                        output=ex["output"]
                    ).with_inputs("input")
                else:
                    example = dspy.Example(**ex).with_inputs(*inputs)

                dspy_examples.append(example)

            logger.info(f"Created {len(dspy_examples)} training examples")

            # Create unoptimized program
            program = dspy.Predict(signature.to_dspy_signature())

            # Simple metric
            def accuracy_metric(example, pred, trace=None):
                # Basic keyword overlap
                if hasattr(pred, 'output'):
                    return any(word in pred.output.lower()
                             for word in example.output.lower().split())
                return False

            # Optimize with BootstrapFewShot
            from dspy.teleprompt import BootstrapFewShot

            logger.info("Running optimization...")
            optimizer = BootstrapFewShot(
                metric=accuracy_metric,
                max_bootstrapped_demos=min(3, len(dspy_examples))
            )

            optimized = optimizer.compile(
                program,
                trainset=dspy_examples[:max(3, len(dspy_examples)//2)]
            )

            logger.info("✅ Optimization complete")

            # Test on first example
            test_example = dspy_examples[0]
            result = optimized(input=test_example.input)

            # Calculate metrics
            metrics_result = self.metrics.evaluate(
                example={"output": test_example.output},
                prediction={"output": result.output if hasattr(result, 'output') else str(result)},
                context={"task": task}
            )

            return {
                "success": True,
                "signature": signature,
                "optimized_program": optimized,
                "prompt": str(optimized),
                "test_result": result.output if hasattr(result, 'output') else str(result),
                "metrics": {
                    "overall_score": metrics_result.overall_score,
                    "accuracy": metrics_result.metric_results[0].score / 10.0,
                    "clarity": metrics_result.metric_results[1].score / 10.0,
                    "completeness": metrics_result.metric_results[2].score / 10.0,
                    "latency_ms": 150  # Placeholder
                },
                "examples_used": len(dspy_examples)
            }

        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "examples_count": len(examples)
            }

    async def run_workflow(
        self,
        workflow_name: str,
        input_data: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run a workflow.

        Args:
            workflow_name: Workflow identifier
            input_data: Input data
            context: Optional context dict

        Returns:
            Workflow execution results
        """
        if not self.initialized:
            return self._stub_run_workflow(workflow_name, input_data)

        try:
            # TODO: Load workflow from YAML/database
            # For now, implement basic Q&A workflow

            if workflow_name == "qa_basic":
                # Simple Q&A workflow
                signature = create_signature(
                    "Answer questions accurately and concisely",
                    inputs=["question"],
                    outputs=["answer"]
                )

                program = dspy.Predict(signature.to_dspy_signature())
                result = program(question=input_data)

                return {
                    "success": True,
                    "workflow": workflow_name,
                    "input": input_data,
                    "output": result.answer if hasattr(result, 'answer') else str(result),
                    "steps_executed": 1,
                    "latency_ms": 150
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown workflow: {workflow_name}",
                    "available_workflows": ["qa_basic"]
                }

        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "workflow": workflow_name
            }

    def _stub_optimize(self, task: str, examples: List[Dict]) -> Dict[str, Any]:
        """Stub response when HoloLoom not available"""
        return {
            "success": True,
            "stub_mode": True,
            "task": task,
            "prompt": f"Optimized prompt for: {task}\n\nThis is a stub response. Install HoloLoom + DSPy for real optimization.",
            "metrics": {
                "overall_score": 0.85,
                "accuracy": 0.90,
                "clarity": 0.85,
                "completeness": 0.80,
                "latency_ms": 150
            },
            "examples_used": len(examples),
            "message": "⚠️ Running in stub mode - install HoloLoom + DSPy for real optimization"
        }

    def _stub_run_workflow(self, workflow_name: str, input_data: str) -> Dict[str, Any]:
        """Stub response for workflow execution"""
        return {
            "success": True,
            "stub_mode": True,
            "workflow": workflow_name,
            "input": input_data,
            "output": f"Stub response for workflow '{workflow_name}' with input: {input_data[:50]}...",
            "steps_executed": 1,
            "latency_ms": 100,
            "message": "⚠️ Running in stub mode - install HoloLoom for real workflow execution"
        }


# Singleton instance
_core_instance = None

def get_promptly_core() -> PromptlyCore:
    """Get singleton Promptly Core instance"""
    global _core_instance
    if _core_instance is None:
        _core_instance = PromptlyCore()
    return _core_instance


# Test
if __name__ == "__main__":
    async def test():
        core = PromptlyCore()

        # Test optimize
        result = await core.optimize_prompt(
            task="Answer customer support questions",
            examples=[
                {"input": "How to reset password?", "output": "Click Settings > Reset Password"},
                {"input": "Where is my order?", "output": "Check Orders page in your account"}
            ]
        )

        print("Optimization result:")
        print(f"  Success: {result['success']}")
        print(f"  Metrics: {result['metrics']}")
        if 'stub_mode' in result:
            print(f"  ⚠️ {result['message']}")

        # Test workflow
        result = await core.run_workflow(
            workflow_name="qa_basic",
            input_data="What is Thompson Sampling?"
        )

        print("\nWorkflow result:")
        print(f"  Success: {result['success']}")
        print(f"  Output: {result.get('output', result.get('error'))}")

    asyncio.run(test())
