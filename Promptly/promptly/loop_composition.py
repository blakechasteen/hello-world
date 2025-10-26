#!/usr/bin/env python3
"""
Loop Composition System
=======================
Chain multiple recursive loops together for complex reasoning pipelines.

Examples:
- DECOMPOSE → REFINE → VERIFY
- EXPLORE → HOFSTADTER → REFINE
- CRITIQUE → REFINE (repeat)
"""

from typing import List, Tuple, Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from enum import Enum

try:
    from recursive_loops import RecursiveEngine, LoopConfig, LoopType, LoopResult
    LOOPS_AVAILABLE = True
except ImportError:
    LOOPS_AVAILABLE = False
    print("[WARN] recursive_loops not available")


@dataclass
class CompositionStep:
    """Single step in a loop composition"""
    loop_type: LoopType
    config: LoopConfig
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.loop_type.value} loop"


@dataclass
class CompositionResult:
    """Result from composed loop execution"""
    steps: List[Tuple[str, LoopResult]]  # (description, result) pairs
    final_output: str
    total_iterations: int
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> str:
        """Generate execution report"""
        lines = [
            "# Loop Composition Report",
            "",
            f"**Steps Executed:** {len(self.steps)}",
            f"**Total Iterations:** {self.total_iterations}",
            ""
        ]

        lines.append("## Execution Pipeline")
        for i, (desc, result) in enumerate(self.steps, 1):
            lines.append(f"\n### Step {i}: {desc}")
            lines.append(f"- Iterations: {result.iterations}")
            lines.append(f"- Stop Reason: {result.stop_reason}")
            if result.improvement_history:
                avg_quality = sum(result.improvement_history) / len(result.improvement_history)
                lines.append(f"- Avg Quality: {avg_quality:.2f}")

        lines.append("\n## Final Output")
        lines.append(self.final_output)

        return "\n".join(lines)


class LoopComposer:
    """
    Compose multiple recursive loops into pipelines.

    Enables complex reasoning patterns like:
    1. Decompose problem into subtasks
    2. Refine each subtask
    3. Verify the solutions
    4. Synthesize final answer
    """

    def __init__(self, executor: Callable[[str], str]):
        """
        Initialize composer.

        Args:
            executor: Function that executes prompts (prompt -> output)
        """
        if not LOOPS_AVAILABLE:
            raise ImportError("recursive_loops module required")

        self.executor = executor
        self.engine = RecursiveEngine(executor)

    def compose(
        self,
        task: str,
        steps: List[CompositionStep],
        initial_output: Optional[str] = None
    ) -> CompositionResult:
        """
        Execute composed loop pipeline.

        Args:
            task: The original task
            steps: List of CompositionStep objects
            initial_output: Optional starting output (for first step)

        Returns:
            CompositionResult with full pipeline trace
        """
        results = []
        current_output = initial_output or task
        total_iterations = 0

        for step in steps:
            # Execute this loop step
            if step.loop_type == LoopType.REFINE:
                result = self.engine.execute_refine_loop(
                    task=task,
                    initial_output=current_output,
                    config=step.config
                )
            elif step.loop_type == LoopType.HOFSTADTER:
                result = self.engine.execute_hofstadter_loop(
                    task=task if not initial_output else current_output,
                    config=step.config
                )
            elif step.loop_type == LoopType.DECOMPOSE:
                result = self.engine.execute_decompose_loop(
                    task=current_output,
                    config=step.config
                )
            elif step.loop_type in [LoopType.CRITIQUE, LoopType.VERIFY, LoopType.EXPLORE]:
                # For these types, use refine loop as fallback with appropriate task
                result = self.engine.execute_refine_loop(
                    task=f"Process: {current_output}",
                    initial_output=current_output,
                    config=step.config
                )
            else:
                # Fallback to refine for unknown types
                result = self.engine.execute_refine_loop(
                    task=current_output,
                    initial_output=current_output,
                    config=step.config
                )

            # Store result
            results.append((step.description, result))
            current_output = result.final_output
            total_iterations += result.iterations

        return CompositionResult(
            steps=results,
            final_output=current_output,
            total_iterations=total_iterations,
            metadata={"num_steps": len(steps)}
        )

    def decompose_refine_verify(
        self,
        task: str,
        decompose_iterations: int = 1,
        refine_iterations: int = 3,
        verify_iterations: int = 1
    ) -> CompositionResult:
        """
        Common pattern: Decompose → Refine → Verify.

        Args:
            task: The complex task to solve
            decompose_iterations: Iterations for decomposition
            refine_iterations: Iterations for refinement
            verify_iterations: Iterations for verification

        Returns:
            CompositionResult
        """
        steps = [
            CompositionStep(
                loop_type=LoopType.DECOMPOSE,
                config=LoopConfig(
                    loop_type=LoopType.DECOMPOSE,
                    max_iterations=decompose_iterations
                ),
                description="Decompose complex task into subtasks"
            ),
            CompositionStep(
                loop_type=LoopType.REFINE,
                config=LoopConfig(
                    loop_type=LoopType.REFINE,
                    max_iterations=refine_iterations,
                    quality_threshold=0.85
                ),
                description="Refine each subtask solution"
            ),
            CompositionStep(
                loop_type=LoopType.VERIFY,
                config=LoopConfig(
                    loop_type=LoopType.VERIFY,
                    max_iterations=verify_iterations
                ),
                description="Verify the complete solution"
            )
        ]

        return self.compose(task, steps)

    def explore_synthesize_refine(
        self,
        task: str,
        explore_iterations: int = 3,
        hofstadter_levels: int = 2,
        refine_iterations: int = 2
    ) -> CompositionResult:
        """
        Creative pattern: Explore → Hofstadter → Refine.

        Args:
            task: The creative/complex task
            explore_iterations: Iterations for exploration
            hofstadter_levels: Meta-levels for Hofstadter loop
            refine_iterations: Iterations for final refinement

        Returns:
            CompositionResult
        """
        steps = [
            CompositionStep(
                loop_type=LoopType.EXPLORE,
                config=LoopConfig(
                    loop_type=LoopType.EXPLORE,
                    max_iterations=explore_iterations
                ),
                description="Explore multiple approaches"
            ),
            CompositionStep(
                loop_type=LoopType.HOFSTADTER,
                config=LoopConfig(
                    loop_type=LoopType.HOFSTADTER,
                    max_iterations=hofstadter_levels
                ),
                description="Meta-level synthesis"
            ),
            CompositionStep(
                loop_type=LoopType.REFINE,
                config=LoopConfig(
                    loop_type=LoopType.REFINE,
                    max_iterations=refine_iterations,
                    quality_threshold=0.85
                ),
                description="Final refinement"
            )
        ]

        return self.compose(task, steps)

    def iterative_critique_refine(
        self,
        task: str,
        initial_output: str,
        cycles: int = 2,
        refine_per_cycle: int = 2
    ) -> CompositionResult:
        """
        Iterative pattern: (Critique → Refine) × N.

        Args:
            task: The task
            initial_output: Starting output
            cycles: Number of critique-refine cycles
            refine_per_cycle: Refinements per cycle

        Returns:
            CompositionResult
        """
        steps = []

        for i in range(cycles):
            steps.extend([
                CompositionStep(
                    loop_type=LoopType.CRITIQUE,
                    config=LoopConfig(
                        loop_type=LoopType.CRITIQUE,
                        max_iterations=1
                    ),
                    description=f"Critique cycle {i+1}"
                ),
                CompositionStep(
                    loop_type=LoopType.REFINE,
                    config=LoopConfig(
                        loop_type=LoopType.REFINE,
                        max_iterations=refine_per_cycle,
                        quality_threshold=0.85
                    ),
                    description=f"Refine cycle {i+1}"
                )
            ])

        return self.compose(task, steps, initial_output=initial_output)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_composer(executor: Callable[[str], str]) -> LoopComposer:
    """Create loop composer instance"""
    return LoopComposer(executor)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Loop Composition System")
    print("\nCommon Patterns:")
    print("  1. Decompose -> Refine -> Verify (problem solving)")
    print("  2. Explore -> Hofstadter -> Refine (creative thinking)")
    print("  3. (Critique -> Refine) x N (iterative improvement)")
    print("\nAllows chaining any combination of 6 loop types!")
