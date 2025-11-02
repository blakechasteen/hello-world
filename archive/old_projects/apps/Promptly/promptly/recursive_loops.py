#!/usr/bin/env python3
"""
Promptly Recursive Intelligence System
=======================================
Implements recursive loops, scratchpad reasoning, and iterative refinement.

Inspired by:
- Hofstadter's Strange Loops (GEB)
- Samsung's recursive tiny models
- Scratchpad/chain-of-thought reasoning
- Self-reflective improvement loops
"""

import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class LoopType(Enum):
    """Types of recursive loops"""
    REFINE = "refine"  # Iterative refinement
    CRITIQUE = "critique"  # Self-critique loop
    DECOMPOSE = "decompose"  # Break down → solve → combine
    VERIFY = "verify"  # Generate → verify → improve
    EXPLORE = "explore"  # Multiple approaches → synthesize
    HOFSTADTER = "hofstadter"  # Strange loop (self-reference)


class StopCondition(Enum):
    """When to stop looping"""
    MAX_ITERATIONS = "max_iterations"
    QUALITY_THRESHOLD = "quality_threshold"
    NO_IMPROVEMENT = "no_improvement"
    CONVERGENCE = "convergence"
    USER_SATISFIED = "user_satisfied"


@dataclass
class ScratchpadEntry:
    """Single entry in reasoning scratchpad"""
    iteration: int
    thought: str
    action: str
    observation: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Format as readable text"""
        lines = [
            f"## Iteration {self.iteration}",
            f"**Thought:** {self.thought}",
            f"**Action:** {self.action}",
            f"**Observation:** {self.observation}"
        ]
        if self.score is not None:
            lines.append(f"**Score:** {self.score:.2f}")
        return "\n".join(lines)


@dataclass
class Scratchpad:
    """Reasoning scratchpad for tracking thought process"""
    entries: List[ScratchpadEntry] = field(default_factory=list)
    final_answer: Optional[str] = None

    def add_entry(
        self,
        thought: str,
        action: str,
        observation: str,
        score: Optional[float] = None
    ):
        """Add a reasoning step"""
        self.entries.append(ScratchpadEntry(
            iteration=len(self.entries) + 1,
            thought=thought,
            action=action,
            observation=observation,
            score=score
        ))

    def get_history(self) -> str:
        """Get full reasoning history"""
        return "\n\n".join(e.to_text() for e in self.entries)

    def get_latest(self) -> Optional[ScratchpadEntry]:
        """Get latest entry"""
        return self.entries[-1] if self.entries else None


@dataclass
class LoopConfig:
    """Configuration for recursive loop"""
    loop_type: LoopType
    max_iterations: int = 5
    quality_threshold: float = 0.9
    min_improvement: float = 0.05
    enable_scratchpad: bool = True
    verbose: bool = True


@dataclass
class LoopResult:
    """Result from recursive loop execution"""
    success: bool
    final_output: str
    iterations: int
    scratchpad: Optional[Scratchpad]
    improvement_history: List[float]
    stop_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> str:
        """Generate execution report"""
        lines = [
            "# Recursive Loop Execution Report",
            "",
            f"**Success:** {'Yes' if self.success else 'No'}",
            f"**Iterations:** {self.iterations}",
            f"**Stop Reason:** {self.stop_reason}",
            ""
        ]

        if self.improvement_history:
            lines.append("## Improvement Trajectory")
            for i, score in enumerate(self.improvement_history, 1):
                lines.append(f"Iteration {i}: {score:.2f}")
            lines.append("")

        if self.scratchpad and self.scratchpad.entries:
            lines.append("## Reasoning Process")
            lines.append(self.scratchpad.get_history())
            lines.append("")

        lines.append("## Final Output")
        lines.append(self.final_output)

        return "\n".join(lines)


class RecursiveEngine:
    """Engine for executing recursive loops"""

    def __init__(self, executor: Callable[[str], str]):
        """
        Initialize recursive engine.

        Args:
            executor: Function that executes prompts (prompt -> output)
        """
        self.executor = executor

    def _build_refine_prompt(
        self,
        task: str,
        current_output: str,
        iteration: int,
        scratchpad: Optional[Scratchpad] = None
    ) -> str:
        """Build prompt for refinement iteration"""
        parts = [
            "You are refining your previous response.",
            "",
            f"## Original Task",
            task,
            "",
            f"## Current Output (Iteration {iteration})",
            current_output,
            ""
        ]

        if scratchpad and scratchpad.entries:
            parts.extend([
                "## Previous Reasoning",
                scratchpad.get_history(),
                ""
            ])

        parts.extend([
            "## Instructions",
            "1. Identify weaknesses in the current output",
            "2. Think about how to improve it",
            "3. Produce a refined version",
            "",
            "Provide your response in this format:",
            "THOUGHT: [What needs improvement]",
            "IMPROVED: [Your refined output]"
        ])

        return "\n".join(parts)

    def _build_critique_prompt(
        self,
        task: str,
        output: str
    ) -> str:
        """Build prompt for self-critique"""
        return f"""You are critiquing your own work.

## Task
{task}

## Your Output
{output}

## Critique Instructions
Evaluate your output on:
1. Correctness
2. Completeness
3. Clarity
4. Accuracy

Provide critique in format:
SCORE: [0-10]
STRENGTHS: [What works well]
WEAKNESSES: [What needs improvement]
SUGGESTIONS: [How to improve]
"""

    def _build_decompose_prompt(
        self,
        task: str,
        scratchpad: Scratchpad
    ) -> str:
        """Build prompt for decomposition strategy"""
        return f"""Break down this complex task into subtasks.

## Complex Task
{task}

## Instructions
1. Decompose the task into 3-5 smaller subtasks
2. Solve each subtask independently
3. Synthesize the solutions

Format:
SUBTASKS: [List of subtasks]
SOLUTIONS: [Solution for each]
SYNTHESIS: [Combined final answer]
"""

    def _build_hofstadter_prompt(
        self,
        task: str,
        iteration: int,
        previous_outputs: List[str]
    ) -> str:
        """Build Hofstadter strange loop prompt (self-referential)"""
        parts = [
            "You are in a self-referential thinking loop.",
            "",
            f"## Level {iteration} Task",
            task,
            ""
        ]

        if previous_outputs:
            parts.extend([
                "## Previous Level Outputs",
                *[f"Level {i}: {out[:200]}..." for i, out in enumerate(previous_outputs, 1)],
                ""
            ])

        parts.extend([
            "## Instructions",
            "1. Reflect on the previous levels of thinking",
            "2. Notice patterns in your own reasoning",
            "3. Use this meta-awareness to think at a higher level",
            "4. Generate output that references and builds upon previous levels",
            "",
            "REFLECTION: [What you notice about your thinking]",
            "META_INSIGHT: [Pattern across levels]",
            "OUTPUT: [Your answer incorporating meta-awareness]"
        ])

        return "\n".join(parts)

    def _parse_thought_action(self, response: str) -> tuple[str, str, str]:
        """Parse thought/action/observation from response"""
        thought = ""
        action = ""
        observation = response

        # Try to extract structured parts
        for line in response.split('\n'):
            if line.startswith('THOUGHT:'):
                thought = line.replace('THOUGHT:', '').strip()
            elif line.startswith('ACTION:'):
                action = line.replace('ACTION:', '').strip()
            elif line.startswith('IMPROVED:'):
                observation = line.replace('IMPROVED:', '').strip()
            elif line.startswith('OUTPUT:'):
                observation = line.replace('OUTPUT:', '').strip()

        if not observation or observation == response:
            observation = response

        return thought, action, observation

    def execute_refine_loop(
        self,
        task: str,
        initial_output: str,
        config: LoopConfig
    ) -> LoopResult:
        """
        Execute iterative refinement loop.

        Args:
            task: The original task
            initial_output: Starting output to refine
            config: Loop configuration

        Returns:
            LoopResult with refinement history
        """
        scratchpad = Scratchpad() if config.enable_scratchpad else None
        current_output = initial_output
        improvements = []

        for iteration in range(1, config.max_iterations + 1):
            # Build refinement prompt
            prompt = self._build_refine_prompt(
                task, current_output, iteration, scratchpad
            )

            # Execute
            response = self.executor(prompt)

            # Parse
            thought, action, improved = self._parse_thought_action(response)

            # Critique for quality score
            critique_prompt = self._build_critique_prompt(task, improved)
            critique = self.executor(critique_prompt)

            # Extract score
            score = 0.5
            for line in critique.split('\n'):
                if line.startswith('SCORE:'):
                    try:
                        score = float(line.replace('SCORE:', '').strip()) / 10.0
                    except:
                        pass

            improvements.append(score)

            # Update scratchpad
            if scratchpad:
                scratchpad.add_entry(
                    thought=thought or f"Refining iteration {iteration}",
                    action=action or "Improve output",
                    observation=improved[:200] + "..." if len(improved) > 200 else improved,
                    score=score
                )

            # Check stop conditions
            if score >= config.quality_threshold:
                return LoopResult(
                    success=True,
                    final_output=improved,
                    iterations=iteration,
                    scratchpad=scratchpad,
                    improvement_history=improvements,
                    stop_reason=f"Quality threshold reached ({score:.2f})"
                )

            if iteration > 1 and abs(improvements[-1] - improvements[-2]) < config.min_improvement:
                return LoopResult(
                    success=True,
                    final_output=improved,
                    iterations=iteration,
                    scratchpad=scratchpad,
                    improvement_history=improvements,
                    stop_reason="No significant improvement"
                )

            current_output = improved

        # Max iterations reached
        return LoopResult(
            success=True,
            final_output=current_output,
            iterations=config.max_iterations,
            scratchpad=scratchpad,
            improvement_history=improvements,
            stop_reason="Max iterations reached"
        )

    def execute_hofstadter_loop(
        self,
        task: str,
        config: LoopConfig
    ) -> LoopResult:
        """
        Execute Hofstadter strange loop (self-referential reasoning).

        Each level thinks about the previous level's thinking.
        """
        outputs = []
        scratchpad = Scratchpad() if config.enable_scratchpad else None

        for iteration in range(1, config.max_iterations + 1):
            # Build self-referential prompt
            prompt = self._build_hofstadter_prompt(task, iteration, outputs)

            # Execute
            response = self.executor(prompt)

            # Parse meta-level thinking
            reflection = ""
            meta_insight = ""
            output = response

            for line in response.split('\n'):
                if line.startswith('REFLECTION:'):
                    reflection = line.replace('REFLECTION:', '').strip()
                elif line.startswith('META_INSIGHT:'):
                    meta_insight = line.replace('META_INSIGHT:', '').strip()
                elif line.startswith('OUTPUT:'):
                    output = line.replace('OUTPUT:', '').strip()

            outputs.append(output)

            # Update scratchpad
            if scratchpad:
                scratchpad.add_entry(
                    thought=reflection,
                    action=f"Meta-level {iteration} reasoning",
                    observation=meta_insight
                )

        # Final synthesis: ask to combine all levels
        synthesis_prompt = f"""You've thought about this task at {len(outputs)} different meta-levels.

Task: {task}

Level outputs:
{chr(10).join(f'Level {i}: {o}' for i, o in enumerate(outputs, 1))}

Synthesize all levels into a final answer that incorporates the meta-insights.
"""

        final_output = self.executor(synthesis_prompt)

        return LoopResult(
            success=True,
            final_output=final_output,
            iterations=len(outputs),
            scratchpad=scratchpad,
            improvement_history=[],
            stop_reason="Strange loop complete",
            metadata={"levels": outputs}
        )

    def execute_decompose_loop(
        self,
        task: str,
        config: LoopConfig
    ) -> LoopResult:
        """Execute decomposition strategy (divide and conquer)"""
        scratchpad = Scratchpad() if config.enable_scratchpad else None

        # Step 1: Decompose
        decompose_prompt = self._build_decompose_prompt(task, scratchpad)
        response = self.executor(decompose_prompt)

        # Parse subtasks and solutions
        subtasks = []
        solutions = []
        synthesis = response

        current_section = None
        for line in response.split('\n'):
            if line.startswith('SUBTASKS:'):
                current_section = 'subtasks'
            elif line.startswith('SOLUTIONS:'):
                current_section = 'solutions'
            elif line.startswith('SYNTHESIS:'):
                current_section = 'synthesis'
                synthesis = ""
            elif current_section == 'subtasks' and line.strip().startswith(('-', '*', '1', '2', '3')):
                subtasks.append(line.strip())
            elif current_section == 'synthesis':
                synthesis += line + "\n"

        if scratchpad:
            scratchpad.add_entry(
                thought=f"Decomposed into {len(subtasks)} subtasks",
                action="Divide and conquer",
                observation=synthesis
            )

        return LoopResult(
            success=True,
            final_output=synthesis.strip(),
            iterations=1,
            scratchpad=scratchpad,
            improvement_history=[],
            stop_reason="Decomposition complete",
            metadata={"subtasks": subtasks}
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def refine_iteratively(
    executor: Callable[[str], str],
    task: str,
    initial_output: str,
    max_iterations: int = 3
) -> str:
    """Quick iterative refinement"""
    engine = RecursiveEngine(executor)
    config = LoopConfig(loop_type=LoopType.REFINE, max_iterations=max_iterations)
    result = engine.execute_refine_loop(task, initial_output, config)
    return result.final_output


def think_recursively(
    executor: Callable[[str], str],
    task: str,
    levels: int = 3
) -> str:
    """Quick Hofstadter strange loop"""
    engine = RecursiveEngine(executor)
    config = LoopConfig(loop_type=LoopType.HOFSTADTER, max_iterations=levels)
    result = engine.execute_hofstadter_loop(task, config)
    return result.final_output


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Promptly Recursive Intelligence System")
    print("\nLoop Types:")
    print("- REFINE: Iterative refinement")
    print("- CRITIQUE: Self-critique loop")
    print("- DECOMPOSE: Divide and conquer")
    print("- VERIFY: Generate → verify → improve")
    print("- EXPLORE: Multiple approaches → synthesize")
    print("- HOFSTADTER: Strange loop (self-referential)")
    print("\nExample:")
    print("""
from execution_engine import execute_with_ollama
from recursive_loops import RecursiveEngine, LoopConfig, LoopType

# Setup
executor = lambda p: execute_with_ollama(p).output
engine = RecursiveEngine(executor)

# Configure loop
config = LoopConfig(
    loop_type=LoopType.REFINE,
    max_iterations=5,
    quality_threshold=0.9
)

# Execute refinement loop
result = engine.execute_refine_loop(
    task="Explain quantum entanglement",
    initial_output="Quantum particles are connected...",
    config=config
)

print(result.to_report())
""")
