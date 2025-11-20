"""
Chain Definition - Declarative chain structure

Provides:
- Chain: Complete chain definition
- ChainStep: Single step in a chain
- StepType: Enumeration of step types
- Validation and visualization

Author: HoloLoom Architecture Team
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import json


class StepType(Enum):
    """Step types in a chain."""
    EXECUTE = "execute"                    # Basic execution
    VERIFY = "verify"                      # Verification (e.g., DS-STAR checks)
    REFINE = "refine"                      # Refinement (improve quality)
    UPDATE_STRATEGY = "update_strategy"    # Learning/adaptation
    CONDITION = "condition"                # Conditional branching
    LOOP = "loop"                          # Loop construct
    PARALLEL = "parallel"                  # Parallel execution (future)
    CUSTOM = "custom"                      # Custom step type


@dataclass
class ChainStep:
    """Single step in a chain."""
    step_type: StepType
    params: Dict[str, Any] = field(default_factory=dict)
    next_step: Optional[str] = None  # Next step ID (sequential)
    condition: Optional[Callable] = None  # Conditional branch function
    on_success: Optional[str] = None  # Next step on success (alternative to next_step)
    on_failure: Optional[str] = None  # Next step on failure
    max_iterations: int = 1  # For loops
    timeout_seconds: Optional[float] = None  # Step timeout
    retry_count: int = 0  # Max retries on failure
    skip_condition: Optional[Callable] = None  # Skip step if condition true

    def validate(self) -> List[str]:
        """Validate step configuration."""
        errors = []

        if self.step_type is None:
            errors.append("step_type is required")

        if self.step_type == StepType.CONDITION and not self.condition:
            errors.append("condition() required for CONDITION steps")

        if self.step_type == StepType.LOOP:
            if self.max_iterations < 1:
                errors.append("max_iterations must be >= 1 for LOOP steps")
            if not self.next_step and not self.on_success:
                errors.append("next_step or on_success required for LOOP steps")

        return errors


@dataclass
class Chain:
    """Declarative chain definition."""
    name: str
    entry_point: str = "start"  # Starting step ID
    steps: Dict[str, ChainStep] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate chain on creation."""
        # Empty chain is valid (will be built with add_step)
        pass

    def add_step(self, step_id: str, step: ChainStep) -> None:
        """Add step to chain."""
        if step_id in self.steps:
            raise ValueError(f"Step '{step_id}' already exists")
        self.steps[step_id] = step

    def add_sequential_steps(self, step_ids: List[str], step_type: StepType,
                            params: Dict[str, Any] = None) -> None:
        """Add multiple sequential steps."""
        params = params or {}
        for i, step_id in enumerate(step_ids):
            next_step = step_ids[i + 1] if i < len(step_ids) - 1 else None
            step = ChainStep(
                step_type=step_type,
                params=params.copy(),
                next_step=next_step
            )
            self.add_step(step_id, step)

    def validate(self) -> List[str]:
        """Validate entire chain."""
        errors = []

        if not self.name:
            errors.append("Chain name is required")

        if not self.steps:
            errors.append("Chain has no steps")

        if self.entry_point not in self.steps:
            errors.append(f"Entry point '{self.entry_point}' not found in steps")

        # Validate each step
        for step_id, step in self.steps.items():
            step_errors = step.validate()
            errors.extend([f"Step '{step_id}': {e}" for e in step_errors])

        # Check for orphaned steps (unreachable)
        reachable = self._get_reachable_steps()
        for step_id in self.steps:
            if step_id not in reachable and step_id != self.entry_point:
                errors.append(f"Step '{step_id}' is unreachable")

        # Check for problematic cycles (allow intentional loops)
        # Note: Loops are allowed - they're intentional cycles
        # Only flag problematic cycles (where loop_step doesn't exist)
        problematic_cycles = []
        for cycle in self._detect_cycles():
            # Check if this is an intentional loop
            # (loop step exists and is configured for looping)
            if not any(step_id in self.steps and self.steps[step_id].max_iterations > 1
                      for step_id in cycle):
                problematic_cycles.append(cycle)

        if problematic_cycles:
            errors.append(f"Chain contains problematic cycles: {problematic_cycles}")

        return errors

    def _get_reachable_steps(self) -> Set[str]:
        """Get all reachable steps from entry point."""
        reachable = set()
        queue = [self.entry_point]

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue

            reachable.add(current)

            if current not in self.steps:
                continue

            step = self.steps[current]

            # Add next steps
            if step.next_step:
                queue.append(step.next_step)
            if step.on_success:
                queue.append(step.on_success)
            if step.on_failure:
                queue.append(step.on_failure)

        return reachable

    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the chain graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)

            if node not in self.steps:
                rec_stack.remove(node)
                return

            step = self.steps[node]
            next_steps = [s for s in [step.next_step, step.on_success, step.on_failure] if s]

            for next_step in next_steps:
                dfs(next_step, path + [node])

            rec_stack.remove(node)

        dfs(self.entry_point, [])
        return cycles

    def get_execution_order(self) -> List[str]:
        """Get step execution order (topological sort)."""
        order = []
        visited = set()

        def visit(step_id: str) -> None:
            if step_id in visited or step_id not in self.steps:
                return

            visited.add(step_id)
            order.append(step_id)

            # In chains, we can't do a true topological sort since execution
            # depends on runtime conditions. Instead, return entry point + all reachable.
            step = self.steps[step_id]
            if step.next_step:
                visit(step.next_step)
            if step.on_success:
                visit(step.on_success)
            if step.on_failure:
                visit(step.on_failure)

        visit(self.entry_point)
        return order

    def visualize(self) -> str:
        """Return ASCII visualization of chain."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Chain: {self.name}")
        lines.append(f"{'='*60}")

        visited = set()
        indent_level = {}

        def visit(step_id: str, depth: int = 0) -> None:
            if step_id in visited:
                return

            visited.add(step_id)
            indent_level[step_id] = depth

            if step_id not in self.steps:
                return

            step = self.steps[step_id]
            indent = "  " * depth

            # Print step
            lines.append(f"{indent}┌─ {step_id} [{step.step_type.value}]")

            # Print params
            if step.params:
                for key, value in step.params.items():
                    lines.append(f"{indent}│  {key}: {value}")

            # Print conditions
            if step.condition:
                lines.append(f"{indent}│  [condition: dynamic]")

            # Follow execution paths
            if step.next_step:
                lines.append(f"{indent}│")
                visit(step.next_step, depth + 1)
            elif step.on_success or step.on_failure:
                if step.on_success:
                    lines.append(f"{indent}├─ [success] ↓")
                    visit(step.on_success, depth + 1)
                if step.on_failure:
                    lines.append(f"{indent}└─ [failure] ↓")
                    visit(step.on_failure, depth + 1)
            else:
                lines.append(f"{indent}└─ [end]")

        visit(self.entry_point)

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary (for JSON serialization)."""
        return {
            "name": self.name,
            "entry_point": self.entry_point,
            "steps": {
                step_id: {
                    "step_type": step.step_type.value,
                    "params": step.params,
                    "next_step": step.next_step,
                    "on_success": step.on_success,
                    "on_failure": step.on_failure,
                    "max_iterations": step.max_iterations,
                    "timeout_seconds": step.timeout_seconds,
                    "retry_count": step.retry_count,
                }
                for step_id, step in self.steps.items()
            },
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
