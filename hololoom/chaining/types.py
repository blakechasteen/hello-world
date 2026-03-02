"""
Chain Orchestration Result and Configuration Types

Provides data structures for:
- Step execution results
- Loop configuration
- Conditional branching
- Context management
- Execution metadata

Author: HoloLoom Architecture Team
Date: November 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class StepStatus(Enum):
    """Status of step execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CONDITIONAL_BRANCH = "conditional_branch"


@dataclass
class StepResult:
    """Result of a single step execution."""
    step_id: str
    step_type: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopConfig:
    """Configuration for loop steps."""
    condition: Optional[Callable] = None  # Loop while condition is true
    max_iterations: int = 10  # Max iterations (safety limit)
    exit_on_success: bool = False  # Exit if step succeeds
    exit_on_failure: bool = False  # Exit if step fails
    increment_counter: bool = True  # Auto-increment iteration counter


@dataclass
class ConditionalBranch:
    """Configuration for conditional branching."""
    condition: Callable  # Returns bool
    true_step: str  # Next step if condition is true
    false_step: Optional[str] = None  # Next step if condition is false (optional)


@dataclass
class ExecutionContext:
    """Shared context across chain execution."""
    initial_input: Any = None
    shared_state: Dict[str, Any] = field(default_factory=dict)
    step_outputs: Dict[str, Any] = field(default_factory=dict)  # step_id -> output
    loop_counters: Dict[str, int] = field(default_factory=dict)  # loop_id -> count
    errors: List[str] = field(default_factory=list)

    def get_step_output(self, step_id: str) -> Any:
        """Get output from a previous step."""
        return self.step_outputs.get(step_id)

    def set_step_output(self, step_id: str, output: Any) -> None:
        """Store output from a step."""
        self.step_outputs[step_id] = output

    def get_loop_counter(self, loop_id: str) -> int:
        """Get current iteration count for a loop."""
        return self.loop_counters.get(loop_id, 0)

    def increment_loop_counter(self, loop_id: str) -> int:
        """Increment and return loop counter."""
        count = self.loop_counters.get(loop_id, 0) + 1
        self.loop_counters[loop_id] = count
        return count

    def reset_loop_counter(self, loop_id: str) -> None:
        """Reset loop counter."""
        self.loop_counters[loop_id] = 0


@dataclass
class RollbackPoint:
    """Point in execution that can be rolled back to."""
    step_id: str
    context_snapshot: Dict[str, Any]  # Snapshot of context state
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChainExecutionStats:
    """Statistics from chain execution."""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    total_duration_ms: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    def get_success_rate(self) -> float:
        """Get success rate (completed / total)."""
        if self.total_steps == 0:
            return 0.0
        return self.completed_steps / self.total_steps


@dataclass
class ChainValidationError:
    """Error from chain validation."""
    error_type: str
    message: str
    step_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
