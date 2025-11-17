#!/usr/bin/env python3
"""
Chain Debugger - Step-by-step execution with breakpoints

Provides debugging capabilities including:
- Step-by-step execution mode
- Breakpoints on specific steps
- Variable inspection
- Conditional breakpoints
- Replay with different inputs
- Performance profiling per step
"""

import time
import json
import copy
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum


class DebugAction(Enum):
    """Debug action types"""
    CONTINUE = "continue"      # Continue execution
    STEP_OVER = "step_over"    # Execute current step and pause
    STEP_INTO = "step_into"    # Step into nested execution
    STEP_OUT = "step_out"      # Execute until parent completes
    RUN_TO_CURSOR = "run_to_cursor"  # Run until specific step
    STOP = "stop"              # Stop execution


@dataclass
class Breakpoint:
    """Breakpoint configuration"""
    step_name: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    enabled: bool = True
    hit_count: int = 0
    hit_condition: Optional[str] = None  # e.g., "> 5" means break after 5 hits


@dataclass
class DebugFrame:
    """Debug execution frame"""
    step_name: str
    step_type: str
    input_variables: Dict[str, Any]
    output_variables: Optional[Dict[str, Any]] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    error: Optional[str] = None
    parent_frame: Optional['DebugFrame'] = None
    child_frames: List['DebugFrame'] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Calculate frame duration"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class ChainDebugger:
    """
    Interactive debugger for chain execution.

    Features:
    - Breakpoints on steps
    - Conditional breakpoints
    - Step-by-step execution
    - Variable inspection
    - Performance profiling
    - Execution replay
    """

    def __init__(self, chain_def: Dict[str, Any]):
        self.chain_def = chain_def
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.execution_frames: List[DebugFrame] = []
        self.current_frame: Optional[DebugFrame] = None
        self.current_step_index: int = 0
        self.debug_action: DebugAction = DebugAction.CONTINUE
        self.target_step: Optional[str] = None
        self.paused: bool = False
        self.execution_log: List[Dict[str, Any]] = []

        # Callback hooks
        self.on_step_start: Optional[Callable[[DebugFrame], None]] = None
        self.on_step_end: Optional[Callable[[DebugFrame], None]] = None
        self.on_breakpoint_hit: Optional[Callable[[Breakpoint, DebugFrame], DebugAction]] = None
        self.on_error: Optional[Callable[[DebugFrame, Exception], None]] = None

    def add_breakpoint(
        self,
        step_name: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        hit_condition: Optional[str] = None
    ) -> None:
        """
        Add a breakpoint to a step.

        Args:
            step_name: Name of the step to break on
            condition: Optional condition function that receives variables
            hit_condition: Optional hit count condition (e.g., "> 5", "== 3")
        """
        self.breakpoints[step_name] = Breakpoint(
            step_name=step_name,
            condition=condition,
            hit_condition=hit_condition
        )

    def remove_breakpoint(self, step_name: str) -> None:
        """Remove a breakpoint"""
        if step_name in self.breakpoints:
            del self.breakpoints[step_name]

    def enable_breakpoint(self, step_name: str, enabled: bool = True) -> None:
        """Enable or disable a breakpoint"""
        if step_name in self.breakpoints:
            self.breakpoints[step_name].enabled = enabled

    def list_breakpoints(self) -> List[Dict[str, Any]]:
        """List all breakpoints"""
        return [
            {
                "step": bp.step_name,
                "enabled": bp.enabled,
                "hit_count": bp.hit_count,
                "has_condition": bp.condition is not None,
                "hit_condition": bp.hit_condition
            }
            for bp in self.breakpoints.values()
        ]

    def inspect_variables(self, frame: Optional[DebugFrame] = None) -> Dict[str, Any]:
        """
        Inspect variables at current or specified frame.

        Returns:
            Dictionary of variables and their values
        """
        if frame is None:
            frame = self.current_frame

        if frame is None:
            return {}

        return {
            "input": frame.input_variables,
            "output": frame.output_variables,
            "step": frame.step_name,
            "type": frame.step_type,
            "duration": frame.duration
        }

    def get_call_stack(self) -> List[str]:
        """Get current call stack"""
        stack = []
        frame = self.current_frame

        while frame is not None:
            stack.insert(0, f"{frame.step_name} ({frame.step_type})")
            frame = frame.parent_frame

        return stack

    def evaluate_expression(self, expression: str) -> Any:
        """
        Evaluate an expression in the context of current frame.

        Args:
            expression: Python expression to evaluate

        Returns:
            Result of expression evaluation
        """
        if self.current_frame is None:
            raise RuntimeError("No active frame")

        # Create evaluation context
        context = {
            **self.current_frame.input_variables,
            **(self.current_frame.output_variables or {})
        }

        try:
            return eval(expression, {"__builtins__": {}}, context)
        except Exception as e:
            raise ValueError(f"Expression evaluation error: {e}")

    def _check_breakpoint(self, step_name: str, variables: Dict[str, Any]) -> bool:
        """Check if breakpoint should trigger"""
        if step_name not in self.breakpoints:
            return False

        bp = self.breakpoints[step_name]

        if not bp.enabled:
            return False

        # Increment hit count
        bp.hit_count += 1

        # Check hit condition
        if bp.hit_condition:
            if not self._evaluate_hit_condition(bp.hit_count, bp.hit_condition):
                return False

        # Check custom condition
        if bp.condition:
            try:
                if not bp.condition(variables):
                    return False
            except Exception:
                # If condition evaluation fails, don't break
                return False

        return True

    def _evaluate_hit_condition(self, hit_count: int, condition: str) -> bool:
        """Evaluate hit count condition"""
        try:
            # Simple conditions like "> 5", "== 3", "% 2 == 0"
            return eval(f"{hit_count} {condition}", {"__builtins__": {}})
        except Exception:
            return False

    def step_over(self) -> None:
        """Execute current step and pause"""
        self.debug_action = DebugAction.STEP_OVER
        self.paused = False

    def step_into(self) -> None:
        """Step into nested execution"""
        self.debug_action = DebugAction.STEP_INTO
        self.paused = False

    def step_out(self) -> None:
        """Execute until parent frame completes"""
        self.debug_action = DebugAction.STEP_OUT
        self.paused = False

    def run_to_cursor(self, step_name: str) -> None:
        """Run until specific step"""
        self.debug_action = DebugAction.RUN_TO_CURSOR
        self.target_step = step_name
        self.paused = False

    def continue_execution(self) -> None:
        """Continue normal execution"""
        self.debug_action = DebugAction.CONTINUE
        self.paused = False

    def stop_execution(self) -> None:
        """Stop execution"""
        self.debug_action = DebugAction.STOP
        self.paused = False

    def execute_step(
        self,
        step_name: str,
        step_type: str,
        input_variables: Dict[str, Any],
        executor: Callable[[str, str, Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute a single step with debugging support.

        Args:
            step_name: Name of the step
            step_type: Type of processor
            input_variables: Input variables
            executor: Function to execute the step

        Returns:
            Step output variables
        """
        # Create debug frame
        frame = DebugFrame(
            step_name=step_name,
            step_type=step_type,
            input_variables=copy.deepcopy(input_variables),
            start_time=time.time(),
            parent_frame=self.current_frame
        )

        if self.current_frame:
            self.current_frame.child_frames.append(frame)

        self.current_frame = frame
        self.execution_frames.append(frame)

        # Callback: step start
        if self.on_step_start:
            self.on_step_start(frame)

        # Check for breakpoint
        should_break = self._check_breakpoint(step_name, input_variables)

        # Handle run_to_cursor
        if self.debug_action == DebugAction.RUN_TO_CURSOR and step_name == self.target_step:
            should_break = True
            self.target_step = None

        # Pause if needed
        if should_break or self.debug_action in [DebugAction.STEP_OVER, DebugAction.STEP_INTO]:
            self.paused = True

            # Trigger breakpoint callback
            if should_break and self.on_breakpoint_hit:
                bp = self.breakpoints[step_name]
                action = self.on_breakpoint_hit(bp, frame)
                if action:
                    self.debug_action = action

            # Wait for debug action
            while self.paused:
                time.sleep(0.1)  # In real implementation, use event/condition

        # Execute the step
        try:
            output = executor(step_name, step_type, input_variables)
            frame.output_variables = copy.deepcopy(output)
            frame.end_time = time.time()

            # Log execution
            self.execution_log.append({
                "step": step_name,
                "type": step_type,
                "duration": frame.duration,
                "success": True
            })

            # Callback: step end
            if self.on_step_end:
                self.on_step_end(frame)

            return output

        except Exception as e:
            frame.error = str(e)
            frame.end_time = time.time()

            # Log error
            self.execution_log.append({
                "step": step_name,
                "type": step_type,
                "duration": frame.duration,
                "success": False,
                "error": str(e)
            })

            # Callback: error
            if self.on_error:
                self.on_error(frame, e)

            raise

        finally:
            # Pop frame
            self.current_frame = frame.parent_frame

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total_steps = len(self.execution_frames)
        successful_steps = sum(1 for f in self.execution_frames if f.error is None)
        failed_steps = total_steps - successful_steps
        total_duration = sum(f.duration for f in self.execution_frames if f.duration)

        return {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "total_duration": total_duration,
            "breakpoints_hit": sum(bp.hit_count for bp in self.breakpoints.values()),
            "execution_log": self.execution_log
        }

    def get_performance_profile(self) -> List[Dict[str, Any]]:
        """Get performance profile by step"""
        profile = []

        for frame in self.execution_frames:
            if frame.duration:
                profile.append({
                    "step": frame.step_name,
                    "type": frame.step_type,
                    "duration": frame.duration,
                    "success": frame.error is None
                })

        # Sort by duration
        profile.sort(key=lambda x: x["duration"], reverse=True)

        return profile

    def replay(
        self,
        new_input: Dict[str, Any],
        executor: Callable[[str, str, Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Replay execution with different input.

        Args:
            new_input: New input variables
            executor: Execution function

        Returns:
            Final output variables
        """
        # Reset state
        self.execution_frames = []
        self.current_frame = None
        self.current_step_index = 0
        self.execution_log = []

        # Reset breakpoint hit counts
        for bp in self.breakpoints.values():
            bp.hit_count = 0

        # Re-execute (simplified - actual implementation would use chain executor)
        # This is a placeholder for the replay mechanism
        return {}

    def export_debug_session(self) -> str:
        """Export debug session to JSON"""
        session_data = {
            "chain": self.chain_def.get("name", "unnamed"),
            "breakpoints": self.list_breakpoints(),
            "execution_frames": [
                {
                    "step": f.step_name,
                    "type": f.step_type,
                    "duration": f.duration,
                    "error": f.error,
                    "input_variables": f.input_variables,
                    "output_variables": f.output_variables
                }
                for f in self.execution_frames
            ],
            "summary": self.get_execution_summary(),
            "performance_profile": self.get_performance_profile()
        }

        return json.dumps(session_data, indent=2)


# Convenience functions
def create_debugger(chain_def: Dict[str, Any]) -> ChainDebugger:
    """Create a chain debugger"""
    return ChainDebugger(chain_def)


def add_conditional_breakpoint(
    debugger: ChainDebugger,
    step_name: str,
    condition_expr: str
) -> None:
    """
    Add a conditional breakpoint using string expression.

    Args:
        debugger: ChainDebugger instance
        step_name: Step to break on
        condition_expr: Condition expression (e.g., "score < 0.5")
    """
    def condition_fn(variables: Dict[str, Any]) -> bool:
        try:
            return eval(condition_expr, {"__builtins__": {}}, variables)
        except Exception:
            return False

    debugger.add_breakpoint(step_name, condition=condition_fn)


__all__ = [
    'ChainDebugger',
    'DebugAction',
    'DebugFrame',
    'Breakpoint',
    'create_debugger',
    'add_conditional_breakpoint'
]
