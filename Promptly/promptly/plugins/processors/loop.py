#!/usr/bin/env python3
"""
Loop Processor for Promptly Chains

Implements for-each iteration, while loops, map/reduce, and accumulator patterns.
"""

from typing import Dict, List, Any, Callable, Optional
from enum import Enum
from ..base import BaseChainStepProcessor


class LoopType(Enum):
    """Types of loop operations"""
    FOR_EACH = "for_each"       # Iterate over a list
    WHILE = "while"             # Loop while condition is true
    MAP = "map"                 # Map function over items
    REDUCE = "reduce"           # Reduce items to single value
    ACCUMULATE = "accumulate"   # Accumulate results over iterations


class LoopControl(Exception):
    """Base exception for loop control flow"""
    pass


class BreakLoop(LoopControl):
    """Exception to break out of loop"""
    def __init__(self, result: Any = None):
        self.result = result
        super().__init__()


class ContinueLoop(LoopControl):
    """Exception to continue to next iteration"""
    pass


class LoopProcessor(BaseChainStepProcessor):
    """
    Execute iterative operations.

    Configuration format:
    {
        "type": "loop",
        "loop_type": "for_each|while|map|reduce|accumulate",
        "items": "field_name",          # field containing list to iterate
        "max_iterations": 100,          # safety limit
        "condition": "...",             # for while loops
        "action": {...},                # action to execute each iteration
        "accumulator": "result",        # field to accumulate into
        "initial_value": null,          # initial accumulator value
        "reduce_function": "sum|concat|max|min|custom",  # for reduce
        "break_on": {...},              # condition to break
        "continue_on": {...}            # condition to continue
    }
    """

    def __init__(self, executor: Optional[Callable] = None):
        super().__init__(
            name="loop",
            description="Execute iterative operations"
        )
        self._executor = executor
        self._reduce_functions: Dict[str, Callable] = {
            "sum": sum,
            "concat": lambda items: "".join(str(i) for i in items),
            "max": max,
            "min": min,
            "count": len,
            "first": lambda items: items[0] if items else None,
            "last": lambda items: items[-1] if items else None,
        }

    def register_reduce_function(self, name: str, func: Callable) -> None:
        """Register a custom reduce function"""
        self._reduce_functions[name] = func

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute loop operations.

        Args:
            step_input: Input data with items to iterate
            step_config: Loop configuration

        Returns:
            Results from loop execution
        """
        loop_type = step_config.get("loop_type", "for_each")

        if loop_type == LoopType.FOR_EACH.value:
            return self._for_each(step_input, step_config)
        elif loop_type == LoopType.WHILE.value:
            return self._while_loop(step_input, step_config)
        elif loop_type == LoopType.MAP.value:
            return self._map(step_input, step_config)
        elif loop_type == LoopType.REDUCE.value:
            return self._reduce(step_input, step_config)
        elif loop_type == LoopType.ACCUMULATE.value:
            return self._accumulate(step_input, step_config)
        else:
            return step_input

    def _for_each(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute for-each loop"""
        items_field = step_config.get("items", "items")
        items = self._extract_field(step_input, items_field)

        if not isinstance(items, list):
            items = [items]

        max_iterations = step_config.get("max_iterations", 1000)
        action = step_config.get("action", {})

        results = []
        iteration = 0

        for item in items:
            if iteration >= max_iterations:
                break

            try:
                # Check continue condition
                if self._should_continue(step_config, item, step_input):
                    iteration += 1
                    continue

                # Check break condition
                if self._should_break(step_config, item, step_input):
                    break

                # Execute action for this item
                item_input = {
                    **step_input,
                    "item": item,
                    "index": iteration,
                    "is_first": iteration == 0,
                    "is_last": iteration == len(items) - 1
                }

                result = self._execute_action(action, item_input)
                results.append(result)

                iteration += 1

            except BreakLoop as e:
                if e.result is not None:
                    results.append(e.result)
                break

            except ContinueLoop:
                iteration += 1
                continue

        return {
            **step_input,
            "loop_results": results,
            "iterations": iteration,
            "total_items": len(items)
        }

    def _while_loop(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute while loop"""
        condition = step_config.get("condition", "")
        max_iterations = step_config.get("max_iterations", 1000)
        action = step_config.get("action", {})

        results = []
        iteration = 0
        current_data = step_input.copy()

        while iteration < max_iterations:
            # Evaluate condition
            if not self._eval_condition(condition, current_data):
                break

            try:
                # Check break condition
                if self._should_break(step_config, None, current_data):
                    break

                # Execute action
                loop_input = {
                    **current_data,
                    "iteration": iteration,
                }

                result = self._execute_action(action, loop_input)
                results.append(result)

                # Update current data for next iteration
                current_data = {**current_data, **result}

                iteration += 1

            except BreakLoop as e:
                if e.result is not None:
                    results.append(e.result)
                break

            except ContinueLoop:
                iteration += 1
                continue

        return {
            **step_input,
            "loop_results": results,
            "iterations": iteration,
            "final_state": current_data
        }

    def _map(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute map operation"""
        items_field = step_config.get("items", "items")
        items = self._extract_field(step_input, items_field)

        if not isinstance(items, list):
            items = [items]

        action = step_config.get("action", {})

        mapped_results = []
        for i, item in enumerate(items):
            item_input = {
                **step_input,
                "item": item,
                "index": i
            }

            result = self._execute_action(action, item_input)

            # Extract the mapped value
            map_field = step_config.get("map_field", "output")
            mapped_value = self._extract_field(result, map_field)
            mapped_results.append(mapped_value)

        return {
            **step_input,
            "mapped_results": mapped_results,
            "original_items": items
        }

    def _reduce(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reduce operation"""
        items_field = step_config.get("items", "items")
        items = self._extract_field(step_input, items_field)

        if not isinstance(items, list):
            items = [items]

        reduce_fn_name = step_config.get("reduce_function", "concat")
        initial_value = step_config.get("initial_value")

        # Get reduce function
        reduce_fn = self._reduce_functions.get(reduce_fn_name)

        if reduce_fn and reduce_fn_name in ["sum", "max", "min", "count", "first", "last"]:
            # Built-in reduce functions
            try:
                if initial_value is not None and reduce_fn_name == "sum":
                    result = initial_value + reduce_fn(items)
                else:
                    result = reduce_fn(items)
            except (TypeError, ValueError):
                result = initial_value
        else:
            # Custom reduce or concat
            if reduce_fn_name == "concat":
                result = reduce_fn(items)
            else:
                # Manual reduce
                accumulator = initial_value
                for item in items:
                    if accumulator is None:
                        accumulator = item
                    else:
                        # Custom reduction logic could go here
                        accumulator = self._combine_values(accumulator, item, reduce_fn_name)
                result = accumulator

        return {
            **step_input,
            "reduced_result": result,
            "item_count": len(items)
        }

    def _accumulate(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute accumulation pattern"""
        items_field = step_config.get("items", "items")
        items = self._extract_field(step_input, items_field)

        if not isinstance(items, list):
            items = [items]

        accumulator_field = step_config.get("accumulator", "accumulator")
        initial_value = step_config.get("initial_value", [])
        action = step_config.get("action", {})

        accumulator = initial_value if isinstance(initial_value, list) else [initial_value]

        for i, item in enumerate(items):
            item_input = {
                **step_input,
                "item": item,
                "index": i,
                accumulator_field: accumulator
            }

            result = self._execute_action(action, item_input)

            # Add result to accumulator
            result_value = self._extract_field(result, "output")
            if result_value is not None:
                if isinstance(accumulator, list):
                    accumulator.append(result_value)
                else:
                    accumulator = [accumulator, result_value]

        return {
            **step_input,
            accumulator_field: accumulator,
            "item_count": len(items)
        }

    def _should_break(self, step_config: Dict[str, Any], item: Any, data: Dict[str, Any]) -> bool:
        """Check if loop should break"""
        break_condition = step_config.get("break_on")
        if not break_condition:
            return False

        # Simple condition evaluation
        if isinstance(break_condition, str):
            return self._eval_condition(break_condition, {**data, "item": item})
        elif isinstance(break_condition, dict):
            # Structured condition
            return self._eval_structured_condition(break_condition, {**data, "item": item})

        return False

    def _should_continue(self, step_config: Dict[str, Any], item: Any, data: Dict[str, Any]) -> bool:
        """Check if should skip to next iteration"""
        continue_condition = step_config.get("continue_on")
        if not continue_condition:
            return False

        if isinstance(continue_condition, str):
            return self._eval_condition(continue_condition, {**data, "item": item})
        elif isinstance(continue_condition, dict):
            return self._eval_structured_condition(continue_condition, {**data, "item": item})

        return False

    def _execute_action(self, action: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return result"""
        if self._executor:
            # Use custom executor
            return self._executor(action, data)
        else:
            # Simple pass-through
            return {**data, "output": action.get("value", data.get("item"))}

    def _eval_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Evaluate a condition string"""
        try:
            # Replace field names with values
            eval_str = condition
            for key, value in data.items():
                if key in eval_str:
                    if isinstance(value, str):
                        eval_str = eval_str.replace(key, f"'{value}'")
                    else:
                        eval_str = eval_str.replace(key, str(value))

            return eval(eval_str, {"__builtins__": {}})
        except:
            return False

    def _eval_structured_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate a structured condition"""
        # Simple implementation - can be extended
        field = condition.get("field", "item")
        operator = condition.get("operator", "eq")
        value = condition.get("value")

        actual = self._extract_field(data, field)

        operators = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "ge": lambda a, b: a >= b,
            "le": lambda a, b: a <= b,
            "in": lambda a, b: a in b,
            "contains": lambda a, b: b in a,
        }

        return operators.get(operator, operators["eq"])(actual, value)

    def _extract_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extract field value, supporting nested paths"""
        parts = field_path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                value = value[idx] if 0 <= idx < len(value) else None
            else:
                return None

        return value

    def _combine_values(self, accumulator: Any, item: Any, operation: str) -> Any:
        """Combine accumulator and item based on operation"""
        if operation == "sum":
            return accumulator + item
        elif operation == "concat":
            return str(accumulator) + str(item)
        elif operation == "max":
            return max(accumulator, item)
        elif operation == "min":
            return min(accumulator, item)
        else:
            return item


# Export processors
__all__ = ["LoopProcessor", "LoopType", "BreakLoop", "ContinueLoop"]
