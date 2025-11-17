#!/usr/bin/env python3
"""
Conditional Processor for Promptly Chains

Implements if/else/elif logic based on outputs, pattern matching, and custom predicates.
"""

import re
from typing import Dict, List, Any, Callable, Optional
from ..base import BaseChainStepProcessor


class ConditionalProcessor(BaseChainStepProcessor):
    """
    Execute different branches based on conditions.

    Configuration format:
    {
        "type": "conditional",
        "conditions": [
            {
                "type": "regex|keyword|numeric|custom",
                "field": "output",  # field to check
                "pattern": "...",   # for regex
                "keywords": [...],  # for keyword
                "operator": "gt|lt|eq|ne|ge|le",  # for numeric
                "value": ...,       # comparison value
                "predicate": callable,  # for custom
                "action": {...}     # action to execute if true
            }
        ],
        "default": {...},  # default action if no conditions match
        "short_circuit": true  # stop at first match (default: true)
    }
    """

    def __init__(self):
        super().__init__(
            name="conditional",
            description="Execute different branches based on conditions"
        )
        self._predicates: Dict[str, Callable] = {}

    def register_predicate(self, name: str, predicate: Callable[[Any], bool]) -> None:
        """Register a custom predicate function"""
        self._predicates[name] = predicate

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process conditional logic.

        Args:
            step_input: Input data containing fields to check
            step_config: Configuration with conditions and actions

        Returns:
            Output from executed branch
        """
        conditions = step_config.get("conditions", [])
        default_action = step_config.get("default", {})
        short_circuit = step_config.get("short_circuit", True)

        matched_actions = []

        for condition in conditions:
            if self._evaluate_condition(condition, step_input):
                action = condition.get("action", {})
                matched_actions.append(action)

                if short_circuit:
                    break

        # If no conditions matched, use default
        if not matched_actions:
            if default_action:
                matched_actions = [default_action]
            else:
                # Pass through input unchanged
                return step_input

        # Execute matched actions
        result = step_input.copy()
        for action in matched_actions:
            result = self._execute_action(action, result)

        return result

    def _evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        condition_type = condition.get("type", "keyword")
        field = condition.get("field", "output")

        # Extract field value (supports nested fields like "data.result.score")
        value = self._extract_field(data, field)

        if condition_type == "regex":
            pattern = condition.get("pattern", "")
            if not isinstance(value, str):
                value = str(value)
            return bool(re.search(pattern, value))

        elif condition_type == "keyword":
            keywords = condition.get("keywords", [])
            case_sensitive = condition.get("case_sensitive", False)

            if not isinstance(value, str):
                value = str(value)

            if not case_sensitive:
                value = value.lower()
                keywords = [k.lower() for k in keywords]

            return any(keyword in value for keyword in keywords)

        elif condition_type == "numeric":
            operator = condition.get("operator", "eq")
            threshold = condition.get("value", 0)

            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                return False

            operators = {
                "gt": lambda a, b: a > b,
                "lt": lambda a, b: a < b,
                "eq": lambda a, b: a == b,
                "ne": lambda a, b: a != b,
                "ge": lambda a, b: a >= b,
                "le": lambda a, b: a <= b,
            }

            return operators.get(operator, operators["eq"])(numeric_value, threshold)

        elif condition_type == "custom":
            predicate_name = condition.get("predicate")
            if predicate_name and predicate_name in self._predicates:
                return self._predicates[predicate_name](value)
            return False

        return False

    def _extract_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extract field value, supporting nested paths like 'data.result.score'"""
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

    def _execute_action(self, action: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return result"""
        action_type = action.get("type", "set")

        if action_type == "set":
            # Set a field to a value
            field = action.get("field", "result")
            value = action.get("value", None)
            result = data.copy()
            self._set_field(result, field, value)
            return result

        elif action_type == "transform":
            # Apply a transformation
            transform = action.get("transform", "")
            field = action.get("field", "output")
            value = self._extract_field(data, field)

            if transform == "uppercase":
                value = str(value).upper()
            elif transform == "lowercase":
                value = str(value).lower()
            elif transform == "strip":
                value = str(value).strip()

            result = data.copy()
            self._set_field(result, field, value)
            return result

        elif action_type == "merge":
            # Merge additional data
            merge_data = action.get("data", {})
            result = data.copy()
            result.update(merge_data)
            return result

        else:
            # Unknown action type, pass through
            return data

    def _set_field(self, data: Dict[str, Any], field_path: str, value: Any) -> None:
        """Set a field value, supporting nested paths"""
        parts = field_path.split(".")
        current = data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value


class IfElseProcessor(BaseChainStepProcessor):
    """Simplified if/else processor for common use cases"""

    def __init__(self):
        super().__init__(
            name="if_else",
            description="Simple if/else branching"
        )

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple if/else processing.

        Configuration:
        {
            "condition": "field > 0.5",  # simple condition string
            "if_true": {...},            # action if true
            "if_false": {...}            # action if false
        }
        """
        condition_str = step_config.get("condition", "")
        if_true = step_config.get("if_true", {})
        if_false = step_config.get("if_false", {})

        # Evaluate simple condition
        result = self._eval_simple_condition(condition_str, step_input)

        action = if_true if result else if_false

        return action if action else step_input

    def _eval_simple_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Evaluate a simple condition string like 'score > 0.5'"""
        try:
            # Replace field names with values
            for key, value in data.items():
                if key in condition:
                    if isinstance(value, str):
                        condition = condition.replace(key, f"'{value}'")
                    else:
                        condition = condition.replace(key, str(value))

            # Safely evaluate
            return eval(condition, {"__builtins__": {}})
        except:
            return False


# Export processors
__all__ = ["ConditionalProcessor", "IfElseProcessor"]
