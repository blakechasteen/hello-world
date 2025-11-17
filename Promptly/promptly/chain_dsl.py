#!/usr/bin/env python3
"""
Chain DSL - YAML-based workflow specification and execution engine

Provides a declarative DSL for defining complex prompt chains with advanced
control flow, parallel execution, and error handling.
"""

import yaml
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

from .plugins.processors import (
    ConditionalProcessor,
    ParallelProcessor,
    LoopProcessor,
    RetryProcessor,
    TransformProcessor
)


@dataclass
class ChainExecutionContext:
    """Context for chain execution"""
    variables: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainStep:
    """Represents a single step in the chain"""
    name: str
    processor_type: str
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Dict[str, Any]] = None


class ChainDSL:
    """
    YAML-based chain definition and execution engine.

    DSL Format:
    ```yaml
    name: my_workflow
    description: Complex workflow example
    version: "1.0"

    variables:
      max_retries: 3
      timeout: 30

    steps:
      - name: extract_data
        type: transform
        config:
          transform_type: extract
          method: json
          source_field: input
          target_field: data

      - name: parallel_process
        type: parallel
        depends_on: [extract_data]
        config:
          tasks:
            - name: task1
              prompt: "Process {data}"
            - name: task2
              prompt: "Analyze {data}"
          aggregation: concat

      - name: conditional_branch
        type: conditional
        depends_on: [parallel_process]
        config:
          conditions:
            - type: numeric
              field: score
              operator: gt
              value: 0.7
              action:
                type: set
                field: result
                value: "high_quality"
          default:
            type: set
            field: result
            value: "low_quality"

      - name: retry_final
        type: retry
        depends_on: [conditional_branch]
        config:
          max_attempts: 3
          backoff_strategy: exponential
          action:
            type: execute
            prompt: "Finalize {result}"
    ```
    """

    def __init__(self):
        self.processors = {
            "conditional": ConditionalProcessor(),
            "parallel": ParallelProcessor(),
            "loop": LoopProcessor(),
            "retry": RetryProcessor(),
            "transform": TransformProcessor(),
        }
        self._custom_processors: Dict[str, Any] = {}
        self._executor: Optional[Callable] = None

    def register_processor(self, name: str, processor: Any) -> None:
        """Register a custom processor"""
        self._custom_processors[name] = processor

    def set_executor(self, executor: Callable) -> None:
        """Set executor function for executing prompts"""
        self._executor = executor

        # Pass executor to processors that need it
        if "parallel" in self.processors:
            self.processors["parallel"].set_executor(executor)
        if "loop" in self.processors:
            self.processors["loop"]._executor = executor
        if "retry" in self.processors:
            self.processors["retry"]._executor = executor

    def load_chain(self, yaml_path: str) -> Dict[str, Any]:
        """Load chain definition from YAML file"""
        with open(yaml_path, 'r') as f:
            chain_def = yaml.safe_load(f)

        return self.parse_chain(chain_def)

    def load_chain_from_string(self, yaml_string: str) -> Dict[str, Any]:
        """Load chain definition from YAML string"""
        chain_def = yaml.safe_load(yaml_string)
        return self.parse_chain(chain_def)

    def parse_chain(self, chain_def: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate chain definition"""
        name = chain_def.get("name", "unnamed_chain")
        description = chain_def.get("description", "")
        version = chain_def.get("version", "1.0")
        variables = chain_def.get("variables", {})
        steps = chain_def.get("steps", [])

        # Parse steps
        parsed_steps = []
        for step_def in steps:
            step = ChainStep(
                name=step_def.get("name", f"step_{len(parsed_steps)}"),
                processor_type=step_def.get("type", ""),
                config=step_def.get("config", {}),
                dependencies=step_def.get("depends_on", []),
                condition=step_def.get("condition")
            )
            parsed_steps.append(step)

        return {
            "name": name,
            "description": description,
            "version": version,
            "variables": variables,
            "steps": parsed_steps
        }

    def execute_chain(
        self,
        chain_def: Dict[str, Any],
        initial_input: Dict[str, Any],
        context: Optional[ChainExecutionContext] = None
    ) -> Dict[str, Any]:
        """Execute a chain definition"""
        if context is None:
            context = ChainExecutionContext()

        # Initialize variables
        context.variables.update(chain_def.get("variables", {}))
        context.variables.update(initial_input)

        steps = chain_def.get("steps", [])
        step_results: Dict[str, Dict[str, Any]] = {}

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(steps)

        # Execute steps in topological order
        execution_order = self._topological_sort(dependency_graph)

        for step_name in execution_order:
            step = next((s for s in steps if s.name == step_name), None)
            if not step:
                continue

            # Check if step condition is met
            if step.condition and not self._evaluate_step_condition(step.condition, context):
                context.trace.append({
                    "step": step_name,
                    "status": "skipped",
                    "reason": "condition_not_met"
                })
                continue

            # Collect inputs from dependencies
            step_input = context.variables.copy()
            for dep in step.dependencies:
                if dep in step_results:
                    step_input.update(step_results[dep])

            # Execute step
            try:
                result = self._execute_step(step, step_input, context)
                step_results[step_name] = result

                # Update context variables
                context.variables.update(result)

                # Add to trace
                context.trace.append({
                    "step": step_name,
                    "type": step.processor_type,
                    "status": "completed",
                    "result": result
                })

            except Exception as e:
                context.trace.append({
                    "step": step_name,
                    "type": step.processor_type,
                    "status": "failed",
                    "error": str(e)
                })

                # Check if chain should continue on error
                if not chain_def.get("continue_on_error", False):
                    raise

        return {
            "results": step_results,
            "context": context,
            "trace": context.trace,
            "final_output": context.variables
        }

    def _execute_step(
        self,
        step: ChainStep,
        step_input: Dict[str, Any],
        context: ChainExecutionContext
    ) -> Dict[str, Any]:
        """Execute a single step"""
        processor_type = step.processor_type

        # Get processor
        processor = None
        if processor_type in self.processors:
            processor = self.processors[processor_type]
        elif processor_type in self._custom_processors:
            processor = self._custom_processors[processor_type]
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")

        # Process step
        result = processor.process(step_input, step.config)

        return result

    def _build_dependency_graph(self, steps: List[ChainStep]) -> Dict[str, List[str]]:
        """Build dependency graph from steps"""
        graph = {}

        for step in steps:
            graph[step.name] = step.dependencies

        return graph

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Topologically sort steps based on dependencies"""
        # Kahn's algorithm for topological sorting
        in_degree = {node: 0 for node in graph}

        for node in graph:
            for dep in graph[node]:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Find nodes with no incoming edges
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Remove this node and update in-degrees
            for next_node in graph:
                if node in graph[next_node]:
                    in_degree[next_node] -= 1
                    if in_degree[next_node] == 0:
                        queue.append(next_node)

        # Check for cycles
        if len(result) != len(graph):
            raise ValueError("Cycle detected in chain dependencies")

        return result

    def _evaluate_step_condition(
        self,
        condition: Dict[str, Any],
        context: ChainExecutionContext
    ) -> bool:
        """Evaluate if step should be executed"""
        condition_type = condition.get("type", "simple")

        if condition_type == "simple":
            # Simple field comparison
            field = condition.get("field", "")
            operator = condition.get("operator", "eq")
            value = condition.get("value")

            actual = context.variables.get(field)

            operators = {
                "eq": lambda a, b: a == b,
                "ne": lambda a, b: a != b,
                "gt": lambda a, b: a > b,
                "lt": lambda a, b: a < b,
                "ge": lambda a, b: a >= b,
                "le": lambda a, b: a <= b,
            }

            return operators.get(operator, operators["eq"])(actual, value)

        return True

    def validate_chain(self, chain_def: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chain definition"""
        errors = []
        warnings = []

        # Check required fields
        if "steps" not in chain_def:
            errors.append("Missing 'steps' field")

        steps = chain_def.get("steps", [])

        # Check for duplicate step names
        step_names = [s.name for s in steps]
        if len(step_names) != len(set(step_names)):
            errors.append("Duplicate step names found")

        # Check dependencies
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")

        # Check for cycles
        try:
            graph = self._build_dependency_graph(steps)
            self._topological_sort(graph)
        except ValueError as e:
            errors.append(str(e))

        # Check processor types
        for step in steps:
            if step.processor_type not in self.processors and step.processor_type not in self._custom_processors:
                warnings.append(f"Unknown processor type '{step.processor_type}' in step '{step.name}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def export_chain(self, chain_def: Dict[str, Any], output_path: str) -> None:
        """Export chain definition to YAML file"""
        # Convert ChainStep objects back to dicts
        export_def = {
            "name": chain_def.get("name"),
            "description": chain_def.get("description"),
            "version": chain_def.get("version"),
            "variables": chain_def.get("variables", {}),
            "steps": [
                {
                    "name": step.name,
                    "type": step.processor_type,
                    "config": step.config,
                    "depends_on": step.dependencies,
                    "condition": step.condition
                }
                for step in chain_def.get("steps", [])
            ]
        }

        with open(output_path, 'w') as f:
            yaml.dump(export_def, f, default_flow_style=False, sort_keys=False)


# Convenience functions
def load_chain(yaml_path: str) -> Dict[str, Any]:
    """Load and parse a chain from YAML file"""
    dsl = ChainDSL()
    return dsl.load_chain(yaml_path)


def execute_chain(chain_def: Dict[str, Any], initial_input: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a chain definition"""
    dsl = ChainDSL()
    return dsl.execute_chain(chain_def, initial_input)


__all__ = [
    "ChainDSL",
    "ChainExecutionContext",
    "ChainStep",
    "load_chain",
    "execute_chain"
]
