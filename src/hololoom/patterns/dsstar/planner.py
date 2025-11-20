#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-STAR Planner
===============

Creates execution plans for data analysis tasks.
Leverages HoloLoom's query routing and policy engine.

Created: 2025-01-20
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from .analyzer import DataProfile


class ActionType(Enum):
    """Types of actions in execution plan."""
    LOAD_DATA = "load_data"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    TRANSFORM = "transform"
    VISUALIZE = "visualize"
    ANALYZE = "analyze"
    MODEL = "model"


@dataclass
class PlanStep:
    """Single step in execution plan."""
    step_id: int
    action: ActionType
    description: str
    code_template: str           # Python code template
    inputs: List[str]            # Input variables
    outputs: List[str]           # Output variables
    dependencies: List[int] = field(default_factory=list)  # Step IDs this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query."""
    query: str
    steps: List[PlanStep]
    data_profile: Optional[DataProfile] = None
    estimated_complexity: str = "medium"  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: int) -> Optional[PlanStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_dependencies_satisfied(self, step_id: int, completed: List[int]) -> bool:
        """Check if all dependencies for a step are satisfied."""
        step = self.get_step(step_id)
        if not step:
            return False
        return all(dep in completed for dep in step.dependencies)


class Planner:
    """
    Creates execution plans for data analysis queries.

    Integrates with HoloLoom's query routing and policy engine.
    """

    def __init__(self, use_hololoom_routing: bool = True):
        self.use_hololoom_routing = use_hololoom_routing
        self.step_counter = 0

    def create_plan(self, query: str, data_profile: Optional[DataProfile] = None) -> ExecutionPlan:
        """
        Create execution plan for query.

        Args:
            query: User's data analysis question
            data_profile: Optional data profile for context

        Returns:
            ExecutionPlan with steps
        """
        # Parse query to understand intent
        intent = self._parse_query_intent(query, data_profile)

        # Generate steps based on intent
        steps = self._generate_steps(intent, data_profile)

        # Estimate complexity
        complexity = self._estimate_complexity(steps)

        return ExecutionPlan(
            query=query,
            steps=steps,
            data_profile=data_profile,
            estimated_complexity=complexity,
            metadata={'intent': intent}
        )

    def _parse_query_intent(self, query: str, data_profile: Optional[DataProfile]) -> Dict[str, Any]:
        """Parse query to understand user intent."""
        query_lower = query.lower()

        intent = {
            'type': 'explore',  # explore, aggregate, visualize, model, etc.
            'entities': [],     # Entities mentioned (column names)
            'operations': [],   # Operations (filter, group, etc.)
            'goal': query
        }

        # Detect operation types
        if any(word in query_lower for word in ['average', 'mean', 'sum', 'count', 'total']):
            intent['type'] = 'aggregate'
            intent['operations'].append('aggregate')

        if any(word in query_lower for word in ['plot', 'chart', 'visualize', 'show', 'graph']):
            intent['type'] = 'visualize'
            intent['operations'].append('visualize')

        if any(word in query_lower for word in ['filter', 'where', 'only', 'exclude']):
            intent['operations'].append('filter')

        if any(word in query_lower for word in ['group by', 'by', 'per']):
            intent['operations'].append('group')

        if any(word in query_lower for word in ['predict', 'model', 'forecast', 'classify']):
            intent['type'] = 'model'
            intent['operations'].append('model')

        # Extract entities (column names if profile available)
        if data_profile:
            for anchor in data_profile.anchors:
                if anchor.name.lower() in query_lower:
                    intent['entities'].append(anchor.name)

        return intent

    def _generate_steps(self, intent: Dict[str, Any], data_profile: Optional[DataProfile]) -> List[PlanStep]:
        """Generate execution steps based on intent."""
        steps = []
        self.step_counter = 0

        # Step 1: Always load data first
        if data_profile:
            steps.append(self._create_step(
                action=ActionType.LOAD_DATA,
                description=f"Load data from {data_profile.file_path}",
                code_template=f"df = pd.read_{data_profile.format}('{data_profile.file_path}')",
                inputs=[],
                outputs=['df']
            ))

        # Generate steps based on operations
        current_df = 'df'
        for operation in intent['operations']:
            if operation == 'filter':
                steps.append(self._create_step(
                    action=ActionType.FILTER,
                    description="Filter data based on conditions",
                    code_template=f"{current_df}_filtered = {current_df}[condition]",
                    inputs=[current_df],
                    outputs=[f'{current_df}_filtered'],
                    dependencies=[steps[-1].step_id] if steps else []
                ))
                current_df = f'{current_df}_filtered'

            elif operation == 'aggregate':
                entities_str = ', '.join(intent['entities']) if intent['entities'] else 'column'
                steps.append(self._create_step(
                    action=ActionType.AGGREGATE,
                    description=f"Aggregate {entities_str}",
                    code_template=f"result = {current_df}['{entities_str}'].agg(['mean', 'sum', 'count'])",
                    inputs=[current_df],
                    outputs=['result'],
                    dependencies=[steps[-1].step_id] if steps else []
                ))

            elif operation == 'visualize':
                entities_str = ', '.join(intent['entities']) if intent['entities'] else 'data'
                steps.append(self._create_step(
                    action=ActionType.VISUALIZE,
                    description=f"Visualize {entities_str}",
                    code_template=f"{current_df}.plot()",
                    inputs=[current_df],
                    outputs=['plot'],
                    dependencies=[steps[-1].step_id] if steps else []
                ))

            elif operation == 'model':
                steps.append(self._create_step(
                    action=ActionType.MODEL,
                    description="Build predictive model",
                    code_template="model = build_model(X, y)",
                    inputs=[current_df],
                    outputs=['model', 'predictions'],
                    dependencies=[steps[-1].step_id] if steps else []
                ))

        # If no specific operations, add exploration step
        if len(steps) == 1:  # Only load step
            steps.append(self._create_step(
                action=ActionType.ANALYZE,
                description="Explore data statistics",
                code_template=f"summary = {current_df}.describe()",
                inputs=[current_df],
                outputs=['summary'],
                dependencies=[steps[-1].step_id]
            ))

        return steps

    def _create_step(
        self,
        action: ActionType,
        description: str,
        code_template: str,
        inputs: List[str],
        outputs: List[str],
        dependencies: Optional[List[int]] = None
    ) -> PlanStep:
        """Create a plan step."""
        step_id = self.step_counter
        self.step_counter += 1

        return PlanStep(
            step_id=step_id,
            action=action,
            description=description,
            code_template=code_template,
            inputs=inputs,
            outputs=outputs,
            dependencies=dependencies or [],
            metadata={}
        )

    def _estimate_complexity(self, steps: List[PlanStep]) -> str:
        """Estimate plan complexity."""
        num_steps = len(steps)

        if num_steps <= 2:
            return "low"
        elif num_steps <= 5:
            return "medium"
        else:
            return "high"

    def visualize_plan(self, plan: ExecutionPlan) -> str:
        """
        Generate ASCII visualization of execution plan.

        Args:
            plan: ExecutionPlan to visualize

        Returns:
            ASCII diagram
        """
        lines = [
            f"Execution Plan: {plan.query}",
            f"Complexity: {plan.estimated_complexity}",
            f"Steps: {len(plan.steps)}",
            "",
            "Flow:"
        ]

        for step in plan.steps:
            indent = "  " * len(step.dependencies)
            deps_str = f" (depends on: {step.dependencies})" if step.dependencies else ""
            lines.append(f"{indent}{step.step_id}. {step.action.value}: {step.description}{deps_str}")

        return "\n".join(lines)


# Convenience function
def create_plan(query: str, data_profile: Optional[DataProfile] = None) -> ExecutionPlan:
    """
    Quick plan creation.

    Args:
        query: User query
        data_profile: Optional data profile

    Returns:
        ExecutionPlan
    """
    planner = Planner()
    return planner.create_plan(query, data_profile)
