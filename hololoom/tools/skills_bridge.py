#!/usr/bin/env python3
"""
Skills Bridge Module
====================

Integrates the HoloLoom skills system with the weaving orchestrator.

This module bridges the gap between:
- Skills system (HoloLoom/skills/) - Modular external tool integrations
- Tool executor (HoloLoom/tools/executor.py) - Orchestrator tool dispatch

Key Features:
- Automatic skill discovery and registration
- Dynamic tool list generation from available skills
- Skill execution with caching and parallel support
- Metrics integration for performance monitoring

Author: Claude Code
Date: 2025-11-25
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hololoom.protocols.types import Context, Query
from hololoom.skills.base import SkillStatus, get_registry
from hololoom.skills.executor import SkillExecutor
from hololoom.skills.metrics import SkillMetricsTracker

if TYPE_CHECKING:
    from hololoom.skills.base import BaseSkill


class SkillsBridge:
    """
    Bridge between skills system and orchestrator.

    Automatically discovers registered skills and makes them available
    as orchestrator tools. Handles skill execution, caching, and metrics.

    Example:
        bridge = SkillsBridge()

        # Get available skill tools
        tools = bridge.get_skill_tools()  # ['graphviz', 'pytest', 'file_search', ...]

        # Execute skill as tool
        result = await bridge.execute_skill(
            skill_name="graphviz",
            operation="render_dot",
            parameters={...},
            query=query,
            context=context
        )

    Attributes:
        executor (SkillExecutor): Skill execution engine with caching
        tracker (SkillMetricsTracker): Performance metrics tracker
        logger (logging.Logger): Logger instance
    """

    def __init__(self, enable_caching: bool = True, enable_metrics: bool = True):
        """
        Initialize skills bridge.

        Args:
            enable_caching: Enable skill result caching
            enable_metrics: Enable performance metrics tracking
        """
        self.logger = logging.getLogger(__name__)

        # Initialize skill executor with caching
        self.executor = SkillExecutor()

        # Initialize metrics tracker
        self.tracker = SkillMetricsTracker() if enable_metrics else None

        # Cache of registered skills
        self._skills_cache = None

        self.logger.info(
            f"Initialized SkillsBridge (caching={enable_caching}, "
            f"metrics={enable_metrics})"
        )

    def get_registered_skills(self) -> dict[str, BaseSkill]:
        """
        Get all registered skills.

        Caches result for performance.

        Returns:
            Dict mapping skill names to skill instances
        """
        if self._skills_cache is None:
            registry = get_registry()
            self._skills_cache = registry._skills  # Access registry's skills dict
            self.logger.debug(f"Discovered {len(self._skills_cache)} registered skills")

        return self._skills_cache

    def get_skill_tools(self) -> list[str]:
        """
        Get list of available skill tool names.

        These can be used as tool names in the orchestrator's
        convergence engine.

        Returns:
            List of skill names (e.g., ['graphviz', 'pytest', 'file_search'])
        """
        skills = self.get_registered_skills()
        return list(skills.keys())

    def get_skill_operations(self, skill_name: str) -> list[str]:
        """
        Get available operations for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            List of operation names

        Raises:
            ValueError: If skill not found
        """
        skills = self.get_registered_skills()

        if skill_name not in skills:
            raise ValueError(
                f"Skill '{skill_name}' not found. "
                f"Available: {', '.join(skills.keys())}"
            )

        skill = skills[skill_name]
        metadata = skill.get_metadata()

        # Extract operations from docstrings or metadata
        # For now, return empty list (could be enhanced)
        return []

    async def execute_skill(
        self,
        skill_name: str,
        operation: str,
        parameters: dict,
        query: Query | None = None,
        context: Context | None = None,
        use_cache: bool = True,
        timeout: float | None = None
    ) -> dict:
        """
        Execute a skill as an orchestrator tool.

        Wraps skill execution in orchestrator-compatible format.
        Integrates with caching, metrics, and error handling.

        Args:
            skill_name: Name of the skill to execute
            operation: Skill operation to perform
            parameters: Operation parameters
            query: Optional query context
            context: Optional retrieved context
            use_cache: Enable caching for this execution
            timeout: Optional timeout in seconds

        Returns:
            Dict containing:
                - tool: Skill name
                - operation: Operation performed
                - result: Skill output result
                - status: Execution status (success/error/timeout)
                - confidence: Confidence score (0.0-1.0)
                - execution_time_ms: Execution time
                - cached: Whether result was from cache
                - errors: List of errors (if any)

        Example:
            result = await bridge.execute_skill(
                skill_name="graphviz",
                operation="render_dot",
                parameters={
                    "dot_source": "digraph { A -> B; }",
                    "output": "/tmp/graph.png"
                }
            )

            print(result['status'])  # 'success'
            print(result['result'])  # {'output': '/tmp/graph.png', ...}
        """
        self.logger.info(f"Executing skill: {skill_name}.{operation}")

        # Execute skill via executor
        execution_result = await self.executor.execute_single(
            skill_name=skill_name,
            operation=operation,
            parameters=parameters,
            use_cache=use_cache,
            timeout=timeout
        )

        # Record metrics
        if self.tracker:
            self.tracker.record_execution(
                skill_name=skill_name,
                operation=operation,
                status=execution_result.output.status,
                execution_time_ms=execution_result.output.execution_time_ms,
                cache_hit=execution_result.cache_hit
            )

        # Convert to orchestrator tool result format
        return {
            "tool": skill_name,
            "operation": operation,
            "result": execution_result.output.result,
            "status": execution_result.output.status.value,
            "confidence": self._calculate_confidence(execution_result.output),
            "execution_time_ms": execution_result.output.execution_time_ms,
            "cached": execution_result.cache_hit,
            "errors": execution_result.output.errors or [],
            "message": execution_result.output.message,
            "details": execution_result.output.details or {}
        }

    def _calculate_confidence(self, output) -> float:
        """
        Calculate confidence score from skill output.

        Args:
            output: Skill output

        Returns:
            Confidence score 0.0-1.0
        """
        # Base confidence on status
        if output.status == SkillStatus.SUCCESS:
            return 0.9
        elif output.status == SkillStatus.ERROR:
            return 0.1
        elif output.status == SkillStatus.TIMEOUT:
            return 0.3
        else:
            return 0.5

    def get_metrics_summary(self) -> dict:
        """
        Get performance metrics summary.

        Returns:
            Dict containing metrics for all skills
        """
        if not self.tracker:
            return {"metrics_disabled": True}

        return self.tracker.get_summary()

    def get_skill_info(self, skill_name: str) -> dict:
        """
        Get information about a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Dict containing skill metadata

        Raises:
            ValueError: If skill not found
        """
        skills = self.get_registered_skills()

        if skill_name not in skills:
            raise ValueError(
                f"Skill '{skill_name}' not found. "
                f"Available: {', '.join(skills.keys())}"
            )

        skill = skills[skill_name]
        metadata = skill.get_metadata()

        return {
            "name": metadata.name,
            "version": metadata.version,
            "author": metadata.author,
            "category": metadata.category.value,
            "tags": metadata.tags,
            "description_short": metadata.description_short,
            "description_long": metadata.description_long,
            "requires_code_execution": metadata.requires_code_execution,
            "requires_filesystem_read": metadata.requires_filesystem_read,
            "requires_filesystem_write": metadata.requires_filesystem_write,
            "external_dependencies": metadata.external_dependencies,
            "expected_latency_ms": metadata.expected_latency_ms,
            "token_usage": metadata.token_usage
        }

    def list_skills_by_category(self) -> dict[str, list[str]]:
        """
        List skills grouped by category.

        Returns:
            Dict mapping categories to skill names
        """
        skills = self.get_registered_skills()
        by_category = {}

        for skill_name, skill in skills.items():
            metadata = skill.get_metadata()
            category = metadata.category.value

            if category not in by_category:
                by_category[category] = []

            by_category[category].append(skill_name)

        return by_category


# Singleton instance for easy access
_bridge_instance = None


def get_skills_bridge() -> SkillsBridge:
    """
    Get the singleton skills bridge instance.

    Returns:
        SkillsBridge instance
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = SkillsBridge()
    return _bridge_instance
