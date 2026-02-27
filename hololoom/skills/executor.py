"""Skill Execution Engine - Parallel execution with intelligent caching"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from hololoom.skills.base import (
    BaseSkill, SkillInput, SkillOutput, SkillStatus,
    get_registry
)
from hololoom.skills.cache import SkillCache

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from skill executor."""
    skill_name: str
    operation: str
    output: SkillOutput
    cache_hit: bool
    execution_order: int  # Order in batch


@dataclass
class BatchResult:
    """Result from batch execution."""
    results: List[ExecutionResult]
    total_time_ms: float
    cache_hit_rate: float
    success_count: int
    failure_count: int

    def successful_results(self) -> List[ExecutionResult]:
        """Get only successful results."""
        return [r for r in self.results if r.output.status == SkillStatus.SUCCESS]

    def failed_results(self) -> List[ExecutionResult]:
        """Get only failed results."""
        return [r for r in self.results if r.output.status != SkillStatus.SUCCESS]


class SkillExecutor:
    """
    Skill execution engine with caching and parallel execution.

    Features:
    - Automatic caching of skill results
    - Parallel execution with asyncio.gather()
    - Dependency resolution for sequential execution
    - Timeout enforcement
    - Error handling with partial results
    - Performance metrics tracking
    """

    def __init__(
        self,
        cache: Optional[SkillCache] = None,
        default_timeout: float = 30.0,
        max_parallel: int = 10
    ):
        """
        Initialize skill executor.

        Args:
            cache: Optional SkillCache instance (creates default if None)
            default_timeout: Default timeout in seconds
            max_parallel: Maximum number of parallel executions
        """
        self.cache = cache or SkillCache()
        self.default_timeout = default_timeout
        self.max_parallel = max_parallel
        self.registry = get_registry()

        # Metrics
        self._total_executions = 0
        self._total_time_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

    async def execute_single(
        self,
        skill_name: str,
        operation: str,
        parameters: Dict[str, Any],
        use_cache: bool = True,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """
        Execute a single skill operation.

        Args:
            skill_name: Name of skill to execute
            operation: Operation name
            parameters: Operation parameters
            use_cache: Whether to use caching
            timeout: Timeout in seconds (uses default if None)

        Returns:
            ExecutionResult with output and metadata
        """
        start_time = time.time()
        cache_hit = False

        # Check cache if enabled
        if use_cache:
            cached_result = self.cache.get(skill_name, operation, parameters)
            if cached_result is not None:
                cache_hit = True
                self._cache_hits += 1
                logger.debug(f"Cache hit: {skill_name}.{operation}")

                return ExecutionResult(
                    skill_name=skill_name,
                    operation=operation,
                    output=cached_result,
                    cache_hit=True,
                    execution_order=0
                )

        self._cache_misses += 1

        # Get skill from registry
        skill = self.registry.get(skill_name)
        if skill is None:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                skill_name=skill_name,
                operation=operation,
                output=SkillOutput(
                    status=SkillStatus.ERROR,
                    result=None,
                    message=f"Skill '{skill_name}' not found in registry",
                    execution_time_ms=execution_time,
                    errors=[f"Unknown skill: {skill_name}"]
                ),
                cache_hit=False,
                execution_order=0
            )

        # Create skill input
        skill_input = SkillInput(
            operation=operation,
            parameters=parameters,
            timeout_seconds=timeout or self.default_timeout
        )

        # Execute skill with timeout
        try:
            timeout_seconds = timeout or self.default_timeout
            output = await asyncio.wait_for(
                skill.execute_with_lifecycle(skill_input),
                timeout=timeout_seconds
            )

            # Cache successful results
            if use_cache and output.status == SkillStatus.SUCCESS:
                self.cache.put(skill_name, operation, parameters, output)

            execution_time = (time.time() - start_time) * 1000
            self._total_executions += 1
            self._total_time_ms += execution_time

            return ExecutionResult(
                skill_name=skill_name,
                operation=operation,
                output=output,
                cache_hit=False,
                execution_order=0
            )

        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                skill_name=skill_name,
                operation=operation,
                output=SkillOutput(
                    status=SkillStatus.TIMEOUT,
                    result=None,
                    message=f"Skill execution timed out after {timeout_seconds}s",
                    execution_time_ms=execution_time,
                    errors=["Timeout exceeded"]
                ),
                cache_hit=False,
                execution_order=0
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Skill execution error: {e}")

            return ExecutionResult(
                skill_name=skill_name,
                operation=operation,
                output=SkillOutput(
                    status=SkillStatus.ERROR,
                    result=None,
                    message=f"Skill execution failed: {e}",
                    execution_time_ms=execution_time,
                    errors=[str(e)]
                ),
                cache_hit=False,
                execution_order=0
            )

    async def execute_parallel(
        self,
        executions: List[Dict[str, Any]],
        use_cache: bool = True,
        fail_fast: bool = False
    ) -> BatchResult:
        """
        Execute multiple skills in parallel.

        Args:
            executions: List of execution specs:
                {
                    "skill_name": str,
                    "operation": str,
                    "parameters": dict,
                    "timeout": Optional[float]
                }
            use_cache: Whether to use caching
            fail_fast: Stop on first failure

        Returns:
            BatchResult with all results and metrics
        """
        start_time = time.time()

        # Limit parallelism
        batch_size = min(len(executions), self.max_parallel)

        # Create tasks
        tasks = []
        for i, exec_spec in enumerate(executions):
            task = self.execute_single(
                skill_name=exec_spec["skill_name"],
                operation=exec_spec["operation"],
                parameters=exec_spec["parameters"],
                use_cache=use_cache,
                timeout=exec_spec.get("timeout")
            )
            tasks.append(task)

        # Execute in batches if needed
        all_results = []

        for batch_start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[batch_start:batch_start + batch_size]

            if fail_fast:
                # Return on first failure
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=False)
            else:
                # Collect all results even if some fail
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Convert exception to failed ExecutionResult
                    execution_order = batch_start + i
                    exec_spec = executions[execution_order]

                    result = ExecutionResult(
                        skill_name=exec_spec["skill_name"],
                        operation=exec_spec["operation"],
                        output=SkillOutput(
                            status=SkillStatus.ERROR,
                            result=None,
                            message=f"Execution raised exception: {result}",
                            execution_time_ms=0.0,
                            errors=[str(result)]
                        ),
                        cache_hit=False,
                        execution_order=execution_order
                    )

                # Set execution order
                result.execution_order = batch_start + i
                all_results.append(result)

            # Stop if fail_fast and any failures
            if fail_fast and any(r.output.status != SkillStatus.SUCCESS for r in batch_results):
                break

        # Compute metrics
        total_time = (time.time() - start_time) * 1000
        cache_hits = sum(1 for r in all_results if r.cache_hit)
        cache_hit_rate = cache_hits / len(all_results) if all_results else 0.0
        success_count = sum(1 for r in all_results if r.output.status == SkillStatus.SUCCESS)
        failure_count = len(all_results) - success_count

        return BatchResult(
            results=all_results,
            total_time_ms=total_time,
            cache_hit_rate=cache_hit_rate,
            success_count=success_count,
            failure_count=failure_count
        )

    async def execute_sequential(
        self,
        executions: List[Dict[str, Any]],
        use_cache: bool = True,
        stop_on_failure: bool = False
    ) -> BatchResult:
        """
        Execute skills sequentially (useful for dependencies).

        Args:
            executions: List of execution specs (same format as execute_parallel)
            use_cache: Whether to use caching
            stop_on_failure: Stop on first failure

        Returns:
            BatchResult with all results and metrics
        """
        start_time = time.time()
        all_results = []

        for i, exec_spec in enumerate(executions):
            result = await self.execute_single(
                skill_name=exec_spec["skill_name"],
                operation=exec_spec["operation"],
                parameters=exec_spec["parameters"],
                use_cache=use_cache,
                timeout=exec_spec.get("timeout")
            )

            result.execution_order = i
            all_results.append(result)

            # Stop if requested and execution failed
            if stop_on_failure and result.output.status != SkillStatus.SUCCESS:
                break

        # Compute metrics
        total_time = (time.time() - start_time) * 1000
        cache_hits = sum(1 for r in all_results if r.cache_hit)
        cache_hit_rate = cache_hits / len(all_results) if all_results else 0.0
        success_count = sum(1 for r in all_results if r.output.status == SkillStatus.SUCCESS)
        failure_count = len(all_results) - success_count

        return BatchResult(
            results=all_results,
            total_time_ms=total_time,
            cache_hit_rate=cache_hit_rate,
            success_count=success_count,
            failure_count=failure_count
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Dict with execution metrics
        """
        avg_time = self._total_time_ms / self._total_executions if self._total_executions > 0 else 0.0
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0

        return {
            "total_executions": self._total_executions,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": avg_time,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_stats": self.cache.get_stats()
        }

    def clear_cache(self):
        """Clear the executor's cache."""
        self.cache.clear()
        logger.info("Executor cache cleared")


__all__ = ["SkillExecutor", "ExecutionResult", "BatchResult"]
