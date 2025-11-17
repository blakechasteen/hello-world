#!/usr/bin/env python3
"""
Parallel Processor for Promptly Chains

Execute multiple prompts concurrently with result aggregation and error handling.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
from ..base import BaseChainStepProcessor


class AggregationStrategy(Enum):
    """Strategies for aggregating parallel results"""
    CONCAT = "concat"           # Concatenate all results
    VOTING = "voting"           # Majority vote (for classification)
    WEIGHTED = "weighted"       # Weighted average (needs weights)
    FIRST = "first"            # Return first completed result
    ALL = "all"                # Return all results as list
    BEST = "best"              # Return best result (needs scoring function)


class ErrorStrategy(Enum):
    """Strategies for handling errors in parallel execution"""
    FAIL_FAST = "fail_fast"     # Stop on first error
    BEST_EFFORT = "best_effort" # Continue and return successful results
    RETRY = "retry"             # Retry failed tasks
    FALLBACK = "fallback"       # Use fallback value for failures


class ParallelProcessor(BaseChainStepProcessor):
    """
    Execute multiple tasks concurrently.

    Configuration format:
    {
        "type": "parallel",
        "tasks": [
            {
                "name": "task1",
                "prompt": "...",
                "inputs": {...}
            },
            ...
        ],
        "aggregation": "concat|voting|weighted|first|all|best",
        "weights": [0.5, 0.3, 0.2],  # for weighted aggregation
        "error_strategy": "fail_fast|best_effort|retry|fallback",
        "timeout": 30.0,              # timeout per task in seconds
        "max_workers": 5,             # max concurrent tasks
        "retry_attempts": 3,          # for retry strategy
        "fallback_value": "..."       # for fallback strategy
    }
    """

    def __init__(self, executor: Optional[Callable] = None):
        super().__init__(
            name="parallel",
            description="Execute multiple tasks concurrently"
        )
        self._executor = executor
        self._scoring_function: Optional[Callable] = None

    def set_executor(self, executor: Callable) -> None:
        """Set custom executor function for tasks"""
        self._executor = executor

    def set_scoring_function(self, scoring_fn: Callable[[Any], float]) -> None:
        """Set scoring function for BEST aggregation strategy"""
        self._scoring_function = scoring_fn

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tasks in parallel.

        Args:
            step_input: Input data to distribute to tasks
            step_config: Configuration with tasks and strategies

        Returns:
            Aggregated results from parallel execution
        """
        tasks = step_config.get("tasks", [])
        aggregation = step_config.get("aggregation", "all")
        error_strategy = step_config.get("error_strategy", "best_effort")
        timeout = step_config.get("timeout", 30.0)
        max_workers = step_config.get("max_workers", 5)

        if not tasks:
            return step_input

        # Execute tasks in parallel
        results = self._execute_parallel(
            tasks=tasks,
            input_data=step_input,
            timeout=timeout,
            max_workers=max_workers,
            error_strategy=error_strategy,
            retry_attempts=step_config.get("retry_attempts", 3),
            fallback_value=step_config.get("fallback_value", None)
        )

        # Aggregate results
        aggregated = self._aggregate_results(
            results=results,
            strategy=aggregation,
            weights=step_config.get("weights", []),
            step_config=step_config
        )

        return {
            **step_input,
            "parallel_results": results,
            "aggregated_result": aggregated,
            "success_count": sum(1 for r in results if r.get("success", False)),
            "total_count": len(tasks)
        }

    def _execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        timeout: float,
        max_workers: int,
        error_strategy: str,
        retry_attempts: int,
        fallback_value: Any
    ) -> List[Dict[str, Any]]:
        """Execute tasks in parallel using ThreadPoolExecutor"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for task in tasks:
                future = executor.submit(
                    self._execute_single_task,
                    task,
                    input_data,
                    timeout,
                    retry_attempts if error_strategy == "retry" else 1,
                    fallback_value if error_strategy == "fallback" else None
                )
                futures.append((task, future))

            # Collect results
            for task, future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)

                    # Fail fast on error
                    if error_strategy == "fail_fast" and not result.get("success", False):
                        # Cancel remaining futures
                        for _, f in futures:
                            f.cancel()
                        break

                except FutureTimeoutError:
                    results.append({
                        "task_name": task.get("name", "unknown"),
                        "success": False,
                        "error": "timeout",
                        "output": fallback_value if error_strategy == "fallback" else None
                    })

                    if error_strategy == "fail_fast":
                        break

                except Exception as e:
                    results.append({
                        "task_name": task.get("name", "unknown"),
                        "success": False,
                        "error": str(e),
                        "output": fallback_value if error_strategy == "fallback" else None
                    })

                    if error_strategy == "fail_fast":
                        break

        return results

    def _execute_single_task(
        self,
        task: Dict[str, Any],
        input_data: Dict[str, Any],
        timeout: float,
        max_attempts: int,
        fallback_value: Any
    ) -> Dict[str, Any]:
        """Execute a single task with retry logic"""
        task_name = task.get("name", "unknown")
        attempts = 0
        last_error = None

        while attempts < max_attempts:
            try:
                attempts += 1

                # Execute task
                if self._executor:
                    # Use custom executor
                    task_input = {**input_data, **task.get("inputs", {})}
                    output = self._executor(task.get("prompt", ""), task_input)
                else:
                    # No executor - return task data
                    output = task.get("prompt", "")

                return {
                    "task_name": task_name,
                    "success": True,
                    "output": output,
                    "attempts": attempts
                }

            except Exception as e:
                last_error = e
                if attempts < max_attempts:
                    # Exponential backoff
                    time.sleep(0.1 * (2 ** (attempts - 1)))

        # All attempts failed
        return {
            "task_name": task_name,
            "success": False,
            "error": str(last_error),
            "output": fallback_value,
            "attempts": attempts
        }

    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        strategy: str,
        weights: List[float],
        step_config: Dict[str, Any]
    ) -> Any:
        """Aggregate results using specified strategy"""
        # Filter successful results
        successful = [r for r in results if r.get("success", False)]

        if not successful:
            return None

        if strategy == AggregationStrategy.CONCAT.value:
            # Concatenate all outputs
            outputs = [r.get("output", "") for r in successful]
            return "\n".join(str(o) for o in outputs)

        elif strategy == AggregationStrategy.VOTING.value:
            # Majority vote
            outputs = [r.get("output", "") for r in successful]
            from collections import Counter
            if outputs:
                return Counter(outputs).most_common(1)[0][0]
            return None

        elif strategy == AggregationStrategy.WEIGHTED.value:
            # Weighted average/combination
            if not weights:
                weights = [1.0 / len(successful)] * len(successful)

            # Ensure weights match results
            weights = weights[:len(successful)]
            if len(weights) < len(successful):
                weights.extend([0.0] * (len(successful) - len(weights)))

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            # Weighted combination
            outputs = [r.get("output", "") for r in successful]

            # Try numeric average if possible
            try:
                numeric_outputs = [float(o) for o in outputs]
                return sum(w * o for w, o in zip(weights, numeric_outputs))
            except (ValueError, TypeError):
                # String combination with weights as importance
                # Return output with highest weight
                max_idx = weights.index(max(weights))
                return outputs[max_idx] if max_idx < len(outputs) else outputs[0]

        elif strategy == AggregationStrategy.FIRST.value:
            # Return first successful result
            return successful[0].get("output") if successful else None

        elif strategy == AggregationStrategy.ALL.value:
            # Return all results
            return [r.get("output") for r in successful]

        elif strategy == AggregationStrategy.BEST.value:
            # Return best result using scoring function
            if self._scoring_function:
                scored = [(r, self._scoring_function(r.get("output"))) for r in successful]
                best = max(scored, key=lambda x: x[1])
                return best[0].get("output")
            else:
                # No scoring function, return first
                return successful[0].get("output") if successful else None

        else:
            # Unknown strategy, return all
            return [r.get("output") for r in successful]


class AsyncParallelProcessor(BaseChainStepProcessor):
    """Async version of parallel processor for async/await workflows"""

    def __init__(self, async_executor: Optional[Callable] = None):
        super().__init__(
            name="async_parallel",
            description="Execute multiple tasks concurrently using asyncio"
        )
        self._async_executor = async_executor

    async def process_async(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of process"""
        tasks = step_config.get("tasks", [])
        timeout = step_config.get("timeout", 30.0)

        if not tasks:
            return step_input

        # Create async tasks
        async_tasks = []
        for task in tasks:
            async_task = self._execute_async_task(task, step_input, timeout)
            async_tasks.append(async_task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task_name": tasks[i].get("name", f"task_{i}"),
                    "success": False,
                    "error": str(result),
                    "output": None
                })
            else:
                processed_results.append(result)

        return {
            **step_input,
            "parallel_results": processed_results,
            "success_count": sum(1 for r in processed_results if r.get("success", False)),
            "total_count": len(tasks)
        }

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Sync wrapper for async process"""
        return asyncio.run(self.process_async(step_input, step_config))

    async def _execute_async_task(
        self,
        task: Dict[str, Any],
        input_data: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """Execute a single async task"""
        task_name = task.get("name", "unknown")

        try:
            if self._async_executor:
                task_input = {**input_data, **task.get("inputs", {})}
                output = await asyncio.wait_for(
                    self._async_executor(task.get("prompt", ""), task_input),
                    timeout=timeout
                )
            else:
                output = task.get("prompt", "")

            return {
                "task_name": task_name,
                "success": True,
                "output": output
            }

        except asyncio.TimeoutError:
            return {
                "task_name": task_name,
                "success": False,
                "error": "timeout",
                "output": None
            }
        except Exception as e:
            return {
                "task_name": task_name,
                "success": False,
                "error": str(e),
                "output": None
            }


# Export processors
__all__ = ["ParallelProcessor", "AsyncParallelProcessor", "AggregationStrategy", "ErrorStrategy"]
