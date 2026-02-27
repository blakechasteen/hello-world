"""
Easy Skill Launcher for HoloLoom
================================

One-line skill launching with Zero-G integration.

**Created**: December 30, 2025
**Purpose**: Make it EASY to launch apps and skills onto HoloLoom

Usage:
    from HoloLoom.skills import launch_skill, launch

    # Simple launch
    result = await launch_skill("code_reviewer", code="def foo(): pass")

    # With options
    result = await launch("pytest_runner",
        test_path="tests/",
        verbose=True,
        use_memory=True  # Enable memory-informed decisions
    )
"""

import asyncio
import time
import logging
import random
from typing import Dict, Any, Optional, Union, List, Callable, AsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum
import sys

from HoloLoom.skills.base import (
    BaseSkill,
    SkillInput,
    SkillOutput,
    SkillStatus,
    SkillMetadata,
    SkillCategory,
    get_registry,
    register_skill
)
from HoloLoom.skills.executor import SkillExecutor, ExecutionResult

logger = logging.getLogger(__name__)


# ============================================================================
# Error Handling & Debugging
# ============================================================================

class SkillErrorCode(Enum):
    """Standardized error codes for skill failures."""
    SKILL_NOT_FOUND = "SKILL_NOT_FOUND"
    SKILL_SETUP_FAILED = "SKILL_SETUP_FAILED"
    SKILL_EXECUTION_FAILED = "SKILL_EXECUTION_FAILED"
    SKILL_TIMEOUT = "SKILL_TIMEOUT"
    SKILL_VALIDATION_FAILED = "SKILL_VALIDATION_FAILED"
    MEMORY_ENHANCEMENT_FAILED = "MEMORY_ENHANCEMENT_FAILED"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    PARAMETER_ERROR = "PARAMETER_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SkillError(Exception):
    """
    Rich exception for skill failures.

    Provides detailed error information for debugging including:
    - Error code for programmatic handling
    - Skill name and operation that failed
    - Suggestions for common fixes
    - Stack trace capture
    """

    def __init__(
        self,
        message: str,
        error_code: SkillErrorCode = SkillErrorCode.INTERNAL_ERROR,
        skill_name: str = "",
        operation: str = "",
        suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.skill_name = skill_name
        self.operation = operation
        self.suggestions = suggestions or []
        self.cause = cause
        self.context = context or {}
        self.traceback_str: Optional[str] = None

        # Capture traceback if there's a cause
        if cause:
            import traceback
            self.traceback_str = ''.join(traceback.format_exception(type(cause), cause, cause.__traceback__))

    def __str__(self):
        parts = [f"[{self.error_code.value}] {self.message}"]

        if self.skill_name:
            parts.append(f"  Skill: {self.skill_name}")
        if self.operation:
            parts.append(f"  Operation: {self.operation}")
        if self.suggestions:
            parts.append("  Suggestions:")
            for s in self.suggestions:
                parts.append(f"    - {s}")
        if self.cause:
            parts.append(f"  Caused by: {type(self.cause).__name__}: {self.cause}")

        return '\n'.join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "skill_name": self.skill_name,
            "operation": self.operation,
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
            "cause_type": type(self.cause).__name__ if self.cause else None,
            "context": self.context,
            "traceback": self.traceback_str
        }


def _find_similar_skills(skill_name: str, max_suggestions: int = 3) -> List[str]:
    """Find skills with similar names for suggestions."""
    registry = get_registry()
    all_skills = registry.list_all()

    if not all_skills:
        return []

    # Simple similarity: check for common substrings
    skill_lower = skill_name.lower()
    similar = []

    for name in all_skills:
        name_lower = name.lower()
        # Check if one contains the other, or they share significant prefix
        if (skill_lower in name_lower or
            name_lower in skill_lower or
            skill_lower[:3] == name_lower[:3]):
            similar.append(name)

    return similar[:max_suggestions]


def _format_parameter_error(params: Dict[str, Any], expected: Optional[List[str]] = None) -> str:
    """Format parameter information for error messages."""
    parts = [f"Provided parameters: {list(params.keys())}"]
    if expected:
        parts.append(f"Expected parameters: {expected}")
    return '\n'.join(parts)


# ============================================================================
# Retry Configuration
# ============================================================================

class RetryStrategy(Enum):
    """Retry strategy for failed skill executions."""
    NONE = "none"                    # No retry
    IMMEDIATE = "immediate"           # Retry immediately
    EXPONENTIAL = "exponential"       # Exponential backoff with jitter


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay_ms: float = 100.0
    max_delay_ms: float = 5000.0
    jitter: float = 0.1  # 10% jitter

    @classmethod
    def none(cls) -> "RetryConfig":
        """No retries."""
        return cls(max_retries=0, strategy=RetryStrategy.NONE)

    @classmethod
    def fast(cls) -> "RetryConfig":
        """Fast retries (2 attempts, short delays)."""
        return cls(max_retries=2, base_delay_ms=50.0, max_delay_ms=500.0)

    @classmethod
    def standard(cls) -> "RetryConfig":
        """Standard retry config (3 attempts)."""
        return cls(max_retries=3, base_delay_ms=100.0, max_delay_ms=5000.0)

    @classmethod
    def persistent(cls) -> "RetryConfig":
        """Persistent retries (5 attempts, longer delays)."""
        return cls(max_retries=5, base_delay_ms=200.0, max_delay_ms=10000.0)

    def get_delay_ms(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.strategy == RetryStrategy.NONE:
            return 0
        elif self.strategy == RetryStrategy.IMMEDIATE:
            return 0
        else:  # EXPONENTIAL
            delay = self.base_delay_ms * (2 ** attempt)
            delay = min(delay, self.max_delay_ms)
            # Add jitter
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
            return max(0, delay)


# ============================================================================
# Easy Launch API
# ============================================================================

@dataclass
class LaunchResult:
    """
    Result from skill launch.

    Provides easy access to common result patterns.
    """
    success: bool
    result: Any
    message: str
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    skill_name: str = ""

    # Memory-informed metadata
    memory_informed: bool = False
    memory_count: int = 0
    memory_confidence: float = 0.0

    # Retry metadata
    retry_count: int = 0
    total_attempts: int = 1
    retry_delays_ms: List[float] = field(default_factory=list)

    # Raw output for advanced usage
    raw_output: Optional[SkillOutput] = None

    # Error details for debugging (only populated on failure)
    error_code: Optional[SkillErrorCode] = None
    suggestions: List[str] = field(default_factory=list)
    error_context: Optional[Dict[str, Any]] = None

    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        retry_info = f" (retries: {self.retry_count})" if self.retry_count > 0 else ""
        return f"<LaunchResult [{status}] {self.skill_name}{retry_info}: {self.message[:50]}...>"

    @property
    def total_time_with_retries_ms(self) -> float:
        """Total time including retry delays."""
        return self.execution_time_ms + sum(self.retry_delays_ms)

    def get_debug_info(self) -> str:
        """Get formatted debug information for failed launches."""
        if self.success:
            return "Launch succeeded - no debug info needed."

        lines = [
            f"=== Launch Failed: {self.skill_name} ===",
            f"Error Code: {self.error_code.value if self.error_code else 'UNKNOWN'}",
            f"Message: {self.message}",
        ]

        if self.suggestions:
            lines.append("\nSuggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        if self.retry_count > 0:
            lines.append(f"\nRetry Info: {self.retry_count} retries attempted")
            lines.append(f"  Total attempts: {self.total_attempts}")
            if self.retry_delays_ms:
                lines.append(f"  Retry delays: {[f'{d:.1f}ms' for d in self.retry_delays_ms]}")

        if self.error_context:
            lines.append("\nContext:")
            for key, value in self.error_context.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    @classmethod
    def from_skill_output(cls, output: SkillOutput, skill_name: str = "") -> "LaunchResult":
        """Create LaunchResult from SkillOutput."""
        return cls(
            success=output.status == SkillStatus.SUCCESS,
            result=output.result,
            message=output.message,
            confidence=getattr(output, 'confidence', 0.0) if hasattr(output, 'confidence') else 0.0,
            execution_time_ms=output.execution_time_ms,
            skill_name=skill_name or output.skill_name or "",
            raw_output=output
        )

    @classmethod
    def error(
        cls,
        message: str,
        skill_name: str = "",
        error_code: Optional[SkillErrorCode] = None,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> "LaunchResult":
        """Create error result with rich debugging information."""
        return cls(
            success=False,
            result=None,
            message=message,
            skill_name=skill_name,
            error_code=error_code or SkillErrorCode.INTERNAL_ERROR,
            suggestions=suggestions or [],
            error_context=context
        )

    @classmethod
    def from_skill_error(cls, error: "SkillError") -> "LaunchResult":
        """Create LaunchResult from SkillError exception."""
        return cls(
            success=False,
            result=None,
            message=error.message,
            skill_name=error.skill_name,
            error_code=error.error_code,
            suggestions=error.suggestions,
            error_context=error.context
        )


# Global executor instance (lazy initialization)
_global_executor: Optional[SkillExecutor] = None


def _get_executor() -> SkillExecutor:
    """Get or create global executor."""
    global _global_executor
    if _global_executor is None:
        _global_executor = SkillExecutor()
    return _global_executor


async def launch_skill(
    skill_name: str,
    operation: str = "execute",
    use_cache: bool = True,
    use_memory: bool = False,
    timeout: Optional[float] = None,
    retry: Optional[RetryConfig] = None,
    **parameters
) -> LaunchResult:
    """
    Launch a skill with one line.

    This is the primary easy API for executing skills.

    Args:
        skill_name: Name of skill to launch (e.g., "code_reviewer", "pytest_runner")
        operation: Operation to perform (default: "execute")
        use_cache: Enable result caching for repeated calls
        use_memory: Enable memory-informed decision making (Zero-G)
        timeout: Execution timeout in seconds
        retry: Retry configuration for failed executions (default: no retries)
        **parameters: Skill-specific parameters

    Returns:
        LaunchResult with success status, result, and metadata

    Examples:
        # Simple launch
        result = await launch_skill("code_reviewer", code="def foo(): pass")

        # With timeout
        result = await launch_skill("pytest_runner", test_path="tests/", timeout=60)

        # Memory-informed
        result = await launch_skill("code_reviewer",
            code="def bar(): return 42",
            use_memory=True
        )

        # With retry on failure
        result = await launch_skill("flaky_service",
            retry=RetryConfig.standard(),  # 3 retries with exponential backoff
            data="payload"
        )
    """
    # Validate skill exists before attempting execution
    registry = get_registry()
    if registry.get(skill_name) is None:
        # Find similar skills for helpful suggestions
        similar = _find_similar_skills(skill_name)
        all_skills = registry.list_all()
        available_skills = all_skills[:10]  # Show first 10

        suggestions = []
        if similar:
            suggestions.append(f"Did you mean: {', '.join(similar)}?")
        suggestions.append(f"Available skills: {', '.join(available_skills)}")
        if len(all_skills) > 10:
            suggestions.append(f"Use list_skills() to see all {len(all_skills)} registered skills")
        suggestions.append("Ensure the skill module is imported (skills auto-register on import)")

        return LaunchResult.error(
            message=f"Skill '{skill_name}' not found in registry",
            skill_name=skill_name,
            error_code=SkillErrorCode.SKILL_NOT_FOUND,
            suggestions=suggestions,
            context={
                "requested_skill": skill_name,
                "similar_skills": similar,
                "total_registered": len(all_skills)
            }
        )

    # Default to no retries
    retry_config = retry or RetryConfig.none()
    max_attempts = retry_config.max_retries + 1
    retry_delays: List[float] = []
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            executor = _get_executor()

            # Add memory context if requested
            params = parameters.copy()
            if use_memory:
                params['_use_memory'] = True
                # Try to get Zero-G memory enhancement
                params = await _enhance_with_memory(skill_name, params)

            # Execute skill
            exec_result = await executor.execute_single(
                skill_name=skill_name,
                operation=operation,
                parameters=params,
                use_cache=use_cache,
                timeout=timeout
            )

            # Convert to LaunchResult
            result = LaunchResult.from_skill_output(exec_result.output, skill_name)

            # Add retry metadata
            result.retry_count = attempt
            result.total_attempts = attempt + 1
            result.retry_delays_ms = retry_delays.copy()

            # Add cache info
            if exec_result.cache_hit:
                result.message = f"[CACHED] {result.message}"

            # If successful or no retries configured, return
            if result.success or retry_config.strategy == RetryStrategy.NONE:
                return result

            # Failed but retryable - continue to retry logic
            last_error = Exception(result.message)

        except Exception as e:
            last_error = e
            logger.warning(f"Skill '{skill_name}' attempt {attempt + 1}/{max_attempts} failed: {e}")

        # Check if we should retry
        if attempt < max_attempts - 1:
            delay_ms = retry_config.get_delay_ms(attempt)
            if delay_ms > 0:
                retry_delays.append(delay_ms)
                logger.info(f"Retrying '{skill_name}' in {delay_ms:.1f}ms...")
                await asyncio.sleep(delay_ms / 1000.0)  # Convert to seconds

    # All retries exhausted - build comprehensive error message
    error_msg = f"Launch failed after {max_attempts} attempts: {last_error}"
    logger.error(error_msg)

    # Build helpful suggestions based on the error
    suggestions = [
        f"Skill '{skill_name}' failed {max_attempts} consecutive times",
        "Check skill health with: check_skill_health('{skill_name}')",
        "Increase retry count with: retry=RetryConfig(max_retries=5)",
    ]

    # Add specific suggestions based on error type
    if last_error:
        error_str = str(last_error).lower()
        if "timeout" in error_str:
            suggestions.append("Increase timeout with: timeout=60.0")
        elif "connection" in error_str or "network" in error_str:
            suggestions.append("Check network connectivity or external service availability")
        elif "memory" in error_str or "resource" in error_str:
            suggestions.append("Check system resources (memory, CPU)")
        elif "permission" in error_str or "access" in error_str:
            suggestions.append("Check file/resource permissions")

    result = LaunchResult.error(
        message=error_msg,
        skill_name=skill_name,
        error_code=SkillErrorCode.RETRY_EXHAUSTED,
        suggestions=suggestions,
        context={
            "attempts": max_attempts,
            "retry_delays_ms": retry_delays,
            "last_error": str(last_error),
            "last_error_type": type(last_error).__name__ if last_error else None,
            "parameters_provided": list(parameters.keys())
        }
    )
    result.retry_count = max_attempts - 1
    result.total_attempts = max_attempts
    result.retry_delays_ms = retry_delays
    return result


# Alias for even shorter syntax
launch = launch_skill


async def launch_parallel(
    *skill_specs: Dict[str, Any],
    use_cache: bool = True,
    fail_fast: bool = False
) -> List[LaunchResult]:
    """
    Launch multiple skills in parallel.

    Args:
        *skill_specs: Skill specifications, each dict with:
            - skill_name: str (required)
            - operation: str (default: "execute")
            - parameters: dict (skill params)
            - timeout: float (optional)
        use_cache: Enable caching
        fail_fast: Stop on first failure

    Returns:
        List of LaunchResults in same order as inputs

    Example:
        results = await launch_parallel(
            {"skill_name": "code_reviewer", "parameters": {"code": "..."}},
            {"skill_name": "security_checker", "parameters": {"code": "..."}},
            {"skill_name": "style_linter", "parameters": {"code": "..."}}
        )
    """
    executor = _get_executor()

    # Convert to executor format
    executions = []
    for spec in skill_specs:
        executions.append({
            "skill_name": spec["skill_name"],
            "operation": spec.get("operation", "execute"),
            "parameters": spec.get("parameters", {}),
            "timeout": spec.get("timeout")
        })

    # Execute in parallel
    batch_result = await executor.execute_parallel(
        executions=executions,
        use_cache=use_cache,
        fail_fast=fail_fast
    )

    # Convert results
    return [
        LaunchResult.from_skill_output(r.output, r.skill_name)
        for r in sorted(batch_result.results, key=lambda r: r.execution_order)
    ]


async def launch_chain(
    *skill_specs: Dict[str, Any],
    use_cache: bool = True,
    stop_on_failure: bool = True
) -> List[LaunchResult]:
    """
    Launch skills sequentially (chain).

    Each skill can use the previous skill's output via {{prev}} in parameters.

    Args:
        *skill_specs: Skill specifications (same format as launch_parallel)
        use_cache: Enable caching
        stop_on_failure: Stop chain on first failure

    Returns:
        List of LaunchResults in execution order

    Example:
        results = await launch_chain(
            {"skill_name": "code_parser", "parameters": {"code": "..."}},
            {"skill_name": "code_analyzer", "parameters": {"ast": "{{prev.result}}"}},
            {"skill_name": "report_generator", "parameters": {"analysis": "{{prev.result}}"}}
        )
    """
    executor = _get_executor()

    # Convert and execute
    executions = []
    for spec in skill_specs:
        executions.append({
            "skill_name": spec["skill_name"],
            "operation": spec.get("operation", "execute"),
            "parameters": spec.get("parameters", {}),
            "timeout": spec.get("timeout")
        })

    batch_result = await executor.execute_sequential(
        executions=executions,
        use_cache=use_cache,
        stop_on_failure=stop_on_failure
    )

    return [
        LaunchResult.from_skill_output(r.output, r.skill_name)
        for r in sorted(batch_result.results, key=lambda r: r.execution_order)
    ]


# ============================================================================
# Memory Enhancement (Zero-G Integration)
# ============================================================================

async def _enhance_with_memory(
    skill_name: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhance parameters with memory-informed context.

    Integrates with Zero-G's MemoryInformedDecisionEngine if available.
    """
    enhanced = parameters.copy()

    try:
        # Try to import Zero-G components
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".claude" / "skills"))

        from zero_g_integration import MemoryInformedDecisionEngine, ZeroGConfig

        # Create memory engine
        config = ZeroGConfig(
            memory_query_limit=5,
            memory_confidence_threshold=0.6
        )
        engine = MemoryInformedDecisionEngine(config=config)

        # Query memory
        query = parameters.get('query', parameters.get('code', str(parameters)[:100]))
        memories = await engine.query_memory(query, parameters)

        if memories:
            enhanced['_relevant_memories'] = memories
            enhanced['_memory_informed'] = True
            enhanced['_memory_count'] = len(memories)
            enhanced['_memory_confidence'] = sum(m.get('confidence', 0) for m in memories) / len(memories)
            logger.info(f"Enhanced {skill_name} with {len(memories)} memories")

    except ImportError:
        logger.debug("Zero-G not available, skipping memory enhancement")
    except Exception as e:
        logger.warning(f"Memory enhancement failed: {e}")

    return enhanced


# ============================================================================
# Skill Discovery
# ============================================================================

def list_skills(category: Optional[str] = None) -> List[str]:
    """
    List available skills.

    Args:
        category: Filter by category (domain, testing, code, etc.)

    Returns:
        List of skill names
    """
    registry = get_registry()

    if category:
        try:
            cat = SkillCategory(category)
            return registry.list_by_category(cat)
        except ValueError:
            return []

    return registry.list_all()


def get_skill_info(skill_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a skill.

    Args:
        skill_name: Name of skill

    Returns:
        Dict with skill metadata or None if not found
    """
    registry = get_registry()
    skill = registry.get(skill_name)

    if not skill:
        return None

    metadata = skill.get_metadata()
    return {
        'name': metadata.name,
        'version': metadata.version,
        'category': metadata.category.value,
        'description': metadata.description_short,
        'tags': metadata.tags,
        'requires_network': metadata.requires_network,
        'requires_filesystem_read': metadata.requires_filesystem_read,
        'requires_filesystem_write': metadata.requires_filesystem_write,
        'requires_code_execution': metadata.requires_code_execution,
        'expected_latency': metadata.expected_latency_ms
    }


def get_launch_stats() -> Dict[str, Any]:
    """
    Get launcher statistics.

    Returns:
        Dict with execution metrics
    """
    executor = _get_executor()
    return executor.get_statistics()


# ============================================================================
# Synchronous Wrappers (for non-async contexts)
# ============================================================================

def launch_sync(
    skill_name: str,
    operation: str = "execute",
    **parameters
) -> LaunchResult:
    """
    Synchronous skill launch (for non-async code).

    Warning: This blocks the current thread.

    Args:
        skill_name: Name of skill
        operation: Operation to perform
        **parameters: Skill parameters

    Returns:
        LaunchResult
    """
    return asyncio.run(launch_skill(skill_name, operation, **parameters))


# ============================================================================
# Context Managers
# ============================================================================

@dataclass
class SkillSessionContext:
    """Context for skill session with accumulated results."""
    results: List[LaunchResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    _executor: Optional[SkillExecutor] = None

    @property
    def total_time_ms(self) -> float:
        """Total session time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    @property
    def success_count(self) -> int:
        """Number of successful launches."""
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        """Number of failed launches."""
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        """Success rate as a ratio (0.0-1.0)."""
        if not self.results:
            return 0.0
        return self.success_count / len(self.results)

    def summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            "total_launches": len(self.results),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": sum(r.execution_time_ms for r in self.results) / len(self.results) if self.results else 0.0,
        }


@asynccontextmanager
async def skill_session() -> AsyncContextManager[SkillSessionContext]:
    """
    Context manager for skill sessions.

    Tracks all skill launches within the session and provides
    accumulated statistics.

    Example:
        async with skill_session() as session:
            result1 = await launch_skill("skill_a", param="value")
            session.results.append(result1)

            result2 = await launch_skill("skill_b", param="value")
            session.results.append(result2)

        print(f"Success rate: {session.success_rate:.1%}")
        print(f"Total time: {session.total_time_ms:.1f}ms")
    """
    ctx = SkillSessionContext()
    ctx._executor = _get_executor()
    try:
        yield ctx
    finally:
        # Session cleanup - could flush metrics here
        logger.info(
            f"Skill session completed: {ctx.success_count}/{len(ctx.results)} "
            f"succeeded in {ctx.total_time_ms:.1f}ms"
        )


@asynccontextmanager
async def batch_launch(
    skill_specs: List[Dict[str, Any]],
    parallel: bool = True,
    fail_fast: bool = False
) -> AsyncContextManager[List[LaunchResult]]:
    """
    Context manager for batch skill launching.

    Launches multiple skills and yields results when complete.

    Args:
        skill_specs: List of skill specifications, each dict with:
            - skill_name: str (required)
            - operation: str (default: "execute")
            - parameters: dict (skill params)
            - timeout: float (optional)
        parallel: Execute in parallel (True) or sequentially (False)
        fail_fast: Stop on first failure (only for sequential)

    Example:
        specs = [
            {"skill_name": "skill_a", "parameters": {"x": 1}},
            {"skill_name": "skill_b", "parameters": {"y": 2}},
        ]

        async with batch_launch(specs) as results:
            for result in results:
                print(f"{result.skill_name}: {result.success}")
    """
    start_time = time.time()

    try:
        if parallel:
            results = await launch_parallel(*skill_specs, fail_fast=fail_fast)
        else:
            results = await launch_chain(*skill_specs, stop_on_failure=fail_fast)

        yield results

    finally:
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Batch launch completed in {elapsed:.1f}ms")


# ============================================================================
# Health Checks
# ============================================================================

@dataclass
class SkillHealthStatus:
    """Health status for a skill."""
    skill_name: str
    healthy: bool
    message: str
    latency_ms: float = 0.0
    last_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


async def check_skill_health(
    skill_name: str,
    timeout: float = 5.0
) -> SkillHealthStatus:
    """
    Check if a skill is healthy and responsive.

    Performs a lightweight health check by verifying:
    1. Skill exists in registry
    2. Skill can be instantiated
    3. Skill setup completes (if not already done)

    Args:
        skill_name: Name of skill to check
        timeout: Maximum time for health check in seconds

    Returns:
        SkillHealthStatus with health information

    Example:
        health = await check_skill_health("my_skill")
        if health.healthy:
            print(f"Skill healthy, latency: {health.latency_ms:.1f}ms")
        else:
            print(f"Skill unhealthy: {health.message}")
    """
    start_time = time.time()

    # Check if skill exists
    registry = get_registry()
    skill = registry.get(skill_name)

    if skill is None:
        return SkillHealthStatus(
            skill_name=skill_name,
            healthy=False,
            message=f"Skill '{skill_name}' not found in registry",
            latency_ms=(time.time() - start_time) * 1000
        )

    try:
        # Get metadata
        metadata = skill.get_metadata()

        # Check if setup is needed
        if hasattr(skill, '_setup_done') and not skill._setup_done:
            await asyncio.wait_for(skill.setup(), timeout=timeout)

        latency_ms = (time.time() - start_time) * 1000

        return SkillHealthStatus(
            skill_name=skill_name,
            healthy=True,
            message="Skill is healthy",
            latency_ms=latency_ms,
            metadata={
                "version": metadata.version,
                "category": metadata.category.value,
                "requires_network": metadata.requires_network,
            }
        )

    except asyncio.TimeoutError:
        return SkillHealthStatus(
            skill_name=skill_name,
            healthy=False,
            message=f"Skill setup timed out after {timeout}s",
            latency_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        return SkillHealthStatus(
            skill_name=skill_name,
            healthy=False,
            message=f"Health check failed: {e}",
            latency_ms=(time.time() - start_time) * 1000
        )


@dataclass
class SystemHealthStatus:
    """System-wide health status."""
    healthy: bool
    total_skills: int
    healthy_skills: int
    unhealthy_skills: List[str]
    check_time_ms: float
    skill_statuses: Dict[str, SkillHealthStatus] = field(default_factory=dict)

    @property
    def health_rate(self) -> float:
        """Percentage of healthy skills."""
        if self.total_skills == 0:
            return 1.0
        return self.healthy_skills / self.total_skills


async def get_system_health(
    timeout_per_skill: float = 2.0,
    parallel: bool = True
) -> SystemHealthStatus:
    """
    Get health status for all registered skills.

    Args:
        timeout_per_skill: Timeout for each skill health check
        parallel: Check skills in parallel (faster but more resource-intensive)

    Returns:
        SystemHealthStatus with overall system health

    Example:
        health = await get_system_health()
        print(f"System health: {health.health_rate:.1%}")
        print(f"Unhealthy skills: {health.unhealthy_skills}")
    """
    start_time = time.time()
    skills = list_skills()

    if not skills:
        return SystemHealthStatus(
            healthy=True,
            total_skills=0,
            healthy_skills=0,
            unhealthy_skills=[],
            check_time_ms=0.0
        )

    if parallel:
        # Check all skills in parallel
        checks = [check_skill_health(name, timeout_per_skill) for name in skills]
        statuses = await asyncio.gather(*checks, return_exceptions=True)
    else:
        # Check skills sequentially
        statuses = []
        for name in skills:
            status = await check_skill_health(name, timeout_per_skill)
            statuses.append(status)

    # Process results
    skill_statuses = {}
    healthy_count = 0
    unhealthy = []

    for i, status in enumerate(statuses):
        skill_name = skills[i]
        if isinstance(status, Exception):
            skill_statuses[skill_name] = SkillHealthStatus(
                skill_name=skill_name,
                healthy=False,
                message=str(status)
            )
            unhealthy.append(skill_name)
        else:
            skill_statuses[skill_name] = status
            if status.healthy:
                healthy_count += 1
            else:
                unhealthy.append(skill_name)

    check_time = (time.time() - start_time) * 1000

    return SystemHealthStatus(
        healthy=len(unhealthy) == 0,
        total_skills=len(skills),
        healthy_skills=healthy_count,
        unhealthy_skills=unhealthy,
        check_time_ms=check_time,
        skill_statuses=skill_statuses
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Primary API
    "launch_skill",
    "launch",
    "launch_parallel",
    "launch_chain",
    "launch_sync",

    # Results
    "LaunchResult",

    # Retry Configuration
    "RetryConfig",
    "RetryStrategy",

    # Discovery
    "list_skills",
    "get_skill_info",
    "get_launch_stats",

    # Context Managers
    "skill_session",
    "batch_launch",

    # Health Checks
    "check_skill_health",
    "get_system_health",
    "SkillHealthStatus",
    "SystemHealthStatus",

    # Session Context
    "SkillSessionContext",
]
