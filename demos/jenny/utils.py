"""
Jenny Demo Utilities
====================
Decorator and helper functions for standardized demo output.

Philosophy:
> "Consistent output format makes demos feel polished and professional."

The utilities provide:
- @demo_phase decorator for standardized headers and timing
- Output formatting helpers
- Error recovery wrappers
- Progress indicators

Usage:
    @demo_phase("M1: Renderer Registry", emoji="📋")
    async def demo_m1_renderer_registry() -> dict:
        # Your demo code here
        return {"renderers": 3, "success": True}

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot Extensibility)
"""

import asyncio
import time
import traceback
from functools import wraps
from typing import Callable, Any, Dict, Optional, List
from datetime import datetime


# ============================================================================
# Demo Phase Decorator
# ============================================================================

def demo_phase(
    name: str,
    emoji: str = "🔬",
    show_duration: bool = True,
    catch_errors: bool = True,
    separator_char: str = "=",
    separator_width: int = 60
) -> Callable:
    """
    Decorator for demo phases with standardized output.

    Wraps demo functions with:
    - Formatted header with emoji
    - Execution timing
    - Error handling with graceful degradation
    - Consistent output format

    Args:
        name: Display name for the phase
        emoji: Emoji prefix for header
        show_duration: Whether to print duration after execution
        catch_errors: Whether to catch and format exceptions
        separator_char: Character for header separator line
        separator_width: Width of separator line

    Returns:
        Decorated function that prints formatted output

    Example:
        @demo_phase("M1: Renderer Registry", emoji="📋")
        async def demo_m1():
            return {"renderers": 3}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # Print header
            print(f"\n{separator_char * separator_width}")
            print(f"  {emoji} {name}")
            print(f"{separator_char * separator_width}\n")

            start_time = time.perf_counter()
            result = {
                "phase": name,
                "emoji": emoji,
                "status": "pending",
                "started_at": datetime.now().isoformat(),
            }

            try:
                output = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                result["status"] = "success"
                result["duration_ms"] = duration_ms
                result["output"] = output

                if show_duration:
                    print(f"\n  ✓ Completed in {duration_ms:.1f}ms")

                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000

                result["status"] = "error"
                result["duration_ms"] = duration_ms
                result["error"] = str(e)
                result["traceback"] = traceback.format_exc()

                if catch_errors:
                    print(f"\n  ✗ Error after {duration_ms:.1f}ms: {e}")
                    return result
                else:
                    raise

        return wrapper
    return decorator


# ============================================================================
# Output Formatting Helpers
# ============================================================================

def print_header(title: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_subheader(title: str, char: str = "-", width: int = 50) -> None:
    """Print a formatted subsection header."""
    print(f"\n  {char * width}")
    print(f"    {title}")
    print(f"  {char * width}\n")


def print_metric(name: str, value: Any, unit: str = "", indent: int = 2) -> None:
    """Print a single metric in aligned format."""
    prefix = " " * indent
    if unit:
        print(f"{prefix}{name:.<30} {value} {unit}")
    else:
        print(f"{prefix}{name:.<30} {value}")


def print_metrics(metrics: Dict[str, Any], indent: int = 2) -> None:
    """Print multiple metrics in aligned format."""
    prefix = " " * indent
    max_key_len = max(len(str(k)) for k in metrics.keys()) if metrics else 10

    for key, value in metrics.items():
        print(f"{prefix}{key:.<{max_key_len + 10}} {value}")


def print_success(message: str, indent: int = 2) -> None:
    """Print a success message."""
    prefix = " " * indent
    print(f"{prefix}✓ {message}")


def print_warning(message: str, indent: int = 2) -> None:
    """Print a warning message."""
    prefix = " " * indent
    print(f"{prefix}⚠ {message}")


def print_error(message: str, indent: int = 2) -> None:
    """Print an error message."""
    prefix = " " * indent
    print(f"{prefix}✗ {message}")


def print_info(message: str, indent: int = 2) -> None:
    """Print an info message."""
    prefix = " " * indent
    print(f"{prefix}• {message}")


def print_list(items: List[Any], indent: int = 4, bullet: str = "•") -> None:
    """Print a bulleted list."""
    prefix = " " * indent
    for item in items:
        print(f"{prefix}{bullet} {item}")


# ============================================================================
# Progress Indicators
# ============================================================================

class ProgressTracker:
    """
    Simple progress tracker for demo steps.

    Usage:
        progress = ProgressTracker(total=5, description="Processing")
        progress.start()
        for i in range(5):
            # do work
            progress.update(i + 1)
        progress.complete()
    """

    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start progress tracking."""
        self.start_time = time.perf_counter()
        self.current = 0
        print(f"  {self.description}: Starting ({self.total} steps)")

    def update(self, current: int, message: str = "") -> None:
        """Update progress."""
        self.current = current
        pct = (current / self.total) * 100
        msg_part = f" - {message}" if message else ""
        print(f"    [{current}/{self.total}] {pct:.0f}%{msg_part}")

    def complete(self) -> float:
        """Complete progress and return duration in ms."""
        duration_ms = 0.0
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
        print(f"  {self.description}: Complete ({duration_ms:.1f}ms)")
        return duration_ms


# ============================================================================
# Summary Formatting
# ============================================================================

def format_demo_summary(results: List[Dict[str, Any]]) -> str:
    """
    Format a summary of demo results.

    Args:
        results: List of result dicts from demo phases

    Returns:
        Formatted summary string
    """
    lines = [
        "",
        "=" * 60,
        "  DEMO SUMMARY",
        "=" * 60,
        "",
    ]

    success_count = sum(1 for r in results if r.get("status") == "success")
    total_count = len(results)
    total_duration = sum(r.get("duration_ms", 0) for r in results)

    # Status overview
    lines.append(f"  Total Phases: {total_count}")
    lines.append(f"  Successful: {success_count}")
    lines.append(f"  Failed: {total_count - success_count}")
    lines.append(f"  Total Duration: {total_duration:.1f}ms")
    lines.append("")

    # Per-phase status
    lines.append("  Phase Results:")
    lines.append("  " + "-" * 50)

    for result in results:
        phase = result.get("phase", "Unknown")
        status = result.get("status", "unknown")
        duration = result.get("duration_ms", 0)

        if status == "success":
            status_icon = "✓"
        elif status == "error":
            status_icon = "✗"
        else:
            status_icon = "?"

        lines.append(f"    {status_icon} {phase:.<35} {duration:.1f}ms")

        if status == "error":
            error = result.get("error", "Unknown error")
            lines.append(f"        Error: {error[:50]}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def print_demo_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted demo summary."""
    print(format_demo_summary(results))


# ============================================================================
# Error Recovery
# ============================================================================

def with_fallback(fallback_value: Any = None, log_error: bool = True):
    """
    Decorator for graceful error recovery.

    Args:
        fallback_value: Value to return on error
        log_error: Whether to print error message

    Returns:
        Decorated function

    Example:
        @with_fallback(fallback_value=[], log_error=True)
        async def fetch_data():
            return await risky_operation()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    print_warning(f"{func.__name__} failed: {e}")
                return fallback_value

        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay_ms: float = 100,
    backoff_multiplier: float = 2.0
):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        delay_ms: Initial delay between retries
        backoff_multiplier: Multiplier for delay after each retry

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            delay = delay_ms

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        print_warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.0f}ms..."
                        )
                        await asyncio.sleep(delay / 1000)
                        delay *= backoff_multiplier
                    else:
                        raise last_error

        return wrapper
    return decorator


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Decorators
    'demo_phase',
    'with_fallback',
    'retry_on_error',

    # Output helpers
    'print_header',
    'print_subheader',
    'print_metric',
    'print_metrics',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'print_list',

    # Progress
    'ProgressTracker',

    # Summary
    'format_demo_summary',
    'print_demo_summary',
]
