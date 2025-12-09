"""
Jenny Demo Phase Registry
=========================
Decorator-based registration system for demo phases with dependency resolution.

Philosophy:
> "Adding a new demo should be as simple as writing the function."

The registry enables:
- Decorator-based phase registration
- Automatic dependency ordering
- Tag-based filtering (run only "core" demos)
- Extensibility (third-party demo plugins)

Usage:
    # Register a demo phase
    @register_demo("m1", "Renderer Registry", order=1, tags=["core"])
    async def demo_m1_renderer_registry() -> dict:
        ...

    # Run all demos in order
    results = await run_all_demos()

    # Run demos by tag
    results = await run_demos_by_tag("accessibility")

    # Run specific phase
    result = await run_demo("m1")

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot Extensibility)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional, Set
from functools import wraps
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Demo Phase Definition
# ============================================================================

@dataclass
class DemoPhase:
    """
    Represents a single demo phase in the Jenny system.

    Attributes:
        id: Unique identifier (e.g., "m1", "m2", "thompson")
        name: Human-readable name (e.g., "Renderer Registry")
        order: Execution order (lower = earlier)
        func: The async demo function
        requires: List of phase IDs that must run first
        tags: Categorization tags for filtering
        description: Optional longer description
    """
    id: str
    name: str
    order: int
    func: Callable[..., Any]
    requires: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        """Validate phase configuration."""
        if not self.id:
            raise ValueError("Demo phase must have an id")
        if not self.name:
            raise ValueError("Demo phase must have a name")
        if self.order < 0:
            raise ValueError("Demo phase order must be non-negative")


# ============================================================================
# Demo Phase Registry (Singleton)
# ============================================================================

class DemoPhaseRegistry:
    """
    Singleton registry for Jenny demo phases.

    Thread-safe registry that manages demo phase plugins:
    - Registration via decorator or direct call
    - Dependency-aware execution ordering
    - Tag-based filtering
    - Result collection and reporting

    Example:
        registry = DemoPhaseRegistry()

        @registry.register("m1", "Renderer Registry", order=1)
        async def demo_m1():
            return {"status": "success"}

        results = await registry.run_all()
    """

    _instance: Optional['DemoPhaseRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._phases: Dict[str, DemoPhase] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._phases: Dict[str, DemoPhase] = {}
            self._initialized = True

    # ========================================================================
    # Registration
    # ========================================================================

    def register(
        self,
        phase_id: str,
        name: str,
        order: int,
        requires: List[str] = None,
        tags: List[str] = None,
        description: str = ""
    ) -> Callable:
        """
        Decorator to register a demo phase.

        Args:
            phase_id: Unique identifier (e.g., "m1")
            name: Human-readable name
            order: Execution order (lower = earlier)
            requires: Dependencies (phase IDs that must run first)
            tags: Categorization tags
            description: Optional longer description

        Returns:
            Decorator function

        Example:
            @registry.register("m1", "Renderer Registry", order=1, tags=["core"])
            async def demo_m1():
                return {"status": "success"}
        """
        def decorator(func: Callable) -> Callable:
            phase = DemoPhase(
                id=phase_id,
                name=name,
                order=order,
                func=func,
                requires=requires or [],
                tags=tags or [],
                description=description
            )

            if phase_id in self._phases:
                logger.warning(f"Overwriting existing demo phase: {phase_id}")

            self._phases[phase_id] = phase
            logger.debug(f"Registered demo phase: {phase_id} ({name})")

            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def unregister(self, phase_id: str) -> bool:
        """
        Unregister a demo phase.

        Args:
            phase_id: Phase ID to remove

        Returns:
            True if removed, False if not found
        """
        if phase_id in self._phases:
            del self._phases[phase_id]
            logger.debug(f"Unregistered demo phase: {phase_id}")
            return True
        return False

    # ========================================================================
    # Lookup
    # ========================================================================

    def get(self, phase_id: str) -> Optional[DemoPhase]:
        """Get a phase by ID."""
        return self._phases.get(phase_id)

    def list_phases(self) -> List[DemoPhase]:
        """List all phases sorted by order."""
        return sorted(self._phases.values(), key=lambda p: p.order)

    def list_by_tag(self, tag: str) -> List[DemoPhase]:
        """List phases with a specific tag, sorted by order."""
        return sorted(
            [p for p in self._phases.values() if tag in p.tags],
            key=lambda p: p.order
        )

    def get_tags(self) -> Set[str]:
        """Get all unique tags across all phases."""
        tags = set()
        for phase in self._phases.values():
            tags.update(phase.tags)
        return tags

    # ========================================================================
    # Dependency Resolution
    # ========================================================================

    def _resolve_dependencies(self, phase_ids: List[str]) -> List[str]:
        """
        Resolve dependencies and return execution order.

        Uses topological sort to ensure dependencies run first.

        Args:
            phase_ids: Phase IDs to include

        Returns:
            Ordered list of phase IDs

        Raises:
            ValueError: If circular dependency detected
        """
        # Build dependency graph
        visited = set()
        order = []
        temp_visited = set()

        def visit(phase_id: str):
            if phase_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving: {phase_id}")
            if phase_id in visited:
                return

            temp_visited.add(phase_id)

            phase = self._phases.get(phase_id)
            if phase:
                for dep in phase.requires:
                    if dep in phase_ids or dep in self._phases:
                        visit(dep)

            temp_visited.remove(phase_id)
            visited.add(phase_id)
            order.append(phase_id)

        for phase_id in phase_ids:
            visit(phase_id)

        return order

    # ========================================================================
    # Execution
    # ========================================================================

    async def run(self, phase_id: str, **kwargs) -> Dict[str, Any]:
        """
        Run a single demo phase.

        Args:
            phase_id: Phase to run
            **kwargs: Arguments to pass to phase function

        Returns:
            Phase result dict with status, duration, and output
        """
        import time

        phase = self._phases.get(phase_id)
        if not phase:
            return {
                "phase_id": phase_id,
                "status": "error",
                "error": f"Phase not found: {phase_id}"
            }

        start = time.perf_counter()
        try:
            result = await phase.func(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            return {
                "phase_id": phase_id,
                "name": phase.name,
                "status": "success",
                "duration_ms": duration_ms,
                "output": result
            }
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(f"Demo phase {phase_id} failed: {e}")

            return {
                "phase_id": phase_id,
                "name": phase.name,
                "status": "error",
                "duration_ms": duration_ms,
                "error": str(e)
            }

    async def run_all(self, **kwargs) -> Dict[str, Any]:
        """
        Run all registered demo phases in dependency order.

        Args:
            **kwargs: Arguments to pass to all phase functions

        Returns:
            Summary dict with all results
        """
        import time

        phase_ids = [p.id for p in self.list_phases()]
        ordered_ids = self._resolve_dependencies(phase_ids)

        results = []
        total_start = time.perf_counter()

        for phase_id in ordered_ids:
            result = await self.run(phase_id, **kwargs)
            results.append(result)

        total_duration_ms = (time.perf_counter() - total_start) * 1000

        success_count = sum(1 for r in results if r["status"] == "success")

        return {
            "total_phases": len(results),
            "success_count": success_count,
            "failure_count": len(results) - success_count,
            "total_duration_ms": total_duration_ms,
            "results": results
        }

    async def run_by_tag(self, tag: str, **kwargs) -> Dict[str, Any]:
        """
        Run demo phases with a specific tag.

        Args:
            tag: Tag to filter by
            **kwargs: Arguments to pass to phase functions

        Returns:
            Summary dict with results
        """
        import time

        phases = self.list_by_tag(tag)
        phase_ids = [p.id for p in phases]
        ordered_ids = self._resolve_dependencies(phase_ids)

        results = []
        total_start = time.perf_counter()

        for phase_id in ordered_ids:
            result = await self.run(phase_id, **kwargs)
            results.append(result)

        total_duration_ms = (time.perf_counter() - total_start) * 1000

        success_count = sum(1 for r in results if r["status"] == "success")

        return {
            "tag": tag,
            "total_phases": len(results),
            "success_count": success_count,
            "failure_count": len(results) - success_count,
            "total_duration_ms": total_duration_ms,
            "results": results
        }

    # ========================================================================
    # Utilities
    # ========================================================================

    def clear(self) -> None:
        """Clear all registered phases (for testing)."""
        self._phases.clear()

    def summary(self) -> str:
        """Get a human-readable summary of registered phases."""
        lines = ["Jenny Demo Phase Registry", "=" * 40]

        for phase in self.list_phases():
            tags_str = ", ".join(phase.tags) if phase.tags else "none"
            deps_str = ", ".join(phase.requires) if phase.requires else "none"
            lines.append(f"  [{phase.id}] {phase.name}")
            lines.append(f"      Order: {phase.order}, Tags: {tags_str}")
            if phase.requires:
                lines.append(f"      Requires: {deps_str}")

        lines.append(f"\nTotal: {len(self._phases)} phases")
        return "\n".join(lines)


# ============================================================================
# Global Registry Instance
# ============================================================================

# Singleton instance
_registry = DemoPhaseRegistry()


def get_registry() -> DemoPhaseRegistry:
    """Get the global demo phase registry."""
    return _registry


def register_demo(
    phase_id: str,
    name: str,
    order: int,
    requires: List[str] = None,
    tags: List[str] = None,
    description: str = ""
) -> Callable:
    """
    Convenience decorator using global registry.

    Usage:
        @register_demo("m1", "Renderer Registry", order=1, tags=["core"])
        async def demo_m1_renderer_registry():
            return {"status": "success"}
    """
    return _registry.register(
        phase_id=phase_id,
        name=name,
        order=order,
        requires=requires,
        tags=tags,
        description=description
    )


async def run_demo(phase_id: str, **kwargs) -> Dict[str, Any]:
    """Run a single demo phase."""
    return await _registry.run(phase_id, **kwargs)


async def run_all_demos(**kwargs) -> Dict[str, Any]:
    """Run all registered demos."""
    return await _registry.run_all(**kwargs)


async def run_demos_by_tag(tag: str, **kwargs) -> Dict[str, Any]:
    """Run demos filtered by tag."""
    return await _registry.run_by_tag(tag, **kwargs)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'DemoPhase',
    'DemoPhaseRegistry',
    'get_registry',
    'register_demo',
    'run_demo',
    'run_all_demos',
    'run_demos_by_tag',
]
