"""
Jenny Renderer Registry
========================
Singleton registry for extensible renderer plugins.

Philosophy:
> "Renderers are plugins, not hardcoded switches."

The registry enables:
- Dynamic renderer registration at import time
- Priority-based renderer selection
- Concurrent rendering for stateless renderers
- Plugin discovery for third-party renderers

Usage:
    # Register a renderer
    @register_renderer(priority=10)
    class MyCustomRenderer(JennyRendererBase):
        ...

    # Get renderer by target
    renderer = RendererRegistry.get_for_target(RenderTarget.HTML)

    # Render with automatic renderer selection
    output = await RendererRegistry.render(specs, target=RenderTarget.HTML)

    # Concurrent rendering across multiple targets
    outputs = await RendererRegistry.render_concurrent(
        specs,
        targets=[RenderTarget.HTML, RenderTarget.JSON]
    )

Author: HoloLoom Team
Date: 2025-12-08 (Jenny Moonshot M1)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type, Callable, Any, Set
from dataclasses import dataclass, field
from functools import wraps

from HoloLoom.protocols.jenny import (
    JennyRendererProtocol,
    RenderTarget,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Registry Entry
# ============================================================================

@dataclass
class RendererEntry:
    """
    Registry entry for a renderer.

    Attributes:
        renderer_class: The renderer class (not instance)
        instance: Lazily created singleton instance
        priority: Higher = preferred when multiple match
        targets: Set of supported RenderTarget values
        concurrent: Whether safe for parallel rendering
    """
    renderer_class: Type[JennyRendererProtocol]
    instance: Optional[JennyRendererProtocol] = None
    priority: int = 0
    targets: Set[RenderTarget] = field(default_factory=set)
    concurrent: bool = True

    def get_instance(self) -> JennyRendererProtocol:
        """Get or create singleton instance."""
        if self.instance is None:
            self.instance = self.renderer_class()
        return self.instance


# ============================================================================
# Renderer Registry (Singleton)
# ============================================================================

class RendererRegistryMeta(type):
    """Metaclass for singleton registry."""
    _instance: Optional['RendererRegistry'] = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class RendererRegistry(metaclass=RendererRegistryMeta):
    """
    Singleton registry for Jenny renderers.

    Thread-safe registry that manages renderer plugins:
    - Registration by name or decorator
    - Lookup by target format
    - Priority-based selection when multiple match
    - Concurrent rendering support

    Example:
        # Manual registration
        RendererRegistry().register(HTMLRenderer, priority=10)

        # Decorator registration
        @RendererRegistry.register_decorator(priority=5)
        class MyRenderer:
            ...

        # Get best renderer for target
        renderer = RendererRegistry().get_for_target(RenderTarget.HTML)

        # Render with auto-selection
        html = await RendererRegistry().render(specs, RenderTarget.HTML)
    """

    def __init__(self):
        self._renderers: Dict[str, RendererEntry] = {}
        self._target_index: Dict[RenderTarget, List[str]] = {
            target: [] for target in RenderTarget
        }
        self._lock = asyncio.Lock()
        self._initialized = False

    # ========================================================================
    # Registration
    # ========================================================================

    def register(
        self,
        renderer_class: Type[JennyRendererProtocol],
        priority: int = 0,
        name: Optional[str] = None
    ) -> None:
        """
        Register a renderer class.

        Args:
            renderer_class: Renderer class implementing JennyRendererProtocol
            priority: Higher = preferred (default: 0)
            name: Override name (default: from renderer.name property)
        """
        # Create temporary instance to get metadata
        temp_instance = renderer_class()

        renderer_name = name or temp_instance.name
        targets = set(temp_instance.supported_targets)
        concurrent = temp_instance.supports_concurrent()

        # Create entry
        entry = RendererEntry(
            renderer_class=renderer_class,
            instance=temp_instance,  # Reuse the instance we created
            priority=priority,
            targets=targets,
            concurrent=concurrent
        )

        # Register by name
        if renderer_name in self._renderers:
            existing = self._renderers[renderer_name]
            if existing.priority >= priority:
                logger.debug(
                    f"Renderer '{renderer_name}' already registered with "
                    f"higher priority ({existing.priority} >= {priority})"
                )
                return
            logger.info(
                f"Replacing renderer '{renderer_name}' "
                f"(priority {existing.priority} → {priority})"
            )

        self._renderers[renderer_name] = entry

        # Update target index
        for target in targets:
            if renderer_name not in self._target_index[target]:
                self._target_index[target].append(renderer_name)
            # Sort by priority (highest first)
            self._target_index[target].sort(
                key=lambda n: self._renderers[n].priority,
                reverse=True
            )

        logger.info(
            f"Registered renderer '{renderer_name}' "
            f"(priority={priority}, targets={[t.value for t in targets]}, "
            f"concurrent={concurrent})"
        )

    def unregister(self, name: str) -> bool:
        """
        Unregister a renderer by name.

        Args:
            name: Renderer name to remove

        Returns:
            True if removed, False if not found
        """
        if name not in self._renderers:
            return False

        entry = self._renderers.pop(name)

        # Remove from target index
        for target in entry.targets:
            if name in self._target_index[target]:
                self._target_index[target].remove(name)

        logger.info(f"Unregistered renderer '{name}'")
        return True

    @staticmethod
    def register_decorator(
        priority: int = 0,
        name: Optional[str] = None
    ) -> Callable:
        """
        Decorator for renderer registration.

        Usage:
            @RendererRegistry.register_decorator(priority=10)
            class HTMLRenderer(JennyRendererBase):
                ...
        """
        def decorator(cls: Type[JennyRendererProtocol]) -> Type[JennyRendererProtocol]:
            RendererRegistry().register(cls, priority=priority, name=name)
            return cls
        return decorator

    # ========================================================================
    # Lookup
    # ========================================================================

    def get_by_name(self, name: str) -> Optional[JennyRendererProtocol]:
        """
        Get renderer by name.

        Args:
            name: Renderer name

        Returns:
            Renderer instance or None if not found
        """
        entry = self._renderers.get(name)
        return entry.get_instance() if entry else None

    def get_for_target(
        self,
        target: RenderTarget,
        prefer_concurrent: bool = False
    ) -> Optional[JennyRendererProtocol]:
        """
        Get best renderer for a target format.

        Args:
            target: Desired render target
            prefer_concurrent: If True, prefer concurrent-safe renderers

        Returns:
            Best matching renderer or None
        """
        names = self._target_index.get(target, [])
        if not names:
            return None

        if prefer_concurrent:
            # Find first concurrent-safe renderer
            for name in names:
                entry = self._renderers[name]
                if entry.concurrent:
                    return entry.get_instance()

        # Return highest priority
        return self._renderers[names[0]].get_instance()

    def get_all_for_target(
        self,
        target: RenderTarget
    ) -> List[JennyRendererProtocol]:
        """
        Get all renderers supporting a target.

        Args:
            target: Desired render target

        Returns:
            List of renderers (priority-sorted)
        """
        names = self._target_index.get(target, [])
        return [self._renderers[n].get_instance() for n in names]

    def list_renderers(self) -> List[Dict[str, Any]]:
        """
        List all registered renderers.

        Returns:
            List of renderer info dicts
        """
        return [
            {
                "name": name,
                "priority": entry.priority,
                "targets": [t.value for t in entry.targets],
                "concurrent": entry.concurrent,
            }
            for name, entry in self._renderers.items()
        ]

    # ========================================================================
    # Rendering
    # ========================================================================

    async def render(
        self,
        specs: List[Any],
        target: RenderTarget = RenderTarget.HTML,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render specs using best available renderer.

        Args:
            specs: JennySpec list to render
            target: Output format
            options: Renderer-specific options

        Returns:
            Rendered output string

        Raises:
            ValueError: If no renderer available for target
        """
        renderer = self.get_for_target(target)
        if renderer is None:
            raise ValueError(f"No renderer available for target: {target.value}")

        return await renderer.render(specs, target=target, options=options)

    async def render_concurrent(
        self,
        specs: List[Any],
        targets: List[RenderTarget],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[RenderTarget, str]:
        """
        Render to multiple targets concurrently.

        Only uses concurrent-safe renderers. Falls back to sequential
        for non-concurrent renderers.

        Args:
            specs: JennySpec list to render
            targets: List of output formats
            options: Shared renderer options

        Returns:
            Dict mapping target → rendered output
        """
        results: Dict[RenderTarget, str] = {}
        concurrent_tasks = []
        sequential_targets = []

        for target in targets:
            renderer = self.get_for_target(target, prefer_concurrent=True)
            if renderer is None:
                logger.warning(f"No renderer for target: {target.value}")
                continue

            entry = self._renderers.get(renderer.name)
            if entry and entry.concurrent:
                # Safe for concurrent execution
                concurrent_tasks.append((
                    target,
                    renderer.render(specs, target=target, options=options)
                ))
            else:
                # Must run sequentially
                sequential_targets.append((target, renderer))

        # Run concurrent tasks
        if concurrent_tasks:
            async with self._lock:
                coros = [task for _, task in concurrent_tasks]
                outputs = await asyncio.gather(*coros, return_exceptions=True)

                for (target, _), output in zip(concurrent_tasks, outputs):
                    if isinstance(output, Exception):
                        logger.error(f"Render failed for {target.value}: {output}")
                    else:
                        results[target] = output

        # Run sequential tasks
        for target, renderer in sequential_targets:
            try:
                output = await renderer.render(specs, target=target, options=options)
                results[target] = output
            except Exception as e:
                logger.error(f"Render failed for {target.value}: {e}")

        return results

    # ========================================================================
    # Plugin Discovery
    # ========================================================================

    def discover_plugins(self, package_path: str = "HoloLoom.visualization") -> int:
        """
        Discover and register renderer plugins from a package.

        Looks for classes with @register_renderer decorator
        or subclasses of JennyRendererBase.

        Args:
            package_path: Python package path to scan

        Returns:
            Number of renderers discovered
        """
        import importlib
        import pkgutil

        count = 0
        try:
            package = importlib.import_module(package_path)

            for _, name, _ in pkgutil.iter_modules(package.__path__):
                if name.startswith('jenny_renderer'):
                    try:
                        module = importlib.import_module(f"{package_path}.{name}")
                        # Renderers auto-register via decorator
                        logger.debug(f"Loaded renderer module: {name}")
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {name}: {e}")
        except Exception as e:
            logger.warning(f"Plugin discovery failed: {e}")

        return count

    def clear(self) -> None:
        """Clear all registered renderers (for testing)."""
        self._renderers.clear()
        self._target_index = {target: [] for target in RenderTarget}
        self._initialized = False


# ============================================================================
# Convenience Functions
# ============================================================================

def get_registry() -> RendererRegistry:
    """Get the singleton registry instance."""
    return RendererRegistry()


def register_renderer(
    priority: int = 0,
    name: Optional[str] = None
) -> Callable:
    """
    Decorator to register a renderer class.

    Usage:
        @register_renderer(priority=10)
        class HTMLRenderer(JennyRendererBase):
            name = "html"
            ...
    """
    return RendererRegistry.register_decorator(priority=priority, name=name)


async def render_jenny(
    specs: List[Any],
    target: RenderTarget = RenderTarget.HTML,
    options: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to render JennySpecs.

    Args:
        specs: JennySpec list to render
        target: Output format (default: HTML)
        options: Renderer options

    Returns:
        Rendered output string
    """
    return await get_registry().render(specs, target=target, options=options)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'RendererRegistry',
    'RendererEntry',
    'get_registry',
    'register_renderer',
    'render_jenny',
]
