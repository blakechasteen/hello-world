"""
Jenny Demo Package
==================
Auto-discovering demo system for Jenny Moonshot phases.

This package provides:
- Automatic demo module discovery
- Phase registration via decorators
- Unified execution interface
- Extensible plugin architecture

Usage:
    # Run all demos
    from demos.jenny import run_all_demos
    results = await run_all_demos()

    # Run specific phase
    from demos.jenny import run_demo
    result = await run_demo("m1")

    # Run by tag
    from demos.jenny import run_demos_by_tag
    results = await run_demos_by_tag("core")

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot Extensibility)
"""

import importlib
import pkgutil
import logging
from pathlib import Path
from typing import List

from .phase_registry import (
    DemoPhase,
    DemoPhaseRegistry,
    get_registry,
    register_demo,
    run_demo,
    run_all_demos,
    run_demos_by_tag,
)

from .utils import (
    demo_phase,
    with_fallback,
    retry_on_error,
    print_header,
    print_subheader,
    print_metric,
    print_metrics,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_list,
    ProgressTracker,
    format_demo_summary,
    print_demo_summary,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Plugin Discovery
# ============================================================================

def discover_demo_modules() -> List[str]:
    """
    Auto-discover and import demo modules in this package.

    Looks for modules starting with 'demo_m' (e.g., demo_m1, demo_m2).
    Each module registers its phases via @register_demo decorator.

    Returns:
        List of discovered module names
    """
    demo_dir = Path(__file__).parent
    discovered = []

    for module_info in pkgutil.iter_modules([str(demo_dir)]):
        # Look for demo_m* modules (demo_m1, demo_m2_async, etc.)
        if module_info.name.startswith('demo_m'):
            try:
                full_name = f'demos.jenny.{module_info.name}'
                importlib.import_module(full_name)
                discovered.append(module_info.name)
                logger.debug(f"Discovered demo module: {module_info.name}")
            except Exception as e:
                logger.warning(f"Failed to import {module_info.name}: {e}")

    return discovered


def discover_and_register() -> int:
    """
    Discover and register all demo modules.

    Returns:
        Number of phases registered
    """
    registry = get_registry()
    initial_count = len(registry.list_phases())

    discovered = discover_demo_modules()

    final_count = len(registry.list_phases())
    new_phases = final_count - initial_count

    logger.info(
        f"Discovered {len(discovered)} demo modules, "
        f"registered {new_phases} new phases"
    )

    return new_phases


# ============================================================================
# Convenience Functions
# ============================================================================

def list_all_phases() -> List[DemoPhase]:
    """List all registered demo phases."""
    return get_registry().list_phases()


def get_registry_summary() -> str:
    """Get human-readable registry summary."""
    return get_registry().summary()


# ============================================================================
# Auto-Discovery on Import (Optional)
# ============================================================================

# Uncomment to auto-discover on package import:
# discover_demo_modules()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Registry
    'DemoPhase',
    'DemoPhaseRegistry',
    'get_registry',
    'register_demo',
    'run_demo',
    'run_all_demos',
    'run_demos_by_tag',

    # Discovery
    'discover_demo_modules',
    'discover_and_register',
    'list_all_phases',
    'get_registry_summary',

    # Utilities
    'demo_phase',
    'with_fallback',
    'retry_on_error',
    'print_header',
    'print_subheader',
    'print_metric',
    'print_metrics',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'print_list',
    'ProgressTracker',
    'format_demo_summary',
    'print_demo_summary',
]
