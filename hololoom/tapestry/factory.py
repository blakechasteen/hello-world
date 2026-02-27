"""
Tapestry Backend Factory

Factory with auto-detection and graceful fallback.

Follows HoloLoom's create_memory_backend() pattern:
- Auto-detect best available backend
- Graceful fallback to simpler options
- Clear logging of choices

Created: December 2025
"""

import logging
from typing import Optional

from hololoom.tapestry.protocol import TapestryBackend
from hololoom.tapestry.backends.json_backend import JsonTapestryBackend

logger = logging.getLogger(__name__)


async def create_tapestry_backend(
    backend: Optional[str] = None,
    path: Optional[str] = None
) -> TapestryBackend:
    """
    Factory with graceful fallback.

    Follows HoloLoom's create_memory_backend() pattern.

    Args:
        backend: Backend type ("json" or "sqlite"). None for auto-detect.
        path: Path to storage file. None uses default.

    Returns:
        TapestryBackend implementation.

    Backends:
        - "json": Atomic JSON file storage (always available)
        - "sqlite": SQLite for durability (future)

    Auto-detection prefers JSON (always available, human-readable).
    """
    # Default path
    if path is None:
        path = ".hololoom/tapestry.json"

    # Auto-detect: prefer JSON (always available)
    if backend is None:
        backend = "json"
        logger.debug(f"Auto-detected backend: {backend}")

    # Create backend
    if backend == "json":
        logger.info(f"Creating JSON backend at {path}")
        return JsonTapestryBackend(path)

    elif backend == "sqlite":
        # SQLite backend (optional, for future durability needs)
        sqlite_path = path.replace('.json', '.db')
        try:
            from hololoom.tapestry.backends.sqlite_backend import SqliteTapestryBackend
            logger.info(f"Creating SQLite backend at {sqlite_path}")
            return SqliteTapestryBackend(sqlite_path)
        except ImportError:
            logger.warning(
                f"SQLite backend not available, falling back to JSON. "
                f"Install aiosqlite for SQLite support."
            )
            return JsonTapestryBackend(path)

    else:
        logger.warning(f"Unknown backend '{backend}', falling back to JSON")
        return JsonTapestryBackend(path)


def get_default_tapestry_path() -> str:
    """
    Get default tapestry path.

    Returns:
        Default path: .hololoom/tapestry.json
    """
    return ".hololoom/tapestry.json"


def get_project_tapestry_path(project_root: Optional[str] = None) -> str:
    """
    Get project-specific tapestry path.

    Args:
        project_root: Project root directory. None uses current directory.

    Returns:
        Path to tapestry file within project.
    """
    import os
    root = project_root or os.getcwd()
    return os.path.join(root, ".hololoom", "tapestry.json")
