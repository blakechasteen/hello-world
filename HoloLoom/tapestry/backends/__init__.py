"""
Tapestry Backend Implementations

Pluggable persistence backends for Tapestry state.

Available backends:
- JsonTapestryBackend: Atomic JSON file storage (default)
- SqliteTapestryBackend: SQLite for durability (optional)

Usage:
    from HoloLoom.tapestry.backends import JsonTapestryBackend

    backend = JsonTapestryBackend(".hololoom/tapestry.json")
    await backend.save(tapestry)
    loaded = await backend.load()

Created: December 2025
"""

from HoloLoom.tapestry.backends.json_backend import JsonTapestryBackend

__all__ = [
    "JsonTapestryBackend",
]

# Lazy import for optional SQLite backend
def __getattr__(name):
    if name == "SqliteTapestryBackend":
        from HoloLoom.tapestry.backends.sqlite_backend import SqliteTapestryBackend
        return SqliteTapestryBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
