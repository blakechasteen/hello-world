"""
HoloLoom Simple API - One-line memory operations

The simplest possible interface to HoloLoom's persistent memory system.
Zero configuration, just import and use.

Usage:
    import hololoom_simple as loom

    # Store memories
    loom.remember("I love Python")
    loom.remember("Thompson Sampling balances exploration/exploitation")

    # Retrieve memories
    results = loom.recall("What do I like?")
    print(results[0]['text'])  # "I love Python"

    # View statistics
    stats = loom.metrics()
    print(f"Total memories: {stats['total_memories']}")

    # Clear everything (careful!)
    loom.reset()

Design Philosophy:
- Zero configuration (smart defaults)
- Synchronous API (no async/await)
- Global singleton (no instance management)
- Graceful initialization (lazy loading)
- Maximum simplicity (remember/recall/metrics/reset)
"""

import asyncio
import atexit
from typing import List, Dict, Any, Optional
from pathlib import Path

# Lazy imports - only load when first used
_HoloLoom = None
_Config = None

# Global singleton state
_global_loom: Optional[Any] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_is_initialized = False


def _lazy_import_hololoom():
    """Lazy import HoloLoom to avoid heavy imports at module load time."""
    global _HoloLoom, _Config
    if _HoloLoom is None:
        from HoloLoom import HoloLoom, Config
        _HoloLoom = HoloLoom
        _Config = Config


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Get existing event loop or create new one."""
    global _event_loop

    if _event_loop is None or _event_loop.is_closed():
        # Try to get current loop
        try:
            _event_loop = asyncio.get_event_loop()
            if _event_loop.is_closed():
                raise RuntimeError("Loop is closed")
        except RuntimeError:
            # Create new loop
            _event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_event_loop)

    return _event_loop


def _initialize():
    """Initialize global HoloLoom instance (lazy, only once)."""
    global _global_loom, _is_initialized

    if _is_initialized and _global_loom is not None:
        return

    # Lazy import
    _lazy_import_hololoom()

    # Create instance with fast config (good balance)
    config = _Config.fast()
    _global_loom = _HoloLoom(config=config)

    # Start the async context manager
    loop = _get_or_create_loop()
    loop.run_until_complete(_global_loom.__aenter__())

    _is_initialized = True

    # Register cleanup on exit
    atexit.register(_cleanup)


def _cleanup():
    """Cleanup on program exit."""
    global _global_loom, _event_loop, _is_initialized

    if _global_loom is not None and _event_loop is not None:
        try:
            # Close the async context manager
            _event_loop.run_until_complete(
                _global_loom.__aexit__(None, None, None)
            )
        except Exception:
            pass  # Ignore cleanup errors
        finally:
            _global_loom = None
            _is_initialized = False


def _run_async(coro):
    """Run async coroutine synchronously."""
    _initialize()
    loop = _get_or_create_loop()
    return loop.run_until_complete(coro)


# ============================================================================
# PUBLIC API - The magic one-liners
# ============================================================================

def remember(text: str) -> Dict[str, Any]:
    """
    Store a memory. Simple as that.

    Args:
        text: Anything you want to remember

    Returns:
        Dictionary with memory metadata

    Example:
        >>> import hololoom_simple as loom
        >>> loom.remember("I love TypeScript")
        {'status': 'stored', 'id': '...', 'confidence': 1.0}
    """
    _initialize()
    memory = _run_async(_global_loom.experience(text))

    return {
        'status': 'stored',
        'id': memory.id if hasattr(memory, 'id') else str(hash(text)),
        'text': text,
        'confidence': 1.0
    }


def recall(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve memories. Just ask.

    Args:
        query: What you want to remember
        limit: Maximum number of results (default: 5)

    Returns:
        List of memories with text and confidence scores

    Example:
        >>> import hololoom_simple as loom
        >>> loom.remember("Python has decorators")
        >>> results = loom.recall("What languages do I know?")
        >>> print(results[0]['text'])
        'Python has decorators'
    """
    _initialize()
    memories = _run_async(_global_loom.recall(query, k=limit))

    # Convert to simple dict format
    results = []
    for mem in memories:
        results.append({
            'text': mem.content if hasattr(mem, 'content') else str(mem),
            'confidence': mem.score if hasattr(mem, 'score') else 0.0,
            'id': mem.id if hasattr(mem, 'id') else str(hash(mem))
        })

    return results


def metrics() -> Dict[str, Any]:
    """
    Get memory system statistics.

    Returns:
        Dictionary with system metrics

    Example:
        >>> import hololoom_simple as loom
        >>> stats = loom.metrics()
        >>> print(f"Total memories: {stats['total_memories']}")
        >>> print(f"Active memories: {stats['active_memories']}")
    """
    _initialize()
    raw_metrics = _global_loom.get_metrics()

    # Simplify metrics for readability
    return {
        'total_memories': raw_metrics.get('activation', {}).get('total_nodes', 0),
        'active_memories': raw_metrics.get('activation', {}).get('active_nodes', 0),
        'coherence': raw_metrics.get('coherence', {}).get('global_coherence', 0.0),
        'mean_activation': raw_metrics.get('activation', {}).get('mean_activation', 0.0),
        'system_status': 'healthy' if raw_metrics.get('coherence', {}).get('global_coherence', 0) > 0.5 else 'degraded'
    }


def reset():
    """
    Clear all memories and reset the system.

    WARNING: This deletes all stored memories!

    Example:
        >>> import hololoom_simple as loom
        >>> loom.reset()  # Start fresh
    """
    global _global_loom, _is_initialized

    if _global_loom is not None:
        _cleanup()

    # Reinitialize
    _initialize()


def search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search memories (alias for recall with higher limit).

    Args:
        query: Search query
        limit: Maximum results (default: 10)

    Returns:
        List of matching memories

    Example:
        >>> import hololoom_simple as loom
        >>> results = loom.search("machine learning", limit=20)
    """
    return recall(query, limit=limit)


def status() -> Dict[str, str]:
    """
    Get system status.

    Returns:
        Dictionary with status information

    Example:
        >>> import hololoom_simple as loom
        >>> print(loom.status())
        {'initialized': 'yes', 'backend': 'awareness_graph', 'mode': 'fast'}
    """
    _initialize()

    return {
        'initialized': 'yes' if _is_initialized else 'no',
        'backend': 'awareness_graph',
        'mode': _global_loom.config.mode.value if hasattr(_global_loom.config, 'mode') else 'fast',
        'multimodal': 'available' if _global_loom.router is not None else 'unavailable'
    }


# ============================================================================
# Alternative names (for discoverability)
# ============================================================================

store = remember  # Alias
retrieve = recall  # Alias
find = search      # Alias
stats = metrics    # Alias


# ============================================================================
# Module-level help
# ============================================================================

def help():
    """Print help information."""
    print(__doc__)
    print("\nAvailable functions:")
    print("  remember(text)        - Store a memory")
    print("  recall(query, limit)  - Retrieve memories")
    print("  search(query, limit)  - Search memories (alias)")
    print("  metrics()             - Get statistics")
    print("  status()              - Get system status")
    print("  reset()               - Clear all memories")
    print("  help()                - Show this help")
    print("\nAliases:")
    print("  store = remember")
    print("  retrieve = recall")
    print("  find = search")
    print("  stats = metrics")


if __name__ == '__main__':
    # Demo when run directly
    print("HoloLoom Simple API - Interactive Demo\n")

    print("Storing memories...")
    remember("I love Python programming")
    remember("Thompson Sampling balances exploration and exploitation")
    remember("HoloLoom is a persistent memory system")

    print("\nRecalling memories...")
    results = recall("What programming languages do I like?")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['text']} (confidence: {result['confidence']:.2f})")

    print("\nSystem metrics:")
    m = metrics()
    for key, value in m.items():
        print(f"  {key}: {value}")

    print("\nSystem status:")
    s = status()
    for key, value in s.items():
        print(f"  {key}: {value}")

    print("\nDemo complete!")
