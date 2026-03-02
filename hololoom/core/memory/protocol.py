"""
Memory Protocols - Protocol-based interfaces for memory backends.
All backends implement MemoryStore protocol for easy extension.

UPDATED (Phase 0, Task 7): This module now imports canonical types and protocols
from hololoom.protocols instead of defining them locally.

Import from here for backward compatibility, or import directly from hololoom.protocols.
"""

import os
import logging
from typing import List, Dict, Optional, Any

from hololoom.utils.security import sanitize_uri

# Import canonical types and protocols
from hololoom.protocols import (
    Memory,
    MemoryQuery,
    MemoryRetrievalResult as RetrievalResult,  # Alias for backward compatibility
    Strategy,
    QueryMode,
    MemoryStore,
    shards_to_memories,
)

# Re-export for backward compatibility
__all__ = [
    'Memory',
    'MemoryQuery',
    'RetrievalResult',
    'Strategy',
    'QueryMode',
    'MemoryStore',
    'shards_to_memories',
    'create_unified_memory',  # Factory function below
]


# ============================================================================
# Helper Functions
# ============================================================================

async def create_unified_memory(
    user_id: str = "default",
    backend: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Backwards-compatible async factory with backend detection and config support.

    Many modules import `create_unified_memory` from this protocol module.
    This enhanced factory:
    - Auto-detects available backends (Neo4j, Qdrant, in-memory)
    - Supports explicit backend selection
    - Handles configuration files
    - Provides graceful degradation with logging
    - Constructs UnifiedMemory from hololoom.memory.unified

    Args:
        user_id: User identifier passed to UnifiedMemory constructor
        backend: Optional explicit backend ("neo4j", "qdrant", "in-memory", "hybrid")
                 If None, auto-detects best available backend
        config: Optional config dict with keys:
                - neo4j_uri: Neo4j connection URI
                - qdrant_url: Qdrant server URL
                - enable_mem0: Enable mem0 extraction (default: True)
                - enable_hofstadter: Enable Hofstadter patterns (default: True)
        **kwargs: Additional args forwarded to UnifiedMemory

    Returns:
        UnifiedMemory instance configured with best available backend

    Examples:
        # Auto-detect backend
        memory = await create_unified_memory(user_id="blake")

        # Explicit backend
        memory = await create_unified_memory(
            user_id="blake",
            backend="neo4j",
            config={"neo4j_uri": "bolt://localhost:7687"}
        )

        # In-memory (testing/development)
        memory = await create_unified_memory(
            user_id="test",
            backend="in-memory"
        )
    """
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    config = config or {}
    
    # Try to import UnifiedMemory
    try:
        from hololoom.memory.unified import UnifiedMemory
    except ImportError as e:
        raise ImportError(
            f"UnifiedMemory implementation not available: {e}\n"
            "Ensure HoloLoom.memory.unified exists or pass a memory backend directly."
        )
    
    # Backend detection and configuration
    enable_neo4j = True
    enable_qdrant = True
    enable_mem0 = config.get('enable_mem0', True)
    enable_hofstadter = config.get('enable_hofstadter', True)
    
    if backend:
        # Explicit backend selection
        backend_lower = backend.lower()
        
        if backend_lower == "in-memory":
            # Disable external backends
            enable_neo4j = False
            enable_qdrant = False
            logger.info("Using in-memory backend (no persistence)")
            
        elif backend_lower == "neo4j":
            enable_qdrant = False
            logger.info("Using Neo4j backend")
            
        elif backend_lower == "qdrant":
            enable_neo4j = False
            logger.info("Using Qdrant backend")
            
        elif backend_lower == "hybrid":
            # Use both Neo4j and Qdrant
            logger.info("Using hybrid Neo4j + Qdrant backend")
            
        else:
            logger.warning(f"Unknown backend '{backend}', falling back to auto-detect")
    
    else:
        # Auto-detect available backends
        logger.info("Auto-detecting available memory backends...")
        
        # Check Neo4j availability
        neo4j_uri = config.get('neo4j_uri') or os.getenv('NEO4J_URI')
        if not neo4j_uri:
            enable_neo4j = False
            logger.debug("Neo4j not configured (no URI)")
        else:
            # Try to connect (optional health check)
            try:
                # TODO: Add actual Neo4j connection test
                logger.info(f"Neo4j configured at {sanitize_uri(neo4j_uri)}")
            except Exception as e:
                enable_neo4j = False
                logger.warning(f"Neo4j unavailable: {e}")
        
        # Check Qdrant availability
        qdrant_url = config.get('qdrant_url') or os.getenv('QDRANT_URL')
        if not qdrant_url:
            enable_qdrant = False
            logger.debug("Qdrant not configured (no URL)")
        else:
            # Try to connect (optional health check)
            try:
                # TODO: Add actual Qdrant connection test
                logger.info(f"Qdrant configured at {sanitize_uri(qdrant_url)}")
            except Exception as e:
                enable_qdrant = False
                logger.warning(f"Qdrant unavailable: {e}")
        
        # Log selected backend
        if enable_neo4j and enable_qdrant:
            logger.info("✓ Using hybrid Neo4j + Qdrant backend")
        elif enable_neo4j:
            logger.info("✓ Using Neo4j backend")
        elif enable_qdrant:
            logger.info("✓ Using Qdrant backend")
        else:
            logger.info("✓ Using in-memory backend (fallback)")
    
    # Construct UnifiedMemory with detected/configured backends
    try:
        memory = UnifiedMemory(
            user_id=user_id,
            enable_mem0=enable_mem0,
            enable_neo4j=enable_neo4j,
            enable_qdrant=enable_qdrant,
            enable_hofstadter=enable_hofstadter,
            **kwargs
        )
        
        logger.info(f"✓ UnifiedMemory initialized for user '{user_id}'")
        return memory
        
    except Exception as e:
        logger.error(f"Failed to initialize UnifiedMemory: {e}")
        raise RuntimeError(
            f"UnifiedMemory initialization failed: {e}\n"
            "Check backend configuration and dependencies."
        )