"""
Memory Routes
=============

Endpoints for memory storage and retrieval.

Endpoints:
    POST /memories/add - Add new memory to storage
    POST /api/remember - Store content with IDE context
    POST /api/recall - Search memories semantically
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from hololoom.protocols.types import MemoryShard

from ..state import state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Memory"])


@router.post("/memories/add")
async def add_memory(memory: Dict):
    """
    Add new memory to persistent storage.

    Args:
        memory: Dict with text, episode, entities, motifs, metadata

    Returns:
        Success status and memory ID

    Example:
        POST /memories/add
        {
          "text": "Thompson Sampling balances exploration and exploitation",
          "episode": "algorithms",
          "entities": ["Thompson Sampling"],
          "motifs": ["definition"],
          "metadata": {"topic": "ML", "confidence": 0.9}
        }
    """
    try:
        if not state.memory_backend:
            return {
                "success": False,
                "message": "Persistent backend not available",
                "memory_id": None
            }

        from hololoom.memory.protocol import Memory

        # Create Memory object
        new_memory = Memory(
            id=f"mem_{datetime.now().timestamp()}",
            text=memory.get("text", ""),
            context={
                "episode": memory.get("episode", "default"),
                "entities": memory.get("entities", []),
                "motifs": memory.get("motifs", [])
            },
            metadata=memory.get("metadata", {})
        )

        # Store in persistent backend
        await state.memory_backend.store([new_memory])

        # Also add to in-memory shards for immediate availability
        shard = MemoryShard(
            id=new_memory.id,
            text=new_memory.text,
            episode=new_memory.context.get("episode", "default"),
            entities=new_memory.context.get("entities", []),
            motifs=new_memory.context.get("motifs", []),
            metadata=new_memory.metadata
        )
        state.shards.append(shard)

        logger.info(f"Added memory: {new_memory.id}")

        return {
            "success": True,
            "message": "Memory added successfully",
            "memory_id": new_memory.id
        }

    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/remember")
async def api_remember(request: Dict[str, Any]):
    """
    Store content to HoloLoom memory with IDE context.

    Endpoint for Promptly VS Code extension to capture developer notes,
    decisions, and code context into the knowledge graph.

    Args:
        request: Dict with 'content' and optional 'context'
            - content (str): The content to remember
            - context (dict): IDE context (workspace, file, timestamp, etc.)

    Returns:
        Success status and memory ID

    Example:
        POST /api/remember
        {
          "content": "We decided to use PostgreSQL for authentication",
          "context": {
            "workspace": "my-project",
            "file": "src/auth.ts",
            "timestamp": "2025-11-15T10:30:00Z"
          }
        }
    """
    try:
        content = request.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="Missing 'content' field")

        context = request.get("context", {})

        # Import HoloLoom unified API
        from hololoom import hololoom

        # Create HoloLoom instance with current config
        async with HoloLoom(config=state.config) as loom:
            # Experience the content (stores in knowledge graph)
            memory = await loom.experience(content, context=context)

            logger.info(f"Remembered via /api/remember: {content[:50]}...")

            return {
                "status": "success",
                "message": f"Saved to HoloLoom memory",
                "memory_id": memory.id
            }

    except Exception as e:
        logger.error(f"Failed to remember content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/recall")
async def api_recall(request: Dict[str, Any]):
    """
    Search HoloLoom memories using semantic + keyword search.

    Endpoint for Promptly VS Code extension to retrieve relevant memories
    based on a query. Uses hybrid BM25 + semantic search for best results.

    Args:
        request: Dict with 'query' and optional 'k'
            - query (str): Search query
            - k (int): Number of results to return (default: 5)

    Returns:
        List of matching memories with confidence scores

    Example:
        POST /api/recall
        {
          "query": "What database did we choose?",
          "k": 5
        }

        Response:
        {
          "memories": [
            {
              "content": "We decided to use PostgreSQL for authentication",
              "confidence": 0.92,
              "timestamp": "2025-11-15T10:30:00Z",
              "source": "src/auth.ts"
            },
            ...
          ]
        }
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' field")

        k = request.get("k", 5)

        # Import HoloLoom unified API
        from hololoom import hololoom

        # Create HoloLoom instance with current config
        async with HoloLoom(config=state.config) as loom:
            # Recall memories
            memories = await loom.recall(query, k=k)

            logger.info(f"Recalled {len(memories)} memories for: {query[:50]}...")

            # Format response to match TypeScript interface
            return {
                "memories": [
                    {
                        "content": m.text,
                        "confidence": m.metadata.get("confidence", 0.85),
                        "timestamp": m.metadata.get("timestamp", "unknown"),
                        "source": m.metadata.get("file", "unknown")
                    }
                    for m in memories
                ]
            }

    except Exception as e:
        logger.error(f"Failed to recall memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
