"""
Memory Router
=============

Memory-related endpoints for storing and retrieving knowledge.
Extracted from agentic_api.py as part of W2 (Monolithic Files) SWOT remediation.

Endpoints:
- POST /memories/add - Add memory to persistent storage
- POST /api/remember - Store content with IDE context
- POST /api/recall - Semantic + keyword search
- GET /api/todos - Extract TODOs with importance scoring
"""

import re
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Memory"])


# ============================================================================
# Request/Response Models
# ============================================================================

class MemoryAddRequest(BaseModel):
    """
    Request model for adding memories.

    SECURITY: Pydantic validation prevents resource exhaustion attacks.
    All fields have explicit size limits.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Memory text content (max 100KB)"
    )
    episode: str = Field(
        default="default",
        max_length=1000,
        description="Episode/category for the memory (max 1KB)"
    )
    entities: List[str] = Field(
        default_factory=list,
        max_items=100,
        description="List of entities mentioned (max 100 items)"
    )
    motifs: List[str] = Field(
        default_factory=list,
        max_items=100,
        description="List of motifs/patterns (max 100 items)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @validator('entities', 'motifs', each_item=True)
    def validate_list_items(cls, v):
        """SECURITY: Limit individual list item lengths."""
        if len(v) > 1000:
            raise ValueError("List items must be under 1KB")
        return v

    @validator('metadata')
    def validate_metadata_size(cls, v):
        """SECURITY: Limit metadata size to prevent abuse."""
        import json
        if len(json.dumps(v)) > 10000:
            raise ValueError("Metadata must be under 10KB when serialized")
        return v


# ============================================================================
# Dependencies
# ============================================================================

def get_server_state(request: Request):
    """
    Dependency to get server state from the app.

    The state is attached to the app during startup in agentic_api.py.
    """
    return request.app.state.server_state


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/memories/add")
async def add_memory(memory: MemoryAddRequest, state=Depends(get_server_state)):
    """
    Add new memory to persistent storage.

    SECURITY: Input validated via MemoryAddRequest Pydantic model with:
    - Text limited to 100KB
    - Episode limited to 1KB
    - Entities/motifs limited to 100 items of 1KB each
    - Metadata limited to 10KB serialized

    Args:
        memory: MemoryAddRequest with text, episode, entities, motifs, metadata

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
        from hololoom.protocols.types import MemoryShard

        # Create Memory object - fields already validated by Pydantic
        new_memory = Memory(
            id=f"mem_{datetime.now().timestamp()}",
            text=memory.text,
            context={
                "episode": memory.episode,
                "entities": memory.entities,
                "motifs": memory.motifs
            },
            metadata=memory.metadata
        )

        # Store in persistent backend
        await state.memory_backend.store([new_memory])

        # Also add to in-memory shards for immediate availability
        shard = MemoryShard(
            id=new_memory.id,
            text=new_memory.text,
            episode=memory.episode,
            entities=memory.entities,
            motifs=memory.motifs,
            metadata=memory.metadata
        )
        state.shards.append(shard)

        logger.info(f"Added memory: {new_memory.id}")

        return {
            "success": True,
            "message": "Memory added successfully",
            "memory_id": new_memory.id
        }

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Failed to add memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add memory. Please try again or contact support.")


@router.post("/api/remember")
async def api_remember(request: Dict[str, Any], state=Depends(get_server_state)):
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
        from hololoom import HoloLoom

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
async def api_recall(request: Dict[str, Any], state=Depends(get_server_state)):
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
        from hololoom import HoloLoom

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


@router.get("/api/todos")
async def get_todos(
    workspace_id: Optional[str] = None,
    limit: int = 50,
    state=Depends(get_server_state)
):
    """
    Extract all TODOs from workspace with importance scoring.

    Queries HoloLoom memory for TODO/FIXME/NOTE comments and scores them
    by importance (mention frequency, recency, context).

    Args:
        workspace_id: Optional workspace filter
        limit: Maximum number of TODOs to return (default: 50)

    Returns:
        List of TODOs sorted by importance score

    Example:
        GET /api/todos?limit=20

        Response:
        {
          "todos": [
            {
              "text": "Add rate limiting to auth endpoints",
              "type": "TODO",
              "priority": "HIGH",
              "importance_score": 0.92,
              "mention_count": 3,
              "locations": ["auth.ts:15", "middleware.ts:42"],
              "related_notes": ["Security review needed", "API rate limits"]
            },
            ...
          ],
          "total_count": 47
        }
    """
    try:
        from hololoom import HoloLoom

        # Create HoloLoom instance
        async with HoloLoom(config=state.config) as loom:
            # Query for TODO/FIXME/NOTE comments
            # Search broadly for common task markers
            all_results = []
            for keyword in ['TODO', 'FIXME', 'NOTE', 'HACK', 'XXX']:
                results = await loom.recall(keyword, k=100)
                all_results.extend(results)

            logger.info(f"Found {len(all_results)} potential TODO items")

            # Extract and group TODOs
            todo_groups = defaultdict(list)

            for memory in all_results:
                text = memory.text

                # Extract TODO/FIXME/NOTE lines
                todo_pattern = r'(TODO|FIXME|NOTE|HACK|XXX):\s*(.+?)(?:\n|$)'
                matches = re.finditer(todo_pattern, text, re.IGNORECASE)

                for match in matches:
                    todo_type = match.group(1).upper()
                    todo_text = match.group(2).strip()

                    # Get location from metadata
                    file_path = memory.metadata.get('file', 'unknown')

                    # Normalize TODO text for grouping (lowercase, strip punctuation)
                    normalized = todo_text.lower().strip('.,!?')

                    # Group similar TODOs
                    todo_groups[normalized].append({
                        'type': todo_type,
                        'text': todo_text,
                        'location': file_path,
                        'confidence': memory.metadata.get('confidence', 0.5),
                        'timestamp': memory.metadata.get('timestamp', 'unknown')
                    })

            # Score and rank TODOs
            scored_todos = []

            for normalized_text, occurrences in todo_groups.items():
                # Calculate importance score
                mention_count = len(occurrences)
                avg_confidence = sum(o['confidence'] for o in occurrences) / mention_count

                # Importance factors:
                # 1. Mention count (more mentions = more important)
                # 2. Confidence (higher confidence = more important)
                # 3. TODO type (FIXME > TODO > NOTE)
                type_weights = {
                    'FIXME': 1.0,
                    'TODO': 0.8,
                    'HACK': 0.7,
                    'XXX': 0.7,
                    'NOTE': 0.5
                }

                todo_type = occurrences[0]['type']
                type_weight = type_weights.get(todo_type, 0.5)

                # Importance score formula
                importance_score = (
                    (mention_count / 10) * 0.4 +  # Mention frequency (normalized)
                    avg_confidence * 0.3 +          # Average confidence
                    type_weight * 0.3               # Type weight
                )
                importance_score = min(1.0, importance_score)  # Cap at 1.0

                # Determine priority
                if importance_score >= 0.75:
                    priority = 'HIGH'
                elif importance_score >= 0.5:
                    priority = 'MEDIUM'
                else:
                    priority = 'LOW'

                # Get unique locations
                locations = list(set(o['location'] for o in occurrences))

                scored_todos.append({
                    'text': occurrences[0]['text'],  # Use original text (not normalized)
                    'type': todo_type,
                    'priority': priority,
                    'importance_score': round(importance_score, 2),
                    'mention_count': mention_count,
                    'locations': locations[:5],  # Limit to 5 locations
                    'related_notes': []  # Could add related memories here
                })

            # Sort by importance score (descending)
            scored_todos.sort(key=lambda x: x['importance_score'], reverse=True)

            # Limit results
            scored_todos = scored_todos[:limit]

            return {
                'todos': scored_todos,
                'total_count': len(todo_groups)
            }

    except Exception as e:
        logger.error(f"Failed to extract TODOs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
