"""
Workspace Routes
================

Endpoints for workspace ingestion and TODO extraction.

Endpoints:
    GET /api/todos - Extract TODOs from workspace
    POST /ingest/workspace - Ingest workspace into knowledge graph
    POST /ingest/workspace/legacy - Legacy workspace ingestion
    POST /ingest/file - Ingest single file
"""

import logging
import re
from collections import defaultdict
from typing import Optional, List

from fastapi import APIRouter, HTTPException

from ..state import state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workspace"])


@router.get("/api/todos")
async def get_todos(workspace_id: Optional[str] = None, limit: int = 50):
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


@router.post("/ingest/workspace")
async def ingest_workspace(
    workspace_path: str,
    languages: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
):
    """
    Ingest entire workspace into knowledge graph.

    Args:
        workspace_path: Path to workspace root
        languages: Languages to index (e.g., ["python", "typescript"])
        exclude_patterns: Glob patterns to exclude

    Returns:
        Statistics about ingestion

    Example:
        POST /ingest/workspace
        {
          "workspace_path": "/path/to/project",
          "languages": ["python", "typescript"],
          "exclude_patterns": ["**/node_modules/**", "**/.venv/**"]
        }
    """
    try:
        from hololoom.spinningWheel.workspace import WorkspaceSpinner
        from hololoom import HoloLoom

        logger.info(f"Starting workspace ingestion: {workspace_path}")

        # Create workspace spinner
        spinner = WorkspaceSpinner()

        # Scan workspace
        shards = await spinner.spin_workspace(
            workspace_path=workspace_path,
            languages=languages,
            exclude_patterns=exclude_patterns
        )

        logger.info(f"Created {len(shards)} shards from workspace")

        # Store shards in HoloLoom
        async with HoloLoom(config=state.config) as loom:
            for shard in shards:
                await loom.experience(
                    content=shard.text,
                    context=shard.metadata
                )

        # Also add to server's in-memory shards
        state.shards.extend(shards)

        # Calculate statistics
        total_files = len(shards)
        total_elements = sum(shard.metadata.get('element_count', 0) for shard in shards)
        total_comments = sum(shard.metadata.get('comment_count', 0) for shard in shards)
        total_todos = sum(shard.metadata.get('todo_count', 0) for shard in shards)

        logger.info(f"Workspace ingestion complete: {total_files} files, {total_elements} elements, {total_todos} TODOs")

        return {
            "success": True,
            "files_indexed": total_files,
            "code_elements": total_elements,
            "comments": total_comments,
            "todos": total_todos,
            "workspace_path": workspace_path
        }

    except Exception as e:
        logger.error(f"Workspace ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/workspace/legacy")
async def ingest_workspace_legacy(
    workspace_path: str,
    languages: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
):
    """
    Legacy workspace ingestion using codebase indexer (if available).

    This endpoint is kept for backward compatibility but the main
    /ingest/workspace endpoint now uses WorkspaceSpinner.
    """
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

        from hololoom.agentic.codebase_indexer import CodeLanguage

        # Convert language strings to Language enum
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT,
            "java": CodeLanguage.JAVA,
            "cpp": CodeLanguage.CPP,
            "rust": CodeLanguage.RUST,
            "go": CodeLanguage.GO
        }

        parsed_languages = None
        if languages:
            parsed_languages = [lang_map.get(lang.lower()) for lang in languages]
            parsed_languages = [l for l in parsed_languages if l is not None]

        logger.info(f"Starting workspace ingestion: {workspace_path}")
        stats = await state.codebase_indexer.ingest_workspace(
            workspace_path,
            languages=parsed_languages,
            exclude_patterns=exclude_patterns
        )

        # Convert indexed code to memory shards for HoloLoom
        code_shards = state.codebase_indexer.to_memory_shards()
        logger.info(f"Created {len(code_shards)} code memory shards")

        # Add to server's memory
        state.shards.extend(code_shards)

        # If persistent backend available, store shards
        if state.memory_backend:
            from hololoom.memory.protocol import Memory
            memories = []
            for shard in code_shards:
                memory = Memory(
                    id=shard.id,
                    text=shard.text,
                    context={
                        "episode": shard.episode,
                        "entities": shard.entities,
                        "motifs": shard.motifs
                    },
                    metadata=shard.metadata
                )
                memories.append(memory)

            await state.memory_backend.store(memories)
            logger.info(f"Stored {len(memories)} code memories to persistent backend")

        # Invalidate orchestrator so it gets recreated with new shards
        if state.orchestrator:
            await state.orchestrator.close()
            state.orchestrator = None

        return {
            "success": True,
            "ingestion_stats": stats,
            "memory_shards_created": len(code_shards),
            "message": f"Ingested {stats['files_processed']} files with {stats['entities_found']} entities"
        }

    except Exception as e:
        logger.error(f"Workspace ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file")
async def ingest_file(file_path: str, language: str, content: Optional[str] = None):
    """
    Ingest single file into knowledge graph.

    Args:
        file_path: Path to file
        language: Programming language
        content: File content (if None, reads from file_path)

    Returns:
        Parsed entities and statistics

    Example:
        POST /ingest/file
        {
          "file_path": "/path/to/file.py",
          "language": "python",
          "content": "def foo(): pass"
        }
    """
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

        from hololoom.agentic.codebase_indexer import CodeLanguage

        # Map language string to enum
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT
        }

        lang_enum = lang_map.get(language.lower())
        if not lang_enum:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        # If content provided, create temp file
        if content:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(content)
                temp_path = f.name

            code_file = await state.codebase_indexer.ingest_file(temp_path, lang_enum)

            os.unlink(temp_path)
        else:
            code_file = await state.codebase_indexer.ingest_file(file_path, lang_enum)

        return {
            "success": True,
            "file_path": file_path,
            "language": language,
            "entities_found": len(code_file.entities),
            "imports_found": len(code_file.imports),
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type.value,
                    "line": e.line_number,
                    "signature": e.signature
                }
                for e in code_file.entities
            ]
        }

    except Exception as e:
        logger.error(f"File ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
