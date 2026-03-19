"""
Codebase Router
================
Endpoints for codebase stats, search, and code verification.

Extracted from agentic_api.py (March 2026 Refactor).
"""

import logging

from fastapi import APIRouter, HTTPException

from hololoom.apps.server.server_state import state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["codebase"])


@router.get("/codebase/stats")
async def get_codebase_stats():
    """Get statistics about indexed codebase."""
    if not state.codebase_indexer:
        return {"error": "Codebase indexer not initialized"}
    return state.codebase_indexer.get_statistics()


@router.post("/verify/code")
async def verify_code(
    code: str,
    language: str,
    file_path: str = "temp",
    check_syntax: bool = True,
    check_types: bool = True,
    check_lint: bool = False
):
    """Verify code for syntax/type/lint errors."""
    try:
        if not state.code_verifier:
            raise HTTPException(status_code=500, detail="Code verifier not initialized")

        result = await state.code_verifier.verify_comprehensive(
            code, language, file_path
        )
        return {"success": True, "verification": result}

    except Exception as e:
        logger.error(f"Code verification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/codebase/search")
async def search_codebase(
    query: str,
    entity_type: str | None = None,
    fuzzy: bool = True,
    limit: int = 10
):
    """Search indexed codebase for entities."""
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

        type_map = {
            "function": "function",
            "class": "class",
            "method": "method",
            "variable": "variable",
            "import": "import",
            "module": "module"
        }

        entity_type_enum = None
        if entity_type:
            from hololoom.agentic.codebase_ingestion import EntityType
            mapped_type = type_map.get(entity_type.lower())
            if mapped_type:
                entity_type_enum = EntityType(mapped_type)

        results = await state.codebase_indexer.search_entity(
            query, entity_type=entity_type_enum, fuzzy=fuzzy
        )
        results = results[:limit]

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Codebase search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
