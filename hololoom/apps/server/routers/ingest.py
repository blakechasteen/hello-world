"""
Ingestion Router
=================
Endpoints for workspace and file ingestion into HoloLoom knowledge graph.

Extracted from agentic_api.py (March 2026 Refactor).
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from hololoom.agentic.ml_logic_detector import Language as CodeLanguage
from hololoom.apps.server.server_state import state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion"])


@router.post("/ingest/workspace")
async def ingest_workspace(
    workspace_path: str,
    languages: list[str] | None = None,
    exclude_patterns: list[str] | None = None
):
    """Ingest entire workspace into knowledge graph."""
    try:
        from hololoom import HoloLoom
        from hololoom.spinningWheel.workspace import WorkspaceSpinner

        # SECURITY: Path traversal validation
        allowed_bases_str = os.environ.get("ALLOWED_WORKSPACE_PATHS", "")
        if allowed_bases_str:
            allowed_bases = [Path(p.strip()).resolve() for p in allowed_bases_str.split(",") if p.strip()]
        else:
            allowed_bases = [Path.cwd().resolve(), Path.home().resolve()]

        requested_path = Path(workspace_path).resolve()

        path_allowed = False
        for base in allowed_bases:
            try:
                requested_path.relative_to(base)
                path_allowed = True
                break
            except ValueError:
                continue

        if not path_allowed:
            logger.warning(f"SECURITY: Path traversal attempt blocked: {workspace_path}")
            raise HTTPException(
                status_code=403,
                detail="Path not allowed. Set ALLOWED_WORKSPACE_PATHS environment variable."
            )

        path_str = str(requested_path)
        suspicious_patterns = ["/etc/", "/var/", "/root/", "\\Windows\\", "\\System32\\"]
        for pattern in suspicious_patterns:
            if pattern.lower() in path_str.lower():
                logger.warning(f"SECURITY: Suspicious path blocked: {workspace_path}")
                raise HTTPException(status_code=403, detail="Access to system directories not allowed")

        logger.info(f"Starting workspace ingestion: {requested_path}")

        spinner = WorkspaceSpinner()
        shards = await spinner.spin_workspace(
            workspace_path=str(requested_path),
            languages=languages,
            exclude_patterns=exclude_patterns
        )

        logger.info(f"Created {len(shards)} shards from workspace")

        async with HoloLoom(config=state.config) as loom:
            for shard in shards:
                await loom.experience(content=shard.text, context=shard.metadata)

        state.shards.extend(shards)

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
            "workspace_path": str(requested_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workspace ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Workspace ingestion failed")


@router.post("/ingest/workspace/legacy")
async def ingest_workspace_legacy(
    workspace_path: str,
    languages: list[str] | None = None,
    exclude_patterns: list[str] | None = None
):
    """Legacy workspace ingestion using codebase indexer (if available)."""
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

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

        code_shards = state.codebase_indexer.to_memory_shards()
        logger.info(f"Created {len(code_shards)} code memory shards")
        state.shards.extend(code_shards)

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
async def ingest_file(file_path: str, language: str, content: str | None = None):
    """Ingest single file into knowledge graph."""
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT
        }

        lang_enum = lang_map.get(language.lower())
        if not lang_enum:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}. Allowed: python, typescript, javascript")

        # SECURITY: Path traversal validation
        validated_path = None
        if not content:
            allowed_bases_str = os.environ.get("ALLOWED_WORKSPACE_PATHS", "")
            if allowed_bases_str:
                allowed_bases = [Path(p.strip()).resolve() for p in allowed_bases_str.split(",") if p.strip()]
            else:
                allowed_bases = [Path.cwd().resolve(), Path.home().resolve()]

            requested_path = Path(file_path).resolve()

            path_allowed = False
            for base in allowed_bases:
                try:
                    requested_path.relative_to(base)
                    path_allowed = True
                    break
                except ValueError:
                    continue

            if not path_allowed:
                logger.warning(f"SECURITY: Path traversal attempt blocked: {file_path}")
                raise HTTPException(
                    status_code=403,
                    detail="Path not allowed. Set ALLOWED_WORKSPACE_PATHS environment variable."
                )

            path_str = str(requested_path)
            suspicious_patterns = ["/etc/", "/var/", "/root/", "\\Windows\\", "\\System32\\", ".ssh", ".aws", ".env"]
            for pattern in suspicious_patterns:
                if pattern.lower() in path_str.lower():
                    logger.warning(f"SECURITY: Suspicious path blocked: {file_path}")
                    raise HTTPException(status_code=403, detail="Access to system/sensitive directories not allowed")

            validated_path = requested_path

        if content:
            import tempfile
            suffix_map = {"python": ".py", "typescript": ".ts", "javascript": ".js"}
            suffix = suffix_map.get(language.lower(), ".txt")

            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                f.write(content)
                temp_path = f.name

            try:
                code_file = await state.codebase_indexer.ingest_file(temp_path, lang_enum)
            finally:
                os.unlink(temp_path)
        else:
            code_file = await state.codebase_indexer.ingest_file(str(validated_path), lang_enum)

        return {
            "success": True,
            "file_path": str(validated_path) if validated_path else file_path,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="File ingestion failed")
