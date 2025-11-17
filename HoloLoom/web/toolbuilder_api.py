"""
Tool Builder REST API

FastAPI endpoints for the custom tool builder interface.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Depends, Body, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not installed. Tool Builder API unavailable.")
    FASTAPI_AVAILABLE = False
    APIRouter = None
    HTTPException = None
    BaseModel = object

# Import tool builder components
try:
    from ..toolbuilder import (
        ToolDefinition,
        ToolValidator,
        CodeGenerator,
        ToolRegistry,
        ToolExecutor,
        ParameterType,
        LogicBlockType,
        ActionType,
        OutputFormat,
    )
    from ..toolbuilder.definition import (
        Parameter,
        LogicBlock,
        Condition,
        HTTPRequestAction,
        DatabaseQueryAction,
        FileOperationAction,
        ComputeAction,
        CallToolAction,
        MemoryOperationAction,
    )
    TOOLBUILDER_AVAILABLE = True
except ImportError:
    logger.warning("Tool Builder components not available")
    TOOLBUILDER_AVAILABLE = False


# Pydantic models for API
class ToolCreateRequest(BaseModel):
    """Request to create a new tool"""
    definition: Dict[str, Any]


class ToolUpdateRequest(BaseModel):
    """Request to update a tool"""
    definition: Dict[str, Any]


class ToolTestRequest(BaseModel):
    """Request to test a tool"""
    parameters: Dict[str, Any]
    timeout: Optional[int] = None


class ToolListResponse(BaseModel):
    """Response for tool list"""
    tools: List[Dict[str, Any]]
    total: int
    categories: List[str]
    tags: List[str]


class ToolStatsResponse(BaseModel):
    """Response for registry statistics"""
    registry_stats: Dict[str, Any]
    execution_stats: Dict[str, Any]


# Global instances (initialized when API is created)
registry: Optional[ToolRegistry] = None
executor: Optional[ToolExecutor] = None
validator: Optional[ToolValidator] = None
generator: Optional[CodeGenerator] = None


def create_toolbuilder_router(storage_path: str = "./tools") -> APIRouter:
    """
    Create Tool Builder API router

    Args:
        storage_path: Path to store tool definitions

    Returns:
        FastAPI router
    """
    if not FASTAPI_AVAILABLE or not TOOLBUILDER_AVAILABLE:
        raise RuntimeError("Tool Builder API dependencies not available")

    # Initialize global instances
    global registry, executor, validator, generator
    registry = ToolRegistry(storage_path=storage_path)
    executor = ToolExecutor(registry=registry)
    validator = ToolValidator()
    generator = CodeGenerator()

    router = APIRouter(prefix="/api/toolbuilder", tags=["Tool Builder"])

    # ==================== Tool CRUD Endpoints ====================

    @router.post("/tools")
    async def create_tool(request: ToolCreateRequest) -> JSONResponse:
        """Create a new custom tool"""
        try:
            # Parse tool definition
            tool_def = ToolDefinition.from_dict(request.definition)

            # Validate
            validator.validate(tool_def)

            # Generate code
            code = generator.generate(tool_def)

            # Register
            tool_id = registry.register(tool_def)

            # Save generated code
            registry.save_generated_code(tool_id, code)

            return JSONResponse({
                "status": "success",
                "tool_id": tool_id,
                "message": f"Tool '{tool_def.name}' created successfully",
                "warnings": validator.get_warnings(),
            })

        except Exception as e:
            logger.error(f"Error creating tool: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/tools")
    async def list_tools(
        category: Optional[str] = Query(None),
        tag: Optional[str] = Query(None),
        enabled_only: bool = Query(True),
        search: Optional[str] = Query(None),
    ) -> ToolListResponse:
        """List all tools with optional filtering"""
        try:
            if search:
                tools = registry.search(search)
            else:
                tools = registry.list_all(
                    category=category,
                    tag=tag,
                    enabled_only=enabled_only
                )

            return ToolListResponse(
                tools=[t.to_dict() for t in tools],
                total=len(tools),
                categories=registry.get_categories(),
                tags=registry.get_tags(),
            )

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/tools/{tool_id}")
    async def get_tool(tool_id: str) -> JSONResponse:
        """Get a tool definition by ID"""
        try:
            tool = registry.get(tool_id)
            if not tool:
                raise HTTPException(status_code=404, detail="Tool not found")

            # Get generated code
            code = registry.get_generated_code(tool_id)

            return JSONResponse({
                "status": "success",
                "tool": tool.to_dict(),
                "code": code,
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting tool: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/tools/{tool_id}")
    async def update_tool(tool_id: str, request: ToolUpdateRequest) -> JSONResponse:
        """Update a tool definition"""
        try:
            # Check if tool exists
            existing = registry.get(tool_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Tool not found")

            # Save current version
            registry.save_version(tool_id)

            # Parse updated definition
            tool_def = ToolDefinition.from_dict(request.definition)

            # Ensure ID matches
            if tool_def.id != tool_id:
                raise HTTPException(
                    status_code=400,
                    detail="Tool ID in definition must match URL parameter"
                )

            # Validate
            validator.validate(tool_def)

            # Generate new code
            code = generator.generate(tool_def)

            # Update registry
            registry.register(tool_def, overwrite=True)
            registry.save_generated_code(tool_id, code)

            return JSONResponse({
                "status": "success",
                "message": f"Tool '{tool_def.name}' updated successfully",
                "warnings": validator.get_warnings(),
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating tool: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/tools/{tool_id}")
    async def delete_tool(
        tool_id: str,
        keep_versions: bool = Query(True)
    ) -> JSONResponse:
        """Delete a tool"""
        try:
            success = registry.delete(tool_id, keep_versions=keep_versions)
            if not success:
                raise HTTPException(status_code=404, detail="Tool not found")

            return JSONResponse({
                "status": "success",
                "message": f"Tool {tool_id} deleted successfully",
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting tool: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Tool Management Endpoints ====================

    @router.post("/tools/{tool_id}/enable")
    async def enable_tool(tool_id: str) -> JSONResponse:
        """Enable a tool"""
        try:
            success = registry.enable(tool_id)
            if not success:
                raise HTTPException(status_code=404, detail="Tool not found")

            return JSONResponse({
                "status": "success",
                "message": f"Tool {tool_id} enabled",
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error enabling tool: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/tools/{tool_id}/disable")
    async def disable_tool(tool_id: str) -> JSONResponse:
        """Disable a tool"""
        try:
            success = registry.disable(tool_id)
            if not success:
                raise HTTPException(status_code=404, detail="Tool not found")

            return JSONResponse({
                "status": "success",
                "message": f"Tool {tool_id} disabled",
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error disabling tool: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Version Management ====================

    @router.get("/tools/{tool_id}/versions")
    async def get_versions(tool_id: str) -> JSONResponse:
        """Get all versions of a tool"""
        try:
            versions = registry.get_versions(tool_id)
            return JSONResponse({
                "status": "success",
                "tool_id": tool_id,
                "versions": versions,
                "total": len(versions),
            })

        except Exception as e:
            logger.error(f"Error getting versions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/tools/{tool_id}/restore/{version_id}")
    async def restore_version(tool_id: str, version_id: str) -> JSONResponse:
        """Restore a tool from a version"""
        try:
            tool_def = registry.restore_version(tool_id, version_id)

            # Regenerate code
            code = generator.generate(tool_def)
            registry.save_generated_code(tool_id, code)

            return JSONResponse({
                "status": "success",
                "message": f"Tool {tool_id} restored to version {version_id}",
                "tool": tool_def.to_dict(),
            })

        except Exception as e:
            logger.error(f"Error restoring version: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== Code Generation ====================

    @router.get("/tools/{tool_id}/code")
    async def get_generated_code(tool_id: str) -> JSONResponse:
        """Get generated Python code for a tool"""
        try:
            code = registry.get_generated_code(tool_id)
            if not code:
                # Generate if not exists
                tool = registry.get(tool_id)
                if not tool:
                    raise HTTPException(status_code=404, detail="Tool not found")

                code = generator.generate(tool)
                registry.save_generated_code(tool_id, code)

            return JSONResponse({
                "status": "success",
                "tool_id": tool_id,
                "code": code,
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting code: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/tools/{tool_id}/regenerate")
    async def regenerate_code(tool_id: str) -> JSONResponse:
        """Regenerate Python code for a tool"""
        try:
            tool = registry.get(tool_id)
            if not tool:
                raise HTTPException(status_code=404, detail="Tool not found")

            # Validate first
            validator.validate(tool)

            # Generate new code
            code = generator.generate(tool)
            registry.save_generated_code(tool_id, code)

            return JSONResponse({
                "status": "success",
                "message": "Code regenerated successfully",
                "code": code,
                "warnings": validator.get_warnings(),
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error regenerating code: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== Tool Execution ====================

    @router.post("/tools/{tool_id}/test")
    async def test_tool(tool_id: str, request: ToolTestRequest) -> JSONResponse:
        """Test execute a tool"""
        try:
            result = await executor.execute(
                tool_id=tool_id,
                parameters=request.parameters,
                timeout_override=request.timeout,
            )

            return JSONResponse({
                "status": "success",
                "result": result,
            })

        except Exception as e:
            logger.error(f"Error testing tool: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e),
            }, status_code=400)

    @router.post("/tools/{tool_id}/execute")
    async def execute_tool(tool_id: str, parameters: Dict[str, Any] = Body(...)) -> JSONResponse:
        """Execute a tool (production endpoint)"""
        try:
            result = await executor.execute(
                tool_id=tool_id,
                parameters=parameters,
            )

            return JSONResponse({
                "status": "success",
                "result": result,
            })

        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ==================== Templates ====================

    @router.get("/templates")
    async def get_templates() -> JSONResponse:
        """Get pre-built tool templates"""
        templates = _get_builtin_templates()

        return JSONResponse({
            "status": "success",
            "templates": templates,
            "total": len(templates),
        })

    @router.get("/templates/{template_name}")
    async def get_template(template_name: str) -> JSONResponse:
        """Get a specific template"""
        templates = _get_builtin_templates()
        template = next((t for t in templates if t["id"] == template_name), None)

        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        return JSONResponse({
            "status": "success",
            "template": template,
        })

    # ==================== Statistics ====================

    @router.get("/stats")
    async def get_stats() -> ToolStatsResponse:
        """Get registry and execution statistics"""
        return ToolStatsResponse(
            registry_stats=registry.get_stats(),
            execution_stats=executor.get_execution_stats(),
        )

    @router.get("/executions/recent")
    async def get_recent_executions(limit: int = Query(10, le=100)) -> JSONResponse:
        """Get recent execution records"""
        executions = executor.get_recent_executions(limit=limit)

        return JSONResponse({
            "status": "success",
            "executions": executions,
            "total": len(executions),
        })

    # ==================== Validation ====================

    @router.post("/validate")
    async def validate_tool(definition: Dict[str, Any] = Body(...)) -> JSONResponse:
        """Validate a tool definition without saving"""
        try:
            tool_def = ToolDefinition.from_dict(definition)
            validator.validate(tool_def)

            return JSONResponse({
                "status": "success",
                "valid": True,
                "warnings": validator.get_warnings(),
            })

        except Exception as e:
            return JSONResponse({
                "status": "error",
                "valid": False,
                "errors": [str(e)],
                "warnings": validator.get_warnings(),
            }, status_code=400)

    return router


def _get_builtin_templates() -> List[Dict[str, Any]]:
    """Get built-in tool templates"""
    from .templates import (
        get_calculator_template,
        get_web_scraper_template,
        get_data_transformer_template,
        get_api_client_template,
        get_file_converter_template,
    )

    return [
        get_calculator_template(),
        get_web_scraper_template(),
        get_data_transformer_template(),
        get_api_client_template(),
        get_file_converter_template(),
    ]
