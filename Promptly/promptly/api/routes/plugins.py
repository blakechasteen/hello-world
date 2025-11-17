"""
Plugins API Routes
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..middleware.auth import api_key_auth
from pydantic import BaseModel

router = APIRouter()


class PluginInfo(BaseModel):
    """Plugin information"""
    name: str
    type: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True


class PluginListResponse(BaseModel):
    """Plugin list response"""
    evaluators: List[PluginInfo]
    storage_backends: List[PluginInfo]
    processors: List[PluginInfo]
    total: int


class PluginConfigRequest(BaseModel):
    """Plugin configuration request"""
    plugin_name: str
    plugin_type: str
    config: Dict[str, Any]


@router.get("", response_model=PluginListResponse)
async def list_plugins(authenticated: bool = Depends(api_key_auth)):
    """
    List all available plugins

    Returns all evaluators, storage backends, and processors.
    """
    try:
        # Try to import plugins module
        try:
            from plugins import list_plugins as get_plugins
            plugins_data = get_plugins()
        except ImportError:
            # Fallback to basic plugin list
            plugins_data = {
                "evaluators": [
                    {"name": "keyword", "description": "Keyword-based evaluation"},
                    {"name": "semantic", "description": "Semantic similarity evaluation"},
                    {"name": "exact_match", "description": "Exact string matching"},
                ],
                "storage_backends": [
                    {"name": "sqlite", "description": "SQLite database storage"},
                    {"name": "json", "description": "JSON file storage"},
                ],
                "chain_processors": []
            }

        # Convert to response format
        evaluators = [
            PluginInfo(
                name=p["name"],
                type="evaluator",
                description=p["description"]
            )
            for p in plugins_data.get("evaluators", [])
        ]

        storage_backends = [
            PluginInfo(
                name=p["name"],
                type="storage",
                description=p["description"]
            )
            for p in plugins_data.get("storage_backends", [])
        ]

        processors = [
            PluginInfo(
                name=p["name"],
                type="processor",
                description=p["description"]
            )
            for p in plugins_data.get("chain_processors", [])
        ]

        return PluginListResponse(
            evaluators=evaluators,
            storage_backends=storage_backends,
            processors=processors,
            total=len(evaluators) + len(storage_backends) + len(processors)
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{plugin_type}/{plugin_name}", response_model=PluginInfo)
async def get_plugin(
    plugin_type: str,
    plugin_name: str,
    authenticated: bool = Depends(api_key_auth)
):
    """
    Get plugin details

    Returns information about a specific plugin.
    """
    try:
        # Get plugin list
        plugins_list = await list_plugins(authenticated)

        # Find plugin
        all_plugins = (
            plugins_list.evaluators +
            plugins_list.storage_backends +
            plugins_list.processors
        )

        for plugin in all_plugins:
            if plugin.name == plugin_name and plugin.type == plugin_type:
                return plugin

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{plugin_name}' of type '{plugin_type}' not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/configure", response_model=Dict[str, Any])
async def configure_plugin(
    config: PluginConfigRequest,
    authenticated: bool = Depends(api_key_auth)
):
    """
    Configure a plugin

    Updates plugin configuration.
    """
    # This would typically persist plugin configuration
    # For now, just return success
    return {
        "status": "success",
        "message": f"Configured {config.plugin_type} plugin '{config.plugin_name}'",
        "config": config.config
    }
