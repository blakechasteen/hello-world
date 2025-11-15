"""Tools module: Elle's capabilities and services."""

from .protocol import Tool, BaseTool, ToolRegistry
from .vision_tool import VisionTool, VisionStub
from .layout_tool import LayoutTool, LayoutStub
from .task_tools import TaskPlannerTool, SchedulerTool

__all__ = [
    'Tool',
    'BaseTool',
    'ToolRegistry',
    'VisionTool',
    'VisionStub',
    'LayoutTool',
    'LayoutStub',
    'TaskPlannerTool',
    'SchedulerTool',
]


def create_default_registry() -> ToolRegistry:
    """Create registry with all standard tools."""
    
    registry = ToolRegistry()
    
    # Register tools
    registry.register(VisionStub())
    registry.register(LayoutStub())
    registry.register(TaskPlannerTool())
    registry.register(SchedulerTool())
    
    return registry
