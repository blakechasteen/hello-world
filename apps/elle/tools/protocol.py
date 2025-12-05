"""Tool protocol: interface for Elle's capabilities."""

from __future__ import annotations
from typing import Protocol, Dict, Any, Optional, List, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..domain import Followup


class Tool(Protocol):
    """
    Interface for tools that Elle can use.
    
    Tools are capabilities that extend Elle's perception and action.
    They don't make decisions—they provide information or execute tasks.
    """
    
    name: str
    description: str
    
    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with given input.
        
        Args:
            input: Tool-specific parameters
        
        Returns:
            Tool-specific results
        """
        ...


class BaseTool(ABC):
    """Base class for implementing tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Implement tool logic."""
        pass
    
    def validate_input(self, input: Dict[str, Any], required: List[str]) -> None:
        """Validate that required input fields are present."""
        missing = [field for field in required if field not in input]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")


class ToolRegistry:
    """
    Registry for discovering and dispatching tools.
    
    Central place to register all available tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def dispatch(self, followup: 'Followup') -> Dict[str, Any]:
        """
        Dispatch a followup action to appropriate tool.
        
        Args:
            followup: Followup from ElleAction
        
        Returns:
            Tool execution result
        """
        
        # Map service names to tools
        # For now, simple naming convention
        tool_name = followup.service
        
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Run the tool
        result = tool.run({
            'action': followup.action,
            **followup.params
        })
        
        return result
