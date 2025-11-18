"""Base plugin class"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel


class HookContext(BaseModel):
    """Context passed to hook handlers"""
    user_id: Optional[str] = None
    institution_id: Optional[str] = None
    data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class HookResponse(BaseModel):
    """Response from hook handler"""
    success: bool = True
    data: Dict[str, Any] = {}
    message: Optional[str] = None


class PluginBase(ABC):
    """Base class for all plugins"""

    def __init__(self, plugin_id: str, config: Dict[str, Any] = None):
        self.plugin_id = plugin_id
        self.config = config or {}
        self._hooks: Dict[str, callable] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize plugin (load resources, connect to services, etc.)"""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Cleanup plugin resources"""
        pass

    def register_hook(self, hook_name: str, handler: callable):
        """Register a hook handler"""
        self._hooks[hook_name] = handler

    async def execute_hook(self, hook_name: str, context: HookContext) -> HookResponse:
        """Execute a registered hook"""
        if hook_name not in self._hooks:
            return HookResponse(success=False, message=f"Hook '{hook_name}' not registered")

        try:
            handler = self._hooks[hook_name]
            result = await handler(context)

            if isinstance(result, HookResponse):
                return result
            elif isinstance(result, dict):
                return HookResponse(success=True, data=result)
            else:
                return HookResponse(success=True, data={"result": result})

        except Exception as e:
            return HookResponse(success=False, message=str(e))

    def get_hooks(self) -> List[str]:
        """Get list of registered hooks"""
        return list(self._hooks.keys())

    @property
    def name(self) -> str:
        """Plugin name"""
        return self.config.get("name", self.plugin_id)

    @property
    def version(self) -> str:
        """Plugin version"""
        return self.config.get("version", "1.0.0")

    @property
    def category(self) -> str:
        """Plugin category"""
        return self.config.get("category", "unknown")
