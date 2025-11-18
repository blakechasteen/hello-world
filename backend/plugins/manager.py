"""Plugin manager"""
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional
from sqlalchemy import select

from backend.plugins.base import PluginBase, HookContext, HookResponse


class PluginManager:
    """Manages plugin loading, initialization, and execution"""

    def __init__(self):
        self._plugins: Dict[str, PluginBase] = {}
        self._hook_registry: Dict[str, List[str]] = {}  # hook_name -> [plugin_ids]

    async def load_plugin(self, plugin_id: str, plugin_path: str, config: Dict = None) -> bool:
        """Load a plugin from a Python module"""
        try:
            # Add plugin directory to path if needed
            plugin_dir = Path(plugin_path).parent
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))

            # Import plugin module
            module_name = Path(plugin_path).stem
            module = importlib.import_module(module_name)

            # Find PluginBase subclass
            plugin_class = None
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type) and issubclass(item, PluginBase) and item is not PluginBase:
                    plugin_class = item
                    break

            if not plugin_class:
                print(f"❌ No PluginBase subclass found in {plugin_path}")
                return False

            # Instantiate plugin
            plugin = plugin_class(plugin_id=plugin_id, config=config)

            # Initialize
            success = await plugin.initialize()
            if not success:
                print(f"❌ Failed to initialize plugin {plugin_id}")
                return False

            # Store plugin
            self._plugins[plugin_id] = plugin

            # Register hooks
            for hook_name in plugin.get_hooks():
                if hook_name not in self._hook_registry:
                    self._hook_registry[hook_name] = []
                self._hook_registry[hook_name].append(plugin_id)

            print(f"✅ Loaded plugin: {plugin_id} ({plugin.name} v{plugin.version})")
            return True

        except Exception as e:
            print(f"❌ Error loading plugin {plugin_id}: {str(e)}")
            return False

    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload and cleanup a plugin"""
        if plugin_id not in self._plugins:
            return False

        plugin = self._plugins[plugin_id]

        # Shutdown plugin
        await plugin.shutdown()

        # Remove from hook registry
        for hook_name, plugin_ids in self._hook_registry.items():
            if plugin_id in plugin_ids:
                plugin_ids.remove(plugin_id)

        # Remove plugin
        del self._plugins[plugin_id]

        print(f"✅ Unloaded plugin: {plugin_id}")
        return True

    async def trigger_hook(
        self,
        hook_name: str,
        context: HookContext,
        plugin_id: Optional[str] = None
    ) -> List[HookResponse]:
        """Trigger a hook on one or all plugins"""
        responses = []

        # If specific plugin requested
        if plugin_id:
            if plugin_id in self._plugins:
                plugin = self._plugins[plugin_id]
                response = await plugin.execute_hook(hook_name, context)
                responses.append(response)
            return responses

        # Trigger on all plugins that registered this hook
        if hook_name in self._hook_registry:
            for pid in self._hook_registry[hook_name]:
                if pid in self._plugins:
                    plugin = self._plugins[pid]
                    response = await plugin.execute_hook(hook_name, context)
                    responses.append(response)

        return responses

    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """Get a loaded plugin by ID"""
        return self._plugins.get(plugin_id)

    def list_plugins(self) -> List[Dict]:
        """List all loaded plugins"""
        return [
            {
                "plugin_id": pid,
                "name": plugin.name,
                "version": plugin.version,
                "category": plugin.category,
                "hooks": plugin.get_hooks(),
            }
            for pid, plugin in self._plugins.items()
        ]

    def list_hooks(self) -> Dict[str, List[str]]:
        """List all registered hooks and their plugins"""
        return dict(self._hook_registry)

    async def shutdown_all(self):
        """Shutdown all plugins"""
        for plugin_id in list(self._plugins.keys()):
            await self.unload_plugin(plugin_id)


# Global plugin manager instance
plugin_manager = PluginManager()
