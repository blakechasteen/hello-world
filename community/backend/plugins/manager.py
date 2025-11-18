"""
Plugin manager for loading, managing, and executing plugins
"""
from typing import Dict, List, Optional, Any
from uuid import UUID
import importlib
import logging

from plugins.base import PluginBase, HookType, PluginContext, PluginType

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages plugin lifecycle and hook execution
    """

    def __init__(self):
        """Initialize plugin manager"""
        self._plugins: Dict[str, PluginBase] = {}
        self._hooks: Dict[HookType, List[PluginBase]] = {}
        self._plugin_order: Dict[HookType, List[str]] = {}

    async def load_plugin(
        self,
        plugin_id: str,
        plugin_module: str,
        plugin_class: str,
        config: Dict[str, Any],
    ) -> bool:
        """
        Load and initialize a plugin

        Args:
            plugin_id: Unique plugin identifier
            plugin_module: Python module path (e.g., "plugins.content.polls")
            plugin_class: Plugin class name
            config: Plugin configuration

        Returns:
            bool: True if loaded successfully
        """
        try:
            # Import plugin module
            module = importlib.import_module(plugin_module)

            # Get plugin class
            PluginClass = getattr(module, plugin_class)

            # Instantiate plugin
            plugin = PluginClass(config)

            # Initialize plugin
            if not await plugin.initialize():
                logger.error(f"Failed to initialize plugin: {plugin_id}")
                return False

            # Store plugin
            self._plugins[plugin_id] = plugin

            # Register plugin hooks
            for hook_type in HookType:
                if hook_type in plugin._hooks and plugin._hooks[hook_type]:
                    if hook_type not in self._hooks:
                        self._hooks[hook_type] = []
                        self._plugin_order[hook_type] = []

                    self._hooks[hook_type].append(plugin)
                    self._plugin_order[hook_type].append(plugin_id)

            logger.info(
                f"Loaded plugin: {plugin.name} v{plugin.version} ({plugin_id})"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_id}: {e}")
            return False

    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload and shutdown a plugin

        Args:
            plugin_id: Plugin identifier

        Returns:
            bool: True if unloaded successfully
        """
        if plugin_id not in self._plugins:
            logger.warning(f"Plugin not found: {plugin_id}")
            return False

        try:
            plugin = self._plugins[plugin_id]

            # Shutdown plugin
            await plugin.shutdown()

            # Remove from hooks
            for hook_type in HookType:
                if hook_type in self._hooks:
                    self._hooks[hook_type] = [
                        p for p in self._hooks[hook_type] if p != plugin
                    ]
                    if plugin_id in self._plugin_order.get(hook_type, []):
                        self._plugin_order[hook_type].remove(plugin_id)

            # Remove plugin
            del self._plugins[plugin_id]

            logger.info(f"Unloaded plugin: {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False

    async def trigger_hook(
        self,
        hook_type: HookType,
        context: Optional[PluginContext] = None,
        **kwargs,
    ) -> PluginContext:
        """
        Trigger a hook and execute all registered handlers

        Args:
            hook_type: Type of hook to trigger
            context: Plugin context (created if None)
            **kwargs: Additional context data

        Returns:
            PluginContext: Modified context after all handlers
        """
        if context is None:
            context = PluginContext(data=kwargs)
        else:
            # Merge kwargs into context
            for key, value in kwargs.items():
                context.set(key, value)

        # Get plugins for this hook
        plugins = self._hooks.get(hook_type, [])

        if not plugins:
            return context

        logger.debug(f"Triggering hook {hook_type} with {len(plugins)} plugins")

        # Execute plugins in order
        for plugin in plugins:
            if not plugin.enabled:
                continue

            try:
                # Execute plugin's hook handlers
                context = await plugin.execute_hook(hook_type, context)

            except Exception as e:
                logger.error(
                    f"Error executing hook {hook_type} in plugin {plugin.name}: {e}"
                )
                # Continue with other plugins

        return context

    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """
        Get a loaded plugin by ID

        Args:
            plugin_id: Plugin identifier

        Returns:
            PluginBase or None if not found
        """
        return self._plugins.get(plugin_id)

    def list_plugins(
        self, plugin_type: Optional[PluginType] = None
    ) -> List[Dict[str, Any]]:
        """
        List all loaded plugins

        Args:
            plugin_type: Filter by plugin type (optional)

        Returns:
            List of plugin info dictionaries
        """
        plugins = []

        for plugin_id, plugin in self._plugins.items():
            if plugin_type and plugin.plugin_type != plugin_type:
                continue

            plugins.append(
                {
                    "id": plugin_id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "type": plugin.plugin_type.value,
                    "enabled": plugin.enabled,
                }
            )

        return plugins

    def enable_plugin(self, plugin_id: str):
        """Enable a plugin"""
        if plugin_id in self._plugins:
            self._plugins[plugin_id].enabled = True
            logger.info(f"Enabled plugin: {plugin_id}")

    def disable_plugin(self, plugin_id: str):
        """Disable a plugin"""
        if plugin_id in self._plugins:
            self._plugins[plugin_id].enabled = False
            logger.info(f"Disabled plugin: {plugin_id}")

    async def shutdown_all(self):
        """Shutdown all plugins"""
        logger.info("Shutting down all plugins")

        for plugin_id in list(self._plugins.keys()):
            await self.unload_plugin(plugin_id)


# Global plugin manager instance
plugin_manager = PluginManager()
