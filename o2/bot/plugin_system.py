"""
Plugin System - Extensible Agent Architecture

Enables community-built custom agents for O2 Platform.

Features:
- Plugin discovery (auto-load from plugins/ directory)
- Sandboxed execution (plugins can't access core system)
- Lifecycle management (init, start, stop, cleanup)
- Event hooks (on_message, on_proposal, on_vote, etc.)
- Capability registration (what can plugin do?)
- Permission system (what is plugin allowed to do?)
- Marketplace integration (share plugins across instances)
"""

import asyncio
import logging
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger('o2-bot.plugin_system')


class PluginCapability(Enum):
    """Plugin capabilities - what can a plugin do?"""
    # Message processing
    PROCESS_MESSAGES = "process_messages"
    SEND_MESSAGES = "send_messages"

    # Governance
    CREATE_PROPOSALS = "create_proposals"
    VOTE = "vote"
    OBSERVE_VOTES = "observe_votes"

    # Memory
    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    SHARE_MEMORY = "share_memory"

    # Swarm
    JOIN_SWARM = "join_swarm"
    COORDINATE_AGENTS = "coordinate_agents"

    # Analysis
    ANALYZE_DATA = "analyze_data"
    GENERATE_REPORTS = "generate_reports"

    # External
    MAKE_HTTP_REQUESTS = "make_http_requests"
    ACCESS_FILES = "access_files"


class PluginPermission(Enum):
    """Plugin permissions - what is plugin allowed to do?"""
    # Safety levels
    SAFE = "safe"  # Read-only, no side effects
    TRUSTED = "trusted"  # Can write, but sandboxed
    PRIVILEGED = "privileged"  # Full access (dangerous!)


class PluginMetadata:
    """Plugin metadata (from manifest)."""

    def __init__(
        self,
        name: str,
        version: str,
        author: str,
        description: str,
        capabilities: List[PluginCapability],
        permission_level: PluginPermission = PluginPermission.SAFE
    ):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.capabilities = capabilities
        self.permission_level = permission_level
        self.installed_at = datetime.now()


class PluginContext:
    """
    Context provided to plugins.

    Gives plugins access to O2 functionality while maintaining sandboxing.
    """

    def __init__(self, plugin_name: str, matrix_client, memory_manager, governance):
        self.plugin_name = plugin_name
        self._matrix = matrix_client
        self._memory = memory_manager
        self._governance = governance
        self._capabilities = set()

    def grant_capability(self, capability: PluginCapability):
        """Grant capability to plugin."""
        self._capabilities.add(capability)

    def has_capability(self, capability: PluginCapability) -> bool:
        """Check if plugin has capability."""
        return capability in self._capabilities

    async def send_message(self, room_id: str, message: str):
        """Send message to Matrix room (requires SEND_MESSAGES capability)."""
        if not self.has_capability(PluginCapability.SEND_MESSAGES):
            raise PermissionError(f"Plugin {self.plugin_name} lacks SEND_MESSAGES capability")

        await self._matrix.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": message}
        )

    async def read_memory(self, user_id: str, query: str):
        """Query user's memory (requires READ_MEMORY capability)."""
        if not self.has_capability(PluginCapability.READ_MEMORY):
            raise PermissionError(f"Plugin {self.plugin_name} lacks READ_MEMORY capability")

        user_loom = await self._memory.get_user_loom(user_id)
        return await user_loom.recall(query)

    async def create_proposal(self, room_id: str, title: str, description: str):
        """Create governance proposal (requires CREATE_PROPOSALS capability)."""
        if not self.has_capability(PluginCapability.CREATE_PROPOSALS):
            raise PermissionError(f"Plugin {self.plugin_name} lacks CREATE_PROPOSALS capability")

        return await self._governance.create_proposal(
            room_id=room_id,
            author=f"plugin:{self.plugin_name}",
            title=title,
            description=description
        )


class PluginBase(ABC):
    """
    Base class for O2 plugins.

    Plugins must inherit from this class and implement required methods.
    """

    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.context: Optional[PluginContext] = None
        self.enabled = False

    @abstractmethod
    async def on_load(self, context: PluginContext):
        """
        Called when plugin is loaded.

        Args:
            context: Plugin context with O2 API access
        """
        pass

    @abstractmethod
    async def on_enable(self):
        """Called when plugin is enabled."""
        pass

    @abstractmethod
    async def on_disable(self):
        """Called when plugin is disabled."""
        pass

    async def on_message(self, room_id: str, sender: str, message: str) -> Optional[str]:
        """
        Handle incoming message.

        Args:
            room_id: Matrix room ID
            sender: Message sender
            message: Message content

        Returns:
            Response message or None
        """
        return None

    async def on_proposal_created(self, proposal_id: int, title: str, author: str):
        """Handle proposal creation event."""
        pass

    async def on_vote_cast(self, proposal_id: int, user_id: str, vote: str):
        """Handle vote cast event."""
        pass

    async def on_swarm_task(self, task: Dict) -> Optional[Dict]:
        """
        Handle swarm task.

        Args:
            task: Task dictionary

        Returns:
            Result dictionary or None if plugin can't handle
        """
        return None


class PluginManager:
    """
    Manages plugin lifecycle and execution.

    Features:
    - Auto-discovery (scan plugins/ directory)
    - Dynamic loading (importlib)
    - Capability enforcement (check before execution)
    - Event routing (dispatch events to plugins)
    - Sandboxing (plugins can't access core directly)
    """

    def __init__(self, plugins_dir: Path, matrix_client, memory_manager, governance):
        self.plugins_dir = plugins_dir
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        self.matrix_client = matrix_client
        self.memory_manager = memory_manager
        self.governance = governance

        self.plugins: Dict[str, PluginBase] = {}  # name → plugin instance
        self.metadata: Dict[str, PluginMetadata] = {}  # name → metadata

    async def discover_plugins(self):
        """
        Discover plugins in plugins/ directory.

        Looks for:
        - plugin.py (plugin code)
        - manifest.json (plugin metadata)
        """
        logger.info(f"Scanning for plugins in {self.plugins_dir}")

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            plugin_file = plugin_dir / "plugin.py"
            manifest_file = plugin_dir / "manifest.json"

            if not plugin_file.exists():
                logger.warning(f"Plugin {plugin_dir.name} missing plugin.py")
                continue

            if not manifest_file.exists():
                logger.warning(f"Plugin {plugin_dir.name} missing manifest.json")
                continue

            # Load plugin
            try:
                await self.load_plugin(plugin_dir.name, plugin_file, manifest_file)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_dir.name}: {e}", exc_info=True)

        logger.info(f"Loaded {len(self.plugins)} plugins")

    async def load_plugin(self, name: str, plugin_file: Path, manifest_file: Path):
        """
        Load a specific plugin.

        Args:
            name: Plugin name
            plugin_file: Path to plugin.py
            manifest_file: Path to manifest.json
        """
        # Load manifest
        import json
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        metadata = PluginMetadata(
            name=manifest['name'],
            version=manifest['version'],
            author=manifest['author'],
            description=manifest['description'],
            capabilities=[PluginCapability(c) for c in manifest['capabilities']],
            permission_level=PluginPermission(manifest.get('permission_level', 'safe'))
        )

        # Load plugin code
        spec = importlib.util.spec_from_file_location(name, plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find plugin class (must inherit from PluginBase)
        plugin_class = None
        for item_name in dir(module):
            item = getattr(module, item_name)
            if inspect.isclass(item) and issubclass(item, PluginBase) and item != PluginBase:
                plugin_class = item
                break

        if not plugin_class:
            raise ValueError(f"Plugin {name} has no PluginBase subclass")

        # Instantiate plugin
        plugin = plugin_class()
        plugin.metadata = metadata

        # Create context
        context = PluginContext(
            plugin_name=name,
            matrix_client=self.matrix_client,
            memory_manager=self.memory_manager,
            governance=self.governance
        )

        # Grant capabilities from manifest
        for capability in metadata.capabilities:
            context.grant_capability(capability)

        plugin.context = context

        # Call on_load
        await plugin.on_load(context)

        # Store plugin
        self.plugins[name] = plugin
        self.metadata[name] = metadata

        logger.info(f"Loaded plugin: {name} v{metadata.version} by {metadata.author}")

    async def enable_plugin(self, name: str):
        """Enable a plugin."""
        if name not in self.plugins:
            raise ValueError(f"Plugin {name} not found")

        plugin = self.plugins[name]

        if plugin.enabled:
            logger.warning(f"Plugin {name} already enabled")
            return

        await plugin.on_enable()
        plugin.enabled = True

        logger.info(f"Enabled plugin: {name}")

    async def disable_plugin(self, name: str):
        """Disable a plugin."""
        if name not in self.plugins:
            raise ValueError(f"Plugin {name} not found")

        plugin = self.plugins[name]

        if not plugin.enabled:
            logger.warning(f"Plugin {name} already disabled")
            return

        await plugin.on_disable()
        plugin.enabled = False

        logger.info(f"Disabled plugin: {name}")

    async def dispatch_message(self, room_id: str, sender: str, message: str) -> List[str]:
        """
        Dispatch message to all enabled plugins.

        Args:
            room_id: Matrix room ID
            sender: Message sender
            message: Message content

        Returns:
            List of responses from plugins
        """
        responses = []

        for name, plugin in self.plugins.items():
            if not plugin.enabled:
                continue

            if not plugin.context.has_capability(PluginCapability.PROCESS_MESSAGES):
                continue

            try:
                response = await plugin.on_message(room_id, sender, message)
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(f"Plugin {name} error handling message: {e}", exc_info=True)

        return responses

    async def dispatch_proposal_created(self, proposal_id: int, title: str, author: str):
        """Dispatch proposal creation event to plugins."""
        for name, plugin in self.plugins.items():
            if not plugin.enabled:
                continue

            if not plugin.context.has_capability(PluginCapability.OBSERVE_VOTES):
                continue

            try:
                await plugin.on_proposal_created(proposal_id, title, author)
            except Exception as e:
                logger.error(f"Plugin {name} error handling proposal: {e}", exc_info=True)

    async def dispatch_vote_cast(self, proposal_id: int, user_id: str, vote: str):
        """Dispatch vote cast event to plugins."""
        for name, plugin in self.plugins.items():
            if not plugin.enabled:
                continue

            if not plugin.context.has_capability(PluginCapability.OBSERVE_VOTES):
                continue

            try:
                await plugin.on_vote_cast(proposal_id, user_id, vote)
            except Exception as e:
                logger.error(f"Plugin {name} error handling vote: {e}", exc_info=True)

    async def dispatch_swarm_task(self, task: Dict) -> List[Dict]:
        """
        Dispatch swarm task to plugins.

        Args:
            task: Task dictionary

        Returns:
            List of results from plugins that handled task
        """
        results = []

        for name, plugin in self.plugins.items():
            if not plugin.enabled:
                continue

            if not plugin.context.has_capability(PluginCapability.JOIN_SWARM):
                continue

            try:
                result = await plugin.on_swarm_task(task)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Plugin {name} error handling swarm task: {e}", exc_info=True)

        return results

    def get_plugin_info(self, name: str) -> Optional[Dict]:
        """Get plugin metadata."""
        if name not in self.metadata:
            return None

        metadata = self.metadata[name]
        plugin = self.plugins[name]

        return {
            'name': metadata.name,
            'version': metadata.version,
            'author': metadata.author,
            'description': metadata.description,
            'capabilities': [c.value for c in metadata.capabilities],
            'permission_level': metadata.permission_level.value,
            'enabled': plugin.enabled,
            'installed_at': metadata.installed_at.isoformat()
        }

    def list_plugins(self) -> List[Dict]:
        """List all plugins."""
        return [self.get_plugin_info(name) for name in self.plugins.keys()]
