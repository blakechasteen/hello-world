"""
Dark Trace Plugin Loader

Dynamic plugin loading with hot reload capability for seamless
plugin updates without system restart.

Author: HoloLoom Team
Created: December 2025
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import importlib
import importlib.util
import sys
import os
import json
import threading
import time
import hashlib
import logging

from HoloLoom.dark_trace.plugins.plugin_protocol import (
    DarkTracePlugin,
    LensPlugin,
    ValidatorPlugin,
    AnalyzerPlugin,
    VisualizerPlugin,
    SafetyPlugin,
    SteeringPlugin,
    DatasetPlugin,
    ModelAdapterPlugin,
    PluginMetadata,
    PluginCategory,
    PluginStatus,
    PluginFactory,
)

# Abstract base classes that should not be instantiated directly
_ABSTRACT_PLUGIN_CLASSES = (
    DarkTracePlugin,
    LensPlugin,
    ValidatorPlugin,
    AnalyzerPlugin,
    VisualizerPlugin,
    SafetyPlugin,
    SteeringPlugin,
    DatasetPlugin,
    ModelAdapterPlugin,
)
from HoloLoom.dark_trace.plugins.plugin_signing import (
    PluginVerifier,
    SignedManifest,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedPlugin:
    """Information about a loaded plugin."""
    plugin: DarkTracePlugin
    metadata: PluginMetadata
    module_name: str
    file_path: Path
    status: PluginStatus = PluginStatus.ACTIVE
    loaded_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None
    file_hash: Optional[str] = None
    error_message: Optional[str] = None

    def is_healthy(self) -> bool:
        try:
            return self.plugin.health_check()
        except Exception:
            return False


class PluginLoader:
    """
    Loads and manages Dark Trace plugins.

    Supports:
    - Dynamic loading from Python files
    - Plugin discovery from directories
    - Hot reload on file changes
    - Health monitoring
    """

    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        verifier: Optional[PluginVerifier] = None,
        require_signature: bool = False,
    ):
        self.plugin_dirs = plugin_dirs or []
        self.verifier = verifier or PluginVerifier()
        self.require_signature = require_signature

        self._loaded_plugins: Dict[str, LoadedPlugin] = {}
        self._plugin_factories: Dict[str, PluginFactory] = {}
        self._file_hashes: Dict[str, str] = {}
        self._lock = threading.RLock()

        # Hot reload monitoring
        self._hot_reload_enabled = False
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._hot_reload_stop = threading.Event()
        self._hot_reload_interval = 2.0  # seconds

        # Callbacks
        self._on_load_callbacks: List[Callable[[LoadedPlugin], None]] = []
        self._on_unload_callbacks: List[Callable[[str], None]] = []
        self._on_reload_callbacks: List[Callable[[LoadedPlugin], None]] = []

    def register_factory(self, name: str, factory: PluginFactory) -> None:
        """Register a plugin factory for programmatic plugin creation."""
        with self._lock:
            self._plugin_factories[name] = factory

    def load_from_factory(self, name: str) -> Optional[LoadedPlugin]:
        """Load a plugin from a registered factory."""
        with self._lock:
            if name not in self._plugin_factories:
                logger.warning(f"No factory registered for plugin: {name}")
                return None

            try:
                factory = self._plugin_factories[name]
                plugin = factory()
                metadata = plugin.metadata

                loaded = LoadedPlugin(
                    plugin=plugin,
                    metadata=metadata,
                    module_name=f"factory:{name}",
                    file_path=Path("."),
                    status=PluginStatus.ACTIVE,
                )

                plugin.on_load()
                self._loaded_plugins[metadata.name] = loaded
                self._notify_load(loaded)

                logger.info(f"Loaded plugin from factory: {metadata.name}")
                return loaded

            except Exception as e:
                logger.error(f"Failed to load plugin from factory {name}: {e}")
                return None

    def load_from_file(
        self,
        file_path: Path,
        verify: bool = True,
    ) -> Optional[LoadedPlugin]:
        """
        Load a plugin from a Python file.

        Args:
            file_path: Path to the plugin Python file
            verify: Whether to verify signature (if available)

        Returns:
            LoadedPlugin or None if loading failed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return None

        with self._lock:
            try:
                # Check for signature
                if verify and self.require_signature:
                    manifest_path = file_path.parent / "manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path, 'r') as f:
                            manifest_data = json.load(f)
                        signed_manifest = SignedManifest.from_dict(manifest_data)
                        valid, msg = self.verifier.full_verify(
                            file_path.parent, signed_manifest
                        )
                        if not valid:
                            logger.error(f"Plugin verification failed: {msg}")
                            return None
                    else:
                        logger.error(f"Signature required but manifest not found: {file_path}")
                        return None

                # Compute file hash
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                # Create unique module name
                module_name = f"dark_trace_plugin_{file_path.stem}_{file_hash[:8]}"

                # Load module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    logger.error(f"Could not create module spec for: {file_path}")
                    return None

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Find plugin class
                plugin_class = self._find_plugin_class(module)
                if plugin_class is None:
                    logger.error(f"No DarkTracePlugin subclass found in: {file_path}")
                    del sys.modules[module_name]
                    return None

                # Instantiate plugin
                plugin = plugin_class()
                metadata = plugin.metadata

                # Check if already loaded (reload case)
                if metadata.name in self._loaded_plugins:
                    self._unload_plugin(metadata.name)

                loaded = LoadedPlugin(
                    plugin=plugin,
                    metadata=metadata,
                    module_name=module_name,
                    file_path=file_path,
                    status=PluginStatus.ACTIVE,
                    file_hash=file_hash,
                )

                # Call on_load hook
                plugin.on_load()

                self._loaded_plugins[metadata.name] = loaded
                self._file_hashes[str(file_path)] = file_hash
                self._notify_load(loaded)

                logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
                return loaded

            except Exception as e:
                logger.error(f"Failed to load plugin from {file_path}: {e}")
                return None

    def load_from_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> List[LoadedPlugin]:
        """
        Load all plugins from a directory.

        Args:
            directory: Directory to search for plugins
            recursive: Whether to search subdirectories

        Returns:
            List of loaded plugins
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return []

        loaded = []
        pattern = "**/*.py" if recursive else "*.py"

        for py_file in directory.glob(pattern):
            # Skip __init__.py and test files
            if py_file.name.startswith("_") or "test" in py_file.name.lower():
                continue

            result = self.load_from_file(py_file)
            if result:
                loaded.append(result)

        return loaded

    def discover_plugins(self) -> List[LoadedPlugin]:
        """
        Discover and load plugins from all registered plugin directories.

        Returns:
            List of loaded plugins
        """
        loaded = []
        for plugin_dir in self.plugin_dirs:
            loaded.extend(self.load_from_directory(plugin_dir))
        return loaded

    def unload(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if unloaded successfully
        """
        return self._unload_plugin(plugin_name)

    def reload(self, plugin_name: str) -> Optional[LoadedPlugin]:
        """
        Reload a plugin from its original file.

        Args:
            plugin_name: Name of the plugin to reload

        Returns:
            Reloaded plugin or None if failed
        """
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                logger.warning(f"Plugin not loaded: {plugin_name}")
                return None

            loaded = self._loaded_plugins[plugin_name]
            file_path = loaded.file_path

            if loaded.module_name.startswith("factory:"):
                # Reload from factory
                factory_name = loaded.module_name.split(":", 1)[1]
                return self.load_from_factory(factory_name)
            else:
                # Reload from file
                result = self.load_from_file(file_path)
                if result:
                    self._notify_reload(result)
                return result

    def get_plugin(self, name: str) -> Optional[DarkTracePlugin]:
        """Get a loaded plugin by name."""
        with self._lock:
            if name in self._loaded_plugins:
                return self._loaded_plugins[name].plugin
            return None

    def get_plugins_by_category(
        self,
        category: PluginCategory
    ) -> List[DarkTracePlugin]:
        """Get all plugins of a specific category."""
        with self._lock:
            return [
                loaded.plugin
                for loaded in self._loaded_plugins.values()
                if loaded.metadata.category == category
                and loaded.status == PluginStatus.ACTIVE
            ]

    def list_plugins(self) -> List[PluginMetadata]:
        """List all loaded plugins."""
        with self._lock:
            return [loaded.metadata for loaded in self._loaded_plugins.values()]

    def get_status(self, plugin_name: str) -> Optional[PluginStatus]:
        """Get the status of a plugin."""
        with self._lock:
            if plugin_name in self._loaded_plugins:
                return self._loaded_plugins[plugin_name].status
            return None

    def enable(self, plugin_name: str) -> bool:
        """Enable a disabled plugin."""
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                return False

            loaded = self._loaded_plugins[plugin_name]
            if loaded.status == PluginStatus.DISABLED:
                loaded.plugin.on_enable()
                loaded.status = PluginStatus.ACTIVE
                logger.info(f"Enabled plugin: {plugin_name}")
                return True
            return False

    def disable(self, plugin_name: str) -> bool:
        """Disable an active plugin."""
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                return False

            loaded = self._loaded_plugins[plugin_name]
            if loaded.status == PluginStatus.ACTIVE:
                loaded.plugin.on_disable()
                loaded.status = PluginStatus.DISABLED
                logger.info(f"Disabled plugin: {plugin_name}")
                return True
            return False

    # Hot reload methods
    def enable_hot_reload(self, interval: float = 2.0) -> None:
        """Enable hot reload monitoring."""
        if self._hot_reload_enabled:
            return

        self._hot_reload_interval = interval
        self._hot_reload_enabled = True
        self._hot_reload_stop.clear()
        self._hot_reload_thread = threading.Thread(
            target=self._hot_reload_monitor,
            daemon=True,
        )
        self._hot_reload_thread.start()
        logger.info(f"Hot reload enabled (interval: {interval}s)")

    def disable_hot_reload(self) -> None:
        """Disable hot reload monitoring."""
        if not self._hot_reload_enabled:
            return

        self._hot_reload_stop.set()
        if self._hot_reload_thread:
            self._hot_reload_thread.join(timeout=5.0)
        self._hot_reload_enabled = False
        logger.info("Hot reload disabled")

    def _hot_reload_monitor(self) -> None:
        """Background thread that monitors for file changes."""
        while not self._hot_reload_stop.is_set():
            try:
                self._check_for_changes()
            except Exception as e:
                logger.error(f"Error in hot reload monitor: {e}")

            self._hot_reload_stop.wait(self._hot_reload_interval)

    def _check_for_changes(self) -> None:
        """Check for file changes and reload if necessary."""
        with self._lock:
            for plugin_name, loaded in list(self._loaded_plugins.items()):
                if loaded.module_name.startswith("factory:"):
                    continue

                file_path = loaded.file_path
                if not file_path.exists():
                    continue

                # Check hash
                with open(file_path, 'rb') as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()

                if current_hash != loaded.file_hash:
                    logger.info(f"Detected change in plugin: {plugin_name}")
                    self.reload(plugin_name)

    # Callback registration
    def on_load(self, callback: Callable[[LoadedPlugin], None]) -> None:
        """Register callback for plugin load events."""
        self._on_load_callbacks.append(callback)

    def on_unload(self, callback: Callable[[str], None]) -> None:
        """Register callback for plugin unload events."""
        self._on_unload_callbacks.append(callback)

    def on_reload(self, callback: Callable[[LoadedPlugin], None]) -> None:
        """Register callback for plugin reload events."""
        self._on_reload_callbacks.append(callback)

    def _notify_load(self, loaded: LoadedPlugin) -> None:
        """Notify callbacks of plugin load."""
        for callback in self._on_load_callbacks:
            try:
                callback(loaded)
            except Exception as e:
                logger.error(f"Error in load callback: {e}")

    def _notify_unload(self, plugin_name: str) -> None:
        """Notify callbacks of plugin unload."""
        for callback in self._on_unload_callbacks:
            try:
                callback(plugin_name)
            except Exception as e:
                logger.error(f"Error in unload callback: {e}")

    def _notify_reload(self, loaded: LoadedPlugin) -> None:
        """Notify callbacks of plugin reload."""
        for callback in self._on_reload_callbacks:
            try:
                callback(loaded)
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")

    def _unload_plugin(self, plugin_name: str) -> bool:
        """Internal method to unload a plugin."""
        with self._lock:
            if plugin_name not in self._loaded_plugins:
                return False

            loaded = self._loaded_plugins[plugin_name]

            try:
                loaded.plugin.on_unload()
            except Exception as e:
                logger.warning(f"Error in plugin on_unload: {e}")

            # Remove from module cache
            if loaded.module_name in sys.modules:
                del sys.modules[loaded.module_name]

            # Remove from tracking
            del self._loaded_plugins[plugin_name]
            if str(loaded.file_path) in self._file_hashes:
                del self._file_hashes[str(loaded.file_path)]

            self._notify_unload(plugin_name)
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True

    def _find_plugin_class(self, module) -> Optional[Type[DarkTracePlugin]]:
        """Find a DarkTracePlugin subclass in a module."""
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, DarkTracePlugin)
                and obj not in _ABSTRACT_PLUGIN_CLASSES
            ):
                return obj
        return None

    # Health check
    def check_health(self) -> Dict[str, bool]:
        """Check health of all loaded plugins."""
        results = {}
        with self._lock:
            for name, loaded in self._loaded_plugins.items():
                try:
                    healthy = loaded.is_healthy()
                    loaded.last_health_check = datetime.now()
                    results[name] = healthy
                except Exception:
                    results[name] = False
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded plugins."""
        with self._lock:
            by_category = {}
            for loaded in self._loaded_plugins.values():
                cat = loaded.metadata.category.value
                if cat not in by_category:
                    by_category[cat] = 0
                by_category[cat] += 1

            return {
                "total_loaded": len(self._loaded_plugins),
                "by_category": by_category,
                "by_status": {
                    status.value: sum(
                        1 for p in self._loaded_plugins.values()
                        if p.status == status
                    )
                    for status in PluginStatus
                },
                "hot_reload_enabled": self._hot_reload_enabled,
                "plugin_dirs": [str(d) for d in self.plugin_dirs],
            }


# Factory functions
def create_loader(
    plugin_dirs: Optional[List[Path]] = None,
    require_signature: bool = False,
) -> PluginLoader:
    """Create a plugin loader."""
    return PluginLoader(plugin_dirs, require_signature=require_signature)


def load_plugin(file_path: Path) -> Optional[LoadedPlugin]:
    """Quick function to load a single plugin."""
    loader = PluginLoader()
    return loader.load_from_file(file_path, verify=False)
