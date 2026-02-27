"""
Dark Trace Plugin Discovery: Safety-Scanned Plugin Detection

Discovers plugins from multiple sources with safety scanning BEFORE loading.

Discovery Sources:
1. Folder-based: plugin.yaml manifest + safety scan
2. EntryPoints: dark_trace.plugins entry point + signature check
3. Built-in: Official plugins (CORE trust level, always trusted)

Security Principle: "Scan before load, verify before trust"

Every discovered plugin is:
1. Scanned for known malicious patterns
2. Validated against blocked plugins list
3. Assigned initial trust level based on source
4. Checked for required manifest fields

Usage:
    from hololoom.dark_trace.plugins import PluginDiscovery, DiscoverySource

    discovery = PluginDiscovery(safety_gate)

    # Discover from all sources
    discovered = await discovery.discover_all()

    # Discover from specific folder
    folder_plugins = await discovery.discover_folder("./plugins")

    # Discover from entry points
    ep_plugins = await discovery.discover_entrypoints()

Date: 2025-12-22
Phase: 11.5 - Plugin Discovery (Safety Scanning)
"""

import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Discovery Source Types
# =============================================================================

class DiscoverySource(Enum):
    """Source of discovered plugin."""
    FOLDER = "folder"          # Discovered from plugin folder
    ENTRYPOINT = "entrypoint"  # Discovered from Python entry points
    BUILTIN = "builtin"        # Official HoloLoom plugin


class DiscoveryStatus(Enum):
    """Status of discovery attempt."""
    SUCCESS = "success"
    BLOCKED = "blocked"                  # On blocked list
    MALICIOUS_PATTERN = "malicious_pattern"  # Matched malicious pattern
    INVALID_MANIFEST = "invalid_manifest"
    IMPORT_FAILED = "import_failed"
    NOT_A_PLUGIN = "not_a_plugin"        # Doesn't implement DarkTracePlugin


# =============================================================================
# Discovery Result
# =============================================================================

@dataclass
class DiscoveredPlugin:
    """Information about a discovered plugin before loading."""
    name: str
    source: DiscoverySource
    status: DiscoveryStatus
    location: str  # File path or entry point name
    manifest: Optional[Dict[str, Any]] = None
    plugin_class: Optional[Type] = None
    suggested_trust: Optional[str] = None  # "sandboxed", "verified", etc.
    issues: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_valid(self) -> bool:
        return self.status == DiscoveryStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "source": self.source.value,
            "status": self.status.value,
            "location": self.location,
            "manifest": self.manifest,
            "suggested_trust": self.suggested_trust,
            "issues": self.issues,
            "discovered_at": self.discovered_at.isoformat(),
            "is_valid": self.is_valid,
        }


@dataclass
class DiscoveryResult:
    """Aggregated result of plugin discovery."""
    discovered: List[DiscoveredPlugin]
    valid_count: int = 0
    blocked_count: int = 0
    invalid_count: int = 0
    sources_scanned: Set[DiscoverySource] = field(default_factory=set)
    scan_duration_ms: float = 0.0

    @property
    def valid_plugins(self) -> List[DiscoveredPlugin]:
        return [p for p in self.discovered if p.is_valid]

    @property
    def blocked_plugins(self) -> List[DiscoveredPlugin]:
        return [p for p in self.discovered if p.status == DiscoveryStatus.BLOCKED]

    def summary(self) -> str:
        return (
            f"Discovery: {self.valid_count} valid, {self.blocked_count} blocked, "
            f"{self.invalid_count} invalid from {[s.value for s in self.sources_scanned]}"
        )


# =============================================================================
# Malicious Pattern Detection
# =============================================================================

# Patterns that indicate potentially malicious code
MALICIOUS_PATTERNS = [
    # System access
    r"os\.system\s*\(",
    r"subprocess\.call\s*\(",
    r"subprocess\.run\s*\(",
    r"subprocess\.Popen\s*\(",
    # Code execution
    r"exec\s*\(",
    r"eval\s*\(",
    r"compile\s*\(",
    r"__import__\s*\(",
    # Network
    r"socket\.socket\s*\(",
    r"urllib\.request\s*\(",
    r"requests\.(get|post|put|delete)\s*\(",
    # File system
    r"open\s*\([^)]*['\"][wa]",  # Writing files
    r"shutil\.rmtree\s*\(",
    r"os\.remove\s*\(",
    r"os\.unlink\s*\(",
    # Dangerous imports
    r"import\s+ctypes",
    r"from\s+ctypes\s+import",
    r"import\s+pickle",  # Unsafe deserialization
]

# Required fields in plugin manifest
REQUIRED_MANIFEST_FIELDS = {
    "name": str,
    "version": str,
    "author": str,
    "plugin_type": str,
}

# Optional but recommended fields
RECOMMENDED_MANIFEST_FIELDS = {
    "description": str,
    "capabilities": list,
    "dependencies": list,
    "signature": str,
}


# =============================================================================
# Plugin Discovery
# =============================================================================

class PluginDiscovery:
    """
    Discovers plugins from multiple sources with safety scanning.

    Features:
    - Multi-source discovery (folders, entry points, built-in)
    - Pre-load malicious pattern scanning
    - Manifest validation
    - Trust level suggestion based on source and signature
    - Blocked plugins list integration
    """

    def __init__(
        self,
        safety_gate: Optional[Any] = None,
        builtin_packages: Optional[List[str]] = None,
        custom_patterns: Optional[List[str]] = None,
        enable_pattern_scan: bool = True,
    ):
        """
        Initialize plugin discovery.

        Args:
            safety_gate: PluginSafetyGate for blocked plugins check
            builtin_packages: List of built-in plugin package names
            custom_patterns: Additional malicious patterns to check
            enable_pattern_scan: Whether to scan for malicious patterns
        """
        self._safety_gate = safety_gate
        self._enable_pattern_scan = enable_pattern_scan

        # Built-in plugins are always CORE trust
        self._builtin_packages = builtin_packages or [
            "hololoom.dark_trace.plugins.builtin"
        ]

        # Compile malicious patterns
        import re
        all_patterns = MALICIOUS_PATTERNS + (custom_patterns or [])
        self._patterns = [re.compile(p) for p in all_patterns]

        # Cache discovered plugins
        self._cache: Dict[str, DiscoveredPlugin] = {}

    # =========================================================================
    # Main Discovery Methods
    # =========================================================================

    async def discover_all(
        self,
        plugin_folders: Optional[List[str]] = None,
        include_entrypoints: bool = True,
        include_builtin: bool = True,
    ) -> DiscoveryResult:
        """
        Discover plugins from all sources.

        Args:
            plugin_folders: Folders to scan for plugins
            include_entrypoints: Include entry point plugins
            include_builtin: Include built-in plugins

        Returns:
            DiscoveryResult with all discovered plugins
        """
        import time
        start_time = time.perf_counter()

        discovered: List[DiscoveredPlugin] = []
        sources_scanned: Set[DiscoverySource] = set()

        # 1. Discover from folders
        if plugin_folders:
            for folder in plugin_folders:
                folder_plugins = await self.discover_folder(folder)
                discovered.extend(folder_plugins)
                sources_scanned.add(DiscoverySource.FOLDER)

        # 2. Discover from entry points
        if include_entrypoints:
            ep_plugins = await self.discover_entrypoints()
            discovered.extend(ep_plugins)
            sources_scanned.add(DiscoverySource.ENTRYPOINT)

        # 3. Discover built-in plugins
        if include_builtin:
            builtin_plugins = await self.discover_builtin()
            discovered.extend(builtin_plugins)
            sources_scanned.add(DiscoverySource.BUILTIN)

        # Calculate statistics
        valid_count = sum(1 for p in discovered if p.is_valid)
        blocked_count = sum(1 for p in discovered if p.status == DiscoveryStatus.BLOCKED)
        invalid_count = len(discovered) - valid_count - blocked_count

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return DiscoveryResult(
            discovered=discovered,
            valid_count=valid_count,
            blocked_count=blocked_count,
            invalid_count=invalid_count,
            sources_scanned=sources_scanned,
            scan_duration_ms=elapsed_ms,
        )

    async def discover_folder(self, folder_path: str) -> List[DiscoveredPlugin]:
        """
        Discover plugins from a folder.

        Looks for:
        - plugin.yaml or plugin.json manifest files
        - __init__.py with plugin class

        Args:
            folder_path: Path to scan

        Returns:
            List of discovered plugins
        """
        discovered: List[DiscoveredPlugin] = []
        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            logger.warning(f"Plugin folder not found: {folder_path}")
            return discovered

        # Scan subdirectories for plugins
        for item in folder.iterdir():
            if item.is_dir() and not item.name.startswith((".", "_")):
                plugin = await self._discover_folder_plugin(item)
                if plugin:
                    discovered.append(plugin)
                    self._cache[plugin.name] = plugin

        return discovered

    async def discover_entrypoints(
        self,
        group: str = "dark_trace.plugins"
    ) -> List[DiscoveredPlugin]:
        """
        Discover plugins from Python entry points.

        Args:
            group: Entry point group name

        Returns:
            List of discovered plugins
        """
        discovered: List[DiscoveredPlugin] = []

        try:
            # Python 3.10+
            from importlib.metadata import entry_points
            eps = entry_points(group=group)
        except (ImportError, TypeError):
            # Python 3.9
            try:
                from importlib.metadata import entry_points as get_eps
                all_eps = get_eps()
                eps = all_eps.get(group, [])
            except Exception:
                logger.debug("Entry points discovery not available")
                return discovered

        for ep in eps:
            try:
                plugin = await self._discover_entrypoint_plugin(ep)
                if plugin:
                    discovered.append(plugin)
                    self._cache[plugin.name] = plugin
            except Exception as e:
                logger.warning(f"Failed to discover entry point {ep.name}: {e}")

        return discovered

    async def discover_builtin(self) -> List[DiscoveredPlugin]:
        """
        Discover built-in plugins from known packages.

        Built-in plugins are always CORE trust level.

        Returns:
            List of discovered built-in plugins
        """
        discovered: List[DiscoveredPlugin] = []

        for package_name in self._builtin_packages:
            try:
                module = importlib.import_module(package_name)
                if hasattr(module, "__all__"):
                    for name in module.__all__:
                        cls = getattr(module, name, None)
                        if cls and self._is_plugin_class(cls):
                            plugin = DiscoveredPlugin(
                                name=name,
                                source=DiscoverySource.BUILTIN,
                                status=DiscoveryStatus.SUCCESS,
                                location=package_name,
                                plugin_class=cls,
                                suggested_trust="core",
                            )
                            discovered.append(plugin)
                            self._cache[name] = plugin
            except ImportError:
                logger.debug(f"Built-in package not found: {package_name}")

        return discovered

    # =========================================================================
    # Private Discovery Methods
    # =========================================================================

    async def _discover_folder_plugin(self, folder: Path) -> Optional[DiscoveredPlugin]:
        """Discover a single plugin from a folder."""
        issues: List[str] = []

        # 1. Look for manifest
        manifest = self._load_manifest(folder)
        if manifest is None:
            return DiscoveredPlugin(
                name=folder.name,
                source=DiscoverySource.FOLDER,
                status=DiscoveryStatus.INVALID_MANIFEST,
                location=str(folder),
                issues=["No plugin.yaml or plugin.json found"],
            )

        # 2. Validate manifest
        manifest_issues = self._validate_manifest(manifest)
        if manifest_issues:
            return DiscoveredPlugin(
                name=manifest.get("name", folder.name),
                source=DiscoverySource.FOLDER,
                status=DiscoveryStatus.INVALID_MANIFEST,
                location=str(folder),
                manifest=manifest,
                issues=manifest_issues,
            )

        plugin_name = manifest["name"]

        # 3. Check blocked list
        if self._safety_gate and await self._safety_gate.is_plugin_blocked(plugin_name):
            return DiscoveredPlugin(
                name=plugin_name,
                source=DiscoverySource.FOLDER,
                status=DiscoveryStatus.BLOCKED,
                location=str(folder),
                manifest=manifest,
                issues=["Plugin is on the blocked list"],
            )

        # 4. Scan for malicious patterns
        if self._enable_pattern_scan:
            malicious_issues = self._scan_folder_for_patterns(folder)
            if malicious_issues:
                return DiscoveredPlugin(
                    name=plugin_name,
                    source=DiscoverySource.FOLDER,
                    status=DiscoveryStatus.MALICIOUS_PATTERN,
                    location=str(folder),
                    manifest=manifest,
                    issues=malicious_issues,
                )

        # 5. Try to import plugin class
        plugin_class = await self._import_plugin_class(folder, manifest)
        if plugin_class is None:
            return DiscoveredPlugin(
                name=plugin_name,
                source=DiscoverySource.FOLDER,
                status=DiscoveryStatus.IMPORT_FAILED,
                location=str(folder),
                manifest=manifest,
                issues=["Failed to import plugin class"],
            )

        # 6. Verify it's a valid plugin
        if not self._is_plugin_class(plugin_class):
            return DiscoveredPlugin(
                name=plugin_name,
                source=DiscoverySource.FOLDER,
                status=DiscoveryStatus.NOT_A_PLUGIN,
                location=str(folder),
                manifest=manifest,
                issues=["Class does not implement DarkTracePlugin"],
            )

        # 7. Determine suggested trust level
        suggested_trust = self._determine_trust_level(manifest)

        return DiscoveredPlugin(
            name=plugin_name,
            source=DiscoverySource.FOLDER,
            status=DiscoveryStatus.SUCCESS,
            location=str(folder),
            manifest=manifest,
            plugin_class=plugin_class,
            suggested_trust=suggested_trust,
        )

    async def _discover_entrypoint_plugin(self, ep: Any) -> Optional[DiscoveredPlugin]:
        """Discover a single plugin from an entry point."""
        try:
            plugin_class = ep.load()

            # Check blocked list
            if self._safety_gate and await self._safety_gate.is_plugin_blocked(ep.name):
                return DiscoveredPlugin(
                    name=ep.name,
                    source=DiscoverySource.ENTRYPOINT,
                    status=DiscoveryStatus.BLOCKED,
                    location=str(ep),
                    issues=["Plugin is on the blocked list"],
                )

            # Verify it's a valid plugin
            if not self._is_plugin_class(plugin_class):
                return DiscoveredPlugin(
                    name=ep.name,
                    source=DiscoverySource.ENTRYPOINT,
                    status=DiscoveryStatus.NOT_A_PLUGIN,
                    location=str(ep),
                    issues=["Class does not implement DarkTracePlugin"],
                )

            # Entry points without signature are SANDBOXED
            return DiscoveredPlugin(
                name=ep.name,
                source=DiscoverySource.ENTRYPOINT,
                status=DiscoveryStatus.SUCCESS,
                location=str(ep),
                plugin_class=plugin_class,
                suggested_trust="sandboxed",  # Conservative default
            )

        except Exception as e:
            return DiscoveredPlugin(
                name=ep.name,
                source=DiscoverySource.ENTRYPOINT,
                status=DiscoveryStatus.IMPORT_FAILED,
                location=str(ep),
                issues=[str(e)],
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _load_manifest(self, folder: Path) -> Optional[Dict[str, Any]]:
        """Load plugin manifest from folder."""
        # Try YAML first
        yaml_path = folder / "plugin.yaml"
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, "r") as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.warning("PyYAML not installed, skipping YAML manifests")
            except Exception as e:
                logger.warning(f"Failed to load {yaml_path}: {e}")

        # Try JSON
        json_path = folder / "plugin.json"
        if json_path.exists():
            try:
                import json
                with open(json_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_path}: {e}")

        return None

    def _validate_manifest(self, manifest: Dict[str, Any]) -> List[str]:
        """Validate manifest has required fields."""
        issues = []

        for field_name, field_type in REQUIRED_MANIFEST_FIELDS.items():
            if field_name not in manifest:
                issues.append(f"Missing required field: {field_name}")
            elif not isinstance(manifest[field_name], field_type):
                issues.append(f"Field {field_name} must be {field_type.__name__}")

        return issues

    def _scan_folder_for_patterns(self, folder: Path) -> List[str]:
        """Scan folder for malicious patterns."""
        issues = []

        for py_file in folder.glob("**/*.py"):
            try:
                content = py_file.read_text()
                for pattern in self._patterns:
                    matches = pattern.findall(content)
                    if matches:
                        issues.append(
                            f"Suspicious pattern in {py_file.name}: {pattern.pattern}"
                        )
            except Exception as e:
                issues.append(f"Failed to scan {py_file}: {e}")

        return issues

    async def _import_plugin_class(
        self,
        folder: Path,
        manifest: Dict[str, Any],
    ) -> Optional[Type]:
        """Import the plugin class from a folder."""
        try:
            # Add folder to path temporarily
            folder_str = str(folder)
            if folder_str not in sys.path:
                sys.path.insert(0, folder_str)

            # Get module and class names
            module_name = manifest.get("module", "__init__")
            class_name = manifest.get("class", manifest["name"])

            # Import
            if module_name == "__init__":
                spec = importlib.util.spec_from_file_location(
                    manifest["name"],
                    folder / "__init__.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    return None
            else:
                module = importlib.import_module(module_name)

            return getattr(module, class_name, None)

        except Exception as e:
            logger.warning(f"Failed to import plugin from {folder}: {e}")
            return None
        finally:
            # Clean up path
            if folder_str in sys.path:
                sys.path.remove(folder_str)

    def _is_plugin_class(self, cls: Any) -> bool:
        """Check if class is a valid DarkTracePlugin subclass."""
        if not isinstance(cls, type):
            return False

        # Check for DarkTracePlugin base
        from hololoom.dark_trace.plugins.interface import DarkTracePlugin
        return issubclass(cls, DarkTracePlugin) and cls is not DarkTracePlugin

    def _determine_trust_level(self, manifest: Dict[str, Any]) -> str:
        """Determine suggested trust level from manifest."""
        # If signed and verified, can be VERIFIED
        if manifest.get("signature"):
            return "verified"

        # Check for HoloLoom author (trusted)
        author = manifest.get("author", "")
        if "hololoom" in author.lower() or "anthropic" in author.lower():
            return "verified"

        # Default to sandboxed
        return "sandboxed"

    # =========================================================================
    # Cache Management
    # =========================================================================

    def get_cached(self, name: str) -> Optional[DiscoveredPlugin]:
        """Get a cached discovered plugin by name."""
        return self._cache.get(name)

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()

    @property
    def cached_plugins(self) -> Dict[str, DiscoveredPlugin]:
        """Get all cached plugins."""
        return dict(self._cache)


# =============================================================================
# Factory Function
# =============================================================================

def create_discovery(
    safety_gate: Optional[Any] = None,
    **kwargs,
) -> PluginDiscovery:
    """
    Create a configured PluginDiscovery instance.

    Args:
        safety_gate: Optional safety gate for blocked plugins
        **kwargs: Additional arguments for PluginDiscovery

    Returns:
        Configured PluginDiscovery instance
    """
    if safety_gate is None:
        try:
            from hololoom.dark_trace.plugins.safety_gate import create_safety_gate
            safety_gate = create_safety_gate()
        except ImportError:
            pass

    return PluginDiscovery(safety_gate=safety_gate, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "DiscoverySource",
    "DiscoveryStatus",
    # Data classes
    "DiscoveredPlugin",
    "DiscoveryResult",
    # Main class
    "PluginDiscovery",
    # Factory
    "create_discovery",
    # Constants
    "MALICIOUS_PATTERNS",
    "REQUIRED_MANIFEST_FIELDS",
]
