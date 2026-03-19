"""
Module Version Manager - Track and manage WASM module versions.

Provides semantic versioning, safe rollouts, and version lifecycle management:
- Track all versions of each module
- Set active versions for production use
- Deprecate old versions with grace periods
- SHA256 hashing for content verification
- Support both flat and versioned directory structures

Usage:
    manager = VersionManager(modules_dir=Path("./wasm_modules"))

    # List all versions of a module
    versions = manager.list_versions("my-module")

    # Get active (default) version
    active = manager.get_active_version("my-module")

    # Change active version (safe rollout)
    manager.set_active_version("my-module", "v2.0.0")

    # Deprecate an old version
    manager.deprecate_version("my-module", "v1.0.0")

Directory structure:
    wasm_modules/
      {module_id}/
        v1.0.0/
          module.wasm
          manifest.json
        v1.1.0/
          module.wasm
          manifest.json
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModuleVersion:
    """
    A specific version of a WASM module.

    Tracks:
    - Module identity and version string
    - Content hash for verification
    - File paths (WASM + manifest)
    - Lifecycle state (deprecated, deprecated_at)
    """

    module_id: str
    version: str
    version_hash: str  # SHA256 of WASM content (first 16 chars)
    wasm_path: Path
    manifest_path: Path | None = None
    deprecated: bool = False
    deprecated_at: float | None = None  # Unix timestamp
    size_bytes: int = 0
    loaded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "module_id": self.module_id,
            "version": self.version,
            "version_hash": self.version_hash,
            "wasm_path": str(self.wasm_path),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "deprecated": self.deprecated,
            "deprecated_at": self.deprecated_at,
            "size_bytes": self.size_bytes,
            "loaded_at": self.loaded_at,
        }


@dataclass
class VersionStats:
    """Statistics about version management."""

    total_modules: int = 0
    total_versions: int = 0
    deprecated_versions: int = 0
    total_size_bytes: int = 0


class VersionManager:
    """
    Manages WASM module versions with safe rollout support.

    Supports two directory layouts:

    Versioned (recommended):
        wasm_modules/
          {module_id}/
            v1.0.0/
              module.wasm
              manifest.json
            v1.1.0/
              module.wasm
              manifest.json

    Flat (legacy):
        wasm_modules/
          {module_id}.wasm
          {module_id}.json

    Args:
        modules_dir: Root directory containing WASM modules
        default_version: Version to assign to flat-structure modules
        wasm_filename: Name of WASM file in versioned dirs (default: module.wasm)
        manifest_filename: Name of manifest file (default: manifest.json)
    """

    def __init__(
        self,
        modules_dir: Path,
        default_version: str = "1.0.0",
        wasm_filename: str = "module.wasm",
        manifest_filename: str = "manifest.json",
    ):
        self.modules_dir = Path(modules_dir)
        self.default_version = default_version
        self.wasm_filename = wasm_filename
        self.manifest_filename = manifest_filename

        # module_id -> version -> ModuleVersion
        self._versions: dict[str, dict[str, ModuleVersion]] = {}
        # module_id -> active version string
        self._active: dict[str, str] = {}

        # Scan modules on initialization
        if self.modules_dir.exists():
            self._scan_modules()

    def _hash_file(self, path: Path) -> str:
        """
        Compute SHA256 hash of file contents.

        Returns first 16 characters for a compact identifier.
        """
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()[:16]
        except Exception:
            return "0" * 16  # Return placeholder if hashing fails

    def _get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        try:
            return path.stat().st_size
        except Exception:
            return 0

    def _scan_modules(self) -> None:
        """
        Scan directory for modules (both versioned and flat layouts).

        Called on initialization to discover existing modules.
        """
        if not self.modules_dir.exists():
            return

        for item in self.modules_dir.iterdir():
            if item.is_dir():
                # Could be a module directory with versions
                self._scan_module_directory(item)
            elif item.suffix == ".wasm":
                # Flat layout: module_id.wasm
                self._register_flat_module(item)

    def _scan_module_directory(self, module_dir: Path) -> None:
        """
        Scan a module directory for versioned subdirectories.

        Directory structure:
          module_dir/
            v1.0.0/
              module.wasm
              manifest.json
        """
        module_id = module_dir.name
        has_versions = False

        for version_dir in module_dir.iterdir():
            if not version_dir.is_dir():
                continue

            version = version_dir.name
            wasm_path = version_dir / self.wasm_filename
            manifest_path = version_dir / self.manifest_filename

            if wasm_path.exists():
                has_versions = True
                version_hash = self._hash_file(wasm_path)
                size_bytes = self._get_file_size(wasm_path)

                if module_id not in self._versions:
                    self._versions[module_id] = {}

                self._versions[module_id][version] = ModuleVersion(
                    module_id=module_id,
                    version=version,
                    version_hash=version_hash,
                    wasm_path=wasm_path,
                    manifest_path=manifest_path if manifest_path.exists() else None,
                    size_bytes=size_bytes,
                )

                # Set first discovered version as active (will be overwritten by latest)
                if module_id not in self._active:
                    self._active[module_id] = version

        # If no versioned subdirs but directory contains a wasm file directly
        if not has_versions:
            direct_wasm = module_dir / self.wasm_filename
            if direct_wasm.exists():
                self._register_module_version(
                    module_id=module_id,
                    version=self.default_version,
                    wasm_path=direct_wasm,
                    manifest_path=module_dir / self.manifest_filename,
                )

    def _register_flat_module(self, wasm_path: Path) -> None:
        """Register a flat-layout module (module_id.wasm at root)."""
        module_id = wasm_path.stem  # filename without extension
        manifest_path = wasm_path.with_suffix(".json")

        self._register_module_version(
            module_id=module_id,
            version=self.default_version,
            wasm_path=wasm_path,
            manifest_path=manifest_path,
        )

    def _register_module_version(
        self,
        module_id: str,
        version: str,
        wasm_path: Path,
        manifest_path: Path | None = None,
    ) -> ModuleVersion:
        """
        Register a module version internally.

        Returns the created ModuleVersion.
        """
        version_hash = self._hash_file(wasm_path)
        size_bytes = self._get_file_size(wasm_path)

        if module_id not in self._versions:
            self._versions[module_id] = {}

        module_version = ModuleVersion(
            module_id=module_id,
            version=version,
            version_hash=version_hash,
            wasm_path=wasm_path,
            manifest_path=manifest_path if manifest_path and manifest_path.exists() else None,
            size_bytes=size_bytes,
        )

        self._versions[module_id][version] = module_version

        # Set as active if first version
        if module_id not in self._active:
            self._active[module_id] = version

        return module_version

    def get_active_version(self, module_id: str) -> ModuleVersion | None:
        """
        Get currently active version of a module.

        The active version is the one that should be used for new jobs.

        Args:
            module_id: Module identifier

        Returns:
            ModuleVersion if found, None otherwise
        """
        if module_id not in self._active:
            return None
        version = self._active[module_id]
        return self._versions.get(module_id, {}).get(version)

    def get_version(self, module_id: str, version: str) -> ModuleVersion | None:
        """
        Get a specific version of a module.

        Args:
            module_id: Module identifier
            version: Version string (e.g., "v1.0.0")

        Returns:
            ModuleVersion if found, None otherwise
        """
        return self._versions.get(module_id, {}).get(version)

    def list_versions(self, module_id: str) -> list[ModuleVersion]:
        """
        List all versions of a module.

        Args:
            module_id: Module identifier

        Returns:
            List of ModuleVersion objects (empty if module not found)
        """
        return list(self._versions.get(module_id, {}).values())

    def list_modules(self) -> list[str]:
        """
        List all registered module IDs.

        Returns:
            List of module identifiers
        """
        return list(self._versions.keys())

    def set_active_version(self, module_id: str, version: str) -> bool:
        """
        Set active version for a module.

        The active version is used for new job dispatches.
        This enables safe rollouts by switching traffic to new versions.

        Args:
            module_id: Module identifier
            version: Version to activate

        Returns:
            True if successful, False if module/version not found
        """
        if module_id not in self._versions:
            return False
        if version not in self._versions[module_id]:
            return False

        # Don't activate deprecated versions
        module_version = self._versions[module_id][version]
        if module_version.deprecated:
            return False

        self._active[module_id] = version
        return True

    def deprecate_version(self, module_id: str, version: str) -> bool:
        """
        Mark a version as deprecated.

        Deprecated versions:
        - Cannot be set as active
        - Will not be used for new jobs
        - Can still be queried for historical purposes

        Args:
            module_id: Module identifier
            version: Version to deprecate

        Returns:
            True if successful, False if module/version not found
        """
        if module_id not in self._versions:
            return False
        if version not in self._versions[module_id]:
            return False

        self._versions[module_id][version].deprecated = True
        self._versions[module_id][version].deprecated_at = time.time()

        # If deprecating the active version, switch to another if available
        if self._active.get(module_id) == version:
            # Find a non-deprecated version to activate
            for v, mod in self._versions[module_id].items():
                if not mod.deprecated:
                    self._active[module_id] = v
                    break
            else:
                # All versions deprecated - remove from active
                del self._active[module_id]

        return True

    def undeprecate_version(self, module_id: str, version: str) -> bool:
        """
        Remove deprecation flag from a version.

        Args:
            module_id: Module identifier
            version: Version to undeprecate

        Returns:
            True if successful, False if module/version not found
        """
        if module_id not in self._versions:
            return False
        if version not in self._versions[module_id]:
            return False

        self._versions[module_id][version].deprecated = False
        self._versions[module_id][version].deprecated_at = None

        # If no active version, make this one active
        if module_id not in self._active:
            self._active[module_id] = version

        return True

    def register_version(
        self,
        module_id: str,
        version: str,
        wasm_path: Path,
        manifest_path: Path | None = None,
        set_active: bool = False,
    ) -> ModuleVersion | None:
        """
        Manually register a module version.

        Use this to add modules that weren't discovered by scanning.

        Args:
            module_id: Module identifier
            version: Version string
            wasm_path: Path to WASM file
            manifest_path: Optional path to manifest JSON
            set_active: Whether to set this as the active version

        Returns:
            ModuleVersion if successful, None if WASM file doesn't exist
        """
        if not wasm_path.exists():
            return None

        module_version = self._register_module_version(
            module_id=module_id,
            version=version,
            wasm_path=wasm_path,
            manifest_path=manifest_path,
        )

        if set_active:
            self._active[module_id] = version

        return module_version

    def verify_hash(self, module_id: str, version: str, expected_hash: str) -> bool:
        """
        Verify that a module version's hash matches expected value.

        Useful for security/integrity checks.

        Args:
            module_id: Module identifier
            version: Version string
            expected_hash: Expected SHA256 hash prefix (16 chars)

        Returns:
            True if hashes match, False otherwise
        """
        module_version = self.get_version(module_id, version)
        if module_version is None:
            return False
        return module_version.version_hash == expected_hash

    def rescan(self) -> None:
        """
        Rescan the modules directory.

        Clears existing state and rediscovers all modules.
        Useful after manual file changes.
        """
        self._versions.clear()
        self._active.clear()
        if self.modules_dir.exists():
            self._scan_modules()

    def get_stats(self) -> VersionStats:
        """
        Get version management statistics.

        Returns:
            VersionStats with counts and sizes
        """
        total_versions = 0
        deprecated_versions = 0
        total_size = 0

        for versions in self._versions.values():
            for module_version in versions.values():
                total_versions += 1
                total_size += module_version.size_bytes
                if module_version.deprecated:
                    deprecated_versions += 1

        return VersionStats(
            total_modules=len(self._versions),
            total_versions=total_versions,
            deprecated_versions=deprecated_versions,
            total_size_bytes=total_size,
        )

    def get_manifest(self, module_id: str, version: str | None = None) -> dict[str, Any] | None:
        """
        Load and return the manifest for a module version.

        Args:
            module_id: Module identifier
            version: Version string (uses active if not specified)

        Returns:
            Parsed manifest dict, or None if not found
        """
        if version:
            module_version = self.get_version(module_id, version)
        else:
            module_version = self.get_active_version(module_id)

        if module_version is None or module_version.manifest_path is None:
            return None

        try:
            with open(module_version.manifest_path) as f:
                return json.load(f)
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        """
        Export version state as dictionary.

        Useful for debugging and persistence.
        """
        return {
            "modules_dir": str(self.modules_dir),
            "modules": {
                module_id: {
                    "active_version": self._active.get(module_id),
                    "versions": {
                        v: mod.to_dict() for v, mod in versions.items()
                    },
                }
                for module_id, versions in self._versions.items()
            },
            "stats": {
                "total_modules": len(self._versions),
                "total_versions": sum(len(v) for v in self._versions.values()),
            },
        }
