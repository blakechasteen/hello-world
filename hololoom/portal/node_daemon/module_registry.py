"""
Module Registry - Discovers and manages WASM modules.

Scans a directory for WASM modules with companion manifest.json files.
Provides lookup by module ID and lists available modules.
"""

import json
from pathlib import Path

from ..shared.logging import get_logger
from ..shared.types import ModuleManifest

logger = get_logger(__name__, component="node")


class ModuleRegistry:
    """
    Discovers and manages WASM modules from a directory.

    Each module consists of:
    - {module_id}.wasm - The WASM binary
    - {module_id}.json - Manifest describing the module

    Or a subdirectory with:
    - module.wasm
    - manifest.json
    """

    def __init__(self, module_dir: str):
        """
        Initialize registry and scan for modules.

        Args:
            module_dir: Directory containing WASM modules
        """
        self.module_dir = Path(module_dir)
        self.modules: dict[str, ModuleManifest] = {}
        self._wasm_paths: dict[str, Path] = {}
        self._scan_modules()

    def _scan_modules(self) -> None:
        """Scan module directory for WASM modules with manifests."""
        if not self.module_dir.exists():
            logger.warning(f"Module directory does not exist: {self.module_dir}")
            return

        # Pattern 1: {id}.wasm + {id}.json in root
        for wasm_file in self.module_dir.glob("*.wasm"):
            manifest_file = wasm_file.with_suffix(".json")
            if manifest_file.exists():
                self._load_module(manifest_file, wasm_file)

        # Pattern 2: subdirectory with module.wasm + manifest.json
        for subdir in self.module_dir.iterdir():
            if subdir.is_dir():
                wasm_file = subdir / "module.wasm"
                manifest_file = subdir / "manifest.json"
                if wasm_file.exists() and manifest_file.exists():
                    self._load_module(manifest_file, wasm_file)

        logger.info(f"Loaded {len(self.modules)} WASM modules")

    def _load_module(self, manifest_path: Path, wasm_path: Path) -> None:
        """Load a module from manifest and WASM files."""
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)

            manifest = ModuleManifest(**manifest_data)
            self.modules[manifest.id] = manifest
            self._wasm_paths[manifest.id] = wasm_path

            logger.debug(f"Loaded module: {manifest.id} v{manifest.version}")

        except Exception as e:
            logger.error(f"Failed to load module from {manifest_path}: {e}")

    def get(self, module_id: str) -> ModuleManifest | None:
        """
        Get module manifest by ID.

        Args:
            module_id: Module identifier

        Returns:
            ModuleManifest or None if not found
        """
        return self.modules.get(module_id)

    def get_wasm_path(self, module_id: str) -> Path | None:
        """
        Get path to WASM binary for a module.

        Args:
            module_id: Module identifier

        Returns:
            Path to WASM file or None if not found
        """
        return self._wasm_paths.get(module_id)

    def list(self) -> list[ModuleManifest]:
        """List all available modules."""
        return list(self.modules.values())

    def exists(self, module_id: str) -> bool:
        """Check if a module exists."""
        return module_id in self.modules

    def reload(self) -> None:
        """Reload modules from directory."""
        self.modules.clear()
        self._wasm_paths.clear()
        self._scan_modules()
