"""
Unit tests for Module Version Manager.

Tests VersionManager functionality including directory scanning,
version management, deprecation, and integrity checks.
"""

import pytest
import tempfile
import json
from pathlib import Path

from ..version_manager import (
    VersionManager,
    ModuleVersion,
    VersionStats,
)


class TestModuleVersion:
    """Tests for ModuleVersion dataclass."""

    def test_module_version_creation(self):
        """ModuleVersion stores all fields correctly."""
        version = ModuleVersion(
            module_id="test-module",
            version="v1.0.0",
            version_hash="abcd1234efgh5678",
            wasm_path=Path("/path/to/module.wasm"),
            manifest_path=Path("/path/to/manifest.json"),
            size_bytes=1024,
        )

        assert version.module_id == "test-module"
        assert version.version == "v1.0.0"
        assert version.version_hash == "abcd1234efgh5678"
        assert version.size_bytes == 1024
        assert version.deprecated is False
        assert version.deprecated_at is None

    def test_module_version_to_dict(self):
        """to_dict provides proper JSON structure."""
        version = ModuleVersion(
            module_id="test-module",
            version="v2.0.0",
            version_hash="1234567890abcdef",
            wasm_path=Path("/path/to/module.wasm"),
            size_bytes=2048,
        )

        d = version.to_dict()

        assert d["module_id"] == "test-module"
        assert d["version"] == "v2.0.0"
        assert d["version_hash"] == "1234567890abcdef"
        assert d["size_bytes"] == 2048
        assert d["deprecated"] is False
        assert "loaded_at" in d


class TestVersionManagerBasics:
    """Basic VersionManager tests."""

    def test_empty_directory(self):
        """Manager handles non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            manager = VersionManager(modules_dir=nonexistent)

            assert manager.list_modules() == []
            assert manager.get_active_version("any") is None

    def test_empty_modules_dir(self):
        """Manager handles empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = VersionManager(modules_dir=Path(tmpdir))

            assert manager.list_modules() == []
            assert manager.get_stats().total_modules == 0


class TestFlatLayout:
    """Tests for flat directory layout (module.wasm at root)."""

    def test_scan_flat_module(self):
        """Scans flat-layout modules (module_id.wasm)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create a flat module
            wasm_path = modules_dir / "my-module.wasm"
            wasm_path.write_bytes(b"fake wasm content")

            manager = VersionManager(modules_dir=modules_dir)

            # Should discover the module
            assert "my-module" in manager.list_modules()

            version = manager.get_active_version("my-module")
            assert version is not None
            assert version.module_id == "my-module"
            assert version.version == "1.0.0"  # Default version

    def test_flat_module_with_manifest(self):
        """Flat module with accompanying manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create module and manifest
            wasm_path = modules_dir / "analyzer.wasm"
            wasm_path.write_bytes(b"analyzer wasm")

            manifest_path = modules_dir / "analyzer.json"
            manifest_path.write_text(json.dumps({
                "name": "Analyzer Module",
                "entry_function": "analyze",
            }))

            manager = VersionManager(modules_dir=modules_dir)

            version = manager.get_active_version("analyzer")
            assert version is not None
            assert version.manifest_path is not None

            manifest = manager.get_manifest("analyzer")
            assert manifest is not None
            assert manifest["name"] == "Analyzer Module"


class TestVersionedLayout:
    """Tests for versioned directory layout."""

    def test_scan_versioned_modules(self):
        """Scans versioned directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create versioned module structure
            module_dir = modules_dir / "processor"
            v1_dir = module_dir / "v1.0.0"
            v1_dir.mkdir(parents=True)
            (v1_dir / "module.wasm").write_bytes(b"v1 content")

            v2_dir = module_dir / "v2.0.0"
            v2_dir.mkdir(parents=True)
            (v2_dir / "module.wasm").write_bytes(b"v2 content")

            manager = VersionManager(modules_dir=modules_dir)

            # Should find module
            assert "processor" in manager.list_modules()

            # Should have both versions
            versions = manager.list_versions("processor")
            assert len(versions) == 2

            version_strings = [v.version for v in versions]
            assert "v1.0.0" in version_strings
            assert "v2.0.0" in version_strings

    def test_versioned_with_manifests(self):
        """Versioned module with manifest in each version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create versioned structure with manifests
            module_dir = modules_dir / "transformer"
            v1_dir = module_dir / "v1.0.0"
            v1_dir.mkdir(parents=True)
            (v1_dir / "module.wasm").write_bytes(b"v1 wasm")
            (v1_dir / "manifest.json").write_text(json.dumps({
                "version": "1.0.0",
                "description": "First version",
            }))

            manager = VersionManager(modules_dir=modules_dir)

            manifest = manager.get_manifest("transformer", "v1.0.0")
            assert manifest is not None
            assert manifest["description"] == "First version"


class TestActiveVersionManagement:
    """Tests for active version selection."""

    def test_first_version_becomes_active(self):
        """First discovered version becomes active by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            wasm_path = modules_dir / "test.wasm"
            wasm_path.write_bytes(b"test content")

            manager = VersionManager(modules_dir=modules_dir)

            active = manager.get_active_version("test")
            assert active is not None
            assert active.module_id == "test"

    def test_set_active_version(self):
        """Can change active version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create two versions
            module_dir = modules_dir / "mymod"
            for version in ["v1.0.0", "v2.0.0"]:
                v_dir = module_dir / version
                v_dir.mkdir(parents=True)
                (v_dir / "module.wasm").write_bytes(f"{version} content".encode())

            manager = VersionManager(modules_dir=modules_dir)

            # Change to v2.0.0
            assert manager.set_active_version("mymod", "v2.0.0") is True

            active = manager.get_active_version("mymod")
            assert active is not None
            assert active.version == "v2.0.0"

    def test_set_active_nonexistent_version(self):
        """Cannot set non-existent version as active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            wasm_path = modules_dir / "test.wasm"
            wasm_path.write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)

            assert manager.set_active_version("test", "v99.0.0") is False

    def test_set_active_nonexistent_module(self):
        """Cannot set active for non-existent module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = VersionManager(modules_dir=Path(tmpdir))
            assert manager.set_active_version("nonexistent", "v1.0.0") is False


class TestDeprecation:
    """Tests for version deprecation."""

    def test_deprecate_version(self):
        """Can deprecate a version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create two versions
            module_dir = modules_dir / "mymod"
            for version in ["v1.0.0", "v2.0.0"]:
                v_dir = module_dir / version
                v_dir.mkdir(parents=True)
                (v_dir / "module.wasm").write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)

            # Deprecate v1
            assert manager.deprecate_version("mymod", "v1.0.0") is True

            v1 = manager.get_version("mymod", "v1.0.0")
            assert v1 is not None
            assert v1.deprecated is True
            assert v1.deprecated_at is not None

    def test_deprecate_active_switches_version(self):
        """Deprecating active version switches to another."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            module_dir = modules_dir / "mymod"
            for version in ["v1.0.0", "v2.0.0"]:
                v_dir = module_dir / version
                v_dir.mkdir(parents=True)
                (v_dir / "module.wasm").write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)

            # Set v1 as active
            manager.set_active_version("mymod", "v1.0.0")

            # Deprecate v1 - should switch to v2
            manager.deprecate_version("mymod", "v1.0.0")

            active = manager.get_active_version("mymod")
            assert active is not None
            assert active.version == "v2.0.0"

    def test_cannot_activate_deprecated(self):
        """Cannot set deprecated version as active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            module_dir = modules_dir / "mymod"
            for version in ["v1.0.0", "v2.0.0"]:
                v_dir = module_dir / version
                v_dir.mkdir(parents=True)
                (v_dir / "module.wasm").write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)
            manager.set_active_version("mymod", "v2.0.0")
            manager.deprecate_version("mymod", "v1.0.0")

            # Try to activate deprecated version
            assert manager.set_active_version("mymod", "v1.0.0") is False

    def test_undeprecate_version(self):
        """Can undeprecate a version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            wasm_path = modules_dir / "test.wasm"
            wasm_path.write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)
            manager.deprecate_version("test", "1.0.0")
            manager.undeprecate_version("test", "1.0.0")

            version = manager.get_version("test", "1.0.0")
            assert version is not None
            assert version.deprecated is False


class TestHashVerification:
    """Tests for content hash verification."""

    def test_hash_is_computed(self):
        """WASM files are hashed on discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            wasm_path = modules_dir / "test.wasm"
            wasm_path.write_bytes(b"specific content for hashing")

            manager = VersionManager(modules_dir=modules_dir)

            version = manager.get_active_version("test")
            assert version is not None
            assert len(version.version_hash) == 16

    def test_different_content_different_hash(self):
        """Different content produces different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create two modules with different content
            (modules_dir / "mod1.wasm").write_bytes(b"content one")
            (modules_dir / "mod2.wasm").write_bytes(b"content two")

            manager = VersionManager(modules_dir=modules_dir)

            v1 = manager.get_active_version("mod1")
            v2 = manager.get_active_version("mod2")

            assert v1 is not None
            assert v2 is not None
            assert v1.version_hash != v2.version_hash

    def test_verify_hash(self):
        """Can verify hash matches expected value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            wasm_path = modules_dir / "test.wasm"
            wasm_path.write_bytes(b"content to verify")

            manager = VersionManager(modules_dir=modules_dir)

            version = manager.get_active_version("test")
            assert version is not None

            # Verify with correct hash
            assert manager.verify_hash("test", "1.0.0", version.version_hash) is True

            # Verify with wrong hash
            assert manager.verify_hash("test", "1.0.0", "wronghash12345678") is False


class TestManualRegistration:
    """Tests for manually registering modules."""

    def test_register_version(self):
        """Can manually register a module version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            wasm_path = modules_dir / "external.wasm"
            wasm_path.write_bytes(b"external content")

            # Start with empty manager
            empty_dir = modules_dir / "empty"
            empty_dir.mkdir()
            manager = VersionManager(modules_dir=empty_dir)

            # Register external module
            version = manager.register_version(
                module_id="external",
                version="v3.0.0",
                wasm_path=wasm_path,
                set_active=True,
            )

            assert version is not None
            assert version.module_id == "external"
            assert version.version == "v3.0.0"

            # Should be active
            active = manager.get_active_version("external")
            assert active is not None
            assert active.version == "v3.0.0"

    def test_register_nonexistent_file(self):
        """Cannot register non-existent WASM file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = VersionManager(modules_dir=Path(tmpdir))

            version = manager.register_version(
                module_id="ghost",
                version="v1.0.0",
                wasm_path=Path("/nonexistent/path.wasm"),
            )

            assert version is None


class TestRescan:
    """Tests for directory rescanning."""

    def test_rescan_discovers_new_modules(self):
        """Rescan finds newly added modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            manager = VersionManager(modules_dir=modules_dir)
            assert manager.list_modules() == []

            # Add a module
            (modules_dir / "new.wasm").write_bytes(b"new content")

            # Rescan
            manager.rescan()

            assert "new" in manager.list_modules()

    def test_rescan_clears_removed_modules(self):
        """Rescan removes deleted modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            wasm_path = modules_dir / "temp.wasm"
            wasm_path.write_bytes(b"temp content")

            manager = VersionManager(modules_dir=modules_dir)
            assert "temp" in manager.list_modules()

            # Remove the module
            wasm_path.unlink()

            # Rescan
            manager.rescan()

            assert "temp" not in manager.list_modules()


class TestStatistics:
    """Tests for version statistics."""

    def test_stats_counting(self):
        """Stats accurately count modules and versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            # Create 2 modules with different version counts
            (modules_dir / "mod1.wasm").write_bytes(b"content1")

            mod2_dir = modules_dir / "mod2"
            for v in ["v1.0.0", "v2.0.0", "v3.0.0"]:
                v_dir = mod2_dir / v
                v_dir.mkdir(parents=True)
                (v_dir / "module.wasm").write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)

            stats = manager.get_stats()
            assert stats.total_modules == 2
            assert stats.total_versions == 4  # 1 + 3

    def test_stats_tracks_deprecation(self):
        """Stats count deprecated versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            mod_dir = modules_dir / "mod"
            for v in ["v1.0.0", "v2.0.0"]:
                v_dir = mod_dir / v
                v_dir.mkdir(parents=True)
                (v_dir / "module.wasm").write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)
            manager.deprecate_version("mod", "v1.0.0")

            stats = manager.get_stats()
            assert stats.deprecated_versions == 1

    def test_stats_tracks_size(self):
        """Stats track total size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)

            content1 = b"x" * 100
            content2 = b"y" * 200

            (modules_dir / "mod1.wasm").write_bytes(content1)
            (modules_dir / "mod2.wasm").write_bytes(content2)

            manager = VersionManager(modules_dir=modules_dir)

            stats = manager.get_stats()
            assert stats.total_size_bytes == 300


class TestToDictExport:
    """Tests for state export."""

    def test_to_dict_structure(self):
        """to_dict exports correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modules_dir = Path(tmpdir)
            (modules_dir / "test.wasm").write_bytes(b"content")

            manager = VersionManager(modules_dir=modules_dir)

            d = manager.to_dict()

            assert "modules_dir" in d
            assert "modules" in d
            assert "stats" in d
            assert "test" in d["modules"]
            assert d["modules"]["test"]["active_version"] == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
