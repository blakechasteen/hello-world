"""
Skill Package Manager
======================

**Created**: November 22, 2025
**Purpose**: Install, upgrade, and remove skills

The package manager provides:
- Install skills from catalog
- Upgrade to newer versions
- Remove installed skills
- Dependency resolution
- Rollback capability
- Safe installation (validation + testing)

Architecture:
```
PackageManager
├── Install
│   ├── Download/copy skill files
│   ├── Resolve dependencies
│   ├── Run preflight checks
│   ├── Test installation
│   └── Update catalog metrics
│
├── Upgrade
│   ├── Check for newer version
│   ├── Backup current version
│   ├── Install new version
│   ├── Rollback on failure
│   └── Update catalog
│
└── Remove
    ├── Check dependents
    ├── Backup before removal
    ├── Remove skill files
    ├── Update catalog
    └── Cleanup
```
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import json
import logging
from datetime import datetime
import asyncio

from .catalog import SkillCatalog, SkillMetadata, SkillCategory

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class InstallResult:
    """Result of skill installation."""
    success: bool
    skill_id: str
    message: str
    installed_path: Optional[Path] = None
    dependencies_installed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class UpgradeResult:
    """Result of skill upgrade."""
    success: bool
    skill_id: str
    old_version: str
    new_version: str
    message: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rollback_available: bool = True


@dataclass
class RemoveResult:
    """Result of skill removal."""
    success: bool
    skill_id: str
    message: str
    backup_path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Package Manager
# ============================================================================

class PackageManager:
    """
    Manages skill installation, upgrade, and removal.

    Features:
    - Safe installation with validation
    - Automatic dependency resolution
    - Rollback on failure
    - Backup before upgrade/remove
    - Catalog integration
    """

    def __init__(
        self,
        catalog: SkillCatalog,
        skills_root: Optional[Path] = None,
        backup_root: Optional[Path] = None
    ):
        """
        Initialize package manager.

        Args:
            catalog: Skill catalog
            skills_root: Root directory for installed skills (default: ./skills)
            backup_root: Root directory for backups (default: ./skills_backups)
        """
        self.catalog = catalog
        self.skills_root = skills_root or Path("./skills")
        self.backup_root = backup_root or Path("./skills_backups")

        # Ensure directories exist
        self.skills_root.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)

        # Installed skills tracking
        self.installed_skills: Dict[str, Path] = {}

    async def install(
        self,
        skill_id: str,
        force: bool = False,
        skip_dependencies: bool = False,
        skip_tests: bool = False
    ) -> InstallResult:
        """
        Install a skill.

        Args:
            skill_id: Skill ID to install
            force: Force reinstall if already installed
            skip_dependencies: Skip dependency installation
            skip_tests: Skip post-install tests

        Returns:
            Install result
        """
        # Get skill metadata
        metadata = await self.catalog.get_skill(skill_id)
        if not metadata:
            return InstallResult(
                success=False,
                skill_id=skill_id,
                message=f"Skill not found in catalog: {skill_id}",
                errors=[f"Skill {skill_id} not in catalog"]
            )

        # Check if already installed
        if skill_id in self.installed_skills and not force:
            return InstallResult(
                success=False,
                skill_id=skill_id,
                message=f"Skill already installed: {skill_id}. Use force=True to reinstall.",
                errors=[f"Skill {skill_id} already installed"]
            )

        logger.info(f"Installing skill: {skill_id}")

        # Install dependencies first
        dependencies_installed = []
        if not skip_dependencies and metadata.dependencies:
            logger.info(f"Installing {len(metadata.dependencies)} dependencies")
            for dep in metadata.dependencies:
                dep_result = await self.install(dep, skip_tests=True)
                if not dep_result.success:
                    return InstallResult(
                        success=False,
                        skill_id=skill_id,
                        message=f"Failed to install dependency: {dep}",
                        errors=[f"Dependency installation failed: {dep}"] + dep_result.errors
                    )
                dependencies_installed.append(dep)

        # Determine install path
        if metadata.category == SkillCategory.META:
            install_path = self.skills_root / "meta" / metadata.name
        elif metadata.category == SkillCategory.DOMAIN:
            install_path = self.skills_root / "domain" / metadata.domain / metadata.name
        else:  # AGENTIC
            install_path = self.skills_root / "agentic" / metadata.name

        install_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy skill files
        try:
            source_path = Path(metadata.source_path)
            if source_path.exists():
                if source_path.is_file():
                    # Single file skill
                    install_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, install_path / source_path.name)
                else:
                    # Directory skill
                    if install_path.exists():
                        shutil.rmtree(install_path)
                    shutil.copytree(source_path, install_path)
            else:
                return InstallResult(
                    success=False,
                    skill_id=skill_id,
                    message=f"Source not found: {source_path}",
                    errors=[f"Source path does not exist: {source_path}"]
                )

        except Exception as e:
            return InstallResult(
                success=False,
                skill_id=skill_id,
                message=f"Failed to copy skill files: {e}",
                errors=[str(e)]
            )

        # Run post-install tests (if not skipped)
        warnings = []
        if not skip_tests:
            test_result = await self._run_post_install_tests(metadata, install_path)
            if not test_result['success']:
                # Tests failed - rollback
                logger.warning(f"Post-install tests failed for {skill_id}")
                await self._rollback_install(install_path)
                return InstallResult(
                    success=False,
                    skill_id=skill_id,
                    message="Post-install tests failed",
                    errors=test_result['errors']
                )
            warnings = test_result.get('warnings', [])

        # Update catalog
        await self.catalog.record_install(skill_id)

        # Track installation
        self.installed_skills[skill_id] = install_path

        logger.info(f"Successfully installed: {skill_id} at {install_path}")

        return InstallResult(
            success=True,
            skill_id=skill_id,
            message=f"Successfully installed {skill_id}",
            installed_path=install_path,
            dependencies_installed=dependencies_installed,
            warnings=warnings
        )

    async def upgrade(
        self,
        skill_id: str,
        version: Optional[str] = None,
        skip_tests: bool = False
    ) -> UpgradeResult:
        """
        Upgrade a skill to newer version.

        Args:
            skill_id: Skill ID to upgrade
            version: Target version (default: latest)
            skip_tests: Skip post-upgrade tests

        Returns:
            Upgrade result
        """
        # Check if installed
        if skill_id not in self.installed_skills:
            return UpgradeResult(
                success=False,
                skill_id=skill_id,
                old_version="",
                new_version=version or "",
                message=f"Skill not installed: {skill_id}",
                errors=[f"Skill {skill_id} not installed"]
            )

        # Get current metadata
        current_metadata = await self.catalog.get_skill(skill_id)
        if not current_metadata:
            return UpgradeResult(
                success=False,
                skill_id=skill_id,
                old_version="",
                new_version=version or "",
                message=f"Skill not in catalog: {skill_id}",
                errors=[f"Skill {skill_id} not in catalog"]
            )

        old_version = current_metadata.version

        # If version not specified, find latest
        if not version:
            # TODO: Query catalog for latest version
            # For now, assume same version (no upgrade)
            return UpgradeResult(
                success=False,
                skill_id=skill_id,
                old_version=old_version,
                new_version=old_version,
                message="No newer version available",
                warnings=["Version detection not yet implemented"]
            )

        # Check if version is newer
        # TODO: Proper version comparison
        if version == old_version:
            return UpgradeResult(
                success=False,
                skill_id=skill_id,
                old_version=old_version,
                new_version=version,
                message="Already on target version",
                warnings=[f"Already on version {version}"]
            )

        logger.info(f"Upgrading {skill_id}: {old_version} → {version}")

        # Backup current installation
        backup_path = await self._backup_skill(skill_id)
        if not backup_path:
            return UpgradeResult(
                success=False,
                skill_id=skill_id,
                old_version=old_version,
                new_version=version,
                message="Failed to backup current installation",
                errors=["Backup failed"],
                rollback_available=False
            )

        # Uninstall current version
        remove_result = await self.remove(skill_id, skip_backup=True)  # Already backed up
        if not remove_result.success:
            return UpgradeResult(
                success=False,
                skill_id=skill_id,
                old_version=old_version,
                new_version=version,
                message="Failed to remove current version",
                errors=remove_result.errors
            )

        # Install new version
        # TODO: Modify skill_id to include version
        new_skill_id = f"{skill_id.split(':')[0]}:{version}"
        install_result = await self.install(new_skill_id, skip_tests=skip_tests)

        if not install_result.success:
            # Installation failed - rollback
            logger.error(f"Upgrade failed, rolling back to {old_version}")
            await self._restore_backup(skill_id, backup_path)

            return UpgradeResult(
                success=False,
                skill_id=skill_id,
                old_version=old_version,
                new_version=version,
                message=f"Upgrade failed, rolled back to {old_version}",
                errors=install_result.errors
            )

        logger.info(f"Successfully upgraded {skill_id}: {old_version} → {version}")

        return UpgradeResult(
            success=True,
            skill_id=skill_id,
            old_version=old_version,
            new_version=version,
            message=f"Successfully upgraded from {old_version} to {version}",
            warnings=install_result.warnings
        )

    async def remove(
        self,
        skill_id: str,
        skip_backup: bool = False,
        force: bool = False
    ) -> RemoveResult:
        """
        Remove an installed skill.

        Args:
            skill_id: Skill ID to remove
            skip_backup: Skip backup before removal
            force: Force removal even if other skills depend on it

        Returns:
            Remove result
        """
        # Check if installed
        if skill_id not in self.installed_skills:
            return RemoveResult(
                success=False,
                skill_id=skill_id,
                message=f"Skill not installed: {skill_id}",
                errors=[f"Skill {skill_id} not installed"]
            )

        install_path = self.installed_skills[skill_id]

        # Check for dependent skills (if not forced)
        if not force:
            dependents = await self._find_dependents(skill_id)
            if dependents:
                return RemoveResult(
                    success=False,
                    skill_id=skill_id,
                    message=f"Other skills depend on this: {', '.join(dependents)}",
                    errors=[f"Cannot remove - dependents: {', '.join(dependents)}"],
                    warnings=["Use force=True to remove anyway"]
                )

        logger.info(f"Removing skill: {skill_id}")

        # Backup before removal (if not skipped)
        backup_path = None
        if not skip_backup:
            backup_path = await self._backup_skill(skill_id)
            if not backup_path:
                logger.warning(f"Failed to backup {skill_id} before removal")

        # Remove skill files
        try:
            if install_path.exists():
                if install_path.is_file():
                    install_path.unlink()
                else:
                    shutil.rmtree(install_path)
        except Exception as e:
            return RemoveResult(
                success=False,
                skill_id=skill_id,
                message=f"Failed to remove skill files: {e}",
                backup_path=backup_path,
                errors=[str(e)]
            )

        # Remove from tracking
        del self.installed_skills[skill_id]

        logger.info(f"Successfully removed: {skill_id}")

        return RemoveResult(
            success=True,
            skill_id=skill_id,
            message=f"Successfully removed {skill_id}",
            backup_path=backup_path
        )

    async def list_installed(self) -> List[SkillMetadata]:
        """List all installed skills."""
        installed = []
        for skill_id in self.installed_skills:
            metadata = await self.catalog.get_skill(skill_id)
            if metadata:
                installed.append(metadata)
        return installed

    async def check_updates(self) -> Dict[str, str]:
        """
        Check for available updates.

        Returns:
            Dict of {skill_id: new_version} for skills with updates
        """
        # TODO: Implement version checking
        # For now, return empty dict
        return {}

    async def _run_post_install_tests(
        self,
        metadata: SkillMetadata,
        install_path: Path
    ) -> Dict[str, Any]:
        """
        Run post-install tests.

        Args:
            metadata: Skill metadata
            install_path: Installation path

        Returns:
            Test results
        """
        # TODO: Implement proper testing
        # For now, just check if files exist
        if not install_path.exists():
            return {
                'success': False,
                'errors': [f"Install path does not exist: {install_path}"]
            }

        return {
            'success': True,
            'warnings': []
        }

    async def _backup_skill(self, skill_id: str) -> Optional[Path]:
        """
        Backup a skill.

        Args:
            skill_id: Skill ID to backup

        Returns:
            Backup path or None if failed
        """
        if skill_id not in self.installed_skills:
            return None

        install_path = self.installed_skills[skill_id]
        if not install_path.exists():
            return None

        # Create backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{skill_id.replace(':', '_')}_{timestamp}"
        backup_path = self.backup_root / backup_name

        try:
            if install_path.is_file():
                backup_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(install_path, backup_path / install_path.name)
            else:
                shutil.copytree(install_path, backup_path)

            logger.info(f"Backed up {skill_id} to {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to backup {skill_id}: {e}")
            return None

    async def _restore_backup(self, skill_id: str, backup_path: Path) -> bool:
        """
        Restore from backup.

        Args:
            skill_id: Skill ID to restore
            backup_path: Backup path

        Returns:
            True if restored
        """
        if not backup_path.exists():
            return False

        # Get install path
        metadata = await self.catalog.get_skill(skill_id)
        if not metadata:
            return False

        if metadata.category == SkillCategory.META:
            install_path = self.skills_root / "meta" / metadata.name
        elif metadata.category == SkillCategory.DOMAIN:
            install_path = self.skills_root / "domain" / metadata.domain / metadata.name
        else:
            install_path = self.skills_root / "agentic" / metadata.name

        try:
            # Remove current installation
            if install_path.exists():
                if install_path.is_file():
                    install_path.unlink()
                else:
                    shutil.rmtree(install_path)

            # Restore from backup
            if backup_path.is_file():
                install_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, install_path)
            else:
                shutil.copytree(backup_path, install_path)

            # Track installation
            self.installed_skills[skill_id] = install_path

            logger.info(f"Restored {skill_id} from backup")
            return True

        except Exception as e:
            logger.error(f"Failed to restore {skill_id}: {e}")
            return False

    async def _rollback_install(self, install_path: Path) -> bool:
        """
        Rollback a failed installation.

        Args:
            install_path: Installation path to remove

        Returns:
            True if rolled back
        """
        try:
            if install_path.exists():
                if install_path.is_file():
                    install_path.unlink()
                else:
                    shutil.rmtree(install_path)
            return True
        except Exception as e:
            logger.error(f"Failed to rollback install: {e}")
            return False

    async def _find_dependents(self, skill_id: str) -> List[str]:
        """
        Find skills that depend on given skill.

        Args:
            skill_id: Skill ID to check

        Returns:
            List of dependent skill IDs
        """
        dependents = []

        # Check all installed skills
        for installed_id in self.installed_skills:
            metadata = await self.catalog.get_skill(installed_id)
            if metadata and skill_id in metadata.dependencies:
                dependents.append(installed_id)

        return dependents


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'InstallResult',
    'UpgradeResult',
    'RemoveResult',
    'PackageManager',
]
