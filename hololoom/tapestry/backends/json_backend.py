"""
JSON Tapestry Backend

Atomic JSON file persistence for Tapestry state.

Features:
- Atomic writes via temp file + rename
- Backup before overwrite
- Human-readable JSON output
- Graceful degradation

Pattern follows HoloLoom/tuning/persistence.py

Created: December 2025
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Optional
import os

from HoloLoom.tapestry.protocol import Tapestry, TapestryBackend

logger = logging.getLogger(__name__)


class JsonTapestryBackend:
    """
    JSON file backend with atomic writes.

    Follows pattern from tuning/persistence.py:
    - Write to temp file
    - Atomic rename
    - Backup before write

    Default path: .hololoom/tapestry.json (project-specific)
    """

    def __init__(self, path: str = ".hololoom/tapestry.json"):
        """
        Initialize JSON backend.

        Args:
            path: Path to tapestry JSON file.
                  Defaults to .hololoom/tapestry.json
        """
        self.path = Path(path)
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def exists(self) -> bool:
        """Check if tapestry file exists."""
        return self.path.exists()

    async def load(self) -> Optional[Tapestry]:
        """
        Load tapestry from JSON file.

        Returns:
            Tapestry if file exists and is valid, None otherwise.
        """
        if not self.path.exists():
            logger.debug(f"No tapestry file at {self.path}")
            return None

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tapestry = Tapestry.from_dict(data)
            logger.info(f"Loaded tapestry '{tapestry.loom_id}' with {len(tapestry.threads)} threads")
            return tapestry
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in tapestry file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load tapestry: {e}")
            return None

    async def save(self, tapestry: Tapestry) -> None:
        """
        Save tapestry atomically with backup.

        Steps:
        1. Ensure directory exists
        2. Backup existing file (if any)
        3. Write to temp file
        4. Atomic rename temp -> target

        This ensures we never have a partially written file.
        """
        self._ensure_dir()

        # Backup existing file
        if self.path.exists():
            backup_path = self.path.with_suffix('.json.bak')
            try:
                shutil.copy2(self.path, backup_path)
                logger.debug(f"Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Write to temp file
        temp_path = self.path.with_suffix('.json.tmp')
        try:
            data = tapestry.to_dict()
            content = json.dumps(data, indent=2, ensure_ascii=False)

            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Atomic rename (on Windows, need to remove target first)
            if os.name == 'nt' and self.path.exists():
                self.path.unlink()

            temp_path.rename(self.path)
            logger.info(f"Saved tapestry '{tapestry.loom_id}' to {self.path}")

        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to save tapestry: {e}")
            raise

    async def delete(self) -> None:
        """Delete tapestry file."""
        if self.path.exists():
            # Keep backup before delete
            backup_path = self.path.with_suffix('.json.deleted')
            try:
                shutil.copy2(self.path, backup_path)
            except Exception:
                pass  # Best effort backup

            self.path.unlink()
            logger.info(f"Deleted tapestry at {self.path}")

    def get_backup_path(self) -> Optional[Path]:
        """Get path to most recent backup, if exists."""
        backup = self.path.with_suffix('.json.bak')
        if backup.exists():
            return backup
        return None

    async def restore_from_backup(self) -> bool:
        """
        Restore tapestry from backup file.

        Returns:
            True if restored successfully, False otherwise.
        """
        backup = self.get_backup_path()
        if not backup:
            logger.warning("No backup file found")
            return False

        try:
            shutil.copy2(backup, self.path)
            logger.info(f"Restored tapestry from {backup}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False


# Verify protocol compliance
assert isinstance(JsonTapestryBackend(".test"), TapestryBackend), \
    "JsonTapestryBackend must implement TapestryBackend protocol"
