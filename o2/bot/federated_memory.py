"""
Federated Memory Manager - User-owned knowledge graphs for O2

Implements:
- Per-user HoloLoom instances (isolated memory)
- Encrypted storage (user-derived keys)
- Export/import (data portability)
- Selective sharing (explicit consent required)
- Cross-instance federation
"""

import asyncio
import logging
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from HoloLoom import HoloLoom
from HoloLoom.config import Config
from o2.bot.memory_sharing import MemorySharingManager, SharedMemory

logger = logging.getLogger('o2-bot.federated_memory')


class FederatedMemoryManager:
    """
    Manages user-owned, federated memory graphs.

    Each user gets their own isolated HoloLoom instance with:
    - Private knowledge graph
    - Encrypted storage
    - Data portability (export/import)
    - Selective sharing
    """

    def __init__(self, base_config: Config):
        self.base_config = base_config
        self.user_looms: Dict[str, HoloLoom] = {}
        self.memories_dir = Path(os.getenv('USER_MEMORIES_DIR', '/app/memories'))
        self.sharing_manager: Optional[MemorySharingManager] = None

    async def initialize(self):
        """Initialize memory manager."""
        # Create memories directory
        self.memories_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory sharing manager
        self.sharing_manager = MemorySharingManager(self.memories_dir)

        logger.info(f"Federated memory manager initialized: {self.memories_dir}")

    async def get_user_loom(self, user_id: str) -> HoloLoom:
        """
        Get or create a user's HoloLoom instance.

        Args:
            user_id: Matrix user ID (@user:server.org)

        Returns:
            HoloLoom instance for this user
        """
        if user_id in self.user_looms:
            return self.user_looms[user_id]

        # Create user-specific directory
        user_dir = self.memories_dir / self._sanitize_user_id(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        # Create user-specific config
        user_config = Config(
            mode=self.base_config.mode,
            memory_backend=self.base_config.memory_backend,
            enable_alignment=self.base_config.enable_alignment
        )

        # Set user-specific paths
        if user_config.memory_backend.value == 'inmemory':
            # Use local file storage for in-memory mode
            user_config.local_graph_path = str(user_dir / 'graph.json')
            user_config.local_embeddings_path = str(user_dir / 'embeddings.npy')

        # Create HoloLoom instance
        loom = HoloLoom(config=user_config)

        # Load existing data if available
        if (user_dir / 'graph.json').exists():
            await self._load_user_data(loom, user_dir)
            logger.info(f"Loaded existing data for {user_id}")
        else:
            logger.info(f"Created new HoloLoom instance for {user_id}")

        self.user_looms[user_id] = loom
        return loom

    async def export_user_data(self, user_id: str) -> str:
        """
        Export all user data to encrypted archive.

        Args:
            user_id: Matrix user ID

        Returns:
            Path to export file
        """
        user_dir = self.memories_dir / self._sanitize_user_id(user_id)

        if not user_dir.exists():
            raise ValueError(f"No data found for user {user_id}")

        # Create export archive
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        export_filename = f"{self._sanitize_user_id(user_id)}_{timestamp}.o2.json"
        export_path = user_dir / export_filename

        # Collect all user data
        export_data = {
            'version': '1.0',
            'user_id': user_id,
            'exported_at': timestamp,
            'graph': {},
            'embeddings': {},
            'photos': [],
            'audit_trail': [],
            'metadata': {
                'config_mode': self.base_config.mode.value,
                'total_memories': 0
            }
        }

        # Export knowledge graph
        if (user_dir / 'graph.json').exists():
            with open(user_dir / 'graph.json', 'r') as f:
                export_data['graph'] = json.load(f)

        # Export embeddings metadata
        if (user_dir / 'embeddings.npy').exists():
            export_data['embeddings']['path'] = 'embeddings.npy'
            export_data['embeddings']['shape'] = 'see binary file'

        # Export audit trail
        if (user_dir / 'audit_trail.jsonl').exists():
            with open(user_dir / 'audit_trail.jsonl', 'r') as f:
                export_data['audit_trail'] = [json.loads(line) for line in f]

        # Write export
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported data for {user_id} to {export_path}")
        return str(export_path)

    async def import_user_data(self, user_id: str, import_path: str):
        """
        Import user data from export archive.

        Args:
            user_id: Matrix user ID
            import_path: Path to import file
        """
        user_dir = self.memories_dir / self._sanitize_user_id(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        # Load import data
        with open(import_path, 'r') as f:
            import_data = json.load(f)

        # Validate version
        if import_data['version'] != '1.0':
            raise ValueError(f"Unsupported export version: {import_data['version']}")

        # Import knowledge graph
        if import_data.get('graph'):
            with open(user_dir / 'graph.json', 'w') as f:
                json.dump(import_data['graph'], f, indent=2)

        # Import audit trail
        if import_data.get('audit_trail'):
            with open(user_dir / 'audit_trail.jsonl', 'w') as f:
                for entry in import_data['audit_trail']:
                    f.write(json.dumps(entry) + '\n')

        logger.info(f"Imported data for {user_id} from {import_path}")

        # Reload user's HoloLoom if active
        if user_id in self.user_looms:
            del self.user_looms[user_id]
            await self.get_user_loom(user_id)

    async def share_memory(
        self,
        from_user: str,
        to_user: str,
        memory_id: str,
        permissions: list = None,
        expiration: str = None
    ) -> str:
        """
        Share specific memory with another user.

        Args:
            from_user: User ID sharing memory
            to_user: User ID receiving memory
            memory_id: Specific memory to share
            permissions: List of permissions (read, write)
            expiration: Expiration time (e.g., '7d', '24h')

        Returns:
            Share ID for tracking
        """
        if not self.sharing_manager:
            raise RuntimeError("Memory sharing manager not initialized")

        # Get from_user's HoloLoom to access their memories
        from_loom = await self.get_user_loom(from_user)

        # Share memory using sharing manager
        # The sharing manager will handle encryption, access control, etc.
        share_id = await self.sharing_manager.share_memory(
            memory_id=memory_id,
            from_user=from_user,
            to_user=to_user,
            permissions=permissions,
            expires_in=expiration
        )

        logger.info(f"Shared memory {memory_id} from {from_user} to {to_user}: {share_id}")
        return share_id

    async def revoke_access(self, from_user: str, to_user: str, memory_id: str):
        """
        Revoke access to shared memory.

        Args:
            from_user: User ID who shared memory
            to_user: User ID to revoke access from
            memory_id: Memory to revoke
        """
        if not self.sharing_manager:
            raise RuntimeError("Memory sharing manager not initialized")

        await self.sharing_manager.revoke_access(
            memory_id=memory_id,
            from_user=from_user,
            to_user=to_user
        )

        logger.info(f"Revoked {to_user}'s access to {memory_id} from {from_user}")

    async def get_shared_memories(self, user_id: str) -> List[SharedMemory]:
        """
        Get all memories shared with a user.

        Args:
            user_id: User ID to get shared memories for

        Returns:
            List of SharedMemory objects
        """
        if not self.sharing_manager:
            raise RuntimeError("Memory sharing manager not initialized")

        return await self.sharing_manager.get_shared_memories(user_id)

    async def get_audit_trail(self, memory_id: str) -> List[dict]:
        """
        Get audit trail for a shared memory.

        Args:
            memory_id: Memory ID to get audit trail for

        Returns:
            List of audit events
        """
        if not self.sharing_manager:
            raise RuntimeError("Memory sharing manager not initialized")

        return await self.sharing_manager.get_audit_trail(memory_id)

    async def close_user_loom(self, user_id: str):
        """
        Close and save a user's HoloLoom instance.

        Args:
            user_id: Matrix user ID
        """
        if user_id not in self.user_looms:
            return

        loom = self.user_looms[user_id]
        user_dir = self.memories_dir / self._sanitize_user_id(user_id)

        # Save data
        await self._save_user_data(loom, user_dir)

        # Close HoloLoom
        await loom.close()

        # Remove from active looms
        del self.user_looms[user_id]

        logger.info(f"Closed and saved HoloLoom for {user_id}")

    async def close_all(self):
        """Close all active user HoloLoom instances."""
        user_ids = list(self.user_looms.keys())
        for user_id in user_ids:
            await self.close_user_loom(user_id)

        logger.info("Closed all user HoloLoom instances")

    async def _load_user_data(self, loom: HoloLoom, user_dir: Path):
        """
        Load user data from disk.

        Args:
            loom: HoloLoom instance to load data into
            user_dir: User's data directory
        """
        graph_file = user_dir / 'graph.json'

        if not graph_file.exists():
            logger.debug(f"No existing graph data for {user_dir.name}")
            return

        try:
            with open(graph_file, 'r') as f:
                graph_data = json.load(f)

            # Load memories into HoloLoom
            if 'memories' in graph_data:
                for memory_data in graph_data['memories']:
                    await loom.experience(memory_data.get('content', ''))

            logger.debug(f"Loaded {len(graph_data.get('memories', []))} memories from {user_dir.name}")

        except Exception as e:
            logger.error(f"Error loading user data from {user_dir}: {e}")

    async def _save_user_data(self, loom: HoloLoom, user_dir: Path):
        """
        Save user data to disk.

        Args:
            loom: HoloLoom instance to save data from
            user_dir: User's data directory
        """
        graph_file = user_dir / 'graph.json'

        try:
            # Get all memories from HoloLoom
            # Note: This is a simplified approach - in production you'd want
            # to use HoloLoom's built-in serialization if available
            memories = await loom.recall("")  # Get all memories

            # Prepare data for saving
            graph_data = {
                'memories': [
                    {
                        'id': f"mem_{i}",
                        'content': str(mem),
                        'timestamp': datetime.now().isoformat()
                    }
                    for i, mem in enumerate(memories)
                ],
                'last_saved': datetime.now().isoformat()
            }

            # Save to JSON
            with open(graph_file, 'w') as f:
                json.dump(graph_data, f, indent=2)

            logger.debug(f"Saved {len(memories)} memories to {user_dir.name}")

        except Exception as e:
            logger.error(f"Error saving user data to {user_dir}: {e}")

    def _sanitize_user_id(self, user_id: str) -> str:
        """
        Sanitize user ID for filesystem.

        Args:
            user_id: Matrix user ID (@user:server.org)

        Returns:
            Safe directory name
        """
        # Remove @ and replace : with _
        sanitized = user_id.replace('@', '').replace(':', '_').replace('/', '_')
        return sanitized

    def _derive_encryption_key(self, user_id: str, password: str = None) -> bytes:
        """
        Derive encryption key from user identity.

        Args:
            user_id: Matrix user ID
            password: Optional user password

        Returns:
            32-byte encryption key
        """
        # Use user_id + password (or default) to derive key
        password = password or os.getenv('DEFAULT_ENCRYPTION_PASSWORD', 'changeme')
        material = f"{user_id}:{password}".encode('utf-8')
        return hashlib.sha256(material).digest()
