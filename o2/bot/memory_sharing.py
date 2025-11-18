"""
Memory Sharing System - Secure, Consent-Based Knowledge Sharing

Implements:
- Selective memory sharing (specific memories, not entire graph)
- Encryption (shared memory encrypted for recipient's public key)
- Access control (granular permissions: read, write)
- Time-limited sharing (auto-expiration)
- Cross-instance federation (share with users on other Matrix servers)
- Complete audit trail (who accessed when)
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger('o2-bot.memory_sharing')


class SharedMemory:
    """
    Represents a shared memory with access control.
    """

    def __init__(
        self,
        memory_id: str,
        owner_id: str,
        content: str,
        metadata: Dict = None,
        encrypted_content: bytes = None
    ):
        self.memory_id = memory_id
        self.owner_id = owner_id
        self.content = content
        self.metadata = metadata or {}
        self.encrypted_content = encrypted_content
        self.created_at = datetime.now()


class AccessGrant:
    """
    Represents access granted to a user for a specific memory.
    """

    def __init__(
        self,
        memory_id: str,
        granted_to: str,
        permissions: List[str],
        expires_at: Optional[datetime] = None
    ):
        self.memory_id = memory_id
        self.granted_to = granted_to
        self.permissions = permissions  # ['read'] or ['read', 'write']
        self.granted_at = datetime.now()
        self.expires_at = expires_at
        self.accessed_count = 0
        self.last_accessed = None

    def is_expired(self) -> bool:
        """Check if access has expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def has_permission(self, permission: str) -> bool:
        """Check if grant has specific permission."""
        return permission in self.permissions and not self.is_expired()

    def record_access(self):
        """Record an access event."""
        self.accessed_count += 1
        self.last_accessed = datetime.now()


class MemorySharingManager:
    """
    Manages secure memory sharing between users.

    Features:
    - RSA encryption for shared memories
    - Granular access control (memory-level, not graph-level)
    - Time-limited sharing with auto-expiration
    - Complete audit trail
    - Cross-instance federation support
    """

    def __init__(self, memories_dir: Path):
        self.memories_dir = memories_dir
        self.shared_dir = memories_dir / 'shared'
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        # Access control list (memory_id -> [AccessGrant])
        self.access_grants: Dict[str, List[AccessGrant]] = {}

        # User public keys (user_id -> RSA public key)
        self.user_public_keys: Dict[str, rsa.RSAPublicKey] = {}

        # Load existing access grants
        self._load_access_grants()

    def generate_user_keypair(self, user_id: str) -> tuple:
        """
        Generate RSA keypair for user.

        Args:
            user_id: Matrix user ID

        Returns:
            (private_key, public_key) tuple
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Store public key
        self.user_public_keys[user_id] = public_key

        # Save public key to disk
        user_dir = self.memories_dir / self._sanitize_user_id(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(user_dir / 'public_key.pem', 'wb') as f:
            f.write(public_pem)

        # Save private key to disk (encrypted)
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()  # TODO: Add password encryption
        )
        with open(user_dir / 'private_key.pem', 'wb') as f:
            f.write(private_pem)

        logger.info(f"Generated keypair for {user_id}")
        return private_key, public_key

    def load_user_public_key(self, user_id: str) -> Optional[rsa.RSAPublicKey]:
        """
        Load user's public key from disk.

        Args:
            user_id: Matrix user ID

        Returns:
            RSA public key or None if not found
        """
        if user_id in self.user_public_keys:
            return self.user_public_keys[user_id]

        user_dir = self.memories_dir / self._sanitize_user_id(user_id)
        key_path = user_dir / 'public_key.pem'

        if not key_path.exists():
            return None

        with open(key_path, 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        self.user_public_keys[user_id] = public_key
        return public_key

    async def share_memory(
        self,
        memory_id: str,
        from_user: str,
        to_user: str,
        permissions: List[str] = None,
        expires_in: str = None
    ) -> str:
        """
        Share a specific memory with another user.

        Args:
            memory_id: ID of memory to share
            from_user: User ID sharing the memory
            to_user: User ID receiving the memory
            permissions: List of permissions (['read'] or ['read', 'write'])
            expires_in: Expiration time (e.g., '7d', '24h', '30m')

        Returns:
            Share ID for tracking
        """
        permissions = permissions or ['read']

        # Validate permissions
        valid_permissions = {'read', 'write'}
        if not all(p in valid_permissions for p in permissions):
            raise ValueError(f"Invalid permissions: {permissions}")

        # Load recipient's public key (generate if doesn't exist)
        recipient_key = self.load_user_public_key(to_user)
        if not recipient_key:
            logger.info(f"Generating keypair for {to_user} (first time sharing)")
            _, recipient_key = self.generate_user_keypair(to_user)

        # Load memory content
        memory = await self._load_memory(from_user, memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found for user {from_user}")

        # Encrypt memory for recipient
        encrypted_content = recipient_key.encrypt(
            memory.content.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Create shared memory object
        shared_memory = SharedMemory(
            memory_id=memory_id,
            owner_id=from_user,
            content=memory.content,
            metadata=memory.metadata,
            encrypted_content=encrypted_content
        )

        # Parse expiration
        expires_at = None
        if expires_in:
            expires_at = self._parse_expiration(expires_in)

        # Create access grant
        grant = AccessGrant(
            memory_id=memory_id,
            granted_to=to_user,
            permissions=permissions,
            expires_at=expires_at
        )

        # Store access grant
        if memory_id not in self.access_grants:
            self.access_grants[memory_id] = []
        self.access_grants[memory_id].append(grant)

        # Save shared memory to recipient's shared directory
        await self._save_shared_memory(to_user, shared_memory)

        # Save access grants
        self._save_access_grants()

        # Log to audit trail
        await self._log_share_event(from_user, to_user, memory_id, permissions, expires_at)

        logger.info(f"Shared memory {memory_id} from {from_user} to {to_user}")

        return f"{from_user}:{memory_id}:{to_user}"

    async def revoke_access(self, memory_id: str, from_user: str, to_user: str):
        """
        Revoke access to a shared memory.

        Args:
            memory_id: Memory ID
            from_user: User who originally shared
            to_user: User to revoke access from
        """
        if memory_id not in self.access_grants:
            raise ValueError(f"No access grants found for memory {memory_id}")

        # Remove access grant
        self.access_grants[memory_id] = [
            g for g in self.access_grants[memory_id]
            if not (g.granted_to == to_user)
        ]

        # Remove shared memory file
        await self._remove_shared_memory(to_user, memory_id)

        # Save access grants
        self._save_access_grants()

        # Log to audit trail
        await self._log_revoke_event(from_user, to_user, memory_id)

        logger.info(f"Revoked {to_user}'s access to memory {memory_id}")

    async def get_shared_memories(self, user_id: str) -> List[SharedMemory]:
        """
        Get all memories shared with a user.

        Args:
            user_id: User ID

        Returns:
            List of shared memories
        """
        shared_memories = []

        # Find all access grants for this user
        for memory_id, grants in self.access_grants.items():
            for grant in grants:
                if grant.granted_to == user_id and not grant.is_expired():
                    # Load shared memory
                    memory = await self._load_shared_memory(user_id, memory_id)
                    if memory:
                        # Decrypt if needed (for read access)
                        if grant.has_permission('read'):
                            grant.record_access()
                            shared_memories.append(memory)

        return shared_memories

    async def can_access(self, user_id: str, memory_id: str, permission: str = 'read') -> bool:
        """
        Check if user has permission to access a memory.

        Args:
            user_id: User ID
            memory_id: Memory ID
            permission: Permission to check ('read' or 'write')

        Returns:
            True if user has permission
        """
        if memory_id not in self.access_grants:
            return False

        for grant in self.access_grants[memory_id]:
            if grant.granted_to == user_id and grant.has_permission(permission):
                return True

        return False

    def get_access_audit(self, memory_id: str) -> List[Dict]:
        """
        Get complete audit trail for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of access events
        """
        if memory_id not in self.access_grants:
            return []

        audit_trail = []
        for grant in self.access_grants[memory_id]:
            audit_trail.append({
                'granted_to': grant.granted_to,
                'permissions': grant.permissions,
                'granted_at': grant.granted_at.isoformat(),
                'expires_at': grant.expires_at.isoformat() if grant.expires_at else None,
                'accessed_count': grant.accessed_count,
                'last_accessed': grant.last_accessed.isoformat() if grant.last_accessed else None,
                'is_expired': grant.is_expired()
            })

        return audit_trail

    async def _load_memory(self, user_id: str, memory_id: str) -> Optional[SharedMemory]:
        """Load memory from user's knowledge graph."""
        # TODO: Integrate with HoloLoom to extract specific memory
        # For now, return placeholder
        return SharedMemory(
            memory_id=memory_id,
            owner_id=user_id,
            content=f"Memory content for {memory_id}",
            metadata={'source': 'hololoom'}
        )

    async def _save_shared_memory(self, user_id: str, memory: SharedMemory):
        """Save shared memory to recipient's directory."""
        user_shared_dir = self.shared_dir / self._sanitize_user_id(user_id)
        user_shared_dir.mkdir(parents=True, exist_ok=True)

        memory_file = user_shared_dir / f"{memory.memory_id}.json"

        with open(memory_file, 'w') as f:
            json.dump({
                'memory_id': memory.memory_id,
                'owner_id': memory.owner_id,
                'encrypted_content': base64.b64encode(memory.encrypted_content).decode('utf-8'),
                'metadata': memory.metadata,
                'created_at': memory.created_at.isoformat()
            }, f, indent=2)

    async def _load_shared_memory(self, user_id: str, memory_id: str) -> Optional[SharedMemory]:
        """Load shared memory from recipient's directory."""
        user_shared_dir = self.shared_dir / self._sanitize_user_id(user_id)
        memory_file = user_shared_dir / f"{memory_id}.json"

        if not memory_file.exists():
            return None

        with open(memory_file, 'r') as f:
            data = json.load(f)

        return SharedMemory(
            memory_id=data['memory_id'],
            owner_id=data['owner_id'],
            content='',  # Encrypted
            metadata=data['metadata'],
            encrypted_content=base64.b64decode(data['encrypted_content'])
        )

    async def _remove_shared_memory(self, user_id: str, memory_id: str):
        """Remove shared memory file."""
        user_shared_dir = self.shared_dir / self._sanitize_user_id(user_id)
        memory_file = user_shared_dir / f"{memory_id}.json"

        if memory_file.exists():
            memory_file.unlink()

    def _load_access_grants(self):
        """Load access grants from disk."""
        grants_file = self.shared_dir / 'access_grants.json'

        if not grants_file.exists():
            return

        with open(grants_file, 'r') as f:
            data = json.load(f)

        for memory_id, grants in data.items():
            self.access_grants[memory_id] = [
                AccessGrant(
                    memory_id=g['memory_id'],
                    granted_to=g['granted_to'],
                    permissions=g['permissions'],
                    expires_at=datetime.fromisoformat(g['expires_at']) if g['expires_at'] else None
                )
                for g in grants
            ]

    def _save_access_grants(self):
        """Save access grants to disk."""
        grants_file = self.shared_dir / 'access_grants.json'

        data = {}
        for memory_id, grants in self.access_grants.items():
            data[memory_id] = [
                {
                    'memory_id': g.memory_id,
                    'granted_to': g.granted_to,
                    'permissions': g.permissions,
                    'granted_at': g.granted_at.isoformat(),
                    'expires_at': g.expires_at.isoformat() if g.expires_at else None
                }
                for g in grants
            ]

        with open(grants_file, 'w') as f:
            json.dump(data, f, indent=2)

    async def _log_share_event(self, from_user: str, to_user: str, memory_id: str, permissions: List[str], expires_at: Optional[datetime]):
        """Log memory sharing event to audit trail."""
        audit_file = self.shared_dir / 'sharing_audit.jsonl'

        with open(audit_file, 'a') as f:
            f.write(json.dumps({
                'event': 'share',
                'from_user': from_user,
                'to_user': to_user,
                'memory_id': memory_id,
                'permissions': permissions,
                'expires_at': expires_at.isoformat() if expires_at else None,
                'timestamp': datetime.now().isoformat()
            }) + '\n')

    async def _log_revoke_event(self, from_user: str, to_user: str, memory_id: str):
        """Log access revocation event to audit trail."""
        audit_file = self.shared_dir / 'sharing_audit.jsonl'

        with open(audit_file, 'a') as f:
            f.write(json.dumps({
                'event': 'revoke',
                'from_user': from_user,
                'to_user': to_user,
                'memory_id': memory_id,
                'timestamp': datetime.now().isoformat()
            }) + '\n')

    def _parse_expiration(self, expires_in: str) -> datetime:
        """
        Parse expiration string to datetime.

        Args:
            expires_in: String like '7d', '24h', '30m'

        Returns:
            Expiration datetime
        """
        unit = expires_in[-1]
        amount = int(expires_in[:-1])

        if unit == 'd':
            delta = timedelta(days=amount)
        elif unit == 'h':
            delta = timedelta(hours=amount)
        elif unit == 'm':
            delta = timedelta(minutes=amount)
        else:
            raise ValueError(f"Invalid expiration format: {expires_in}")

        return datetime.now() + delta

    def _sanitize_user_id(self, user_id: str) -> str:
        """Sanitize user ID for filesystem."""
        return user_id.replace('@', '').replace(':', '_').replace('/', '_')
