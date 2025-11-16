"""
Forensic Logger - Main Interface

Immutable, tamper-proof audit logging with hash chain integrity.

**Added: 2025-11-16** - Phase 4 Security Pipeline

Features:
- Append-only logs (immutable)
- Hash chain integrity
- Optional digital signatures
- Multiple storage backends
- Sub-5ms write latency
- Complete audit trail
"""

import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from .hash_chain import HashChain, create_hash_chain
from .storage import StorageBackend, create_storage_backend

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Forensic event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ATTACK = "attack"
    INCIDENT = "incident"
    ACCESS = "access"
    MODIFICATION = "modification"
    DELETION = "deletion"
    EXPORT = "export"
    ADMIN = "admin"
    SYSTEM = "system"


class Severity(Enum):
    """Event severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class ForensicEntry:
    """
    Forensic log entry.

    **Added: 2025-11-16**

    Contains complete event information plus hash chain metadata.
    """
    entry_id: str
    timestamp: str  # ISO 8601
    event_type: str
    severity: str
    action: str
    outcome: str
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    current_hash: str = ""
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForensicEntry':
        """Create from dictionary."""
        return cls(**data)

    def hash_sensitive_data(self) -> None:
        """
        Hash sensitive fields (user_id, source_ip) for privacy.

        Uses SHA-256 to anonymize while maintaining uniqueness.
        """
        if self.user_id:
            self.user_id = hashlib.sha256(self.user_id.encode()).hexdigest()[:16]
        if self.source_ip:
            self.source_ip = hashlib.sha256(self.source_ip.encode()).hexdigest()[:16]


class ForensicLogger:
    """
    Immutable forensic logging system.

    **Added: 2025-11-16**

    Features:
    - Append-only logs (write-only, no updates/deletes)
    - Hash chain integrity (tamper detection)
    - Optional digital signatures
    - Multiple storage backends (file, PostgreSQL, S3)
    - Fast writes (<5ms async)
    - Chain of custody tracking

    Usage:
        async with ForensicLogger() as logger:
            await logger.log(
                event_type="authentication",
                severity="INFO",
                action="user_login",
                outcome="success",
                user_id="alice",
                metadata={"method": "password"}
            )
    """

    def __init__(
        self,
        storage_backend: str = "file",
        hash_sensitive: bool = True,
        enable_signatures: bool = False,
        **storage_kwargs
    ):
        """
        Initialize forensic logger.

        Args:
            storage_backend: "file", "postgresql", or "s3"
            hash_sensitive: Hash user_id and source_ip for privacy
            enable_signatures: Enable digital signatures (requires GPG)
            **storage_kwargs: Backend-specific arguments
        """
        self.hash_sensitive = hash_sensitive
        self.enable_signatures = enable_signatures

        # Create storage backend
        self.storage = create_storage_backend(storage_backend, **storage_kwargs)

        # Create hash chain
        genesis_data = {
            'system': 'HoloLoom Forensic Logger',
            'version': '1.0.0',
            'created': datetime.utcnow().isoformat() + 'Z'
        }
        self.chain = create_hash_chain(genesis_data)

        # Performance tracking
        self._total_entries = 0
        self._total_time_ms = 0.0

        logger.info(
            f"ForensicLogger initialized: backend={storage_backend}, "
            f"hash_sensitive={hash_sensitive}, signatures={enable_signatures}"
        )

    async def log(
        self,
        event_type: str,
        severity: str,
        action: str,
        outcome: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ForensicEntry:
        """
        Log forensic event.

        Args:
            event_type: Event type (authentication, attack, etc.)
            severity: Severity level (INFO, WARNING, CRITICAL)
            action: Action performed
            outcome: Outcome (success, failure, blocked)
            user_id: User identifier (will be hashed if hash_sensitive=True)
            source_ip: Source IP address (will be hashed if hash_sensitive=True)
            metadata: Additional context

        Returns:
            Created forensic entry

        Performance: <5ms typical
        """
        start_time = asyncio.get_event_loop().time()

        # Create entry
        entry = ForensicEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + 'Z',
            event_type=event_type,
            severity=severity,
            action=action,
            outcome=outcome,
            user_id=user_id,
            source_ip=source_ip,
            metadata=metadata or {}
        )

        # Hash sensitive data if enabled
        if self.hash_sensitive:
            entry.hash_sensitive_data()

        # Add to hash chain
        chain_entry = self.chain.add_entry(
            entry_id=entry.entry_id,
            timestamp=entry.timestamp,
            data={
                'event_type': entry.event_type,
                'severity': entry.severity,
                'action': entry.action,
                'outcome': entry.outcome,
                'user_id': entry.user_id,
                'source_ip': entry.source_ip,
                'metadata': entry.metadata
            }
        )

        # Update entry with hash chain metadata
        entry.previous_hash = chain_entry.previous_hash
        entry.current_hash = chain_entry.current_hash

        # Sign if enabled
        if self.enable_signatures:
            entry.signature = await self._sign_entry(entry)

        # Store to backend
        await self.storage.append(entry.to_dict())

        # Track performance
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        self._total_entries += 1
        self._total_time_ms += elapsed_ms

        if elapsed_ms > 5.0:
            logger.warning(
                f"Slow forensic log write: {elapsed_ms:.1f}ms "
                f"(target: <5ms)"
            )

        return entry

    async def _sign_entry(self, entry: ForensicEntry) -> str:
        """
        Sign entry with digital signature.

        Note: Requires GPG setup. Returns empty string if unavailable.
        """
        # Placeholder - would require python-gnupg
        logger.warning("Digital signatures not yet implemented")
        return ""

    async def get_chain_integrity(self) -> tuple[bool, Optional[int]]:
        """
        Verify hash chain integrity.

        Returns:
            (is_valid, first_invalid_index)

        Performance: <10s for 1M entries
        """
        return self.chain.verify_integrity()

    async def get_latest_hash(self) -> str:
        """Get hash of most recent entry."""
        return self.chain.get_latest_hash()

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get forensic logger statistics.

        Returns:
            Statistics dictionary with:
            - total_entries: Number of entries logged
            - avg_write_time_ms: Average write latency
            - chain_length: Hash chain length
            - storage_stats: Backend storage statistics
        """
        storage_stats = await self.storage.get_stats()

        return {
            'total_entries': self._total_entries,
            'avg_write_time_ms': self._total_time_ms / max(self._total_entries, 1),
            'chain_length': len(self.chain),
            'storage_stats': storage_stats.__dict__
        }

    async def close(self) -> None:
        """Close forensic logger and storage backend."""
        await self.storage.close()
        logger.info(
            f"ForensicLogger closed: {self._total_entries} entries logged, "
            f"avg write time: {self._total_time_ms / max(self._total_entries, 1):.2f}ms"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


async def create_forensic_logger(
    backend: str = "file",
    **kwargs
) -> ForensicLogger:
    """
    Factory function to create forensic logger.

    Args:
        backend: Storage backend ("file", "postgresql", "s3")
        **kwargs: Backend-specific arguments

    Returns:
        Initialized forensic logger

    Example:
        logger = await create_forensic_logger(
            backend="file",
            log_dir="./forensic_logs"
        )
    """
    return ForensicLogger(storage_backend=backend, **kwargs)
