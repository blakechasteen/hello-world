"""
Hash Chain Implementation for Forensic Logging

Implements cryptographic hash chaining to detect tampering.

**Added: 2025-11-16** - Phase 4 Security Pipeline

Algorithm:
    Entry 1: hash(genesis_block)
    Entry 2: hash(entry_1_data + hash_1)
    Entry 3: hash(entry_2_data + hash_2)
    ...
    Entry N: hash(entry_N-1_data + hash_N-1)

Any tampering breaks the chain, enabling detection.
"""

import hashlib
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class HashChainEntry:
    """Single entry in the hash chain."""
    entry_id: str
    timestamp: str  # ISO 8601
    data: Dict[str, Any]
    previous_hash: str
    current_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HashChainEntry':
        """Create from dictionary."""
        return cls(**data)


class HashChain:
    """
    Cryptographic hash chain for tamper detection.

    Each entry contains a hash of the previous entry's data and hash,
    creating a chain that breaks if any entry is modified.

    **Added: 2025-11-16**

    Features:
    - SHA-256 hashing
    - Genesis block initialization
    - Automatic chain linking
    - Tamper detection
    - Fast verification
    """

    def __init__(self, genesis_data: Optional[Dict[str, Any]] = None):
        """
        Initialize hash chain.

        Args:
            genesis_data: Optional data for genesis block
        """
        self.chain: list[HashChainEntry] = []
        self._create_genesis_block(genesis_data or {})

    def _create_genesis_block(self, data: Dict[str, Any]) -> None:
        """Create the first block in the chain."""
        genesis_data = {
            **data,
            'type': 'genesis',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        genesis_hash = self._compute_hash("0" * 64, genesis_data)

        genesis_entry = HashChainEntry(
            entry_id="genesis",
            timestamp=genesis_data['timestamp'],
            data=genesis_data,
            previous_hash="0" * 64,
            current_hash=genesis_hash
        )

        self.chain.append(genesis_entry)

    def _compute_hash(self, previous_hash: str, data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of previous hash + data.

        Args:
            previous_hash: Hash of previous entry
            data: Current entry data

        Returns:
            Hex-encoded SHA-256 hash
        """
        # Canonical JSON representation (sorted keys)
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))

        # Combine previous hash + data
        combined = previous_hash + data_str

        # SHA-256 hash
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def add_entry(
        self,
        entry_id: str,
        timestamp: str,
        data: Dict[str, Any]
    ) -> HashChainEntry:
        """
        Add new entry to the chain.

        Args:
            entry_id: Unique entry identifier (UUID)
            timestamp: ISO 8601 timestamp
            data: Entry data (must be JSON-serializable)

        Returns:
            Created hash chain entry

        Raises:
            ValueError: If entry_id already exists
        """
        # Check for duplicate entry_id
        if any(e.entry_id == entry_id for e in self.chain):
            raise ValueError(f"Entry ID {entry_id} already exists in chain")

        # Get previous entry
        previous_entry = self.chain[-1]

        # Compute current hash
        current_hash = self._compute_hash(previous_entry.current_hash, data)

        # Create new entry
        new_entry = HashChainEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            data=data,
            previous_hash=previous_entry.current_hash,
            current_hash=current_hash
        )

        self.chain.append(new_entry)
        return new_entry

    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        """
        Verify entire chain integrity.

        Returns:
            (is_valid, first_invalid_index)
            - is_valid: True if chain is intact
            - first_invalid_index: Index of first tampered entry (None if valid)

        Performance: O(n) where n is chain length
        """
        if not self.chain:
            return True, None

        # Verify genesis block
        genesis = self.chain[0]
        expected_genesis_hash = self._compute_hash("0" * 64, genesis.data)
        if genesis.current_hash != expected_genesis_hash:
            return False, 0

        # Verify each subsequent entry
        for i in range(1, len(self.chain)):
            entry = self.chain[i]
            previous_entry = self.chain[i - 1]

            # Check previous hash matches
            if entry.previous_hash != previous_entry.current_hash:
                return False, i

            # Recompute hash
            expected_hash = self._compute_hash(entry.previous_hash, entry.data)
            if entry.current_hash != expected_hash:
                return False, i

        return True, None

    def get_entry(self, entry_id: str) -> Optional[HashChainEntry]:
        """Get entry by ID."""
        for entry in self.chain:
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_latest_hash(self) -> str:
        """Get hash of most recent entry."""
        return self.chain[-1].current_hash if self.chain else "0" * 64

    def __len__(self) -> int:
        """Return number of entries in chain."""
        return len(self.chain)

    def to_json(self) -> list[Dict[str, Any]]:
        """Export chain as JSON-serializable list."""
        return [entry.to_dict() for entry in self.chain]

    @classmethod
    def from_json(cls, data: list[Dict[str, Any]]) -> 'HashChain':
        """
        Import chain from JSON.

        Args:
            data: List of entry dictionaries

        Returns:
            Reconstructed hash chain

        Raises:
            ValueError: If chain is invalid
        """
        if not data:
            raise ValueError("Cannot create chain from empty data")

        # Create empty chain (will be replaced)
        chain = cls.__new__(cls)
        chain.chain = [HashChainEntry.from_dict(entry) for entry in data]

        # Verify integrity
        is_valid, invalid_idx = chain.verify_integrity()
        if not is_valid:
            raise ValueError(f"Invalid chain: tampering detected at index {invalid_idx}")

        return chain


def create_hash_chain(genesis_data: Optional[Dict[str, Any]] = None) -> HashChain:
    """
    Factory function to create a new hash chain.

    Args:
        genesis_data: Optional data for genesis block

    Returns:
        Initialized hash chain
    """
    return HashChain(genesis_data)
