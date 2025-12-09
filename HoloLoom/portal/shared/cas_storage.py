"""
Content-Addressed Storage (CAS) for Portal System.

Provides deduplicated storage of WASM modules and other binary content
using SHA256 hashing. Identical content is stored only once regardless
of how many modules reference it.

Usage:
    storage = CASStorage(storage_dir=Path("./cas"))

    # Store content
    content_hash = storage.store(wasm_bytes, module_id="my-module", version="v1.0.0")

    # Retrieve by reference
    content = storage.get("my-module", "v1.0.0")

    # Retrieve by hash
    content = storage.get_by_hash(content_hash)

    # Check deduplication
    stats = storage.get_stats()
    print(f"Deduplication ratio: {stats.deduplication_ratio:.1f}x")

Directory Structure:
    cas/
        objects/
            ab/cd1234...  # First 2 chars as prefix directory
        refs/
            {module_id}/
                {version}  # Text file containing content hash
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any
import threading


@dataclass
class CASEntry:
    """
    Entry in content-addressed storage.

    Tracks content location and reference count for garbage collection.
    """

    content_hash: str  # SHA256 hash (64 chars)
    size_bytes: int
    path: Path
    reference_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "path": str(self.path),
            "reference_count": self.reference_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
        }


@dataclass
class CASStats:
    """Statistics about content-addressed storage."""

    unique_objects: int = 0
    total_references: int = 0
    total_size_bytes: int = 0
    deduplication_ratio: float = 1.0
    objects_by_size: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "unique_objects": self.unique_objects,
            "total_references": self.total_references,
            "total_size_bytes": self.total_size_bytes,
            "deduplication_ratio": round(self.deduplication_ratio, 2),
            "size_distribution": self.objects_by_size,
        }


@dataclass
class CASReference:
    """Reference from module/version to content hash."""

    module_id: str
    version: str
    content_hash: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "module_id": self.module_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }


class CASStorage:
    """
    Content-addressed storage for WASM modules.

    Stores modules by SHA256 hash, deduplicating identical content.
    Each module version maintains a reference to the content hash.

    Directory structure:
        storage_dir/
            objects/
                ab/cd1234...  # First 2 chars as prefix directory
            refs/
                {module_id}/
                    {version}  # Text file containing content hash

    Thread-safe for concurrent access.

    Args:
        storage_dir: Root directory for CAS storage
        hash_prefix_length: Length of hash prefix for subdirectory (default: 2)
    """

    def __init__(
        self,
        storage_dir: Path,
        hash_prefix_length: int = 2,
    ):
        self.storage_dir = Path(storage_dir)
        self.objects_dir = self.storage_dir / "objects"
        self.refs_dir = self.storage_dir / "refs"
        self.hash_prefix_length = hash_prefix_length

        # Internal state
        self._entries: Dict[str, CASEntry] = {}
        self._references: Dict[str, Dict[str, CASReference]] = {}  # module_id -> version -> ref
        self._lock = threading.RLock()  # Reentrant lock to allow nested locking

        # Ensure directories exist
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)

        # Scan existing storage
        self._scan_existing()

    def _hash_content(self, content: bytes) -> str:
        """
        Compute SHA256 hash of content.

        Args:
            content: Binary content to hash

        Returns:
            64-character hexadecimal hash string
        """
        return hashlib.sha256(content).hexdigest()

    def _object_path(self, content_hash: str) -> Path:
        """
        Get filesystem path for object by hash.

        Uses first N characters as subdirectory prefix to avoid
        too many files in a single directory.

        Args:
            content_hash: SHA256 hash

        Returns:
            Path to object file
        """
        prefix = content_hash[:self.hash_prefix_length]
        suffix = content_hash[self.hash_prefix_length:]
        return self.objects_dir / prefix / suffix

    def _ref_path(self, module_id: str, version: str) -> Path:
        """
        Get filesystem path for reference file.

        Args:
            module_id: Module identifier
            version: Version string

        Returns:
            Path to reference file
        """
        return self.refs_dir / module_id / version

    def _scan_existing(self) -> None:
        """
        Scan existing storage and populate internal state.

        Called on initialization to rebuild state from filesystem.
        """
        # Scan objects
        for prefix_dir in self.objects_dir.iterdir():
            if prefix_dir.is_dir():
                for obj_file in prefix_dir.iterdir():
                    if obj_file.is_file():
                        content_hash = prefix_dir.name + obj_file.name
                        try:
                            size = obj_file.stat().st_size
                            self._entries[content_hash] = CASEntry(
                                content_hash=content_hash,
                                size_bytes=size,
                                path=obj_file,
                                reference_count=0,  # Will be updated when scanning refs
                            )
                        except OSError:
                            pass

        # Scan references
        for module_dir in self.refs_dir.iterdir():
            if module_dir.is_dir():
                module_id = module_dir.name
                self._references[module_id] = {}

                for version_file in module_dir.iterdir():
                    if version_file.is_file():
                        version = version_file.name
                        try:
                            content_hash = version_file.read_text().strip()
                            ref = CASReference(
                                module_id=module_id,
                                version=version,
                                content_hash=content_hash,
                            )
                            self._references[module_id][version] = ref

                            # Update reference count
                            if content_hash in self._entries:
                                self._entries[content_hash].reference_count += 1
                        except (OSError, IOError):
                            pass

    def store(
        self,
        content: bytes,
        module_id: str,
        version: str,
    ) -> str:
        """
        Store content with module reference.

        If content already exists, increments reference count.
        Otherwise stores new content and creates object file.

        Args:
            content: Binary content to store
            module_id: Module identifier
            version: Version string (e.g., "v1.0.0")

        Returns:
            Content hash (SHA256)
        """
        content_hash = self._hash_content(content)
        obj_path = self._object_path(content_hash)

        with self._lock:
            # Store object if new
            if content_hash not in self._entries:
                # Create prefix directory if needed
                obj_path.parent.mkdir(parents=True, exist_ok=True)

                # Write content
                with open(obj_path, "wb") as f:
                    f.write(content)

                self._entries[content_hash] = CASEntry(
                    content_hash=content_hash,
                    size_bytes=len(content),
                    path=obj_path,
                    reference_count=0,
                )

            # Create reference
            ref_path = self._ref_path(module_id, version)
            ref_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if reference already exists to different hash
            if module_id in self._references and version in self._references[module_id]:
                old_ref = self._references[module_id][version]
                if old_ref.content_hash != content_hash:
                    # Update reference count for old content
                    if old_ref.content_hash in self._entries:
                        self._entries[old_ref.content_hash].reference_count -= 1

            # Write reference file
            with open(ref_path, "w") as f:
                f.write(content_hash)

            # Update internal state
            if module_id not in self._references:
                self._references[module_id] = {}

            self._references[module_id][version] = CASReference(
                module_id=module_id,
                version=version,
                content_hash=content_hash,
            )

            # Increment reference count
            self._entries[content_hash].reference_count += 1
            self._entries[content_hash].last_accessed = time.time()

        return content_hash

    def get(self, module_id: str, version: str) -> Optional[bytes]:
        """
        Get content by module reference.

        Args:
            module_id: Module identifier
            version: Version string

        Returns:
            Binary content if found, None otherwise
        """
        ref_path = self._ref_path(module_id, version)
        if not ref_path.exists():
            return None

        try:
            content_hash = ref_path.read_text().strip()
            return self.get_by_hash(content_hash)
        except (OSError, IOError):
            return None

    def get_by_hash(self, content_hash: str) -> Optional[bytes]:
        """
        Get content directly by hash.

        Args:
            content_hash: SHA256 hash

        Returns:
            Binary content if found, None otherwise
        """
        with self._lock:
            if content_hash not in self._entries:
                return None

            entry = self._entries[content_hash]
            entry.last_accessed = time.time()

        try:
            with open(entry.path, "rb") as f:
                return f.read()
        except (OSError, IOError):
            return None

    def get_hash(self, module_id: str, version: str) -> Optional[str]:
        """
        Get content hash for a module version without reading content.

        Args:
            module_id: Module identifier
            version: Version string

        Returns:
            Content hash if found, None otherwise
        """
        with self._lock:
            if module_id in self._references and version in self._references[module_id]:
                return self._references[module_id][version].content_hash
        return None

    def exists(self, module_id: str, version: str) -> bool:
        """
        Check if module version exists.

        Args:
            module_id: Module identifier
            version: Version string

        Returns:
            True if reference exists
        """
        with self._lock:
            return (
                module_id in self._references
                and version in self._references[module_id]
            )

    def exists_hash(self, content_hash: str) -> bool:
        """
        Check if content with given hash exists.

        Args:
            content_hash: SHA256 hash

        Returns:
            True if object exists
        """
        with self._lock:
            return content_hash in self._entries

    def delete_ref(self, module_id: str, version: str) -> bool:
        """
        Delete a reference (does not delete content).

        Content may be garbage collected later if no references remain.

        Args:
            module_id: Module identifier
            version: Version string

        Returns:
            True if reference was deleted, False if not found
        """
        ref_path = self._ref_path(module_id, version)

        with self._lock:
            if module_id not in self._references:
                return False
            if version not in self._references[module_id]:
                return False

            ref = self._references[module_id][version]
            content_hash = ref.content_hash

            # Delete reference file
            try:
                ref_path.unlink()
            except OSError:
                pass

            # Update internal state
            del self._references[module_id][version]
            if not self._references[module_id]:
                del self._references[module_id]
                # Try to remove module directory
                try:
                    (self.refs_dir / module_id).rmdir()
                except OSError:
                    pass

            # Decrement reference count
            if content_hash in self._entries:
                self._entries[content_hash].reference_count -= 1

        return True

    def garbage_collect(self) -> int:
        """
        Remove unreferenced objects.

        Deletes objects with zero reference count to reclaim disk space.

        Returns:
            Number of objects deleted
        """
        deleted = 0

        with self._lock:
            to_delete = [
                h for h, entry in self._entries.items()
                if entry.reference_count <= 0
            ]

            for content_hash in to_delete:
                entry = self._entries[content_hash]
                try:
                    entry.path.unlink()
                    # Try to remove prefix directory if empty
                    try:
                        entry.path.parent.rmdir()
                    except OSError:
                        pass
                    del self._entries[content_hash]
                    deleted += 1
                except OSError:
                    pass

        return deleted

    def list_modules(self) -> List[str]:
        """
        List all module IDs.

        Returns:
            List of module identifiers
        """
        with self._lock:
            return list(self._references.keys())

    def list_versions(self, module_id: str) -> List[str]:
        """
        List all versions of a module.

        Args:
            module_id: Module identifier

        Returns:
            List of version strings
        """
        with self._lock:
            if module_id not in self._references:
                return []
            return list(self._references[module_id].keys())

    def get_reference(self, module_id: str, version: str) -> Optional[CASReference]:
        """
        Get reference information.

        Args:
            module_id: Module identifier
            version: Version string

        Returns:
            CASReference if found, None otherwise
        """
        with self._lock:
            if module_id in self._references and version in self._references[module_id]:
                return self._references[module_id][version]
        return None

    def get_entry(self, content_hash: str) -> Optional[CASEntry]:
        """
        Get entry information by hash.

        Args:
            content_hash: SHA256 hash

        Returns:
            CASEntry if found, None otherwise
        """
        with self._lock:
            return self._entries.get(content_hash)

    def verify_integrity(self, content_hash: str) -> bool:
        """
        Verify that stored content matches its hash.

        Args:
            content_hash: SHA256 hash to verify

        Returns:
            True if content is intact, False if corrupted or missing
        """
        content = self.get_by_hash(content_hash)
        if content is None:
            return False
        return self._hash_content(content) == content_hash

    def verify_all(self) -> Dict[str, bool]:
        """
        Verify integrity of all stored objects.

        Returns:
            Dict mapping content_hash to integrity status
        """
        results = {}
        with self._lock:
            hashes = list(self._entries.keys())

        for content_hash in hashes:
            results[content_hash] = self.verify_integrity(content_hash)

        return results

    def get_stats(self) -> CASStats:
        """
        Get storage statistics.

        Returns:
            CASStats with storage metrics
        """
        with self._lock:
            total_size = sum(e.size_bytes for e in self._entries.values())
            total_refs = sum(e.reference_count for e in self._entries.values())
            unique = len(self._entries)

            # Size distribution
            size_dist = {}
            for entry in self._entries.values():
                if entry.size_bytes < 1024:
                    bucket = "< 1KB"
                elif entry.size_bytes < 1024 * 1024:
                    bucket = "1KB - 1MB"
                elif entry.size_bytes < 10 * 1024 * 1024:
                    bucket = "1MB - 10MB"
                else:
                    bucket = "> 10MB"
                size_dist[bucket] = size_dist.get(bucket, 0) + 1

        return CASStats(
            unique_objects=unique,
            total_references=total_refs,
            total_size_bytes=total_size,
            deduplication_ratio=total_refs / unique if unique > 0 else 1.0,
            objects_by_size=size_dist,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Export storage state as dictionary.

        Useful for debugging and monitoring.

        Returns:
            Dict with storage state
        """
        with self._lock:
            return {
                "storage_dir": str(self.storage_dir),
                "stats": self.get_stats().to_dict(),
                "modules": {
                    module_id: {
                        version: ref.to_dict()
                        for version, ref in versions.items()
                    }
                    for module_id, versions in self._references.items()
                },
            }
