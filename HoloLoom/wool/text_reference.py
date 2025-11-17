"""
Text Reference - Pointer to Text in Wool Storage
=================================================

Lightweight reference to text stored in wool storage.
Stored in graph nodes instead of full text for zero-copy semantics.

Author: Claude Code
Date: November 17, 2025
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import WoolStorage, WoolReference


@dataclass
class TextReference:
    """
    Reference to text stored in wool storage.

    Stored as graph node property instead of full text.
    Enables zero-copy graph storage.

    Attributes:
        file_id: SHA-256 hash of source file
        offset: Byte offset of text in file
        length: Byte length of text
        encoding: Text encoding (default: utf-8)

    Example:
        # Store text in wool
        wool = WoolStorage()
        text_bytes = "Hello, World!".encode('utf-8')
        ref = wool.store(text_bytes)

        # Create TextReference
        text_ref = TextReference(
            file_id=ref.file_id,
            offset=0,
            length=len(text_bytes),
            encoding='utf-8'
        )

        # Store reference in graph (not full text!)
        graph.add_node("greeting", text_ref=text_ref)

        # Resolve later (lazy)
        text = text_ref.resolve(wool)
        # → "Hello, World!"

    Memory Savings:
        - Full text in graph: 1000 bytes
        - TextReference in graph: 88 bytes
        - Savings: 11.4x smaller!
    """
    file_id: str    # SHA-256 hash (64 hex chars)
    offset: int     # Byte offset
    length: int     # Byte length
    encoding: str = 'utf-8'  # Text encoding

    def __post_init__(self):
        """Validate fields."""
        if len(self.file_id) != 64:
            raise ValueError(f"Invalid file_id length: {len(self.file_id)}")
        if self.offset < 0:
            raise ValueError(f"Invalid offset: {self.offset}")
        if self.length < 0:
            raise ValueError(f"Invalid length: {self.length}")

    def to_dict(self) -> dict:
        """
        Serialize for graph storage.

        Returns:
            Dictionary suitable for Neo4j/NetworkX node properties
        """
        return {
            'file_id': self.file_id,
            'offset': self.offset,
            'length': self.length,
            'encoding': self.encoding
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TextReference':
        """
        Deserialize from graph storage.

        Args:
            data: Dictionary from graph node

        Returns:
            TextReference instance
        """
        return cls(
            file_id=data['file_id'],
            offset=data['offset'],
            length=data['length'],
            encoding=data.get('encoding', 'utf-8')
        )

    def resolve(self, wool_storage: 'WoolStorage') -> str:
        """
        Resolve reference to actual text (on-demand).

        Args:
            wool_storage: WoolStorage instance

        Returns:
            Decoded text string

        Raises:
            FileNotFoundError: If referenced file doesn't exist
            UnicodeDecodeError: If data is not valid text
        """
        from .storage import WoolReference

        ref = WoolReference(
            file_id=self.file_id,
            offset=self.offset,
            length=self.length
        )
        return wool_storage.read_text(ref, encoding=self.encoding)

    def to_wool_reference(self) -> 'WoolReference':
        """
        Convert to WoolReference.

        Returns:
            WoolReference for reading raw bytes
        """
        from .storage import WoolReference

        return WoolReference(
            file_id=self.file_id,
            offset=self.offset,
            length=self.length,
            content_type='text/plain'
        )

    def __sizeof__(self) -> int:
        """
        Report memory footprint.

        Returns:
            Size in bytes (approximate)
        """
        # file_id: 64 bytes (string)
        # offset: 8 bytes (int64)
        # length: 8 bytes (int64)
        # encoding: ~8 bytes (small string)
        return 88

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TextReference(file_id={self.file_id[:8]}..., "
            f"offset={self.offset}, length={self.length}, "
            f"encoding={self.encoding})"
        )
