"""
ChatOps Scratch Pad - Serialization
===================================
Binary content handling with proper serialization.

Fixes the `<binary>` placeholder issue in Spacetime artifacts by
providing proper serialization/deserialization of binary content.

Key Features:
- Preserves binary content (no data loss)
- Compression support for large content
- Format detection and content-type inference
- Safe handling of various content types

Created: December 2025
"""

import base64
import gzip
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ContentFormat(str, Enum):
    """Content format for serialization."""
    RAW_BYTES = "raw_bytes"         # Raw binary content
    UTF8_TEXT = "utf8_text"         # UTF-8 encoded text
    BASE64 = "base64"               # Base64 encoded binary
    GZIP_BASE64 = "gzip_base64"     # Gzip compressed, then base64
    JSON = "json"                   # JSON serializable


class ContentType(str, Enum):
    """MIME-like content type detection."""
    TEXT_PLAIN = "text/plain"
    TEXT_MARKDOWN = "text/markdown"
    TEXT_CODE = "text/code"
    APPLICATION_JSON = "application/json"
    APPLICATION_PYTHON = "application/x-python"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_WEBP = "image/webp"
    IMAGE_GIF = "image/gif"
    AUDIO_WAV = "audio/wav"
    AUDIO_MP3 = "audio/mpeg"
    VIDEO_MP4 = "video/mp4"
    APPLICATION_PDF = "application/pdf"
    APPLICATION_ZIP = "application/zip"
    APPLICATION_OCTET = "application/octet-stream"


# Magic bytes for format detection
MAGIC_BYTES = {
    b'\x89PNG\r\n\x1a\n': ContentType.IMAGE_PNG,
    b'\xff\xd8\xff': ContentType.IMAGE_JPEG,
    b'RIFF': ContentType.IMAGE_WEBP,  # Actually needs more checks
    b'GIF87a': ContentType.IMAGE_GIF,
    b'GIF89a': ContentType.IMAGE_GIF,
    b'%PDF': ContentType.APPLICATION_PDF,
    b'PK\x03\x04': ContentType.APPLICATION_ZIP,
    b'ID3': ContentType.AUDIO_MP3,
    b'\xff\xfb': ContentType.AUDIO_MP3,
    b'\x1f\x8b': ContentType.APPLICATION_OCTET,  # Gzip
}


@dataclass
class SerializedContent:
    """
    Serialized content with metadata.

    Wraps binary content with format information for safe storage
    and retrieval without data loss.
    """
    data: bytes                     # The actual serialized bytes
    format: ContentFormat           # How it was serialized
    content_type: ContentType       # Detected content type
    original_size: int              # Size before compression (if any)
    compressed: bool                # Whether compression was applied
    encoding: str | None = None  # Text encoding if applicable

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'data': base64.b64encode(self.data).decode('ascii'),
            'format': self.format.value,
            'content_type': self.content_type.value,
            'original_size': self.original_size,
            'compressed': self.compressed,
            'encoding': self.encoding
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SerializedContent':
        """Create from dictionary."""
        return cls(
            data=base64.b64decode(data['data']),
            format=ContentFormat(data['format']),
            content_type=ContentType(data['content_type']),
            original_size=data['original_size'],
            compressed=data['compressed'],
            encoding=data.get('encoding')
        )


def detect_content_type(content: bytes) -> ContentType:
    """
    Detect content type from magic bytes.

    Args:
        content: Raw bytes to analyze

    Returns:
        Detected ContentType
    """
    if not content:
        return ContentType.APPLICATION_OCTET

    # Check magic bytes
    for magic, content_type in MAGIC_BYTES.items():
        if content.startswith(magic):
            return content_type

    # Try to decode as UTF-8 text
    try:
        text = content.decode('utf-8')

        # Check for JSON
        if text.strip().startswith(('{', '[')):
            try:
                json.loads(text)
                return ContentType.APPLICATION_JSON
            except json.JSONDecodeError:
                pass

        # Check for Python code indicators
        if any(marker in text for marker in ['def ', 'class ', 'import ', 'from ']):
            return ContentType.APPLICATION_PYTHON

        # Check for Markdown
        if any(marker in text for marker in ['# ', '## ', '```', '**']):
            return ContentType.TEXT_MARKDOWN

        return ContentType.TEXT_PLAIN

    except UnicodeDecodeError:
        return ContentType.APPLICATION_OCTET


def serialize_content(
    content: str | bytes | Any,
    compress: bool = True,
    compression_threshold: int = 1024
) -> tuple[bytes, SerializedContent]:
    """
    Serialize content for storage.

    Handles text, bytes, and JSON-serializable objects.
    Applies compression for content above threshold.

    Args:
        content: The content to serialize
        compress: Whether to compress large content
        compression_threshold: Size threshold for compression (bytes)

    Returns:
        Tuple of (raw_bytes, metadata)

    Raises:
        ValueError: If content cannot be serialized
    """
    # Convert to bytes based on type
    if isinstance(content, bytes):
        raw_bytes = content
        encoding = None
    elif isinstance(content, str):
        raw_bytes = content.encode('utf-8')
        encoding = 'utf-8'
    else:
        # Try JSON serialization
        try:
            json_str = json.dumps(content, indent=2, default=str)
            raw_bytes = json_str.encode('utf-8')
            encoding = 'utf-8'
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot serialize content of type {type(content)}: {e}")

    original_size = len(raw_bytes)
    content_type = detect_content_type(raw_bytes)

    # Apply compression if beneficial
    compressed = False
    if compress and original_size > compression_threshold:
        compressed_bytes = gzip.compress(raw_bytes, compresslevel=6)
        # Only use compression if it actually saves space
        if len(compressed_bytes) < original_size * 0.9:  # At least 10% savings
            raw_bytes = compressed_bytes
            compressed = True

    # Determine format
    if compressed:
        format_type = ContentFormat.GZIP_BASE64
    elif encoding == 'utf-8':
        format_type = ContentFormat.UTF8_TEXT
    else:
        format_type = ContentFormat.RAW_BYTES

    metadata = SerializedContent(
        data=raw_bytes,
        format=format_type,
        content_type=content_type,
        original_size=original_size,
        compressed=compressed,
        encoding=encoding
    )

    return raw_bytes, metadata


def deserialize_content(
    raw_bytes: bytes,
    metadata: SerializedContent | None = None
) -> str | bytes:
    """
    Deserialize content from storage.

    Args:
        raw_bytes: The stored bytes
        metadata: Optional serialization metadata

    Returns:
        Original content (str for text, bytes for binary)
    """
    if metadata is None:
        # Try to auto-detect
        content_type = detect_content_type(raw_bytes)

        # Check if gzip compressed
        if raw_bytes[:2] == b'\x1f\x8b':
            try:
                raw_bytes = gzip.decompress(raw_bytes)
            except gzip.BadGzipFile:
                pass

        # Try UTF-8 decode
        if content_type in (
            ContentType.TEXT_PLAIN,
            ContentType.TEXT_MARKDOWN,
            ContentType.TEXT_CODE,
            ContentType.APPLICATION_JSON,
            ContentType.APPLICATION_PYTHON
        ):
            try:
                return raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                pass

        return raw_bytes

    # Use metadata for proper deserialization
    data = raw_bytes

    # Decompress if needed
    if metadata.compressed or metadata.format == ContentFormat.GZIP_BASE64:
        try:
            data = gzip.decompress(data)
        except gzip.BadGzipFile:
            logger.warning("Failed to decompress content, returning raw bytes")

    # Decode text if applicable
    if metadata.encoding:
        try:
            return data.decode(metadata.encoding)
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode with {metadata.encoding}, returning bytes")

    return data


class ContentSerializer:
    """
    High-level content serializer with caching.

    Provides a stateful interface for serialization with
    configurable compression settings.
    """

    def __init__(
        self,
        compress: bool = True,
        compression_threshold: int = 1024,
        compression_level: int = 6
    ):
        """
        Initialize the serializer.

        Args:
            compress: Whether to enable compression
            compression_threshold: Min size for compression (bytes)
            compression_level: Gzip compression level (1-9)
        """
        self.compress = compress
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level

    def serialize(
        self,
        content: str | bytes | Any
    ) -> tuple[bytes, SerializedContent]:
        """
        Serialize content.

        Args:
            content: Content to serialize

        Returns:
            Tuple of (bytes, metadata)
        """
        return serialize_content(
            content,
            compress=self.compress,
            compression_threshold=self.compression_threshold
        )

    def deserialize(
        self,
        raw_bytes: bytes,
        metadata: SerializedContent | None = None
    ) -> str | bytes:
        """
        Deserialize content.

        Args:
            raw_bytes: Stored bytes
            metadata: Serialization metadata

        Returns:
            Original content
        """
        return deserialize_content(raw_bytes, metadata)

    def serialize_for_json(
        self,
        content: str | bytes | Any
    ) -> dict[str, Any]:
        """
        Serialize content to a JSON-safe dictionary.

        Unlike the placeholder approach, this preserves binary content
        by encoding it as base64.

        Args:
            content: Content to serialize

        Returns:
            JSON-serializable dictionary with content and metadata
        """
        raw_bytes, metadata = self.serialize(content)

        return {
            'content_base64': base64.b64encode(raw_bytes).decode('ascii'),
            'metadata': metadata.to_dict()
        }

    def deserialize_from_json(
        self,
        data: dict[str, Any]
    ) -> str | bytes:
        """
        Deserialize content from JSON dictionary.

        Args:
            data: Dictionary from serialize_for_json

        Returns:
            Original content
        """
        raw_bytes = base64.b64decode(data['content_base64'])
        metadata = SerializedContent.from_dict(data['metadata'])
        return self.deserialize(raw_bytes, metadata)


# ============================================================================
# Spacetime Integration Helpers
# ============================================================================

def serialize_spacetime_artifact(artifact: Any) -> dict[str, Any]:
    """
    Serialize a Spacetime Artifact without losing binary content.

    Replaces the `<binary>` placeholder approach with proper serialization.

    Args:
        artifact: Spacetime Artifact object

    Returns:
        JSON-serializable dictionary
    """
    serializer = ContentSerializer()

    result = {
        'name': artifact.name,
        'type': artifact.type.value if hasattr(artifact.type, 'value') else str(artifact.type),
        'destination_path': artifact.destination_path,
        'metadata': artifact.metadata
    }

    if artifact.content is not None:
        if isinstance(artifact.content, (str, bytes)):
            result['content_serialized'] = serializer.serialize_for_json(artifact.content)
        else:
            # For other types, try JSON serialization
            try:
                result['content_serialized'] = serializer.serialize_for_json(
                    json.dumps(artifact.content, default=str)
                )
            except Exception as e:
                logger.warning(f"Could not serialize artifact content: {e}")
                result['content'] = str(artifact.content)
    else:
        result['content'] = None

    return result


def deserialize_spacetime_artifact(data: dict[str, Any]) -> dict[str, Any]:
    """
    Deserialize a Spacetime Artifact from storage.

    Restores binary content from proper serialization.

    Args:
        data: Serialized artifact dictionary

    Returns:
        Dictionary with restored content
    """
    serializer = ContentSerializer()

    result = {
        'name': data['name'],
        'type': data['type'],
        'destination_path': data.get('destination_path'),
        'metadata': data.get('metadata', {})
    }

    if 'content_serialized' in data:
        result['content'] = serializer.deserialize_from_json(data['content_serialized'])
    else:
        result['content'] = data.get('content')

    return result
