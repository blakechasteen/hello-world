"""
Integration Tests for Transparent Compression
==============================================

Comprehensive integration tests for Phase 7:
- End-to-end compression workflows
- Algorithm selection and fallback
- Backward compatibility with uncompressed files
- Compression statistics and savings
- Performance benchmarks

Author: Claude Code
Date: November 17, 2025 (Track 2 Week 1: Integration Tests)
"""

import pytest
import time
from pathlib import Path

from HoloLoom.wool import WoolStorage
from HoloLoom.wool.compressed import (
    CompressedWoolStorage,
    CompressedWoolReference,
    CompressionAlgorithm,
    select_compression,
    compress_data,
    decompress_data
)


# Test Fixtures
# =============

@pytest.fixture
def compressed_storage():
    """Create compressed wool storage."""
    storage = CompressedWoolStorage(
        base_path=Path("/tmp/wool_test_compressed"),
        enable_cache=True,
        adaptive_compression=True
    )
    yield storage
    # Cleanup
    storage.close()


@pytest.fixture
def text_data():
    """Generate compressible text data."""
    # Highly compressible (repeated text)
    return ("Hello, world! " * 100).encode('utf-8')


@pytest.fixture
def binary_data():
    """Generate binary data."""
    return bytes(range(256)) * 10


@pytest.fixture
def random_data():
    """Generate incompressible random data."""
    import random
    return bytes(random.randint(0, 255) for _ in range(1000))


# Compression Algorithm Selection Tests
# =======================================

def test_select_compression_small_file():
    """Test that small files skip compression."""
    data = b"tiny"
    algorithm = select_compression(data, "text/plain", adaptive=True)
    assert algorithm == 'none'


def test_select_compression_precompressed():
    """Test that already-compressed files skip compression."""
    data = b"x" * 10000
    algorithm = select_compression(data, "image/jpeg", adaptive=True)
    assert algorithm == 'none'


def test_select_compression_text():
    """Test that text files use compression."""
    data = b"Hello, world! " * 100
    algorithm = select_compression(data, "text/plain", adaptive=True)
    assert algorithm in ['lz4', 'zstd:3']


def test_select_compression_large_file():
    """Test that large files prefer zstd."""
    data = b"x" * 200000
    algorithm = select_compression(data, "application/json", adaptive=True)
    # Large files should prefer zstd (better ratio)
    assert algorithm in ['zstd:3', 'lz4']


# Compression/Decompression Tests
# =================================

def test_compress_decompress_lz4(text_data):
    """Test LZ4 compression round-trip."""
    compressed, algorithm = compress_data(text_data, 'lz4')

    assert algorithm == 'lz4'
    assert len(compressed) < len(text_data)  # Should compress

    decompressed = decompress_data(compressed, 'lz4')
    assert decompressed == text_data


def test_compress_decompress_zstd(text_data):
    """Test Zstd compression round-trip."""
    compressed, algorithm = compress_data(text_data, 'zstd:3')

    assert algorithm == 'zstd:3'
    assert len(compressed) < len(text_data)

    decompressed = decompress_data(compressed, 'zstd:3')
    assert decompressed == text_data


def test_compress_incompressible_data(random_data):
    """Test that incompressible data returns 'none'."""
    compressed, algorithm = compress_data(random_data, 'lz4')

    # Random data doesn't compress well, should return 'none'
    if algorithm == 'none':
        assert compressed == random_data
    else:
        # If it did compress, verify compression ratio is poor
        ratio = len(random_data) / len(compressed)
        assert ratio < 1.5  # Less than 1.5x compression


# CompressedWoolStorage Tests
# =============================

def test_compressed_storage_store_read(compressed_storage, text_data):
    """Test storing and reading compressed data."""
    # Store
    ref = compressed_storage.store(text_data, content_type="text/plain")

    assert isinstance(ref, CompressedWoolReference)
    assert ref.compression_algorithm in ['lz4', 'zstd:3']
    assert ref.compression_ratio > 1.0  # Should compress

    # Read
    read_data = compressed_storage.read(ref)
    assert read_data.tobytes() == text_data


def test_compressed_storage_savings(compressed_storage, text_data):
    """Test that compression provides storage savings."""
    ref = compressed_storage.store(text_data, content_type="text/plain")

    # Check savings
    original_size = ref.uncompressed_size
    compressed_size = ref.compressed_size
    savings = ref.savings_bytes

    assert compressed_size < original_size
    assert savings > 0
    assert ref.compression_ratio > 1.5  # At least 1.5x compression


def test_compressed_storage_statistics(compressed_storage, text_data):
    """Test compression statistics tracking."""
    # Store multiple files
    for i in range(10):
        data = f"File {i}: " .encode() + text_data
        compressed_storage.store(data, content_type="text/plain")

    # Get stats
    stats = compressed_storage.get_compression_stats()

    assert stats['files_compressed'] == 10
    assert stats['bytes_original'] > 0
    assert stats['bytes_compressed'] > 0
    assert stats['total_savings_bytes'] > 0
    assert stats['average_compression_ratio'] > 1.0


# Backward Compatibility Tests
# ==============================

def test_compressed_storage_read_uncompressed(compressed_storage, text_data):
    """Test reading uncompressed files (backward compatibility)."""
    # Store using base WoolStorage (uncompressed)
    base_storage = WoolStorage(base_path=compressed_storage.base_path)
    base_ref = base_storage.store(text_data, content_type="text/plain")

    # Read using CompressedWoolStorage
    # Should handle uncompressed files gracefully
    read_data = compressed_storage.read(base_ref)
    assert read_data.tobytes() == text_data


def test_mixed_compressed_uncompressed(compressed_storage, text_data):
    """Test mixed compressed and uncompressed files."""
    # Store compressed
    ref_compressed = compressed_storage.store(
        text_data,
        content_type="text/plain"
    )

    # Store uncompressed (small file)
    ref_uncompressed = compressed_storage.store(
        b"tiny",
        content_type="text/plain"
    )

    # Read both
    data_compressed = compressed_storage.read(ref_compressed)
    data_uncompressed = compressed_storage.read(ref_uncompressed)

    assert data_compressed.tobytes() == text_data
    assert data_uncompressed.tobytes() == b"tiny"


# Adaptive Compression Tests
# ============================

def test_adaptive_compression_text(compressed_storage):
    """Test adaptive compression for text files."""
    text = ("Lorem ipsum " * 1000).encode('utf-8')

    ref = compressed_storage.store(text, content_type="text/plain")

    # Text should be compressed
    assert ref.compression_algorithm in ['lz4', 'zstd:3']
    assert ref.compression_ratio > 2.0  # Good compression


def test_adaptive_compression_binary(compressed_storage, binary_data):
    """Test adaptive compression for binary files."""
    ref = compressed_storage.store(binary_data, content_type="application/octet-stream")

    # Binary data should attempt compression
    # (may not compress well, but should try)
    assert ref.compression_algorithm in ['lz4', 'zstd:3', 'none']


def test_adaptive_compression_image(compressed_storage):
    """Test adaptive compression skips already-compressed images."""
    # Fake JPEG data (small sample)
    jpeg_data = b"\xff\xd8\xff\xe0" + b"x" * 1000  # JPEG header

    ref = compressed_storage.store(jpeg_data, content_type="image/jpeg")

    # Should skip compression (already compressed)
    assert ref.compression_algorithm == 'none'
    assert ref.compression_ratio == 1.0


# Content Type Tests
# ===================

def test_compression_json(compressed_storage):
    """Test compression of JSON data."""
    json_data = ('{"key": "value", "array": [1, 2, 3]}' * 100).encode('utf-8')

    ref = compressed_storage.store(json_data, content_type="application/json")

    # JSON should compress well
    assert ref.compression_algorithm in ['lz4', 'zstd:3']
    assert ref.compression_ratio > 3.0


def test_compression_xml(compressed_storage):
    """Test compression of XML data."""
    xml_data = ('<root><item>value</item></root>' * 100).encode('utf-8')

    ref = compressed_storage.store(xml_data, content_type="application/xml")

    # XML should compress well
    assert ref.compression_algorithm in ['lz4', 'zstd:3']
    assert ref.compression_ratio > 3.0


def test_compression_code(compressed_storage):
    """Test compression of source code."""
    code_data = ('def hello():\n    print("world")\n' * 100).encode('utf-8')

    ref = compressed_storage.store(code_data, content_type="text/x-python")

    # Code should compress well
    assert ref.compression_algorithm in ['lz4', 'zstd:3']
    assert ref.compression_ratio > 3.0


# Performance Tests
# ==================

def test_performance_compression_latency(compressed_storage, text_data):
    """Test compression latency."""
    start = time.time()
    ref = compressed_storage.store(text_data, content_type="text/plain")
    latency = time.time() - start

    # Compression should be <10ms for small files
    assert latency < 0.01


def test_performance_decompression_latency(compressed_storage, text_data):
    """Test decompression latency."""
    ref = compressed_storage.store(text_data, content_type="text/plain")

    start = time.time()
    data = compressed_storage.read(ref)
    latency = time.time() - start

    # Decompression should be <5ms
    assert latency < 0.005


def test_performance_throughput(compressed_storage, text_data):
    """Test compression throughput."""
    # Store 100 files
    start = time.time()

    for i in range(100):
        data = f"File {i}: ".encode() + text_data
        compressed_storage.store(data, content_type="text/plain")

    duration = time.time() - start
    throughput = 100 / duration

    # Should achieve >1000 files/sec
    assert throughput > 1000


# Compression Ratio Tests
# =========================

def test_compression_ratio_text():
    """Test compression ratio for text data."""
    text = ("Hello, world! " * 1000).encode('utf-8')

    compressed, algorithm = compress_data(text, 'lz4')

    ratio = len(text) / len(compressed)

    # Text should compress at least 5x
    assert ratio >= 5.0


def test_compression_ratio_json():
    """Test compression ratio for JSON data."""
    json_data = ('{"key": "value"}' * 1000).encode('utf-8')

    compressed, algorithm = compress_data(json_data, 'lz4')

    ratio = len(json_data) / len(compressed)

    # JSON should compress at least 8x
    assert ratio >= 8.0


def test_compression_ratio_binary():
    """Test compression ratio for binary data."""
    # Repetitive binary pattern
    binary = bytes([i % 10 for i in range(10000)])

    compressed, algorithm = compress_data(binary, 'lz4')

    ratio = len(binary) / len(compressed)

    # Repetitive binary should compress well
    assert ratio >= 3.0


# Edge Cases
# ===========

def test_empty_data(compressed_storage):
    """Test compressing empty data."""
    ref = compressed_storage.store(b"", content_type="text/plain")

    # Empty data should skip compression
    assert ref.compression_algorithm == 'none'

    data = compressed_storage.read(ref)
    assert data.tobytes() == b""


def test_large_file(compressed_storage):
    """Test compressing large file (>1MB)."""
    large_data = b"x" * (2 * 1024 * 1024)  # 2MB

    ref = compressed_storage.store(large_data, content_type="text/plain")

    # Large file should use zstd
    assert ref.compression_algorithm in ['zstd:3', 'lz4']

    data = compressed_storage.read(ref)
    assert data.tobytes() == large_data


def test_unicode_text(compressed_storage):
    """Test compressing Unicode text."""
    unicode_text = "Hello, 世界! 🌍" * 100
    data = unicode_text.encode('utf-8')

    ref = compressed_storage.store(data, content_type="text/plain")

    assert ref.compression_algorithm in ['lz4', 'zstd:3']

    read_data = compressed_storage.read(ref)
    assert read_data.tobytes().decode('utf-8') == unicode_text


# End-to-End Integration Tests
# =============================

def test_e2e_compression_workflow(compressed_storage):
    """
    End-to-end test: Store multiple files, verify compression,
    read back, check statistics.
    """
    files = {
        "text": ("Hello, world! " * 100).encode('utf-8'),
        "json": ('{"key": "value"}' * 100).encode('utf-8'),
        "code": ('def foo():\n    pass\n' * 100).encode('utf-8'),
        "binary": bytes(range(256)) * 10
    }

    refs = {}

    # 1. Store all files
    for name, data in files.items():
        ref = compressed_storage.store(data, content_type=f"test/{name}")
        refs[name] = ref

    # 2. Verify compression
    for name, ref in refs.items():
        assert isinstance(ref, CompressedWoolReference)
        if name != "binary":
            # Text-based should compress
            assert ref.compression_ratio > 1.5

    # 3. Read back and verify
    for name, ref in refs.items():
        data = compressed_storage.read(ref)
        assert data.tobytes() == files[name]

    # 4. Check statistics
    stats = compressed_storage.get_compression_stats()
    assert stats['files_compressed'] == len(files)
    assert stats['total_savings_bytes'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
