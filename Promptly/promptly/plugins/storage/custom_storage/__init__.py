"""
Advanced Custom Storage Backend Examples

This module showcases advanced storage backend implementations
demonstrating the extensibility of Promptly's storage system.

Available custom backends:
- S3Storage: S3/MinIO object storage with versioning and CDN
- HybridStorage: Multi-tier storage (Redis + PostgreSQL + S3)
- BlockchainStorage: Immutable prompt history with IPFS
- CustomDBStorage: Template for creating custom backends
"""

# Import custom backends with graceful degradation
try:
    from .s3 import S3Storage
    S3_AVAILABLE = True
except ImportError:
    S3Storage = None
    S3_AVAILABLE = False

try:
    from .hybrid import HybridStorage
    HYBRID_AVAILABLE = True
except ImportError:
    HybridStorage = None
    HYBRID_AVAILABLE = False

try:
    from .blockchain import BlockchainStorage
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BlockchainStorage = None
    BLOCKCHAIN_AVAILABLE = False

from .custom_db import CustomDBStorage  # Always available (template)

__all__ = [
    'S3Storage',
    'HybridStorage',
    'BlockchainStorage',
    'CustomDBStorage',
    'S3_AVAILABLE',
    'HYBRID_AVAILABLE',
    'BLOCKCHAIN_AVAILABLE',
    'get_available_custom_backends',
]


def get_available_custom_backends() -> dict:
    """
    Get list of available custom storage backends

    Returns:
        Dict mapping backend names to availability status
    """
    return {
        's3': S3_AVAILABLE,
        'hybrid': HYBRID_AVAILABLE,
        'blockchain': BLOCKCHAIN_AVAILABLE,
        'custom_db': True,  # Always available as template
    }
