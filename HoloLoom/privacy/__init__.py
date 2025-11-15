"""
Privacy Module

Privacy-preserving data collection and learning for HoloLoom.

Features:
- Secure data collection with PII anonymization
- Differential privacy for aggregate statistics
- GDPR/CCPA compliance helpers
- Encrypted storage (AES-256-GCM)

Usage:
    >>> from HoloLoom.privacy import SecureDataCollectionLoop
    >>> collector = SecureDataCollectionLoop(retention_days=30)
    >>> await collector.collect_interaction(
    ...     user_id="user123",
    ...     query="What is Thompson Sampling?",
    ...     confidence=0.92
    ... )
"""

from HoloLoom.privacy.secure_collection import (
    SecureDataCollectionLoop,
    DifferentialPrivacy,
    SecureDataStore,
    AnonymizedInteraction,
    QueryType
)

__all__ = [
    "SecureDataCollectionLoop",
    "DifferentialPrivacy",
    "SecureDataStore",
    "AnonymizedInteraction",
    "QueryType"
]
