"""
HoloLoom Forensic Logging System

Immutable, tamper-proof audit logging with hash chain integrity.

**Added: 2025-11-16** - Phase 4 Security Pipeline

Components:
- ForensicLogger: Main logging interface
- HashChain: Cryptographic hash chain for tamper detection
- Storage backends: File, PostgreSQL, S3
- Search: Fast retrieval and filtering
- Export: Compliance-ready exports (GDPR, SOC2, ISO27001)
- Verification: Integrity checking

Usage:
    from HoloLoom.security.forensics import ForensicLogger

    async with ForensicLogger() as logger:
        await logger.log(
            event_type="authentication",
            severity="INFO",
            action="user_login",
            outcome="success",
            metadata={"user": "alice"}
        )
"""

from .logger import ForensicLogger, ForensicEntry
from .hash_chain import HashChain
from .verification import verify_chain, verify_signature
from .export import export_gdpr, export_soc2, export_iso27001

__all__ = [
    'ForensicLogger',
    'ForensicEntry',
    'HashChain',
    'verify_chain',
    'verify_signature',
    'export_gdpr',
    'export_soc2',
    'export_iso27001',
]
