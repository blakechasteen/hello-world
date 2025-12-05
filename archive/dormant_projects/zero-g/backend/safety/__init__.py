"""
Zero-G OpSec Layer

Production-grade security for Zero-G's enterprise data platform.

Modules:
- encryption.py: At-rest (AES-256-GCM) + In-transit (TLS 1.3, mTLS) encryption
- rate_limiter.py: Token bucket rate limiting (global + per-user)
- permissions.py: Permission levels, row-level security, audit logging
- classification.py: Auto-classification (PUBLIC → RESTRICTED), data masking
- opsec.py: Main OpSec orchestrator integrating all modules

Security Principles:
- Defense in Depth: Multiple security layers
- Fail Closed: Deny by default
- Least Privilege: Minimum permissions required
- Complete Provenance: All decisions auditable
- Zero Trust: Never trust, always verify

Version: 1.0.0
Status: Phase 2 Production Security Architecture
"""

from .opsec import OpSecLayer, SecuredRequest, SecuredResponse

__all__ = [
    "OpSecLayer",
    "SecuredRequest",
    "SecuredResponse",
]

__version__ = "1.0.0"
