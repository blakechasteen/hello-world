"""
HoloLoom Department Protocol — Re-export Shim
==============================================

The canonical definitions now live in ``HoloLoom.protocols.department``.
This file re-exports every public symbol so that existing
``from HoloLoom.departments.protocol import …`` continues to work.

A deprecation warning is emitted once per process to encourage migration.

Moved: 2026-02-26
"""

import warnings as _warnings

_warnings.warn(
    "Importing from HoloLoom.departments.protocol is deprecated. "
    "Use HoloLoom.protocols.department instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location
from HoloLoom.protocols.department import (  # noqa: F401, E402
    # Confidence
    ConfidenceLevel,
    ConfidenceMetadata,
    # Privacy
    PrivacyLevel,
    PrivacyEnvelope,
    # Requests and Responses
    DepartmentRequest,
    DepartmentResponse,
    # Verification
    VerificationStatus,
    VerificationCheck,
    DSStarCheck,
    VerificationResult,
    # Protocols
    Department,
    DepartmentProtocol,
    # Helpers
    create_simple_request,
    create_simple_response,
    # Type Aliases
    DepartmentFactory,
    VerificationFunction,
    # Configuration
    DepartmentConfig,
    DepartmentManifest,
    # Learning Functions
    compute_learning_rate,
    should_update_now,
)

__all__ = [
    'ConfidenceLevel',
    'ConfidenceMetadata',
    'PrivacyLevel',
    'PrivacyEnvelope',
    'DepartmentRequest',
    'DepartmentResponse',
    'VerificationStatus',
    'VerificationCheck',
    'DSStarCheck',
    'VerificationResult',
    'Department',
    'DepartmentProtocol',
    'create_simple_request',
    'create_simple_response',
    'DepartmentFactory',
    'VerificationFunction',
    'DepartmentConfig',
    'DepartmentManifest',
    'compute_learning_rate',
    'should_update_now',
]
