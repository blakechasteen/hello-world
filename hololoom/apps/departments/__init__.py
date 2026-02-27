#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Departments - Modular nested learning architecture.

This package provides the core framework for building modular, marketplace-ready
departments that compose through confidence negotiation.

Quick Start:

    # Import core types
    from hololoom.departments import (
        Department,
        BaseDepartment,
        DepartmentRegistry,
        DepartmentRequest,
        DepartmentResponse,
        ConfidenceLevel,
        ConfidenceMetadata
    )

    # Create a department
    class MyDepartment(BaseDepartment):
        def __init__(self):
            super().__init__(
                name="my_dept",
                domain="my_domain",
                version="1.0.0",
                supported_tasks=["my_task"],
                confidence_range=(0.6, 0.9)
            )

        async def execute(self, request):
            # Your logic here
            return DepartmentResponse(
                task_id=request.task_id,
                result={"answer": "42"},
                confidence=ConfidenceMetadata.from_score(0.85)
            )

        async def verify(self, response):
            return VerificationResult(
                sufficient=True,
                confidence_valid=True,
                reasoning_sound=True
            )

        async def refine(self, request, prior_response, verification):
            # Improvement logic
            return await self.execute(request)

    # Register and use
    registry = DepartmentRegistry()
    await registry.register(MyDepartment())

    request = DepartmentRequest(
        task_id="req_001",
        task_type="my_task",
        parameters={}
    )

    response = await registry.route_request(request)
"""

# ============================================================================
# Core Protocol
# ============================================================================

# Import from canonical location (protocols/) directly to avoid
# triggering the deprecation warning in the .protocol shim.
from hololoom.protocols.department import (
    # Confidence System
    ConfidenceLevel,
    ConfidenceMetadata,

    # Request/Response
    DepartmentRequest,
    DepartmentResponse,

    # Verification
    VerificationResult,

    # Marketplace
    DepartmentManifest,
    DepartmentConfig,

    # Protocol
    Department,

    # Utilities
    compute_learning_rate,
    should_update_now,
)

# ============================================================================
# Base Implementation
# ============================================================================

from .base import BaseDepartment

# ============================================================================
# Registry
# ============================================================================

from .registry import DepartmentRegistry, DepartmentInstance


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Protocol Types
    "ConfidenceLevel",
    "ConfidenceMetadata",
    "DepartmentRequest",
    "DepartmentResponse",
    "VerificationResult",
    "DepartmentManifest",
    "DepartmentConfig",
    "Department",

    # Base Classes
    "BaseDepartment",

    # Registry
    "DepartmentRegistry",
    "DepartmentInstance",

    # Utilities
    "compute_learning_rate",
    "should_update_now"
]


__version__ = "1.0.0"
