"""
Dream Department - Phase 2 NeuroHood-HoloLoom Integration
=========================================================

Production-grade department for dream processing with DS-STAR verification.

Usage:
    from HoloLoom.departments.dream import DreamDepartment, create_dream_department

    dept = create_dream_department()
    await dept.start()

    result = await dept.process({
        "action": "store",
        "narrative": narrative,
        "resident_id": "r001"
    })

    await dept.stop()
"""

from .department import (
    DreamDepartment,
    DreamDSStarVerifier,
    DreamAction,
    DreamRequest,
    DreamResponse,
    DSStarCheck,
    VerificationResult,
    create_dream_department,
    create_and_start_dream_department
)

__all__ = [
    "DreamDepartment",
    "DreamDSStarVerifier",
    "DreamAction",
    "DreamRequest",
    "DreamResponse",
    "DSStarCheck",
    "VerificationResult",
    "create_dream_department",
    "create_and_start_dream_department"
]
