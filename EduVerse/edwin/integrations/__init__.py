"""
EdWIN LMS Integration Package

Provides seamless integration with Canvas LMS and Google Classroom.

Author: Agent D
Date: 2025-11-15
"""

from .lms_base import LMSIntegration, RosterData, Assignment, Submission
from .oauth_manager import OAuth2Manager, OAuth2Provider
from .canvas_integration import CanvasIntegration
from .google_classroom import GoogleClassroomIntegration
from .assignment_mapper import AssignmentMapper, GradingScheme
from .gradebook_sync import GradebookSync, SyncStrategy
from .roster_manager import RosterManager
from .lms_api import create_lms_api_routes

__all__ = [
    "LMSIntegration",
    "RosterData",
    "Assignment",
    "Submission",
    "OAuth2Manager",
    "OAuth2Provider",
    "CanvasIntegration",
    "GoogleClassroomIntegration",
    "AssignmentMapper",
    "GradingScheme",
    "GradebookSync",
    "SyncStrategy",
    "RosterManager",
    "create_lms_api_routes",
]
