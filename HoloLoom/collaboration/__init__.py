# HoloLoom Phase 3: Multi-User Collaboration
# November 2025
#
# Team knowledge management with:
# - User identification and profiles
# - Contribution tracking and attribution
# - Access controls and permissions
# - Knowledge sharing and export

from HoloLoom.collaboration.user_manager import UserManager, User, UserRole
from HoloLoom.collaboration.contribution_tracker import ContributionTracker, Contribution
from HoloLoom.collaboration.access_control import AccessController, Permission, AccessLevel
from HoloLoom.collaboration.knowledge_sharing import KnowledgeSharing, SharedKnowledge

__all__ = [
    # User management
    'UserManager',
    'User',
    'UserRole',
    # Contribution tracking
    'ContributionTracker',
    'Contribution',
    # Access control
    'AccessController',
    'Permission',
    'AccessLevel',
    # Knowledge sharing
    'KnowledgeSharing',
    'SharedKnowledge',
]
