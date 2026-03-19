"""
Context-Aware Routing System

Integrates ContextDepartment with QueryClassifier for intelligent, personalized routing.

Features:
- User context integration (session history, preferences)
- Personalized routing decisions
- Learning from user feedback
- A/B testing for routing strategies

Author: HoloLoom B2B Framework
Date: November 2025
"""

from .ab_testing import ABTestRouter, RoutingVariant
from .context_router import ContextAwareRouter, RoutingDecision, UserContext
from .personalization import PersonalizationEngine, UserProfile

__all__ = [
    "ContextAwareRouter",
    "UserContext",
    "RoutingDecision",
    "PersonalizationEngine",
    "UserProfile",
    "ABTestRouter",
    "RoutingVariant",
]

__version__ = "1.0.0"
