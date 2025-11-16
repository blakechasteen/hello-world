"""Incident Response System - NIST SP 800-61 + GDPR/CCPA Compliance."""

from .core import (
    IncidentRegistry,
    IncidentMetadata,
    BreachNotification,
    IncidentAction,
    IncidentSeverity,
    IncidentType,
    IncidentStatus,
    NotificationStatus,
    get_notification_deadline,
    assess_high_risk
)

from .notification import (
    BreachNotificationBuilder
)

__all__ = [
    "IncidentRegistry",
    "IncidentMetadata",
    "BreachNotification",
    "IncidentAction",
    "IncidentSeverity",
    "IncidentType",
    "IncidentStatus",
    "NotificationStatus",
    "BreachNotificationBuilder",
    "get_notification_deadline",
    "assess_high_risk"
]

__version__ = "1.0.0"
__status__ = "Production Ready"
__created__ = "2025-11-16"
