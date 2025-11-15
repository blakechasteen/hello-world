"""
SIEM Integration for HoloLoom Security Pipeline

Provides structured security logging with support for multiple SIEM backends.

Created: 2025-11-15
"""

from .core import SIEMIntegration, SIEMConfig, LogBuffer
from .taxonomy import (
    SecurityEventCategory,
    SecurityEventSubcategory,
    SecurityEvent,
    create_security_event
)
from .splunk_backend import SplunkBackend
from .elk_backend import ELKBackend
from .datadog_backend import DatadogBackend

__all__ = [
    'SIEMIntegration',
    'SIEMConfig',
    'LogBuffer',
    'SecurityEventCategory',
    'SecurityEventSubcategory',
    'SecurityEvent',
    'create_security_event',
    'SplunkBackend',
    'ELKBackend',
    'DatadogBackend',
]
