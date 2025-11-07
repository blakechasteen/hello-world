"""
Writing Templates Module
========================

Reusable templates for structured content generation.
"""

from .email import EmailTemplate
from .report import ReportTemplate

__all__ = [
    'EmailTemplate',
    'ReportTemplate'
]
