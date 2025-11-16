"""
Trough - AI Code Quality Detection System
==========================================

Detects and helps fix common issues in AI-generated code.

Components:
- AI Slop Detector: 15 categories of common AI code issues
- ML Logic Detector: 9 ML-based algorithms for subtle logic errors

Usage:
    from trough import AISlopDetector, MLLogicDetector, Language

    # Detect AI slop
    slop_detector = AISlopDetector()
    issues = await slop_detector.detect_all(code, Language.PYTHON, "example.py")

    # Detect logic errors
    logic_detector = MLLogicDetector()
    errors = await logic_detector.detect(code, Language.PYTHON, "example.py")

Author: mythRL Team
Date: November 8, 2025
"""

__version__ = '1.0.0'
__author__ = 'mythRL Team'

# Core exports
from .trough_types import Language, Severity, SlopCategory, ErrorCategory, SlopIssue, LogicError
from .ml_logic_detector import MLLogicDetector
from .ai_slop_detector import AISlopDetector
from .typescript_detector import TypeScriptSlopDetector, detect_typescript_issues

__all__ = [
    # Types
    'Language',
    'Severity',
    'SlopCategory',
    'ErrorCategory',
    'SlopIssue',
    'LogicError',

    # Detectors
    'MLLogicDetector',
    'AISlopDetector',
    'TypeScriptSlopDetector',
    'detect_typescript_issues',
]
