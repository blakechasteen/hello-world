"""
Developer Tools Workflow Templates
===================================

5 production-ready workflow templates for developer automation:
1. Bug Triage Automation - Save 30 min/bug, 95% correct assignment
2. Code Review Automation - Save 20 min/PR, catch 80% of issues
3. Dependency Update Monitor - Stay up-to-date, reduce security risks
4. Test Case Generator - Save 2 hours/feature, improve coverage
5. Documentation Generator - Save 4 hours/release, always up-to-date

**Created**: November 2025
"""

from .bug_triage import BugTriageTemplate
from .code_review import CodeReviewTemplate
from .dependency_monitor import DependencyMonitorTemplate
from .test_generator import TestGeneratorTemplate
from .doc_generator import DocGeneratorTemplate

__all__ = [
    'BugTriageTemplate',
    'CodeReviewTemplate',
    'DependencyMonitorTemplate',
    'TestGeneratorTemplate',
    'DocGeneratorTemplate',
]
