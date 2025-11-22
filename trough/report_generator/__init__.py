"""
Trough Interactive HTML Report Generator

Converts Trough findings into interactive, click-through HTML reports.

Features:
- 3-panel layout (Findings | Code Preview | Details)
- Syntax highlighting with Pygments
- VS Code integration (click to open in editor)
- Auto-fix integration with diff preview
- Zero external dependencies (embedded CSS/JS)

Usage:
    from trough.report_generator import generate_html_report

    findings = detector.analyze("code.py")
    html = generate_html_report(
        findings=findings,
        output_path="report.html",
        enable_vscode_integration=True
    )

Author: Agent A (Haiku)
Week: 1-3
Status: Week 1 Complete (31/31 tests passing)
Date: 2025-11-22
"""

from .generator import (
    generate_html_report,
    generate_report_from_detector
)

__all__ = [
    'generate_html_report',
    'generate_report_from_detector'
]
