"""
Trough MCP Server - AI Code Quality Detection
==============================================

Model Context Protocol (MCP) server providing Trough detection tools
for cross-department integration in HoloLoom.

Tools:
- trough_scan_file: Scan single file for issues
- trough_scan_directory: Scan multiple files
- trough_classify_issues: Classify issues by fixability
- trough_get_metrics: Get quality metrics

Usage:
    python -m trough.mcp_server

Author: mythRL Team
Date: November 15, 2025
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import glob

from trough import (
    AISlopDetector,
    MLLogicDetector,
    TypeScriptSlopDetector,
    Language,
    Severity,
    SlopIssue
)


@dataclass
class MCPServerInfo:
    """MCP server metadata."""
    name: str = "trough-detection"
    version: str = "1.0.0"
    description: str = "AI code quality detection - 15 categories, 9 ML algorithms"
    vendor: str = "mythRL"
    context_budget: int = 20000
    session_aware: bool = True


class TroughMCPServer:
    """
    MCP server for Trough code quality detection.

    Provides tools for detecting AI slop and logic errors in code,
    enabling cross-department quality assurance in HoloLoom.
    """

    def __init__(self):
        self.info = MCPServerInfo()
        self.slop_detector = AISlopDetector()
        self.logic_detector = MLLogicDetector()
        self.typescript_detector = TypeScriptSlopDetector()
        self.session_state: Dict[str, Any] = {
            'total_scans': 0,
            'total_issues': 0,
            'scans_by_file': {},
            'issues_by_category': defaultdict(int)
        }

    def _detect_language(self, file_path: str) -> Language:
        """Detect language from file extension."""
        ext = file_path.lower().split('.')[-1]
        ext_map = {
            'py': Language.PYTHON,
            'ts': Language.TYPESCRIPT,
            'tsx': Language.TSX,
            'js': Language.JAVASCRIPT,
            'jsx': Language.JSX
        }
        return ext_map.get(ext, Language.PYTHON)

    async def trough_scan_file(
        self,
        file_path: str,
        categories: Optional[List[str]] = None,
        min_severity: str = "low",
        include_ml_logic: bool = True
    ) -> Dict[str, Any]:
        """
        Scan a single file for AI slop and logic errors.

        Args:
            file_path: Path to Python file to scan
            categories: Filter to specific categories (empty = all)
            min_severity: Minimum severity to report (info/low/medium/high/critical)
            include_ml_logic: Run ML logic detector (slower but deeper)

        Returns:
            {
                'file_path': str,
                'total_issues': int,
                'issues_by_severity': dict,
                'issues': list,
                'scan_duration_ms': float
            }
        """
        start_time = time.time()

        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return {
                'file_path': file_path,
                'error': f"Failed to read file: {str(e)}",
                'total_issues': 0,
                'issues_by_severity': {},
                'issues': [],
                'scan_duration_ms': 0
            }

        # Detect language from file extension
        language = self._detect_language(file_path)

        # Detect AI slop (use appropriate detector)
        if language == Language.PYTHON:
            slop_issues = await self.slop_detector.detect_all(
                code,
                language,
                file_path
            )
        else:
            # Use TypeScript detector for TS/JS files
            slop_issues = await self.typescript_detector.detect_all(
                code,
                language,
                file_path
            )

        # Detect logic errors (if enabled and Python file)
        logic_errors = []
        if include_ml_logic and language == Language.PYTHON:
            logic_errors = await self.logic_detector.detect(
                code,
                language,
                file_path
            )

        # Convert LogicError to SlopIssue for unified handling
        converted_logic_errors = []
        for error in logic_errors:
            # LogicError has: category, severity, line, message, confidence
            # Convert to SlopIssue format
            from trough.trough_types import SlopCategory

            # Map ErrorCategory to SlopCategory
            category_map = {
                'division_by_zero': SlopCategory.ERROR_HANDLING,
                'null_dereference': SlopCategory.ERROR_HANDLING,
                'logic_contradiction': SlopCategory.INCOMPLETE,
                'missing_return': SlopCategory.INCOMPLETE,
                'constant_condition': SlopCategory.DEAD_CODE,
                'array_bounds': SlopCategory.OFF_BY_ONE,
                'wrong_operator': SlopCategory.INCOMPLETE
            }

            slop_category = category_map.get(
                error.category.value,
                SlopCategory.ERROR_HANDLING
            )

            # Use fields from LogicError
            slop_issue = SlopIssue(
                category=slop_category,
                severity=error.severity,
                message=error.message,
                line_number=error.line_number,  # Changed from error.line
                code_snippet=error.code_snippet,  # LogicError DOES have snippet
                suggestion=error.suggested_fix or "Review logic flow and add error handling",
                confidence=error.confidence,
                file_path=error.file_path
            )
            converted_logic_errors.append(slop_issue)

        # Combine issues
        all_issues = slop_issues + converted_logic_errors

        # Filter by categories
        if categories:
            all_issues = [
                issue for issue in all_issues
                if issue.category.value in categories
            ]

        # Filter by severity
        severity_order = {
            'info': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4
        }
        min_severity_level = severity_order.get(min_severity, 1)
        all_issues = [
            issue for issue in all_issues
            if severity_order.get(issue.severity.value, 0) >= min_severity_level
        ]

        # Count by severity
        issues_by_severity = defaultdict(int)
        for issue in all_issues:
            issues_by_severity[issue.severity.value] += 1

        # Convert to dict format
        issues_list = [
            {
                'category': issue.category.value,
                'severity': issue.severity.value,
                'message': issue.message,
                'line_number': issue.line_number,
                'code_snippet': issue.code_snippet,
                'suggestion': issue.suggestion,
                'confidence': issue.confidence,
                'file_path': issue.file_path
            }
            for issue in all_issues
        ]

        # Update session state
        self.session_state['total_scans'] += 1
        self.session_state['total_issues'] += len(all_issues)
        self.session_state['scans_by_file'][file_path] = len(all_issues)
        for issue in all_issues:
            self.session_state['issues_by_category'][issue.category.value] += 1

        duration_ms = (time.time() - start_time) * 1000

        return {
            'file_path': file_path,
            'total_issues': len(all_issues),
            'issues_by_severity': dict(issues_by_severity),
            'issues': issues_list,
            'scan_duration_ms': duration_ms
        }

    async def trough_scan_directory(
        self,
        directory: str,
        max_files: int = 100,
        file_pattern: str = "**/*.{py,ts,tsx,js,jsx}",
        min_severity: str = "low",
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Scan multiple files in directory with parallel processing.

        Args:
            directory: Path to scan (recursive)
            max_files: Maximum files to scan
            file_pattern: Glob pattern (default **/*.py)
            min_severity: Minimum severity to report
            categories: Category filter

        Returns:
            {
                'directory': str,
                'files_scanned': int,
                'total_issues': int,
                'issues_by_severity': dict,
                'files': [{'file': str, 'issues': int}, ...],
                'scan_duration_ms': float
            }
        """
        start_time = time.time()

        # Find files
        pattern = str(Path(directory) / file_pattern)
        py_files = glob.glob(pattern, recursive=True)
        py_files = [f for f in py_files if '__pycache__' not in f]
        py_files = py_files[:max_files]

        # Scan all files in parallel
        tasks = [
            self.trough_scan_file(
                file_path=f,
                categories=categories,
                min_severity=min_severity,
                include_ml_logic=True
            )
            for f in py_files
        ]

        results = await asyncio.gather(*tasks)

        # Aggregate results
        total_issues = sum(r['total_issues'] for r in results)
        issues_by_severity = defaultdict(int)
        for result in results:
            for severity, count in result['issues_by_severity'].items():
                issues_by_severity[severity] += count

        files_summary = [
            {'file': r['file_path'], 'issues': r['total_issues']}
            for r in results
        ]

        duration_ms = (time.time() - start_time) * 1000

        return {
            'directory': directory,
            'files_scanned': len(py_files),
            'total_issues': total_issues,
            'issues_by_severity': dict(issues_by_severity),
            'files': files_summary,
            'scan_duration_ms': duration_ms
        }

    async def trough_classify_issues(
        self,
        issues: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify issues by fixability for xTerminator routing.

        Args:
            issues: Array of issue dicts

        Returns:
            {
                'auto_fixable': [...],      # High confidence fixes
                'needs_review': [...],       # Medium confidence
                'manual_only': [...],        # Security issues
                'false_positives': [...]     # Low confidence
            }
        """
        # Convert dicts back to SlopIssue objects for classification
        from xterminator import ClassificationEngine

        classifier = ClassificationEngine()

        auto_fixable = []
        needs_review = []
        manual_only = []
        false_positives = []

        # Classify each issue
        for issue_dict in issues:
            # Simple classification based on severity and category
            severity = issue_dict.get('severity', 'medium')
            confidence = issue_dict.get('confidence', 0.5)
            category = issue_dict.get('category', '')

            # Security issues always manual
            if severity == 'critical' or 'security' in category:
                manual_only.append(issue_dict)
            # High confidence + low severity = auto-fixable
            elif severity == 'low' and confidence >= 0.85:
                auto_fixable.append(issue_dict)
            # Low confidence = likely false positive
            elif confidence < 0.4:
                false_positives.append(issue_dict)
            # Everything else needs review
            else:
                needs_review.append(issue_dict)

        return {
            'auto_fixable': auto_fixable,
            'needs_review': needs_review,
            'manual_only': manual_only,
            'false_positives': false_positives,
            'classification_summary': {
                'auto_fixable_count': len(auto_fixable),
                'needs_review_count': len(needs_review),
                'manual_only_count': len(manual_only),
                'false_positives_count': len(false_positives),
                'auto_fixable_percent': len(auto_fixable) / len(issues) * 100 if issues else 0
            }
        }

    async def trough_get_metrics(self) -> Dict[str, Any]:
        """
        Get quality metrics from current session.

        Returns:
            {
                'total_scans': int,
                'total_issues': int,
                'avg_issues_per_file': float,
                'top_categories': list,
                'files_scanned': int
            }
        """
        total_scans = self.session_state['total_scans']
        total_issues = self.session_state['total_issues']

        # Top categories by count
        top_categories = sorted(
            self.session_state['issues_by_category'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_scans': total_scans,
            'total_issues': total_issues,
            'avg_issues_per_file': total_issues / total_scans if total_scans > 0 else 0,
            'top_categories': [
                {'category': cat, 'count': count}
                for cat, count in top_categories
            ],
            'files_scanned': len(self.session_state['scans_by_file'])
        }

    def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server metadata."""
        return asdict(self.info)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {
                'name': 'trough_scan_file',
                'description': 'Scan a single file for AI slop and logic errors',
                'parameters': {
                    'file_path': {'type': 'string', 'required': True},
                    'categories': {'type': 'array', 'items': {'type': 'string'}, 'default': []},
                    'min_severity': {'type': 'string', 'enum': ['info', 'low', 'medium', 'high', 'critical'], 'default': 'low'},
                    'include_ml_logic': {'type': 'boolean', 'default': True}
                }
            },
            {
                'name': 'trough_scan_directory',
                'description': 'Scan multiple files in directory',
                'parameters': {
                    'directory': {'type': 'string', 'required': True},
                    'max_files': {'type': 'integer', 'default': 100},
                    'file_pattern': {'type': 'string', 'default': '**/*.py'},
                    'min_severity': {'type': 'string', 'default': 'low'},
                    'categories': {'type': 'array', 'items': {'type': 'string'}, 'default': []}
                }
            },
            {
                'name': 'trough_classify_issues',
                'description': 'Classify issues by fixability for xTerminator routing',
                'parameters': {
                    'issues': {'type': 'array', 'items': {'type': 'object'}, 'required': True}
                }
            },
            {
                'name': 'trough_get_metrics',
                'description': 'Get quality metrics from current session',
                'parameters': {}
            }
        ]


# MCP server entry point (if needed for standalone use)
async def main():
    """Run Trough MCP server."""
    server = TroughMCPServer()

    print("=" * 70)
    print("TROUGH MCP SERVER")
    print(f"  Name: {server.info.name}")
    print(f"  Version: {server.info.version}")
    print(f"  Description: {server.info.description}")
    print("=" * 70)
    print()
    print("Available Tools:")
    for i, tool in enumerate(server.get_tools(), 1):
        print(f"  {i}. {tool['name']} - {tool['description']}")
    print()
    print("Server ready for cross-department integration!")
    print("Use TroughMCPServer() class for programmatic access.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
