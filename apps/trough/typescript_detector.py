"""
TypeScript/JavaScript Slop Detector
====================================

Detects common quality issues in TypeScript and JavaScript code using
regex-based pattern matching (similar to Python detector).

Detects:
- Error handling issues (missing try/catch, unhandled promises)
- Hardcoded values (API keys, magic numbers)
- Resource leaks (unclosed connections, event listeners)
- Security issues (XSS, injection, eval usage)
- TypeScript-specific issues (any types, type assertions)
- React-specific issues (missing keys, unsafe lifecycle)
- Async/await issues (unhandled rejections, missing await)

Author: mythRL Team
Date: November 15, 2025
"""

import re
from typing import List, Optional
from dataclasses import dataclass

from trough.trough_types import Language, Severity, SlopCategory, SlopIssue


class TypeScriptSlopDetector:
    """
    Detects AI slop and quality issues in TypeScript/JavaScript code.

    Uses regex patterns to identify common issues without requiring
    a full TypeScript parser (which isn't available in Python stdlib).
    """

    def __init__(self):
        self.patterns = self._init_patterns()

    def _init_patterns(self) -> dict:
        """Initialize detection patterns for TypeScript/JavaScript."""
        return {
            # Error handling issues
            'unhandled_promise': {
                'pattern': r'\.then\([^)]+\)(?!\s*\.catch)',
                'category': SlopCategory.ERROR_HANDLING,
                'severity': Severity.HIGH,
                'message': 'Promise without .catch() handler',
                'suggestion': 'Add .catch() to handle promise rejections'
            },
            'missing_try_catch': {
                'pattern': r'await\s+\w+\([^)]*\)(?!\s*catch)',
                'category': SlopCategory.ERROR_HANDLING,
                'severity': Severity.MEDIUM,
                'message': 'Await without try/catch block',
                'suggestion': 'Wrap await in try/catch to handle potential errors'
            },
            'bare_throw': {
                'pattern': r'throw\s+[\'"][^\'"]+[\'"]',
                'category': SlopCategory.ERROR_HANDLING,
                'severity': Severity.LOW,
                'message': 'Throwing string instead of Error object',
                'suggestion': 'Throw new Error(...) instead of string'
            },

            # Hardcoded values
            'hardcoded_api_key': {
                'pattern': r'(?:api[_-]?key|token|secret)\s*[:=]\s*[\'"][a-zA-Z0-9_-]{20,}[\'"]',
                'category': SlopCategory.HARDCODED_VALUES,
                'severity': Severity.CRITICAL,
                'message': 'Hardcoded API key or secret detected',
                'suggestion': 'Move to environment variables (process.env.API_KEY)'
            },
            'magic_number': {
                'pattern': r'(?<![a-zA-Z_])\d{4,}(?![a-zA-Z_])',
                'category': SlopCategory.HARDCODED_VALUES,
                'severity': Severity.LOW,
                'message': 'Magic number detected',
                'suggestion': 'Extract to named constant'
            },

            # Resource leaks
            'unclosed_connection': {
                'pattern': r'(?:new\s+)?(?:WebSocket|EventSource)\([^)]+\)(?!.*\.close\(\))',
                'category': SlopCategory.RESOURCE_LEAK,
                'severity': Severity.MEDIUM,
                'message': 'Connection opened but never closed',
                'suggestion': 'Add cleanup logic to close connection'
            },
            'event_listener_leak': {
                'pattern': r'addEventListener\([^)]+\)(?!.*removeEventListener)',
                'category': SlopCategory.RESOURCE_LEAK,
                'severity': Severity.LOW,
                'message': 'Event listener added but never removed',
                'suggestion': 'Add removeEventListener in cleanup/unmount'
            },

            # Security issues
            'dangerous_eval': {
                'pattern': r'\beval\s*\(',
                'category': SlopCategory.SECURITY,
                'severity': Severity.CRITICAL,
                'message': 'Use of eval() detected - code injection risk',
                'suggestion': 'Avoid eval(), use JSON.parse() or safer alternatives'
            },
            'dangerous_innerhtml': {
                'pattern': r'\.innerHTML\s*=\s*[^;]+',
                'category': SlopCategory.SECURITY,
                'severity': Severity.HIGH,
                'message': 'Direct innerHTML assignment - XSS risk',
                'suggestion': 'Use textContent or sanitize HTML before insertion'
            },
            'console_log': {
                'pattern': r'console\.log\(',
                'category': SlopCategory.SECURITY,
                'severity': Severity.LOW,
                'message': 'Console.log in production code',
                'suggestion': 'Remove debug statements before deployment'
            },

            # TypeScript-specific
            'any_type': {
                'pattern': r':\s*any\b',
                'category': SlopCategory.INCOMPLETE,  # Changed from TYPE_CONFUSION
                'severity': Severity.MEDIUM,
                'message': 'Use of "any" type defeats TypeScript safety',
                'suggestion': 'Use specific type or "unknown" instead of "any"'
            },
            'type_assertion': {
                'pattern': r'as\s+\w+(?!;)',
                'category': SlopCategory.INCOMPLETE,  # Changed from TYPE_CONFUSION
                'severity': Severity.LOW,
                'message': 'Type assertion (as) bypasses type checking',
                'suggestion': 'Ensure type assertion is necessary and safe'
            },
            'non_null_assertion': {
                'pattern': r'!\s*\.',
                'category': SlopCategory.INCOMPLETE,  # Changed from TYPE_CONFUSION
                'severity': Severity.MEDIUM,
                'message': 'Non-null assertion operator (!) used',
                'suggestion': 'Add null check instead of forcing non-null'
            },

            # React-specific
            'missing_key': {
                'pattern': r'\.map\([^)]*=>\s*<[^>]+>(?![^<]*key=)',
                'category': SlopCategory.PERFORMANCE,
                'severity': Severity.MEDIUM,
                'message': 'Missing key prop in mapped React elements',
                'suggestion': 'Add unique key prop to each mapped element'
            },
            'unsafe_lifecycle': {
                'pattern': r'componentWillMount|componentWillReceiveProps|componentWillUpdate',
                'category': SlopCategory.DEAD_CODE,  # Changed from DEPRECATED (doesn't exist)
                'severity': Severity.HIGH,
                'message': 'Unsafe React lifecycle method',
                'suggestion': 'Use componentDidMount or useEffect hook instead'
            },

            # Async/await issues
            'missing_await': {
                'pattern': r'async\s+function[^{]*\{[^}]*(?<!await\s)\w+\([^)]*\)\.then\(',
                'category': SlopCategory.ERROR_HANDLING,
                'severity': Severity.LOW,
                'message': 'Using .then() in async function instead of await',
                'suggestion': 'Use await instead of .then() in async functions'
            },
            'promise_constructor': {
                'pattern': r'new\s+Promise\s*\(\s*\(\s*resolve\s*,\s*reject\s*\)',
                'category': SlopCategory.PERFORMANCE,
                'severity': Severity.LOW,
                'message': 'Manual Promise constructor',
                'suggestion': 'Consider using async/await for simpler code'
            },

            # Dead code
            'unused_import': {
                'pattern': r'import\s+\{[^}]+\}\s+from\s+[\'"][^\'"]+[\'"]',
                'category': SlopCategory.DEAD_CODE,
                'severity': Severity.LOW,
                'message': 'Potentially unused import (requires static analysis)',
                'suggestion': 'Remove unused imports'
            },
            'commented_code': {
                'pattern': r'//\s*(?:function|const|let|var|if|for|while)\s+\w+',
                'category': SlopCategory.DEAD_CODE,
                'severity': Severity.LOW,
                'message': 'Commented out code',
                'suggestion': 'Remove commented code or document why it\'s kept'
            },

            # Documentation
            'missing_jsdoc': {
                'pattern': r'(?:export\s+)?function\s+\w+\([^)]*\)(?!\s*/\*\*)',
                'category': SlopCategory.DOCUMENTATION,
                'severity': Severity.LOW,
                'message': 'Exported function missing JSDoc comment',
                'suggestion': 'Add JSDoc comment describing function purpose'
            },

            # Incomplete code
            'todo_comment': {
                'pattern': r'//\s*TODO:',
                'category': SlopCategory.INCOMPLETE,
                'severity': Severity.LOW,
                'message': 'TODO comment found',
                'suggestion': 'Complete the TODO or create a tracking issue'
            },
            'fixme_comment': {
                'pattern': r'//\s*FIXME:',
                'category': SlopCategory.INCOMPLETE,
                'severity': Severity.MEDIUM,
                'message': 'FIXME comment found',
                'suggestion': 'Fix the issue or create a tracking ticket'
            },
        }

    async def detect_all(
        self,
        code: str,
        language: Language,
        file_path: str = ""
    ) -> List[SlopIssue]:
        """
        Detect all TypeScript/JavaScript quality issues in code.

        Args:
            code: TypeScript/JavaScript source code
            language: Language type (TYPESCRIPT, JAVASCRIPT, TSX, JSX)
            file_path: Path to file being analyzed

        Returns:
            List of detected issues
        """
        issues = []
        lines = code.split('\n')

        # Detect issues using regex patterns
        for pattern_name, pattern_info in self.patterns.items():
            pattern = pattern_info['pattern']
            category = pattern_info['category']
            severity = pattern_info['severity']
            message = pattern_info['message']
            suggestion = pattern_info['suggestion']

            # Search for pattern in code
            for line_num, line in enumerate(lines, start=1):
                # Skip if line is a comment (basic check)
                if line.strip().startswith('//') or line.strip().startswith('/*'):
                    continue

                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Create issue
                    issue = SlopIssue(
                        category=category,
                        severity=severity,
                        message=message,
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip()[:100],  # Limit snippet length
                        suggestion=suggestion,
                        confidence=0.7  # Regex-based detection has medium confidence
                    )
                    issues.append(issue)

        # Add TypeScript-specific checks
        if language in [Language.TYPESCRIPT, Language.TSX]:
            issues.extend(await self._detect_typescript_specific(code, file_path))

        # Add React-specific checks
        if language in [Language.TSX, Language.JSX]:
            issues.extend(await self._detect_react_specific(code, file_path))

        return issues

    async def _detect_typescript_specific(
        self,
        code: str,
        file_path: str
    ) -> List[SlopIssue]:
        """Detect TypeScript-specific issues."""
        issues = []
        lines = code.split('\n')

        # Check for missing type annotations
        for line_num, line in enumerate(lines, start=1):
            # Function parameters without types
            if re.search(r'function\s+\w+\s*\(\s*\w+\s*[,\)]', line):
                issues.append(SlopIssue(
                    category=SlopCategory.INCOMPLETE,  # Changed from TYPE_CONFUSION
                    severity=Severity.LOW,
                    message='Function parameter missing type annotation',
                    file_path=file_path,
                    line_number=line_num,
                    code_snippet=line.strip()[:100],
                    suggestion='Add type annotation to parameter',
                    confidence=0.6
                ))

        return issues

    async def _detect_react_specific(
        self,
        code: str,
        file_path: str
    ) -> List[SlopIssue]:
        """Detect React-specific issues."""
        issues = []
        lines = code.split('\n')

        # Check for useState without type parameter
        for line_num, line in enumerate(lines, start=1):
            if re.search(r'useState\s*\(\s*[^<]', line):
                issues.append(SlopIssue(
                    category=SlopCategory.INCOMPLETE,  # Changed from TYPE_CONFUSION
                    severity=Severity.LOW,
                    message='useState without type parameter',
                    file_path=file_path,
                    line_number=line_num,
                    code_snippet=line.strip()[:100],
                    suggestion='Add type parameter: useState<Type>(initialValue)',
                    confidence=0.5
                ))

            # Check for useEffect without dependencies
            if re.search(r'useEffect\s*\(\s*\(\s*\)\s*=>\s*\{', line):
                # Check if next few lines have dependency array
                remaining_lines = '\n'.join(lines[line_num:line_num+5])
                if not re.search(r'\}\s*,\s*\[', remaining_lines):
                    issues.append(SlopIssue(
                        category=SlopCategory.PERFORMANCE,
                        severity=Severity.MEDIUM,
                        message='useEffect without dependency array',
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip()[:100],
                        suggestion='Add dependency array to useEffect',
                        confidence=0.6
                    ))

        return issues


# Convenience function
async def detect_typescript_issues(
    code: str,
    language: Language = Language.TYPESCRIPT,
    file_path: str = ""
) -> List[SlopIssue]:
    """
    Convenience function to detect TypeScript/JavaScript issues.

    Usage:
        issues = await detect_typescript_issues(code, Language.TYPESCRIPT)
    """
    detector = TypeScriptSlopDetector()
    return await detector.detect_all(code, language, file_path)
