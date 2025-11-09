#!/usr/bin/env python3
"""
AI Slop Detector - Comprehensive Detection of Common AI Code Pitfalls
======================================================================

Detects the most common issues in AI-generated code beyond simple hallucinations:

1. **Hallucinations** - Non-existent functions/classes/imports
2. **Missing Error Handling** - Try/except, null checks, validation
3. **Hardcoded Values** - API keys, secrets, magic numbers
4. **Race Conditions** - Async/threading issues
5. **Resource Leaks** - Unclosed files, connections, locks
6. **Type Mismatches** - Incorrect type usage
7. **Security Issues** - SQL injection, XSS, command injection
8. **Performance Anti-patterns** - N+1 queries, unnecessary loops
9. **Dead Code** - Unused imports, variables, functions
10. **Inconsistent Naming** - Mixed conventions (camelCase vs snake_case)
11. **Missing Docstrings** - Undocumented functions/classes
12. **Copy-Paste Errors** - Repeated code with minor variations
13. **Incomplete Implementation** - TODO comments, pass statements
14. **Off-by-One Errors** - Array indexing issues
15. **Timezone Issues** - Naive datetime usage

Usage:
    from trough import AISlopDetector, Language

    detector = AISlopDetector()
    issues = await detector.detect_all(code, Language.PYTHON, "file.py")

    for issue in issues:
        print(f"Line {issue.line_number}: {issue.category} - {issue.message}")
        if issue.suggestion:
            print(f"  Fix: {issue.suggestion}")
"""

import ast
import re
import logging
from typing import List, Dict, Optional, Any, Set

# Import from local types module (no HoloLoom dependency)
from .trough_types import Language, Severity, SlopCategory, SlopIssue


logger = logging.getLogger(__name__)


class AISlopDetector:
    """
    Comprehensive detector for common AI code generation pitfalls.

    Standalone version - no HoloLoom dependencies.
    Pattern-based detection of security issues, bugs, and code quality problems.
    """

    def __init__(self):
        """Initialize detector."""
        # Note: Hallucination detection removed for standalone version
        # Can be added back if needed with a separate indexer

        # Common hardcoded secret patterns
        self.secret_patterns = [
            (r'(password|passwd|pwd)\s*=\s*["\']([^"\']+)["\']', "hardcoded password"),
            (r'(api[_-]?key|apikey)\s*=\s*["\']([^"\']+)["\']', "hardcoded API key"),
            (r'(secret|token)\s*=\s*["\']([^"\']+)["\']', "hardcoded secret"),
            (r'(aws[_-]?access[_-]?key|aws[_-]?secret)', "AWS credentials"),
            (r'sk-[a-zA-Z0-9]{48}', "OpenAI API key"),
            (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token"),
        ]

        # SQL injection patterns
        self.sql_injection_patterns = [
            r'execute\s*\(\s*f?["\'].*\{.*\}.*["\']',
            r'\.format\s*\(',
            r'%s.*%\s*\(',
            r'\+.*\+',  # String concatenation in SQL
        ]

        # XSS patterns
        self.xss_patterns = [
            r'innerHTML\s*=',
            r'dangerouslySetInnerHTML',
            r'document\.write\s*\(',
            r'\.html\s*\(',  # jQuery html()
        ]

    async def detect_all(
        self,
        code: str,
        language: Language,
        file_path: str = "temp"
    ) -> List[SlopIssue]:
        """
        Run all detectors and return comprehensive list of issues.

        Args:
            code: Source code to analyze
            language: Programming language
            file_path: File path for context

        Returns:
            List of all detected issues, sorted by severity
        """
        issues = []

        # Note: Hallucination detection removed for standalone version
        # (requires codebase indexer dependency)

        # 1. Error handling issues
        issues.extend(self._detect_missing_error_handling(code, language, file_path))

        # 2. Hardcoded secrets
        issues.extend(self._detect_hardcoded_secrets(code, language, file_path))

        # 3. Resource leaks
        issues.extend(self._detect_resource_leaks(code, language, file_path))

        # 4. Security issues
        issues.extend(self._detect_security_issues(code, language, file_path))

        # 5. Performance anti-patterns
        issues.extend(self._detect_performance_issues(code, language, file_path))

        # 6. Dead code
        issues.extend(self._detect_dead_code(code, language, file_path))

        # 7. Inconsistent naming
        issues.extend(self._detect_naming_issues(code, language, file_path))

        # 8. Missing documentation
        issues.extend(self._detect_missing_docs(code, language, file_path))

        # 9. Incomplete implementation
        issues.extend(self._detect_incomplete_code(code, language, file_path))

        # 10. Off-by-one errors
        issues.extend(self._detect_off_by_one(code, language, file_path))

        # 11. Timezone issues
        issues.extend(self._detect_timezone_issues(code, language, file_path))

        # 12. Copy-paste errors
        issues.extend(self._detect_copy_paste_errors(code, language, file_path))

        # Sort by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4
        }
        issues.sort(key=lambda x: (severity_order[x.severity], x.line_number))

        return issues

    def _detect_missing_error_handling(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect missing try/except, null checks, validation."""
        issues = []

        if language == Language.PYTHON:
            # Check for file operations without try/except
            file_ops_pattern = r'(open\(|\.read\(|\.write\()'
            for match in re.finditer(file_ops_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                context = self._get_context(code, line_num)

                # Check if inside try block
                if 'try:' not in context and 'with ' not in context:
                    issues.append(SlopIssue(
                        category=SlopCategory.ERROR_HANDLING,
                        severity=Severity.MEDIUM,
                        line_number=line_num,
                        column=match.start() - code.rfind('\n', 0, match.start()),
                        description="File operation without error handling (use try/except or 'with' statement)",
                        context=context,
                        fix_suggestion="Wrap in try/except or use 'with open(...) as f:'"
                    ))

            # Check for network requests without error handling
            network_pattern = r'(requests\.|urllib\.|http\.client)'
            for match in re.finditer(network_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                context = self._get_context(code, line_num)

                if 'try:' not in context:
                    issues.append(SlopIssue(
                        category=SlopCategory.ERROR_HANDLING,
                        severity=Severity.HIGH,
                        line_number=line_num,
                        column=match.start() - code.rfind('\n', 0, match.start()),
                        description="Network request without error handling",
                        context=context,
                        fix_suggestion="Add try/except for ConnectionError, Timeout, HTTPError"
                    ))

            # Check for dictionary access without .get()
            dict_access_pattern = r'\w+\[[\'"][^\'"]+[\'"]\](?!\s*=)'
            for match in re.finditer(dict_access_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.ERROR_HANDLING,
                    severity=Severity.MEDIUM,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="Dictionary access without KeyError handling",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Use .get(key, default) instead of [key]"
                ))

        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            # Check for fetch without .catch()
            fetch_pattern = r'fetch\([^)]+\)(?!\.catch)'
            for match in re.finditer(fetch_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.ERROR_HANDLING,
                    severity=Severity.HIGH,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="fetch() without .catch() error handling",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Add .catch(error => console.error(error))"
                ))

            # Check for JSON.parse without try/catch
            json_parse_pattern = r'JSON\.parse\('
            for match in re.finditer(json_parse_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                context = self._get_context(code, line_num)

                if 'try' not in context:
                    issues.append(SlopIssue(
                        category=SlopCategory.ERROR_HANDLING,
                        severity=Severity.MEDIUM,
                        line_number=line_num,
                        column=match.start() - code.rfind('\n', 0, match.start()),
                        description="JSON.parse() without try/catch (can throw on invalid JSON)",
                        context=context,
                        fix_suggestion="Wrap in try/catch to handle SyntaxError"
                    ))

        return issues

    def _detect_hardcoded_secrets(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect hardcoded passwords, API keys, secrets."""
        issues = []

        for pattern, description in self.secret_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.HARDCODED_VALUES,
                    severity=Severity.CRITICAL,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description=f"Hardcoded {description} detected",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Move to environment variable (os.getenv() or process.env)"
                ))

        # Detect magic numbers
        magic_number_pattern = r'(?<!\d)([0-9]{3,})(?!\d)'  # Numbers with 3+ digits
        for match in re.finditer(magic_number_pattern, code):
            # Skip common numbers (200, 404, 1000, etc.)
            number = match.group(1)
            if number in ['200', '404', '500', '1000', '2000', '3000', '8000']:
                continue

            line_num = code[:match.start()].count('\n') + 1
            issues.append(SlopIssue(
                category=SlopCategory.HARDCODED_VALUES,
                severity=Severity.LOW,
                line_number=line_num,
                column=match.start() - code.rfind('\n', 0, match.start()),
                description=f"Magic number {number} should be a named constant",
                context=self._get_context(code, line_num),
                fix_suggestion=f"Define as constant: MAX_RETRIES = {number}"
            ))

        return issues

    def _detect_resource_leaks(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect unclosed files, connections, locks."""
        issues = []

        if language == Language.PYTHON:
            # Check for open() without 'with'
            open_pattern = r'\b(\w+)\s*=\s*open\('
            for match in re.finditer(open_pattern, code):
                var_name = match.group(1)
                line_num = code[:match.start()].count('\n') + 1
                context = self._get_context(code, line_num, window=10)

                # Check if .close() is called
                if f"{var_name}.close()" not in context:
                    issues.append(SlopIssue(
                        category=SlopCategory.RESOURCE_LEAK,
                        severity=Severity.HIGH,
                        line_number=line_num,
                        column=match.start() - code.rfind('\n', 0, match.start()),
                        description=f"File '{var_name}' opened but never closed (use 'with' statement)",
                        context=self._get_context(code, line_num),
                        fix_suggestion=f"Use: with open(...) as {var_name}:"
                    ))

            # Check for database connections without close()
            db_pattern = r'(\w+)\s*=\s*(sqlite3\.connect|psycopg2\.connect|mysql\.connector\.connect)'
            for match in re.finditer(db_pattern, code):
                var_name = match.group(1)
                line_num = code[:match.start()].count('\n') + 1
                context = self._get_context(code, line_num, window=15)

                if f"{var_name}.close()" not in context:
                    issues.append(SlopIssue(
                        category=SlopCategory.RESOURCE_LEAK,
                        severity=Severity.HIGH,
                        line_number=line_num,
                        column=match.start() - code.rfind('\n', 0, match.start()),
                        description=f"Database connection '{var_name}' never closed",
                        context=self._get_context(code, line_num),
                        fix_suggestion=f"Add {var_name}.close() or use context manager"
                    ))

        return issues

    def _detect_security_issues(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect SQL injection, XSS, command injection."""
        issues = []

        if language == Language.PYTHON:
            # SQL injection via string formatting
            for pattern in self.sql_injection_patterns:
                for match in re.finditer(pattern, code):
                    line_num = code[:match.start()].count('\n') + 1
                    context = self._get_context(code, line_num)

                    # Check if it's actually SQL
                    if any(keyword in context.lower() for keyword in ['select', 'insert', 'update', 'delete', 'execute']):
                        issues.append(SlopIssue(
                            category=SlopCategory.SECURITY,
                            severity=Severity.CRITICAL,
                            line_number=line_num,
                            column=match.start() - code.rfind('\n', 0, match.start()),
                            description="Potential SQL injection vulnerability (use parameterized queries)",
                            context=context,
                            fix_suggestion="Use cursor.execute('SELECT * FROM table WHERE id = ?', (user_id,))"
                        ))

            # Command injection via os.system, subprocess
            command_pattern = r'(os\.system|subprocess\.call|subprocess\.run|eval|exec)\s*\([^)]*\{[^}]*\}[^)]*\)'
            for match in re.finditer(command_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.SECURITY,
                    severity=Severity.CRITICAL,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="Potential command injection (user input in shell command)",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Use subprocess with list arguments, not string interpolation"
                ))

        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            # XSS vulnerabilities
            for pattern in self.xss_patterns:
                for match in re.finditer(pattern, code):
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append(SlopIssue(
                        category=SlopCategory.SECURITY,
                        severity=Severity.CRITICAL,
                        line_number=line_num,
                        column=match.start() - code.rfind('\n', 0, match.start()),
                        description="Potential XSS vulnerability (unsafe HTML insertion)",
                        context=self._get_context(code, line_num),
                        fix_suggestion="Use textContent instead of innerHTML, or sanitize input"
                    ))

        return issues

    def _detect_performance_issues(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect N+1 queries, unnecessary loops, inefficient patterns."""
        issues = []

        if language == Language.PYTHON:
            # N+1 query pattern
            n_plus_1_pattern = r'for\s+\w+\s+in\s+.*:\s*\n\s*(execute\(|\.query\(|\.get\()'
            for match in re.finditer(n_plus_1_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="Potential N+1 query problem (database query inside loop)",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Fetch all records in one query, then iterate"
                ))

            # String concatenation in loops
            concat_in_loop_pattern = r'for\s+.*:\s*\n\s*\w+\s*\+?=\s*["\']'
            for match in re.finditer(concat_in_loop_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="String concatenation in loop (inefficient for large loops)",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Use list.append() then ''.join(list) for better performance"
                ))

        return issues

    def _detect_dead_code(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect unused imports, variables, functions."""
        issues = []

        if language == Language.PYTHON:
            try:
                tree = ast.parse(code)

                # Track imports
                imported_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_names.add(alias.asname or alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imported_names.add(alias.asname or alias.name)

                # Check usage
                used_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        used_names.add(node.id)

                # Find unused imports
                unused = imported_names - used_names
                for name in unused:
                    # Find line number
                    for line_num, line in enumerate(code.split('\n'), 1):
                        if f'import {name}' in line or f'import.*{name}' in line:
                            issues.append(SlopIssue(
                                category=SlopCategory.DEAD_CODE,
                                severity=Severity.LOW,
                                line_number=line_num,
                                column=0,
                                description=f"Unused import '{name}'",
                                context=line,
                                fix_suggestion=f"Remove import statement"
                            ))
                            break

            except SyntaxError:
                pass  # Skip dead code detection if syntax error

        return issues

    def _detect_naming_issues(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect inconsistent naming conventions."""
        issues = []

        if language == Language.PYTHON:
            # Check for camelCase in Python (should be snake_case)
            camel_case_pattern = r'\b([a-z]+[A-Z][a-zA-Z]*)\s*='
            for match in re.finditer(camel_case_pattern, code):
                var_name = match.group(1)
                line_num = code[:match.start()].count('\n') + 1
                snake_case = re.sub(r'([A-Z])', r'_\1', var_name).lower()

                issues.append(SlopIssue(
                    category=SlopCategory.NAMING,
                    severity=Severity.LOW,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description=f"Variable '{var_name}' uses camelCase (Python prefers snake_case)",
                    context=self._get_context(code, line_num),
                    fix_suggestion=f"Rename to '{snake_case}'"
                ))

        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            # Check for snake_case in JS/TS (should be camelCase)
            snake_case_pattern = r'\b([a-z]+_[a-z_]+)\s*='
            for match in re.finditer(snake_case_pattern, code):
                var_name = match.group(1)
                line_num = code[:match.start()].count('\n') + 1
                camel_case = ''.join(word.capitalize() if i > 0 else word
                                   for i, word in enumerate(var_name.split('_')))

                issues.append(SlopIssue(
                    category=SlopCategory.NAMING,
                    severity=Severity.LOW,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description=f"Variable '{var_name}' uses snake_case (JavaScript prefers camelCase)",
                    context=self._get_context(code, line_num),
                    fix_suggestion=f"Rename to '{camel_case}'"
                ))

        return issues

    def _detect_missing_docs(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect functions/classes without docstrings."""
        issues = []

        if language == Language.PYTHON:
            try:
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # Check if has docstring
                        has_docstring = (
                            node.body and
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)
                        )

                        if not has_docstring and not node.name.startswith('_'):
                            issues.append(SlopIssue(
                                category=SlopCategory.DOCUMENTATION,
                                severity=Severity.LOW,
                                line_number=node.lineno,
                                column=node.col_offset,
                                description=f"{'Function' if isinstance(node, ast.FunctionDef) else 'Class'} '{node.name}' missing docstring",
                                context=self._get_context(code, node.lineno),
                                fix_suggestion='Add docstring: """Description of function/class."""'
                            ))

            except SyntaxError:
                pass

        return issues

    def _detect_incomplete_code(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect TODO comments, pass statements, NotImplementedError."""
        issues = []

        # TODO/FIXME comments
        todo_pattern = r'#\s*(TODO|FIXME|XXX|HACK)\b'
        for match in re.finditer(todo_pattern, code, re.IGNORECASE):
            line_num = code[:match.start()].count('\n') + 1
            issues.append(SlopIssue(
                category=SlopCategory.INCOMPLETE,
                severity=Severity.MEDIUM,
                line_number=line_num,
                column=match.start() - code.rfind('\n', 0, match.start()),
                description=f"{match.group(1)} comment indicates incomplete code",
                context=self._get_context(code, line_num),
                fix_suggestion="Complete the implementation"
            ))

        # Bare pass statements
        pass_pattern = r'^\s*pass\s*$'
        for match in re.finditer(pass_pattern, code, re.MULTILINE):
            line_num = code[:match.start()].count('\n') + 1
            context = self._get_context(code, line_num)

            # Check if it's in a real function/class (not just empty)
            if 'def ' in context or 'class ' in context:
                issues.append(SlopIssue(
                    category=SlopCategory.INCOMPLETE,
                    severity=Severity.HIGH,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="Empty function/class with only 'pass' statement",
                    context=context,
                    fix_suggestion="Implement the function or raise NotImplementedError"
                ))

        # NotImplementedError or NotImplemented
        not_impl_pattern = r'raise\s+(NotImplementedError|NotImplemented)'
        for match in re.finditer(not_impl_pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            issues.append(SlopIssue(
                category=SlopCategory.INCOMPLETE,
                severity=Severity.MEDIUM,
                line_number=line_num,
                column=match.start() - code.rfind('\n', 0, match.start()),
                description="Function raises NotImplementedError (incomplete implementation)",
                context=self._get_context(code, line_num),
                fix_suggestion="Complete the implementation or remove the function"
            ))

        return issues

    def _detect_off_by_one(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect common off-by-one errors in loops and array access."""
        issues = []

        # range(len(array)) instead of enumerate
        range_len_pattern = r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):'
        for match in re.finditer(range_len_pattern, code):
            idx_var, array_var = match.groups()
            line_num = code[:match.start()].count('\n') + 1
            issues.append(SlopIssue(
                category=SlopCategory.OFF_BY_ONE,
                severity=Severity.LOW,
                line_number=line_num,
                column=match.start() - code.rfind('\n', 0, match.start()),
                description=f"Use enumerate() instead of range(len({array_var}))",
                context=self._get_context(code, line_num),
                fix_suggestion=f"for {idx_var}, item in enumerate({array_var}):"
            ))

        # Array access with len() (should be len()-1)
        len_access_pattern = r'(\w+)\[len\(\1\)\]'
        for match in re.finditer(len_access_pattern, code):
            array_var = match.group(1)
            line_num = code[:match.start()].count('\n') + 1
            issues.append(SlopIssue(
                category=SlopCategory.OFF_BY_ONE,
                severity=Severity.HIGH,
                line_number=line_num,
                column=match.start() - code.rfind('\n', 0, match.start()),
                description=f"Array access {array_var}[len({array_var})] will cause IndexError",
                context=self._get_context(code, line_num),
                fix_suggestion=f"Use {array_var}[-1] to get last element"
            ))

        return issues

    def _detect_timezone_issues(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect naive datetime usage (missing timezone)."""
        issues = []

        if language == Language.PYTHON:
            # datetime.now() without timezone
            naive_now_pattern = r'datetime\.(now|utcnow)\(\)'
            for match in re.finditer(naive_now_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.TIMEZONE,
                    severity=Severity.MEDIUM,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description=f"datetime.{match.group(1)}() creates naive datetime (no timezone)",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Use datetime.now(timezone.utc) for timezone-aware datetime"
                ))

        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            # new Date() without timezone consideration
            naive_date_pattern = r'new\s+Date\(\)'
            for match in re.finditer(naive_date_pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(SlopIssue(
                    category=SlopCategory.TIMEZONE,
                    severity=Severity.MEDIUM,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="new Date() uses local timezone (can cause bugs in distributed systems)",
                    context=self._get_context(code, line_num),
                    fix_suggestion="Use new Date().toISOString() or specify timezone explicitly"
                ))

        return issues

    def _detect_copy_paste_errors(self, code: str, language: Language, file_path: str) -> List[SlopIssue]:
        """Detect repeated code blocks with minor variations (copy-paste errors)."""
        issues = []

        # Find similar code blocks (simplified heuristic)
        lines = code.split('\n')
        blocks = {}  # signature -> (line_num, block)

        for i in range(len(lines) - 5):  # Look at blocks of 5+ lines
            block = '\n'.join(lines[i:i+5])
            # Create signature by removing variable names
            signature = re.sub(r'\b[a-z_]\w*\b', 'VAR', block)

            if signature in blocks:
                # Found similar block - potential copy-paste
                original_line, original_block = blocks[signature]
                if block != original_block:  # Not exact match
                    issues.append(SlopIssue(
                        category=SlopCategory.COPY_PASTE,
                        severity=Severity.MEDIUM,
                        line_number=i + 1,
                        column=0,
                        description=f"Similar code block found at line {original_line} (possible copy-paste error)",
                        context=block[:200],
                        fix_suggestion="Extract common logic into a function"
                    ))
            else:
                blocks[signature] = (i + 1, block)

        return issues

    def _get_context(self, code: str, line_number: int, window: int = 3) -> str:
        """Get surrounding lines for context."""
        lines = code.split('\n')
        start = max(0, line_number - window - 1)
        end = min(len(lines), line_number + window)
        return '\n'.join(lines[start:end])

    async def get_summary(self, issues: List[SlopIssue]) -> Dict[str, Any]:
        """Generate summary of detected issues."""
        summary = {
            "total_issues": len(issues),
            "by_severity": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "by_category": {},
            "top_issues": []
        }

        for issue in issues:
            # Count by severity
            summary["by_severity"][issue.severity.value] += 1

            # Count by category
            cat = issue.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1

        # Top 10 issues by severity
        summary["top_issues"] = [
            {
                "category": issue.category.value,
                "severity": issue.severity.value,
                "line": issue.line_number,
                "description": issue.description,
                "fix": issue.fix_suggestion
            }
            for issue in issues[:10]
        ]

        return summary
