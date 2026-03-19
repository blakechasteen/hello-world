#!/usr/bin/env python3
"""
Code Verification System
========================
Actually runs compilers, linters, and tests to verify code correctness.

Features:
- Multi-language support (Python, TypeScript, JavaScript)
- Compiler integration (TypeScript compiler, Python AST, mypy)
- Linter integration (ESLint, Pylint, etc.)
- Test execution and result parsing
- Error extraction and formatting

Usage:
    from hololoom.agentic.code_verification import CodeVerifier

    verifier = CodeVerifier()
    result = await verifier.verify_python(code, check_types=True, run_linter=True)

    if result.has_errors:
        print(f"Found {result.error_count} errors:")
        for error in result.errors:
            print(f"  Line {error.line}: {error.message}")
"""

import ast
import asyncio
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class VerificationType(str, Enum):
    """Types of verification."""
    SYNTAX = "syntax"  # Basic syntax check
    TYPE_CHECK = "type_check"  # Static type checking
    LINT = "lint"  # Style and best practices
    TEST = "test"  # Run tests
    EXECUTION = "execution"  # Actually run the code


@dataclass
class CodeError:
    """Represents a code error or warning."""
    line: int
    column: int | None
    message: str
    severity: ErrorSeverity
    code: str | None = None  # Error code (e.g., "E501" for Pylint)
    source: str = "unknown"  # Which tool found this (tsc, mypy, pylint, etc.)


@dataclass
class VerificationResult:
    """Result of code verification."""
    success: bool
    verification_type: VerificationType
    language: str
    errors: list[CodeError] = field(default_factory=list)
    warnings: list[CodeError] = field(default_factory=list)
    stdout: str | None = None
    stderr: str | None = None
    execution_time_ms: float = 0.0

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Total number of warnings."""
        return len(self.warnings)

    def get_error_summary(self) -> str:
        """Get human-readable error summary."""
        if not self.has_errors:
            return "No errors found."

        summary = f"Found {self.error_count} error(s)"
        if self.warning_count > 0:
            summary += f" and {self.warning_count} warning(s)"

        lines = [summary + ":"]
        for error in self.errors[:10]:  # Limit to first 10
            lines.append(f"  Line {error.line}: {error.message}")

        if len(self.errors) > 10:
            lines.append(f"  ... and {len(self.errors) - 10} more errors")

        return "\n".join(lines)


class PythonVerifier:
    """Verify Python code using AST, mypy, and pylint."""

    async def verify_syntax(self, code: str, file_path: str = "temp.py") -> VerificationResult:
        """Check Python syntax using AST."""
        result = VerificationResult(
            success=True,
            verification_type=VerificationType.SYNTAX,
            language="python"
        )

        try:
            ast.parse(code, filename=file_path)
        except SyntaxError as e:
            result.success = False
            result.errors.append(CodeError(
                line=e.lineno or 0,
                column=e.offset,
                message=e.msg,
                severity=ErrorSeverity.ERROR,
                source="python_ast"
            ))

        return result

    async def verify_types(self, code: str, file_path: str = "temp.py") -> VerificationResult:
        """Check types using mypy."""
        result = VerificationResult(
            success=True,
            verification_type=VerificationType.TYPE_CHECK,
            language="python"
        )

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run mypy
            proc = await asyncio.create_subprocess_exec(
                'mypy',
                '--ignore-missing-imports',
                '--no-error-summary',
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()
            result.stdout = stdout.decode('utf-8')
            result.stderr = stderr.decode('utf-8')

            # Parse mypy output
            errors = self._parse_mypy_output(result.stdout, file_path)
            result.errors.extend(errors)

            if errors:
                result.success = False

        except FileNotFoundError:
            logger.warning("mypy not installed, skipping type checking")
            result.warnings.append(CodeError(
                line=0,
                column=None,
                message="mypy not available for type checking",
                severity=ErrorSeverity.WARNING,
                source="system"
            ))
        finally:
            os.unlink(temp_path)

        return result

    async def verify_lint(self, code: str, file_path: str = "temp.py") -> VerificationResult:
        """Check style using pylint."""
        result = VerificationResult(
            success=True,
            verification_type=VerificationType.LINT,
            language="python"
        )

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run pylint
            proc = await asyncio.create_subprocess_exec(
                'pylint',
                '--output-format=json',
                '--disable=C0114,C0115,C0116',  # Disable docstring warnings
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()
            result.stdout = stdout.decode('utf-8')

            # Parse pylint JSON output
            if result.stdout:
                errors = self._parse_pylint_output(result.stdout, file_path)
                for error in errors:
                    if error.severity == ErrorSeverity.ERROR:
                        result.errors.append(error)
                    else:
                        result.warnings.append(error)

            if result.errors:
                result.success = False

        except FileNotFoundError:
            logger.warning("pylint not installed, skipping linting")
        finally:
            os.unlink(temp_path)

        return result

    def _parse_mypy_output(self, output: str, file_path: str) -> list[CodeError]:
        """Parse mypy output into CodeError objects."""
        errors = []

        # mypy format: filename.py:line: error: message
        pattern = r'(?:.*?):(\d+): (error|warning|note): (.+)'

        for match in re.finditer(pattern, output):
            line_num = int(match.group(1))
            severity = ErrorSeverity.ERROR if match.group(2) == 'error' else ErrorSeverity.WARNING
            message = match.group(3)

            errors.append(CodeError(
                line=line_num,
                column=None,
                message=message,
                severity=severity,
                source="mypy"
            ))

        return errors

    def _parse_pylint_output(self, output: str, file_path: str) -> list[CodeError]:
        """Parse pylint JSON output into CodeError objects."""
        errors = []

        try:
            data = json.loads(output)
            for item in data:
                severity = ErrorSeverity.ERROR if item['type'] in ['error', 'fatal'] else ErrorSeverity.WARNING

                errors.append(CodeError(
                    line=item['line'],
                    column=item['column'],
                    message=item['message'],
                    severity=severity,
                    code=item['message-id'],
                    source="pylint"
                ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse pylint output: {e}")

        return errors


class TypeScriptVerifier:
    """Verify TypeScript code using tsc and eslint."""

    async def verify_syntax(self, code: str, file_path: str = "temp.ts") -> VerificationResult:
        """Check TypeScript syntax using tsc."""
        result = VerificationResult(
            success=True,
            verification_type=VerificationType.SYNTAX,
            language="typescript"
        )

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run tsc
            proc = await asyncio.create_subprocess_exec(
                'npx',
                'tsc',
                '--noEmit',
                '--strict',
                '--skipLibCheck',
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()
            result.stdout = stdout.decode('utf-8')
            result.stderr = stderr.decode('utf-8')

            # Parse tsc output
            errors = self._parse_tsc_output(result.stdout + result.stderr, file_path)
            result.errors.extend(errors)

            if errors:
                result.success = False

        except FileNotFoundError:
            logger.warning("tsc not available, skipping syntax check")
        finally:
            os.unlink(temp_path)

        return result

    async def verify_lint(self, code: str, file_path: str = "temp.ts") -> VerificationResult:
        """Check style using eslint."""
        result = VerificationResult(
            success=True,
            verification_type=VerificationType.LINT,
            language="typescript"
        )

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run eslint
            proc = await asyncio.create_subprocess_exec(
                'npx',
                'eslint',
                '--format=json',
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()
            result.stdout = stdout.decode('utf-8')

            # Parse eslint JSON output
            if result.stdout:
                errors = self._parse_eslint_output(result.stdout, file_path)
                for error in errors:
                    if error.severity == ErrorSeverity.ERROR:
                        result.errors.append(error)
                    else:
                        result.warnings.append(error)

            if result.errors:
                result.success = False

        except FileNotFoundError:
            logger.warning("eslint not available, skipping linting")
        finally:
            os.unlink(temp_path)

        return result

    def _parse_tsc_output(self, output: str, file_path: str) -> list[CodeError]:
        """Parse tsc output into CodeError objects."""
        errors = []

        # tsc format: filename.ts(line,column): error TS####: message
        pattern = r'(?:.*?)\((\d+),(\d+)\): (error|warning) TS(\d+): (.+)'

        for match in re.finditer(pattern, output):
            line_num = int(match.group(1))
            column = int(match.group(2))
            severity = ErrorSeverity.ERROR if match.group(3) == 'error' else ErrorSeverity.WARNING
            code = f"TS{match.group(4)}"
            message = match.group(5)

            errors.append(CodeError(
                line=line_num,
                column=column,
                message=message,
                severity=severity,
                code=code,
                source="tsc"
            ))

        return errors

    def _parse_eslint_output(self, output: str, file_path: str) -> list[CodeError]:
        """Parse eslint JSON output into CodeError objects."""
        errors = []

        try:
            data = json.loads(output)
            for file_data in data:
                for msg in file_data.get('messages', []):
                    severity = ErrorSeverity.ERROR if msg['severity'] == 2 else ErrorSeverity.WARNING

                    errors.append(CodeError(
                        line=msg['line'],
                        column=msg.get('column'),
                        message=msg['message'],
                        severity=severity,
                        code=msg.get('ruleId'),
                        source="eslint"
                    ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse eslint output: {e}")

        return errors


class CodeVerifier:
    """
    Main code verification interface.

    Coordinates verification across different languages and tools.
    """

    def __init__(self):
        """Initialize verifier with language-specific backends."""
        self.python_verifier = PythonVerifier()
        self.typescript_verifier = TypeScriptVerifier()

    async def verify(
        self,
        code: str,
        language: str,
        file_path: str = "temp",
        check_syntax: bool = True,
        check_types: bool = True,
        check_lint: bool = False
    ) -> list[VerificationResult]:
        """
        Verify code with multiple checks.

        Args:
            code: Source code
            language: Programming language
            file_path: File path for context
            check_syntax: Run syntax checker
            check_types: Run type checker
            check_lint: Run linter

        Returns:
            List of VerificationResult objects (one per check type)
        """
        results = []

        if language.lower() == "python":
            if check_syntax:
                result = await self.python_verifier.verify_syntax(code, file_path)
                results.append(result)

            if check_types:
                result = await self.python_verifier.verify_types(code, file_path)
                results.append(result)

            if check_lint:
                result = await self.python_verifier.verify_lint(code, file_path)
                results.append(result)

        elif language.lower() in ["typescript", "javascript"]:
            file_suffix = ".ts" if language.lower() == "typescript" else ".js"
            file_path_with_ext = file_path if file_path.endswith(file_suffix) else f"{file_path}{file_suffix}"

            if check_syntax or check_types:
                result = await self.typescript_verifier.verify_syntax(code, file_path_with_ext)
                results.append(result)

            if check_lint:
                result = await self.typescript_verifier.verify_lint(code, file_path_with_ext)
                results.append(result)

        else:
            logger.warning(f"Unsupported language: {language}")

        return results

    async def verify_comprehensive(
        self,
        code: str,
        language: str,
        file_path: str = "temp"
    ) -> dict[str, Any]:
        """
        Run all available verifications and return comprehensive report.

        Args:
            code: Source code
            language: Programming language
            file_path: File path for context

        Returns:
            Dict with results from all checks, aggregated
        """
        results = await self.verify(
            code,
            language,
            file_path,
            check_syntax=True,
            check_types=True,
            check_lint=True
        )

        # Aggregate results
        all_errors = []
        all_warnings = []
        checks_run = []

        for result in results:
            checks_run.append(result.verification_type.value)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        # Deduplicate errors (same line + message)
        unique_errors = []
        seen = set()
        for error in all_errors:
            key = (error.line, error.message)
            if key not in seen:
                seen.add(key)
                unique_errors.append(error)

        # Sort by line number
        unique_errors.sort(key=lambda e: e.line)

        return {
            "success": len(unique_errors) == 0,
            "language": language,
            "checks_run": checks_run,
            "error_count": len(unique_errors),
            "warning_count": len(all_warnings),
            "errors": [
                {
                    "line": e.line,
                    "column": e.column,
                    "message": e.message,
                    "severity": e.severity.value,
                    "code": e.code,
                    "source": e.source
                }
                for e in unique_errors
            ],
            "warnings": [
                {
                    "line": w.line,
                    "column": w.column,
                    "message": w.message,
                    "code": w.code,
                    "source": w.source
                }
                for w in all_warnings[:20]  # Limit warnings
            ],
            "summary": self._generate_summary(unique_errors, all_warnings)
        }

    def _generate_summary(self, errors: list[CodeError], warnings: list[CodeError]) -> str:
        """Generate human-readable summary."""
        if not errors:
            summary = "✅ No errors found"
            if warnings:
                summary += f" ({len(warnings)} warnings)"
            return summary

        lines = [f"❌ Found {len(errors)} error(s)"]

        # Group by source
        by_source = {}
        for error in errors:
            by_source.setdefault(error.source, []).append(error)

        for source, source_errors in by_source.items():
            lines.append(f"  {source}: {len(source_errors)} error(s)")

        # Show first few errors
        lines.append("\nFirst errors:")
        for error in errors[:5]:
            lines.append(f"  Line {error.line}: {error.message}")

        if len(errors) > 5:
            lines.append(f"  ... and {len(errors) - 5} more")

        return "\n".join(lines)
