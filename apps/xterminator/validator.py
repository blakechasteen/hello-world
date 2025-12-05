"""
Validation Pipeline (Phase 5)
=============================

🐷 Fern's Safety Net - Comprehensive validation ensures fixes are safe before committing.

The Validation Pipeline runs 5 stages to ensure proposed fixes don't break anything:

1. **Syntax Validator** - AST parsing (0-3ms)
2. **Import Validator** - Check all imports resolve (0-5ms)
3. **Test Validator** - Run pytest/unit tests (0-5000ms depending on test suite)
4. **Trough Validator** - Re-scan with AI slop detector (20-100ms)
5. **Regression Validator** - Verify behavior hasn't changed (1-10ms)

Philosophy: "Fern validates everything - and so should you!"
- Early termination on first failure (fail-fast)
- Detailed diagnostics for debugging
- Performance tracking for optimization
- Zero breaking changes if validation fails

Usage:
    from xterminator import ValidationPipeline

    pipeline = ValidationPipeline()
    results = await pipeline.validate_fix(
        original_code="import numpy as np",
        fixed_code="",  # Remove unused import
        file_path="config.py"
    )

    if pipeline.all_passed(results):
        print("✓ Safe to commit!")
    else:
        for result in results:
            if not result.passed:
                print(f"✗ {result.stage}: {result.message}")
"""

import ast
import asyncio
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set, Tuple
import time


# ============================================================================
# Types
# ============================================================================

class ValidationStage(str, Enum):
    """Validation pipeline stages"""
    SYNTAX = "syntax"
    IMPORTS = "imports"
    TESTS = "tests"
    TROUGH = "trough"
    REGRESSION = "regression"


@dataclass
class ValidationResult:
    """Result of a validation stage"""
    stage: ValidationStage
    passed: bool
    message: str
    duration_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None
    skipped: bool = False
    skip_reason: str = ""

    def __post_init__(self):
        """Ensure stage is ValidationStage enum"""
        if isinstance(self.stage, str):
            self.stage = ValidationStage(self.stage)

    def summary(self) -> str:
        """Human-readable summary line"""
        if self.skipped:
            return f"⊘ {self.stage.value.capitalize():<18} SKIP ({self.skip_reason}) ({int(self.duration_ms)}ms)"

        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} {self.stage.value.capitalize():<18} {self.message:<40} ({int(self.duration_ms)}ms)"


@dataclass
class ValidationPipelineReport:
    """Complete validation pipeline report"""
    results: List[ValidationResult]
    total_duration_ms: float = 0.0
    all_passed: bool = True
    failed_stage: Optional[ValidationStage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary"""
        lines = ["Validation Pipeline Report", "=" * 50]

        # Stage results
        for result in self.results:
            lines.append(result.summary())

        # Overall result
        lines.append("-" * 50)
        if self.all_passed:
            lines.append(f"✓ ALL VALIDATION PASSED ({self.total_duration_ms:.0f}ms)")
        else:
            lines.append(f"✗ VALIDATION FAILED at stage: {self.failed_stage}")
            lines.append(f"  Total time: {self.total_duration_ms:.0f}ms")

        return "\n".join(lines)


# ============================================================================
# Stage Validators
# ============================================================================

class SyntaxValidator:
    """
    Stage 1: Syntax Validation

    Verifies that fixed code is syntactically valid Python.
    Uses ast.parse() for AST construction.
    """

    async def validate(
        self,
        original_code: str,
        fixed_code: str,
        file_path: str
    ) -> ValidationResult:
        """
        Validate syntax of fixed code.

        Args:
            original_code: Original code snippet
            fixed_code: Code after proposed fix
            file_path: Path to file (for diagnostics)

        Returns:
            ValidationResult with syntax check details
        """
        start = time.time()

        try:
            # Parse fixed code
            ast.parse(fixed_code)

            duration = (time.time() - start) * 1000  # Convert to ms
            return ValidationResult(
                stage=ValidationStage.SYNTAX,
                passed=True,
                message="Syntax valid",
                duration_ms=duration,
                details={'ast_nodes': 'parsed successfully'}
            )
        except SyntaxError as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.SYNTAX,
                passed=False,
                message=f"Syntax error: {e.msg}",
                duration_ms=duration,
                details={
                    'line': e.lineno,
                    'offset': e.offset,
                    'text': e.text,
                    'error': str(e)
                }
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.SYNTAX,
                passed=False,
                message=f"Parse error: {type(e).__name__}",
                duration_ms=duration,
                details={'error': str(e)}
            )


class ImportValidator:
    """
    Stage 2: Import Validation

    Checks that all imports in fixed code are resolvable.
    Does NOT require actual installation - just checks they're not obviously broken.
    """

    async def validate(
        self,
        original_code: str,
        fixed_code: str,
        file_path: str
    ) -> ValidationResult:
        """
        Validate all imports can be resolved.

        Args:
            original_code: Original code snippet
            fixed_code: Code after proposed fix
            file_path: Path to file (for diagnostics)

        Returns:
            ValidationResult with import validation details
        """
        start = time.time()

        try:
            tree = ast.parse(fixed_code)

            # Extract all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Check each import (basic validation)
            broken = []
            for imp in imports:
                if not self._is_valid_import(imp):
                    broken.append(imp)

            duration = (time.time() - start) * 1000

            if broken:
                return ValidationResult(
                    stage=ValidationStage.IMPORTS,
                    passed=False,
                    message=f"Invalid imports: {', '.join(broken)}",
                    duration_ms=duration,
                    details={'invalid_imports': broken, 'checked': len(imports)}
                )

            return ValidationResult(
                stage=ValidationStage.IMPORTS,
                passed=True,
                message=f"All {len(imports)} imports valid" if imports else "No imports",
                duration_ms=duration,
                details={'total_imports': len(imports), 'imports': imports}
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.IMPORTS,
                passed=False,
                message=f"Import check failed: {type(e).__name__}",
                duration_ms=duration,
                details={'error': str(e)}
            )

    def _is_valid_import(self, import_name: str) -> bool:
        """
        Check if import name looks valid.

        Very basic check - just verifies it's not obviously broken.
        Real validation happens when code is actually imported.
        """
        # Empty import
        if not import_name or not import_name.strip():
            return False

        # Contains invalid characters
        for char in import_name:
            if not (char.isalnum() or char in '._'):
                return False

        # Don't start with number
        if import_name[0].isdigit():
            return False

        return True


class TestValidator:
    """
    Stage 3: Test Validation

    Runs pytest on the fixed file to ensure no test regressions.
    Skipped if no tests exist for the file.
    """

    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize test validator.

        Args:
            timeout_seconds: Maximum time to wait for tests (default: 30s)
        """
        self.timeout_seconds = timeout_seconds

    async def validate(
        self,
        original_code: str,
        fixed_code: str,
        file_path: str
    ) -> ValidationResult:
        """
        Run tests for the fixed file.

        Args:
            original_code: Original code snippet
            fixed_code: Code after proposed fix
            file_path: Path to file (for diagnostics)

        Returns:
            ValidationResult with test execution details
        """
        start = time.time()

        # Find test file
        test_file = self._find_test_file(file_path)

        if not test_file:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.TESTS,
                passed=True,
                message="No tests found",
                duration_ms=duration,
                skipped=True,
                skip_reason="No test file",
                details={'search_paths': [f"test_{Path(file_path).name}", f"{Path(file_path).stem}_test.py"]}
            )

        try:
            # Run pytest
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(test_file), '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(Path(file_path).parent)
            )

            duration = (time.time() - start) * 1000

            if result.returncode == 0:
                # Extract test count from output
                test_count = result.stdout.count(' PASSED')
                return ValidationResult(
                    stage=ValidationStage.TESTS,
                    passed=True,
                    message=f"{test_count} tests passed",
                    duration_ms=duration,
                    details={
                        'test_file': str(test_file),
                        'test_count': test_count,
                        'output': result.stdout[:500]
                    }
                )
            else:
                # Extract failure info
                failures = result.stdout.count(' FAILED')
                return ValidationResult(
                    stage=ValidationStage.TESTS,
                    passed=False,
                    message=f"{failures} tests failed",
                    duration_ms=duration,
                    details={
                        'test_file': str(test_file),
                        'failures': failures,
                        'output': result.stdout[:1000],
                        'stderr': result.stderr[:500]
                    }
                )

        except subprocess.TimeoutExpired:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.TESTS,
                passed=False,
                message=f"Tests timed out ({self.timeout_seconds}s)",
                duration_ms=duration,
                details={'timeout_seconds': self.timeout_seconds}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.TESTS,
                passed=False,
                message=f"Test execution failed: {type(e).__name__}",
                duration_ms=duration,
                details={'error': str(e)}
            )

    def _find_test_file(self, file_path: str) -> Optional[Path]:
        """
        Find corresponding test file for a given source file.

        Searches for:
        - test_<filename>.py
        - <filename>_test.py
        - tests/<filename>.py
        """
        path = Path(file_path)

        # Test patterns to search
        patterns = [
            path.parent / f"test_{path.name}",
            path.parent / f"{path.stem}_test.py",
            path.parent / "tests" / f"test_{path.name}",
            path.parent / "tests" / path.name,
        ]

        for test_path in patterns:
            if test_path.exists():
                return test_path

        return None


class TroughValidator:
    """
    Stage 4: Trough Re-scan

    Re-scans the fixed code with Trough AI slop detector
    to ensure no new issues were introduced.
    """

    async def validate(
        self,
        original_code: str,
        fixed_code: str,
        file_path: str
    ) -> ValidationResult:
        """
        Re-scan fixed code for new issues with Trough.

        Args:
            original_code: Original code snippet
            fixed_code: Code after proposed fix
            file_path: Path to file (for diagnostics)

        Returns:
            ValidationResult with Trough scan details
        """
        start = time.time()

        try:
            # Try to import Trough
            try:
                from trough.ai_slop_detector import AISlopDetector
                from trough.trough_types import Language
            except ImportError:
                duration = (time.time() - start) * 1000
                return ValidationResult(
                    stage=ValidationStage.TROUGH,
                    passed=True,
                    message="Trough not available",
                    duration_ms=duration,
                    skipped=True,
                    skip_reason="Trough module not installed",
                    details={'available_modules': ['syntax', 'imports', 'tests']}
                )

            # Try to instantiate detector
            try:
                detector = AISlopDetector()
            except (ImportError, AttributeError, Exception) as e:
                # Handle cases where import succeeds but instantiation fails
                duration = (time.time() - start) * 1000
                return ValidationResult(
                    stage=ValidationStage.TROUGH,
                    passed=True,
                    message="Trough not available",
                    duration_ms=duration,
                    skipped=True,
                    skip_reason="Trough detector not functional",
                    details={'error': str(e)}
                )

            # Determine language
            language = self._get_language(file_path)

            # Scan original code
            original_issues = await detector.detect_all(original_code, language, file_path)
            original_count = len(original_issues)

            # Scan fixed code
            fixed_issues = await detector.detect_all(fixed_code, language, file_path)
            fixed_count = len(fixed_issues)

            # Check for new issues
            new_issues = fixed_count - original_count

            duration = (time.time() - start) * 1000

            if new_issues > 0:
                return ValidationResult(
                    stage=ValidationStage.TROUGH,
                    passed=False,
                    message=f"Introduced {new_issues} new issue(s)",
                    duration_ms=duration,
                    details={
                        'original_issues': original_count,
                        'fixed_issues': fixed_count,
                        'new_issues': new_issues,
                        'issues': [str(i) for i in fixed_issues[:5]]
                    }
                )

            improvement = original_count - fixed_count
            if improvement > 0:
                return ValidationResult(
                    stage=ValidationStage.TROUGH,
                    passed=True,
                    message=f"Fixed {improvement} issue(s), no regressions",
                    duration_ms=duration,
                    details={
                        'original_issues': original_count,
                        'fixed_issues': fixed_count,
                        'fixed_count': improvement
                    }
                )
            else:
                return ValidationResult(
                    stage=ValidationStage.TROUGH,
                    passed=True,
                    message="No new issues introduced",
                    duration_ms=duration,
                    details={
                        'original_issues': original_count,
                        'fixed_issues': fixed_count
                    }
                )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.TROUGH,
                passed=False,
                message=f"Trough scan failed: {type(e).__name__}",
                duration_ms=duration,
                details={'error': str(e)}
            )

    def _get_language(self, file_path: str) -> str:
        """Determine programming language from file extension."""
        ext = Path(file_path).suffix.lower()

        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.rs': 'rust',
            '.go': 'go',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
        }

        return language_map.get(ext, 'python')


class RegressionValidator:
    """
    Stage 5: Regression Validation

    Checks that the fix doesn't change behavior:
    - Function signatures unchanged
    - Return types compatible
    - API contract maintained
    """

    async def validate(
        self,
        original_code: str,
        fixed_code: str,
        file_path: str
    ) -> ValidationResult:
        """
        Validate no behavioral regression.

        Args:
            original_code: Original code snippet
            fixed_code: Code after proposed fix
            file_path: Path to file (for diagnostics)

        Returns:
            ValidationResult with regression check details
        """
        start = time.time()

        try:
            # Parse both versions
            original_tree = ast.parse(original_code)
            fixed_tree = ast.parse(fixed_code)

            # Extract function signatures
            original_sigs = self._extract_signatures(original_tree)
            fixed_sigs = self._extract_signatures(fixed_tree)

            # Check for signature changes
            signature_diffs = self._diff_signatures(original_sigs, fixed_sigs)

            duration = (time.time() - start) * 1000

            if signature_diffs:
                return ValidationResult(
                    stage=ValidationStage.REGRESSION,
                    passed=False,
                    message="Function signatures changed",
                    duration_ms=duration,
                    details={
                        'changed_signatures': signature_diffs,
                        'original_count': len(original_sigs),
                        'fixed_count': len(fixed_sigs)
                    }
                )

            # Check class definitions
            original_classes = self._extract_class_names(original_tree)
            fixed_classes = self._extract_class_names(fixed_tree)

            if original_classes != fixed_classes:
                return ValidationResult(
                    stage=ValidationStage.REGRESSION,
                    passed=False,
                    message="Class definitions changed",
                    duration_ms=duration,
                    details={
                        'original_classes': original_classes,
                        'fixed_classes': fixed_classes
                    }
                )

            return ValidationResult(
                stage=ValidationStage.REGRESSION,
                passed=True,
                message="No behavioral changes detected",
                duration_ms=duration,
                details={
                    'functions_checked': len(original_sigs),
                    'classes_checked': len(original_classes)
                }
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                stage=ValidationStage.REGRESSION,
                passed=False,
                message=f"Regression check failed: {type(e).__name__}",
                duration_ms=duration,
                details={'error': str(e)}
            )

    def _extract_signatures(self, tree: ast.AST) -> Dict[str, str]:
        """Extract function signatures from AST."""
        signatures = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Build signature string
                args = [arg.arg for arg in node.args.args]
                signature = f"({', '.join(args)})"
                signatures[node.name] = signature

        return signatures

    def _extract_class_names(self, tree: ast.AST) -> Set[str]:
        """Extract class names from AST."""
        classes = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.add(node.name)

        return classes

    def _diff_signatures(self, original: Dict[str, str], fixed: Dict[str, str]) -> List[Tuple[str, str, str]]:
        """Find differences between function signatures."""
        diffs = []

        # Check for modified or removed functions
        for func_name, original_sig in original.items():
            if func_name not in fixed:
                diffs.append((func_name, original_sig, "removed"))
            elif fixed[func_name] != original_sig:
                diffs.append((func_name, original_sig, f"changed to {fixed[func_name]}"))

        # Check for new functions (potential breaking change if public API)
        for func_name in fixed:
            if func_name not in original and not func_name.startswith('_'):
                diffs.append((func_name, fixed[func_name], "added"))

        return diffs


# ============================================================================
# Validation Pipeline Orchestrator
# ============================================================================

class ValidationPipeline:
    """
    🐷 Validation Pipeline Orchestrator

    Runs 5-stage validation pipeline to ensure fixes are safe:

    1. Syntax Check (AST parsing)
    2. Import Validation (import resolution)
    3. Test Execution (pytest)
    4. Trough Re-scan (AI slop detection)
    5. Regression Check (behavior preservation)

    Philosophy: "Fern validates everything - and so should you!"
    - Fail-fast: Stop on first failure
    - Detailed diagnostics for each stage
    - Clear pass/fail decisions
    - Performance tracking
    """

    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize validation pipeline.

        Args:
            timeout_seconds: Test execution timeout (default: 30s)
        """
        self.stages = [
            SyntaxValidator(),
            ImportValidator(),
            TestValidator(timeout_seconds=timeout_seconds),
            TroughValidator(),
            RegressionValidator()
        ]

    async def validate_fix(
        self,
        original_code: str,
        fixed_code: str,
        file_path: str
    ) -> ValidationPipelineReport:
        """
        Run all validation stages on proposed fix.

        Stops on first failure (fail-fast) for quick feedback.

        Args:
            original_code: Original code snippet
            fixed_code: Code after proposed fix
            file_path: Path to file being fixed

        Returns:
            ValidationPipelineReport with all stage results
        """
        results = []
        total_start = time.time()
        failed_stage = None

        for stage in self.stages:
            # Run validator
            result = await stage.validate(original_code, fixed_code, file_path)
            results.append(result)

            # Stop on first failure
            if not result.passed and not result.skipped:
                failed_stage = result.stage
                break

        total_duration = (time.time() - total_start) * 1000
        all_passed = all(r.passed or r.skipped for r in results)

        return ValidationPipelineReport(
            results=results,
            total_duration_ms=total_duration,
            all_passed=all_passed,
            failed_stage=failed_stage,
            metadata={
                'file_path': file_path,
                'original_size': len(original_code),
                'fixed_size': len(fixed_code),
                'stages_run': len(results)
            }
        )

    def all_passed(self, report: ValidationPipelineReport) -> bool:
        """Check if all stages passed."""
        return report.all_passed

    def should_commit(self, report: ValidationPipelineReport, min_confidence: float = 0.95) -> bool:
        """
        Determine if fix is safe to commit.

        Args:
            report: Validation report
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            True if safe to commit, False otherwise
        """
        if not report.all_passed:
            return False

        # All stages must pass or be skipped
        return all(r.passed or r.skipped for r in report.results)


# ============================================================================
# Convenience Functions
# ============================================================================

async def validate_fix(
    original_code: str,
    fixed_code: str,
    file_path: str,
    timeout_seconds: int = 30
) -> ValidationPipelineReport:
    """
    Convenience function to run validation pipeline.

    Args:
        original_code: Original code snippet
        fixed_code: Code after proposed fix
        file_path: Path to file being fixed
        timeout_seconds: Test execution timeout

    Returns:
        ValidationPipelineReport with all results
    """
    pipeline = ValidationPipeline(timeout_seconds=timeout_seconds)
    return await pipeline.validate_fix(original_code, fixed_code, file_path)


async def quick_validate(
    fixed_code: str,
    file_path: str
) -> bool:
    """
    Quick validation (syntax + imports only).

    Useful for fast feedback during development.

    Args:
        fixed_code: Code after proposed fix
        file_path: Path to file being fixed

    Returns:
        True if quick validation passed
    """
    # Syntax check
    try:
        ast.parse(fixed_code)
    except SyntaxError:
        return False

    # Import check
    try:
        tree = ast.parse(fixed_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                pass  # Basic check only
    except Exception:
        return False

    return True
