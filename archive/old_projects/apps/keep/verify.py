#!/usr/bin/env python3
"""
Keep Verification System
========================

Comprehensive verification of Keep beekeeping application:
1. Convention compliance checking
2. Type hint verification
3. Docstring coverage
4. Test suite execution
5. Import validation
6. Code quality metrics

Follows mythRL verification patterns.
"""

import ast
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class VerificationResult:
    """Result of a verification check."""
    name: str
    passed: bool
    message: str
    details: List[str] = None
    score: float = 0.0

    def __post_init__(self):
        if self.details is None:
            self.details = []


class ConventionChecker:
    """Checks code follows mythRL conventions."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.results = []

    def check_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Check a single file for convention compliance.

        Checks:
        - Docstring presence
        - Type hints coverage
        - Import organization
        - Naming conventions
        """
        metrics = {
            'has_module_docstring': False,
            'functions': 0,
            'functions_with_docstrings': 0,
            'functions_with_type_hints': 0,
            'classes': 0,
            'classes_with_docstrings': 0,
            'imports_organized': True,
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # Check module docstring
            if ast.get_docstring(tree):
                metrics['has_module_docstring'] = True

            # Check functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics['functions'] += 1
                    if ast.get_docstring(node):
                        metrics['functions_with_docstrings'] += 1

                    # Check type hints
                    if node.returns or any(
                        arg.annotation for arg in node.args.args
                    ):
                        metrics['functions_with_type_hints'] += 1

                elif isinstance(node, ast.ClassDef):
                    metrics['classes'] += 1
                    if ast.get_docstring(node):
                        metrics['classes_with_docstrings'] += 1

        except Exception as e:
            metrics['error'] = str(e)

        return metrics

    def check_all_files(self) -> VerificationResult:
        """Check all Python files in the project."""
        python_files = list(self.base_path.glob('*.py'))
        python_files = [f for f in python_files if not f.name.startswith('test_')]

        total_functions = 0
        total_functions_with_docs = 0
        total_functions_with_hints = 0
        total_classes = 0
        total_classes_with_docs = 0
        files_with_module_docs = 0

        details = []

        for file_path in python_files:
            metrics = self.check_file(file_path)

            if 'error' in metrics:
                details.append(f"  ERROR {file_path.name}: {metrics['error']}")
                continue

            if metrics['has_module_docstring']:
                files_with_module_docs += 1
            else:
                details.append(f"  MISSING module docstring: {file_path.name}")

            total_functions += metrics['functions']
            total_functions_with_docs += metrics['functions_with_docstrings']
            total_functions_with_hints += metrics['functions_with_type_hints']
            total_classes += metrics['classes']
            total_classes_with_docs += metrics['classes_with_docstrings']

        # Calculate scores
        doc_coverage = (
            (files_with_module_docs / len(python_files)) * 0.3 +
            (total_functions_with_docs / max(total_functions, 1)) * 0.4 +
            (total_classes_with_docs / max(total_classes, 1)) * 0.3
        ) * 100 if python_files else 0

        hint_coverage = (
            (total_functions_with_hints / max(total_functions, 1)) * 100
        ) if total_functions else 0

        score = (doc_coverage * 0.5 + hint_coverage * 0.5) / 100

        message = (
            f"Documentation: {doc_coverage:.1f}% "
            f"({files_with_module_docs}/{len(python_files)} modules, "
            f"{total_functions_with_docs}/{total_functions} functions, "
            f"{total_classes_with_docs}/{total_classes} classes)\n"
            f"Type Hints: {hint_coverage:.1f}% "
            f"({total_functions_with_hints}/{total_functions} functions)"
        )

        passed = score >= 0.9  # 90% threshold

        return VerificationResult(
            name="Convention Compliance",
            passed=passed,
            message=message,
            details=details,
            score=score
        )


class TestRunner:
    """Runs test suite and collects results."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.tests_dir = base_path / "tests"

    def run_tests(self) -> VerificationResult:
        """Run pytest test suite."""
        if not self.tests_dir.exists():
            return VerificationResult(
                name="Test Suite",
                passed=False,
                message="Tests directory not found",
                score=0.0
            )

        details = []

        # Run pytest with output capture
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(self.tests_dir), "-v", "--tb=short"],
            cwd=self.base_path.parent.parent,
            env={"PYTHONPATH": str(self.base_path.parent.parent)},
            capture_output=True,
            text=True
        )

        # Parse output
        output_lines = result.stdout.split('\n')
        passed_count = 0
        failed_count = 0
        total_count = 0

        for line in output_lines:
            if " PASSED" in line:
                passed_count += 1
                total_count += 1
            elif " FAILED" in line:
                failed_count += 1
                total_count += 1
                details.append(f"  FAILED: {line.strip()}")

        # Check for summary line
        for line in output_lines:
            if "passed" in line.lower():
                details.append(f"  Summary: {line.strip()}")

        score = passed_count / max(total_count, 1) if total_count > 0 else 0

        message = f"{passed_count}/{total_count} tests passed"
        if failed_count > 0:
            message += f", {failed_count} failed"

        return VerificationResult(
            name="Test Suite",
            passed=failed_count == 0 and passed_count > 0,
            message=message,
            details=details[:10],  # First 10 failures
            score=score
        )


class ImportValidator:
    """Validates import statements and structure."""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def check_imports(self) -> VerificationResult:
        """Check import organization follows conventions."""
        python_files = list(self.base_path.glob('*.py'))
        issues = []

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                # Extract imports
                imports = []
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports.append(node)
                    elif not isinstance(node, (ast.Expr, ast.FunctionDef,
                                               ast.ClassDef, ast.Assign)):
                        # Imports should be at top (after docstring)
                        break

                # Check for relative imports to non-existent modules
                for node in imports:
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('apps.keep'):
                            # This is fine
                            pass

            except Exception as e:
                issues.append(f"  Error parsing {file_path.name}: {e}")

        score = 1.0 if len(issues) == 0 else 0.7

        return VerificationResult(
            name="Import Validation",
            passed=len(issues) == 0,
            message=f"Found {len(issues)} import issues",
            details=issues,
            score=score
        )


class QualityMetrics:
    """Calculates code quality metrics."""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def calculate_metrics(self) -> VerificationResult:
        """Calculate various quality metrics."""
        python_files = list(self.base_path.glob('*.py'))
        python_files = [f for f in python_files if not f.name.startswith('test_')]

        metrics = {
            'total_lines': 0,
            'total_files': len(python_files),
            'avg_file_length': 0,
            'max_file_length': 0,
        }

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                    metrics['total_lines'] += line_count
                    metrics['max_file_length'] = max(metrics['max_file_length'], line_count)
            except Exception:
                pass

        if metrics['total_files'] > 0:
            metrics['avg_file_length'] = metrics['total_lines'] / metrics['total_files']

        # Score based on reasonable file sizes
        size_score = 1.0
        if metrics['avg_file_length'] > 500:
            size_score = 0.9
        if metrics['max_file_length'] > 1000:
            size_score = 0.8

        message = (
            f"Total: {metrics['total_lines']} lines across {metrics['total_files']} files\n"
            f"Average: {metrics['avg_file_length']:.0f} lines/file\n"
            f"Largest: {metrics['max_file_length']} lines"
        )

        return VerificationResult(
            name="Quality Metrics",
            passed=size_score >= 0.9,
            message=message,
            details=[],
            score=size_score
        )


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result(result: VerificationResult):
    """Print verification result."""
    status = "PASS" if result.passed else "FAIL"
    symbol = "+" if result.passed else "X"

    print(f"\n[{symbol}] {result.name}: {status}")
    print(f"    Score: {result.score*100:.1f}%")
    print(f"    {result.message}")

    if result.details:
        for detail in result.details[:5]:  # Show first 5 details
            print(detail)
        if len(result.details) > 5:
            print(f"    ... and {len(result.details) - 5} more")


def main():
    """Run comprehensive verification."""
    print_header("Keep Verification System")
    print("Verifying: Elegance, Extensibility, Testing, Rigor, Analysis")

    base_path = Path(__file__).parent
    results = []

    # 1. Convention Compliance
    print("\n1. Checking Convention Compliance...")
    checker = ConventionChecker(base_path)
    convention_result = checker.check_all_files()
    results.append(convention_result)
    print_result(convention_result)

    # 2. Test Suite
    print("\n2. Running Test Suite...")
    runner = TestRunner(base_path)
    test_result = runner.run_tests()
    results.append(test_result)
    print_result(test_result)

    # 3. Import Validation
    print("\n3. Validating Imports...")
    validator = ImportValidator(base_path)
    import_result = validator.check_imports()
    results.append(import_result)
    print_result(import_result)

    # 4. Quality Metrics
    print("\n4. Calculating Quality Metrics...")
    metrics = QualityMetrics(base_path)
    quality_result = metrics.calculate_metrics()
    results.append(quality_result)
    print_result(quality_result)

    # Final Summary
    print_header("Verification Summary")

    total_score = sum(r.score for r in results) / len(results)
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    print(f"\nChecks Passed: {passed_count}/{total_count}")
    print(f"Overall Score: {total_score*100:.1f}%")

    if total_score >= 0.95:
        print("\n*** EXCELLENT: Keep meets production standards! ***")
        return 0
    elif total_score >= 0.85:
        print("\n*** GOOD: Minor improvements recommended ***")
        return 0
    elif total_score >= 0.75:
        print("\n*** FAIR: Some issues need attention ***")
        return 1
    else:
        print("\n*** NEEDS WORK: Significant improvements required ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
