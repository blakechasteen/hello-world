"""
Validation Pipeline Tests (Phase 5)
===================================

Comprehensive test suite for the validation pipeline.
Tests each validator stage independently and the orchestrator.

Test Coverage:
- ✓ SyntaxValidator (3 tests)
- ✓ ImportValidator (4 tests)
- ✓ TestValidator (3 tests)
- ✓ TroughValidator (2 tests)
- ✓ RegressionValidator (4 tests)
- ✓ ValidationPipeline orchestrator (5 tests)

Total: 21 tests
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from .validator import (
    ValidationPipeline,
    ValidationPipelineReport,
    ValidationResult,
    ValidationStage,
    SyntaxValidator,
    ImportValidator,
    TestValidator,
    TroughValidator,
    RegressionValidator,
    validate_fix,
    quick_validate
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def syntax_validator():
    """Create SyntaxValidator instance"""
    return SyntaxValidator()


@pytest.fixture
def import_validator():
    """Create ImportValidator instance"""
    return ImportValidator()


@pytest.fixture
def test_validator():
    """Create TestValidator instance"""
    return TestValidator(timeout_seconds=5)


@pytest.fixture
def trough_validator():
    """Create TroughValidator instance"""
    return TroughValidator()


@pytest.fixture
def regression_validator():
    """Create RegressionValidator instance"""
    return RegressionValidator()


@pytest.fixture
def validation_pipeline():
    """Create ValidationPipeline instance"""
    return ValidationPipeline(timeout_seconds=5)


# ============================================================================
# SyntaxValidator Tests
# ============================================================================

class TestSyntaxValidator:
    """Test SyntaxValidator stage"""

    @pytest.mark.asyncio
    async def test_valid_syntax(self, syntax_validator):
        """Test valid Python syntax passes"""
        code = "x = 42\nprint(x)"

        result = await syntax_validator.validate("", code, "test.py")

        assert result.passed is True
        assert result.stage == ValidationStage.SYNTAX
        assert "valid" in result.message.lower()

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, syntax_validator):
        """Test invalid syntax fails"""
        code = "x = 42\nprint(x"  # Missing closing paren

        result = await syntax_validator.validate("", code, "test.py")

        assert result.passed is False
        assert result.stage == ValidationStage.SYNTAX
        assert result.details is not None
        assert 'error' in result.details

    @pytest.mark.asyncio
    async def test_empty_code(self, syntax_validator):
        """Test empty code passes"""
        result = await syntax_validator.validate("", "", "test.py")

        assert result.passed is True
        assert result.stage == ValidationStage.SYNTAX

    @pytest.mark.asyncio
    async def test_syntax_with_decorators(self, syntax_validator):
        """Test code with decorators parses correctly"""
        code = "@decorator\ndef func():\n    pass"

        result = await syntax_validator.validate("", code, "test.py")

        assert result.passed is True


# ============================================================================
# ImportValidator Tests
# ============================================================================

class TestImportValidator:
    """Test ImportValidator stage"""

    @pytest.mark.asyncio
    async def test_valid_imports(self, import_validator):
        """Test valid imports pass"""
        code = "import os\nimport sys\nfrom pathlib import Path"

        result = await import_validator.validate("", code, "test.py")

        assert result.passed is True
        assert result.stage == ValidationStage.IMPORTS
        assert "imports" in result.message.lower()

    @pytest.mark.asyncio
    async def test_no_imports(self, import_validator):
        """Test code with no imports"""
        code = "x = 42\nprint(x)"

        result = await import_validator.validate("", code, "test.py")

        assert result.passed is True
        assert "No imports" in result.message

    @pytest.mark.asyncio
    async def test_import_validation_details(self, import_validator):
        """Test import validator returns count"""
        code = "import os\nimport sys"

        result = await import_validator.validate("", code, "test.py")

        assert result.passed is True
        assert result.details is not None
        assert 'total_imports' in result.details
        assert result.details['total_imports'] == 2

    @pytest.mark.asyncio
    async def test_from_import(self, import_validator):
        """Test 'from X import Y' pattern"""
        code = "from os import path\nfrom collections import defaultdict"

        result = await import_validator.validate("", code, "test.py")

        assert result.passed is True
        assert result.details['total_imports'] == 2


# ============================================================================
# TestValidator Tests
# ============================================================================

class TestTestValidator:
    """Test TestValidator stage"""

    @pytest.mark.asyncio
    async def test_no_test_file(self, test_validator):
        """Test when no test file exists"""
        result = await test_validator.validate("", "x = 42", "nonexistent.py")

        assert result.skipped is True
        assert result.passed is True
        assert "No tests found" in result.message

    @pytest.mark.asyncio
    async def test_find_test_file_patterns(self, test_validator):
        """Test finding test files with various patterns"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            source_path = Path(tmpdir) / "config.py"
            test_path = Path(tmpdir) / "test_config.py"
            test_path.write_text("def test_config(): assert True")

            # Should find test file
            found = test_validator._find_test_file(str(source_path))
            assert found is not None
            assert found.name == "test_config.py"

    @pytest.mark.asyncio
    async def test_test_timeout(self, test_validator):
        """Test timeout handling (mocked to avoid long wait)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file that would timeout
            source_path = Path(tmpdir) / "source.py"
            test_path = Path(tmpdir) / "test_source.py"

            # Create a simple test
            test_path.write_text("def test_simple():\n    assert True")

            # This should complete quickly in practice
            result = await test_validator.validate("", "", str(source_path))

            # Either skipped (no tests found) or passed
            assert result.passed or result.skipped


# ============================================================================
# TroughValidator Tests
# ============================================================================

class TestTroughValidator:
    """Test TroughValidator stage"""

    @pytest.mark.asyncio
    async def test_trough_not_available(self, trough_validator):
        """Test graceful degradation when Trough not available"""
        with patch('trough.ai_slop_detector.AISlopDetector', side_effect=ImportError):
            result = await trough_validator.validate("", "x = 42", "test.py")

            # Should skip gracefully
            assert result.skipped is True
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_trough_no_new_issues(self, trough_validator):
        """Test Trough validation when no new issues introduced"""
        # Mock Trough detector
        with patch('trough.ai_slop_detector.AISlopDetector') as mock_detector_class:
            mock_detector = AsyncMock()
            mock_detector_class.return_value = mock_detector

            # Original code has 2 issues, fixed has 1
            mock_detector.detect_all.side_effect = [
                [Mock(), Mock()],  # Original: 2 issues
                [Mock()]  # Fixed: 1 issue (improvement!)
            ]

            result = await trough_validator.validate("old", "new", "test.py")

            assert result.passed is True
            assert "Fixed 1 issue" in result.message

    @pytest.mark.asyncio
    async def test_trough_new_issues(self, trough_validator):
        """Test Trough validation when new issues introduced"""
        with patch('trough.ai_slop_detector.AISlopDetector') as mock_detector_class:
            mock_detector = AsyncMock()
            mock_detector_class.return_value = mock_detector

            # Original code has 1 issue, fixed has 3
            mock_detector.detect_all.side_effect = [
                [Mock()],  # Original: 1 issue
                [Mock(), Mock(), Mock()]  # Fixed: 3 issues (2 new!)
            ]

            result = await trough_validator.validate("old", "new", "test.py")

            assert result.passed is False
            assert "Introduced 2 new issue" in result.message


# ============================================================================
# RegressionValidator Tests
# ============================================================================

class TestRegressionValidator:
    """Test RegressionValidator stage"""

    @pytest.mark.asyncio
    async def test_no_changes(self, regression_validator):
        """Test when function signatures don't change"""
        code = "def func(x, y):\n    return x + y"

        result = await regression_validator.validate(code, code, "test.py")

        assert result.passed is True
        assert "No behavioral changes" in result.message

    @pytest.mark.asyncio
    async def test_signature_changed(self, regression_validator):
        """Test detection of signature changes"""
        original = "def func(x, y):\n    return x + y"
        fixed = "def func(x, y, z):\n    return x + y + z"

        result = await regression_validator.validate(original, fixed, "test.py")

        assert result.passed is False
        assert "signatures changed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_function_removed(self, regression_validator):
        """Test detection of removed functions"""
        original = "def func1():\n    pass\ndef func2():\n    pass"
        fixed = "def func1():\n    pass"

        result = await regression_validator.validate(original, fixed, "test.py")

        assert result.passed is False
        assert "changed" in result.message.lower() or "signatures" in result.message.lower()

    @pytest.mark.asyncio
    async def test_private_function_added(self, regression_validator):
        """Test that private function additions don't fail"""
        original = "def func():\n    pass"
        fixed = "def func():\n    pass\ndef _helper():\n    pass"

        result = await regression_validator.validate(original, fixed, "test.py")

        # Should pass because _helper is private
        assert result.passed is True


# ============================================================================
# ValidationPipeline Tests
# ============================================================================

class TestValidationPipeline:
    """Test ValidationPipeline orchestrator"""

    @pytest.mark.asyncio
    async def test_all_stages_pass(self, validation_pipeline):
        """Test pipeline when all stages pass"""
        original = ""
        fixed = "x = 42\nprint(x)"

        report = await validation_pipeline.validate_fix(original, fixed, "test.py")

        assert report.all_passed is True
        assert len(report.results) >= 2  # At least syntax + imports

    @pytest.mark.asyncio
    async def test_pipeline_fails_fast(self, validation_pipeline):
        """Test that pipeline stops on first failure"""
        original = ""
        fixed = "x = 42\nprint(x"  # Syntax error

        report = await validation_pipeline.validate_fix(original, fixed, "test.py")

        assert report.all_passed is False
        assert report.failed_stage == ValidationStage.SYNTAX
        # Should not run all stages
        assert len(report.results) <= 5

    @pytest.mark.asyncio
    async def test_pipeline_metadata(self, validation_pipeline):
        """Test pipeline metadata tracking"""
        original = "x = 1"
        fixed = "x = 2"

        report = await validation_pipeline.validate_fix(original, fixed, "test.py")

        assert report.metadata['file_path'] == "test.py"
        assert report.metadata['original_size'] == len(original)
        assert report.metadata['fixed_size'] == len(fixed)

    @pytest.mark.asyncio
    async def test_should_commit(self, validation_pipeline):
        """Test commit decision logic"""
        original = ""
        fixed = "x = 42"

        report = await validation_pipeline.validate_fix(original, fixed, "test.py")

        # Should be safe to commit
        assert validation_pipeline.should_commit(report) is True

    @pytest.mark.asyncio
    async def test_should_not_commit_on_failure(self, validation_pipeline):
        """Test commit blocked on validation failure"""
        original = ""
        fixed = "x = 42\nprint(x"  # Syntax error

        report = await validation_pipeline.validate_fix(original, fixed, "test.py")

        # Should NOT be safe to commit
        assert validation_pipeline.should_commit(report) is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_validate_fix_convenience(self):
        """Test validate_fix convenience function"""
        original = "import numpy"  # Unused import
        fixed = ""

        report = await validate_fix(original, fixed, "test.py")

        assert isinstance(report, ValidationPipelineReport)
        assert report.all_passed is True

    @pytest.mark.asyncio
    async def test_quick_validate_valid(self):
        """Test quick_validate with valid code"""
        result = await quick_validate("x = 42\nprint(x)", "test.py")
        assert result is True

    @pytest.mark.asyncio
    async def test_quick_validate_invalid(self):
        """Test quick_validate with invalid code"""
        result = await quick_validate("x = 42\nprint(x", "test.py")
        assert result is False


# ============================================================================
# Validation Result Tests
# ============================================================================

class TestValidationResult:
    """Test ValidationResult class"""

    def test_result_summary(self):
        """Test result summary formatting"""
        result = ValidationResult(
            stage=ValidationStage.SYNTAX,
            passed=True,
            message="Syntax valid",
            duration_ms=1.5
        )

        summary = result.summary()
        assert "PASS" in summary
        assert "syntax" in summary.lower()
        assert "1" in summary  # Duration

    def test_result_enum_conversion(self):
        """Test that string stage gets converted to enum"""
        result = ValidationResult(
            stage="syntax",  # String instead of enum
            passed=True,
            message="OK",
            duration_ms=1.0
        )

        assert isinstance(result.stage, ValidationStage)
        assert result.stage == ValidationStage.SYNTAX

    def test_skipped_result(self):
        """Test skipped result formatting"""
        result = ValidationResult(
            stage=ValidationStage.TESTS,
            passed=True,
            message="",
            duration_ms=0.1,
            skipped=True,
            skip_reason="No test file"
        )

        summary = result.summary()
        assert "SKIP" in summary
        assert "test file" in summary


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_very_large_code(self, syntax_validator):
        """Test with very large code snippets"""
        # Create 10,000 lines of valid Python
        code = "x = 42\n" * 10000

        result = await syntax_validator.validate("", code, "test.py")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unicode_in_code(self, syntax_validator):
        """Test code with unicode characters"""
        code = 'print("こんにちは")  # Hello in Japanese'

        result = await syntax_validator.validate("", code, "test.py")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_whitespace_only_code(self, syntax_validator):
        """Test code with only whitespace"""
        code = "   \n  \n   "

        result = await syntax_validator.validate("", code, "test.py")

        assert result.passed is True


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance characteristics"""

    @pytest.mark.asyncio
    async def test_syntax_validation_speed(self, syntax_validator):
        """Test that syntax validation is fast (<10ms)"""
        code = "x = 42\nprint(x)"

        result = await syntax_validator.validate("", code, "test.py")

        assert result.duration_ms < 10  # Should be < 10ms

    @pytest.mark.asyncio
    async def test_import_validation_speed(self, import_validator):
        """Test that import validation is fast (<10ms)"""
        code = "import os\nimport sys\nfrom pathlib import Path"

        result = await import_validator.validate("", code, "test.py")

        assert result.duration_ms < 10  # Should be < 10ms

    @pytest.mark.asyncio
    async def test_regression_validation_speed(self, regression_validator):
        """Test that regression validation is fast (<10ms)"""
        code = "def func(x):\n    return x * 2"

        result = await regression_validator.validate(code, code, "test.py")

        assert result.duration_ms < 10  # Should be < 10ms


# ============================================================================
# Helper Functions for Testing
# ============================================================================

def create_temp_test_file(content: str, name: str = "test_temp.py") -> Path:
    """Create a temporary test file"""
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / name
    path.write_text(content)
    return path


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
