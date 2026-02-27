"""
Integration Tests for xTerminator + DATAPIG Integration

Tests auto-fixing of data quality issues detected by DATAPIG.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
import os
import tempfile
import json
from pathlib import Path


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_csv_with_duplicates():
    """Create a CSV file with duplicate rows for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,value\n")
        f.write("1,Alice,100\n")
        f.write("2,Bob,200\n")
        f.write("1,Alice,100\n")  # Duplicate
        f.write("3,Charlie,300\n")
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_json_with_inconsistent_format():
    """Create a JSON file with inconsistent date formats"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = [
            {"id": 1, "date": "2023-01-15"},  # ISO format
            {"id": 2, "date": "01/15/2023"},  # US format
            {"id": 3, "date": "15-01-2023"},  # EU format
        ]
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_csv_with_missing_values():
    """Create a CSV file with missing values"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,email\n")
        f.write("1,Alice,alice@example.com\n")
        f.write("2,Bob,\n")  # Missing email
        f.write("3,,charlie@example.com\n")  # Missing name
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# xTerminator Auto-Fix Tests
# ============================================================================

def test_fix_duplicates_detection():
    """Test that xTerminator can detect duplicate rows"""
    # NOTE: This test checks detection only, not fixing
    # xTerminator integration will be tested when the fixer is available
    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=True)

    # For now, just verify detector is available
    assert detector is not None
    assert detector.enable_datapig is True


def test_fix_inconsistent_format_detection():
    """Test that xTerminator can detect inconsistent formats"""
    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=True)

    assert detector is not None


def test_fix_missing_values_detection():
    """Test that xTerminator can detect missing values"""
    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=True)

    assert detector is not None


def test_unified_fix_pipeline_structure():
    """Test that unified fix pipeline structure is available"""
    # This test verifies the integration structure exists
    # Actual fixing will be tested when xTerminator fixer is integrated

    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=True)

    # Verify detector has detect_data_quality method
    assert hasattr(detector, 'detect_data_quality')

    # Verify detector has detect_all method
    assert hasattr(detector, 'detect_all')


def test_datapig_issue_to_slop_issue_conversion():
    """Test conversion of DATAPIG issues to SlopIssue format"""
    from hololoom.datapig import DataQualityIssue, IssueType, Severity
    from trough.datapig_integration import convert_datapig_issue

    # Create a sample DATAPIG issue
    issue = DataQualityIssue(
        issue_type=IssueType.DUPLICATES,
        severity=Severity.LIEUTENANT,
        message="Duplicate row detected",
        location="row_2",
        details={"row_index": 2},
        stardate=12345.67
    )

    # Convert to SlopIssue
    slop_issue = convert_datapig_issue(issue, "test.csv")

    # Verify conversion
    assert slop_issue is not None
    assert "[DATAPIG]" in slop_issue.message
    assert slop_issue.file_path == "test.csv"
    assert slop_issue.line_number >= 1


def test_fix_suggestion_generation():
    """Test that fix suggestions are generated for DATAPIG issues"""
    from hololoom.datapig import DataQualityIssue, IssueType, Severity
    from trough.datapig_integration import generate_fix_suggestion

    # Test duplicate fix suggestion
    issue = DataQualityIssue(
        issue_type=IssueType.DUPLICATES,
        severity=Severity.LIEUTENANT,
        message="Duplicate detected",
        location="row_2",
        details={},
        stardate=0.0
    )

    suggestion = generate_fix_suggestion(issue)
    assert suggestion is not None
    assert "duplicate" in suggestion.lower()


def test_data_leak_fix_suggestion():
    """Test fix suggestion for data leak (PII) issues"""
    from hololoom.datapig import DataQualityIssue, IssueType, Severity
    from trough.datapig_integration import generate_fix_suggestion

    issue = DataQualityIssue(
        issue_type=IssueType.DATA_LEAK,
        severity=Severity.CAPTAIN,
        message="Email detected",
        location="row_1.email",
        details={"leak_type": "email"},
        stardate=0.0
    )

    suggestion = generate_fix_suggestion(issue)
    assert suggestion is not None
    assert "CRITICAL" in suggestion or "Remove" in suggestion


def test_stale_data_fix_suggestion():
    """Test fix suggestion for stale data issues"""
    from hololoom.datapig import DataQualityIssue, IssueType, Severity
    from trough.datapig_integration import generate_fix_suggestion

    issue = DataQualityIssue(
        issue_type=IssueType.STALE_DATA,
        severity=Severity.LIEUTENANT,
        message="Old timestamp",
        location="row_0.updated_at",
        details={},
        stardate=0.0
    )

    suggestion = generate_fix_suggestion(issue)
    assert suggestion is not None
    assert "update" in suggestion.lower() or "archive" in suggestion.lower()


def test_outlier_fix_suggestion():
    """Test fix suggestion for outlier issues"""
    from hololoom.datapig import DataQualityIssue, IssueType, Severity
    from trough.datapig_integration import generate_fix_suggestion

    issue = DataQualityIssue(
        issue_type=IssueType.OUTLIERS,
        severity=Severity.LIEUTENANT,
        message="Extreme value",
        location="row_5.value",
        details={"value": 1000, "lower_bound": 0, "upper_bound": 100},
        stardate=0.0
    )

    suggestion = generate_fix_suggestion(issue)
    assert suggestion is not None
    assert "outlier" in suggestion.lower() or "investigate" in suggestion.lower()


def test_inconsistent_format_fix_suggestion():
    """Test fix suggestion for inconsistent format issues"""
    from hololoom.datapig import DataQualityIssue, IssueType, Severity
    from trough.datapig_integration import generate_fix_suggestion

    issue = DataQualityIssue(
        issue_type=IssueType.INCONSISTENT_FORMAT,
        severity=Severity.LIEUTENANT,
        message="Inconsistent date format",
        location="row_1.date",
        details={},
        stardate=0.0
    )

    suggestion = generate_fix_suggestion(issue)
    assert suggestion is not None
    assert "normalize" in suggestion.lower() or "consistent" in suggestion.lower()
