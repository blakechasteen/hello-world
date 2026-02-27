"""
Unit Tests for DATAPIG Issue Reporting

Tests DataQualityIssue structure and formatting.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
from HoloLoom.datapig import DataQualityIssue, IssueType, Severity


# ============================================================================
# DataQualityIssue Tests
# ============================================================================

def test_issue_creation():
    """Test creating a DataQualityIssue"""
    issue = DataQualityIssue(
        issue_type=IssueType.SCHEMA_DRIFT,
        severity=Severity.CAPTAIN,
        message="Test issue",
        location="row_0.field",
        details={"test": "value"},
        stardate=12345.67
    )

    assert issue.issue_type == IssueType.SCHEMA_DRIFT
    assert issue.severity == Severity.CAPTAIN
    assert issue.message == "Test issue"
    assert issue.location == "row_0.field"
    assert issue.details == {"test": "value"}
    assert issue.stardate == 12345.67


def test_issue_str_formatting():
    """Test issue string representation"""
    issue = DataQualityIssue(
        issue_type=IssueType.DATA_LEAK,
        severity=Severity.CAPTAIN,
        message="PII detected",
        location="row_5.email",
        details={"leak_type": "email"},
        stardate=12345.67
    )

    issue_str = str(issue)

    assert "CAPTAIN" in issue_str
    assert "DATA_LEAK" in issue_str
    assert "PII detected" in issue_str
    assert "row_5.email" in issue_str


# ============================================================================
# Severity Tests
# ============================================================================

def test_severity_levels():
    """Test all severity levels exist"""
    assert Severity.CAPTAIN.value == "CAPTAIN"
    assert Severity.COMMANDER.value == "COMMANDER"
    assert Severity.LIEUTENANT.value == "LIEUTENANT"
    assert Severity.ENSIGN.value == "ENSIGN"


def test_severity_hierarchy():
    """Test severity ordering (for future priority sorting)"""
    severities = [
        Severity.CAPTAIN,
        Severity.COMMANDER,
        Severity.LIEUTENANT,
        Severity.ENSIGN
    ]

    # Just verify all severities exist and are distinct
    assert len(set(severities)) == 4


# ============================================================================
# IssueType Tests
# ============================================================================

def test_all_issue_types_exist():
    """Test all 10 issue types exist"""
    expected_types = [
        IssueType.SCHEMA_DRIFT,
        IssueType.DATA_LEAK,
        IssueType.STALE_DATA,
        IssueType.DUPLICATES,
        IssueType.OUTLIERS,
        IssueType.INCONSISTENT_FORMAT,
        IssueType.MISSING_RELATIONS,
        IssueType.DISTRIBUTION_SHIFT,
        IssueType.SAMPLING_BIAS,
        IssueType.LABEL_NOISE,
    ]

    # All 10 types should be distinct
    assert len(set(expected_types)) == 10


def test_issue_type_values():
    """Test issue type values match enum names"""
    assert IssueType.SCHEMA_DRIFT.value == "SCHEMA_DRIFT"
    assert IssueType.DATA_LEAK.value == "DATA_LEAK"
    assert IssueType.STALE_DATA.value == "STALE_DATA"


# ============================================================================
# Issue Details Tests
# ============================================================================

def test_issue_details_schema_drift():
    """Test schema drift issue details"""
    issue = DataQualityIssue(
        issue_type=IssueType.SCHEMA_DRIFT,
        severity=Severity.LIEUTENANT,
        message="Missing field",
        location="row_1",
        details={
            "missing_fields": ["email"],
            "row_index": 1
        },
        stardate=0.0
    )

    assert "missing_fields" in issue.details
    assert "email" in issue.details["missing_fields"]


def test_issue_details_data_leak():
    """Test data leak issue details"""
    issue = DataQualityIssue(
        issue_type=IssueType.DATA_LEAK,
        severity=Severity.CAPTAIN,
        message="Email detected",
        location="row_0.contact",
        details={
            "leak_type": "email",
            "field": "contact",
            "row_index": 0
        },
        stardate=0.0
    )

    assert issue.details["leak_type"] == "email"
    assert issue.details["field"] == "contact"


def test_issue_details_outliers():
    """Test outlier issue details"""
    issue = DataQualityIssue(
        issue_type=IssueType.OUTLIERS,
        severity=Severity.LIEUTENANT,
        message="Extreme value detected",
        location="row_5.value",
        details={
            "value": 1000,
            "lower_bound": -10.0,
            "upper_bound": 50.0,
            "field": "value"
        },
        stardate=0.0
    )

    assert issue.details["value"] == 1000
    assert "lower_bound" in issue.details
    assert "upper_bound" in issue.details
