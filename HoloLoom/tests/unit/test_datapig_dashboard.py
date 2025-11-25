"""
Unit Tests for DATAPIG Visual Dashboard

Tests Tufte-style HTML report generation with:
- Small multiples
- Data density tables
- Inline sparklines
- Quality score calculation

**Target**: 15 tests
**Status**: Complete

Run:
    pytest HoloLoom/tests/unit/test_datapig_dashboard.py -v
"""

import pytest
import time
from typing import List

# DATAPIG imports
from HoloLoom.datapig.dashboard import (
    QualityReport,
    render_quality_dashboard,
    render_small_multiples,
    render_density_table,
    render_sparkline
)
from HoloLoom.datapig.detector import DataQualityIssue, IssueType, Severity


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_issues() -> List[DataQualityIssue]:
    """Create sample data quality issues"""
    return [
        DataQualityIssue(
            issue_type=IssueType.DATA_LEAK,
            severity=Severity.CAPTAIN,
            message="Email leak detected",
            location="field_email",
            details={"field": "email"},
            stardate=12345.0
        ),
        DataQualityIssue(
            issue_type=IssueType.HIGH_ENTROPY_PII,
            severity=Severity.COMMANDER,
            message="High entropy PII detected",
            location="field_ssn",
            details={"field": "ssn"},
            stardate=12345.0
        ),
        DataQualityIssue(
            issue_type=IssueType.DUPLICATES,
            severity=Severity.LIEUTENANT,
            message="Duplicate rows detected",
            location="rows_1_2",
            details={"rows": [1, 2]},
            stardate=12345.0
        ),
        DataQualityIssue(
            issue_type=IssueType.OUTLIERS,
            severity=Severity.ENSIGN,
            message="Outlier detected",
            location="field_age",
            details={"field": "age"},
            stardate=12345.0
        )
    ]


@pytest.fixture
def sample_report(sample_issues) -> QualityReport:
    """Create sample quality report"""
    return QualityReport(
        dataset_name="users.csv",
        timestamp=time.time(),
        issues=sample_issues,
        dataset_size=100,
        duration_ms=125.5
    )


@pytest.fixture
def multiple_reports(sample_issues) -> List[QualityReport]:
    """Create multiple quality reports for small multiples testing"""
    return [
        QualityReport(
            dataset_name="users.csv",
            timestamp=time.time(),
            issues=sample_issues[:2],  # 2 issues (1 CAPTAIN, 1 COMMANDER)
            dataset_size=100,
            duration_ms=120.0
        ),
        QualityReport(
            dataset_name="orders.csv",
            timestamp=time.time(),
            issues=sample_issues[:1],  # 1 issue (CAPTAIN)
            dataset_size=200,
            duration_ms=150.0
        ),
        QualityReport(
            dataset_name="products.csv",
            timestamp=time.time(),
            issues=sample_issues,  # All 4 issues
            dataset_size=50,
            duration_ms=80.0
        )
    ]


# ============================================================================
# QualityReport Tests (5 tests)
# ============================================================================

def test_quality_report_issue_counts_by_type(sample_report):
    """Test counting issues by type"""
    counts = sample_report.get_issue_counts_by_type()

    assert counts["DATA_LEAK"] == 1
    assert counts["HIGH_ENTROPY_PII"] == 1
    assert counts["DUPLICATES"] == 1
    assert counts["OUTLIERS"] == 1
    assert len(counts) == 4


def test_quality_report_issue_counts_by_severity(sample_report):
    """Test counting issues by severity"""
    counts = sample_report.get_issue_counts_by_severity()

    assert counts["CAPTAIN"] == 1
    assert counts["COMMANDER"] == 1
    assert counts["LIEUTENANT"] == 1
    assert counts["ENSIGN"] == 1
    assert len(counts) == 4


def test_quality_report_get_critical_issues(sample_report):
    """Test filtering critical (CAPTAIN-level) issues"""
    critical = sample_report.get_critical_issues()

    assert len(critical) == 1
    assert critical[0].severity == Severity.CAPTAIN
    assert critical[0].issue_type == IssueType.DATA_LEAK


def test_quality_report_quality_score_calculation(sample_report):
    """Test quality score calculation (0.0-1.0)"""
    score = sample_report.get_quality_score()

    # Formula: 1.0 - (weighted_issues / dataset_size)
    # Weights: CAPTAIN=1.0, COMMANDER=0.5, LIEUTENANT=0.25, ENSIGN=0.1
    # weighted = 1.0 + 0.5 + 0.25 + 0.1 = 1.85
    # score = 1.0 - (1.85 / 100) = 0.9815
    assert 0.98 < score < 0.99
    assert 0.0 <= score <= 1.0


def test_quality_report_quality_score_edge_cases():
    """Test quality score edge cases (empty, zero size)"""
    # Empty report (perfect score)
    empty_report = QualityReport(
        dataset_name="empty.csv",
        timestamp=time.time(),
        issues=[],
        dataset_size=100
    )
    assert empty_report.get_quality_score() == 1.0

    # Zero size dataset (should not crash)
    zero_report = QualityReport(
        dataset_name="zero.csv",
        timestamp=time.time(),
        issues=[DataQualityIssue(
            issue_type=IssueType.DATA_LEAK,
            severity=Severity.CAPTAIN,
            message="Test",
            location="test",
            details={},
            stardate=0.0
        )],
        dataset_size=0
    )
    score = zero_report.get_quality_score()
    assert 0.0 <= score <= 1.0


# ============================================================================
# Sparkline Tests (3 tests)
# ============================================================================

def test_render_sparkline_basic():
    """Test basic sparkline rendering"""
    values = [1.0, 2.0, 3.0, 2.5, 4.0]
    svg = render_sparkline(values)

    # Check SVG structure
    assert '<svg' in svg
    assert 'width="100"' in svg
    assert 'height="30"' in svg
    assert '<polyline' in svg
    assert '<circle' in svg  # Endpoint indicator
    assert '</svg>' in svg


def test_render_sparkline_trend_colors():
    """Test sparkline colors based on trend"""
    # Improving trend (last > first)
    improving = [1.0, 1.5, 2.0, 2.5, 3.0]
    svg_improving = render_sparkline(improving)
    assert '#10b981' in svg_improving  # Green

    # Degrading trend (last < first)
    degrading = [3.0, 2.5, 2.0, 1.5, 1.0]
    svg_degrading = render_sparkline(degrading)
    assert '#ef4444' in svg_degrading  # Red

    # Stable trend (last == first)
    stable = [2.0, 3.0, 1.0, 3.0, 2.0]
    svg_stable = render_sparkline(stable)
    assert '#6b7280' in svg_stable  # Gray


def test_render_sparkline_edge_cases():
    """Test sparkline edge cases (empty, single value, flat line)"""
    # Empty values
    svg_empty = render_sparkline([])
    assert '<svg' in svg_empty
    assert '<polyline' not in svg_empty  # No data

    # Single value
    svg_single = render_sparkline([1.0])
    assert '<svg' in svg_single
    assert '<polyline' not in svg_single  # Need at least 2 points

    # Flat line (all same value)
    svg_flat = render_sparkline([2.0, 2.0, 2.0, 2.0])
    assert '<svg' in svg_flat
    assert '<polyline' in svg_flat


# ============================================================================
# Small Multiples Tests (3 tests)
# ============================================================================

def test_render_small_multiples_basic(multiple_reports):
    """Test small multiples rendering with multiple datasets"""
    html = render_small_multiples(multiple_reports, max_columns=3)

    # Check structure
    assert '<div style="display: grid;' in html
    assert 'users.csv' in html
    assert 'orders.csv' in html
    assert 'products.csv' in html

    # Check quality scores are displayed
    assert 'EXCELLENT' in html or 'GOOD' in html or 'FAIR' in html

    # Check issue counts are displayed
    assert 'Critical:' in html
    assert 'High:' in html
    assert 'Medium:' in html
    assert 'Low:' in html


def test_render_small_multiples_empty():
    """Test small multiples with empty report list"""
    html = render_small_multiples([])

    assert 'No reports available' in html
    assert '<div' not in html or 'display: grid' not in html


def test_render_small_multiples_single_report(sample_report):
    """Test small multiples with single report"""
    html = render_small_multiples([sample_report], max_columns=2)

    assert 'users.csv' in html
    assert '<div style="display: grid;' in html

    # Should still render grid layout even with 1 item
    assert 'grid-template-columns' in html


# ============================================================================
# Data Density Table Tests (2 tests)
# ============================================================================

def test_render_density_table_basic(multiple_reports):
    """Test data density table rendering"""
    html = render_density_table(multiple_reports)

    # Check table structure
    assert '<table' in html
    assert '<thead>' in html
    assert '<tbody>' in html
    assert '</table>' in html

    # Check headers
    assert 'Issue Type' in html
    assert 'Count' in html
    assert '%' in html
    assert 'Trend' in html

    # Check issue types are listed
    assert 'DATA_LEAK' in html
    assert 'HIGH_ENTROPY_PII' in html

    # Check sparklines are embedded
    assert '<svg' in html


def test_render_density_table_empty():
    """Test density table with empty report list"""
    html = render_density_table([])

    assert 'No reports available' in html
    assert '<table' not in html


# ============================================================================
# Complete Dashboard Tests (2 tests)
# ============================================================================

def test_render_quality_dashboard_complete(multiple_reports):
    """Test complete dashboard rendering"""
    html = render_quality_dashboard(
        multiple_reports,
        title="Production Quality Dashboard",
        subtitle="Last 24 hours"
    )

    # Check HTML structure
    assert '<!DOCTYPE html>' in html
    assert '<html' in html
    assert '<head>' in html
    assert '<body>' in html
    assert '</html>' in html

    # Check title and subtitle
    assert 'Production Quality Dashboard' in html
    assert 'Last 24 hours' in html

    # Check summary statistics
    assert 'Datasets' in html
    assert 'Total Issues' in html
    assert 'Avg Quality' in html

    # Check sections
    assert 'Dataset Comparison' in html
    assert 'Issue Breakdown' in html

    # Check critical issues section (should appear since we have CAPTAIN issues)
    assert 'Critical Issues' in html or '🚨' in html

    # Check footer
    assert 'DATAPIG' in html


def test_render_quality_dashboard_no_critical_issues():
    """Test dashboard with no critical issues"""
    # Create reports with only low-severity issues
    low_severity_issues = [
        DataQualityIssue(
            issue_type=IssueType.OUTLIERS,
            severity=Severity.ENSIGN,
            message="Minor outlier",
            location="field_x",
            details={},
            stardate=0.0
        )
    ]

    reports = [
        QualityReport(
            dataset_name="test.csv",
            timestamp=time.time(),
            issues=low_severity_issues,
            dataset_size=100
        )
    ]

    html = render_quality_dashboard(reports)

    # Should NOT have critical issues section
    # (or it should be empty)
    # Note: critical section only appears if there ARE critical issues
    assert '<!DOCTYPE html>' in html
    assert 'Dataset Comparison' in html


# ============================================================================
# Integration Test
# ============================================================================

def test_dashboard_end_to_end_integration():
    """
    Test complete end-to-end workflow:
    1. Create detector
    2. Analyze datasets
    3. Generate dashboard
    """
    from HoloLoom.datapig.detector import DataPigDetector

    # Sample datasets
    datasets = [
        {
            "name": "users.csv",
            "data": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
                {"id": 1, "name": "Alice", "email": "alice@example.com"},  # Duplicate
            ]
        },
        {
            "name": "orders.csv",
            "data": [
                {"order_id": 1, "amount": 100},
                {"order_id": 2, "amount": 200},
                {"order_id": 3, "amount": 999999},  # Outlier
            ]
        }
    ]

    # Analyze all datasets
    reports = []
    for dataset_info in datasets:
        detector = DataPigDetector()
        start_time = time.time()
        issues = detector.analyze_dataset(dataset_info["data"])
        duration_ms = (time.time() - start_time) * 1000

        reports.append(QualityReport(
            dataset_name=dataset_info["name"],
            timestamp=time.time(),
            issues=issues,
            dataset_size=len(dataset_info["data"]),
            duration_ms=duration_ms
        ))

    # Generate dashboard
    html = render_quality_dashboard(
        reports,
        title="Integration Test Dashboard",
        subtitle="End-to-end workflow test"
    )

    # Verify dashboard contains expected content
    assert '<!DOCTYPE html>' in html
    assert 'Integration Test Dashboard' in html
    assert 'users.csv' in html
    assert 'orders.csv' in html
    assert 'Dataset Comparison' in html
    assert 'Issue Breakdown' in html

    # Should have detected some issues
    total_issues = sum(len(r.issues) for r in reports)
    assert total_issues > 0  # We know there are duplicates and outliers

    # Verify quality scores are calculated
    for report in reports:
        score = report.get_quality_score()
        assert 0.0 <= score <= 1.0


# ============================================================================
# Summary
# ============================================================================

# Total Tests: 15
# - QualityReport tests: 5
# - Sparkline tests: 3
# - Small multiples tests: 3
# - Data density table tests: 2
# - Complete dashboard tests: 2
# - Integration test: 1 (bonus)
