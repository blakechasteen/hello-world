"""
Integration Tests for QA Department + DATAPIG Integration

Tests DATAPIG integration with HoloLoom's Quality Assurance Department
for orchestrated code + data quality workflows.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
import os
import tempfile
import json


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_mixed_quality_dataset():
    """Create a dataset with multiple quality issues for department testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,email,value\n")
        f.write("1,Alice,alice@example.com,100\n")
        f.write("2,Bob,,200\n")  # Missing email
        f.write("1,Alice,alice@example.com,100\n")  # Duplicate
        f.write("3,Charlie,charlie@example.com,9999\n")  # Outlier
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# QA Department Integration Tests
# ============================================================================

def test_qa_department_datapig_availability():
    """Test that DATAPIG is available for QA department workflows"""
    from HoloLoom.datapig import DataPigDetector

    detector = DataPigDetector(enable_verbose=False)
    assert detector is not None


def test_qa_department_unified_detection():
    """Test QA department can use unified code+data detection"""
    from trough.datapig_integration import UnifiedDetector

    # QA department should be able to create unified detector
    detector = UnifiedDetector(enable_datapig=True)

    assert detector is not None
    assert detector.enable_datapig is True
    assert hasattr(detector, 'detect_data_quality')
    assert hasattr(detector, 'detect_all')


def test_qa_department_issue_classification():
    """Test QA department can classify issues by severity"""
    from HoloLoom.datapig import DataPigDetector, Severity

    detector = DataPigDetector(enable_verbose=False)

    # Test data with various issues
    data = [
        {"id": 1, "name": "Alice", "ssn": "123-45-6789"},  # DATA_LEAK (CAPTAIN severity)
        {"id": 2, "name": "Bob", "value": 10},
        {"id": 2, "name": "Bob", "value": 10},  # Duplicate (LIEUTENANT severity)
    ]

    issues = detector.analyze_dataset(data)

    # Should have detected issues
    assert len(issues) >= 1

    # Issues should have severity levels
    severities = [issue.severity for issue in issues]
    assert all(isinstance(s, Severity) for s in severities)


def test_qa_department_batch_processing():
    """Test QA department can batch process multiple datasets"""
    from HoloLoom.datapig import DataPigDetector

    detector = DataPigDetector(enable_verbose=False)

    # Simulate batch processing of multiple datasets
    datasets = [
        [{"id": 1, "value": 10}],  # Clean data
        [{"id": 1, "value": 10}, {"id": 1, "value": 10}],  # Has duplicates
        [{"id": 1, "email": "test@example.com"}],  # Has PII
    ]

    results = []
    for dataset in datasets:
        issues = detector.analyze_dataset(dataset)
        results.append({
            "dataset_size": len(dataset),
            "issues_found": len(issues),
            "issues": issues
        })

    # Should have processed all datasets
    assert len(results) == 3

    # Second dataset should have issues (duplicates)
    assert results[1]["issues_found"] >= 1

    # Third dataset should have issues (PII)
    assert results[2]["issues_found"] >= 1


def test_qa_department_configuration_presets():
    """Test QA department can use different detector configurations"""
    from HoloLoom.datapig import DataPigDetector
    from HoloLoom.datapig.config import PresetConfig

    # Test that different presets can be created
    configs = [
        PresetConfig.default(),
        PresetConfig.strict(),
        PresetConfig.lenient(),
        PresetConfig.fast(),
        PresetConfig.pii_focused(),
        PresetConfig.ml_validation(),
    ]

    # All configs should be valid
    for config in configs:
        assert config is not None
        assert config.stale_threshold_days > 0

    # QA department should be able to create detector with any preset
    # (Actual configuration not yet implemented in detector constructor)
    detector = DataPigDetector(enable_verbose=False)
    assert detector is not None


def test_qa_department_reporting_structure(temp_mixed_quality_dataset):
    """Test QA department can generate structured quality reports"""
    from trough.datapig_integration import analyze_file_unified

    # Analyze file and get structured report
    report = analyze_file_unified(temp_mixed_quality_dataset)

    # Verify report structure for QA department
    assert "file_path" in report
    assert "file_type" in report
    assert "issues" in report
    assert "summary" in report

    # Summary should have useful metrics
    summary = report["summary"]
    assert "total_issues" in summary
    assert "by_severity" in summary
    assert "by_category" in summary

    # Should have detected multiple issues in mixed quality dataset
    assert summary["total_issues"] >= 2  # At least duplicates + outliers/missing
