"""
Integration Tests for Trough + DATAPIG Integration

Tests unified code + data quality detection pipeline.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
import os
import tempfile
from pathlib import Path
from HoloLoom.datapig import IssueType, Severity


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,email\n")
        f.write("1,Alice,alice@example.com\n")
        f.write("2,Bob,bob@example.com\n")
        f.write("1,Alice,alice@example.com\n")  # Duplicate
        temp_path = f.name

    # File is now closed and flushed to disk
    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing"""
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 12},
            {"id": 3, "value": 1000}  # Outlier
        ]
        json.dump(data, f)
        temp_path = f.name

    # File is now closed and flushed to disk
    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def example():\n")
        f.write("    x = 1\n")
        f.write("    return x\n")
        temp_path = f.name

    # File is now closed and flushed to disk
    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# Data File Detection Tests
# ============================================================================

def test_is_data_file_csv():
    """Test CSV file detection"""
    from trough.datapig_integration import is_data_file

    assert is_data_file("test.csv") is True
    assert is_data_file("test.py") is False


def test_is_data_file_json():
    """Test JSON file detection"""
    from trough.datapig_integration import is_data_file

    assert is_data_file("data.json") is True
    assert is_data_file("data.jsonl") is True
    assert is_data_file("code.js") is False


def test_detect_data_format():
    """Test data format detection from extension"""
    from trough.datapig_integration import detect_data_format

    assert detect_data_format("test.csv") == "csv"
    assert detect_data_format("test.json") == "json"
    assert detect_data_format("test.jsonl") == "jsonl"
    assert detect_data_format("test.tsv") == "tsv"
    assert detect_data_format("test.py") is None


# ============================================================================
# Data Loading Tests
# ============================================================================

def test_load_csv_file(temp_csv_file):
    """Test loading CSV file"""
    from trough.datapig_integration import load_csv

    rows = load_csv(temp_csv_file)

    assert len(rows) == 3  # 3 data rows
    assert rows[0]["id"] == "1"
    assert rows[0]["name"] == "Alice"


def test_load_json_file(temp_json_file):
    """Test loading JSON file"""
    from trough.datapig_integration import load_json

    rows = load_json(temp_json_file)

    assert len(rows) == 3
    assert rows[0]["id"] == 1
    assert rows[2]["value"] == 1000


# ============================================================================
# UnifiedDetector Tests
# ============================================================================

def test_unified_detector_creation():
    """Test creating UnifiedDetector"""
    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=True)

    assert detector is not None
    assert detector.enable_datapig is True


def test_unified_detector_data_quality(temp_csv_file):
    """Test detecting data quality issues in CSV file"""
    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=True)
    issues = detector.detect_data_quality(temp_csv_file)

    # Should detect duplicates and email leaks
    assert len(issues) >= 1  # At least one issue detected

    # Check for DATAPIG prefix in messages
    datapig_issues = [i for i in issues if "[DATAPIG]" in i.message]
    assert len(datapig_issues) >= 1


def test_unified_detector_no_datapig():
    """Test UnifiedDetector with DATAPIG disabled"""
    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=False)

    assert detector.datapig is None
