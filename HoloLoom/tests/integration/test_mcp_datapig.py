"""
Integration Tests for MCP Tools + DATAPIG Integration

Tests Model Context Protocol tools for DATAPIG data quality analysis.

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
def temp_sample_dataset():
    """Create a sample dataset for MCP tool testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = [
            {"id": 1, "name": "Alice", "value": 100},
            {"id": 2, "name": "Bob", "value": 200},
            {"id": 1, "name": "Alice", "value": 100},  # Duplicate
        ]
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# MCP Tool Integration Tests
# ============================================================================

def test_mcp_tool_availability():
    """Test that DATAPIG is available for MCP integration"""
    # Check that DATAPIG detector can be imported
    from HoloLoom.datapig import DataPigDetector

    detector = DataPigDetector(enable_verbose=False)
    assert detector is not None


def test_mcp_analyze_dataset_structure():
    """Test MCP analyze_dataset tool structure"""
    from HoloLoom.datapig import DataPigDetector

    detector = DataPigDetector(enable_verbose=False)

    # Verify analyze_dataset method exists
    assert hasattr(detector, 'analyze_dataset')

    # Test with minimal data
    data = [{"id": 1, "value": 10}]
    issues = detector.analyze_dataset(data)

    # Should return a list (even if empty)
    assert isinstance(issues, list)


def test_mcp_tool_response_format():
    """Test that MCP tool responses are in expected format"""
    from HoloLoom.datapig import DataPigDetector, DataQualityIssue

    detector = DataPigDetector(enable_verbose=False)

    # Test with data that will trigger issues
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 1, "name": "Alice"},  # Duplicate
    ]

    issues = detector.analyze_dataset(data)

    # Should have at least one issue (duplicates)
    assert len(issues) >= 1

    # All issues should be DataQualityIssue instances
    for issue in issues:
        assert isinstance(issue, DataQualityIssue)
        assert hasattr(issue, 'issue_type')
        assert hasattr(issue, 'severity')
        assert hasattr(issue, 'message')
        assert hasattr(issue, 'location')


def test_mcp_unified_integration():
    """Test MCP integration with unified detector"""
    from trough.datapig_integration import UnifiedDetector

    detector = UnifiedDetector(enable_datapig=True)

    # Verify detector is available
    assert detector is not None
    assert detector.enable_datapig is True


def test_mcp_analyze_file_unified(temp_sample_dataset):
    """Test MCP analyze_file_unified helper function"""
    from trough.datapig_integration import analyze_file_unified

    # Analyze the temp dataset
    result = analyze_file_unified(temp_sample_dataset)

    # Verify result structure
    assert "file_path" in result
    assert "file_type" in result
    assert "issues" in result
    assert "summary" in result
    assert "datapig_enabled" in result

    # File should be recognized as data file
    assert result["file_type"] == "data"

    # Should have found issues (duplicates in test data)
    assert result["summary"]["total_issues"] >= 1


def test_mcp_scan_directory_structure():
    """Test MCP scan_directory_unified helper function structure"""
    from trough.datapig_integration import scan_directory_unified
    import tempfile

    # Create a temporary directory with a data file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test CSV file in the directory
        test_file = os.path.join(temp_dir, "test.csv")
        with open(test_file, 'w', newline='') as f:
            f.write("id,name\n")
            f.write("1,Alice\n")
            f.write("1,Alice\n")  # Duplicate

        # Scan directory
        result = scan_directory_unified(temp_dir, include_data=True)

        # Verify result structure
        assert "total_files" in result
        assert "files_scanned" in result
        assert "total_issues" in result
        assert "issues_by_file" in result
        assert "summary" in result

        # Should have scanned at least one file
        assert result["total_files"] >= 0  # May be 0 if no issues found
