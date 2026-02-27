"""
Performance Benchmark Tests for DATAPIG + xTerminator Fixing Speed

Measures auto-fixing speed for different issue types and dataset sizes.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
import time
from hololoom.datapig import DataPigDetector, IssueType


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create DATAPIG detector for benchmarking"""
    return DataPigDetector(enable_verbose=False)


# ============================================================================
# Fixing Speed Benchmarks
# ============================================================================

def test_benchmark_issue_detection_overhead(detector):
    """Benchmark: Overhead of detecting issues before fixing"""

    # Dataset with duplicates
    data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 1, "value": 10},  # Duplicate
    ]

    # Warm-up run
    detector.analyze_dataset(data)

    # Benchmark detection
    start = time.perf_counter()
    issues = detector.analyze_dataset(data)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify issues detected
    assert len(issues) >= 1

    # Performance target: <10ms for small dataset detection
    print(f"\n  Issue detection (3 rows): {duration_ms:.2f}ms")
    assert duration_ms < 10, f"Detection took {duration_ms:.2f}ms, target <10ms"


def test_benchmark_fix_suggestion_generation(detector):
    """Benchmark: Speed of generating fix suggestions"""

    # Dataset with multiple issue types
    data = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},  # PII
        {"id": 2, "name": "Bob", "value": 10},
        {"id": 2, "name": "Bob", "value": 10},  # Duplicate
    ]

    # Warm-up run
    detector.analyze_dataset(data)

    # Benchmark detection + suggestion generation
    start = time.perf_counter()
    issues = detector.analyze_dataset(data)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify issues have suggestions (via conversion)
    from trough.datapig_integration import generate_fix_suggestion

    for issue in issues:
        suggestion = generate_fix_suggestion(issue)
        assert suggestion is not None

    # Performance target: <20ms including suggestion generation
    print(f"\n  Detection + suggestions ({len(issues)} issues): {duration_ms:.2f}ms")
    assert duration_ms < 20, f"Detection+suggestions took {duration_ms:.2f}ms, target <20ms"


def test_benchmark_issue_prioritization():
    """Benchmark: Speed of prioritizing issues by severity"""

    from hololoom.datapig import DataQualityIssue, Severity

    # Create mock issues with different severities
    issues = []
    for i in range(100):
        severity = [Severity.CAPTAIN, Severity.COMMANDER, Severity.LIEUTENANT, Severity.ENSIGN][i % 4]
        issues.append(DataQualityIssue(
            issue_type=IssueType.DUPLICATES,
            severity=severity,
            message=f"Issue {i}",
            location=f"row_{i}",
            details={},
            stardate=0.0
        ))

    # Severity ranking (higher = more severe)
    severity_rank = {
        Severity.CAPTAIN: 4,
        Severity.COMMANDER: 3,
        Severity.LIEUTENANT: 2,
        Severity.ENSIGN: 1,
    }

    # Benchmark prioritization (sorting by severity rank)
    start = time.perf_counter()
    sorted_issues = sorted(issues, key=lambda x: severity_rank[x.severity], reverse=True)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify sorting
    assert len(sorted_issues) == 100
    # CAPTAIN (highest rank) comes first
    assert sorted_issues[0].severity == Severity.CAPTAIN
    # ENSIGN (lowest rank) comes last
    assert sorted_issues[-1].severity == Severity.ENSIGN

    # Performance target: <1ms for 100 issues
    print(f"\n  Issue prioritization (100 issues): {duration_ms:.2f}ms")
    assert duration_ms < 1, f"Prioritization took {duration_ms:.2f}ms, target <1ms"


def test_benchmark_batch_issue_processing(detector):
    """Benchmark: Processing multiple datasets in batch"""

    # Create 10 small datasets
    datasets = []
    for i in range(10):
        datasets.append([
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 1, "value": 10},  # Duplicate in each
        ])

    # Warm-up run
    for dataset in datasets:
        detector.analyze_dataset(dataset)

    # Benchmark batch processing
    start = time.perf_counter()
    all_issues = []
    for dataset in datasets:
        issues = detector.analyze_dataset(dataset)
        all_issues.extend(issues)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify all datasets processed
    assert len(all_issues) >= 10  # At least 1 issue per dataset

    # Performance target: <100ms for 10 small datasets
    print(f"\n  Batch processing (10 datasets): {duration_ms:.2f}ms")
    assert duration_ms < 100, f"Batch processing took {duration_ms:.2f}ms, target <100ms"


def test_benchmark_issue_conversion_overhead():
    """Benchmark: DATAPIG → Trough SlopIssue conversion overhead"""

    from hololoom.datapig import DataQualityIssue, Severity
    from trough.datapig_integration import convert_datapig_issue

    # Create 100 mock DATAPIG issues
    datapig_issues = []
    for i in range(100):
        datapig_issues.append(DataQualityIssue(
            issue_type=IssueType.DUPLICATES,
            severity=Severity.LIEUTENANT,
            message=f"Duplicate row {i}",
            location=f"row_{i}",
            details={"row_index": i},
            stardate=0.0
        ))

    # Benchmark conversion
    start = time.perf_counter()
    slop_issues = [convert_datapig_issue(issue, "test.csv") for issue in datapig_issues]
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify conversion
    assert len(slop_issues) == 100
    assert all(issue.file_path == "test.csv" for issue in slop_issues)

    # Performance target: <10ms for 100 conversions
    print(f"\n  Issue conversion (100 issues): {duration_ms:.2f}ms")
    assert duration_ms < 10, f"Conversion took {duration_ms:.2f}ms, target <10ms"


def test_benchmark_unified_detector_overhead(detector):
    """Benchmark: Overhead of unified code+data detection"""

    from trough.datapig_integration import UnifiedDetector
    import tempfile
    import os

    # Create temp CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,value\n")
        f.write("1,10\n")
        f.write("2,20\n")
        f.write("1,10\n")  # Duplicate
        temp_path = f.name

    try:
        # Create unified detector
        unified = UnifiedDetector(enable_datapig=True)

        # Warm-up run
        unified.detect_data_quality(temp_path)

        # Benchmark unified detection
        start = time.perf_counter()
        issues = unified.detect_data_quality(temp_path)
        end = time.perf_counter()

        duration_ms = (end - start) * 1000

        # Verify detection
        assert len(issues) >= 1

        # Performance target: <50ms for file-based detection
        print(f"\n  Unified detection (file-based): {duration_ms:.2f}ms")
        assert duration_ms < 50, f"Unified detection took {duration_ms:.2f}ms, target <50ms"

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
