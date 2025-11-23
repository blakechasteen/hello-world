"""
Performance Benchmark Tests for DATAPIG End-to-End Pipeline

Measures complete detection → fixing → validation pipeline performance.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
import time
import tempfile
import os
import json
from HoloLoom.datapig import DataPigDetector


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create DATAPIG detector for benchmarking"""
    return DataPigDetector(enable_verbose=False)


# ============================================================================
# End-to-End Pipeline Benchmarks
# ============================================================================

def test_benchmark_e2e_csv_detection():
    """Benchmark: End-to-end CSV file detection pipeline"""

    from trough.datapig_integration import analyze_file_unified

    # Create temp CSV with issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,value\n")
        f.write("1,Alice,100\n")
        f.write("2,Bob,200\n")
        f.write("1,Alice,100\n")  # Duplicate
        temp_path = f.name

    try:
        # Warm-up run
        analyze_file_unified(temp_path)

        # Benchmark end-to-end analysis
        start = time.perf_counter()
        result = analyze_file_unified(temp_path)
        end = time.perf_counter()

        duration_ms = (end - start) * 1000

        # Verify analysis
        assert result["file_type"] == "data"
        assert result["summary"]["total_issues"] >= 1

        # Performance target: <50ms for small CSV
        print(f"\n  E2E CSV detection (4 rows): {duration_ms:.2f}ms")
        assert duration_ms < 50, f"E2E analysis took {duration_ms:.2f}ms, target <50ms"

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_benchmark_e2e_json_detection():
    """Benchmark: End-to-end JSON file detection pipeline"""

    from trough.datapig_integration import analyze_file_unified

    # Create temp JSON with issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 1, "value": 10},  # Duplicate
        ]
        json.dump(data, f)
        temp_path = f.name

    try:
        # Warm-up run
        analyze_file_unified(temp_path)

        # Benchmark end-to-end analysis
        start = time.perf_counter()
        result = analyze_file_unified(temp_path)
        end = time.perf_counter()

        duration_ms = (end - start) * 1000

        # Verify analysis
        assert result["file_type"] == "data"
        assert result["summary"]["total_issues"] >= 1

        # Performance target: <50ms for small JSON
        print(f"\n  E2E JSON detection (3 rows): {duration_ms:.2f}ms")
        assert duration_ms < 50, f"E2E analysis took {duration_ms:.2f}ms, target <50ms"

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_benchmark_e2e_directory_scan():
    """Benchmark: End-to-end directory scanning pipeline"""

    from trough.datapig_integration import scan_directory_unified

    # Create temp directory with multiple files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create 5 CSV files with issues
        for i in range(5):
            file_path = os.path.join(temp_dir, f"test_{i}.csv")
            with open(file_path, 'w', newline='') as f:
                f.write("id,value\n")
                f.write("1,10\n")
                f.write("1,10\n")  # Duplicate in each file

        # Warm-up run
        scan_directory_unified(temp_dir, include_data=True)

        # Benchmark directory scan
        start = time.perf_counter()
        result = scan_directory_unified(temp_dir, include_data=True)
        end = time.perf_counter()

        duration_ms = (end - start) * 1000

        # Verify scan
        assert result["total_files"] >= 1
        assert result["total_issues"] >= 5  # At least 1 per file

        # Performance target: <200ms for 5 small files
        print(f"\n  E2E directory scan (5 files): {duration_ms:.2f}ms")
        assert duration_ms < 200, f"Directory scan took {duration_ms:.2f}ms, target <200ms"


def test_benchmark_e2e_issue_reporting():
    """Benchmark: End-to-end issue detection and reporting"""

    from trough.datapig_integration import UnifiedDetector, convert_datapig_issue

    detector = UnifiedDetector(enable_datapig=True)

    # Create dataset with multiple issue types
    data = [
        {"id": 1, "name": "Alice", "email": "test@example.com"},  # PII
        {"id": 2, "name": "Bob", "value": 10},
        {"id": 2, "name": "Bob", "value": 10},  # Duplicate
        {"id": 3, "name": "Charlie", "value": 9999},  # Outlier
    ]

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        # Warm-up run
        detector.detect_data_quality(temp_path)

        # Benchmark detection + conversion + reporting
        start = time.perf_counter()
        trough_issues = detector.detect_data_quality(temp_path)
        end = time.perf_counter()

        duration_ms = (end - start) * 1000

        # Verify reporting
        assert len(trough_issues) >= 2  # Multiple issue types

        # All issues should have proper structure
        for issue in trough_issues:
            assert hasattr(issue, 'category')
            assert hasattr(issue, 'severity')
            assert hasattr(issue, 'message')
            assert hasattr(issue, 'file_path')
            assert issue.file_path == temp_path

        # Performance target: <100ms for comprehensive reporting
        print(f"\n  E2E issue reporting ({len(trough_issues)} issues): {duration_ms:.2f}ms")
        assert duration_ms < 100, f"Issue reporting took {duration_ms:.2f}ms, target <100ms"

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_benchmark_e2e_memory_usage(detector):
    """Benchmark: Memory usage during large dataset processing"""

    import psutil
    import os as os_module

    # Create large dataset (1000 rows)
    large_dataset = []
    for i in range(1000):
        large_dataset.append({
            "id": i,
            "name": f"User{i}",
            "value": i * 10,
        })

    # Add some duplicates
    for i in range(10):
        large_dataset.append(large_dataset[i].copy())

    # Get process
    process = psutil.Process(os_module.getpid())

    # Measure memory before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Run analysis
    start = time.perf_counter()
    issues = detector.analyze_dataset(large_dataset)
    end = time.perf_counter()

    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_delta = mem_after - mem_before

    duration_ms = (end - start) * 1000

    # Verify analysis
    assert len(issues) >= 10  # Should find duplicates

    # Performance target: <100MB memory increase, <500ms processing
    print(f"\n  E2E memory (1010 rows): {duration_ms:.2f}ms, {mem_delta:.1f}MB delta")
    assert duration_ms < 500, f"Processing took {duration_ms:.2f}ms, target <500ms"
    assert mem_delta < 100, f"Memory increased {mem_delta:.1f}MB, target <100MB"


def test_benchmark_e2e_throughput(detector):
    """Benchmark: Throughput (datasets per second)"""

    # Create 100 small datasets
    datasets = []
    for i in range(100):
        datasets.append([
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 1, "value": 10},  # Duplicate
        ])

    # Warm-up run
    for dataset in datasets[:10]:
        detector.analyze_dataset(dataset)

    # Benchmark throughput
    start = time.perf_counter()
    for dataset in datasets:
        detector.analyze_dataset(dataset)
    end = time.perf_counter()

    duration_sec = end - start
    throughput = len(datasets) / duration_sec

    # Verify all processed
    assert throughput > 0

    # Performance target: >100 datasets/second
    print(f"\n  E2E throughput: {throughput:.0f} datasets/sec")
    assert throughput > 100, f"Throughput {throughput:.0f} datasets/sec, target >100"
