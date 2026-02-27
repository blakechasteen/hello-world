"""
Performance Benchmark Tests for DATAPIG Detection Speed

Measures detection speed across different dataset sizes and issue types.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
import time
from hololoom.datapig import DataPigDetector


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create DATAPIG detector for benchmarking"""
    return DataPigDetector(enable_verbose=False)


@pytest.fixture
def small_dataset():
    """10 rows"""
    return [{"id": i, "value": i * 10} for i in range(10)]


@pytest.fixture
def medium_dataset():
    """100 rows"""
    return [{"id": i, "value": i * 10} for i in range(100)]


@pytest.fixture
def large_dataset():
    """1000 rows"""
    return [{"id": i, "value": i * 10} for i in range(1000)]


@pytest.fixture
def dataset_with_issues():
    """Dataset with multiple issue types for comprehensive detection"""
    data = []

    # Clean rows
    for i in range(50):
        data.append({
            "id": i,
            "name": f"User{i}",
            "value": i * 10,
            "updated_at": "2023-01-15T00:00:00"
        })

    # Add duplicates
    data.append(data[0].copy())
    data.append(data[1].copy())

    # Add PII leak
    data.append({
        "id": 100,
        "name": "Test",
        "email": "test@example.com",
        "value": 10,
        "updated_at": "2023-01-15T00:00:00"
    })

    # Add outlier
    data.append({
        "id": 101,
        "name": "Outlier",
        "value": 99999,
        "updated_at": "2023-01-15T00:00:00"
    })

    return data


# ============================================================================
# Detection Speed Benchmarks
# ============================================================================

def test_benchmark_small_dataset_speed(detector, small_dataset):
    """Benchmark: Detection speed on small dataset (10 rows)"""

    # Warm-up run
    detector.analyze_dataset(small_dataset)

    # Benchmark run
    start = time.perf_counter()
    result = detector.analyze_dataset(small_dataset)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify detection works
    assert isinstance(result, list)

    # Performance target: <100ms for small datasets
    print(f"\n  Small dataset (10 rows): {duration_ms:.2f}ms")
    assert duration_ms < 100, f"Detection took {duration_ms:.2f}ms, target <100ms"


def test_benchmark_medium_dataset_speed(detector, medium_dataset):
    """Benchmark: Detection speed on medium dataset (100 rows)"""

    # Warm-up run
    detector.analyze_dataset(medium_dataset)

    # Benchmark run
    start = time.perf_counter()
    result = detector.analyze_dataset(medium_dataset)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify detection works
    assert isinstance(result, list)

    # Performance target: <200ms for medium datasets
    print(f"\n  Medium dataset (100 rows): {duration_ms:.2f}ms")
    assert duration_ms < 200, f"Detection took {duration_ms:.2f}ms, target <200ms"


def test_benchmark_large_dataset_speed(detector, large_dataset):
    """Benchmark: Detection speed on large dataset (1000 rows)"""

    # Warm-up run
    detector.analyze_dataset(large_dataset)

    # Benchmark run
    start = time.perf_counter()
    result = detector.analyze_dataset(large_dataset)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify detection works
    assert isinstance(result, list)

    # Performance target: <500ms for large datasets
    print(f"\n  Large dataset (1000 rows): {duration_ms:.2f}ms")
    assert duration_ms < 500, f"Detection took {duration_ms:.2f}ms, target <500ms"


def test_benchmark_detection_scaling(detector):
    """Benchmark: Detection speed scaling with dataset size"""

    sizes = [10, 50, 100, 500, 1000]
    timings = []

    for size in sizes:
        dataset = [{"id": i, "value": i * 10} for i in range(size)]

        start = time.perf_counter()
        detector.analyze_dataset(dataset)
        end = time.perf_counter()

        duration_ms = (end - start) * 1000
        timings.append(duration_ms)

    # Verify scaling is reasonable (should be roughly linear or sub-linear)
    # Time for 1000 rows should be < 100x time for 10 rows (allows for overhead)
    # Linear scaling would be exactly 100x, so this allows for sub-linear scaling
    assert timings[-1] < timings[0] * 100

    # Print scaling info for analysis
    for size, timing in zip(sizes, timings):
        print(f"  {size} rows: {timing:.2f}ms")


def test_benchmark_comprehensive_detection(detector, dataset_with_issues):
    """Benchmark: Detection across all issue types"""

    # Warm-up run
    detector.analyze_dataset(dataset_with_issues)

    # Benchmark run
    start = time.perf_counter()
    result = detector.analyze_dataset(dataset_with_issues)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify multiple issue types detected
    assert len(result) >= 2  # At least duplicates + PII/outliers

    # Performance target: <300ms for comprehensive detection
    print(f"\n  Comprehensive detection ({len(dataset_with_issues)} rows): {duration_ms:.2f}ms")
    assert duration_ms < 300, f"Detection took {duration_ms:.2f}ms, target <300ms"


def test_benchmark_detector_initialization():
    """Benchmark: Detector initialization overhead"""

    # Warm-up run
    DataPigDetector(enable_verbose=False)

    # Benchmark run
    start = time.perf_counter()
    detector = DataPigDetector(enable_verbose=False)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Verify detector created
    assert detector is not None

    # Performance target: <100ms for initialization
    print(f"\n  Detector initialization: {duration_ms:.2f}ms")
    assert duration_ms < 100, f"Initialization took {duration_ms:.2f}ms, target <100ms"
