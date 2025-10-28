"""
Unit tests for functional transformations.

Tests composability, correctness, and edge cases of transform functions.
"""

import pytest
from datetime import datetime, timedelta

from apps.keep.models import Colony, Inspection, HarvestRecord
from apps.keep.types import HealthStatus, QueenStatus, InspectionType
from apps.keep.transforms import (
    # Filters
    filter_by_health,
    filter_healthy,
    filter_concerning,
    filter_by_queen_status,
    filter_queenless,
    filter_by_time_range,
    filter_recent,
    # Sorts
    sort_by_health,
    sort_by_population,
    sort_by_timestamp,
    # Aggregations
    take,
    group_by,
    count_by,
    compute_health_distribution,
    compute_average_population,
    compute_harvest_totals,
    # Composition
    pipe,
    compose,
    get_top_healthy_colonies,
    get_concerning_colonies,
    # Statistical
    extract_time_series,
    compute_trend,
)


@pytest.fixture
def sample_colonies():
    """Create sample colonies for testing."""
    return [
        Colony(
            colony_id="c1",
            hive_id="h1",
            health_status=HealthStatus.EXCELLENT,
            queen_status=QueenStatus.PRESENT_LAYING,
            population_estimate=60000,
        ),
        Colony(
            colony_id="c2",
            hive_id="h2",
            health_status=HealthStatus.GOOD,
            queen_status=QueenStatus.PRESENT_LAYING,
            population_estimate=45000,
        ),
        Colony(
            colony_id="c3",
            hive_id="h3",
            health_status=HealthStatus.FAIR,
            queen_status=QueenStatus.PRESENT_NOT_LAYING,
            population_estimate=30000,
        ),
        Colony(
            colony_id="c4",
            hive_id="h4",
            health_status=HealthStatus.POOR,
            queen_status=QueenStatus.ABSENT,
            population_estimate=15000,
        ),
        Colony(
            colony_id="c5",
            hive_id="h5",
            health_status=HealthStatus.CRITICAL,
            queen_status=QueenStatus.ABSENT,
            population_estimate=5000,
        ),
    ]


@pytest.fixture
def sample_inspections():
    """Create sample inspections for testing."""
    now = datetime.now()
    return [
        Inspection(
            inspection_id="i1",
            hive_id="h1",
            timestamp=now - timedelta(days=1),
            inspection_type=InspectionType.ROUTINE,
        ),
        Inspection(
            inspection_id="i2",
            hive_id="h2",
            timestamp=now - timedelta(days=7),
            inspection_type=InspectionType.HEALTH_CHECK,
        ),
        Inspection(
            inspection_id="i3",
            hive_id="h3",
            timestamp=now - timedelta(days=14),
            inspection_type=InspectionType.ROUTINE,
        ),
        Inspection(
            inspection_id="i4",
            hive_id="h4",
            timestamp=now - timedelta(days=30),
            inspection_type=InspectionType.SWARM_CHECK,
        ),
    ]


class TestFilters:
    """Test filter functions."""

    def test_filter_by_health(self, sample_colonies):
        """Test filtering by health status."""
        excellent = filter_by_health(HealthStatus.EXCELLENT)(sample_colonies)
        assert len(excellent) == 1
        assert excellent[0].health_status == HealthStatus.EXCELLENT

        multiple = filter_by_health(
            HealthStatus.EXCELLENT,
            HealthStatus.GOOD
        )(sample_colonies)
        assert len(multiple) == 2

    def test_filter_healthy(self, sample_colonies):
        """Test filtering to healthy colonies."""
        healthy = filter_healthy(sample_colonies)
        assert len(healthy) == 2
        assert all(
            c.health_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD]
            for c in healthy
        )

    def test_filter_concerning(self, sample_colonies):
        """Test filtering to concerning colonies."""
        concerning = filter_concerning(sample_colonies)
        assert len(concerning) == 3
        assert all(
            c.health_status in [HealthStatus.FAIR, HealthStatus.POOR, HealthStatus.CRITICAL]
            for c in concerning
        )

    def test_filter_by_queen_status(self, sample_colonies):
        """Test filtering by queen status."""
        laying = filter_by_queen_status(QueenStatus.PRESENT_LAYING)(sample_colonies)
        assert len(laying) == 2

        absent = filter_by_queen_status(QueenStatus.ABSENT)(sample_colonies)
        assert len(absent) == 2

    def test_filter_queenless(self, sample_colonies):
        """Test filtering to potentially queenless colonies."""
        queenless = filter_queenless(sample_colonies)
        assert len(queenless) == 3  # ABSENT + PRESENT_NOT_LAYING

    def test_filter_by_time_range(self, sample_inspections):
        """Test filtering by time range."""
        now = datetime.now()
        week_ago = now - timedelta(days=7)

        recent = filter_by_time_range(week_ago)(sample_inspections)
        assert len(recent) == 2

    def test_filter_recent(self, sample_inspections):
        """Test filtering to recent inspections."""
        recent = filter_recent(7)(sample_inspections)
        assert len(recent) == 2

    def test_filter_empty_list(self, sample_colonies):
        """Test that filters work on empty lists."""
        result = filter_healthy([])
        assert result == []


class TestSorts:
    """Test sort functions."""

    def test_sort_by_health(self, sample_colonies):
        """Test sorting by health status."""
        sorted_asc = sort_by_health(reverse=False)(sample_colonies)
        assert sorted_asc[0].health_status == HealthStatus.CRITICAL
        assert sorted_asc[-1].health_status == HealthStatus.EXCELLENT

        sorted_desc = sort_by_health(reverse=True)(sample_colonies)
        assert sorted_desc[0].health_status == HealthStatus.EXCELLENT
        assert sorted_desc[-1].health_status == HealthStatus.CRITICAL

    def test_sort_by_population(self, sample_colonies):
        """Test sorting by population."""
        sorted_asc = sort_by_population(reverse=False)(sample_colonies)
        assert sorted_asc[0].population_estimate == 5000
        assert sorted_asc[-1].population_estimate == 60000

        sorted_desc = sort_by_population(reverse=True)(sample_colonies)
        assert sorted_desc[0].population_estimate == 60000
        assert sorted_desc[-1].population_estimate == 5000

    def test_sort_by_timestamp(self, sample_inspections):
        """Test sorting by timestamp."""
        sorted_recent = sort_by_timestamp(reverse=True)(sample_inspections)
        # Most recent first
        assert sorted_recent[0].inspection_id == "i1"
        assert sorted_recent[-1].inspection_id == "i4"

    def test_sort_stability(self, sample_colonies):
        """Test that sort is stable."""
        # Create colonies with same health
        equal_colonies = [
            Colony(colony_id=f"c{i}", health_status=HealthStatus.GOOD)
            for i in range(5)
        ]

        sorted_colonies = sort_by_health()(equal_colonies)
        # Order should be preserved for equal elements
        assert [c.colony_id for c in sorted_colonies] == [f"c{i}" for i in range(5)]


class TestAggregations:
    """Test aggregation functions."""

    def test_take(self, sample_colonies):
        """Test take function."""
        assert len(take(3)(sample_colonies)) == 3
        assert len(take(10)(sample_colonies)) == 5  # Only 5 colonies
        assert len(take(0)(sample_colonies)) == 0

    def test_group_by(self, sample_colonies):
        """Test grouping by key."""
        grouped = group_by(lambda c: c.health_status)(sample_colonies)

        assert len(grouped[HealthStatus.EXCELLENT]) == 1
        assert len(grouped[HealthStatus.GOOD]) == 1
        assert len(grouped[HealthStatus.FAIR]) == 1
        assert len(grouped[HealthStatus.POOR]) == 1
        assert len(grouped[HealthStatus.CRITICAL]) == 1

    def test_count_by(self, sample_colonies):
        """Test counting by key."""
        counts = count_by(lambda c: c.health_status)(sample_colonies)

        assert counts[HealthStatus.EXCELLENT] == 1
        assert counts[HealthStatus.GOOD] == 1
        assert counts[HealthStatus.FAIR] == 1
        assert counts[HealthStatus.POOR] == 1
        assert counts[HealthStatus.CRITICAL] == 1

    def test_compute_health_distribution(self, sample_colonies):
        """Test health distribution computation."""
        dist = compute_health_distribution(sample_colonies)
        assert dist[HealthStatus.EXCELLENT] == 1
        assert dist[HealthStatus.GOOD] == 1
        assert sum(dist.values()) == 5

    def test_compute_average_population(self, sample_colonies):
        """Test average population computation."""
        avg = compute_average_population(sample_colonies)
        expected = (60000 + 45000 + 30000 + 15000 + 5000) / 5
        assert avg == expected

    def test_compute_average_population_empty(self):
        """Test average on empty list."""
        avg = compute_average_population([])
        assert avg == 0.0

    def test_compute_harvest_totals(self):
        """Test harvest total computation."""
        harvests = [
            HarvestRecord(hive_id="h1", product_type="honey", quantity=40.0),
            HarvestRecord(hive_id="h2", product_type="honey", quantity=35.0),
            HarvestRecord(hive_id="h3", product_type="wax", quantity=2.0),
        ]

        by_product = compute_harvest_totals(harvests, by_product=True)
        assert by_product["honey"] == 75.0
        assert by_product["wax"] == 2.0

        total = compute_harvest_totals(harvests, by_product=False)
        assert total["total"] == 77.0


class TestComposition:
    """Test function composition."""

    def test_pipe(self, sample_colonies):
        """Test pipe composition (left-to-right)."""
        transform = pipe(
            filter_healthy,
            sort_by_population(reverse=True),
            take(1)
        )

        result = transform(sample_colonies)
        assert len(result) == 1
        assert result[0].population_estimate == 60000

    def test_compose(self, sample_colonies):
        """Test compose (right-to-left)."""
        transform = compose(
            take(1),
            sort_by_population(reverse=True),
            filter_healthy
        )

        result = transform(sample_colonies)
        assert len(result) == 1
        assert result[0].population_estimate == 60000

    def test_pipe_vs_compose(self, sample_colonies):
        """Test that pipe and compose are inverses."""
        pipe_result = pipe(filter_healthy, take(2))(sample_colonies)
        compose_result = compose(take(2), filter_healthy)(sample_colonies)

        assert pipe_result == compose_result

    def test_get_top_healthy_colonies(self, sample_colonies):
        """Test composite get_top_healthy_colonies."""
        top_2 = get_top_healthy_colonies(2)(sample_colonies)

        assert len(top_2) == 2
        assert all(
            c.health_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD]
            for c in top_2
        )
        assert top_2[0].population_estimate == 60000
        assert top_2[1].population_estimate == 45000

    def test_get_concerning_colonies(self, sample_colonies):
        """Test composite get_concerning_colonies."""
        concerning = get_concerning_colonies()(sample_colonies)

        assert len(concerning) == 3
        # Should be sorted by severity (worst first)
        assert concerning[0].health_status == HealthStatus.CRITICAL
        assert concerning[1].health_status == HealthStatus.POOR
        assert concerning[2].health_status == HealthStatus.FAIR


class TestStatistical:
    """Test statistical analysis functions."""

    def test_extract_time_series(self, sample_inspections):
        """Test time series extraction."""
        def extract_value(insp):
            # Mock: return day offset as value
            return (datetime.now() - insp.timestamp).days

        series = extract_time_series(sample_inspections, extract_value)

        assert len(series) == 4
        # Should be sorted by timestamp
        assert series[0][1] > series[-1][1]  # Older inspections have higher values

    def test_extract_time_series_with_none(self, sample_inspections):
        """Test that None values are filtered out."""
        def extract_value(insp):
            return None if insp.inspection_id == "i2" else 1.0

        series = extract_time_series(sample_inspections, extract_value)
        assert len(series) == 3  # One filtered out

    def test_compute_trend_increasing(self):
        """Test trend computation for increasing series."""
        now = datetime.now()
        series = [
            (now - timedelta(days=10), 1.0),
            (now - timedelta(days=5), 2.0),
            (now, 3.0),
        ]

        trend = compute_trend(series)
        assert trend["slope"] > 0
        assert trend["direction"] == 1

    def test_compute_trend_decreasing(self):
        """Test trend computation for decreasing series."""
        now = datetime.now()
        series = [
            (now - timedelta(days=10), 3.0),
            (now - timedelta(days=5), 2.0),
            (now, 1.0),
        ]

        trend = compute_trend(series)
        assert trend["slope"] < 0
        assert trend["direction"] == -1

    def test_compute_trend_stable(self):
        """Test trend computation for stable series."""
        now = datetime.now()
        series = [
            (now - timedelta(days=10), 2.0),
            (now - timedelta(days=5), 2.0),
            (now, 2.0),
        ]

        trend = compute_trend(series)
        assert abs(trend["slope"]) < 0.01
        assert trend["direction"] == 0

    def test_compute_trend_insufficient_data(self):
        """Test trend with insufficient data."""
        trend = compute_trend([])
        assert trend["slope"] == 0.0
        assert trend["direction"] == 0

        trend = compute_trend([(datetime.now(), 1.0)])
        assert trend["slope"] == 0.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_transforms_on_empty_lists(self):
        """Test that all transforms handle empty lists."""
        empty = []

        assert filter_healthy(empty) == []
        assert sort_by_health()(empty) == []
        assert take(5)(empty) == []
        assert group_by(lambda x: x)(empty) == {}
        assert compute_average_population(empty) == 0.0

    def test_transforms_preserve_original(self, sample_colonies):
        """Test that transforms don't mutate original list."""
        original_len = len(sample_colonies)
        original_first = sample_colonies[0]

        filter_healthy(sample_colonies)
        sort_by_health()(sample_colonies)
        take(2)(sample_colonies)

        assert len(sample_colonies) == original_len
        assert sample_colonies[0] is original_first

    def test_chained_transforms_correctness(self, sample_colonies):
        """Test complex chained transforms for correctness."""
        result = pipe(
            filter_concerning,
            sort_by_health(reverse=False),
            take(2)
        )(sample_colonies)

        # Should have 2 worst colonies
        assert len(result) == 2
        assert result[0].health_status == HealthStatus.CRITICAL
        assert result[1].health_status == HealthStatus.POOR


# Performance tests
class TestPerformance:
    """Test performance characteristics."""

    def test_filter_performance(self):
        """Test filter performance on large dataset."""
        large_dataset = [
            Colony(
                colony_id=f"c{i}",
                health_status=HealthStatus.GOOD if i % 2 == 0 else HealthStatus.POOR,
                population_estimate=50000,
            )
            for i in range(1000)
        ]

        import time
        start = time.time()
        result = filter_healthy(large_dataset)
        elapsed = time.time() - start

        assert len(result) == 500
        assert elapsed < 0.1  # Should be very fast

    def test_sort_performance(self):
        """Test sort performance on large dataset."""
        import random
        large_dataset = [
            Colony(
                colony_id=f"c{i}",
                health_status=random.choice(list(HealthStatus)),
                population_estimate=random.randint(1000, 100000),
            )
            for i in range(1000)
        ]

        import time
        start = time.time()
        result = sort_by_population()(large_dataset)
        elapsed = time.time() - start

        assert len(result) == 1000
        assert elapsed < 0.1  # Should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
