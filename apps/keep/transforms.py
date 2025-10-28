"""
Elegant data transformation utilities for Keep.

Provides functional, composable transformations for apiary data,
enabling clean data flow through the system.
"""

from typing import List, Dict, Any, Callable, Optional, TypeVar
from datetime import datetime, timedelta
from functools import reduce
from dataclasses import asdict

from apps.keep.models import Hive, Colony, Inspection, HarvestRecord, Alert
from apps.keep.types import HealthStatus, QueenStatus, AlertLevel


T = TypeVar('T')


# =============================================================================
# Core Transformation Utilities
# =============================================================================

def compose(*funcs: Callable) -> Callable:
    """
    Compose functions right-to-left.

    Example:
        transform = compose(filter_healthy, sort_by_health, take(5))
        result = transform(colonies)
    """
    def composed(x):
        return reduce(lambda acc, f: f(acc), reversed(funcs), x)
    return composed


def pipe(*funcs: Callable) -> Callable:
    """
    Compose functions left-to-right.

    Example:
        transform = pipe(filter_healthy, sort_by_health, take(5))
        result = transform(colonies)
    """
    def piped(x):
        return reduce(lambda acc, f: f(acc), funcs, x)
    return piped


def curry(func: Callable, *args, **kwargs) -> Callable:
    """Partially apply a function."""
    def curried(*more_args, **more_kwargs):
        return func(*args, *more_args, **kwargs, **more_kwargs)
    return curried


# =============================================================================
# Filtering Utilities
# =============================================================================

def filter_by_health(*statuses: HealthStatus) -> Callable[[List[Colony]], List[Colony]]:
    """Filter colonies by health status."""
    def filter_fn(colonies: List[Colony]) -> List[Colony]:
        return [c for c in colonies if c.health_status in statuses]
    return filter_fn


def filter_healthy(colonies: List[Colony]) -> List[Colony]:
    """Filter to healthy colonies only."""
    return filter_by_health(HealthStatus.EXCELLENT, HealthStatus.GOOD)(colonies)


def filter_concerning(colonies: List[Colony]) -> List[Colony]:
    """Filter to colonies needing attention."""
    return filter_by_health(
        HealthStatus.POOR,
        HealthStatus.CRITICAL,
        HealthStatus.FAIR
    )(colonies)


def filter_by_queen_status(*statuses: QueenStatus) -> Callable[[List[Colony]], List[Colony]]:
    """Filter colonies by queen status."""
    def filter_fn(colonies: List[Colony]) -> List[Colony]:
        return [c for c in colonies if c.queen_status in statuses]
    return filter_fn


def filter_queenless(colonies: List[Colony]) -> List[Colony]:
    """Filter to potentially queenless colonies."""
    return filter_by_queen_status(
        QueenStatus.ABSENT,
        QueenStatus.PRESENT_NOT_LAYING
    )(colonies)


def filter_by_time_range(
    start: datetime,
    end: Optional[datetime] = None
) -> Callable[[List[Inspection]], List[Inspection]]:
    """Filter inspections by time range."""
    end = end or datetime.now()

    def filter_fn(inspections: List[Inspection]) -> List[Inspection]:
        return [
            i for i in inspections
            if start <= i.timestamp <= end
        ]
    return filter_fn


def filter_recent(days: int) -> Callable[[List[Inspection]], List[Inspection]]:
    """Filter to recent inspections."""
    cutoff = datetime.now() - timedelta(days=days)
    return filter_by_time_range(cutoff)


# =============================================================================
# Sorting Utilities
# =============================================================================

def sort_by(key: Callable, reverse: bool = False) -> Callable[[List[T]], List[T]]:
    """Generic sorting function."""
    def sort_fn(items: List[T]) -> List[T]:
        return sorted(items, key=key, reverse=reverse)
    return sort_fn


def sort_by_health(reverse: bool = False) -> Callable[[List[Colony]], List[Colony]]:
    """Sort colonies by health status."""
    health_order = {
        HealthStatus.CRITICAL: 0,
        HealthStatus.POOR: 1,
        HealthStatus.FAIR: 2,
        HealthStatus.GOOD: 3,
        HealthStatus.EXCELLENT: 4,
    }
    return sort_by(lambda c: health_order[c.health_status], reverse=reverse)


def sort_by_population(reverse: bool = True) -> Callable[[List[Colony]], List[Colony]]:
    """Sort colonies by population estimate."""
    return sort_by(lambda c: c.population_estimate, reverse=reverse)


def sort_by_timestamp(reverse: bool = True) -> Callable[[List[Inspection]], List[Inspection]]:
    """Sort inspections by timestamp."""
    return sort_by(lambda i: i.timestamp, reverse=reverse)


def sort_alerts_by_priority(reverse: bool = True) -> Callable[[List[Alert]], List[Alert]]:
    """Sort alerts by priority level."""
    priority_order = {
        AlertLevel.CRITICAL: 0,
        AlertLevel.URGENT: 1,
        AlertLevel.WARNING: 2,
        AlertLevel.INFO: 3,
    }
    return sort_by(lambda a: priority_order[a.level], reverse=reverse)


# =============================================================================
# Aggregation Utilities
# =============================================================================

def take(n: int) -> Callable[[List[T]], List[T]]:
    """Take first n items."""
    def take_fn(items: List[T]) -> List[T]:
        return items[:n]
    return take_fn


def group_by(key: Callable) -> Callable[[List[T]], Dict[Any, List[T]]]:
    """Group items by a key function."""
    def group_fn(items: List[T]) -> Dict[Any, List[T]]:
        groups = {}
        for item in items:
            k = key(item)
            if k not in groups:
                groups[k] = []
            groups[k].append(item)
        return groups
    return group_fn


def count_by(key: Callable) -> Callable[[List[T]], Dict[Any, int]]:
    """Count items by a key function."""
    def count_fn(items: List[T]) -> Dict[Any, int]:
        groups = group_by(key)(items)
        return {k: len(v) for k, v in groups.items()}
    return count_fn


# =============================================================================
# Projection Utilities
# =============================================================================

def project_fields(*fields: str) -> Callable[[List[Any]], List[Dict[str, Any]]]:
    """Project specific fields from dataclass objects."""
    def project_fn(items: List[Any]) -> List[Dict[str, Any]]:
        result = []
        for item in items:
            item_dict = asdict(item) if hasattr(item, '__dataclass_fields__') else item
            result.append({f: item_dict.get(f) for f in fields if f in item_dict})
        return result
    return project_fn


def to_dict(item: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary."""
    return asdict(item) if hasattr(item, '__dataclass_fields__') else item


def to_dicts(items: List[Any]) -> List[Dict[str, Any]]:
    """Convert list of dataclasses to dictionaries."""
    return [to_dict(item) for item in items]


# =============================================================================
# Statistical Aggregations
# =============================================================================

def compute_health_distribution(colonies: List[Colony]) -> Dict[HealthStatus, int]:
    """Compute distribution of colony health statuses."""
    return count_by(lambda c: c.health_status)(colonies)


def compute_average_population(colonies: List[Colony]) -> float:
    """Compute average colony population."""
    if not colonies:
        return 0.0
    return sum(c.population_estimate for c in colonies) / len(colonies)


def compute_inspection_frequency(inspections: List[Inspection]) -> Dict[str, float]:
    """
    Compute average days between inspections per hive.

    Returns:
        Dict mapping hive_id to average days between inspections
    """
    by_hive = group_by(lambda i: i.hive_id)(inspections)
    frequencies = {}

    for hive_id, hive_inspections in by_hive.items():
        if len(hive_inspections) < 2:
            frequencies[hive_id] = 0.0
            continue

        sorted_inspections = sort_by_timestamp(reverse=True)(hive_inspections)
        gaps = [
            (sorted_inspections[i].timestamp - sorted_inspections[i + 1].timestamp).days
            for i in range(len(sorted_inspections) - 1)
        ]
        frequencies[hive_id] = sum(gaps) / len(gaps) if gaps else 0.0

    return frequencies


def compute_harvest_totals(
    harvests: List[HarvestRecord],
    by_product: bool = True
) -> Dict[str, float]:
    """
    Compute total harvest quantities.

    Args:
        harvests: List of harvest records
        by_product: If True, group by product_type

    Returns:
        Dict mapping product_type to total quantity (or 'total' if not grouped)
    """
    if by_product:
        by_product_dict = group_by(lambda h: h.product_type)(harvests)
        return {
            product: sum(h.quantity for h in records)
            for product, records in by_product_dict.items()
        }
    else:
        return {"total": sum(h.quantity for h in harvests)}


# =============================================================================
# Time-Series Analysis
# =============================================================================

def extract_time_series(
    inspections: List[Inspection],
    field_extractor: Callable[[Inspection], Optional[float]]
) -> List[tuple[datetime, float]]:
    """
    Extract time series data from inspections.

    Args:
        inspections: List of inspections
        field_extractor: Function to extract numeric value from inspection

    Returns:
        List of (timestamp, value) tuples
    """
    series = []
    for inspection in inspections:
        value = field_extractor(inspection)
        if value is not None:
            series.append((inspection.timestamp, value))

    return sorted(series, key=lambda x: x[0])


def compute_trend(
    time_series: List[tuple[datetime, float]]
) -> Dict[str, float]:
    """
    Compute simple linear trend from time series.

    Returns:
        Dict with 'slope', 'intercept', 'direction' (-1, 0, 1)
    """
    if len(time_series) < 2:
        return {"slope": 0.0, "intercept": 0.0, "direction": 0}

    # Convert timestamps to days from start
    start_time = time_series[0][0]
    points = [
        ((t - start_time).days, v)
        for t, v in time_series
    ]

    # Simple linear regression
    n = len(points)
    x_mean = sum(x for x, y in points) / n
    y_mean = sum(y for x, y in points) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    denominator = sum((x - x_mean) ** 2 for x, y in points)

    slope = numerator / denominator if denominator != 0 else 0.0
    intercept = y_mean - slope * x_mean

    direction = 1 if slope > 0.01 else (-1 if slope < -0.01 else 0)

    return {
        "slope": slope,
        "intercept": intercept,
        "direction": direction
    }


# =============================================================================
# Composite Transformations (Examples)
# =============================================================================

def get_top_healthy_colonies(n: int = 5) -> Callable[[List[Colony]], List[Colony]]:
    """Get top N healthiest colonies by population."""
    return pipe(
        filter_healthy,
        sort_by_population(reverse=True),
        take(n)
    )


def get_concerning_colonies() -> Callable[[List[Colony]], List[Colony]]:
    """Get colonies needing attention, sorted by severity."""
    return pipe(
        filter_concerning,
        sort_by_health(reverse=False)
    )


def get_critical_alerts() -> Callable[[List[Alert]], List[Alert]]:
    """Get unresolved critical/urgent alerts."""
    def filter_critical(alerts: List[Alert]) -> List[Alert]:
        return [
            a for a in alerts
            if not a.resolved and a.level in [AlertLevel.CRITICAL, AlertLevel.URGENT]
        ]

    return pipe(
        filter_critical,
        sort_alerts_by_priority(reverse=True)
    )


def analyze_recent_activity(days: int = 30) -> Callable[[List[Inspection]], Dict[str, Any]]:
    """Analyze recent inspection activity."""
    def analyze(inspections: List[Inspection]) -> Dict[str, Any]:
        recent = filter_recent(days)(inspections)

        return {
            "total_inspections": len(recent),
            "by_type": count_by(lambda i: i.inspection_type)(recent),
            "by_hive": count_by(lambda i: i.hive_id)(recent),
            "average_duration": (
                sum(i.duration_minutes or 0 for i in recent) / len(recent)
                if recent else 0
            ),
        }

    return analyze
