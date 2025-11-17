"""
Promptly Analytics - Comprehensive observability and monitoring

Provides performance tracking, usage analytics, quality metrics,
visualization, and reporting capabilities.

Quick Start:
    >>> from promptly import Promptly
    >>> from promptly.analytics import enable_analytics
    >>>
    >>> # Enable analytics for your Promptly instance
    >>> promptly = Promptly()
    >>> promptly = enable_analytics(promptly)
    >>>
    >>> # Use Promptly normally - all operations are tracked
    >>> promptly.add('greeting', 'Hello {name}!')
    >>>
    >>> # View statistics
    >>> stats = promptly.get_performance_stats()
    >>> usage = promptly.get_usage_stats()
    >>> quality = promptly.get_quality_stats()

CLI Usage:
    # View statistics
    $ python -m promptly.analytics.cli stats performance
    $ python -m promptly.analytics.cli stats usage --days 30
    $ python -m promptly.analytics.cli stats quality --prompt greeting

    # Generate reports
    $ python -m promptly.analytics.cli report daily --output ./report.md
    $ python -m promptly.analytics.cli report dashboard --output ./dashboard.html

    # Export data
    $ python -m promptly.analytics.cli export json --output ./analytics.json
    $ python -m promptly.analytics.cli export grafana --output-dir ./grafana

    # Cleanup old data
    $ python -m promptly.analytics.cli cleanup
"""

from .performance import PerformanceMonitor, OperationTimer
from .usage import UsageAnalytics, UsageTracker
from .quality import QualityMetrics, QualityTracker
from .visualize import Visualizer, ChartType
from .reports import ReportGenerator, ReportFormat, ReportPeriod, ReportConfig
from .integrations import (
    PrometheusExporter,
    OpenTelemetryIntegration,
    GrafanaExporter,
    AnalyticsHub,
    StructuredLogger
)
from .instrumentation import enable_analytics, PromptlyAnalytics, track_operation

__all__ = [
    # Core analytics
    'PerformanceMonitor',
    'OperationTimer',
    'UsageAnalytics',
    'UsageTracker',
    'QualityMetrics',
    'QualityTracker',

    # Visualization and reporting
    'Visualizer',
    'ChartType',
    'ReportGenerator',
    'ReportFormat',
    'ReportPeriod',
    'ReportConfig',

    # Integrations
    'PrometheusExporter',
    'OpenTelemetryIntegration',
    'GrafanaExporter',
    'AnalyticsHub',
    'StructuredLogger',

    # Instrumentation
    'enable_analytics',
    'PromptlyAnalytics',
    'track_operation',
]

__version__ = '0.1.0'
