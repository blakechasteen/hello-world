#!/usr/bin/env python3
"""
Intelligent Widget Builder for HoloLoom Dashboards
===================================================
Auto-constructs optimal visualizations from raw data.

Philosophy: "Show me the data, I'll build the dashboard"

The widget builder analyzes data and automatically:
1. Detects data types and patterns
2. Selects optimal visualization types
3. Generates intelligence insights
4. Constructs complete dashboard panels

Author: Claude Code
Date: October 29, 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import math


# ============================================================================
# Data Analysis
# ============================================================================

class DataType(str, Enum):
    """Detected data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BOOLEAN = "boolean"


class DataPattern(str, Enum):
    """Detected data patterns."""
    TIME_SERIES = "time_series"           # Sequential temporal data
    CORRELATION = "correlation"           # Two numeric variables related
    DISTRIBUTION = "distribution"         # Single variable spread
    COMPARISON = "comparison"             # Categorical groups
    RELATIONSHIP = "relationship"         # Multi-variate connections
    TREND = "trend"                       # Directional change
    OUTLIER = "outlier"                   # Anomalous values


@dataclass
class DataColumn:
    """Analyzed column information."""
    name: str
    data_type: DataType
    values: List[Any]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    unique_count: int = 0
    null_count: int = 0

    def is_numeric(self) -> bool:
        return self.data_type == DataType.NUMERIC

    def is_categorical(self) -> bool:
        return self.data_type == DataType.CATEGORICAL

    def is_temporal(self) -> bool:
        return self.data_type == DataType.TEMPORAL


class DataAnalyzer:
    """
    Analyzes raw data to detect types, patterns, and relationships.

    Takes raw data (dict, DataFrame-like, etc.) and produces structured
    metadata about the data suitable for visualization selection.
    """

    @staticmethod
    def analyze_column(name: str, values: List[Any]) -> DataColumn:
        """
        Analyze a single column to detect type and statistics.

        Args:
            name: Column name
            values: List of values

        Returns:
            DataColumn with metadata
        """
        # Detect type
        data_type = DataAnalyzer._detect_type(values)

        # Calculate statistics
        numeric_vals = []
        if data_type == DataType.NUMERIC:
            numeric_vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]

        unique_count = len(set(v for v in values if v is not None))
        null_count = sum(1 for v in values if v is None)

        return DataColumn(
            name=name,
            data_type=data_type,
            values=values,
            min_val=min(numeric_vals) if numeric_vals else None,
            max_val=max(numeric_vals) if numeric_vals else None,
            mean=sum(numeric_vals) / len(numeric_vals) if numeric_vals else None,
            std=DataAnalyzer._std(numeric_vals) if numeric_vals else None,
            unique_count=unique_count,
            null_count=null_count
        )

    @staticmethod
    def _detect_type(values: List[Any]) -> DataType:
        """Detect data type from values."""
        non_null = [v for v in values if v is not None]
        if not non_null:
            return DataType.TEXT

        # Check if numeric
        numeric_count = sum(1 for v in non_null if isinstance(v, (int, float)))
        if numeric_count / len(non_null) > 0.8:
            return DataType.NUMERIC

        # Check if boolean
        bool_count = sum(1 for v in non_null if isinstance(v, bool))
        if bool_count / len(non_null) > 0.8:
            return DataType.BOOLEAN

        # Check if temporal (simple heuristic - look for month names, numbers in sequence)
        temporal_keywords = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                           'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        str_vals = [str(v).lower() for v in non_null[:10]]
        if any(kw in sv for sv in str_vals for kw in temporal_keywords):
            return DataType.TEMPORAL

        # Check if categorical (low unique count)
        unique_count = len(set(non_null))
        if unique_count < len(non_null) * 0.5:  # < 50% unique
            return DataType.CATEGORICAL

        return DataType.TEXT

    @staticmethod
    def _std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    @staticmethod
    def detect_patterns(columns: List[DataColumn]) -> List[DataPattern]:
        """
        Detect high-level patterns in the dataset.

        Args:
            columns: List of analyzed columns

        Returns:
            List of detected patterns
        """
        patterns = []

        # Time series: temporal column + numeric column(s)
        temporal_cols = [c for c in columns if c.is_temporal()]
        numeric_cols = [c for c in columns if c.is_numeric()]

        if temporal_cols and numeric_cols:
            patterns.append(DataPattern.TIME_SERIES)

        # Correlation: 2+ numeric columns
        if len(numeric_cols) >= 2:
            patterns.append(DataPattern.CORRELATION)

        # Distribution: single numeric column
        if len(numeric_cols) >= 1:
            patterns.append(DataPattern.DISTRIBUTION)

        # Comparison: categorical + numeric
        categorical_cols = [c for c in columns if c.is_categorical()]
        if categorical_cols and numeric_cols:
            patterns.append(DataPattern.COMPARISON)

        # Detect trends (increasing/decreasing values)
        for col in numeric_cols:
            if DataAnalyzer._has_trend(col.values):
                patterns.append(DataPattern.TREND)
                break

        # Detect outliers
        for col in numeric_cols:
            if DataAnalyzer._has_outliers(col):
                patterns.append(DataPattern.OUTLIER)
                break

        return patterns

    @staticmethod
    def _has_trend(values: List[float]) -> bool:
        """Check if values show a trend (simple linear check)."""
        if len(values) < 3:
            return False

        # Count increases vs decreases
        increases = sum(1 for i in range(len(values)-1) if values[i+1] > values[i])
        decreases = sum(1 for i in range(len(values)-1) if values[i+1] < values[i])

        # Strong trend if >70% going one direction
        total = increases + decreases
        if total == 0:
            return False

        return max(increases, decreases) / total > 0.7

    @staticmethod
    def _has_outliers(col: DataColumn) -> bool:
        """Check if column has outliers (simple 2-sigma check)."""
        if col.mean is None or col.std is None or col.std == 0:
            return False

        # Check if any value is >2 std deviations from mean
        for val in col.values:
            if val is not None and abs(val - col.mean) > 2 * col.std:
                return True

        return False

    @staticmethod
    def calculate_correlation(col1: DataColumn, col2: DataColumn) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            col1: First numeric column
            col2: Second numeric column

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if not (col1.is_numeric() and col2.is_numeric()):
            return 0.0

        # Get paired values (exclude nulls)
        pairs = [(v1, v2) for v1, v2 in zip(col1.values, col2.values)
                 if v1 is not None and v2 is not None]

        if len(pairs) < 2:
            return 0.0

        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]

        n = len(pairs)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in pairs)
        sum_x2 = sum(x * x for x in x_vals)
        sum_y2 = sum(y * y for y in y_vals)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

        if denominator == 0:
            return 0.0

        return numerator / denominator


# ============================================================================
# Visualization Selection
# ============================================================================

from .dashboard import PanelType, PanelSize


@dataclass
class VisualizationRecommendation:
    """Recommended visualization with reasoning."""
    panel_type: PanelType
    priority: int  # 0-100, higher = more important
    reasoning: str
    data_mapping: Dict[str, str]  # 'x' -> 'column_name', etc.
    size: PanelSize = PanelSize.LARGE


class VisualizationSelector:
    """
    Selects optimal visualization types based on data analysis.

    Uses heuristics and pattern matching to recommend chart types.
    """

    @staticmethod
    def recommend_visualizations(
        columns: List[DataColumn],
        patterns: List[DataPattern]
    ) -> List[VisualizationRecommendation]:
        """
        Recommend visualizations for the dataset.

        Args:
            columns: Analyzed columns
            patterns: Detected patterns

        Returns:
            List of visualization recommendations sorted by priority
        """
        recommendations = []

        numeric_cols = [c for c in columns if c.is_numeric()]
        categorical_cols = [c for c in columns if c.is_categorical()]
        temporal_cols = [c for c in columns if c.is_temporal()]

        # Time series → Line chart (high priority)
        if DataPattern.TIME_SERIES in patterns and temporal_cols and numeric_cols:
            recommendations.append(VisualizationRecommendation(
                panel_type=PanelType.LINE,
                priority=90,
                reasoning="Temporal data with numeric values - perfect for trend analysis",
                data_mapping={
                    'x': temporal_cols[0].name,
                    'y': [c.name for c in numeric_cols]
                },
                size=PanelSize.FULL_WIDTH
            ))

        # Correlation → Scatter plot
        if DataPattern.CORRELATION in patterns and len(numeric_cols) >= 2:
            recommendations.append(VisualizationRecommendation(
                panel_type=PanelType.SCATTER,
                priority=85,
                reasoning="Two numeric variables - investigate correlation",
                data_mapping={
                    'x': numeric_cols[0].name,
                    'y': numeric_cols[1].name
                },
                size=PanelSize.LARGE
            ))

        # Comparison → Bar chart
        if DataPattern.COMPARISON in patterns and categorical_cols and numeric_cols:
            recommendations.append(VisualizationRecommendation(
                panel_type=PanelType.BAR,
                priority=80,
                reasoning="Categorical groups with values - compare across categories",
                data_mapping={
                    'categories': categorical_cols[0].name,
                    'values': numeric_cols[0].name
                },
                size=PanelSize.LARGE
            ))

        # Key metrics → Metric cards
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            recommendations.append(VisualizationRecommendation(
                panel_type=PanelType.METRIC,
                priority=70,
                reasoning=f"Key numeric metric: {col.name}",
                data_mapping={'value': col.name},
                size=PanelSize.SMALL
            ))

        # Distribution → Histogram/heatmap
        if DataPattern.DISTRIBUTION in patterns and numeric_cols:
            recommendations.append(VisualizationRecommendation(
                panel_type=PanelType.DISTRIBUTION,
                priority=60,
                reasoning="Show data distribution and variance",
                data_mapping={'values': numeric_cols[0].name},
                size=PanelSize.MEDIUM
            ))

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        return recommendations


# ============================================================================
# Insight Generation
# ============================================================================

@dataclass
class GeneratedInsight:
    """Auto-generated intelligence insight."""
    type: str  # 'trend', 'correlation', 'outlier', 'pattern', 'recommendation'
    title: str
    message: str
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    priority: int = 50


class InsightGenerator:
    """
    Auto-generates intelligence insights from data analysis.

    Detects patterns, correlations, trends, outliers and generates
    human-readable insight cards.
    """

    @staticmethod
    def generate_insights(
        columns: List[DataColumn],
        patterns: List[DataPattern]
    ) -> List[GeneratedInsight]:
        """
        Generate all possible insights from analyzed data.

        Args:
            columns: Analyzed columns
            patterns: Detected patterns

        Returns:
            List of generated insights
        """
        insights = []

        numeric_cols = [c for c in columns if c.is_numeric()]

        # Trend insights
        if DataPattern.TREND in patterns:
            for col in numeric_cols:
                insight = InsightGenerator._generate_trend_insight(col)
                if insight:
                    insights.append(insight)

        # Correlation insights
        if DataPattern.CORRELATION in patterns and len(numeric_cols) >= 2:
            insight = InsightGenerator._generate_correlation_insight(
                numeric_cols[0], numeric_cols[1]
            )
            if insight:
                insights.append(insight)

        # Outlier insights
        if DataPattern.OUTLIER in patterns:
            for col in numeric_cols:
                insight = InsightGenerator._generate_outlier_insight(col)
                if insight:
                    insights.append(insight)

        # Statistical summary insights
        for col in numeric_cols:
            insight = InsightGenerator._generate_summary_insight(col)
            if insight:
                insights.append(insight)

        # Sort by priority
        insights.sort(key=lambda i: i.priority, reverse=True)

        return insights

    @staticmethod
    def _generate_trend_insight(col: DataColumn) -> Optional[GeneratedInsight]:
        """Generate trend insight for a column."""
        if col.mean is None or len(col.values) < 3:
            return None

        # Determine trend direction
        first_third = [v for v in col.values[:len(col.values)//3] if v is not None]
        last_third = [v for v in col.values[-len(col.values)//3:] if v is not None]

        if not first_third or not last_third:
            return None

        start_avg = sum(first_third) / len(first_third)
        end_avg = sum(last_third) / len(last_third)

        change_pct = ((end_avg - start_avg) / start_avg * 100) if start_avg != 0 else 0

        if abs(change_pct) < 5:  # Insignificant change
            return None

        direction = "increasing" if change_pct > 0 else "decreasing"
        magnitude = "sharply" if abs(change_pct) > 20 else "steadily"

        return GeneratedInsight(
            type='trend',
            title=f"{col.name.title()} is {magnitude} {direction}",
            message=f"The {col.name} shows a {magnitude} {direction} trend with {abs(change_pct):.1f}% change from start to end.",
            confidence=min(0.95, abs(change_pct) / 100),
            details={
                'Direction': direction.title(),
                'Change': f'{change_pct:+.1f}%',
                'Start': f'{start_avg:.2f}',
                'End': f'{end_avg:.2f}'
            },
            priority=85
        )

    @staticmethod
    def _generate_correlation_insight(
        col1: DataColumn,
        col2: DataColumn
    ) -> Optional[GeneratedInsight]:
        """Generate correlation insight between two columns."""
        corr = DataAnalyzer.calculate_correlation(col1, col2)

        if abs(corr) < 0.3:  # Weak correlation
            return None

        strength = "strong" if abs(corr) > 0.7 else "moderate"
        direction = "positive" if corr > 0 else "negative"

        return GeneratedInsight(
            type='correlation',
            title=f"{strength.title()} {direction} correlation detected",
            message=f"{col1.name.title()} and {col2.name.title()} show a {strength} {direction} relationship (r={corr:.3f}).",
            confidence=abs(corr),
            details={
                'Correlation': f'{corr:.3f}',
                'Strength': strength.title(),
                'Direction': direction.title(),
                'Variables': f'{col1.name} vs {col2.name}'
            },
            priority=90
        )

    @staticmethod
    def _generate_outlier_insight(col: DataColumn) -> Optional[GeneratedInsight]:
        """Generate outlier insight for a column."""
        if col.mean is None or col.std is None or col.std == 0:
            return None

        # Find outliers (>2 sigma)
        outliers = []
        for i, val in enumerate(col.values):
            if val is not None and abs(val - col.mean) > 2 * col.std:
                outliers.append((i, val))

        if not outliers:
            return None

        outlier_val = outliers[0][1]
        outlier_idx = outliers[0][0]

        return GeneratedInsight(
            type='outlier',
            title=f"Anomalous value detected in {col.name}",
            message=f"Value at position {outlier_idx} ({outlier_val:.2f}) is {abs(outlier_val - col.mean) / col.std:.1f} standard deviations from the mean.",
            confidence=0.85,
            details={
                'Outlier Value': f'{outlier_val:.2f}',
                'Mean': f'{col.mean:.2f}',
                'Std Dev': f'{col.std:.2f}',
                'Total Outliers': len(outliers)
            },
            priority=75
        )

    @staticmethod
    def _generate_summary_insight(col: DataColumn) -> Optional[GeneratedInsight]:
        """Generate statistical summary insight."""
        if col.mean is None:
            return None

        return GeneratedInsight(
            type='pattern',
            title=f"{col.name.title()} statistical summary",
            message=f"Mean: {col.mean:.2f}, Range: {col.min_val:.2f} to {col.max_val:.2f}, Std Dev: {col.std:.2f}",
            confidence=1.0,
            details={
                'Mean': f'{col.mean:.2f}',
                'Min': f'{col.min_val:.2f}',
                'Max': f'{col.max_val:.2f}',
                'Std Dev': f'{col.std:.2f}'
            },
            priority=50
        )


# ============================================================================
# Widget Builder (Main Orchestrator)
# ============================================================================

from .dashboard import Panel, Dashboard, LayoutType, ComplexityLevel


class WidgetBuilder:
    """
    Main widget builder orchestrator.

    Takes raw data → analyzes → selects visualizations → generates insights
    → constructs complete dashboard.

    Usage:
        builder = WidgetBuilder()
        dashboard = builder.build_from_data(
            data={'month': [...], 'survival': [...], 'treatment': [...]},
            title="My Analysis"
        )
    """

    def __init__(self):
        self.analyzer = DataAnalyzer()
        self.selector = VisualizationSelector()
        self.insight_gen = InsightGenerator()

    def build_from_data(
        self,
        data: Dict[str, List[Any]],
        title: str = "Auto-Generated Dashboard",
        max_panels: int = 12,
        spacetime: Any = None
    ) -> Dashboard:
        """
        Build complete dashboard from raw data.

        Args:
            data: Dictionary of column_name -> values
            title: Dashboard title
            max_panels: Maximum number of panels to generate
            spacetime: Optional Spacetime object (for metadata)

        Returns:
            Complete Dashboard object ready for rendering
        """
        # Step 1: Analyze data
        columns = [
            self.analyzer.analyze_column(name, values)
            for name, values in data.items()
        ]

        patterns = self.analyzer.detect_patterns(columns)

        print(f"\n[WidgetBuilder] Analyzed {len(columns)} columns")
        print(f"[WidgetBuilder] Detected patterns: {[p.value for p in patterns]}")

        # Step 2: Get visualization recommendations
        viz_recs = self.selector.recommend_visualizations(columns, patterns)

        print(f"[WidgetBuilder] Generated {len(viz_recs)} visualization recommendations")

        # Step 3: Generate insights
        insights = self.insight_gen.generate_insights(columns, patterns)

        print(f"[WidgetBuilder] Generated {len(insights)} insights")

        # Step 4: Build panels
        panels = []

        # Add visualization panels
        for i, rec in enumerate(viz_recs[:max_panels//2]):  # Leave room for insights
            panel = self._build_panel_from_recommendation(rec, columns, data, i)
            if panel:
                panels.append(panel)

        # Add insight panels
        for i, insight in enumerate(insights[:max_panels//3]):  # Top insights
            panel = self._build_insight_panel(insight, len(panels) + i)
            if panel:
                panels.append(panel)

        print(f"[WidgetBuilder] Built {len(panels)} panels total")

        # Step 5: Select layout based on panel count
        if len(panels) <= 3:
            layout = LayoutType.METRIC
        elif len(panels) <= 6:
            layout = LayoutType.FLOW
        else:
            layout = LayoutType.RESEARCH

        # Step 6: Create dashboard
        from datetime import datetime

        # Mock spacetime if not provided
        if spacetime is None:
            @dataclass
            class MockSpacetime:
                query_text: str = title
                response: str = "Auto-generated analysis"
                tool_used: str = "widget_builder"
                confidence: float = 0.95
                trace: Any = None
                metadata: Dict[str, Any] = field(default_factory=dict)

                def to_dict(self):
                    return {'query': self.query_text}

            spacetime = MockSpacetime()

        dashboard = Dashboard(
            title=title,
            layout=layout,
            panels=panels,
            spacetime=spacetime,
            metadata={
                'complexity': ComplexityLevel.FULL,
                'panel_count': len(panels),
                'generated_at': datetime.now().isoformat(),
                'auto_generated': True,
                'patterns_detected': [p.value for p in patterns]
            }
        )

        return dashboard

    def _build_panel_from_recommendation(
        self,
        rec: VisualizationRecommendation,
        columns: List[DataColumn],
        data: Dict[str, List[Any]],
        panel_idx: int
    ) -> Optional[Panel]:
        """Build a Panel from a visualization recommendation."""

        panel_id = f"panel_{panel_idx}"

        # Build based on panel type
        if rec.panel_type == PanelType.LINE:
            return self._build_line_panel(rec, columns, data, panel_id)
        elif rec.panel_type == PanelType.SCATTER:
            return self._build_scatter_panel(rec, columns, data, panel_id)
        elif rec.panel_type == PanelType.BAR:
            return self._build_bar_panel(rec, columns, data, panel_id)
        elif rec.panel_type == PanelType.METRIC:
            return self._build_metric_panel(rec, columns, data, panel_id)

        return None

    def _build_line_panel(self, rec, columns, data, panel_id) -> Panel:
        """Build line chart panel."""
        x_col = rec.data_mapping['x']
        y_cols = rec.data_mapping['y']

        if isinstance(y_cols, str):
            y_cols = [y_cols]

        traces = []
        colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

        for i, y_col in enumerate(y_cols):
            traces.append({
                'name': y_col.replace('_', ' ').title(),
                'x': data[x_col],
                'y': data[y_col],
                'color': colors[i % len(colors)]
            })

        return Panel(
            id=panel_id,
            type=PanelType.LINE,
            title=f"Trends: {', '.join(y_cols)}",
            subtitle=rec.reasoning,
            data={
                'traces': traces,
                'x_label': x_col.replace('_', ' ').title(),
                'y_label': 'Value',
                'show_points': True
            },
            size=rec.size
        )

    def _build_scatter_panel(self, rec, columns, data, panel_id) -> Panel:
        """Build scatter plot panel."""
        x_col = rec.data_mapping['x']
        y_col = rec.data_mapping['y']

        # Find column objects
        x_column = next((c for c in columns if c.name == x_col), None)
        y_column = next((c for c in columns if c.name == y_col), None)

        if not (x_column and y_column):
            return None

        # Calculate correlation
        corr = self.analyzer.calculate_correlation(x_column, y_column)

        return Panel(
            id=panel_id,
            type=PanelType.SCATTER,
            title=f"{y_col.title()} vs {x_col.title()}",
            subtitle=rec.reasoning,
            data={
                'x': data[x_col],
                'y': data[y_col],
                'labels': [f'Point {i+1}' for i in range(len(data[x_col]))],
                'x_label': x_col.replace('_', ' ').title(),
                'y_label': y_col.replace('_', ' ').title(),
                'correlation': corr,
                'colors': ['#6366f1'] * len(data[x_col]),
                'sizes': [10] * len(data[x_col])
            },
            size=rec.size
        )

    def _build_bar_panel(self, rec, columns, data, panel_id) -> Panel:
        """Build bar chart panel."""
        cat_col = rec.data_mapping['categories']
        val_col = rec.data_mapping['values']

        return Panel(
            id=panel_id,
            type=PanelType.BAR,
            title=f"{val_col.title()} by {cat_col.title()}",
            subtitle=rec.reasoning,
            data={
                'categories': data[cat_col],
                'values': data[val_col],
                'orientation': 'h' if len(data[cat_col]) > 5 else 'v',
                'x_label': cat_col.replace('_', ' ').title(),
                'y_label': val_col.replace('_', ' ').title(),
                'colors': ['#6366f1'] * len(data[cat_col])
            },
            size=rec.size
        )

    def _build_metric_panel(self, rec, columns, data, panel_id) -> Panel:
        """Build metric card panel."""
        val_col = rec.data_mapping['value']

        # Find column
        column = next((c for c in columns if c.name == val_col), None)
        if not column or column.mean is None:
            return None

        # Use mean as the metric value
        value = column.mean

        # Determine color based on value (simple heuristic)
        if value > 80:
            color = 'green'
        elif value > 50:
            color = 'blue'
        else:
            color = 'yellow'

        return Panel(
            id=panel_id,
            type=PanelType.METRIC,
            title=val_col.replace('_', ' ').title(),
            data={
                'value': value,
                'formatted': f'{value:.1f}',
                'label': val_col.replace('_', ' ').title(),
                'color': color
            },
            size=PanelSize.SMALL
        )

    def _build_insight_panel(self, insight: GeneratedInsight, panel_idx: int) -> Panel:
        """Build insight card panel from generated insight."""
        return Panel(
            id=f"insight_{panel_idx}",
            type=PanelType.INSIGHT,
            title=insight.title,
            data={
                'type': insight.type,
                'message': insight.message,
                'confidence': insight.confidence,
                'details': insight.details
            },
            size=PanelSize.MEDIUM
        )
