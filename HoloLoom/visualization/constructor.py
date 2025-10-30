#!/usr/bin/env python3
"""
DashboardConstructor - Builds Dashboards from Spacetime
=======================================================
Extracts data from Spacetime artifacts and constructs Dashboard objects
using StrategySelector's intelligent panel selection.

Author: Claude Code with HoloLoom architecture
Date: October 28, 2025
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from .dashboard import Dashboard, Panel, PanelSpec, PanelType
from .strategy import StrategySelector, UserPreferences


class DashboardConstructor:
    """
    Constructs Dashboard from Spacetime using selected strategy.

    Responsibilities:
    1. Use StrategySelector to determine panels
    2. Extract data from Spacetime based on PanelSpecs
    3. Create Panel objects with extracted data
    4. Assemble complete Dashboard

    Usage:
        constructor = DashboardConstructor()
        dashboard = constructor.construct(spacetime)
        # dashboard contains panels with data from spacetime
    """

    def __init__(self, strategy_selector: Optional[StrategySelector] = None):
        """
        Initialize dashboard constructor.

        Args:
            strategy_selector: Optional custom StrategySelector
                             (default: creates new one)
        """
        self.strategy_selector = strategy_selector or StrategySelector()

    def construct(self, spacetime) -> Dashboard:
        """
        Construct dashboard from Spacetime.

        Args:
            spacetime: Spacetime artifact from weaving

        Returns:
            Dashboard with panels containing extracted data
        """
        # 1. Select strategy (panels to show)
        strategy = self.strategy_selector.select(spacetime)

        # 2. Extract data for each panel
        panels = []
        for spec in strategy.panels:
            panel = self._create_panel(spec, spacetime)
            if panel:  # Only add if data extraction succeeded
                panels.append(panel)

        # 3. Assemble dashboard
        # Handle complexity as enum or string
        if isinstance(strategy.complexity_level, str):
            complexity_str = strategy.complexity_level
        elif hasattr(strategy.complexity_level, 'value'):
            complexity_str = strategy.complexity_level.value
        else:
            complexity_str = str(strategy.complexity_level)

        return Dashboard(
            title=strategy.title,
            layout=strategy.layout_type,
            panels=panels,
            spacetime=spacetime,
            metadata={
                'complexity': complexity_str,
                'panel_count': len(panels),
                'generated_at': datetime.now().isoformat()
            }
        )

    def _create_panel(self, spec: PanelSpec, spacetime) -> Optional[Panel]:
        """
        Create panel by extracting data from Spacetime.

        Args:
            spec: Panel specification (what to show)
            spacetime: Spacetime artifact (where to get data)

        Returns:
            Panel with data, or None if extraction failed
        """
        # Extract data based on data_source
        data = self._extract_data(spec.data_source, spacetime)

        if data is None:
            return None  # Skip panel if no data available

        # Create unique panel ID
        panel_id = f"{spec.type.value}_{spec.data_source.replace('.', '_')}"

        return Panel(
            id=panel_id,
            type=spec.type,
            title=spec.title or spec.data_source,
            subtitle=spec.subtitle,
            data=data,
            size=spec.size
        )

    def _extract_data(
        self,
        data_source: str,
        spacetime
    ) -> Optional[Dict[str, Any]]:
        """
        Extract data from Spacetime based on data_source path.

        Supports dot notation: 'trace.stage_durations'

        Args:
            data_source: Dot-separated path to data
            spacetime: Spacetime artifact

        Returns:
            Dict with extracted data formatted for visualization,
            or None if extraction failed
        """
        # Handle special/computed data sources
        if data_source == 'confidence':
            return self._format_metric(
                value=spacetime.confidence,
                label='Confidence',
                unit='%',
                color=self._confidence_color(spacetime.confidence)
            )

        elif data_source == 'trace.duration_ms':
            return self._format_metric(
                value=spacetime.trace.duration_ms,
                label='Duration',
                unit='ms',
                color=self._duration_color(spacetime.trace.duration_ms)
            )

        elif data_source == 'trace.stage_durations':
            return self._format_timeline(spacetime.trace.stage_durations)

        elif data_source == 'trace.threads_activated':
            return self._format_network(spacetime.trace.threads_activated)

        elif data_source == 'trace.errors':
            if hasattr(spacetime.trace, 'errors') and spacetime.trace.errors:
                return self._format_errors(spacetime.trace.errors)
            return None

        elif data_source == 'query_text':
            return self._format_text(spacetime.query_text, 'Query')

        elif data_source == 'response':
            return self._format_text(spacetime.response, 'Response')

        elif data_source == 'semantic_dimensions':
            # Extract from metadata if available
            if hasattr(spacetime, 'metadata') and 'semantic_cache' in spacetime.metadata:
                return self._format_semantic_profile(spacetime.metadata)
            return None

        elif data_source == 'bottleneck':
            # Compute bottleneck from stage durations
            return self._format_bottleneck(spacetime.trace.stage_durations)

        # Generic extraction using dot notation
        return self._extract_generic(data_source, spacetime)

    def _extract_generic(
        self,
        data_source: str,
        spacetime
    ) -> Optional[Dict[str, Any]]:
        """
        Generic data extraction using dot notation.

        Example: 'trace.stage_durations' → spacetime.trace.stage_durations

        Args:
            data_source: Dot-separated path
            spacetime: Spacetime object

        Returns:
            Extracted data or None
        """
        parts = data_source.split('.')
        value = spacetime

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return None  # Path doesn't exist

        # Wrap raw value
        return {'raw': value}

    # ========================================================================
    # Data Formatters (format extracted data for visualization)
    # ========================================================================

    def _format_metric(
        self,
        value: float,
        label: str,
        unit: str = '',
        color: str = 'blue'
    ) -> Dict[str, Any]:
        """
        Format single metric value.

        Returns:
            Dict with metric data for rendering
        """
        return {
            'value': value,
            'label': label,
            'unit': unit,
            'color': color,
            'formatted': f"{value:.2f}{unit}"
        }

    def _format_timeline(
        self,
        stage_durations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Format timeline data for waterfall chart with bottleneck detection.

        Returns:
            Dict with stages, durations, and bottleneck information for Plotly
        """
        if not stage_durations:
            return None

        stages = list(stage_durations.keys())
        durations = list(stage_durations.values())

        # Calculate percentages
        total = sum(durations)
        percentages = [d / total * 100 for d in durations] if total > 0 else [0] * len(durations)

        # Detect bottleneck (stage taking >40% of total time)
        BOTTLENECK_THRESHOLD = 40.0
        bottleneck_idx = None
        bottleneck_stage = None
        bottleneck_percentage = 0.0

        if total > 0:
            for i, (stage, duration) in enumerate(zip(stages, durations)):
                pct = (duration / total) * 100
                if pct > BOTTLENECK_THRESHOLD:
                    bottleneck_idx = i
                    bottleneck_stage = stage
                    bottleneck_percentage = pct
                    break  # First bottleneck found

        # Generate optimization suggestions based on bottleneck
        optimization_suggestion = None
        if bottleneck_stage:
            optimization_suggestion = self._get_optimization_suggestion(
                bottleneck_stage,
                durations[bottleneck_idx],
                bottleneck_percentage
            )

        # Assign colors to stages (red/orange for bottleneck, blue for normal)
        colors = []
        for i, pct in enumerate(percentages):
            if i == bottleneck_idx:
                # Red for severe bottleneck (>50%), orange for moderate (40-50%)
                colors.append('#ef4444' if pct > 50 else '#f97316')  # red-500 : orange-500
            else:
                colors.append('#3b82f6')  # blue-500 (normal)

        return {
            'type': 'timeline',
            'stages': stages,
            'durations': durations,
            'percentages': percentages,
            'total': total,
            'colors': colors,
            'bottleneck': {
                'detected': bottleneck_idx is not None,
                'stage': bottleneck_stage,
                'index': bottleneck_idx,
                'percentage': bottleneck_percentage,
                'threshold': BOTTLENECK_THRESHOLD,
                'optimization': optimization_suggestion
            }
        }

    def _get_optimization_suggestion(
        self,
        stage: str,
        duration_ms: float,
        percentage: float
    ) -> str:
        """
        Generate actionable optimization suggestion based on bottleneck stage.

        Args:
            stage: Name of bottleneck stage
            duration_ms: Duration in milliseconds
            percentage: Percentage of total time

        Returns:
            Human-readable optimization suggestion
        """
        # Map stage names to optimization suggestions
        suggestions = {
            'retrieval': (
                "Consider enabling semantic cache for faster retrieval. "
                "Current retrieval time is {:.1f}ms ({:.0f}% of total). "
                "Expected speedup: 3-10x with caching."
            ),
            'pattern_selection': (
                "Pattern selection is taking {:.1f}ms ({:.0f}% of total). "
                "Consider using BARE mode for simpler queries or pre-selecting patterns."
            ),
            'feature_extraction': (
                "Feature extraction is slow ({:.1f}ms, {:.0f}% of total). "
                "Try reducing embedding scales or disabling spectral features for faster processing."
            ),
            'convergence': (
                "Decision convergence is taking {:.1f}ms ({:.0f}% of total). "
                "Consider simplifying the decision space or using epsilon-greedy strategy."
            ),
            'tool_execution': (
                "Tool execution is the bottleneck ({:.1f}ms, {:.0f}% of total). "
                "This is often expected for complex tools. Consider caching tool results if applicable."
            ),
            'warp_space': (
                "Warp space operations are slow ({:.1f}ms, {:.0f}% of total). "
                "Consider reducing the number of activated threads or using simpler tensor operations."
            ),
        }

        # Find matching suggestion (case-insensitive, partial match)
        stage_lower = stage.lower().replace('_', ' ')
        for key, template in suggestions.items():
            if key.replace('_', ' ') in stage_lower or stage_lower in key.replace('_', ' '):
                return template.format(duration_ms, percentage)

        # Generic suggestion if no specific match
        return (
            f"Stage '{stage}' is taking {duration_ms:.1f}ms ({percentage:.0f}% of total). "
            "Consider profiling this stage for optimization opportunities."
        )

    def _format_network(
        self,
        threads: List[str]
    ) -> Dict[str, Any]:
        """
        Format network graph data with enhanced structure.

        Returns:
            Dict with nodes and edges for force-directed visualization
        """
        if not threads:
            return None

        # Create central query node
        nodes = [
            {
                'id': 'query',
                'label': 'Query',
                'type': 'query',
                'color': '#8b5cf6',  # Purple for query
                'size': 20
            }
        ]

        # Add thread nodes connected to query
        edges = []
        for i, thread in enumerate(threads):
            node_id = f'thread_{i}'
            nodes.append({
                'id': node_id,
                'label': thread[:30] + ('...' if len(thread) > 30 else ''),  # Truncate long labels
                'fullLabel': thread,  # Full label for tooltip
                'type': 'thread',
                'color': '#6366f1',  # Indigo for threads
                'size': 12
            })

            # Connect thread to query
            edges.append({
                'source': 'query',
                'target': node_id
            })

        # Create inter-thread connections (every 3rd thread for reasonable density)
        for i in range(len(threads)):
            if i % 3 == 0 and i + 1 < len(threads):
                edges.append({
                    'source': f'thread_{i}',
                    'target': f'thread_{i+1}'
                })

        return {
            'type': 'network',
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'threads': threads  # Keep original list for fallback
        }

    def _format_errors(self, errors: List) -> Dict[str, Any]:
        """
        Format error list for display.

        Returns:
            Dict with error information
        """
        return {
            'type': 'errors',
            'errors': errors,
            'count': len(errors),
            'severity': 'high' if errors else 'none'
        }

    def _format_text(self, text: str, label: str) -> Dict[str, Any]:
        """
        Format text content.

        Returns:
            Dict with text data
        """
        return {
            'type': 'text',
            'content': text,
            'label': label,
            'length': len(text)
        }

    def _format_semantic_profile(
        self,
        metadata: Dict
    ) -> Dict[str, Any]:
        """
        Format semantic dimension data for heatmap.

        Extracts top N most activated semantic dimensions and their scores.

        Returns:
            Dict with semantic dimensions and scores for Plotly heatmap
        """
        cache_info = metadata.get('semantic_cache', {})

        if not cache_info.get('enabled'):
            return None

        # Extract dimension scores from cache (if available)
        dimension_scores = cache_info.get('dimension_scores', {})

        # If no scores available, try to compute from available data
        if not dimension_scores:
            # Try to get query embedding and dimensions
            query_embedding = cache_info.get('query_embedding')
            dimension_axes = cache_info.get('dimension_axes', {})

            if query_embedding is not None and dimension_axes:
                # Compute projections
                dimension_scores = self._compute_dimension_projections(
                    query_embedding,
                    dimension_axes
                )

        if not dimension_scores:
            # Fallback: generate sample data for demonstration
            dimension_scores = self._generate_sample_dimensions()

        # Sort dimensions by absolute score (most activated first)
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Take top N dimensions (configurable, default 20)
        top_n = 20
        top_dimensions = sorted_dims[:top_n]

        # Format for Plotly heatmap
        dim_names = [name for name, _ in top_dimensions]
        dim_scores = [score for _, score in top_dimensions]

        return {
            'type': 'semantic_heatmap',
            'cache_enabled': True,
            'hit_rate': cache_info.get('hit_rate', 0),
            'dimension_names': dim_names,
            'dimension_scores': dim_scores,
            'total_dimensions': len(dimension_scores),
            'showing_top': top_n
        }

    def _compute_dimension_projections(
        self,
        query_embedding,
        dimension_axes: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute projections of query onto semantic dimension axes.

        Args:
            query_embedding: Query embedding vector (numpy array or list)
            dimension_axes: Dict mapping dimension names to axis vectors

        Returns:
            Dict mapping dimension names to projection scores
        """
        import numpy as np

        # Convert query embedding to numpy array if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        projections = {}
        for dim_name, axis in dimension_axes.items():
            if isinstance(axis, (list, tuple)):
                axis = np.array(axis)

            # Compute dot product (projection)
            projection = float(np.dot(query_norm, axis))
            projections[dim_name] = projection

        return projections

    def _generate_sample_dimensions(self) -> Dict[str, float]:
        """
        Generate sample dimension scores for demonstration.

        Returns diverse dimension activations for visualization testing.

        Returns:
            Dict mapping dimension names to sample scores
        """
        import random
        random.seed(42)  # Reproducible samples

        # Common semantic dimensions with varied scores
        sample_dims = {
            'Warmth': 0.72,
            'Formality': -0.45,
            'Technical': 0.85,
            'Abstract': 0.63,
            'Concrete': -0.58,
            'Valence': 0.41,
            'Arousal': 0.28,
            'Dominance': 0.19,
            'Complexity': 0.76,
            'Urgency': -0.32,
            'Certainty': 0.54,
            'Specificity': 0.67,
            'Generality': -0.61,
            'Creativity': 0.39,
            'Analytical': 0.81,
            'Emotional': -0.23,
            'Objective': 0.58,
            'Subjective': -0.52,
            'Action-oriented': 0.44,
            'Reflective': 0.36,
            'Progressive': 0.29,
            'Traditional': -0.41,
            'Collaborative': 0.47,
            'Individual': -0.25
        }

        return sample_dims

    def _format_bottleneck(
        self,
        stage_durations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Identify and format bottleneck stage.

        Returns:
            Dict with bottleneck information
        """
        if not stage_durations:
            return None

        # Find slowest stage
        slowest_stage = max(stage_durations.items(), key=lambda x: x[1])
        stage_name, duration = slowest_stage

        total = sum(stage_durations.values())
        percentage = (duration / total * 100) if total > 0 else 0

        return {
            'value': duration,
            'label': 'Bottleneck',
            'stage': stage_name,
            'percentage': percentage,
            'unit': 'ms',
            'color': 'red' if percentage > 50 else 'orange',
            'formatted': f"{stage_name} ({duration:.1f}ms, {percentage:.0f}%)"
        }

    # ========================================================================
    # Color Helpers (semantic color coding)
    # ========================================================================

    def _confidence_color(self, confidence: float) -> str:
        """Semantic color for confidence score."""
        if confidence >= 0.8:
            return 'green'
        elif confidence >= 0.6:
            return 'yellow'
        else:
            return 'red'

    def _duration_color(self, duration_ms: float) -> str:
        """Semantic color for duration."""
        if duration_ms < 100:
            return 'green'
        elif duration_ms < 500:
            return 'yellow'
        else:
            return 'red'


# Convenience function
def create_dashboard_constructor(
    strategy_selector: Optional[StrategySelector] = None
) -> DashboardConstructor:
    """
    Create a DashboardConstructor instance.

    Args:
        strategy_selector: Optional custom StrategySelector

    Returns:
        DashboardConstructor instance
    """
    return DashboardConstructor(strategy_selector=strategy_selector)
