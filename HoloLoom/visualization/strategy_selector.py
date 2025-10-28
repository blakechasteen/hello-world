"""
Strategy Selector - Intelligent Dashboard Generation
=====================================================
Implements Edward Tufte principles for data visualization.
"""

import re
from dataclasses import dataclass
from typing import List
from enum import Enum

from .dashboard import (
    PanelSpec, PanelType, PanelSize, LayoutType,
    ComplexityLevel, DashboardStrategy, SpacetimeLike
)


class QueryIntent(str, Enum):
    """User intent inferred from query."""
    FACTUAL = "factual"
    EXPLORATORY = "exploratory"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"


@dataclass
class QueryCharacteristics:
    """Analysis of query and data."""
    intent: QueryIntent
    complexity_level: ComplexityLevel
    has_timeline: bool
    has_graph_data: bool
    has_errors: bool
    richness_score: float


class QueryAnalyzer:
    """Analyze query to infer intent."""

    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL: [r"what is", r"define"],
            QueryIntent.DEBUGGING: [r"why.*fail", r"error"],
            QueryIntent.OPTIMIZATION: [r"optimize", r"speed up"],
            QueryIntent.EXPLORATORY: [r"how", r"why"],
        }

    def analyze(self, spacetime: SpacetimeLike) -> QueryCharacteristics:
        """Analyze query and spacetime."""
        intent = self._detect_intent(spacetime.query_text)
        complexity = ComplexityLevel(spacetime.metadata.get("complexity", "FAST"))
        
        return QueryCharacteristics(
            intent=intent,
            complexity_level=complexity,
            has_timeline=self._has_timeline(spacetime),
            has_graph_data=self._has_graph(spacetime),
            has_errors=self._has_errors(spacetime),
            richness_score=self._calc_richness(spacetime)
        )

    def _detect_intent(self, query_text: str) -> QueryIntent:
        query_lower = query_text.lower()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        return QueryIntent.FACTUAL

    def _has_timeline(self, spacetime: SpacetimeLike) -> bool:
        if not hasattr(spacetime, "trace"):
            return False
        return bool(getattr(spacetime.trace, "stage_durations", {}))

    def _has_graph(self, spacetime: SpacetimeLike) -> bool:
        if not hasattr(spacetime, "trace"):
            return False
        return len(getattr(spacetime.trace, "threads_activated", [])) > 2

    def _has_errors(self, spacetime: SpacetimeLike) -> bool:
        if not hasattr(spacetime, "trace"):
            return False
        return bool(getattr(spacetime.trace, "errors", []))

    def _calc_richness(self, spacetime: SpacetimeLike) -> float:
        score = 0.0
        if self._has_timeline(spacetime):
            score += 0.33
        if self._has_graph(spacetime):
            score += 0.33
        if hasattr(spacetime, "metadata") and len(spacetime.metadata) > 2:
            score += 0.34
        return min(score, 1.0)


class StrategySelector:
    """Intelligent dashboard strategy selection."""

    def __init__(self):
        self.analyzer = QueryAnalyzer()

    def select(self, spacetime: SpacetimeLike) -> DashboardStrategy:
        """Select optimal dashboard strategy."""
        characteristics = self.analyzer.analyze(spacetime)
        candidates = self._generate_candidates(characteristics, spacetime)
        prioritized = sorted(candidates, key=lambda p: p.priority, reverse=True)
        selected = self._select_top_panels(prioritized, characteristics.complexity_level)
        narrative = self._arrange_narrative(selected)
        layout = self._select_layout(narrative)
        title = self._generate_title(spacetime, characteristics)

        return DashboardStrategy(
            layout_type=layout,
            panels=tuple(narrative),
            title=title,
            complexity_level=characteristics.complexity_level
        )

    def _generate_candidates(self, characteristics: QueryCharacteristics, spacetime: SpacetimeLike) -> List[PanelSpec]:
        """Generate candidate panels."""
        candidates = [PanelSpec(PanelType.METRIC, "confidence", PanelSize.SMALL, 100, "Confidence")]
        
        if characteristics.intent == QueryIntent.FACTUAL:
            candidates.append(PanelSpec(PanelType.TEXT, "response", PanelSize.LARGE, 90, "Answer"))
        
        elif characteristics.intent == QueryIntent.EXPLORATORY:
            if characteristics.has_timeline:
                candidates.append(PanelSpec(PanelType.TIMELINE, "trace.stage_durations", PanelSize.FULL_WIDTH, 85, "Timeline"))
            if characteristics.has_graph_data:
                candidates.append(PanelSpec(PanelType.NETWORK, "trace.threads_activated", PanelSize.MEDIUM, 80, "Graph"))
        
        elif characteristics.intent == QueryIntent.DEBUGGING:
            if characteristics.has_errors:
                candidates.append(PanelSpec(PanelType.TEXT, "trace.errors", PanelSize.FULL_WIDTH, 95, "Errors"))
            if characteristics.has_timeline:
                candidates.append(PanelSpec(PanelType.TIMELINE, "trace.stage_durations", PanelSize.LARGE, 88, "Timeline"))
        
        elif characteristics.intent == QueryIntent.OPTIMIZATION:
            if characteristics.has_timeline:
                candidates.append(PanelSpec(PanelType.TIMELINE, "trace.stage_durations", PanelSize.FULL_WIDTH, 92, "Bottlenecks"))
        
        return candidates

    def _select_top_panels(self, prioritized: List[PanelSpec], complexity: ComplexityLevel) -> List[PanelSpec]:
        max_panels = {ComplexityLevel.LITE: 3, ComplexityLevel.FAST: 6, ComplexityLevel.FULL: 12, ComplexityLevel.RESEARCH: 999}
        return prioritized[:max_panels.get(complexity, 6)]

    def _arrange_narrative(self, panels: List[PanelSpec]) -> List[PanelSpec]:
        """Arrange for narrative flow."""
        metrics = [p for p in panels if p.type == PanelType.METRIC]
        text = [p for p in panels if p.type == PanelType.TEXT]
        timelines = [p for p in panels if p.type == PanelType.TIMELINE]
        networks = [p for p in panels if p.type == PanelType.NETWORK]
        return metrics + text + timelines + networks

    def _select_layout(self, panels: List[PanelSpec]) -> LayoutType:
        panel_count = len(panels)
        if panel_count <= 3:
            return LayoutType.METRIC
        elif panel_count <= 8:
            return LayoutType.FLOW
        else:
            return LayoutType.RESEARCH

    def _generate_title(self, spacetime: SpacetimeLike, characteristics: QueryCharacteristics) -> str:
        intent_titles = {
            QueryIntent.FACTUAL: "Query Response",
            QueryIntent.EXPLORATORY: "Exploratory Analysis",
            QueryIntent.DEBUGGING: "Debug Report",
            QueryIntent.OPTIMIZATION: "Performance Analysis"
        }
        base = intent_titles.get(characteristics.intent, "Dashboard")
        query_preview = spacetime.query_text[:50]
        if len(spacetime.query_text) > 50:
            query_preview += "..."
        return f"{base}: {query_preview}"
