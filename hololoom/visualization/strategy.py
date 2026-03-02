#!/usr/bin/env python3
"""
StrategySelector - Intelligent Dashboard Panel Selection
=========================================================
Implements Edward Tufte principles for dashboard generation:
- Maximize data-ink ratio
- Show causality, mechanism, explanation
- Enable micro/macro readings
- Create narrative flow

Author: Claude Code with HoloLoom architecture
Date: October 28, 2025
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re
from enum import Enum

from .dashboard import (
    PanelSpec, PanelType, PanelSize, LayoutType,
    ComplexityLevel, DashboardStrategy
)


class QueryIntent(Enum):
    """User's intent inferred from query text."""
    FACTUAL = "factual"           # "What is X?" - Simple explanation
    EXPLORATORY = "exploratory"   # "How does X work?" - Mechanism
    COMPARISON = "comparison"     # "X vs Y" - Comparative analysis
    TREND = "trend"               # "How has X changed?" - Temporal
    DEBUGGING = "debugging"       # "Why did X fail?" - Error analysis
    OPTIMIZATION = "optimization" # "How to improve X?" - Performance


@dataclass
class QueryCharacteristics:
    """Analysis of query for dashboard generation."""
    intent: QueryIntent
    has_timeline: bool
    has_graph: bool
    has_semantic_flow: bool
    complexity: ComplexityLevel
    data_richness_score: float
    word_count: int


class QueryAnalyzer:
    """
    Analyzes query text to infer user intent.

    Uses regex patterns to detect intent from query structure.
    """

    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r"\bwhat is\b",
                r"\bdefine\b",
                r"\bexplain\b",
                r"\btell me about\b",
            ],
            QueryIntent.EXPLORATORY: [
                r"\bhow (?:does|do|did)\b",
                r"\bwhy (?:does|do|did)\b",
                r"\bshow me how\b",
                r"\bwalk through\b",
            ],
            QueryIntent.COMPARISON: [
                r"\bcompare\b",
                r"\bversus\b|\bvs\.?\b",
                r"\bdifference between\b",
                r"\bbetter than\b",
                r"\bwhich (?:is|are)\b",
            ],
            QueryIntent.TREND: [
                r"\bhow (?:has|have|did) .* change",
                r"\bover time\b",
                r"\bevolution of\b",
                r"\btrend\b",
                r"\bhistory of\b",
            ],
            QueryIntent.DEBUGGING: [
                r"\bwhy (?:did|is) .* (?:fail|error|wrong|broken)\b",
                r"\bwhat went wrong\b",
                r"\berror\b",
                r"\bfailed\b",
                r"\bdebug\b",
            ],
            QueryIntent.OPTIMIZATION: [
                r"\bhow to (?:improve|optimize|speed up|fix)\b",
                r"\bmake .* faster\b",
                r"\bperformance\b",
                r"\boptimize\b",
            ],
        }

    def detect_intent(self, query_text: str) -> QueryIntent:
        """
        Detect user intent from query text.

        Args:
            query_text: User query

        Returns:
            QueryIntent enum value
        """
        query_lower = query_text.lower()

        # Check patterns in priority order
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        # Default to factual if no pattern matches
        return QueryIntent.FACTUAL


@dataclass
class UserPreferences:
    """
    User's dashboard preferences for customization.

    Allows users to customize dashboard generation without
    changing core logic. Persists in localStorage on frontend.

    Fields:
        preferred_panels: Panel types to prioritize
        hidden_panels: Panel types to never show
        layout_preference: Preferred layout (METRIC/FLOW/RESEARCH)
        color_scheme: 'light' or 'dark' theme
        detail_level: 'minimal', 'standard', or 'detailed'
        max_panels: Maximum panels to show (overrides complexity)
        enable_animations: Enable panel animations/transitions
        auto_expand_errors: Auto-expand error panels
        panel_sizes: Custom size overrides per panel type
    """
    preferred_panels: List[PanelType] = None
    hidden_panels: List[PanelType] = None
    layout_preference: Optional[LayoutType] = None
    color_scheme: str = 'light'
    detail_level: str = 'standard'  # 'minimal', 'standard', 'detailed'
    max_panels: Optional[int] = None  # Override complexity-based limits
    enable_animations: bool = True
    auto_expand_errors: bool = True
    panel_sizes: Dict[PanelType, PanelSize] = None  # Custom panel sizes

    def __post_init__(self):
        if self.preferred_panels is None:
            self.preferred_panels = []
        if self.hidden_panels is None:
            self.hidden_panels = []
        if self.panel_sizes is None:
            self.panel_sizes = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (localStorage)."""
        return {
            'preferred_panels': [p.value for p in self.preferred_panels],
            'hidden_panels': [p.value for p in self.hidden_panels],
            'layout_preference': self.layout_preference.value if self.layout_preference else None,
            'color_scheme': self.color_scheme,
            'detail_level': self.detail_level,
            'max_panels': self.max_panels,
            'enable_animations': self.enable_animations,
            'auto_expand_errors': self.auto_expand_errors,
            'panel_sizes': {k.value: v.value for k, v in self.panel_sizes.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create from dictionary (loaded from localStorage)."""
        return cls(
            preferred_panels=[PanelType(p) for p in data.get('preferred_panels', [])],
            hidden_panels=[PanelType(p) for p in data.get('hidden_panels', [])],
            layout_preference=LayoutType(data['layout_preference']) if data.get('layout_preference') else None,
            color_scheme=data.get('color_scheme', 'light'),
            detail_level=data.get('detail_level', 'standard'),
            max_panels=data.get('max_panels'),
            enable_animations=data.get('enable_animations', True),
            auto_expand_errors=data.get('auto_expand_errors', True),
            panel_sizes={PanelType(k): PanelSize(v) for k, v in data.get('panel_sizes', {}).items()}
        )


class StrategySelector:
    """
    Intelligently selects dashboard panels based on query characteristics.

    Implements Edward Tufte principles:
    - Maximize data-ink ratio (show only relevant panels)
    - Show causality (add explanation panels for complex queries)
    - Enable micro/macro readings (combine detail + overview)
    - Create narrative flow (arrange panels to tell a story)

    Usage:
        selector = StrategySelector()
        strategy = selector.select(spacetime)
        # strategy.panels contains optimal panel specifications
    """

    def __init__(self, user_prefs: Optional[UserPreferences] = None):
        """
        Initialize strategy selector.

        Args:
            user_prefs: Optional user preferences (default: None)
        """
        self.user_prefs = user_prefs or UserPreferences()
        self.query_analyzer = QueryAnalyzer()

    def select(self, spacetime) -> DashboardStrategy:
        """
        Select optimal dashboard strategy for Spacetime.

        Returns strategy with:
        - Layout type (metric/flow/research)
        - Panel specifications (what to show)
        - Title (generated from query)
        - Complexity level

        Args:
            spacetime: Spacetime artifact from weaving

        Returns:
            DashboardStrategy with panels and layout
        """
        # 1. Analyze query
        characteristics = self.analyze_query(spacetime)

        # 2. Generate candidate panels
        candidates = self.generate_candidates(characteristics, spacetime)

        # 3. Apply user priorities
        candidates = self.apply_user_prefs(candidates)

        # 4. Choose layout
        layout = self.optimize_layout(candidates)

        # 5. Arrange for narrative
        panels = self.arrange_narrative(candidates, characteristics.intent)

        # 6. Generate title
        title = self.generate_title(spacetime, characteristics.intent)

        return DashboardStrategy(
            layout_type=layout,
            panels=tuple(panels),
            title=title,
            complexity_level=characteristics.complexity
        )

    def analyze_query(self, spacetime) -> QueryCharacteristics:
        """
        Analyze Spacetime to determine visualization strategy.

        Args:
            spacetime: Spacetime artifact

        Returns:
            QueryCharacteristics with intent and data flags
        """
        # Detect intent from query text
        intent = self.query_analyzer.detect_intent(spacetime.query_text)

        # Check for temporal data
        has_timeline = (
            hasattr(spacetime, 'trace') and
            hasattr(spacetime.trace, 'stage_durations') and
            bool(spacetime.trace.stage_durations)
        )

        # Check for graph data
        has_graph = (
            hasattr(spacetime, 'trace') and
            hasattr(spacetime.trace, 'threads_activated') and
            len(spacetime.trace.threads_activated) > 0
        )

        # Check for semantic flow
        has_semantic_flow = (
            hasattr(spacetime, 'metadata') and
            spacetime.metadata.get('semantic_cache', {}).get('enabled', False)
        )

        # Determine complexity
        complexity = spacetime.complexity if hasattr(spacetime, 'complexity') else ComplexityLevel.FAST

        # Calculate data richness (0-1)
        richness_factors = [
            has_timeline,
            has_graph,
            has_semantic_flow,
            spacetime.confidence > 0.7,
            len(spacetime.query_text.split()) > 10
        ]
        data_richness_score = sum(richness_factors) / len(richness_factors)

        return QueryCharacteristics(
            intent=intent,
            has_timeline=has_timeline,
            has_graph=has_graph,
            has_semantic_flow=has_semantic_flow,
            complexity=complexity,
            data_richness_score=data_richness_score,
            word_count=len(spacetime.query_text.split())
        )

    def generate_candidates(
        self,
        characteristics: QueryCharacteristics,
        spacetime
    ) -> List[PanelSpec]:
        """
        Generate candidate panels based on query intent.

        Tufte principle: Maximize data-ink ratio by showing only relevant panels.

        Args:
            characteristics: Query analysis
            spacetime: Spacetime artifact

        Returns:
            List of candidate panel specifications
        """
        candidates = []

        # ALWAYS show core metrics (Tufte: essential data)
        candidates.append(PanelSpec(
            type=PanelType.METRIC,
            data_source='confidence',
            size=PanelSize.SMALL,
            priority=10,
            title='Confidence',
            subtitle=f'{spacetime.confidence:.0%}'
        ))

        candidates.append(PanelSpec(
            type=PanelType.METRIC,
            data_source='trace.duration_ms',
            size=PanelSize.SMALL,
            priority=9,
            title='Duration',
            subtitle=f'{spacetime.trace.duration_ms:.0f}ms'
        ))

        # Intent-specific panels
        if characteristics.intent == QueryIntent.EXPLORATORY:
            # Show mechanism (Tufte: show causality)
            candidates.extend([
                PanelSpec(
                    PanelType.TIMELINE,
                    'trace.stage_durations',
                    PanelSize.FULL_WIDTH,
                    9,
                    'Execution Timeline',
                    'How the query was processed'
                ),
                PanelSpec(
                    PanelType.NETWORK,
                    'trace.threads_activated',
                    PanelSize.MEDIUM,
                    7,
                    'Knowledge Threads',
                    'Memory connections activated'
                )
            ])

        elif characteristics.intent == QueryIntent.DEBUGGING:
            # Show errors and causality
            if spacetime.trace.errors:
                candidates.append(PanelSpec(
                    PanelType.TEXT,
                    'trace.errors',
                    PanelSize.FULL_WIDTH,
                    10,
                    'Error Trace',
                    'What went wrong'
                ))

            candidates.append(PanelSpec(
                PanelType.TIMELINE,
                'trace.stage_durations',
                PanelSize.LARGE,
                9,
                'Execution Breakdown',
                'Identify bottlenecks'
            ))

        elif characteristics.intent == QueryIntent.COMPARISON:
            # Show side-by-side comparisons (small multiples)
            candidates.append(PanelSpec(
                PanelType.HEATMAP,
                'comparison_matrix',
                PanelSize.LARGE,
                8,
                'Comparison Matrix',
                'Feature comparison'
            ))

        elif characteristics.intent == QueryIntent.OPTIMIZATION:
            # Show performance metrics
            candidates.extend([
                PanelSpec(
                    PanelType.TIMELINE,
                    'trace.stage_durations',
                    PanelSize.FULL_WIDTH,
                    9,
                    'Performance Profile',
                    'Time spent per stage'
                ),
                PanelSpec(
                    PanelType.METRIC,
                    'bottleneck',
                    PanelSize.SMALL,
                    8,
                    'Bottleneck',
                    'Slowest stage'
                )
            ])

        # Add timeline if temporal data exists (not already added)
        if characteristics.has_timeline and not any(
            p.type == PanelType.TIMELINE for p in candidates
        ):
            candidates.append(PanelSpec(
                PanelType.TIMELINE,
                'trace.stage_durations',
                PanelSize.LARGE,
                8,
                'Timeline',
                'Execution stages'
            ))

        # Add semantic flow if available
        if characteristics.has_semantic_flow:
            candidates.append(PanelSpec(
                PanelType.HEATMAP,
                'semantic_dimensions',
                PanelSize.MEDIUM,
                6,
                'Semantic Profile',
                '244D meaning projection'
            ))

        # Add query/response text (low priority, but useful for context)
        candidates.append(PanelSpec(
            PanelType.TEXT,
            'query_text',
            PanelSize.SMALL,
            5,
            'Query',
            spacetime.query_text[:50] + '...' if len(spacetime.query_text) > 50 else spacetime.query_text
        ))

        return candidates

    def apply_user_prefs(self, candidates: List[PanelSpec]) -> List[PanelSpec]:
        """
        Apply user preferences to filter/prioritize/customize panels.

        Applies:
        - Hidden panels filter
        - Preferred panels priority boost
        - Custom panel sizes
        - Detail level adjustments
        - Max panels limit

        Args:
            candidates: List of candidate panels

        Returns:
            Filtered and customized list based on preferences
        """
        if not self.user_prefs:
            return candidates

        # 1. Filter out hidden panels
        if self.user_prefs.hidden_panels:
            candidates = [
                p for p in candidates
                if p.type not in self.user_prefs.hidden_panels
            ]

        # 2. Boost priority of preferred panels
        if self.user_prefs.preferred_panels:
            for panel in candidates:
                if panel.type in self.user_prefs.preferred_panels:
                    panel.priority += 20  # Significant boost

        # 3. Apply custom panel sizes
        if self.user_prefs.panel_sizes:
            for panel in candidates:
                if panel.type in self.user_prefs.panel_sizes:
                    panel.size = self.user_prefs.panel_sizes[panel.type]

        # 4. Adjust for detail level
        if self.user_prefs.detail_level == 'minimal':
            # Remove lower-priority panels in minimal mode
            candidates = [p for p in candidates if p.priority >= 70]
        elif self.user_prefs.detail_level == 'detailed':
            # Include more panels in detailed mode (lower threshold)
            pass  # Keep all candidates

        # 5. Apply max panels limit
        if self.user_prefs.max_panels:
            # Sort by priority and take top N
            candidates = sorted(candidates, key=lambda p: p.priority, reverse=True)
            candidates = candidates[:self.user_prefs.max_panels]

        return candidates

    def optimize_layout(self, panels: List[PanelSpec]) -> LayoutType:
        """
        Choose layout based on panel count.

        Tufte principle: Layering and separation for clarity.

        Args:
            panels: List of panel specs

        Returns:
            Optimal layout type
        """
        panel_count = len(panels)

        if panel_count <= 3:
            return LayoutType.METRIC  # Single column (1×3 grid)
        elif panel_count <= 8:
            return LayoutType.FLOW  # Two columns (2×4 grid)
        else:
            return LayoutType.RESEARCH  # Three columns (3×n grid)

    def arrange_narrative(
        self,
        panels: List[PanelSpec],
        intent: QueryIntent
    ) -> List[PanelSpec]:
        """
        Arrange panels for narrative flow.

        Tufte principle: Narrative graphics - guide discovery.

        Story structure: Hook → Context → Mechanism → Conclusion

        Args:
            panels: List of panel specs
            intent: Query intent

        Returns:
            Panels arranged in narrative order
        """
        # Separate panels by narrative role
        hooks = []      # Surprising/important findings
        context = []    # Background information
        mechanism = []  # How/why (causality)
        conclusion = [] # Recommendations

        for panel in panels:
            if panel.priority >= 9:
                hooks.append(panel)  # High priority = hook
            elif panel.type in [PanelType.TEXT, PanelType.METRIC]:
                context.append(panel)  # Context panels
            elif panel.type in [PanelType.TIMELINE, PanelType.NETWORK]:
                mechanism.append(panel)  # Mechanism panels
            else:
                conclusion.append(panel)  # Everything else

        # Arrange: Hook → Context → Mechanism → Conclusion
        return hooks + context + mechanism + conclusion

    def generate_title(self, spacetime, intent: QueryIntent) -> str:
        """
        Generate dashboard title based on query and intent.

        Args:
            spacetime: Spacetime artifact
            intent: Query intent

        Returns:
            Human-readable title
        """
        query = spacetime.query_text

        # Truncate long queries
        if len(query) > 60:
            query = query[:57] + "..."

        # Add intent-based context
        intent_prefixes = {
            QueryIntent.FACTUAL: "Answer:",
            QueryIntent.EXPLORATORY: "Exploring:",
            QueryIntent.COMPARISON: "Comparing:",
            QueryIntent.TREND: "Trend Analysis:",
            QueryIntent.DEBUGGING: "Debugging:",
            QueryIntent.OPTIMIZATION: "Optimizing:",
        }

        prefix = intent_prefixes.get(intent, "Query:")
        return f"{prefix} {query}"


# Convenience function
def create_strategy_selector(user_prefs: Optional[UserPreferences] = None) -> StrategySelector:
    """
    Create a StrategySelector instance.

    Args:
        user_prefs: Optional user preferences

    Returns:
        StrategySelector instance
    """
    return StrategySelector(user_prefs=user_prefs)
