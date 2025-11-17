"""
HoloLoom Memory Inspector - Complete Retrieval Transparency
==========================================================

Visualizes WHY each memory was retrieved with complete scoring breakdown.

Features:
- Retrieval path visualization (knowledge graph)
- Scoring breakdown (BM25, semantic, temporal, graph proximity)
- Thompson bandit statistics
- Cache hit/miss indicators
- Confidence calibration
- Complete audit trail

Tufte Principles:
- Maximize data-ink ratio (dense information display)
- Direct labeling (inline explanations)
- Small multiples (compare multiple retrievals)
- Meaning first (critical factors highlighted)

Integration:
- Works with HoloLoom.recall() results
- Accepts Spacetime fabric with traces
- Programmatic API for automated inspection

Author: HoloLoom Team
Date: 2025-11-17
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import math


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RetrievalScore:
    """Breakdown of why a memory was retrieved."""
    memory_id: str
    memory_text: str
    total_score: float

    # Component scores (0.0-1.0)
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    graph_proximity_score: float = 0.0
    recency_score: float = 0.0

    # Metadata
    cache_hit: bool = False
    retrieval_path: List[str] = field(default_factory=list)
    timestamp: Optional[float] = None
    access_count: int = 0

    # Weighted contribution to final score
    component_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class InspectionResult:
    """Complete inspection of a recall operation."""
    query: str
    retrieved_memories: List[RetrievalScore]

    # System state
    total_memories_searched: int = 0
    cache_hit_rate: float = 0.0
    avg_retrieval_time_ms: float = 0.0

    # Thompson bandit state (if available)
    bandit_stats: Optional[Dict[str, Any]] = None

    # Knowledge graph context
    knowledge_graph_nodes: List[str] = field(default_factory=list)
    knowledge_graph_edges: List[Tuple[str, str, str]] = field(default_factory=list)


class ScoreComponent(Enum):
    """Components that contribute to retrieval score."""
    BM25 = "bm25"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    GRAPH_PROXIMITY = "graph_proximity"
    RECENCY = "recency"


# ============================================================================
# Scoring Visualization
# ============================================================================

def _render_score_breakdown_table(scores: List[RetrievalScore], top_n: int = 10) -> str:
    """
    Render dense table showing score components for each retrieved memory.

    Tufte-style: Maximum data density, minimal decoration.
    """
    # Take top N results
    top_scores = sorted(scores, key=lambda s: s.total_score, reverse=True)[:top_n]

    html = [
        '<div class="score-breakdown-table">',
        '<table>',
        '<thead>',
        '<tr>',
        '<th>Rank</th>',
        '<th>Memory</th>',
        '<th>Total</th>',
        '<th>BM25</th>',
        '<th>Semantic</th>',
        '<th>Temporal</th>',
        '<th>Graph</th>',
        '<th>Recency</th>',
        '<th>Cache</th>',
        '</tr>',
        '</thead>',
        '<tbody>'
    ]

    for i, score in enumerate(top_scores, 1):
        # Highlight top result
        row_class = 'top-result' if i == 1 else ''

        # Cache indicator
        cache_icon = '✓' if score.cache_hit else '✗'
        cache_class = 'cache-hit' if score.cache_hit else 'cache-miss'

        # Truncate memory text
        text = score.memory_text[:60] + '...' if len(score.memory_text) > 60 else score.memory_text

        html.append(f'<tr class="{row_class}">')
        html.append(f'<td class="rank">{i}</td>')
        html.append(f'<td class="memory-text" title="{score.memory_text}">{text}</td>')
        html.append(f'<td class="total-score"><strong>{score.total_score:.3f}</strong></td>')
        html.append(f'<td class="component-score">{score.bm25_score:.3f}</td>')
        html.append(f'<td class="component-score">{score.semantic_score:.3f}</td>')
        html.append(f'<td class="component-score">{score.temporal_score:.3f}</td>')
        html.append(f'<td class="component-score">{score.graph_proximity_score:.3f}</td>')
        html.append(f'<td class="component-score">{score.recency_score:.3f}</td>')
        html.append(f'<td class="cache-indicator {cache_class}">{cache_icon}</td>')
        html.append('</tr>')

    html.append('</tbody>')
    html.append('</table>')
    html.append('</div>')

    return '\n'.join(html)


def _render_component_weights_chart(scores: List[RetrievalScore]) -> str:
    """
    Visualize how much each component contributed to final scores.

    Shows stacked bar chart of component contributions.
    """
    if not scores or not scores[0].component_weights:
        return '<p class="no-data">Component weights not available</p>'

    # Calculate average contribution of each component
    components = ['bm25', 'semantic', 'temporal', 'graph_proximity', 'recency']
    avg_weights = {comp: 0.0 for comp in components}

    for score in scores:
        for comp in components:
            avg_weights[comp] += score.component_weights.get(comp, 0.0)

    count = len(scores)
    for comp in components:
        avg_weights[comp] /= count if count > 0 else 1

    # Normalize to percentages
    total = sum(avg_weights.values())
    if total > 0:
        avg_weights = {k: (v/total)*100 for k, v in avg_weights.items()}

    html = [
        '<div class="component-weights-chart">',
        '<h3>Score Component Contributions</h3>',
        '<div class="stacked-bar">'
    ]

    # Color mapping
    colors = {
        'bm25': '#3498db',          # Blue
        'semantic': '#2ecc71',      # Green
        'temporal': '#e74c3c',      # Red
        'graph_proximity': '#9b59b6', # Purple
        'recency': '#f39c12'        # Orange
    }

    for comp in components:
        width = avg_weights[comp]
        color = colors.get(comp, '#95a5a6')
        label = comp.replace('_', ' ').title()

        html.append(
            f'<div class="bar-segment" style="width: {width:.1f}%; background-color: {color};" '
            f'title="{label}: {width:.1f}%">'
        )
        if width > 10:  # Only show label if wide enough
            html.append(f'<span>{label} {width:.0f}%</span>')
        html.append('</div>')

    html.append('</div>')
    html.append('</div>')

    return '\n'.join(html)


def _render_bandit_stats(bandit_stats: Optional[Dict[str, Any]]) -> str:
    """Render Thompson Sampling bandit statistics."""
    if not bandit_stats:
        return '<p class="no-data">Bandit statistics not available</p>'

    html = [
        '<div class="bandit-stats">',
        '<h3>Thompson Sampling Statistics</h3>',
        '<table>',
        '<thead>',
        '<tr><th>Tool</th><th>Alpha (α)</th><th>Beta (β)</th><th>Expected Reward</th><th>Selections</th></tr>',
        '</thead>',
        '<tbody>'
    ]

    # Sort by expected reward
    tools = sorted(
        bandit_stats.items(),
        key=lambda x: x[1].get('expected_reward', 0),
        reverse=True
    )

    for tool_name, stats in tools:
        alpha = stats.get('alpha', 1.0)
        beta = stats.get('beta', 1.0)
        expected = stats.get('expected_reward', alpha / (alpha + beta))
        selections = stats.get('selections', 0)

        html.append('<tr>')
        html.append(f'<td>{tool_name}</td>')
        html.append(f'<td>{alpha:.2f}</td>')
        html.append(f'<td>{beta:.2f}</td>')
        html.append(f'<td><strong>{expected:.3f}</strong></td>')
        html.append(f'<td>{selections}</td>')
        html.append('</tr>')

    html.append('</tbody>')
    html.append('</table>')
    html.append('</div>')

    return '\n'.join(html)


def _render_retrieval_path_graph(
    scores: List[RetrievalScore],
    graph_nodes: List[str],
    graph_edges: List[Tuple[str, str, str]]
) -> str:
    """
    Render knowledge graph showing retrieval paths.

    Uses existing knowledge_graph.py visualization.
    """
    # Import here to avoid circular dependency
    try:
        from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_edges
    except ImportError:
        return '<p class="no-data">Knowledge graph visualization not available</p>'

    if not graph_nodes or not graph_edges:
        return '<p class="no-data">No graph data available for this query</p>'

    # Highlight paths for top 3 results
    highlighted_paths = []
    for score in scores[:3]:
        if score.retrieval_path:
            highlighted_paths.extend(score.retrieval_path)

    # Render using existing visualization
    return render_knowledge_graph_from_edges(
        edges=graph_edges,
        title="Retrieval Paths in Knowledge Graph",
        highlighted_nodes=set(highlighted_paths),
        width=600,
        height=400
    )


def _render_cache_performance(scores: List[RetrievalScore], cache_hit_rate: float) -> str:
    """Render cache performance metrics."""
    total = len(scores)
    hits = sum(1 for s in scores if s.cache_hit)
    misses = total - hits

    html = [
        '<div class="cache-performance">',
        '<h3>Cache Performance</h3>',
        '<div class="cache-metrics">',
        f'<div class="metric"><span class="label">Hit Rate:</span> <span class="value">{cache_hit_rate:.1%}</span></div>',
        f'<div class="metric"><span class="label">Hits:</span> <span class="value cache-hit">{hits}</span></div>',
        f'<div class="metric"><span class="label">Misses:</span> <span class="value cache-miss">{misses}</span></div>',
        '</div>',
        '</div>'
    ]

    return '\n'.join(html)


# ============================================================================
# Main Rendering Function
# ============================================================================

def render_memory_inspector(
    inspection: InspectionResult,
    title: str = "Memory Inspector",
    subtitle: str = "",
    show_graph: bool = True,
    show_bandit_stats: bool = True,
    top_n: int = 10
) -> str:
    """
    Render complete memory inspection visualization.

    Args:
        inspection: InspectionResult with retrieval data
        title: Main title
        subtitle: Optional subtitle
        show_graph: Whether to show knowledge graph
        show_bandit_stats: Whether to show Thompson bandit stats
        top_n: Number of top results to show

    Returns:
        HTML string with complete visualization

    Example:
        >>> from HoloLoom.visualization.memory_inspector import render_memory_inspector
        >>> from HoloLoom import HoloLoom
        >>>
        >>> loom = HoloLoom()
        >>> memories = await loom.recall("What is Thompson Sampling?")
        >>>
        >>> # Create inspection result from memories
        >>> scores = [
        ...     RetrievalScore(
        ...         memory_id=m.id,
        ...         memory_text=m.content,
        ...         total_score=m.score,
        ...         semantic_score=0.92,
        ...         cache_hit=True
        ...     ) for m in memories
        ... ]
        >>>
        >>> inspection = InspectionResult(
        ...     query="What is Thompson Sampling?",
        ...     retrieved_memories=scores
        ... )
        >>>
        >>> html = render_memory_inspector(inspection)
        >>> with open('inspector.html', 'w') as f:
        ...     f.write(html)
    """

    html = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<meta charset="utf-8">',
        '<title>Memory Inspector</title>',
        _render_styles(),
        '</head>',
        '<body>',
        '<div class="inspector-container">',

        # Header
        f'<h1>{title}</h1>',
        f'<p class="subtitle">{subtitle}</p>' if subtitle else '',
        f'<p class="query"><strong>Query:</strong> {inspection.query}</p>',

        # Summary metrics
        '<div class="summary-metrics">',
        f'<div class="metric"><span class="label">Retrieved:</span> <span class="value">{len(inspection.retrieved_memories)}</span></div>',
        f'<div class="metric"><span class="label">Searched:</span> <span class="value">{inspection.total_memories_searched}</span></div>',
        f'<div class="metric"><span class="label">Avg Time:</span> <span class="value">{inspection.avg_retrieval_time_ms:.1f}ms</span></div>',
        '</div>',

        # Score breakdown table
        '<h2>Score Breakdown</h2>',
        _render_score_breakdown_table(inspection.retrieved_memories, top_n=top_n),

        # Component weights
        _render_component_weights_chart(inspection.retrieved_memories),

        # Cache performance
        _render_cache_performance(inspection.retrieved_memories, inspection.cache_hit_rate),

        # Thompson bandit stats (optional)
        '<h2>Thompson Sampling Statistics</h2>' if show_bandit_stats else '',
        _render_bandit_stats(inspection.bandit_stats) if show_bandit_stats else '',

        # Knowledge graph (optional)
        '<h2>Retrieval Paths</h2>' if show_graph else '',
        _render_retrieval_path_graph(
            inspection.retrieved_memories,
            inspection.knowledge_graph_nodes,
            inspection.knowledge_graph_edges
        ) if show_graph else '',

        '</div>',
        '</body>',
        '</html>'
    ]

    return '\n'.join(html)


def _render_styles() -> str:
    """CSS styles for memory inspector (Tufte-inspired)."""
    return """
    <style>
        body {
            font-family: "Gill Sans", "Gill Sans MT", Calibri, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fefefe;
        }

        .inspector-container {
            background: white;
            padding: 30px;
            border: 1px solid #ddd;
        }

        h1 {
            font-size: 24px;
            font-weight: normal;
            margin: 0 0 5px 0;
            letter-spacing: -0.5px;
        }

        h2 {
            font-size: 18px;
            font-weight: normal;
            margin: 30px 0 10px 0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }

        h3 {
            font-size: 14px;
            font-weight: bold;
            margin: 15px 0 8px 0;
        }

        .subtitle {
            font-size: 12px;
            color: #666;
            margin: 0 0 15px 0;
        }

        .query {
            font-size: 14px;
            background: #f5f5f5;
            padding: 10px;
            border-left: 3px solid #3498db;
            margin: 15px 0;
        }

        .summary-metrics {
            display: flex;
            gap: 30px;
            margin: 20px 0;
            font-size: 13px;
        }

        .metric {
            display: flex;
            gap: 8px;
        }

        .metric .label {
            color: #666;
        }

        .metric .value {
            font-weight: bold;
        }

        /* Score breakdown table */
        .score-breakdown-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin: 15px 0;
        }

        .score-breakdown-table th {
            text-align: left;
            font-weight: normal;
            padding: 6px 8px;
            border-bottom: 2px solid #333;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .score-breakdown-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
        }

        .score-breakdown-table tr.top-result {
            background: #fffacd;
        }

        .rank {
            width: 40px;
            text-align: center;
            color: #666;
        }

        .memory-text {
            max-width: 400px;
            font-size: 11px;
        }

        .total-score {
            font-family: "Courier New", monospace;
            text-align: right;
        }

        .component-score {
            font-family: "Courier New", monospace;
            font-size: 11px;
            text-align: right;
            color: #666;
        }

        .cache-indicator {
            text-align: center;
            font-size: 14px;
        }

        .cache-hit {
            color: #2ecc71;
        }

        .cache-miss {
            color: #e74c3c;
        }

        /* Component weights chart */
        .component-weights-chart {
            margin: 20px 0;
        }

        .stacked-bar {
            display: flex;
            width: 100%;
            height: 40px;
            margin: 10px 0;
            border: 1px solid #ddd;
        }

        .bar-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 11px;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }

        /* Cache performance */
        .cache-performance {
            margin: 20px 0;
        }

        .cache-metrics {
            display: flex;
            gap: 30px;
            margin: 10px 0;
        }

        /* Bandit stats table */
        .bandit-stats table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin: 15px 0;
        }

        .bandit-stats th {
            text-align: left;
            font-weight: normal;
            padding: 6px 8px;
            border-bottom: 2px solid #333;
            font-size: 11px;
        }

        .bandit-stats td {
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
            font-family: "Courier New", monospace;
            font-size: 11px;
        }

        .no-data {
            color: #999;
            font-style: italic;
            font-size: 12px;
        }
    </style>
    """


# ============================================================================
# Helper: Create InspectionResult from HoloLoom recall
# ============================================================================

def create_inspection_from_recall(
    query: str,
    memories: List[Any],
    metadata: Optional[Dict[str, Any]] = None
) -> InspectionResult:
    """
    Create InspectionResult from HoloLoom recall() output.

    Args:
        query: Original query string
        memories: List of Memory objects from recall()
        metadata: Optional metadata (cache stats, bandit stats, etc.)

    Returns:
        InspectionResult ready for visualization

    Example:
        >>> from HoloLoom import HoloLoom
        >>> from HoloLoom.visualization.memory_inspector import create_inspection_from_recall
        >>>
        >>> loom = HoloLoom()
        >>> memories = await loom.recall("What is Thompson Sampling?")
        >>>
        >>> inspection = create_inspection_from_recall(
        ...     query="What is Thompson Sampling?",
        ...     memories=memories
        ... )
        >>>
        >>> html = render_memory_inspector(inspection)
    """
    metadata = metadata or {}

    # Convert memories to scores
    scores = []
    for mem in memories:
        score = RetrievalScore(
            memory_id=getattr(mem, 'id', str(hash(mem))),
            memory_text=getattr(mem, 'content', str(mem)),
            total_score=getattr(mem, 'score', 0.0),
            semantic_score=metadata.get('semantic_scores', {}).get(str(mem), 0.0),
            cache_hit=metadata.get('cache_hits', {}).get(str(mem), False)
        )
        scores.append(score)

    return InspectionResult(
        query=query,
        retrieved_memories=scores,
        total_memories_searched=metadata.get('total_searched', len(memories)),
        cache_hit_rate=metadata.get('cache_hit_rate', 0.0),
        avg_retrieval_time_ms=metadata.get('avg_time_ms', 0.0),
        bandit_stats=metadata.get('bandit_stats')
    )
