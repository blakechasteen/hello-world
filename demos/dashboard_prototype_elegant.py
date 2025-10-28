#!/usr/bin/env python3
"""
Self-Constructing Dashboard - Elegant Implementation
=====================================================
Ruthlessly refactored for clarity, modularity, and testability.

Key improvements:
- Composable component methods
- Class-based with dependency injection
- Full type hints
- Clear separation of concerns
- Zero repetition
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants
STAGE_COLORS = {
    'features': '#6366f1',
    'retrieval': '#10b981',
    'decision': '#f59e0b',
    'execution': '#ef4444'
}

METRIC_COLORS = {
    'confidence': 'green',
    'duration': 'blue',
    'tool': 'purple',
    'threads': 'indigo'
}


@dataclass
class MockTrace:
    """Mock trace for testing."""
    start_time: datetime
    end_time: datetime
    duration_ms: float
    stage_durations: Dict[str, float] = field(default_factory=dict)
    threads_activated: List[str] = field(default_factory=list)


@dataclass
class MockSpacetime:
    """Mock Spacetime for testing."""
    query_text: str
    response: str
    tool_used: str
    confidence: float
    trace: MockTrace
    metadata: Dict[str, any] = field(default_factory=dict)


class DashboardGenerator:
    """
    Elegant, composable dashboard generator.

    Philosophy: Each method generates one atomic component.
    Composition creates the full dashboard.
    """

    def __init__(self, stage_colors: Dict[str, str] = None,
                 metric_colors: Dict[str, str] = None):
        """Initialize with configurable colors (dependency injection)."""
        self.stage_colors = stage_colors or STAGE_COLORS
        self.metric_colors = metric_colors or METRIC_COLORS

    def metric_card(self, label: str, value: str, color: str) -> str:
        """Generate single metric card component."""
        return f'''<div class="bg-white p-4 rounded-lg shadow border border-gray-200">
            <p class="text-sm text-gray-500">{label}</p>
            <p class="text-2xl font-bold text-{color}-600 mt-1">{value}</p>
        </div>'''

    def summary_cards(self, spacetime: MockSpacetime) -> str:
        """Generate all 4 summary cards (composed from metric_card)."""
        cards = [
            self.metric_card("Confidence", f"{spacetime.confidence:.2f}",
                           self.metric_colors['confidence']),
            self.metric_card("Duration", f"{spacetime.trace.duration_ms:.1f}ms",
                           self.metric_colors['duration']),
            self.metric_card("Tool", spacetime.tool_used,
                           self.metric_colors['tool']),
            self.metric_card("Threads", str(len(spacetime.trace.threads_activated)),
                           self.metric_colors['threads'])
        ]
        return '\n'.join(cards)

    def timeline_chart(self, stages: List[str], durations: List[float]) -> str:
        """Generate Plotly waterfall chart component."""
        colors = [self.stage_colors.get(s.lower(), '#6b7280') for s in stages]
        labels = [f"{d:.1f}ms" for d in durations]

        return f'''<div class="bg-white p-6 rounded-lg shadow border border-gray-200 mb-6">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Execution Timeline</h2>
            <div id="timeline-chart" style="height:300px"></div>
            <script>
            Plotly.newPlot("timeline-chart", [{{
                type: "waterfall",
                x: {json.dumps(stages)},
                y: {json.dumps(durations)},
                marker: {{color: {json.dumps(colors)}}},
                text: {json.dumps(labels)},
                textposition: "outside"
            }}], {{
                showlegend: false,
                xaxis: {{title: "Stage"}},
                yaxis: {{title: "Duration (ms)"}},
                margin: {{t: 20, r: 20, b: 60, l: 60}}
            }}, {{responsive: true}});
            </script>
        </div>'''

    def content_panel(self, title: str, content: str) -> str:
        """Generate text content panel component."""
        return f'''<div class="bg-white p-6 rounded-lg shadow border border-gray-200">
            <h3 class="text-lg font-semibold mb-3">{title}</h3>
            <p class="text-sm text-gray-700">{content}</p>
        </div>'''

    def generate(self, spacetime: MockSpacetime) -> str:
        """
        Generate complete dashboard HTML.

        This is the main entry point - composes all components.
        Clear data flow: spacetime -> components -> HTML
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        complexity = spacetime.metadata.get('complexity', 'UNKNOWN')

        stages = list(spacetime.trace.stage_durations.keys())
        durations = list(spacetime.trace.stage_durations.values())

        # Compose dashboard from components
        header = f'''<div class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 rounded-lg shadow-lg mb-8">
            <h1 class="text-3xl font-bold mb-2">Self-Constructing Dashboard</h1>
            <p class="text-indigo-100">Elegant, composable implementation</p>
            <p class="text-sm text-indigo-200 mt-2">Generated {timestamp} - Complexity: {complexity}</p>
        </div>'''

        feature_highlight = '''<div class="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
            <h3 class="text-lg font-bold text-gray-900 mb-2">Ruthlessly Elegant!</h3>
            <p class="text-sm text-gray-700 mb-4"><strong>Refactored for</strong>: Composability, testability, clarity</p>
            <div class="grid grid-cols-3 gap-4 text-sm">
                <div><strong class="text-indigo-600">Composable</strong><br/>Each method = one component</div>
                <div><strong class="text-indigo-600">Testable</strong><br/>Class-based, dependency injection</div>
                <div><strong class="text-indigo-600">Type-safe</strong><br/>Full type hints throughout</div>
            </div>
        </div>'''

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Constructing Dashboard | mythRL</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 p-8">
    {header}
    <div class="grid grid-cols-4 gap-4 mb-6">{self.summary_cards(spacetime)}</div>
    {self.timeline_chart(stages, durations)}
    <div class="grid grid-cols-2 gap-6 mb-6">
        {self.content_panel("Query", spacetime.query_text)}
        {self.content_panel("Response", spacetime.response)}
    </div>
    {feature_highlight}
</body>
</html>'''


def create_sample_spacetime() -> MockSpacetime:
    """Create realistic test data."""
    trace = MockTrace(
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_ms=100.5,
        stage_durations={
            'features': 25.3,
            'retrieval': 45.2,
            'decision': 15.1,
            'execution': 14.9
        },
        threads_activated=['thread_001', 'thread_002', 'thread_003']
    )

    return MockSpacetime(
        query_text='What is Thompson Sampling and how does it balance exploration vs exploitation?',
        response='Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that maintains probability distributions over the expected reward of each action.',
        tool_used='answer',
        confidence=0.87,
        trace=trace,
        metadata={'complexity': 'FAST', 'execution_mode': 'fast'}
    )


def main() -> int:
    """Run the elegant dashboard demo."""
    logger.info('[1/3] Creating sample Spacetime...')
    spacetime = create_sample_spacetime()
    logger.info(f'      OK Query: {spacetime.query_text[:50]}...')
    logger.info(f'      OK Confidence: {spacetime.confidence}, Duration: {spacetime.trace.duration_ms}ms')

    logger.info('\n[2/3] Generating elegant dashboard...')
    generator = DashboardGenerator()
    html = generator.generate(spacetime)
    logger.info(f'      OK Generated {len(html)} chars, {len(html.splitlines())} lines')

    logger.info('\n[3/3] Saving dashboard...')
    output = Path(__file__).parent / 'output' / 'dashboard_elegant.html'

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding='utf-8')
        logger.info(f'      OK Saved to: {output}')
    except Exception as e:
        logger.error(f'      ERROR: Failed to save - {e}')
        return 1

    logger.info('\n' + '='*80)
    logger.info('RUTHLESS ELEGANCE PASS COMPLETE')
    logger.info('='*80)
    logger.info('\nKey Improvements:')
    logger.info('  [+] Composable: Each method generates one atomic component')
    logger.info('  [+] Testable: Class-based with dependency injection')
    logger.info('  [+] Type-safe: Full type hints throughout')
    logger.info('  [+] Clear: Obvious data flow, no magic')
    logger.info('  [+] DRY: Zero repetition, reusable components')
    logger.info(f'\nOpen in browser: file:///{output.absolute()}')

    try:
        import webbrowser
        webbrowser.open(str(output.absolute()))
        logger.info('\n[+] Opened in default browser')
    except:
        logger.info('\n[!] Could not auto-open browser - please open manually')

    return 0


if __name__ == '__main__':
    sys.exit(main())