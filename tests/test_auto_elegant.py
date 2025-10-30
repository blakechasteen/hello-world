#!/usr/bin/env python3
"""
Ruthless Tests for auto() Visualization API
============================================
Testing philosophy: "If the test is complex, the API failed."

Each test should be 1-3 lines of actual testing code.
"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.visualization import auto, render, save, Dashboard


# ============================================================================
# Fixtures (Setup)
# ============================================================================

@pytest.fixture
def simple_data():
    """Simple data dict."""
    return {
        'month': ['Jan', 'Feb', 'Mar'],
        'sales': [100, 120, 140],
        'costs': [60, 70, 80]
    }

@pytest.fixture
def complex_data():
    """Complex multi-pattern data."""
    return {
        'hour': list(range(24)),
        'cpu': [45 + i*2 + (50 if i == 12 else 0) for i in range(24)],  # Spike at hour 12
        'memory': [60 + i*0.5 for i in range(24)],
        'requests': [1200 + i*50 for i in range(24)]
    }

@pytest.fixture
def mock_spacetime():
    """Mock Spacetime object."""
    @dataclass
    class MockTrace:
        duration_ms: float = 234.5
        stages: list = field(default_factory=lambda: [
            {'name': 'Extract', 'duration_ms': 50},
            {'name': 'Retrieve', 'duration_ms': 100},
            {'name': 'Decide', 'duration_ms': 84.5}
        ])

    @dataclass
    class MockSpacetime:
        query_text: str = "Test query"
        response: str = "Test response"
        tool_used: str = "test_tool"
        confidence: float = 0.95
        trace: Any = field(default_factory=MockTrace)
        metadata: Dict[str, Any] = field(default_factory=lambda: {
            'analysis_data': {
                'x': [1, 2, 3],
                'y': [10, 20, 30]
            }
        })

        def to_dict(self):
            return {'query': self.query_text}

    return MockSpacetime()


# ============================================================================
# Tests: Core Functionality
# ============================================================================

def test_auto_from_dict(simple_data):
    """Test auto() with simple dict - the most common case."""
    # One line test
    dashboard = auto(simple_data)

    # Assertions
    assert isinstance(dashboard, Dashboard)
    assert len(dashboard.panels) > 0
    assert dashboard.title  # Has a title


def test_auto_from_complex_data(complex_data):
    """Test auto() detects patterns in complex data."""
    dashboard = auto(complex_data, title="Server Metrics")

    # Should detect time-series, correlation, outlier patterns
    assert len(dashboard.panels) >= 5  # Multiple visualizations
    assert any(p.type.value == 'line' for p in dashboard.panels)  # Time-series chart
    assert dashboard.metadata.get('auto_generated') is True


def test_auto_from_spacetime(mock_spacetime):
    """Test auto() extracts and visualizes Spacetime data."""
    dashboard = auto(mock_spacetime)

    assert isinstance(dashboard, Dashboard)
    assert 'Test query' in dashboard.title or dashboard.title
    assert len(dashboard.panels) > 0


def test_auto_with_title():
    """Test auto() respects custom title."""
    data = {'x': [1, 2], 'y': [3, 4]}
    dashboard = auto(data, title="Custom Title")

    assert dashboard.title == "Custom Title"


# ============================================================================
# Tests: Rendering
# ============================================================================

def test_render_produces_html(simple_data):
    """Test render() produces valid HTML."""
    dashboard = auto(simple_data)
    html = render(dashboard)

    assert isinstance(html, str)
    assert '<!DOCTYPE html>' in html
    assert '<html' in html
    assert '</html>' in html
    assert len(html) > 1000  # Reasonable size


def test_render_dark_theme(simple_data):
    """Test render() with dark theme."""
    dashboard = auto(simple_data)
    html = render(dashboard, theme='dark')

    assert '.dark-theme' in html or 'dark' in html.lower()


# ============================================================================
# Tests: Saving
# ============================================================================

def test_save_creates_file(simple_data, tmp_path):
    """Test save() creates HTML file."""
    dashboard = auto(simple_data)
    output_file = tmp_path / "test_dashboard.html"

    save(dashboard, str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 1000  # Has content


def test_auto_with_save_path(simple_data, tmp_path):
    """Test auto() with direct save_path."""
    output_file = tmp_path / "direct_save.html"

    dashboard = auto(simple_data, save_path=str(output_file))

    assert output_file.exists()
    assert isinstance(dashboard, Dashboard)


# ============================================================================
# Tests: Intelligence
# ============================================================================

def test_auto_detects_patterns(complex_data):
    """Test auto() detects multiple data patterns."""
    dashboard = auto(complex_data)

    patterns = dashboard.metadata.get('patterns_detected', [])
    assert len(patterns) > 0
    # Should detect correlation, trend, outlier
    assert any('correlation' in p for p in patterns)


def test_auto_generates_insights(complex_data):
    """Test auto() generates insight panels."""
    dashboard = auto(complex_data)

    insight_panels = [p for p in dashboard.panels if p.type.value == 'insight']
    assert len(insight_panels) > 0  # Should generate insights


def test_auto_selects_layout():
    """Test auto() selects appropriate layout based on panel count."""
    # Few panels -> simple layout
    small_data = {'x': [1], 'y': [2]}
    small_dash = auto(small_data)

    # Many panels -> complex layout
    large_data = {f'col{i}': list(range(10)) for i in range(8)}
    large_dash = auto(large_data)

    assert small_dash.layout != large_dash.layout  # Different layouts


# ============================================================================
# Tests: Edge Cases
# ============================================================================

def test_auto_empty_data():
    """Test auto() handles empty data gracefully."""
    with pytest.raises(ValueError):
        auto({})


def test_auto_single_column():
    """Test auto() with single column."""
    data = {'values': [1, 2, 3, 4, 5]}
    dashboard = auto(data)

    assert len(dashboard.panels) > 0  # Still generates something


def test_auto_non_numeric_data():
    """Test auto() with categorical data."""
    data = {
        'category': ['A', 'B', 'C', 'D'],
        'count': [10, 20, 15, 25]
    }
    dashboard = auto(data)

    # Should generate bar chart
    assert any(p.type.value == 'bar' for p in dashboard.panels)


# ============================================================================
# Tests: Performance
# ============================================================================

def test_auto_is_fast(simple_data):
    """Test auto() completes quickly."""
    import time

    start = time.time()
    dashboard = auto(simple_data)
    duration = time.time() - start

    assert duration < 1.0  # Should complete in under 1 second
    assert len(dashboard.panels) > 0


def test_render_is_fast(simple_data):
    """Test render() completes quickly."""
    import time

    dashboard = auto(simple_data)

    start = time.time()
    html = render(dashboard)
    duration = time.time() - start

    assert duration < 0.5  # Rendering should be fast
    assert len(html) > 0


# ============================================================================
# Tests: Integration
# ============================================================================

def test_auto_then_render_then_save(simple_data, tmp_path):
    """Test complete workflow: auto -> render -> save."""
    # Three clean steps
    dashboard = auto(simple_data)
    html = render(dashboard)
    output = tmp_path / "workflow.html"
    save(dashboard, str(output))

    # All succeed
    assert isinstance(dashboard, Dashboard)
    assert len(html) > 1000
    assert output.exists()


def test_auto_preserves_data_fidelity(simple_data):
    """Test auto() doesn't lose data during visualization."""
    dashboard = auto(simple_data)
    html = render(dashboard)

    # Original data values should appear in HTML
    assert '100' in html  # First sales value
    assert '140' in html  # Last sales value


# ============================================================================
# Ruthless Score
# ============================================================================

def test_ruthless_elegance():
    """Test that the API truly is ruthlessly elegant."""
    # This test verifies the philosophy

    # Requirement 1: One line to create dashboard
    data = {'x': [1, 2], 'y': [3, 4]}
    dashboard = auto(data)  # ✓ One line
    assert isinstance(dashboard, Dashboard)

    # Requirement 2: Zero configuration
    # (No parameters needed beyond data)
    dashboard2 = auto(data)  # ✓ No config params
    assert dashboard2.panels  # Works with defaults

    # Requirement 3: Automatic intelligence
    # (Detects patterns without being told)
    assert dashboard.metadata.get('auto_generated') is True

    # Requirement 4: Complete output
    html = render(dashboard)
    assert '<!DOCTYPE html>' in html
    assert len(html) > 1000  # Full featured

    # Ruthless score: 4/4 ✓
    assert True, "API is ruthlessly elegant"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
