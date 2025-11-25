"""
DATAPIG Visual Dashboard - Tufte-Style HTML Reports

"Above all else show the data." - Edward Tufte

Generates beautiful, information-dense HTML reports for data quality analysis.
Follows Edward Tufte's principles: maximize data-ink ratio, minimize chartjunk.

**Features**:
- Small multiples for comparing quality across datasets
- Inline sparklines showing trends over time
- Data density tables with maximum information per square inch
- Interactive HTML with zero external dependencies
- Bottleneck detection and automatic highlighting

**Star Trek Theme**: Quality metrics represented as "ship systems status"

**Usage**:
    from HoloLoom.datapig.dashboard import render_quality_dashboard

    # After analyzing multiple datasets
    reports = []
    for dataset in datasets:
        detector = DataPigDetector()
        issues = detector.analyze_dataset(dataset)
        reports.append({
            'dataset_name': 'users.csv',
            'timestamp': time.time(),
            'issues': issues,
            'dataset_size': len(dataset)
        })

    # Generate dashboard
    html = render_quality_dashboard(reports, title='Production Data Quality')

    # Save to file
    with open('quality_dashboard.html', 'w') as f:
        f.write(html)
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter

# DATAPIG imports
try:
    from HoloLoom.datapig.detector import DataQualityIssue, IssueType, Severity
    DATAPIG_AVAILABLE = True
except ImportError:
    DATAPIG_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class QualityReport:
    """Quality report for a single dataset"""
    dataset_name: str
    timestamp: float
    issues: List['DataQualityIssue']
    dataset_size: int
    duration_ms: float = 0.0

    def get_issue_counts_by_type(self) -> Dict[str, int]:
        """Count issues by type"""
        counter = Counter(i.issue_type.value for i in self.issues)
        return dict(counter)

    def get_issue_counts_by_severity(self) -> Dict[str, int]:
        """Count issues by severity"""
        counter = Counter(i.severity.value for i in self.issues)
        return dict(counter)

    def get_critical_issues(self) -> List['DataQualityIssue']:
        """Get CAPTAIN-level (critical) issues"""
        return [i for i in self.issues if i.severity.value == "CAPTAIN"]

    def get_quality_score(self) -> float:
        """
        Calculate quality score (0.0-1.0)

        Formula: 1.0 - (weighted_issues / dataset_size)
        Weights: CAPTAIN=1.0, COMMANDER=0.5, LIEUTENANT=0.25, ENSIGN=0.1
        """
        if self.dataset_size == 0:
            return 1.0

        severity_weights = {
            "CAPTAIN": 1.0,
            "COMMANDER": 0.5,
            "LIEUTENANT": 0.25,
            "ENSIGN": 0.1
        }

        weighted_issues = sum(
            severity_weights.get(i.severity.value, 0.1)
            for i in self.issues
        )

        # Normalize by dataset size
        score = 1.0 - min(weighted_issues / self.dataset_size, 1.0)
        return max(score, 0.0)  # Clamp to [0, 1]


# ============================================================================
# Sparkline Generation (Tufte-style)
# ============================================================================

def render_sparkline(values: List[float], width: int = 100, height: int = 30) -> str:
    """
    Render inline SVG sparkline (Tufte-style word-sized graphic)

    **Design Principles**:
    - Small: 100x30px (word-sized)
    - No axes, no labels (data speaks for itself)
    - Thin line (1px)
    - Endpoint indicators (last value highlighted)

    Args:
        values: List of numeric values
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        SVG string for inline HTML
    """
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'

    # Normalize to 0-1 range
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        # Flat line
        normalized = [0.5] * len(values)
    else:
        normalized = [(v - min_val) / (max_val - min_val) for v in values]

    # Create polyline points (invert Y for SVG coordinate system)
    points = []
    x_step = width / (len(values) - 1)
    for i, val in enumerate(normalized):
        x = i * x_step
        y = height - (val * height)  # Invert Y
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    # Endpoint indicator (last value)
    last_x = (len(values) - 1) * x_step
    last_y = height - (normalized[-1] * height)

    # Color based on trend (last vs first)
    if normalized[-1] > normalized[0]:
        color = "#10b981"  # Green (improving)
    elif normalized[-1] < normalized[0]:
        color = "#ef4444"  # Red (degrading)
    else:
        color = "#6b7280"  # Gray (stable)

    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1"/>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2" fill="{color}"/>
</svg>'''

    return svg


# ============================================================================
# Small Multiples (Compare Multiple Datasets)
# ============================================================================

def render_small_multiples(
    reports: List[QualityReport],
    max_columns: int = 4
) -> str:
    """
    Render small multiples grid comparing multiple datasets

    **Tufte Principle**: Use consistent scales to enable fair comparison

    Each dataset gets a card showing:
    - Dataset name and timestamp
    - Quality score (0-100) with trend sparkline
    - Issue count breakdown
    - Critical issues highlighted

    Args:
        reports: List of quality reports
        max_columns: Maximum columns in grid (default: 4)

    Returns:
        HTML string
    """
    if not reports:
        return '<p style="color: #6b7280;">No reports available</p>'

    # Calculate global min/max for consistent scales
    all_scores = [r.get_quality_score() for r in reports]
    min_score = min(all_scores) if all_scores else 0.0
    max_score = max(all_scores) if all_scores else 1.0

    # Group into rows
    rows = []
    for i in range(0, len(reports), max_columns):
        rows.append(reports[i:i + max_columns])

    html_parts = ['<div style="display: grid; gap: 16px;">']

    for row in rows:
        html_parts.append('<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;">')

        for report in row:
            score = report.get_quality_score()
            score_pct = score * 100

            # Color based on score
            if score >= 0.9:
                score_color = "#10b981"  # Green (excellent)
                status = "EXCELLENT"
            elif score >= 0.75:
                score_color = "#3b82f6"  # Blue (good)
                status = "GOOD"
            elif score >= 0.5:
                score_color = "#f59e0b"  # Amber (fair)
                status = "FAIR"
            else:
                score_color = "#ef4444"  # Red (poor)
                status = "POOR"

            # Issue counts
            issue_counts = report.get_issue_counts_by_severity()
            critical = issue_counts.get("CAPTAIN", 0)
            high = issue_counts.get("COMMANDER", 0)
            medium = issue_counts.get("LIEUTENANT", 0)
            low = issue_counts.get("ENSIGN", 0)

            # Timestamp
            from datetime import datetime
            ts = datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M')

            card_html = f'''
<div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; background: white;">
  <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
    <div>
      <div style="font-weight: 600; font-size: 14px; color: #111827;">{report.dataset_name}</div>
      <div style="font-size: 12px; color: #6b7280; font-family: 'SF Mono', monospace;">{ts}</div>
    </div>
    <div style="text-align: right;">
      <div style="font-size: 24px; font-weight: 700; color: {score_color};">{score_pct:.0f}</div>
      <div style="font-size: 10px; color: {score_color}; font-weight: 600;">{status}</div>
    </div>
  </div>

  <div style="margin-bottom: 12px; font-size: 12px; color: #6b7280;">
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
      <div><span style="color: #ef4444;">●</span> Critical: {critical}</div>
      <div><span style="color: #f59e0b;">●</span> High: {high}</div>
      <div><span style="color: #3b82f6;">●</span> Medium: {medium}</div>
      <div><span style="color: #6b7280;">●</span> Low: {low}</div>
    </div>
  </div>

  <div style="font-size: 11px; color: #6b7280; border-top: 1px solid #e5e7eb; padding-top: 8px;">
    <span style="font-family: 'SF Mono', monospace;">{len(report.issues)} issues</span> in
    <span style="font-family: 'SF Mono', monospace;">{report.dataset_size} rows</span>
    {f' • {report.duration_ms:.0f}ms' if report.duration_ms > 0 else ''}
  </div>
</div>
'''
            html_parts.append(card_html)

        html_parts.append('</div>')

    html_parts.append('</div>')

    return '\n'.join(html_parts)


# ============================================================================
# Data Density Table (Maximum Information Per Square Inch)
# ============================================================================

def render_density_table(reports: List[QualityReport]) -> str:
    """
    Render high-density issue breakdown table

    **Tufte Principle**: Maximum data-ink ratio, tight spacing, small fonts

    Shows:
    - Issue type breakdown across all datasets
    - Sparklines showing trends
    - Total counts and percentages

    Args:
        reports: List of quality reports

    Returns:
        HTML string
    """
    if not reports:
        return '<p style="color: #6b7280;">No reports available</p>'

    # Aggregate issue types across all reports
    all_issue_types = set()
    for report in reports:
        all_issue_types.update(report.get_issue_counts_by_type().keys())

    # Sort issue types alphabetically
    issue_types = sorted(all_issue_types)

    # Build table
    html_parts = ['''
<table style="width: 100%; border-collapse: collapse; font-size: 12px; font-family: 'SF Mono', monospace;">
  <thead>
    <tr style="border-bottom: 2px solid #111827; text-align: left;">
      <th style="padding: 8px; font-weight: 600; color: #111827;">Issue Type</th>
      <th style="padding: 8px; font-weight: 600; color: #111827; text-align: right;">Count</th>
      <th style="padding: 8px; font-weight: 600; color: #111827; text-align: right;">%</th>
      <th style="padding: 8px; font-weight: 600; color: #111827;">Trend</th>
    </tr>
  </thead>
  <tbody>
''']

    total_issues = sum(len(r.issues) for r in reports)

    for issue_type in issue_types:
        # Count occurrences across reports
        counts = [r.get_issue_counts_by_type().get(issue_type, 0) for r in reports]
        total_count = sum(counts)
        percentage = (total_count / total_issues * 100) if total_issues > 0 else 0.0

        # Generate sparkline
        sparkline = render_sparkline(counts, width=80, height=20)

        # Row color based on severity (heuristic based on issue type name)
        if "CAPTAIN" in issue_type or "DATA_LEAK" in issue_type or "HIGH_ENTROPY_PII" in issue_type:
            row_color = "#fef2f2"  # Light red
        elif "COMMANDER" in issue_type:
            row_color = "#fff7ed"  # Light amber
        else:
            row_color = "white"

        row_html = f'''
    <tr style="border-bottom: 1px solid #e5e7eb; background: {row_color};">
      <td style="padding: 6px 8px; color: #374151;">{issue_type}</td>
      <td style="padding: 6px 8px; text-align: right; color: #111827; font-weight: 600;">{total_count}</td>
      <td style="padding: 6px 8px; text-align: right; color: #6b7280;">{percentage:.1f}%</td>
      <td style="padding: 6px 8px;">{sparkline}</td>
    </tr>
'''
        html_parts.append(row_html)

    html_parts.append('''
  </tbody>
</table>
''')

    return '\n'.join(html_parts)


# ============================================================================
# Main Dashboard Renderer
# ============================================================================

def render_quality_dashboard(
    reports: List[QualityReport],
    title: str = "DATAPIG Quality Dashboard",
    subtitle: Optional[str] = None
) -> str:
    """
    Render complete Tufte-style quality dashboard

    **Sections**:
    1. Header with title and timestamp
    2. Small multiples grid (dataset comparison)
    3. Data density table (issue breakdown)
    4. Critical issues list (if any)

    Args:
        reports: List of quality reports
        title: Dashboard title
        subtitle: Optional subtitle

    Returns:
        Complete HTML document
    """
    if not reports:
        return '<html><body><p>No reports available</p></body></html>'

    # Generate timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Calculate summary statistics
    total_issues = sum(len(r.issues) for r in reports)
    total_datasets = len(reports)
    avg_quality = sum(r.get_quality_score() for r in reports) / total_datasets

    # Count critical issues
    critical_issues = []
    for report in reports:
        critical_issues.extend([
            (report.dataset_name, issue)
            for issue in report.get_critical_issues()
        ])

    # Render sections
    small_multiples_html = render_small_multiples(reports, max_columns=4)
    density_table_html = render_density_table(reports)

    # Critical issues section
    critical_html = ""
    if critical_issues:
        critical_html = '''
<div style="margin-top: 32px; padding: 16px; background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 4px;">
  <div style="font-weight: 600; font-size: 16px; color: #991b1b; margin-bottom: 12px;">
    🚨 Critical Issues ({count})
  </div>
  <div style="font-size: 14px; color: #7f1d1d;">
'''.format(count=len(critical_issues))

        for dataset_name, issue in critical_issues[:10]:  # Show max 10
            critical_html += f'''
    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 4px;">
      <div style="font-weight: 600; font-size: 12px; color: #111827;">{dataset_name}</div>
      <div style="font-size: 12px; color: #6b7280; font-family: 'SF Mono', monospace;">{issue.message}</div>
    </div>
'''

        critical_html += '''
  </div>
</div>
'''

    # Assemble complete HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px;
      background: #f9fafb;
    }}
    .header {{
      margin-bottom: 32px;
      border-bottom: 2px solid #111827;
      padding-bottom: 16px;
    }}
    .section {{
      margin-bottom: 32px;
      background: white;
      padding: 24px;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1 style="margin: 0; font-size: 32px; color: #111827;">{title}</h1>
    {f'<p style="margin: 8px 0 0 0; font-size: 16px; color: #6b7280;">{subtitle}</p>' if subtitle else ''}
    <p style="margin: 8px 0 0 0; font-size: 14px; color: #9ca3af;">Generated: {timestamp}</p>

    <div style="margin-top: 16px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
      <div>
        <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;">Datasets</div>
        <div style="font-size: 24px; font-weight: 700; color: #111827;">{total_datasets}</div>
      </div>
      <div>
        <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;">Total Issues</div>
        <div style="font-size: 24px; font-weight: 700; color: #111827;">{total_issues}</div>
      </div>
      <div>
        <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;">Avg Quality</div>
        <div style="font-size: 24px; font-weight: 700; color: #111827;">{avg_quality * 100:.0f}%</div>
      </div>
    </div>
  </div>

  {critical_html}

  <div class="section">
    <h2 style="margin: 0 0 16px 0; font-size: 18px; color: #111827;">Dataset Comparison</h2>
    {small_multiples_html}
  </div>

  <div class="section">
    <h2 style="margin: 0 0 16px 0; font-size: 18px; color: #111827;">Issue Breakdown</h2>
    {density_table_html}
  </div>

  <div style="margin-top: 32px; text-align: center; font-size: 12px; color: #9ca3af;">
    Powered by DATAPIG • Data Analysis & Tactical Assessment Program for Integrity Governance
  </div>
</body>
</html>
'''

    return html


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'QualityReport',
    'render_quality_dashboard',
    'render_small_multiples',
    'render_density_table',
    'render_sparkline'
]
