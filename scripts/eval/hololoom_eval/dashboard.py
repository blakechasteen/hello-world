"""
HTML Dashboard Report Generator

Generates a self-contained HTML report with:
- Overall score gauge
- Benchmark results table with pass/fail
- Per-benchmark detail cards
- Statistical comparison charts (SVG)
- Recommendations section
- Metadata and reproducibility info
"""

import json
import html as html_lib
from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_dashboard(report_dict: Dict[str, Any], output_path: str) -> str:
    """
    Generate a self-contained HTML dashboard from an evaluation report.

    Args:
        report_dict: Report from EvaluationReport.to_dict()
        output_path: Path to write HTML file

    Returns:
        Path to generated HTML file
    """
    html_content = _build_html(report_dict)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path


def _build_html(report: Dict[str, Any]) -> str:
    """Build complete HTML document."""
    overall_score = report.get("overall_score", 0)
    overall_passed = report.get("overall_passed", False)
    overall_verdict = report.get("overall_verdict", "Unknown")
    mode = report.get("mode", "standard")
    timestamp = report.get("timestamp", datetime.now().isoformat())
    duration = report.get("duration_seconds", 0)
    benchmarks = report.get("benchmarks", [])
    dependencies = report.get("dependencies", {})
    recommendations = report.get("recommendations", [])

    # Build sections
    benchmark_rows = _build_benchmark_rows(benchmarks)
    benchmark_cards = _build_benchmark_cards(benchmarks)
    dep_rows = _build_dependency_rows(dependencies)
    rec_items = _build_recommendations(recommendations)
    score_gauge = _build_score_gauge(overall_score)
    bar_chart = _build_bar_chart(benchmarks)

    status_class = "pass" if overall_passed else "fail"
    status_text = "PASS" if overall_passed else "FAIL"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HoloLoom Evaluation Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #f0f6fc; font-size: 28px; margin-bottom: 4px; }}
h2 {{ color: #f0f6fc; font-size: 20px; margin: 30px 0 15px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
h3 {{ color: #e6edf3; font-size: 16px; margin-bottom: 8px; }}

.header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 30px; }}
.header-left h1 {{ margin-bottom: 4px; }}
.header-meta {{ color: #8b949e; font-size: 13px; }}
.header-meta span {{ margin-right: 16px; }}

.status {{ display: inline-block; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 14px; }}
.status.pass {{ background: #238636; color: #fff; }}
.status.fail {{ background: #da3633; color: #fff; }}

.score-section {{ display: flex; gap: 30px; align-items: center; margin-bottom: 30px; padding: 20px; background: #161b22; border: 1px solid #21262d; border-radius: 8px; }}
.gauge-container {{ flex-shrink: 0; }}

.score-details {{ flex: 1; }}
.score-details .verdict {{ font-size: 18px; margin-bottom: 10px; }}

.card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
.card.pass {{ border-left: 3px solid #238636; }}
.card.fail {{ border-left: 3px solid #da3633; }}
.card.error {{ border-left: 3px solid #d29922; }}

.card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
.card-title {{ font-weight: 600; font-size: 15px; }}
.card-score {{ font-family: 'SF Mono', monospace; font-size: 14px; }}
.card-verdict {{ color: #8b949e; font-size: 13px; }}

table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th {{ text-align: left; padding: 8px 12px; background: #161b22; color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #21262d; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 14px; }}
tr:hover td {{ background: rgba(56, 139, 253, 0.05); }}

.mono {{ font-family: 'SF Mono', 'Consolas', monospace; }}
.pass-text {{ color: #3fb950; }}
.fail-text {{ color: #f85149; }}
.warn-text {{ color: #d29922; }}

.chart-container {{ margin: 15px 0; }}

.rec-list {{ list-style: none; }}
.rec-list li {{ padding: 8px 12px; margin: 4px 0; background: #161b22; border-radius: 4px; border-left: 3px solid #388bfd; font-size: 14px; }}

.dep-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }}
.dep-item {{ padding: 8px 12px; background: #161b22; border-radius: 4px; font-size: 13px; display: flex; align-items: center; gap: 8px; }}
.dep-icon {{ font-size: 16px; }}

.details-toggle {{ cursor: pointer; color: #388bfd; font-size: 13px; }}
.details-content {{ margin-top: 8px; padding: 8px; background: #0d1117; border-radius: 4px; font-size: 12px; font-family: monospace; white-space: pre-wrap; max-height: 200px; overflow-y: auto; display: none; }}
.details-content.open {{ display: block; }}

.footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #21262d; color: #484f58; font-size: 12px; text-align: center; }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>HoloLoom Evaluation Report</h1>
    <div class="header-meta">
      <span>Mode: <strong>{html_lib.escape(mode.upper())}</strong></span>
      <span>Duration: <strong>{duration:.1f}s</strong></span>
      <span>{html_lib.escape(timestamp[:19])}</span>
    </div>
  </div>
  <span class="status {status_class}">{status_text} ({overall_score:.2f})</span>
</div>

<div class="score-section">
  <div class="gauge-container">
    {score_gauge}
  </div>
  <div class="score-details">
    <div class="verdict">{html_lib.escape(overall_verdict)}</div>
    <div style="color: #8b949e; font-size: 14px;">
      {len(benchmarks)} benchmarks evaluated |
      {sum(1 for b in benchmarks if b.get('passed'))} passed |
      {sum(1 for b in benchmarks if not b.get('passed'))} failed
    </div>
  </div>
</div>

<h2>Benchmark Results</h2>
<div class="chart-container">
{bar_chart}
</div>

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Score</th>
      <th>Status</th>
      <th>Verdict</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    {benchmark_rows}
  </tbody>
</table>

<h2>Benchmark Details</h2>
{benchmark_cards}

<h2>Dependencies</h2>
<div class="dep-grid">
  {dep_rows}
</div>

<h2>Recommendations</h2>
<ul class="rec-list">
  {rec_items}
</ul>

<div class="footer">
  HoloLoom Evaluation Framework v2.0 | Generated {html_lib.escape(timestamp[:19])} | Mode: {html_lib.escape(mode)}
</div>

<script>
document.querySelectorAll('.details-toggle').forEach(el => {{
  el.addEventListener('click', () => {{
    const content = el.parentElement.querySelector('.details-content');
    content.classList.toggle('open');
    el.textContent = content.classList.contains('open') ? 'Hide details' : 'Show details';
  }});
}});
</script>

</body>
</html>"""


def _build_score_gauge(score: float) -> str:
    """Build SVG radial gauge for overall score."""
    pct = score * 100
    angle = score * 270  # 270 degree sweep
    radius = 50
    cx, cy = 60, 60
    stroke_width = 10

    # Color based on score
    if score >= 0.8:
        color = "#238636"
    elif score >= 0.6:
        color = "#3fb950"
    elif score >= 0.4:
        color = "#d29922"
    else:
        color = "#f85149"

    # Arc calculation
    circumference = 2 * 3.14159 * radius
    arc_length = (angle / 360) * circumference
    dash = f"{arc_length} {circumference}"

    return f"""<svg width="120" height="120" viewBox="0 0 120 120">
  <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#21262d" stroke-width="{stroke_width}"
    stroke-dasharray="{0.75 * circumference} {circumference}" stroke-linecap="round"
    transform="rotate(135, {cx}, {cy})" />
  <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
    stroke-dasharray="{dash}" stroke-linecap="round"
    transform="rotate(135, {cx}, {cy})" />
  <text x="{cx}" y="{cy - 5}" text-anchor="middle" fill="{color}" font-size="24" font-weight="700" font-family="monospace">{pct:.0f}</text>
  <text x="{cx}" y="{cy + 12}" text-anchor="middle" fill="#8b949e" font-size="11">/ 100</text>
</svg>"""


def _build_bar_chart(benchmarks: List[Dict]) -> str:
    """Build horizontal bar chart SVG."""
    if not benchmarks:
        return ""

    bar_height = 28
    gap = 6
    label_width = 180
    chart_width = 700
    bar_area = chart_width - label_width - 40
    total_height = len(benchmarks) * (bar_height + gap) + 20

    bars = []
    for i, b in enumerate(benchmarks):
        y = i * (bar_height + gap) + 10
        score = b.get("score", 0)
        passed = b.get("passed", False)
        name = b.get("name", "Unknown")[:25]
        color = "#238636" if passed else "#f85149"
        if b.get("error"):
            color = "#d29922"

        w = max(2, score * bar_area)

        bars.append(f"""  <text x="{label_width - 8}" y="{y + bar_height / 2 + 4}" text-anchor="end"
    fill="#c9d1d9" font-size="12" font-family="sans-serif">{html_lib.escape(name)}</text>
  <rect x="{label_width}" y="{y}" width="{w}" height="{bar_height}" rx="3" fill="{color}" opacity="0.85" />
  <text x="{label_width + w + 6}" y="{y + bar_height / 2 + 4}" fill="#8b949e" font-size="11" font-family="monospace">{score:.2f}</text>""")

    return f"""<svg width="{chart_width}" height="{total_height}" viewBox="0 0 {chart_width} {total_height}">
{''.join(bars)}
</svg>"""


def _build_benchmark_rows(benchmarks: List[Dict]) -> str:
    """Build table rows for benchmarks."""
    rows = []
    for b in benchmarks:
        name = html_lib.escape(b.get("name", "Unknown"))
        score = b.get("score", 0)
        passed = b.get("passed", False)
        verdict = html_lib.escape(str(b.get("verdict", ""))[:80])
        duration = b.get("duration_seconds", 0)
        error = b.get("error")

        if error:
            status_html = '<span class="warn-text">ERROR</span>'
        elif passed:
            status_html = '<span class="pass-text">PASS</span>'
        else:
            status_html = '<span class="fail-text">FAIL</span>'

        rows.append(f"""<tr>
  <td><strong>{name}</strong></td>
  <td class="mono">{score:.3f}</td>
  <td>{status_html}</td>
  <td class="card-verdict">{verdict}</td>
  <td class="mono">{duration:.1f}s</td>
</tr>""")

    return "\n".join(rows)


def _build_benchmark_cards(benchmarks: List[Dict]) -> str:
    """Build detail cards for each benchmark."""
    cards = []
    for b in benchmarks:
        name = html_lib.escape(b.get("name", "Unknown"))
        score = b.get("score", 0)
        passed = b.get("passed", False)
        verdict = html_lib.escape(str(b.get("verdict", "")))
        error = b.get("error")
        details = b.get("details", {})

        if error:
            card_class = "error"
        elif passed:
            card_class = "pass"
        else:
            card_class = "fail"

        details_json = json.dumps(details, indent=2, default=str)[:2000]

        cards.append(f"""<div class="card {card_class}">
  <div class="card-header">
    <span class="card-title">{name}</span>
    <span class="card-score mono">{score:.3f}</span>
  </div>
  <div class="card-verdict">{verdict}</div>
  {f'<div class="card-verdict warn-text">Error: {html_lib.escape(str(error)[:100])}</div>' if error else ''}
  <div>
    <span class="details-toggle">Show details</span>
    <div class="details-content">{html_lib.escape(details_json)}</div>
  </div>
</div>""")

    return "\n".join(cards)


def _build_dependency_rows(dependencies: Dict[str, bool]) -> str:
    """Build dependency grid items."""
    items = []
    for dep, available in dependencies.items():
        icon = "&#10003;" if available else "&#10007;"
        cls = "pass-text" if available else "fail-text"
        items.append(f"""<div class="dep-item">
  <span class="dep-icon {cls}">{icon}</span>
  <span>{html_lib.escape(dep)}</span>
</div>""")
    return "\n".join(items)


def _build_recommendations(recommendations: List[str]) -> str:
    """Build recommendation list items."""
    if not recommendations:
        return '<li>No recommendations - all benchmarks passed.</li>'
    items = []
    for i, rec in enumerate(recommendations, 1):
        items.append(f'<li>{i}. {html_lib.escape(rec)}</li>')
    return "\n".join(items)
