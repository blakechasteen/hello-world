"""
Tufte-Style Verification Visualization Panel

Implements a high-information-density verification analysis panel that displays
claim extraction, question generation, contradiction detection, and quality metrics
in a compact 4-quadrant layout following Edward Tufte's design principles.

Key Features:
- Maximum data-ink ratio (~70% vs typical ~30%)
- No chartjunk, semantic colors for severity
- Inline sparklines for trend visualization
- Compact 4-quadrant layout for quick analysis
- Zero external dependencies (pure HTML/CSS/SVG)

Author: Claude Code
Created: 2025-12-09
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class SeverityLevel(Enum):
    """Severity levels for contradictions and issues."""
    NONE = "none"        # Green: No issues
    MINOR = "minor"      # Amber: Minor contradictions
    MAJOR = "major"      # Red: Major contradictions


class ClaimType(Enum):
    """Types of claims that can be extracted."""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    UNKNOWN = "unknown"


@dataclass
class Contradiction:
    """Represents a contradiction detected during verification."""
    contradiction_type: str      # Type: logical, temporal, reference, etc.
    severity: SeverityLevel      # MINOR, MAJOR
    description: str             # Human-readable description
    claim_ids: List[int] = field(default_factory=list)  # Conflicting claim indices
    confidence: float = 0.0      # Contradiction confidence (0.0-1.0)


@dataclass
class VerificationPanelData:
    """Data structure for verification panel visualization."""
    claims_count: int                              # Total claims extracted
    claim_types: Dict[str, int]                    # Type -> count mapping
    questions_generated: int                       # Number of verification questions
    contradictions: List[Contradiction]            # Detected contradictions
    quality_before: float                          # Quality before verification (0.0-1.0)
    quality_after: float                           # Quality after verification (0.0-1.0)
    verification_time_ms: float                    # Time spent verifying (milliseconds)
    trend_data: Optional[List[float]] = None       # Historical verification times
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


def _render_sparkline(
    values: List[float],
    width: int = 100,
    height: int = 30,
    color: str = "#6366F1"
) -> str:
    """
    Render an SVG sparkline (word-sized graphic).

    Generates a minimal SVG sparkline showing trends inline with metrics.
    Automatically normalizes values to fit within the viewport.

    Args:
        values: List of numeric values to plot
        width: SVG width in pixels (default: 100)
        height: SVG height in pixels (default: 30)
        color: Line color (default: indigo #6366F1)

    Returns:
        SVG string for inline embedding
    """
    if not values or len(values) < 2:
        return '<svg width="100" height="30" style="display:inline;"></svg>'

    # Normalize values to fit in SVG
    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val if max_val > min_val else 1.0

    # Calculate points
    points = []
    x_step = (width - 4) / (len(values) - 1) if len(values) > 1 else width - 4
    for i, v in enumerate(values):
        x = 2 + i * x_step
        y_normalized = (v - min_val) / value_range if value_range > 0 else 0.5
        y = height - 2 - (y_normalized * (height - 4))
        points.append(f"{x},{y}")

    points_str = " ".join(points)

    # Minimal SVG with just the line
    return f'''<svg width="{width}" height="{height}" style="display:inline;margin:0 2px;vertical-align:middle;">
        <polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
    </svg>'''


def _determine_severity(contradictions: List[Contradiction]) -> SeverityLevel:
    """
    Determine overall severity from contradictions list.

    Aggregates individual contradiction severities to a single level:
    - Any MAJOR → overall MAJOR
    - Any MINOR → overall MINOR
    - No contradictions → NONE

    Args:
        contradictions: List of detected contradictions

    Returns:
        Overall SeverityLevel
    """
    if not contradictions:
        return SeverityLevel.NONE

    has_major = any(c.severity == SeverityLevel.MAJOR for c in contradictions)
    if has_major:
        return SeverityLevel.MAJOR

    return SeverityLevel.MINOR


def _get_severity_color(severity: SeverityLevel) -> str:
    """Get semantic color for severity level."""
    color_map = {
        SeverityLevel.NONE: "#10B981",   # Green
        SeverityLevel.MINOR: "#F59E0B",  # Amber
        SeverityLevel.MAJOR: "#EF4444"   # Red
    }
    return color_map.get(severity, "#6B7280")


def _get_severity_emoji(severity: SeverityLevel) -> str:
    """Get emoji indicator for severity level."""
    emoji_map = {
        SeverityLevel.NONE: "✓",    # Checkmark
        SeverityLevel.MINOR: "⚠",   # Warning
        SeverityLevel.MAJOR: "✗"    # X mark
    }
    return emoji_map.get(severity, "?")


def _render_claims_section(claim_types: Dict[str, int]) -> str:
    """
    Render claims extraction mini-visualization.

    Creates a simple bar chart showing claim type distribution.
    Maximizes data density with minimal decoration.

    Args:
        claim_types: Dict mapping claim type -> count

    Returns:
        HTML string for claims section
    """
    if not claim_types:
        return '<div class="claims-section"><p style="color:#9CA3AF;">No claims extracted</p></div>'

    total = sum(claim_types.values())
    max_count = max(claim_types.values()) if claim_types else 1

    # Sort by count (descending)
    sorted_types = sorted(claim_types.items(), key=lambda x: x[1], reverse=True)

    rows = []
    for claim_type, count in sorted_types:
        percentage = (count / total * 100) if total > 0 else 0
        bar_width = (count / max_count * 100) if max_count > 0 else 0

        rows.append(f'''
        <div style="display:flex;gap:8px;align-items:center;margin:4px 0;font-size:12px;">
            <span style="width:60px;font-weight:500;text-transform:capitalize;">{claim_type}</span>
            <div style="flex:1;background:#E5E7EB;height:16px;position:relative;border-radius:2px;">
                <div style="background:#6366F1;height:100%;width:{bar_width}%;border-radius:2px;"></div>
            </div>
            <span style="width:40px;text-align:right;font-feature-settings:'tnum';">{count}</span>
        </div>
        ''')

    return f'''
    <div class="claims-section" style="padding:12px;">
        <div style="font-weight:600;font-size:11px;color:#6B7280;text-transform:uppercase;margin-bottom:8px;">Claims: {total}</div>
        {''.join(rows)}
    </div>
    '''


def _render_contradictions_section(contradictions: List[Contradiction]) -> str:
    """
    Render contradictions detection section.

    Lists detected contradictions with severity indicators and descriptions.
    Semantic coloring based on severity level.

    Args:
        contradictions: List of contradictions to display

    Returns:
        HTML string for contradictions section
    """
    if not contradictions:
        return '''
        <div class="contradictions-section" style="padding:12px;">
            <div style="font-weight:600;font-size:11px;color:#6B7280;text-transform:uppercase;margin-bottom:8px;">Contradictions</div>
            <div style="color:#10B981;font-weight:500;font-size:12px;">✓ None detected</div>
        </div>
        '''

    # Sort by severity (MAJOR first)
    sorted_contradictions = sorted(
        contradictions,
        key=lambda c: (c.severity != SeverityLevel.MAJOR, -c.confidence)
    )

    rows = []
    for i, contradiction in enumerate(sorted_contradictions[:5], 1):  # Show top 5
        severity_color = _get_severity_color(contradiction.severity)
        emoji = _get_severity_emoji(contradiction.severity)

        rows.append(f'''
        <div style="margin:6px 0;padding:6px 8px;background:#F9FAFB;border-left:3px solid {severity_color};border-radius:2px;">
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:2px;">
                <span style="color:{severity_color};font-weight:700;font-size:13px;">{emoji}</span>
                <span style="font-weight:500;font-size:11px;text-transform:capitalize;color:#374151;">{contradiction.contradiction_type}</span>
                <span style="font-size:10px;color:#9CA3AF;">({contradiction.confidence:.0%} confident)</span>
            </div>
            <div style="font-size:11px;color:#6B7280;line-height:1.3;">{contradiction.description}</div>
        </div>
        ''')

    count = len(contradictions)
    more_text = f"<div style='font-size:10px;color:#9CA3AF;margin-top:4px;'>+{count - 5} more...</div>" if count > 5 else ""

    return f'''
    <div class="contradictions-section" style="padding:12px;">
        <div style="font-weight:600;font-size:11px;color:#6B7280;text-transform:uppercase;margin-bottom:8px;">Contradictions: {count}</div>
        {''.join(rows)}
        {more_text}
    </div>
    '''


def _render_quality_section(
    quality_before: float,
    quality_after: float,
    verification_time_ms: float,
    trend_data: Optional[List[float]] = None
) -> str:
    """
    Render quality improvement section.

    Shows quality metrics before/after verification with directional indicator
    and trend sparkline if available.

    Args:
        quality_before: Quality score before verification (0.0-1.0)
        quality_after: Quality score after verification (0.0-1.0)
        verification_time_ms: Time spent verifying
        trend_data: Optional historical trend data for sparkline

    Returns:
        HTML string for quality section
    """
    improvement = quality_after - quality_before
    improvement_pct = (improvement / quality_before * 100) if quality_before > 0 else 0
    improvement_color = "#10B981" if improvement >= 0 else "#EF4444"
    arrow = "↑" if improvement >= 0 else "↓"

    # Format time display
    if verification_time_ms < 1000:
        time_display = f"{verification_time_ms:.0f}ms"
    else:
        time_display = f"{verification_time_ms / 1000:.1f}s"

    # Optional sparkline
    sparkline_html = ""
    if trend_data:
        sparkline_html = f'''
        <div style="margin-top:6px;font-size:10px;color:#9CA3AF;">
            Trend: {_render_sparkline(trend_data, width=80, height=20, color="#6366F1")}
        </div>
        '''

    return f'''
    <div class="quality-section" style="padding:12px;">
        <div style="font-weight:600;font-size:11px;color:#6B7280;text-transform:uppercase;margin-bottom:8px;">Quality</div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
            <div>
                <div style="font-size:10px;color:#9CA3AF;margin-bottom:2px;">Before</div>
                <div style="font-size:18px;font-weight:600;font-feature-settings:'tnum';">{quality_before:.2f}</div>
            </div>
            <div>
                <div style="font-size:10px;color:#9CA3AF;margin-bottom:2px;">After</div>
                <div style="font-size:18px;font-weight:600;font-feature-settings:'tnum';">{quality_after:.2f}</div>
            </div>
        </div>

        <div style="padding:6px 8px;background:#F9FAFB;border-radius:2px;">
            <div style="display:flex;align-items:center;gap:4px;">
                <span style="color:{improvement_color};font-weight:700;font-size:16px;">{arrow}</span>
                <span style="font-weight:600;color:{improvement_color};font-feature-settings:'tnum';">{improvement:+.2f}</span>
                <span style="font-size:11px;color:#6B7280;">({improvement_pct:+.0f}%)</span>
            </div>
        </div>

        <div style="font-size:10px;color:#9CA3AF;margin-top:6px;">
            Time: {time_display}
        </div>

        {sparkline_html}
    </div>
    '''


def _render_questions_section(questions_generated: int) -> str:
    """
    Render questions generated section.

    Simple display of verification question count.

    Args:
        questions_generated: Number of verification questions

    Returns:
        HTML string for questions section
    """
    return f'''
    <div class="questions-section" style="padding:12px;">
        <div style="font-weight:600;font-size:11px;color:#6B7280;text-transform:uppercase;margin-bottom:12px;">Questions Generated</div>
        <div style="font-size:32px;font-weight:600;font-feature-settings:'tnum';text-align:center;color:#6366F1;margin:8px 0;">
            {questions_generated}
        </div>
        <div style="font-size:11px;color:#9CA3AF;text-align:center;">verification questions</div>
    </div>
    '''


def render_verification_panel(
    data: VerificationPanelData,
    title: str = "Verification Analysis",
    subtitle: Optional[str] = None,
    show_timestamp: bool = False
) -> str:
    """
    Render complete verification analysis panel.

    Creates a Tufte-style 4-quadrant panel showing verification results:
    - Top left: Claims extraction breakdown
    - Top right: Questions generated count
    - Bottom left: Contradictions detected
    - Bottom right: Quality improvement metrics

    Follows Edward Tufte's principles:
    - Maximize data-ink ratio (~70%)
    - No decoration/chartjunk
    - Semantic colors for severity
    - Compact information density

    Args:
        data: VerificationPanelData with all metrics
        title: Panel title (default: "Verification Analysis")
        subtitle: Optional subtitle for context
        show_timestamp: Whether to include generation timestamp

    Returns:
        HTML string containing complete self-contained panel

    Example:
        >>> data = VerificationPanelData(
        ...     claims_count=12,
        ...     claim_types={"factual": 8, "procedural": 4},
        ...     questions_generated=5,
        ...     contradictions=[],
        ...     quality_before=0.72,
        ...     quality_after=0.89,
        ...     verification_time_ms=245.5,
        ...     trend_data=[250, 248, 245]
        ... )
        >>> html = render_verification_panel(data)
    """
    overall_severity = _determine_severity(data.contradictions)
    severity_color = _get_severity_color(overall_severity)

    # Render quadrants
    claims_html = _render_claims_section(data.claim_types)
    questions_html = _render_questions_section(data.questions_generated)
    contradictions_html = _render_contradictions_section(data.contradictions)
    quality_html = _render_quality_section(
        data.quality_before,
        data.quality_after,
        data.verification_time_ms,
        data.trend_data
    )

    # Optional timestamp
    timestamp_html = ""
    if show_timestamp:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_html = f'<div style="font-size:10px;color:#9CA3AF;margin-top:12px;">Generated: {ts}</div>'

    subtitle_html = f'<div style="font-size:12px;color:#9CA3AF;margin-bottom:12px;">{subtitle}</div>' if subtitle else ""

    return f'''
    <div class="verification-panel" style="font-family:system-ui,-apple-system,sans-serif;background:#FFFFFF;border:1px solid #E5E7EB;border-radius:6px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,0.05);">

        <!-- Header -->
        <div style="padding:12px 16px;border-bottom:1px solid #E5E7EB;background:#F9FAFB;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="color:{severity_color};font-weight:700;font-size:13px;">●</span>
                <h3 style="margin:0;font-size:13px;font-weight:600;color:#111827;">{title}</h3>
            </div>
            {subtitle_html}
        </div>

        <!-- 4-Quadrant Grid -->
        <div style="display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;">

            <!-- Top-Left: Claims -->
            <div style="border-right:1px solid #E5E7EB;border-bottom:1px solid #E5E7EB;">
                {claims_html}
            </div>

            <!-- Top-Right: Questions -->
            <div style="border-bottom:1px solid #E5E7EB;">
                {questions_html}
            </div>

            <!-- Bottom-Left: Contradictions -->
            <div style="border-right:1px solid #E5E7EB;">
                {contradictions_html}
            </div>

            <!-- Bottom-Right: Quality -->
            <div>
                {quality_html}
            </div>

        </div>

        {timestamp_html}

    </div>
    '''


def create_verification_panel_from_dict(
    result: Dict[str, Any],
    title: str = "Verification Analysis"
) -> str:
    """
    Create verification panel from dictionary result.

    Convenience factory function for creating panels from untyped dictionaries,
    useful when coming from JSON APIs or legacy code.

    Args:
        result: Dictionary with verification results
        title: Panel title

    Returns:
        HTML string for rendered panel

    Example:
        >>> result = {
        ...     "claims_count": 12,
        ...     "claim_types": {"factual": 8, "procedural": 4},
        ...     "questions_generated": 5,
        ...     "contradictions": [],
        ...     "quality_before": 0.72,
        ...     "quality_after": 0.89,
        ...     "verification_time_ms": 245.5
        ... }
        >>> html = create_verification_panel_from_dict(result)
    """
    # Parse contradictions
    contradictions = []
    for contra in result.get("contradictions", []):
        contradictions.append(Contradiction(
            contradiction_type=contra.get("type", "unknown"),
            severity=SeverityLevel(contra.get("severity", "minor")),
            description=contra.get("description", ""),
            claim_ids=contra.get("claim_ids", []),
            confidence=contra.get("confidence", 0.0)
        ))

    data = VerificationPanelData(
        claims_count=result.get("claims_count", 0),
        claim_types=result.get("claim_types", {}),
        questions_generated=result.get("questions_generated", 0),
        contradictions=contradictions,
        quality_before=result.get("quality_before", 0.0),
        quality_after=result.get("quality_after", 0.0),
        verification_time_ms=result.get("verification_time_ms", 0.0),
        trend_data=result.get("trend_data"),
        metadata=result.get("metadata", {})
    )

    return render_verification_panel(
        data,
        title=title,
        subtitle=result.get("subtitle")
    )


if __name__ == "__main__":
    # Example: Generate sample verification panel
    example_data = VerificationPanelData(
        claims_count=12,
        claim_types={
            "factual": 8,
            "procedural": 3,
            "analytical": 1
        },
        questions_generated=7,
        contradictions=[
            Contradiction(
                contradiction_type="temporal",
                severity=SeverityLevel.MINOR,
                description="Claim 3 contradicts timeline in claim 7",
                claim_ids=[3, 7],
                confidence=0.82
            ),
            Contradiction(
                contradiction_type="logical",
                severity=SeverityLevel.MAJOR,
                description="Claim 5 logically incompatible with claim 8",
                claim_ids=[5, 8],
                confidence=0.95
            )
        ],
        quality_before=0.72,
        quality_after=0.89,
        verification_time_ms=245.5,
        trend_data=[280, 260, 250, 245],
    )

    html = render_verification_panel(
        example_data,
        title="Claim Verification Results",
        subtitle="Automated verification on research paper abstract"
    )

    # Save example
    with open("verification_panel_example.html", "w") as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Verification Panel Example</title>
            <style>
                body {{ font-family: system-ui, -apple-system, sans-serif; margin: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Verification Panel Examples</h1>
                {html}
            </div>
        </body>
        </html>
        ''')

    print("✓ Verification panel created successfully")
    print(f"  Example saved to: verification_panel_example.html")
