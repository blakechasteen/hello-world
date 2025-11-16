"""
AutoFix Monitoring Dashboard Generator
=======================================

Generates comprehensive HTML dashboard for monitoring autofix success rates,
confidence calibration, and category performance metrics.

Features:
- Fix success rate over time (interactive line chart)
- Confidence calibration curve (predicted vs actual)
- Category breakdown with success rates
- Session comparison table
- Statistical summaries and recommendations
- Responsive design with real-time data loading

Author: mythRL Team
Date: November 16, 2025
"""

import json
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import statistics


@dataclass
class SessionStats:
    """Statistics for a single session."""
    session_id: str
    timestamp: datetime
    total_attempted: int
    total_applied: int
    success_rate: float
    avg_confidence: float
    duration_seconds: float
    categories: List[str]

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'total_attempted': self.total_attempted,
            'total_applied': self.total_applied,
            'success_rate': self.success_rate,
            'avg_confidence': self.avg_confidence,
            'duration_seconds': duration_seconds
        }


class AutoFixDashboard:
    """Generate interactive HTML dashboard for autofix metrics."""

    def __init__(self, tracking_dir: str = "./autofix_tracking"):
        self.tracking_dir = Path(tracking_dir)
        self.sessions: List[SessionStats] = []
        self.all_fixes: List[Dict] = []
        self.category_stats: Dict[str, Dict] = {}

    def load_data(self) -> bool:
        """Load tracking data from JSON files."""
        sessions_file = self.tracking_dir / "all_sessions.json"

        if not sessions_file.exists():
            print(f"⚠ Warning: {sessions_file} not found. Using sample data.")
            self._generate_sample_data()
            return True

        try:
            with open(sessions_file, 'r') as f:
                data = json.load(f)

            # Parse sessions
            for session_data in data.get('sessions', []):
                if session_data.get('total_fixes_attempted', 0) == 0:
                    # Skip empty sessions
                    continue

                try:
                    start_time = datetime.fromisoformat(session_data['start_time'])
                except:
                    start_time = datetime.now()

                try:
                    end_time = datetime.fromisoformat(session_data['end_time'])
                except:
                    end_time = start_time + timedelta(seconds=1)

                duration = (end_time - start_time).total_seconds()

                stats = SessionStats(
                    session_id=session_data['session_id'],
                    timestamp=start_time,
                    total_attempted=session_data.get('total_fixes_attempted', 0),
                    total_applied=session_data.get('total_fixes_applied', 0),
                    success_rate=session_data.get('success_rate', 0.0),
                    avg_confidence=session_data.get('avg_confidence', 0.0),
                    duration_seconds=duration,
                    categories=session_data.get('categories', [])
                )
                self.sessions.append(stats)

            # If no valid sessions, generate sample data
            if not self.sessions:
                print("⚠ Warning: No valid sessions found. Using sample data.")
                self._generate_sample_data()

            return True

        except Exception as e:
            print(f"⚠ Warning: Failed to load data: {e}. Using sample data.")
            self._generate_sample_data()
            return True

    def _generate_sample_data(self):
        """Generate synthetic sample data for demonstration."""
        categories_list = ['dead_code', 'hardcoded_values', 'missing_docstrings', 'incomplete']

        # Generate 30 days of data with realistic patterns
        for day_offset in range(30):
            date = datetime.now() - timedelta(days=30-day_offset)

            # Add some variation to success rates (improving trend)
            base_success = 0.65 + (day_offset / 30) * 0.25
            success_rate = max(0.5, min(0.95, base_success + (10 - abs(15 - day_offset)) * 0.02))

            # Confidence tends to correlate with success
            avg_confidence = 0.75 + (success_rate - 0.7) * 0.2
            avg_confidence = max(0.5, min(0.95, avg_confidence))

            # Attempted fixes vary
            attempted = 15 + (day_offset % 7) * 5
            applied = int(attempted * success_rate)

            duration = 30 + (attempted * 2.5)

            session = SessionStats(
                session_id=f"session_{date.strftime('%Y%m%d_%H%M%S')}",
                timestamp=date,
                total_attempted=attempted,
                total_applied=applied,
                success_rate=success_rate,
                avg_confidence=avg_confidence,
                duration_seconds=duration,
                categories=categories_list
            )
            self.sessions.append(session)

    def compute_calibration_curve(self) -> Tuple[List[float], List[float]]:
        """
        Compute confidence calibration curve.

        Returns:
            (predicted_confidences, actual_success_rates)

        Bins confidence scores and computes actual success rate per bin.
        """
        if not self.sessions:
            return [], []

        # Create bins for confidence scores
        bins = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]

        calibration_data = {}
        for bin_start, bin_end in bins:
            calibration_data[f"{bin_start:.1f}-{bin_end:.1f}"] = {
                'confidence': (bin_start + bin_end) / 2,
                'sessions': [],
                'success_rates': []
            }

        # Assign sessions to bins based on average confidence
        for session in self.sessions:
            conf = session.avg_confidence
            for bin_start, bin_end in bins:
                if bin_start <= conf < bin_end:
                    bin_key = f"{bin_start:.1f}-{bin_end:.1f}"
                    calibration_data[bin_key]['sessions'].append(session)
                    calibration_data[bin_key]['success_rates'].append(session.success_rate)
                    break

        # Compute actual success rates per bin
        predicted = []
        actual = []

        for bin_key in sorted(calibration_data.keys()):
            data = calibration_data[bin_key]
            if data['success_rates']:
                predicted.append(data['confidence'])
                actual.append(statistics.mean(data['success_rates']))

        return predicted, actual

    def compute_category_stats(self) -> Dict[str, Dict]:
        """Compute statistics by category."""
        stats = {}

        all_categories = set()
        for session in self.sessions:
            all_categories.update(session.categories)

        for category in sorted(all_categories):
            stats[category] = {
                'total_sessions': 0,
                'avg_success_rate': 0.0,
                'avg_confidence': 0.0,
                'total_attempted': 0,
                'total_applied': 0
            }

        # Aggregate statistics
        for session in self.sessions:
            for category in session.categories:
                cat_stats = stats[category]
                cat_stats['total_sessions'] += 1
                cat_stats['total_attempted'] += session.total_attempted
                cat_stats['total_applied'] += session.total_applied
                cat_stats['avg_success_rate'] += session.success_rate
                cat_stats['avg_confidence'] += session.avg_confidence

        # Normalize averages
        for category in stats:
            cat_stats = stats[category]
            if cat_stats['total_sessions'] > 0:
                cat_stats['avg_success_rate'] /= cat_stats['total_sessions']
                cat_stats['avg_confidence'] /= cat_stats['total_sessions']
            if cat_stats['total_attempted'] > 0:
                cat_stats['actual_success_rate'] = (
                    cat_stats['total_applied'] / cat_stats['total_attempted']
                )
            else:
                cat_stats['actual_success_rate'] = 0.0

        return stats

    def generate_html(self, output_file: str = "dashboard.html") -> str:
        """Generate comprehensive HTML dashboard."""

        # Compute statistics
        self.category_stats = self.compute_category_stats()
        predicted_conf, actual_success = self.compute_calibration_curve()

        # Compute overall metrics
        if self.sessions:
            overall_success = statistics.mean(s.success_rate for s in self.sessions)
            overall_confidence = statistics.mean(s.avg_confidence for s in self.sessions)
            total_fixed = sum(s.total_applied for s in self.sessions)
            total_attempted = sum(s.total_attempted for s in self.sessions)
            avg_duration = statistics.mean(s.duration_seconds for s in self.sessions)
        else:
            overall_success = 0.0
            overall_confidence = 0.0
            total_fixed = 0
            total_attempted = 0
            avg_duration = 0.0

        # Prepare data for charts
        timestamps = [s.timestamp.isoformat() for s in self.sessions]
        success_rates = [s.success_rate * 100 for s in self.sessions]
        confidences = [s.avg_confidence * 100 for s in self.sessions]
        attempted = [s.total_attempted for s in self.sessions]
        applied = [s.total_applied for s in self.sessions]

        # Calibration curve
        pred_conf_pct = [c * 100 for c in predicted_conf]
        actual_success_pct = [s * 100 for s in actual_success]

        # Category data
        categories = sorted(self.category_stats.keys())
        cat_success_rates = [self.category_stats[c]['actual_success_rate'] * 100 for c in categories]
        cat_confidences = [self.category_stats[c]['avg_confidence'] * 100 for c in categories]
        cat_attempted = [self.category_stats[c]['total_attempted'] for c in categories]

        # Build HTML
        html = self._build_html_template(
            overall_success, overall_confidence, total_fixed, total_attempted, avg_duration,
            timestamps, success_rates, confidences, attempted, applied,
            pred_conf_pct, actual_success_pct,
            categories, cat_success_rates, cat_confidences, cat_attempted
        )

        # Save to file
        output_path = self.tracking_dir / output_file
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"✓ Dashboard generated: {output_path}")
        return str(output_path)

    def _build_html_template(
        self,
        overall_success: float,
        overall_confidence: float,
        total_fixed: int,
        total_attempted: int,
        avg_duration: float,
        timestamps: List[str],
        success_rates: List[float],
        confidences: List[float],
        attempted: List[int],
        applied: List[int],
        pred_conf: List[float],
        actual_success: List[float],
        categories: List[str],
        cat_success_rates: List[float],
        cat_confidences: List[float],
        cat_attempted: List[int]
    ) -> str:
        """Build HTML dashboard template."""

        # Prepare data for JSON serialization
        timestamps_json = json.dumps(timestamps)
        success_rates_json = json.dumps(success_rates)
        confidences_json = json.dumps(confidences)
        attempted_json = json.dumps(attempted)
        applied_json = json.dumps(applied)
        pred_conf_json = json.dumps(pred_conf)
        actual_success_json = json.dumps(actual_success)
        categories_json = json.dumps(categories)
        cat_success_json = json.dumps(cat_success_rates)
        cat_conf_json = json.dumps(cat_confidences)
        cat_att_json = json.dumps(cat_attempted)

        # Generate session table rows
        session_rows = ""
        for session in sorted(self.sessions, key=lambda s: s.timestamp, reverse=True)[:20]:
            status_color = "green" if session.success_rate > 0.8 else "orange" if session.success_rate > 0.6 else "red"
            session_rows += f"""
            <tr>
                <td>{session.session_id}</td>
                <td>{session.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                <td>{session.total_attempted}</td>
                <td>{session.total_applied}</td>
                <td><span style="color: {status_color}; font-weight: bold;">{session.success_rate*100:.1f}%</span></td>
                <td>{session.avg_confidence*100:.1f}%</td>
                <td>{session.duration_seconds:.1f}s</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoFix Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            border-left: 5px solid #2196F3;
        }}

        h1 {{
            font-size: 2.5em;
            color: #2196F3;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-top: 4px solid #2196F3;
        }}

        .metric-card.success {{
            border-top-color: #4CAF50;
        }}

        .metric-card.warning {{
            border-top-color: #FF9800;
        }}

        .metric-label {{
            font-size: 0.9em;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #2196F3;
        }}

        .metric-value.success {{
            color: #4CAF50;
        }}

        .metric-value.warning {{
            color: #FF9800;
        }}

        .metric-value.danger {{
            color: #f44336;
        }}

        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }}

        .chart {{
            min-height: 400px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        th {{
            background: #f5f5f5;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #2196F3;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}

        tr:hover {{
            background: #f9f9f9;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.9em;
            margin-top: 40px;
        }}

        .insights {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #2196F3;
        }}

        .insights-title {{
            font-weight: 600;
            color: #1976d2;
            margin-bottom: 8px;
        }}

        .insights-text {{
            color: #424242;
            font-size: 0.95em;
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}

        @media (max-width: 1200px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AutoFix Monitoring Dashboard</h1>
            <p class="subtitle">Real-time tracking of autofix success rates, confidence calibration, and category performance</p>
            <p style="margin-top: 10px; color: #999; font-size: 0.9em;">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="metrics-grid">
            <div class="metric-card success">
                <div class="metric-label">Overall Success Rate</div>
                <div class="metric-value success">{overall_success*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Confidence</div>
                <div class="metric-value">{overall_confidence*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Fixes Applied</div>
                <div class="metric-value">{total_fixed}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Attempted</div>
                <div class="metric-value">{total_attempted}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Duration/Session</div>
                <div class="metric-value">{avg_duration:.1f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sessions Tracked</div>
                <div class="metric-value">{len(self.sessions)}</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="chart-container">
                <div class="chart-title">Fix Success Rate Over Time</div>
                <div id="successChart" class="chart"></div>
                <div class="insights">
                    <div class="insights-title">📊 Trend Analysis</div>
                    <div class="insights-text">
                        Success rate shows the percentage of fixes applied relative to attempted.
                        Upward trend indicates improving fix quality or better confidence calibration.
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">Average Confidence Over Time</div>
                <div id="confidenceChart" class="chart"></div>
                <div class="insights">
                    <div class="insights-title">🎯 Confidence Tracking</div>
                    <div class="insights-text">
                        Average confidence reflects how certain the system is about its fixes.
                        Higher confidence + higher success = good calibration.
                    </div>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Confidence Calibration Curve</div>
            <div id="calibrationChart" class="chart"></div>
            <div class="insights">
                <div class="insights-title">📈 Perfect Calibration</div>
                <div class="insights-text">
                    A perfectly calibrated model would show actual success rate = predicted confidence.
                    Points above the diagonal indicate overconfidence. Points below indicate underconfidence.
                    The closer to the diagonal, the better the calibration.
                </div>
            </div>
        </div>

        <div class="grid-2">
            <div class="chart-container">
                <div class="chart-title">Success Rate by Category</div>
                <div id="categoryChart" class="chart"></div>
                <div class="insights">
                    <div class="insights-title">🏷️ Category Performance</div>
                    <div class="insights-text">
                        Different issue categories may have different fix success rates.
                        Use this to identify which categories need improvement.
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">Attempted Fixes by Category</div>
                <div id="categoryVolumeChart" class="chart"></div>
                <div class="insights">
                    <div class="insights-title">📊 Coverage</div>
                    <div class="insights-text">
                        Shows the volume of fixes attempted per category.
                        Higher volume indicates more frequent issues in that category.
                    </div>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Session Comparison (Last 20 Sessions)</div>
            <table>
                <thead>
                    <tr>
                        <th>Session ID</th>
                        <th>Timestamp</th>
                        <th>Attempted</th>
                        <th>Applied</th>
                        <th>Success Rate</th>
                        <th>Avg Confidence</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {session_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>AutoFix Monitoring Dashboard v1.0 | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | mythRL Team</p>
        </div>
    </div>

    <script>
        // Data
        const timestamps = {timestamps_json};
        const successRates = {success_rates_json};
        const confidences = {confidences_json};
        const attempted = {attempted_json};
        const applied = {applied_json};
        const predConfidence = {pred_conf_json};
        const actualSuccess = {actual_success_json};
        const categories = {categories_json};
        const catSuccessRates = {cat_success_json};
        const catConfidences = {cat_conf_json};
        const catAttempted = {cat_att_json};

        // Chart 1: Success Rate Over Time
        const trace1 = {{
            x: timestamps,
            y: successRates,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Success Rate (%)',
            line: {{color: '#4CAF50', width: 2}},
            fill: 'tozeroy',
            fillcolor: 'rgba(76, 175, 80, 0.2)'
        }};

        const layout1 = {{
            title: '',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Success Rate (%)', range: [0, 100]}},
            hovermode: 'x unified',
            margin: {{l: 60, r: 40, t: 40, b: 60}}
        }};

        Plotly.newPlot('successChart', [trace1], layout1, {{responsive: true}});

        // Chart 2: Confidence Over Time
        const trace2 = {{
            x: timestamps,
            y: confidences,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Avg Confidence (%)',
            line: {{color: '#2196F3', width: 2}},
            fill: 'tozeroy',
            fillcolor: 'rgba(33, 150, 243, 0.2)'
        }};

        const layout2 = {{
            title: '',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Confidence (%)', range: [0, 100]}},
            hovermode: 'x unified',
            margin: {{l: 60, r: 40, t: 40, b: 60}}
        }};

        Plotly.newPlot('confidenceChart', [trace2], layout2, {{responsive: true}});

        // Chart 3: Calibration Curve
        const traceCalib = {{
            x: predConfidence,
            y: actualSuccess,
            type: 'scatter',
            mode: 'markers+lines',
            name: 'Actual Success Rate',
            marker: {{size: 8, color: '#FF9800'}},
            line: {{color: '#FF9800', width: 2}}
        }};

        const traceDiagonal = {{
            x: [0, 100],
            y: [0, 100],
            type: 'scatter',
            mode: 'lines',
            name: 'Perfect Calibration',
            line: {{color: '#999', width: 1, dash: 'dash'}},
            hoverinfo: 'skip'
        }};

        const layoutCalib = {{
            title: '',
            xaxis: {{title: 'Predicted Confidence (%)', range: [0, 100]}},
            yaxis: {{title: 'Actual Success Rate (%)', range: [0, 100]}},
            hovermode: 'closest',
            margin: {{l: 60, r: 40, t: 40, b: 60}},
            shapes: [
                {{
                    type: 'rect',
                    x0: 0, x1: 100,
                    y0: 0, y1: 100,
                    fillcolor: 'rgba(0, 0, 0, 0)',
                    line: {{color: '#999', width: 1, dash: 'dash'}}
                }}
            ]
        }};

        Plotly.newPlot('calibrationChart', [traceCalib, traceDiagonal], layoutCalib, {{responsive: true}});

        // Chart 4: Category Success Rates
        const traceCatSuccess = {{
            x: categories,
            y: catSuccessRates,
            type: 'bar',
            name: 'Success Rate',
            marker: {{color: '#4CAF50'}}
        }};

        const layoutCatSuccess = {{
            title: '',
            xaxis: {{title: 'Category'}},
            yaxis: {{title: 'Success Rate (%)', range: [0, 100]}},
            hovermode: 'x',
            margin: {{l: 60, r: 40, t: 40, b: 60}}
        }};

        Plotly.newPlot('categoryChart', [traceCatSuccess], layoutCatSuccess, {{responsive: true}});

        // Chart 5: Category Volume
        const traceCatVolume = {{
            x: categories,
            y: catAttempted,
            type: 'bar',
            name: 'Attempted Fixes',
            marker: {{color: '#2196F3'}}
        }};

        const layoutCatVolume = {{
            title: '',
            xaxis: {{title: 'Category'}},
            yaxis: {{title: 'Number of Fixes'}},
            hovermode: 'x',
            margin: {{l: 60, r: 40, t: 40, b: 60}}
        }};

        Plotly.newPlot('categoryVolumeChart', [traceCatVolume], layoutCatVolume, {{responsive: true}});
    </script>
</body>
</html>
"""
        return html


def main():
    """Generate dashboard from tracking data."""
    print("\n" + "="*70)
    print("AutoFix Monitoring Dashboard Generator")
    print("="*70 + "\n")

    # Create dashboard
    dashboard = AutoFixDashboard()

    print("📊 Loading tracking data...")
    dashboard.load_data()
    print(f"✓ Loaded {len(dashboard.sessions)} sessions\n")

    print("📈 Generating dashboard...")
    output_file = dashboard.generate_html("dashboard.html")

    print("\n" + "="*70)
    print("Dashboard Generation Complete!")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"\nKey Metrics:")
    if dashboard.sessions:
        overall_success = statistics.mean(s.success_rate for s in dashboard.sessions)
        overall_confidence = statistics.mean(s.avg_confidence for s in dashboard.sessions)
        total_fixed = sum(s.total_applied for s in dashboard.sessions)
        total_attempted = sum(s.total_attempted for s in dashboard.sessions)

        print(f"  • Overall Success Rate: {overall_success*100:.1f}%")
        print(f"  • Average Confidence: {overall_confidence*100:.1f}%")
        print(f"  • Total Fixes Applied: {total_fixed}/{total_attempted}")
        print(f"  • Sessions Tracked: {len(dashboard.sessions)}")

    print("\n📖 Dashboard Features:")
    print("  • Real-time success rate tracking over time")
    print("  • Confidence calibration curve (predicted vs actual)")
    print("  • Category breakdown with success rates")
    print("  • Session comparison table")
    print("  • Statistical summaries and insights")
    print("  • Responsive design for mobile/tablet")
    print("\n✅ Open the dashboard in your browser to explore!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
