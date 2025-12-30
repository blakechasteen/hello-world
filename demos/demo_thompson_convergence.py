"""
Thompson Sampling Convergence Visualization

Tracks and visualizes how Thompson Sampling priors evolve over pulls,
showing convergence behavior, exploration/exploitation balance, and
uncertainty reduction over time.

Outputs: demos/output/thompson_convergence.html
"""

import asyncio
import random
import sys
import os
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, '.')

from HoloLoom.policy.unified import create_policy, BanditStrategy


def generate_html_visualization(
    history: List[Dict],
    tools: List[str],
    final_stats: Dict,
    total_pulls: int
) -> str:
    """Generate Tufte-style HTML visualization of Thompson Sampling convergence."""

    # Calculate chart dimensions
    width = 800
    height_per_chart = 200
    margin = 40
    chart_width = width - 2 * margin

    # Generate SVG for expected value convergence
    def generate_expected_value_chart() -> str:
        lines = []
        colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea']  # Blue, Red, Green, Purple

        for tool_idx, tool in enumerate(tools):
            points = []
            for i, h in enumerate(history):
                x = margin + (i / max(len(history) - 1, 1)) * chart_width
                stats = h['stats'][tool_idx]
                a = stats['success'] + 1
                b = stats['fail'] + 1
                expected = a / (a + b)
                y = height_per_chart - margin - (expected * (height_per_chart - 2 * margin))
                points.append(f"{x:.1f},{y:.1f}")

            if points:
                lines.append(f'<polyline points="{" ".join(points)}" '
                           f'fill="none" stroke="{colors[tool_idx % len(colors)]}" '
                           f'stroke-width="2" opacity="0.8"/>')

        # Add legend
        legend = []
        for i, tool in enumerate(tools):
            legend.append(f'<text x="{margin + i * 120}" y="20" '
                         f'fill="{colors[i % len(colors)]}" font-size="12" font-family="monospace">'
                         f'{tool}</text>')

        # Add axes
        axes = f'''
        <line x1="{margin}" y1="{height_per_chart - margin}"
              x2="{width - margin}" y2="{height_per_chart - margin}"
              stroke="#666" stroke-width="1"/>
        <line x1="{margin}" y1="{margin}"
              x2="{margin}" y2="{height_per_chart - margin}"
              stroke="#666" stroke-width="1"/>
        <text x="{width/2}" y="{height_per_chart - 5}"
              text-anchor="middle" font-size="11" fill="#666">Pulls</text>
        <text x="15" y="{height_per_chart/2}"
              text-anchor="middle" font-size="11" fill="#666"
              transform="rotate(-90, 15, {height_per_chart/2})">E[X]</text>
        '''

        # Y-axis labels
        for val in [0, 0.5, 1.0]:
            y = height_per_chart - margin - (val * (height_per_chart - 2 * margin))
            axes += f'<text x="{margin - 5}" y="{y + 4}" text-anchor="end" font-size="10" fill="#666">{val}</text>'

        return f'''
        <svg width="{width}" height="{height_per_chart}" style="background: #fafafa; border-radius: 4px;">
            <text x="{width/2}" y="35" text-anchor="middle" font-size="14" font-weight="bold">
                Expected Value Convergence
            </text>
            {axes}
            {"".join(lines)}
            {"".join(legend)}
        </svg>
        '''

    # Generate SVG for uncertainty reduction
    def generate_uncertainty_chart() -> str:
        lines = []
        colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea']

        for tool_idx, tool in enumerate(tools):
            points = []
            for i, h in enumerate(history):
                x = margin + (i / max(len(history) - 1, 1)) * chart_width
                stats = h['stats'][tool_idx]
                a = stats['success'] + 1
                b = stats['fail'] + 1
                variance = (a * b) / ((a + b)**2 * (a + b + 1))
                # Scale variance (typically 0-0.1 range)
                y = height_per_chart - margin - (min(variance * 10, 1.0) * (height_per_chart - 2 * margin))
                points.append(f"{x:.1f},{y:.1f}")

            if points:
                lines.append(f'<polyline points="{" ".join(points)}" '
                           f'fill="none" stroke="{colors[tool_idx % len(colors)]}" '
                           f'stroke-width="2" opacity="0.8"/>')

        axes = f'''
        <line x1="{margin}" y1="{height_per_chart - margin}"
              x2="{width - margin}" y2="{height_per_chart - margin}"
              stroke="#666" stroke-width="1"/>
        <line x1="{margin}" y1="{margin}"
              x2="{margin}" y2="{height_per_chart - margin}"
              stroke="#666" stroke-width="1"/>
        <text x="{width/2}" y="{height_per_chart - 5}"
              text-anchor="middle" font-size="11" fill="#666">Pulls</text>
        <text x="15" y="{height_per_chart/2}"
              text-anchor="middle" font-size="11" fill="#666"
              transform="rotate(-90, 15, {height_per_chart/2})">Variance</text>
        '''

        return f'''
        <svg width="{width}" height="{height_per_chart}" style="background: #fafafa; border-radius: 4px;">
            <text x="{width/2}" y="35" text-anchor="middle" font-size="14" font-weight="bold">
                Uncertainty Reduction Over Time
            </text>
            {axes}
            {"".join(lines)}
        </svg>
        '''

    # Generate exploration/exploitation ratio chart
    def generate_explore_exploit_chart() -> str:
        explore_points = []
        exploit_points = []

        running_explore = 0
        running_exploit = 0

        for i, h in enumerate(history):
            x = margin + (i / max(len(history) - 1, 1)) * chart_width
            if h.get('was_exploration', False):
                running_explore += 1
            else:
                running_exploit += 1

            total = running_explore + running_exploit
            explore_ratio = running_explore / total if total > 0 else 0.5

            y_explore = height_per_chart - margin - (explore_ratio * (height_per_chart - 2 * margin))
            explore_points.append(f"{x:.1f},{y_explore:.1f}")

        axes = f'''
        <line x1="{margin}" y1="{height_per_chart - margin}"
              x2="{width - margin}" y2="{height_per_chart - margin}"
              stroke="#666" stroke-width="1"/>
        <line x1="{margin}" y1="{margin}"
              x2="{margin}" y2="{height_per_chart - margin}"
              stroke="#666" stroke-width="1"/>
        <text x="{width/2}" y="{height_per_chart - 5}"
              text-anchor="middle" font-size="11" fill="#666">Pulls</text>
        '''

        # Add 50% reference line
        y_50 = height_per_chart - margin - (0.5 * (height_per_chart - 2 * margin))
        ref_line = f'<line x1="{margin}" y1="{y_50}" x2="{width - margin}" y2="{y_50}" stroke="#ccc" stroke-dasharray="5,5"/>'

        return f'''
        <svg width="{width}" height="{height_per_chart}" style="background: #fafafa; border-radius: 4px;">
            <text x="{width/2}" y="35" text-anchor="middle" font-size="14" font-weight="bold">
                Exploration Ratio Over Time
            </text>
            {axes}
            {ref_line}
            <polyline points="{" ".join(explore_points)}"
                     fill="none" stroke="#f59e0b" stroke-width="2.5"/>
            <text x="{width - margin - 80}" y="55" fill="#f59e0b" font-size="11">Exploration %</text>
        </svg>
        '''

    # Generate final stats table
    def generate_stats_table() -> str:
        rows = []
        for i, tool in enumerate(tools):
            stats = final_stats[i]
            a = stats['success'] + 1
            b = stats['fail'] + 1
            expected = a / (a + b)
            variance = (a * b) / ((a + b)**2 * (a + b + 1))
            pulls = int(stats['pulls'])

            # Color code by expected value
            if expected >= 0.7:
                color = '#16a34a'
            elif expected >= 0.5:
                color = '#ca8a04'
            else:
                color = '#dc2626'

            rows.append(f'''
            <tr>
                <td style="font-weight: bold;">{tool}</td>
                <td style="text-align: right;">{a:.2f}</td>
                <td style="text-align: right;">{b:.2f}</td>
                <td style="text-align: right;">{pulls}</td>
                <td style="text-align: right; color: {color}; font-weight: bold;">{expected:.3f}</td>
                <td style="text-align: right;">{variance:.6f}</td>
            </tr>
            ''')

        return f'''
        <table style="width: 100%; border-collapse: collapse; font-family: monospace; font-size: 13px;">
            <thead>
                <tr style="background: #f3f4f6; border-bottom: 2px solid #e5e7eb;">
                    <th style="text-align: left; padding: 8px;">Tool</th>
                    <th style="text-align: right; padding: 8px;">Alpha</th>
                    <th style="text-align: right; padding: 8px;">Beta</th>
                    <th style="text-align: right; padding: 8px;">Pulls</th>
                    <th style="text-align: right; padding: 8px;">E[X]</th>
                    <th style="text-align: right; padding: 8px;">Variance</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        '''

    # Calculate convergence status
    avg_variance = sum(
        (final_stats[i]['success'] + 1) * (final_stats[i]['fail'] + 1) /
        ((final_stats[i]['success'] + final_stats[i]['fail'] + 2)**2 *
         (final_stats[i]['success'] + final_stats[i]['fail'] + 3))
        for i in range(len(tools))
    ) / len(tools)

    if avg_variance < 0.005:
        status = ("CONVERGED", "#16a34a", "Bandit has stable, confident estimates")
    elif avg_variance < 0.02:
        status = ("CONVERGING", "#ca8a04", "Estimates becoming stable, some uncertainty remains")
    else:
        status = ("LEARNING", "#dc2626", "More exploration needed for stable estimates")

    # Generate full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thompson Sampling Convergence</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
            color: #1f2937;
        }}
        h1 {{
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #6b7280;
            margin-bottom: 20px;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stats-section {{
            margin: 30px 0;
        }}
        .insight {{
            background: #f0f9ff;
            border-left: 4px solid #0284c7;
            padding: 12px 16px;
            margin: 15px 0;
            font-size: 14px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            color: #9ca3af;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>Thompson Sampling Convergence</h1>
    <p class="subtitle">HoloLoom Policy Engine - {total_pulls} pulls | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

    <div class="status-badge" style="background: {status[1]}20; color: {status[1]};">
        {status[0]}: {status[2]}
    </div>

    <div class="chart-container">
        {generate_expected_value_chart()}
    </div>

    <div class="insight">
        <strong>Expected Value (E[X])</strong> shows the probability of success for each tool.
        Higher values indicate the tool is more likely to succeed. Watch how the lines converge
        as the bandit learns which tools work best.
    </div>

    <div class="chart-container">
        {generate_uncertainty_chart()}
    </div>

    <div class="insight">
        <strong>Variance</strong> measures uncertainty in our estimates. As we gather more data,
        variance decreases (lines trend downward), meaning we're more confident in our estimates.
        Target: variance &lt; 0.01 for stable decisions.
    </div>

    <div class="chart-container">
        {generate_explore_exploit_chart()}
    </div>

    <div class="insight">
        <strong>Exploration Ratio</strong> shows the balance between trying new tools (exploration)
        and using the best-known tool (exploitation). Early high exploration is healthy -
        it should decrease as the bandit converges to the best option.
    </div>

    <div class="stats-section">
        <h2>Final Statistics</h2>
        {generate_stats_table()}
    </div>

    <div class="insight">
        <strong>Alpha/Beta</strong> are the Beta distribution parameters. Alpha ~ successes + prior,
        Beta ~ failures + prior. The ratio alpha/(alpha+beta) gives the expected success rate.
    </div>

    <div class="footer">
        Generated by HoloLoom Thompson Sampling Advisor |
        Avg Variance: {avg_variance:.6f} |
        Convergence Threshold: 0.01
    </div>
</body>
</html>
'''
    return html


async def run_convergence_experiment(num_pulls: int = 150):
    """Run Thompson Sampling experiment and track convergence."""

    print('=' * 70)
    print('THOMPSON SAMPLING CONVERGENCE VISUALIZATION')
    print('=' * 70)
    print()

    # Create policy
    policy = create_policy(
        mem_dim=384,
        emb=None,
        scales=[96, 192, 384],
        bandit_strategy=BanditStrategy.PURE_THOMPSON,
        epsilon=0.1
    )

    tools = policy.core.tools
    print(f'Policy: {len(tools)} tools - {tools}')
    print(f'Running {num_pulls} pulls with history tracking...')
    print()

    # Query templates with expected outcomes
    queries = [
        ('Simple factual', 0.95, 'answer'),
        ('Define concept', 0.92, 'answer'),
        ('What is X?', 0.94, 'answer'),
        ('Research topic', 0.78, 'search'),
        ('Compare A vs B', 0.80, 'search'),
        ('Analyze tradeoffs', 0.75, 'search'),
        ('Calculate value', 0.88, 'calc'),
        ('Math problem', 0.90, 'calc'),
        ('Write note', 0.85, 'notion_write'),
        ('Save to Notion', 0.82, 'notion_write'),
    ]

    # Track history
    history = []
    decisions = {tool: 0 for tool in tools}

    for pull in range(num_pulls):
        # Pick random query
        query_type, expected_conf, expected_tool = random.choice(queries)

        # Get bandit decision
        probs = [1.0 / len(tools)] * len(tools)
        selected_idx, _ = policy.bandit.select_with_strategy(probs)
        selected_tool = tools[selected_idx]

        # Simulate outcome
        if selected_tool == expected_tool:
            success = random.random() < expected_conf
            confidence = expected_conf if success else expected_conf * 0.5
        else:
            success = random.random() < 0.35
            confidence = 0.45 if success else 0.25

        # Update bandit
        reward = confidence if success else 0.0
        policy.bandit.update(selected_idx, reward)

        # Track decision
        decisions[selected_tool] += 1

        # Determine if exploration
        current_stats = policy.bandit.get_stats()
        best_idx = max(
            range(len(tools)),
            key=lambda x: (current_stats[x]['success'] + 1) /
                         (current_stats[x]['success'] + current_stats[x]['fail'] + 2)
        )
        was_exploration = selected_idx != best_idx

        # Record history
        history.append({
            'pull': pull + 1,
            'tool': selected_tool,
            'success': success,
            'confidence': confidence,
            'was_exploration': was_exploration,
            'stats': {i: dict(current_stats[i]) for i in range(len(tools))}
        })

        # Progress
        if (pull + 1) % 25 == 0:
            explore_count = sum(1 for h in history if h['was_exploration'])
            print(f'  Pull {pull + 1:3d}: Explore={explore_count}/{pull+1} ({explore_count/(pull+1)*100:.1f}%)')

    print()

    # Get final stats
    final_stats = policy.bandit.get_stats()

    # Generate HTML
    html = generate_html_visualization(history, tools, final_stats, num_pulls)

    # Save
    output_dir = 'demos/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'thompson_convergence.html')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'Visualization saved to: {output_path}')
    print()

    # Print summary
    print('FINAL RESULTS:')
    print('-' * 50)
    for i, tool in enumerate(tools):
        stats = final_stats[i]
        a = stats['success'] + 1
        b = stats['fail'] + 1
        expected = a / (a + b)
        print(f'  {tool:<12}: E[X]={expected:.3f}, pulls={decisions[tool]}')

    return output_path


if __name__ == '__main__':
    path = asyncio.run(run_convergence_experiment(150))
    print()
    print(f'Open in browser: file:///{os.path.abspath(path).replace(chr(92), "/")}')
