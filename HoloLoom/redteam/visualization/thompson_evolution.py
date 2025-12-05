"""Thompson Sampling evolution visualization for CARTS red team analytics.

Tracks how Thompson Sampling strategies (alpha/beta distributions) evolve over time,
enabling analysis of:
- Which strategies converge fastest
- Which achieve highest expected reward
- When convergence occurs (stabilization point)
- Strategy dominance patterns

Follows Edward Tufte's visualization principles:
- Maximize information density
- Minimize decoration (chartjunk)
- Show convergence patterns clearly
- Enable strategy comparison

Philosophy: "Show the learning evolution."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import math


@dataclass
class ArmSnapshot:
    """Thompson Sampling arm state at a point in time.

    Represents the Bayesian posterior Beta(α, β) for a single strategy.
    """

    strategy_name: str  # e.g., "prompt_injection", "jailbreak"
    alpha: float  # Beta distribution shape parameter (successes + prior)
    beta: float  # Beta distribution shape parameter (failures + prior)
    expected_reward: float  # α / (α + β), the posterior mean
    pull_count: int  # Total times this arm was selected
    timestamp: Optional[float] = None  # Unix timestamp
    metadata: Optional[Dict[str, Any]] = None  # Extra context

    def __post_init__(self):
        """Validate data consistency."""
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"alpha and beta must be > 0, got alpha={self.alpha}, beta={self.beta}"
            )
        if not (0.0 <= self.expected_reward <= 1.0):
            raise ValueError(
                f"expected_reward must be 0.0-1.0, got {self.expected_reward}"
            )
        if self.pull_count < 0:
            raise ValueError(f"pull_count must be >= 0, got {self.pull_count}")
        if self.metadata is None:
            self.metadata = {}

    @property
    def variance(self) -> float:
        """Variance of Beta distribution: αβ / ((α+β)²(α+β+1))"""
        total = self.alpha + self.beta
        if total <= 1:
            return 0.25  # Maximum variance for Beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def uncertainty(self) -> float:
        """Uncertainty measure: sqrt(variance), 0.0-0.5."""
        return math.sqrt(self.variance)

    @property
    def convergence_score(self) -> float:
        """How converged is this distribution? 0.0 (wide) to 1.0 (narrow).

        Inverse of uncertainty normalized to [0, 1].
        """
        max_uncertainty = 0.5
        return 1.0 - (self.uncertainty / max_uncertainty)


@dataclass
class StrategyEvolution:
    """Complete evolution of a single strategy over time."""

    strategy_name: str
    snapshots: List[ArmSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: ArmSnapshot) -> None:
        """Add a snapshot to the evolution."""
        if snapshot.strategy_name != self.strategy_name:
            raise ValueError(
                f"Snapshot strategy '{snapshot.strategy_name}' "
                f"doesn't match evolution '{self.strategy_name}'"
            )
        self.snapshots.append(snapshot)

    @property
    def is_converged(self) -> bool:
        """Has this strategy converged? (last 3 snapshots stable)."""
        if len(self.snapshots) < 3:
            return False

        last_three = self.snapshots[-3:]
        rewards = [s.expected_reward for s in last_three]

        # Converged if variance in last 3 is < 0.01
        if len(rewards) < 2:
            return False
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)

        return variance < 0.01

    @property
    def convergence_time(self) -> Optional[float]:
        """When did this strategy converge? Returns timestamp or None."""
        if len(self.snapshots) < 3:
            return None

        # Find first index where convergence holds for remaining points
        for i in range(len(self.snapshots) - 2):
            window = self.snapshots[i : i + 3]
            rewards = [s.expected_reward for s in window]
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)

            if variance < 0.01:
                # Check if it stays converged
                remaining = self.snapshots[i + 3 :]
                if len(remaining) == 0:
                    return window[0].timestamp

                all_stable = all(
                    abs(s.expected_reward - mean_reward) < 0.05 for s in remaining
                )
                if all_stable:
                    return window[0].timestamp

        return None


class ThompsonEvolutionRenderer:
    """Render Thompson Sampling strategy evolution using Tufte principles.

    Produces pure HTML/CSS/SVG visualization showing:
    - Sparklines for each strategy's α/(α+β) evolution
    - Expected reward trajectories
    - Convergence detection and timing
    - Strategy comparison table with color coding
    - Uncertainty (confidence interval) visualization

    No external dependencies: All CSS/SVG inline.
    """

    # Color palette
    COLOR_CONVERGENT = "#2ecc71"  # Green: converged
    COLOR_EXPLORING = "#3498db"  # Blue: exploring
    COLOR_UNCERTAIN = "#f39c12"  # Orange: high uncertainty
    COLOR_EXCELLENT = "#27ae60"  # Dark green: excellent reward
    COLOR_GOOD = "#16a085"  # Teal: good reward
    COLOR_FAIR = "#f39c12"  # Orange: fair reward
    COLOR_POOR = "#e74c3c"  # Red: poor reward

    def __init__(self, width: int = 900, height: int = 300):
        """Initialize renderer.

        Args:
            width: Chart width in pixels
            height: Chart height in pixels
        """
        self.width = width
        self.height = height
        self.evolutions: Dict[str, StrategyEvolution] = {}

    def add_snapshot(self, snapshot: ArmSnapshot) -> None:
        """Add a Thompson Sampling snapshot.

        Args:
            snapshot: ArmSnapshot containing Thompson Sampling state
        """
        if snapshot.strategy_name not in self.evolutions:
            self.evolutions[snapshot.strategy_name] = StrategyEvolution(
                snapshot.strategy_name
            )
        self.evolutions[snapshot.strategy_name].add_snapshot(snapshot)

    def render_html(
        self,
        title: str = "Thompson Sampling Evolution",
        subtitle: Optional[str] = None,
    ) -> str:
        """Render complete HTML visualization.

        Args:
            title: Main title
            subtitle: Optional subtitle

        Returns:
            Complete HTML string ready for display
        """
        if not self.evolutions:
            return self._render_empty_state()

        html_parts = [
            self._render_html_header(),
            self._render_styles(),
            "<body>",
            self._render_header(title, subtitle),
            self._render_sparklines(),
            self._render_comparison_table(),
            self._render_convergence_analysis(),
            "</body>",
            "</html>",
        ]

        return "\n".join(html_parts)

    # ========== Header & Title ==========

    def _render_html_header(self) -> str:
        """Render HTML5 document header."""
        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '  <meta charset="UTF-8">\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '  <title>Thompson Sampling Evolution</title>\n'
            '</head>'
        )

    def _render_header(
        self,
        title: str,
        subtitle: Optional[str] = None,
    ) -> str:
        """Render title and subtitle."""
        html = f'<div class="header">\n  <h1>{title}</h1>'

        if subtitle:
            html += f'\n  <p class="subtitle">{subtitle}</p>'

        # Summary badges
        best_strategy = max(
            self.evolutions.values(),
            key=lambda e: e.snapshots[-1].expected_reward
            if e.snapshots
            else 0,
            default=None,
        )

        if best_strategy and best_strategy.snapshots:
            best_reward = best_strategy.snapshots[-1].expected_reward
            converged_count = sum(
                1 for e in self.evolutions.values() if e.is_converged
            )

            html += f'''
  <div class="metrics-badges">
    <span class="badge" title="Strategy with highest expected reward">
      Best: {best_strategy.strategy_name} ({best_reward:.1%})
    </span>
    <span class="badge" title="Strategies that have converged">
      Converged: {converged_count}/{len(self.evolutions)}
    </span>
    <span class="badge" title="Total strategies tracked">
      Strategies: {len(self.evolutions)}
    </span>
  </div>'''

        html += "\n</div>"
        return html

    # ========== Sparklines ==========

    def _render_sparklines(self) -> str:
        """Render sparklines for each strategy showing expected reward evolution.

        Design: Small multiples (one per strategy) with color coding by performance
        """
        html = '<div class="sparklines-section">'
        html += '<h2>Strategy Expected Reward Evolution</h2>'
        html += '<div class="sparklines-container">'

        for strategy_name in sorted(self.evolutions.keys()):
            evolution = self.evolutions[strategy_name]
            if not evolution.snapshots:
                continue

            rewards = [s.expected_reward for s in evolution.snapshots]
            final_reward = rewards[-1]
            is_converged = evolution.is_converged

            # Color code by final reward and convergence
            if final_reward > 0.7:
                bg_color = "#d5f4e6"  # Light green
                text_color = self.COLOR_EXCELLENT
            elif final_reward > 0.5:
                bg_color = "#d6eaf8"  # Light blue
                text_color = self.COLOR_GOOD
            elif final_reward > 0.3:
                bg_color = "#fdebd0"  # Light orange
                text_color = self.COLOR_FAIR
            else:
                bg_color = "#fadbd8"  # Light red
                text_color = self.COLOR_POOR

            convergence_badge = "✓ Converged" if is_converged else "⟳ Exploring"
            convergence_color = self.COLOR_CONVERGENT if is_converged else self.COLOR_EXPLORING

            sparkline_svg = self._build_sparkline(rewards, 120, 30)

            html += f'''
  <div class="sparkline-card" style="background: {bg_color};">
    <div class="sparkline-header">
      <h3>{strategy_name}</h3>
      <span class="convergence-badge" style="color: {convergence_color};">
        {convergence_badge}
      </span>
    </div>
    <div class="sparkline-chart">
      {sparkline_svg}
    </div>
    <div class="sparkline-footer">
      <span class="reward-value" style="color: {text_color};">
        {final_reward:.1%}
      </span>
      <span class="reward-label">Final Reward</span>
    </div>
  </div>'''

        html += '\n</div>\n</div>'
        return html

    def _build_sparkline(
        self,
        data: List[float],
        width: int = 120,
        height: int = 30,
    ) -> str:
        """Build inline SVG sparkline showing reward evolution.

        Tufte principle: Show trend clearly without clutter
        """
        if not data or len(data) < 2:
            return '<span style="color: #bdc3c7;">—</span>'

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val or 0.1

        x_step = (width - 4) / (len(data) - 1)
        y_scale = (height - 4) / range_val

        # Build path
        path_points = []
        for i, val in enumerate(data):
            x = 2 + i * x_step
            y = height - 2 - ((val - min_val) * y_scale)
            path_points.append(f"{x},{y}")

        path_data = "M " + " L ".join(path_points)

        # Color by trend
        trend_direction = "up" if data[-1] >= data[0] else "down"
        line_color = self.COLOR_EXPLORING if trend_direction == "up" else self.COLOR_CONVERGENT

        # Fill area under curve (low opacity)
        fill_path = f"M {path_data} L {2 + (len(data) - 1) * x_step} {height - 2} L 2 {height - 2} Z"

        svg = f'''<svg width="{width}" height="{height}" style="vertical-align: middle;">
  <defs>
    <linearGradient id="grad-{id(data)}" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:{line_color};stop-opacity:0.3" />
      <stop offset="100%" style="stop-color:{line_color};stop-opacity:0" />
    </linearGradient>
  </defs>
  <path d="{fill_path}" fill="url(#grad-{id(data)})"/>
  <path d="{path_data}" stroke="{line_color}" stroke-width="2" fill="none" stroke-linecap="round"/>
  <circle cx="2" cy="{height - 2 - ((data[0] - min_val) * y_scale)}" r="1.5" fill="{line_color}"/>
  <circle cx="{2 + (len(data) - 1) * x_step}" cy="{height - 2 - ((data[-1] - min_val) * y_scale)}" r="1.5" fill="{line_color}"/>
</svg>'''

        return svg

    # ========== Comparison Table ==========

    def _render_comparison_table(self) -> str:
        """Render table comparing all strategies.

        Design: High data density with convergence status, uncertainty, and trends
        """
        html = '<div class="comparison-section">'
        html += '<h2>Strategy Comparison</h2>'
        html += '<table class="thompson-table">'
        html += '''<thead><tr>
  <th>Strategy</th>
  <th>Expected Reward</th>
  <th>Alpha / Beta</th>
  <th>Uncertainty</th>
  <th>Pulls</th>
  <th>Status</th>
</tr></thead>
<tbody>'''

        for strategy_name in sorted(
            self.evolutions.keys(),
            key=lambda s: self.evolutions[s].snapshots[-1].expected_reward
            if self.evolutions[s].snapshots
            else 0,
            reverse=True,
        ):
            evolution = self.evolutions[strategy_name]
            if not evolution.snapshots:
                continue

            latest = evolution.snapshots[-1]
            convergence_score = latest.convergence_score

            # Color code by reward
            if latest.expected_reward > 0.7:
                reward_color = self.COLOR_EXCELLENT
            elif latest.expected_reward > 0.5:
                reward_color = self.COLOR_GOOD
            elif latest.expected_reward > 0.3:
                reward_color = self.COLOR_FAIR
            else:
                reward_color = self.COLOR_POOR

            # Uncertainty color
            uncertainty_color = (
                self.COLOR_UNCERTAIN if latest.uncertainty > 0.2 else self.COLOR_CONVERGENT
            )

            # Status badge
            if evolution.is_converged:
                status = "✓ Converged"
                status_color = self.COLOR_CONVERGENT
            else:
                status = "⟳ Exploring"
                status_color = self.COLOR_EXPLORING

            html += f'''<tr>
  <td class="strategy-name">{strategy_name}</td>
  <td class="reward" style="color: {reward_color}; font-weight: bold;">{latest.expected_reward:.1%}</td>
  <td class="parameters" title="Beta distribution shape parameters">{latest.alpha:.1f} / {latest.beta:.1f}</td>
  <td class="uncertainty" style="color: {uncertainty_color};">{latest.uncertainty:.3f}</td>
  <td class="pulls" title="Times this strategy was selected">{latest.pull_count}</td>
  <td class="status" style="color: {status_color};">{status}</td>
</tr>'''

        html += '</tbody>\n</table>\n</div>'
        return html

    # ========== Convergence Analysis ==========

    def _render_convergence_analysis(self) -> str:
        """Render convergence analysis with timing information.

        Shows which strategies converged and when
        """
        html = '<div class="convergence-section">'
        html += '<h2>Convergence Analysis</h2>'

        convergence_info = self.get_convergence_info()

        html += '<div class="convergence-info">'
        html += f'''<div class="convergence-stat">
  <span class="stat-label">Converged:</span>
  <span class="stat-value">{convergence_info['converged_count']}/{convergence_info['total_strategies']}</span>
</div>'''

        if convergence_info['fastest_strategy']:
            fastest = convergence_info['fastest_strategy']
            html += f'''<div class="convergence-stat">
  <span class="stat-label">Fastest Convergence:</span>
  <span class="stat-value">{fastest['name']}</span>
  <span class="stat-detail">{fastest['snapshots']} iterations</span>
</div>'''

        if convergence_info['best_converged']:
            best = convergence_info['best_converged']
            html += f'''<div class="convergence-stat">
  <span class="stat-label">Best Converged Reward:</span>
  <span class="stat-value" style="color: {self.COLOR_EXCELLENT};">{best['reward']:.1%}</span>
  <span class="stat-detail">{best['name']}</span>
</div>'''

        html += '</div>'

        # Convergence timeline
        converged_strategies = [
            (name, evol)
            for name, evol in self.evolutions.items()
            if evol.is_converged and evol.snapshots
        ]

        if converged_strategies:
            html += '<div class="timeline">'
            html += '<h3>Convergence Timeline</h3>'

            for name, evol in sorted(
                converged_strategies,
                key=lambda x: len(x[1].snapshots),
            ):
                snapshots = len(evol.snapshots)
                final_reward = evol.snapshots[-1].expected_reward

                html += f'''<div class="timeline-item">
  <span class="timeline-name">{name}</span>
  <div class="timeline-bar" style="width: {snapshots * 20}px; background: {self._get_reward_color(final_reward)};" title="{snapshots} iterations, {final_reward:.1%} final reward"></div>
  <span class="timeline-value">{snapshots} iterations</span>
</div>'''

            html += '</div>'

        html += '</div>'
        return html

    def _get_reward_color(self, reward: float) -> str:
        """Get color for reward level."""
        if reward > 0.7:
            return self.COLOR_EXCELLENT
        elif reward > 0.5:
            return self.COLOR_GOOD
        elif reward > 0.3:
            return self.COLOR_FAIR
        else:
            return self.COLOR_POOR

    # ========== Data Analysis ==========

    def get_convergence_info(self) -> Dict[str, Any]:
        """Analyze convergence patterns.

        Returns:
            Dictionary with convergence statistics
        """
        converged = [
            (name, evol)
            for name, evol in self.evolutions.items()
            if evol.is_converged
        ]

        fastest_strategy = None
        min_snapshots = float('inf')

        for name, evol in self.evolutions.items():
            if evol.is_converged and len(evol.snapshots) < min_snapshots:
                min_snapshots = len(evol.snapshots)
                fastest_strategy = {
                    "name": name,
                    "snapshots": len(evol.snapshots),
                    "reward": evol.snapshots[-1].expected_reward,
                }

        best_converged = None
        best_reward = -1

        for name, evol in converged:
            if evol.snapshots:
                reward = evol.snapshots[-1].expected_reward
                if reward > best_reward:
                    best_reward = reward
                    best_converged = {
                        "name": name,
                        "reward": reward,
                        "snapshots": len(evol.snapshots),
                    }

        return {
            "total_strategies": len(self.evolutions),
            "converged_count": len(converged),
            "fastest_strategy": fastest_strategy,
            "best_converged": best_converged,
            "convergence_rate": len(converged) / len(self.evolutions)
            if self.evolutions
            else 0.0,
        }

    # ========== Styles ==========

    def _render_styles(self) -> str:
        """Render inline CSS with Tufte-inspired design."""
        return '''<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.6;
  color: #34495e;
  background: #f8f9fa;
  padding: 40px 20px;
}

.header {
  max-width: 1200px;
  margin: 0 auto 40px;
  padding-bottom: 20px;
  border-bottom: 2px solid #34495e;
}

.header h1 {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 5px;
  color: #1a252f;
}

.header .subtitle {
  font-size: 14px;
  color: #7f8c8d;
  font-style: italic;
  margin-bottom: 15px;
}

.metrics-badges {
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
}

.badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: 500;
  background: #ecf0f1;
  border-left: 3px solid #95a5a6;
}

.sparklines-section,
.comparison-section,
.convergence-section {
  max-width: 1200px;
  margin: 0 auto 40px;
  background: white;
  padding: 25px;
  border: 1px solid #bdc3c7;
}

.sparklines-section h2,
.comparison-section h2,
.convergence-section h2 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #ecf0f1;
  color: #1a252f;
}

.sparklines-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
}

.sparkline-card {
  padding: 15px;
  border-radius: 4px;
  border: 1px solid #bdc3c7;
}

.sparkline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.sparkline-header h3 {
  font-size: 14px;
  font-weight: 600;
  margin: 0;
}

.convergence-badge {
  font-size: 11px;
  font-weight: 500;
}

.sparkline-footer {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.reward-value {
  font-weight: 600;
  font-family: "Courier New", monospace;
  font-size: 14px;
}

.reward-label {
  font-size: 11px;
  color: #7f8c8d;
}

.thompson-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.thompson-table td,
.thompson-table th {
  padding: 8px 12px;
  border-bottom: 1px solid #ecf0f1;
  text-align: left;
}

.thompson-table th {
  background: #f8f9fa;
  font-weight: 600;
  color: #1a252f;
  border-bottom: 2px solid #34495e;
}

.thompson-table tbody tr:hover {
  background: #f8f9fa;
}

.strategy-name {
  font-weight: 500;
  color: #34495e;
}

.reward,
.parameters,
.pulls {
  text-align: right;
  font-family: "Courier New", monospace;
}

.uncertainty {
  text-align: center;
  font-family: "Courier New", monospace;
}

.status {
  text-align: center;
  font-weight: 500;
}

.convergence-info {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 25px;
}

.convergence-stat {
  display: flex;
  flex-direction: column;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 4px;
  border-left: 3px solid #3498db;
}

.stat-label {
  font-size: 11px;
  color: #7f8c8d;
  font-weight: 600;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.stat-value {
  font-size: 18px;
  font-weight: 600;
  color: #1a252f;
  font-family: "Courier New", monospace;
}

.stat-detail {
  font-size: 11px;
  color: #95a5a6;
  margin-top: 3px;
}

.timeline {
  margin-top: 20px;
}

.timeline h3 {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #34495e;
}

.timeline-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #ecf0f1;
}

.timeline-name {
  width: 150px;
  font-weight: 500;
  font-size: 12px;
}

.timeline-bar {
  height: 20px;
  border-radius: 2px;
  min-width: 20px;
  transition: width 0.2s ease;
}

.timeline-value {
  width: 80px;
  text-align: right;
  font-size: 11px;
  font-family: "Courier New", monospace;
  color: #7f8c8d;
}

.empty-state {
  max-width: 1200px;
  margin: 40px auto;
  padding: 40px;
  text-align: center;
  background: white;
  border: 1px solid #bdc3c7;
  border-radius: 4px;
  color: #7f8c8d;
}

@media (max-width: 768px) {
  body {
    padding: 20px 10px;
  }

  .header h1 {
    font-size: 24px;
  }

  .metrics-badges {
    flex-direction: column;
  }

  .badge {
    width: 100%;
  }

  .sparklines-container {
    grid-template-columns: 1fr;
  }

  .convergence-info {
    grid-template-columns: 1fr;
  }

  .timeline-item {
    flex-wrap: wrap;
  }

  .thompson-table {
    font-size: 11px;
  }

  .thompson-table td,
  .thompson-table th {
    padding: 6px 8px;
  }
}
</style>'''

    # ========== Empty State ==========

    def _render_empty_state(self) -> str:
        """Render empty state when no data available."""
        return f'''{self._render_html_header()}
{self._render_styles()}
<body>
<div class="empty-state">
  <h1>No Data Available</h1>
  <p>Add Thompson Sampling snapshots to visualize evolution</p>
</div>
</body>
</html>'''


# ========== Convenience Function ==========


def render_thompson_evolution(
    strategies: List[str],
    snapshots_per_strategy: Dict[str, List[tuple]],
    title: str = "Thompson Sampling Evolution",
    subtitle: Optional[str] = None,
) -> str:
    """Convenience function to render Thompson Sampling evolution.

    Args:
        strategies: List of strategy names
        snapshots_per_strategy: Dict mapping strategy name to list of
            (alpha, beta, expected_reward, pull_count) tuples
        title: Visualization title
        subtitle: Optional subtitle

    Returns:
        Complete HTML string

    Example:
        >>> snapshots_per_strategy = {
        ...     "prompt_injection": [
        ...         (1.0, 1.0, 0.5, 0),
        ...         (2.0, 1.5, 0.57, 1),
        ...         (3.0, 2.0, 0.60, 2),
        ...     ],
        ...     "jailbreak": [
        ...         (1.0, 1.0, 0.5, 0),
        ...         (1.5, 2.5, 0.38, 1),
        ...         (1.5, 3.5, 0.30, 2),
        ...     ],
        ... }
        >>> html = render_thompson_evolution(
        ...     ["prompt_injection", "jailbreak"],
        ...     snapshots_per_strategy,
        ...     title="Red Team Thompson Learning"
        ... )
    """
    renderer = ThompsonEvolutionRenderer()

    for strategy in strategies:
        if strategy not in snapshots_per_strategy:
            continue

        snapshots = snapshots_per_strategy[strategy]
        for i, snapshot_tuple in enumerate(snapshots):
            alpha, beta, expected_reward, pull_count = snapshot_tuple

            snapshot = ArmSnapshot(
                strategy_name=strategy,
                alpha=alpha,
                beta=beta,
                expected_reward=expected_reward,
                pull_count=pull_count,
                timestamp=float(i),  # Use index as timestamp
            )
            renderer.add_snapshot(snapshot)

    return renderer.render_html(title=title, subtitle=subtitle)
