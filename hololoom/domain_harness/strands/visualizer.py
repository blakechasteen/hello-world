#!/usr/bin/env python3
"""
Strands Tension Visualizer
==========================

Real-time visualization of loom tension state.

Modes:
- ASCII: Terminal-based, no dependencies
- Rich: Terminal with colors and animations (requires 'rich')
- Plot: Matplotlib graph (requires 'matplotlib')
- Daemon: Background process with periodic updates

Usage:
    python visualizer.py                    # One-shot ASCII
    python visualizer.py --daemon           # Background daemon
    python visualizer.py --mode rich        # Rich terminal UI
    python visualizer.py --mode plot        # Matplotlib graph
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hololoom.domain_harness.schema import Feature, FeatureStatus
from hololoom.domain_harness.strands.tension_model import LoomState, StrandState, TensionUpdater
from hololoom.domain_harness.templates import FlatFileBackend

# ============================================================================
# ASCII Renderer
# ============================================================================

class ASCIIRenderer:
    """
    Renders loom state as ASCII art in terminal.
    """

    def __init__(self, width: int = 50):
        self.width = width

    def render_tension_bar(
        self,
        label: str,
        tension: float,
        max_tension: float = 10.0
    ) -> str:
        """Render single tension bar."""
        # Normalize to bar width
        bar_width = self.width - 15  # Leave room for label and value
        filled = int((tension / max_tension) * bar_width)
        empty = bar_width - filled

        # Choose character based on tension level
        if tension < 3:
            char = "░"
            indicator = "✓"
        elif tension < 6:
            char = "▒"
            indicator = "○"
        elif tension < 8:
            char = "▓"
            indicator = "!"
        else:
            char = "█"
            indicator = "⚠"

        bar = char * filled + "·" * empty
        return f"{label:8} {indicator} [{bar}] {tension:4.1f}"

    def render_loom(
        self,
        features: list[Feature],
        loom_state: LoomState
    ) -> str:
        """Render complete loom state."""
        lines = []

        # Header
        lines.append("╔" + "═" * (self.width + 8) + "╗")
        lines.append("║" + " LOOM TENSION STATE ".center(self.width + 8) + "║")
        lines.append("╠" + "═" * (self.width + 8) + "╣")

        # Aggregate stats
        lines.append(f"║ Total: {loom_state.total_tension:5.1f}  " +
                    f"Avg: {loom_state.average_tension:4.1f}  " +
                    f"Status: {'STABLE' if loom_state.is_stable else 'UNSTABLE':8}" +
                    " " * 5 + "║")

        if loom_state.is_critical:
            lines.append("║" + " ⚡ CRITICAL TENSION DETECTED ⚡ ".center(self.width + 8) + "║")

        lines.append("╠" + "─" * (self.width + 8) + "╣")

        # Individual strands
        for feature in features:
            strand = loom_state.strands.get(feature.id)
            if strand:
                bar = self.render_tension_bar(feature.id, strand.tension)

                # Add status indicator
                status_char = {
                    FeatureStatus.PASSING: "✓",
                    FeatureStatus.FAILING: "✗",
                    FeatureStatus.PENDING: "○",
                    FeatureStatus.IN_PROGRESS: "◐",
                    FeatureStatus.BLOCKED: "⊘"
                }.get(feature.status, "?")

                line = f"║ {status_char} {bar} ║"
                lines.append(line)

        # Footer
        lines.append("╠" + "─" * (self.width + 8) + "╣")

        # Legend
        lines.append("║ Legend: ░ Low  ▒ Medium  ▓ High  █ Critical" + " " * 10 + "║")
        lines.append("╚" + "═" * (self.width + 8) + "╝")

        return "\n".join(lines)

    def render_mini(self, loom_state: LoomState) -> str:
        """Compact one-line render."""
        if not loom_state.strands:
            return "[No strands]"

        chars = []
        for strand in sorted(loom_state.strands.values(), key=lambda s: s.feature_id):
            if strand.tension < 3:
                chars.append("░")
            elif strand.tension < 6:
                chars.append("▒")
            elif strand.tension < 8:
                chars.append("▓")
            else:
                chars.append("█")

        return f"[{''.join(chars)}] avg:{loom_state.average_tension:.1f}"


# ============================================================================
# Rich Renderer (optional)
# ============================================================================

def render_rich(features: list[Feature], loom_state: LoomState):
    """Render with rich library (if available)."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Table
    except ImportError:
        print("Install 'rich' for enhanced terminal UI: pip install rich")
        return

    console = Console()

    # Create table
    table = Table(title="🧵 Loom Tension State", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Feature", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Tension", justify="right")
    table.add_column("Bar", width=30)

    for feature in features:
        strand = loom_state.strands.get(feature.id)
        if strand:
            tension = strand.tension

            # Color based on tension
            if tension < 3:
                color = "green"
            elif tension < 6:
                color = "yellow"
            elif tension < 8:
                color = "orange1"
            else:
                color = "red"

            # Build bar
            bar_len = int(tension * 3)
            bar = "█" * bar_len + "░" * (30 - bar_len)

            status_map = {
                FeatureStatus.PASSING: "[green]✓ PASS[/]",
                FeatureStatus.FAILING: "[red]✗ FAIL[/]",
                FeatureStatus.PENDING: "[yellow]○ PEND[/]",
                FeatureStatus.IN_PROGRESS: "[blue]◐ WORK[/]",
                FeatureStatus.BLOCKED: "[red]⊘ BLOCK[/]"
            }

            table.add_row(
                feature.id,
                feature.name[:30],
                status_map.get(feature.status, "?"),
                f"[{color}]{tension:.1f}[/]",
                f"[{color}]{bar}[/]"
            )

    console.print(table)

    # Summary panel
    summary = (
        f"Total: {loom_state.total_tension:.1f}  "
        f"Average: {loom_state.average_tension:.1f}  "
        f"{'✓ STABLE' if loom_state.is_stable else '⚠ UNSTABLE'}"
    )
    style = "green" if loom_state.is_stable else "red"
    console.print(Panel(summary, title="Summary", style=style))


# ============================================================================
# Matplotlib Renderer (optional)
# ============================================================================

def render_plot(features: list[Feature], loom_state: LoomState, save_path: Path | None = None):
    """Render with matplotlib (if available)."""
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install 'matplotlib' for plot mode: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of tensions
    feature_ids = [f.id for f in features]
    tensions = [loom_state.strands.get(f.id, StrandState(f.id)).tension for f in features]

    colors = []
    for t in tensions:
        if t < 3:
            colors.append('#2ecc71')  # Green
        elif t < 6:
            colors.append('#f39c12')  # Yellow
        elif t < 8:
            colors.append('#e67e22')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    ax1.barh(feature_ids, tensions, color=colors)
    ax1.set_xlabel('Tension')
    ax1.set_title('Feature Tensions')
    ax1.set_xlim(0, 10)
    ax1.axvline(x=7, color='red', linestyle='--', alpha=0.5, label='Warning')
    ax1.legend()

    # Weave pattern visualization
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, len(features) + 1)
    ax2.set_title('Weave Pattern')

    for i, feature in enumerate(features):
        strand = loom_state.strands.get(feature.id, StrandState(feature.id))
        y = len(features) - i

        # Draw thread
        weft = strand.weft_position * 10
        color = colors[i]

        # Thread line
        ax2.plot([0, weft], [y, y], linewidth=2, color=color)
        ax2.plot([weft, 10], [y, y], linewidth=1, color=color, alpha=0.3)

        # Progress marker
        ax2.scatter([weft], [y], s=100, color=color, zorder=5)

        # Label
        ax2.text(-0.5, y, feature.id, ha='right', va='center', fontsize=8)

    ax2.set_xticks([0, 5, 10])
    ax2.set_xticklabels(['Start', 'Middle', 'Complete'])
    ax2.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ============================================================================
# Daemon Mode
# ============================================================================

def run_daemon(
    domain_memory_dir: Path,
    interval: float = 3.0,
    mode: str = "ascii"
):
    """Run as background daemon with periodic updates."""
    backend = FlatFileBackend(domain_memory_dir)
    updater = TensionUpdater()
    renderer = ASCIIRenderer()
    loom_state = LoomState()

    print("Starting Strands Visualizer Daemon")
    print(f"Domain memory: {domain_memory_dir}")
    print(f"Update interval: {interval}s")
    print(f"Mode: {mode}")
    print("-" * 40)

    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")

            # Load current state
            features = backend.load_features()
            progress = []  # Could load from progress.log

            if not features:
                print("No features found. Waiting...")
                time.sleep(interval)
                continue

            # Update tensions
            loom_state = updater.update_all(features, progress, loom_state)

            # Render
            if mode == "ascii":
                print(renderer.render_loom(features, loom_state))
            elif mode == "rich":
                render_rich(features, loom_state)
            elif mode == "mini":
                print(renderer.render_mini(loom_state))

            # Timestamp
            print(f"\nLast update: {datetime.now().strftime('%H:%M:%S')}")
            print("Press Ctrl+C to stop")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nDaemon stopped.")


# ============================================================================
# One-shot Mode
# ============================================================================

def run_once(domain_memory_dir: Path, mode: str = "ascii"):
    """Run once and exit."""
    backend = FlatFileBackend(domain_memory_dir)
    updater = TensionUpdater()
    renderer = ASCIIRenderer()

    features = backend.load_features()
    if not features:
        print("No features found.")
        return

    loom_state = updater.update_all(features, [], LoomState())

    if mode == "ascii":
        print(renderer.render_loom(features, loom_state))
    elif mode == "rich":
        render_rich(features, loom_state)
    elif mode == "plot":
        render_plot(features, loom_state)
    elif mode == "mini":
        print(renderer.render_mini(loom_state))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strands Tension Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dir', '-d',
        default='./domain_memory',
        help='Domain memory directory'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['ascii', 'rich', 'plot', 'mini'],
        default='ascii',
        help='Visualization mode'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as background daemon'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=3.0,
        help='Daemon update interval (seconds)'
    )
    parser.add_argument(
        '--save',
        type=Path,
        help='Save plot to file (plot mode only)'
    )

    args = parser.parse_args()

    domain_dir = Path(args.dir)

    if not domain_dir.exists():
        print(f"Error: Domain memory directory not found: {domain_dir}")
        return 1

    if args.daemon:
        run_daemon(domain_dir, args.interval, args.mode)
    else:
        if args.mode == "plot" and args.save:
            backend = FlatFileBackend(domain_dir)
            features = backend.load_features()
            loom_state = TensionUpdater().update_all(features, [], LoomState())
            render_plot(features, loom_state, args.save)
        else:
            run_once(domain_dir, args.mode)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
