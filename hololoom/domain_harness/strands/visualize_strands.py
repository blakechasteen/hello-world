#!/usr/bin/env python3
"""
Strands Tension Visualizer

Monitors domain memory and visualizes tension across all strands.

Modes:
- ASCII: Terminal-based visualization (default)
- Rich: Using rich library for better terminal UI
- JSON: Outputs machine-readable tension snapshots

Usage:
    python visualize_strands.py                    # ASCII mode
    python visualize_strands.py --mode rich        # Rich terminal UI
    python visualize_strands.py --mode json        # JSON output
    python visualize_strands.py --once             # Single snapshot, no loop
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hololoom.domain_harness.strands.tension_model import (
    TENSION_CRITICAL,
    TENSION_MAX,
    TENSION_STABLE,
    TENSION_STRESSED,
    LoomState,
    StrandState,
    TensionModel,
)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DOMAIN_MEMORY = Path("domain_memory")
POLL_INTERVAL = 2.0  # seconds
BAR_WIDTH = 40


# ============================================================================
# Data Loading
# ============================================================================

def load_features(domain_memory: Path) -> list:
    """Load features from features.json."""
    features_path = domain_memory / "features.json"
    if not features_path.exists():
        return []

    with open(features_path) as f:
        data = json.load(f)
    return data.get("features", [])


def load_loom_state(domain_memory: Path) -> LoomState:
    """Load or create loom state from features."""
    features = load_features(domain_memory)

    # Build loom state from features
    loom = LoomState()

    for f in features:
        feature_id = f["id"]

        # Check for existing strand_state in feature
        if "strand_state" in f:
            strand = StrandState.from_dict(feature_id, f["strand_state"])
        else:
            # Create strand with tension based on status
            strand = StrandState(feature_id=feature_id)
            status = f.get("status", "failing")

            # Map status to initial tension
            tension_map = {
                "passing": 1.0,
                "failing": 6.0,
                "pending": 5.0,
                "in_progress": 4.0,
                "blocked": 7.0
            }
            strand.tension = tension_map.get(status, 5.0)

        loom.strands[feature_id] = strand

    return loom


# ============================================================================
# ASCII Visualization
# ============================================================================

def tension_bar(tension: float, width: int = BAR_WIDTH) -> str:
    """Create ASCII tension bar."""
    filled = int((tension / TENSION_MAX) * width)
    empty = width - filled

    # Color coding via characters
    if tension <= TENSION_STABLE:
        char = "═"
    elif tension <= TENSION_STRESSED:
        char = "▓"
    elif tension <= TENSION_CRITICAL:
        char = "█"
    else:
        char = "▒"

    return f"[{char * filled}{'·' * empty}]"


def render_ascii(loom: LoomState) -> str:
    """Render full ASCII visualization."""
    lines = []

    # Header
    lines.append("")
    lines.append("╔════════════════════════════════════════════════════════════╗")
    lines.append("║               STRANDS TENSION VISUALIZER                   ║")
    lines.append("╠════════════════════════════════════════════════════════════╣")

    # Overall stats
    avg = loom.average_tension()
    total = len(loom.strands)
    stressed = len(loom.stressed_strands())
    critical = len(loom.critical_strands())

    lines.append(f"║  Strands: {total:<6} Avg Tension: {avg:.1f}                        ║")
    lines.append(f"║  Stressed: {stressed:<5} Critical: {critical:<5}                        ║")
    lines.append("╠════════════════════════════════════════════════════════════╣")

    # Individual strands
    model = TensionModel()

    for feature_id, strand in sorted(loom.strands.items()):
        emoji = model.get_status_emoji(strand.tension)
        bar = tension_bar(strand.tension, width=30)
        motion = strand.last_motion.value[:12] if strand.last_motion else "---"

        line = f"║ {emoji} {feature_id:8} {bar} {strand.tension:4.1f} {motion:12} ║"
        lines.append(line)

    # Footer
    lines.append("╠════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Last update: {datetime.now().strftime('%H:%M:%S'):40}    ║")
    lines.append("╚════════════════════════════════════════════════════════════╝")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Rich Visualization (if available)
# ============================================================================

def render_rich(loom: LoomState):
    """Render using rich library for better terminal UI."""
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress
        from rich.table import Table
    except ImportError:
        print("Rich library not installed. Run: pip install rich")
        return render_ascii(loom)

    console = Console()

    table = Table(title="Strands Tension")
    table.add_column("ID", style="cyan")
    table.add_column("Tension", justify="right")
    table.add_column("Bar", style="green")
    table.add_column("Motion", style="yellow")
    table.add_column("Status")

    model = TensionModel()

    for feature_id, strand in sorted(loom.strands.items()):
        emoji = model.get_status_emoji(strand.tension)

        # Tension bar using Unicode blocks
        bar_width = 20
        filled = int((strand.tension / TENSION_MAX) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color based on tension
        if strand.tension <= TENSION_STABLE:
            bar = f"[green]{bar}[/green]"
        elif strand.tension <= TENSION_STRESSED:
            bar = f"[yellow]{bar}[/yellow]"
        else:
            bar = f"[red]{bar}[/red]"

        motion = strand.last_motion.value if strand.last_motion else "---"

        table.add_row(
            feature_id,
            f"{strand.tension:.1f}",
            bar,
            motion,
            emoji
        )

    console.clear()
    console.print(Panel(table, title="🧵 Loom Status"))
    console.print(f"Average tension: {loom.average_tension():.1f}")
    console.print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")


# ============================================================================
# JSON Output
# ============================================================================

def render_json(loom: LoomState) -> str:
    """Output machine-readable JSON snapshot."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_strands": len(loom.strands),
            "average_tension": loom.average_tension(),
            "total_tension": loom.total_tension(),
            "stressed_strands": loom.stressed_strands(),
            "critical_strands": loom.critical_strands()
        },
        "strands": {
            fid: {
                "tension": s.tension,
                "last_motion": s.last_motion.value if s.last_motion else None,
                "consecutive_fails": s.consecutive_fails,
                "regression_count": s.regression_count
            }
            for fid, s in loom.strands.items()
        }
    }
    return json.dumps(snapshot, indent=2)


# ============================================================================
# Main Loop
# ============================================================================

def run_visualizer(
    domain_memory: Path,
    mode: str = "ascii",
    once: bool = False,
    interval: float = POLL_INTERVAL
):
    """Run the visualizer."""

    print("🧵 Strands Visualizer starting...")
    print(f"   Domain memory: {domain_memory}")
    print(f"   Mode: {mode}")
    print(f"   Interval: {interval}s")
    print("")

    while True:
        try:
            loom = load_loom_state(domain_memory)

            if mode == "ascii":
                # Clear screen and render
                print("\033[2J\033[H", end="")  # ANSI clear
                print(render_ascii(loom))
            elif mode == "rich":
                render_rich(loom)
            elif mode == "json":
                print(render_json(loom))
            else:
                print(render_ascii(loom))

            if once:
                break

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🧵 Visualizer stopped.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            if once:
                break
            time.sleep(interval)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strands Tension Visualizer"
    )
    parser.add_argument(
        "--dir", "-d",
        default="./domain_memory",
        help="Domain memory directory"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["ascii", "rich", "json"],
        default="ascii",
        help="Visualization mode"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single snapshot, no loop"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=POLL_INTERVAL,
        help="Poll interval in seconds"
    )

    args = parser.parse_args()

    run_visualizer(
        domain_memory=Path(args.dir),
        mode=args.mode,
        once=args.once,
        interval=args.interval
    )


if __name__ == "__main__":
    main()
