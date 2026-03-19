#!/usr/bin/env python3
"""
run_worker.py - Domain Harness Worker Orchestrator

Runs worker cycles against flat-file or Neo4j domain memory.

Usage:
    # Initialize a new project
    python run_worker.py init "Build a REST API with user auth and blog posts"
    
    # Run one worker cycle
    python run_worker.py run
    
    # Run until complete (or max cycles)
    python run_worker.py loop --max-cycles 50
    
    # Check status
    python run_worker.py status
    
    # View progress log
    python run_worker.py log

Options:
    --dir PATH          Domain memory directory (default: ./domain_memory)
    --backend TYPE      Storage backend: flat, neo4j (default: flat)
    --dry-run           Don't actually modify anything
    --verbose           Show detailed output
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hololoom.domain_harness.initializer import FeatureParser
from hololoom.domain_harness.schema import Feature, FeatureStatus
from hololoom.domain_harness.templates import FlatFileBackend, FlatFileGenerator

# Strands integration (optional)
try:
    from hololoom.domain_harness.strands import (
        GitAutoCommit,
        GitConfig,
        LoomState,
        TensionModel,
        WeaveDecoder,
        decode_worker_result,
    )
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False


# ============================================================================
# CLI Colors
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.END}"


# ============================================================================
# Worker Implementation (Subprocess)
# ============================================================================

class SubprocessWorker:
    """
    Worker that runs tests via subprocess.
    
    Implementation is a no-op (human or LLM does the actual coding).
    This just handles the test/status/log cycle.
    """

    def __init__(
        self,
        backend: FlatFileBackend,
        test_command: str = "pytest -q",
        dry_run: bool = False,
        verbose: bool = True
    ):
        self.backend = backend
        self.test_command = test_command
        self.dry_run = dry_run
        self.verbose = verbose
        self.worker_id = f"cli_{datetime.now().strftime('%H%M%S')}"

    def run_once(self) -> dict:
        """
        Execute one worker cycle.
        
        Returns status dict.
        """
        # Boot: Load state
        features = self.backend.get_actionable_features()

        if not features:
            return {
                "status": "complete",
                "message": "No actionable features remaining"
            }

        # Select first actionable feature
        target = features[0]

        if self.verbose:
            print(f"\n{colored('▶ Selected feature:', Colors.CYAN)} {target.id} - {target.name}")

        if self.dry_run:
            return {
                "status": "dry_run",
                "feature_id": target.id,
                "message": f"Would work on: {target.name}"
            }

        # Mark in progress
        self.backend.update_feature_status(target.id, FeatureStatus.IN_PROGRESS)

        # Run tests for this feature
        test_file = self.backend.dir / "tests" / f"test_feature_{target.id.lower().replace('-', '_')}.py"

        if test_file.exists():
            cmd = f"{self.test_command} {test_file}"
        else:
            cmd = self.test_command

        if self.verbose:
            print(f"{colored('▶ Running:', Colors.CYAN)} {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            passed = False
            output = "Test timeout exceeded"
        except Exception as e:
            passed = False
            output = str(e)

        # Update status
        new_status = FeatureStatus.PASSING if passed else FeatureStatus.FAILING
        self.backend.update_feature_status(target.id, new_status)

        # Log progress
        action = "PASS" if passed else "FAIL"
        summary = output[:200].replace('\n', ' ').strip()
        self.backend.append_progress(target.id, action, summary, self.worker_id)

        status_icon = colored("✓", Colors.GREEN) if passed else colored("✗", Colors.RED)
        if self.verbose:
            print(f"{status_icon} {target.name}: {new_status.value}")

        return {
            "status": "completed",
            "feature_id": target.id,
            "passed": passed,
            "output": output[:500]
        }

    def run_loop(self, max_cycles: int = 100) -> list[dict]:
        """Run until complete or max cycles."""
        results = []

        for i in range(max_cycles):
            if self.verbose:
                print(f"\n{colored(f'═══ Cycle {i+1}/{max_cycles} ═══', Colors.BOLD)}")

            result = self.run_once()
            results.append(result)

            if result["status"] == "complete":
                break

        return results


# ============================================================================
# Commands
# ============================================================================

def cmd_init(args):
    """Initialize domain memory from prompt."""
    prompt = args.prompt
    project_name = args.name or "AgentProject"
    output_dir = Path(args.dir)

    print(f"\n{colored('═══ Initializing Domain Memory ═══', Colors.BOLD)}")
    print(f"Project: {project_name}")
    print(f"Output: {output_dir}")

    # Parse features from prompt
    parser = FeatureParser()
    parsed = parser.parse(prompt)

    # Convert to Feature objects
    features = []
    for i, p in enumerate(parsed):
        fid = f"F{str(i+1).zfill(3)}"
        features.append(Feature(
            id=fid,
            name=p['name'],
            description=p['description'],
            priority=p.get('priority', len(parsed) - i),
            depends_on=[]  # Could resolve from parsed
        ))

    print(f"\nParsed {len(features)} features:")
    for f in features:
        print(f"  [{f.id}] {f.name}")

    # Generate files
    generator = FlatFileGenerator(output_dir)
    created = generator.generate(
        features=features,
        project_name=project_name,
        language=args.language or "python"
    )

    print(f"\n{colored('Created files:', Colors.GREEN)}")
    for name, path in created.items():
        print(f"  ✓ {path}")

    print(f"\n{colored('Next steps:', Colors.CYAN)}")
    print("  1. Review features.json and adjust priorities")
    print("  2. Implement features (or let an LLM do it)")
    print("  3. Run: python run_worker.py run")


def cmd_run(args):
    """Run one worker cycle."""
    backend = FlatFileBackend(Path(args.dir))

    if not backend.features_path.exists():
        print(colored("Error: Domain memory not initialized. Run 'init' first.", Colors.RED))
        return 1

    state = backend.load_state()
    test_cmd = state.environment.get("test_command", "pytest -q")

    worker = SubprocessWorker(
        backend=backend,
        test_command=test_cmd,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    result = worker.run_once()

    if result["status"] == "complete":
        print(colored("\n✓ All features passing!", Colors.GREEN))

    return 0 if result.get("passed", True) else 1


def cmd_loop(args):
    """Run worker cycles until complete."""
    backend = FlatFileBackend(Path(args.dir))

    if not backend.features_path.exists():
        print(colored("Error: Domain memory not initialized. Run 'init' first.", Colors.RED))
        return 1

    state = backend.load_state()
    test_cmd = state.environment.get("test_command", "pytest -q")

    worker = SubprocessWorker(
        backend=backend,
        test_command=test_cmd,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    print(f"\n{colored('═══ Running Worker Loop ═══', Colors.BOLD)}")
    print(f"Max cycles: {args.max_cycles}")

    results = worker.run_loop(max_cycles=args.max_cycles)

    # Summary
    passed = len([r for r in results if r.get("passed")])
    failed = len([r for r in results if r.get("passed") == False])

    print(f"\n{colored('═══ Summary ═══', Colors.BOLD)}")
    print(f"Cycles: {len(results)}")
    print(f"Passed: {colored(str(passed), Colors.GREEN)}")
    print(f"Failed: {colored(str(failed), Colors.RED)}")

    return 0


def cmd_status(args):
    """Show current status."""
    backend = FlatFileBackend(Path(args.dir))

    if not backend.features_path.exists():
        print(colored("Error: Domain memory not initialized.", Colors.RED))
        return 1

    features = backend.load_features()
    state = backend.load_state()

    print(f"\n{colored('═══ Domain Status ═══', Colors.BOLD)}")
    print(f"Project: {state.project_name}")
    print(f"Total features: {len(features)}")

    # Count by status
    status_counts = {}
    for f in features:
        status_counts[f.status.value] = status_counts.get(f.status.value, 0) + 1

    print("\nStatus breakdown:")
    status_colors = {
        "passing": Colors.GREEN,
        "failing": Colors.RED,
        "pending": Colors.YELLOW,
        "in_progress": Colors.CYAN,
        "blocked": Colors.RED
    }
    for status, count in sorted(status_counts.items()):
        color = status_colors.get(status, Colors.END)
        print(f"  {colored(status, color)}: {count}")

    # Completion rate
    passing = status_counts.get("passing", 0)
    total = len(features)
    rate = (passing / total * 100) if total > 0 else 0
    print(f"\nCompletion: {colored(f'{rate:.0f}%', Colors.GREEN if rate == 100 else Colors.YELLOW)}")

    # Show features
    print(f"\n{colored('Features:', Colors.CYAN)}")
    for f in features:
        icon = {
            FeatureStatus.PASSING: colored("✓", Colors.GREEN),
            FeatureStatus.FAILING: colored("✗", Colors.RED),
            FeatureStatus.PENDING: colored("○", Colors.YELLOW),
            FeatureStatus.IN_PROGRESS: colored("◐", Colors.CYAN),
            FeatureStatus.BLOCKED: colored("⊘", Colors.RED)
        }.get(f.status, "?")
        print(f"  {icon} [{f.id}] {f.name}")

    return 0


def cmd_log(args):
    """Show progress log."""
    backend = FlatFileBackend(Path(args.dir))

    if not backend.progress_path.exists():
        print(colored("Error: No progress log found.", Colors.RED))
        return 1

    print(f"\n{colored('═══ Progress Log ═══', Colors.BOLD)}")

    with open(backend.progress_path) as f:
        content = f.read()

    # Colorize output
    for line in content.split('\n'):
        if line.startswith('#'):
            print(colored(line, Colors.CYAN))
        elif 'PASS' in line:
            print(colored(line, Colors.GREEN))
        elif 'FAIL' in line:
            print(colored(line, Colors.RED))
        elif '[INIT]' in line:
            print(colored(line, Colors.YELLOW))
        else:
            print(line)

    return 0


def cmd_tension(args):
    """Show Strands tension visualization."""
    if not STRANDS_AVAILABLE:
        print(colored("Error: Strands module not available.", Colors.RED))
        return 1

    from hololoom.domain_harness.strands.visualize_strands import (
        load_loom_state,
        render_ascii,
        render_json,
    )

    domain_memory = Path(args.dir)

    if not (domain_memory / "features.json").exists():
        print(colored("Error: Domain memory not initialized.", Colors.RED))
        return 1

    if args.watch:
        # Watch mode - continuous updates
        import time
        print("Watching tension (Ctrl+C to stop)...")
        while True:
            try:
                loom = load_loom_state(domain_memory)
                print("\033[2J\033[H", end="")  # Clear screen
                if args.mode == "json":
                    print(render_json(loom))
                else:
                    print(render_ascii(loom))
                time.sleep(2)
            except KeyboardInterrupt:
                break
    else:
        # Single snapshot
        loom = load_loom_state(domain_memory)
        if args.mode == "json":
            print(render_json(loom))
        else:
            print(render_ascii(loom))

    return 0


def cmd_reset(args):
    """Reset all features to failing status."""
    backend = FlatFileBackend(Path(args.dir))

    if not backend.features_path.exists():
        print(colored("Error: Domain memory not initialized.", Colors.RED))
        return 1

    if not args.force:
        confirm = input("Reset all features to 'failing'? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return 0

    features = backend.load_features()
    for f in features:
        f.status = FeatureStatus.FAILING
    backend.save_features(features)

    # Add log entry
    backend.append_progress("ALL", "RESET", "All features reset to failing status")

    print(colored("✓ All features reset to failing.", Colors.YELLOW))
    return 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Domain Harness Worker Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init "Build a REST API with user auth"
  %(prog)s run
  %(prog)s loop --max-cycles 20
  %(prog)s status
  %(prog)s log
        """
    )

    parser.add_argument(
        '--dir', '-d',
        default='./domain_memory',
        help='Domain memory directory (default: ./domain_memory)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't actually modify anything"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # init
    init_parser = subparsers.add_parser('init', help='Initialize domain memory')
    init_parser.add_argument('prompt', help='Project description')
    init_parser.add_argument('--name', '-n', help='Project name')
    init_parser.add_argument('--language', '-l', default='python', help='Language')

    # run
    subparsers.add_parser('run', help='Run one worker cycle')

    # loop
    loop_parser = subparsers.add_parser('loop', help='Run until complete')
    loop_parser.add_argument('--max-cycles', '-m', type=int, default=100, help='Max cycles')

    # status
    subparsers.add_parser('status', help='Show status')

    # log
    subparsers.add_parser('log', help='Show progress log')

    # reset
    reset_parser = subparsers.add_parser('reset', help='Reset features to failing')
    reset_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')

    # tension (strands)
    tension_parser = subparsers.add_parser('tension', help='Show tension visualization')
    tension_parser.add_argument('--mode', '-m', choices=['ascii', 'json'], default='ascii')
    tension_parser.add_argument('--watch', '-w', action='store_true', help='Watch mode (continuous)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command
    commands = {
        'init': cmd_init,
        'run': cmd_run,
        'loop': cmd_loop,
        'status': cmd_status,
        'log': cmd_log,
        'reset': cmd_reset,
        'tension': cmd_tension
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
