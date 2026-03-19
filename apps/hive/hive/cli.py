"""Hive CLI — entry point for cron and interactive use.

Usage:
    hive check [--hive ID] [--mode MODE] [--dry-run]
    hive status [--hive ID]
    hive resolve FINDING_ID [--notes TEXT] [--failure]
    hive learn
    hive init
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hololoom.core.memory.observation_reflector import ObservationReflector
from hololoom.core.memory.scored_observation import advance_observation
from hololoom.core.memory.bus import MemoryQuery

from .bus_client import HiveBusClient
from .models import load_hives, save_hives, SAMPLE_HIVES
from .jobs import job_inspect, JOB_REGISTRY
from .modes import detect_mode
from .notifier import ThresholdNotifier

logger = logging.getLogger(__name__)

HIVE_DIR = Path.home() / ".hive"
HIVES_PATH = HIVE_DIR / "hives.json"
REFLECTOR_PATH = HIVE_DIR / "reflector.json"


# =============================================================================
# Commands
# =============================================================================

async def cmd_check(args):
    """Run scoring pipeline over all hives."""
    hives = load_hives(HIVES_PATH)
    if not hives:
        print("No hives found. Run `hive init` first.")
        return

    reflector = ObservationReflector()
    reflector.load(REFLECTOR_PATH)
    notifier = ThresholdNotifier()

    async with HiveBusClient() as client:
        print(f"Bus mode: {client.mode}")

        # Wire notifier as post-store hook (local mode)
        if client._bus:
            client._bus._post_store_hooks.append(notifier.hook_store)

        results = await job_inspect(
            client, reflector, hives,
            mode=args.mode,
            hive_id=args.hive,
            dry_run=args.dry_run,
        )

    reflector.save(REFLECTOR_PATH)

    if not results:
        print("No inspections to score.")
        return

    # Print summary
    print(f"\n{'Hive':<20} {'Band':<6} {'Score':<8} {'Summary'}")
    print("-" * 70)
    for r in results:
        print(f"{r['hive_name']:<20} {r['band']:<6} {r['composite']:<8.1f} {r['content'][:40]}")
    print(f"\nMode: {results[0]['mode']} | {'DRY RUN' if args.dry_run else 'Stored to bus'}")

    # Alert summary
    if notifier.pending_count > 0:
        notifier.save_alerts()
        print(f"\n{notifier.summary()}")
    else:
        print("\nNo threshold alerts.")


async def cmd_status(args):
    """Show open findings."""
    async with HiveBusClient() as client:
        props = {"status": "open"}
        q_kwargs = {
            "intent": "open hive findings",
            "memory_type": "hive",
            "properties_match": props,
            "max_results": 20,
        }
        result = await client.query(MemoryQuery(**q_kwargs))

    if not result.items:
        print("No open findings.")
        return

    print(f"\n{'ID':<14} {'Band':<6} {'Content'}")
    print("-" * 60)
    for item in result.items:
        item_id = item.get("id", "?")[:12]
        band = item.get("band", "?")
        content = item.get("content", "")[:50]
        print(f"{item_id:<14} {band:<6} {content}")
    print(f"\n{len(result.items)} open finding(s) | via {result.resolution_path}")


async def cmd_resolve(args):
    """Resolve a finding and feed outcome to reflector."""
    reflector = ObservationReflector()
    reflector.load(REFLECTOR_PATH)
    success = not args.failure

    async with HiveBusClient() as client:
        if client.mode != "local":
            print("Resolve requires local bus mode (snapshot). Set HOLOLOOM_API_URL to invalid to force local.")
            return

        try:
            res_id = await advance_observation(
                client._bus, args.finding_id, "resolved", args.notes
            )
        except KeyError:
            print(f"Finding {args.finding_id} not found.")
            return

        # Feed back to reflector
        original = client._bus.get_item(args.finding_id)
        if original and original.properties.get("breakdown"):
            mode = original.properties.get("mode", "default")
            reflector.record_outcome(mode, original.properties["breakdown"], success=success)

    reflector.save(REFLECTOR_PATH)
    print(f"Resolved {args.finding_id} -> {res_id} ({'success' if success else 'failure'})")


async def cmd_learn(args):
    """Show reflector learning stats."""
    reflector = ObservationReflector()
    reflector.load(REFLECTOR_PATH)

    stats = reflector.stats()
    if not stats:
        print("No learning data yet. Run `hive check` and `hive resolve` first.")
        return

    print(f"\n{'Mode:Term':<30} {'Mean':<8} {'Mult':<8} {'Pulls':<8} {'Certainty'}")
    print("-" * 70)
    for key, s in sorted(stats.items()):
        print(f"{key:<30} {s['mean']:<8.3f} {s['multiplier']:<8.3f} {s['total_pulls']:<8} {s['certainty']:.3f}")
    print(f"\n{reflector.arm_count} arm(s)")


async def cmd_alerts(args):
    """Show unacknowledged alerts."""
    notifier = ThresholdNotifier()
    alerts = notifier.unacknowledged()

    if not alerts:
        print("No unacknowledged alerts.")
        return

    print(f"\n{'Band':<6} {'Score':<8} {'Time':<22} {'Content'}")
    print("-" * 70)
    for a in alerts:
        t = a.get("alerted_at", "")[:19]
        print(f"{a['band']:<6} {a['composite']:<8.1f} {t:<22} {a['content'][:40]}")
    print(f"\n{len(alerts)} unacknowledged alert(s)")


async def cmd_init(args):
    """Initialize ~/.hive/ with sample data."""
    if HIVES_PATH.exists() and not args.force:
        print(f"{HIVES_PATH} already exists. Use --force to overwrite.")
        return

    HIVE_DIR.mkdir(parents=True, exist_ok=True)
    save_hives(SAMPLE_HIVES, HIVES_PATH)
    print(f"Created {HIVES_PATH} with {len(SAMPLE_HIVES)} sample hives")
    print(f"  Sunflower (hive-1): healthy baseline")
    print(f"  Clover (hive-2): multiple issues (queenless, varroa, disease)")
    print(f"  Wildflower (hive-3): healthy, heavy stores")
    print(f"\nRun `hive check` to score them.")


# =============================================================================
# Parser
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hive",
        description="Hive management CLI — scored observations for apiary health",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command")

    # check
    p = subparsers.add_parser("check", help="Run scoring pipeline")
    p.add_argument("--hive", help="Score specific hive ID (default: all)")
    p.add_argument("--mode", help="Override seasonal mode (spring/nectar_flow/fall/winter)")
    p.add_argument("--dry-run", action="store_true", help="Score but don't store")
    p.set_defaults(func=cmd_check)

    # status
    p = subparsers.add_parser("status", help="Show open findings")
    p.add_argument("--hive", help="Filter by hive ID")
    p.set_defaults(func=cmd_status)

    # resolve
    p = subparsers.add_parser("resolve", help="Resolve a finding")
    p.add_argument("finding_id", help="ID of the finding to resolve")
    p.add_argument("--notes", default="", help="Resolution notes")
    p.add_argument("--failure", action="store_true", help="Mark as failed outcome")
    p.set_defaults(func=cmd_resolve)

    # alerts
    p = subparsers.add_parser("alerts", help="Show unacknowledged threshold alerts")
    p.set_defaults(func=cmd_alerts)

    # learn
    p = subparsers.add_parser("learn", help="Show reflector learning stats")
    p.set_defaults(func=cmd_learn)

    # init
    p = subparsers.add_parser("init", help="Initialize ~/.hive/ with sample data")
    p.add_argument("--force", action="store_true", help="Overwrite existing data")
    p.set_defaults(func=cmd_init)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if hasattr(args, "func"):
        asyncio.run(args.func(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
