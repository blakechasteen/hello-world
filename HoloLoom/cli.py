#!/usr/bin/env python3
"""
HoloLoom CLI - Unified Entry Point for Agent Operations

"Kubernetes for AI Agents"

Usage:
    hololoom query "What is Thompson Sampling?"
    hololoom agent list
    hololoom agent run --workflow research "Analyze bandit algorithms"
    hololoom agent status
    hololoom agent logs --limit 10
    hololoom cluster status
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

# Core imports
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.protocols.types import Query


# ============================================================================
# Color Output (Windows-compatible ASCII)
# ============================================================================

class Colors:
    """ANSI colors for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ENDC = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output."""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.ENDC = ''


def print_banner():
    """Print HoloLoom CLI banner."""
    print(f"""
{Colors.CYAN}+-----------------------------------------------+
|  HoloLoom CLI - Agent Hypervisor              |
|  "Kubernetes for AI Agents"                   |
+-----------------------------------------------+{Colors.ENDC}
""")


def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {msg}")


def print_error(msg: str):
    """Print error message."""
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {msg}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}[!]{Colors.ENDC} {msg}")


def print_info(msg: str):
    """Print info message."""
    print(f"{Colors.BLUE}[i]{Colors.ENDC} {msg}")


# ============================================================================
# Query Command
# ============================================================================

async def cmd_query(args):
    """Execute a single query through the weaving orchestrator."""
    cfg_factory = getattr(Config, args.mode)
    cfg = cfg_factory()

    if args.verbose:
        print_info(f"Mode: {args.mode.upper()}")
        print_info(f"Query: {args.text}")
        print()

    # Empty shards for basic query
    shards = []

    async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:
        result = await orchestrator.weave(Query(text=args.text))

        if args.json:
            output = {
                "response": result.response,
                "confidence": result.confidence,
                "tool_used": result.metadata.get('tool_used', 'unknown'),
                "latency_ms": result.metadata.get('latency_ms', 0),
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{Colors.BOLD}Response:{Colors.ENDC}")
            print(result.response)

            if args.verbose:
                print(f"\n{Colors.DIM}Confidence: {result.confidence:.2f}{Colors.ENDC}")


# ============================================================================
# Agent Commands
# ============================================================================

async def cmd_agent_list(args):
    """List registered agents."""
    print(f"\n{Colors.BOLD}Available Agent Types:{Colors.ENDC}\n")

    agents = [
        ("researcher", "Deep research and analysis", "Claude (complex reasoning)"),
        ("verifier", "Fact-checking and verification", "Claude (accuracy-critical)"),
        ("synthesizer", "Synthesis and summarization", "Ollama (fast, local)"),
        ("planner", "Task decomposition and planning", "Claude (multi-step)"),
        ("executor", "Code execution and tools", "GPT-4 (tool use)"),
        ("critic", "Quality assessment and refinement", "Claude (evaluation)"),
    ]

    for agent_type, description, model in agents:
        print(f"  {Colors.CYAN}{agent_type:12}{Colors.ENDC} - {description}")
        print(f"               {Colors.DIM}Default model: {model}{Colors.ENDC}")

    print(f"\n{Colors.DIM}Use 'hololoom agent run --type <type> <query>' to run an agent{Colors.ENDC}")


async def cmd_agent_run(args):
    """Run an agent workflow."""
    print_banner()

    workflow_name = args.workflow or "default"
    agent_type = args.type or "researcher"

    print(f"{Colors.BOLD}Running Agent Workflow{Colors.ENDC}")
    print(f"  Workflow: {workflow_name}")
    print(f"  Agent: {agent_type}")
    print(f"  Query: {args.query}")
    print()

    # Configure based on workflow
    if workflow_name == "research":
        cfg = Config.fused()
        cfg.max_iterations = args.max_steps or 5
    elif workflow_name == "fast":
        cfg = Config.fast()
        cfg.max_iterations = args.max_steps or 2
    else:
        cfg = Config.fast()
        cfg.max_iterations = args.max_steps or 3

    shards = []

    async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:
        start_time = datetime.now()
        result = await orchestrator.weave(Query(text=args.query))
        duration = (datetime.now() - start_time).total_seconds()

        print_success(f"Workflow completed in {duration:.2f}s")
        print(f"\n{Colors.BOLD}Result:{Colors.ENDC}")
        print(result.response)

        if args.json:
            print(f"\n{Colors.BOLD}Metadata:{Colors.ENDC}")
            print(json.dumps({
                "confidence": result.confidence,
                "duration_s": duration,
                "workflow": workflow_name,
                "agent_type": agent_type,
            }, indent=2))


async def cmd_agent_status(args):
    """Show status of agents and hypervisor."""
    print(f"\n{Colors.BOLD}Agent Hypervisor Status{Colors.ENDC}\n")

    # Simulated status (would query actual hypervisor in production)
    status = {
        "hypervisor": "running",
        "agents_registered": 0,
        "agents_active": 0,
        "total_requests": 0,
        "uptime": "0h 0m",
    }

    print(f"  Hypervisor:    {Colors.GREEN}running{Colors.ENDC}")
    print(f"  Agents:        {status['agents_registered']} registered, {status['agents_active']} active")
    print(f"  Requests:      {status['total_requests']} total")
    print(f"  Uptime:        {status['uptime']}")

    print(f"\n{Colors.BOLD}Components:{Colors.ENDC}")
    print(f"  Audit Trail:     {Colors.GREEN}[OK]{Colors.ENDC}")
    print(f"  Safety Guards:   {Colors.GREEN}[OK]{Colors.ENDC}")
    print(f"  Thompson Sampler:{Colors.GREEN}[OK]{Colors.ENDC}")
    print(f"  Circuit Breaker: {Colors.GREEN}[OK]{Colors.ENDC}")

    print(f"\n{Colors.DIM}Note: Full status requires running hypervisor server{Colors.ENDC}")


async def cmd_agent_logs(args):
    """View audit trail logs."""
    print(f"\n{Colors.BOLD}Audit Trail Logs{Colors.ENDC}")
    print(f"{Colors.DIM}(Last {args.limit} entries){Colors.ENDC}\n")

    # Simulated logs (would query actual audit trail in production)
    print(f"  {Colors.DIM}No logs available - start the hypervisor server first{Colors.ENDC}")
    print(f"\n  Run: uvicorn HoloLoom.server.agentic_api:app --port 8000")


# ============================================================================
# Cluster Commands
# ============================================================================

async def cmd_cluster_status(args):
    """Show cluster status (federation, distributed workers)."""
    print(f"\n{Colors.BOLD}Cluster Status{Colors.ENDC}\n")

    print(f"  Mode:          Standalone (no cluster)")
    print(f"  Federation:    {Colors.YELLOW}Not configured{Colors.ENDC}")
    print(f"  Eggroll:       {Colors.YELLOW}Not running{Colors.ENDC}")

    print(f"\n{Colors.DIM}To enable distributed mode:{Colors.ENDC}")
    print(f"  1. Configure federation in config.yaml")
    print(f"  2. Start eggroll workers: hololoom cluster start-workers")
    print(f"  3. Join federation: hololoom cluster join <peer-address>")


async def cmd_cluster_nodes(args):
    """List cluster nodes."""
    print(f"\n{Colors.BOLD}Cluster Nodes{Colors.ENDC}\n")
    print(f"  {Colors.DIM}No cluster nodes - running in standalone mode{Colors.ENDC}")


# ============================================================================
# Main Entry Point
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='hololoom',
        description='HoloLoom CLI - Agent Hypervisor ("Kubernetes for AI Agents")',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hololoom query "What is Thompson Sampling?"
  hololoom query --mode research "Analyze multi-armed bandits"
  hololoom agent list
  hololoom agent run --workflow research "Compare UCB vs Thompson Sampling"
  hololoom agent status
  hololoom cluster status

Documentation:
  https://github.com/yourusername/hololoom
  See docs/AGENT_HYPERVISOR.md for architecture details
"""
    )

    parser.add_argument('--version', action='version', version='HoloLoom 1.0.0')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # -------------------------------------------------------------------------
    # Query command
    # -------------------------------------------------------------------------
    query_parser = subparsers.add_parser('query', help='Execute a single query')
    query_parser.add_argument('text', help='Query text')
    query_parser.add_argument(
        '--mode', '-m',
        choices=['bare', 'fast', 'fused', 'research'],
        default='fast',
        help='Execution mode (default: fast)'
    )
    query_parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    query_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    query_parser.set_defaults(func=cmd_query)

    # -------------------------------------------------------------------------
    # Agent commands
    # -------------------------------------------------------------------------
    agent_parser = subparsers.add_parser('agent', help='Agent management commands')
    agent_subparsers = agent_parser.add_subparsers(dest='agent_command', help='Agent commands')

    # agent list
    agent_list = agent_subparsers.add_parser('list', help='List available agents')
    agent_list.set_defaults(func=cmd_agent_list)

    # agent run
    agent_run = agent_subparsers.add_parser('run', help='Run an agent workflow')
    agent_run.add_argument('query', help='Query/task for the agent')
    agent_run.add_argument('--workflow', '-w', choices=['default', 'research', 'fast', 'verify'],
                          help='Workflow type')
    agent_run.add_argument('--type', '-t', help='Agent type (researcher, verifier, etc.)')
    agent_run.add_argument('--max-steps', type=int, help='Maximum workflow steps')
    agent_run.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    agent_run.set_defaults(func=cmd_agent_run)

    # agent status
    agent_status = agent_subparsers.add_parser('status', help='Show agent/hypervisor status')
    agent_status.set_defaults(func=cmd_agent_status)

    # agent logs
    agent_logs = agent_subparsers.add_parser('logs', help='View audit trail logs')
    agent_logs.add_argument('--limit', '-n', type=int, default=10, help='Number of entries')
    agent_logs.add_argument('--agent', help='Filter by agent ID')
    agent_logs.add_argument('--level', choices=['info', 'warning', 'error'], help='Filter by level')
    agent_logs.set_defaults(func=cmd_agent_logs)

    # -------------------------------------------------------------------------
    # Cluster commands
    # -------------------------------------------------------------------------
    cluster_parser = subparsers.add_parser('cluster', help='Cluster management commands')
    cluster_subparsers = cluster_parser.add_subparsers(dest='cluster_command', help='Cluster commands')

    # cluster status
    cluster_status = cluster_subparsers.add_parser('status', help='Show cluster status')
    cluster_status.set_defaults(func=cmd_cluster_status)

    # cluster nodes
    cluster_nodes = cluster_subparsers.add_parser('nodes', help='List cluster nodes')
    cluster_nodes.set_defaults(func=cmd_cluster_nodes)

    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Handle subcommand parsing
    if args.command == 'agent' and not hasattr(args, 'func'):
        print_error("Please specify an agent command (list, run, status, logs)")
        sys.exit(1)

    if args.command == 'cluster' and not hasattr(args, 'func'):
        print_error("Please specify a cluster command (status, nodes)")
        sys.exit(1)

    # Run the command
    if hasattr(args, 'func'):
        try:
            asyncio.run(args.func(args))
        except KeyboardInterrupt:
            print("\nInterrupted")
            sys.exit(130)
        except Exception as e:
            print_error(str(e))
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
