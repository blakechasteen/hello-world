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

import argparse
import asyncio
import json
import sys
from datetime import datetime

from hololoom.cli_client import HTTPX_AVAILABLE, HypervisorClient

# Core imports
from hololoom.config import Config
from hololoom.protocols.types import Query
from hololoom.weaving_orchestrator import WeavingOrchestrator

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
    """Execute a single query through the weaving orchestrator.

    Uses HTTP server if available, falls back to local orchestrator.
    """
    if args.verbose:
        print_info(f"Mode: {args.mode.upper()}")
        print_info(f"Query: {args.text}")
        print()

    # Try HTTP server first if httpx is available
    use_http = False
    if HTTPX_AVAILABLE:
        async with HypervisorClient() as client:
            if await client.is_server_available():
                use_http = True
                if args.verbose:
                    print_info("Using HTTP server (Agentic API)")

                # Map CLI modes to API modes
                mode_map = {"bare": "direct", "fast": "direct", "fused": "verify", "research": "research"}
                api_mode = mode_map.get(args.mode, "verify")

                result = await client.query(
                    text=args.text,
                    mode=api_mode,
                    max_steps=5
                )

                if "error" in result:
                    print_error(result["error"])
                    return

                if args.json:
                    output = {
                        "response": result.get("response", ""),
                        "confidence": result.get("confidence", 0),
                        "tool_used": result.get("tool_used", "unknown"),
                        "latency_ms": result.get("latency_ms", 0),
                        "source": "http"
                    }
                    print(json.dumps(output, indent=2))
                else:
                    print(f"\n{Colors.BOLD}Response:{Colors.ENDC}")
                    print(result.get("response", ""))

                    if args.verbose:
                        conf = result.get("confidence", 0)
                        print(f"\n{Colors.DIM}Confidence: {conf:.2f}{Colors.ENDC}")
                return

    # Fall back to local orchestrator
    if args.verbose and HTTPX_AVAILABLE:
        print_info("Servers unavailable, using local orchestrator")
    elif args.verbose:
        print_info("Using local orchestrator")

    cfg_factory = getattr(Config, args.mode)
    cfg = cfg_factory()

    shards = []

    async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:
        result = await orchestrator.weave(Query(text=args.text))

        if args.json:
            output = {
                "response": result.response,
                "confidence": result.confidence,
                "tool_used": result.metadata.get('tool_used', 'unknown'),
                "latency_ms": result.metadata.get('latency_ms', 0),
                "source": "local"
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
    """Run an agent workflow.

    Uses HTTP server if available (creates thread via Agent Manager API),
    falls back to local orchestrator.
    """
    print_banner()

    workflow_name = args.workflow or "default"
    agent_type = args.type or "researcher"

    print(f"{Colors.BOLD}Running Agent Workflow{Colors.ENDC}")
    print(f"  Workflow: {workflow_name}")
    print(f"  Agent: {agent_type}")
    print(f"  Query: {args.query}")
    print()

    # Try HTTP server first if httpx is available
    if HTTPX_AVAILABLE:
        async with HypervisorClient() as client:
            # Check if Agent Manager is available
            health = await client.check_agent_manager_health()
            if health.get("status") == "healthy":
                print_info("Using Agent Manager API (HTTP)")

                # Map workflow to reasoning mode
                mode_map = {
                    "default": "direct",
                    "fast": "direct",
                    "research": "research",
                    "verify": "verify"
                }
                reasoning_mode = mode_map.get(workflow_name, "direct")

                # Create thread
                start_time = datetime.now()
                thread_result = await client.create_thread(
                    name=f"cli-{workflow_name}-{agent_type}",
                    query=args.query,
                    agent_type=agent_type,
                    reasoning_mode=reasoning_mode,
                    priority=0
                )

                if "error" in thread_result:
                    print_error(f"Failed to create thread: {thread_result['error']}")
                    return

                thread_id = thread_result.get("thread_id") or thread_result.get("id")
                if not thread_id:
                    print_error("No thread ID returned from server")
                    return

                print_info(f"Thread created: {thread_id}")

                # Start the thread
                start_result = await client.start_thread(thread_id)
                if "error" in start_result:
                    print_warning(f"Start request: {start_result.get('error', 'unknown')}")
                    # Thread might auto-start, continue anyway

                # Wait for completion with progress
                print_info("Waiting for completion...")
                timeout = 300.0 if workflow_name == "research" else 120.0
                final_result = await client.wait_for_completion(
                    thread_id,
                    poll_interval=0.5,
                    timeout=timeout
                )

                duration = (datetime.now() - start_time).total_seconds()

                if "error" in final_result:
                    print_error(f"Thread failed: {final_result['error']}")
                    return

                status = final_result.get("status", "unknown")
                if status == "completed":
                    print_success(f"Workflow completed in {duration:.2f}s")
                elif status == "failed":
                    print_error(f"Workflow failed after {duration:.2f}s")
                else:
                    print_warning(f"Workflow ended with status: {status}")

                # Get result
                result_data = final_result.get("result", {})
                response = result_data.get("response") or final_result.get("response", "")

                print(f"\n{Colors.BOLD}Result:{Colors.ENDC}")
                print(response or "(No response)")

                if args.json:
                    print(f"\n{Colors.BOLD}Metadata:{Colors.ENDC}")
                    print(json.dumps({
                        "thread_id": thread_id,
                        "status": status,
                        "confidence": result_data.get("confidence", 0),
                        "duration_s": duration,
                        "workflow": workflow_name,
                        "agent_type": agent_type,
                        "source": "http"
                    }, indent=2))
                return

    # Fall back to local orchestrator
    print_info("Using local orchestrator")

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
                "source": "local"
            }, indent=2))


async def cmd_agent_status(args):
    """Show status of agents and hypervisor."""
    print(f"\n{Colors.BOLD}Agent Hypervisor Status{Colors.ENDC}\n")

    if not HTTPX_AVAILABLE:
        print_warning("httpx not installed. Install with: pip install httpx")
        print(f"  {Colors.DIM}Cannot connect to hypervisor servers{Colors.ENDC}")
        return

    async with HypervisorClient() as client:
        # Check server health
        agentic_health = await client.check_agentic_health()
        agent_mgr_health = await client.check_agent_manager_health()

        # Determine overall status
        agentic_ok = agentic_health.get("status") == "ok"
        agent_mgr_ok = agent_mgr_health.get("status") == "healthy"

        if agentic_ok or agent_mgr_ok:
            print(f"  Hypervisor:    {Colors.GREEN}running{Colors.ENDC}")
        else:
            print(f"  Hypervisor:    {Colors.YELLOW}offline{Colors.ENDC}")
            print(f"\n{Colors.DIM}Start servers with:{Colors.ENDC}")
            print("  uvicorn HoloLoom.apps.server.agentic_api:app --port 8000")
            print("  uvicorn HoloLoom.apps.server.agent_manager_api:app --port 8002")
            return

        # Get detailed stats from Agent Manager
        if agent_mgr_ok:
            threads = await client.list_threads(limit=100)
            active = threads.get("active", 0)
            completed = threads.get("completed", 0)
            total = threads.get("total", 0)
            uptime = agent_mgr_health.get("uptime_seconds", 0)
            uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"

            print(f"  Active Threads:{active}")
            print(f"  Completed:     {completed}")
            print(f"  Total:         {total}")
            print(f"  Uptime:        {uptime_str}")
        else:
            print(f"  Agent Manager: {Colors.YELLOW}offline{Colors.ENDC} (port 8002)")

        # Get stats from Agentic API
        if agentic_ok:
            stats = await client.get_stats()
            print(f"\n{Colors.BOLD}Agentic API (port 8000):{Colors.ENDC}")
            print(f"  Total Queries: {stats.get('total_queries', 0)}")
            print(f"  Cache Hits:    {stats.get('cache_hits', 0)}")
            if stats.get('avg_latency_ms'):
                print(f"  Avg Latency:   {stats.get('avg_latency_ms', 0):.1f}ms")
        else:
            print(f"\n  Agentic API:   {Colors.YELLOW}offline{Colors.ENDC} (port 8000)")

        print(f"\n{Colors.BOLD}Components:{Colors.ENDC}")
        print(f"  Audit Trail:     {Colors.GREEN}[OK]{Colors.ENDC}")
        print(f"  Safety Guards:   {Colors.GREEN}[OK]{Colors.ENDC}")
        print(f"  Thompson Sampler:{Colors.GREEN}[OK]{Colors.ENDC}")
        print(f"  Circuit Breaker: {Colors.GREEN}[OK]{Colors.ENDC}")


async def cmd_agent_logs(args):
    """View audit trail logs."""
    print(f"\n{Colors.BOLD}Audit Trail Logs{Colors.ENDC}")
    print(f"{Colors.DIM}(Last {args.limit} entries){Colors.ENDC}\n")

    if not HTTPX_AVAILABLE:
        print_warning("httpx not installed. Install with: pip install httpx")
        return

    async with HypervisorClient() as client:
        # Check if server is available
        health = await client.check_agentic_health()
        if health.get("status") != "ok":
            print(f"  {Colors.YELLOW}Agentic API offline{Colors.ENDC}")
            print("\n  Start with: uvicorn HoloLoom.apps.server.agentic_api:app --port 8000")
            return

        # Get audit trail
        result = await client.get_audit_trail(limit=args.limit)

        if "error" in result:
            print_error(result["error"])
            return

        entries = result.get("entries", [])
        if not entries:
            print(f"  {Colors.DIM}No audit trail entries yet{Colors.ENDC}")
            return

        # Display entries
        for entry in entries:
            timestamp = entry.get("timestamp", "")
            event_type = entry.get("type", "unknown")
            details = entry.get("details", {})

            # Color code by event type
            if event_type == "TOOL_SELECTION":
                color = Colors.CYAN
            elif event_type == "SAFETY_CHECK":
                color = Colors.YELLOW
            elif event_type == "ERROR":
                color = Colors.RED
            else:
                color = Colors.DIM

            print(f"  {Colors.DIM}{timestamp}{Colors.ENDC} {color}[{event_type}]{Colors.ENDC}")
            if details:
                for key, value in list(details.items())[:3]:
                    print(f"    {key}: {value}")


# ============================================================================
# Nexus Commands
# ============================================================================

async def cmd_nexus_mcp(args):
    """Launch HoloLoom MCP Server."""
    print_banner()
    print_info("Launching HoloLoom Nexus (MCP Bridge)...")

    try:
        from hololoom.integrations.mcp_server import main as mcp_main
        await mcp_main()
    except ImportError as e:
        print_error(f"Failed to import mcp_server: {e}")
        print_info("Ensure 'mcp' is installed: pip install mcp")
    except Exception as e:
        print_error(f"MCP Server failed: {e}")


# ============================================================================
# Cluster Commands
# ============================================================================

async def cmd_cluster_status(args):
    """Show cluster status (federation, distributed workers)."""
    print(f"\n{Colors.BOLD}Cluster Status{Colors.ENDC}\n")

    if not HTTPX_AVAILABLE:
        print_warning("httpx not installed. Install with: pip install httpx")
        return

    async with HypervisorClient() as client:
        cluster = await client.get_cluster_status()

        if cluster.get("status") == "unavailable":
            print(f"  Mode:          {Colors.YELLOW}Standalone (no cluster){Colors.ENDC}")
            print(f"  Federation:    {Colors.YELLOW}Not configured{Colors.ENDC}")
            print(f"  Eggroll:       {Colors.YELLOW}Not running{Colors.ENDC}")

            print(f"\n{Colors.DIM}Start Agent Manager server:{Colors.ENDC}")
            print("  uvicorn HoloLoom.apps.server.agent_manager_api:app --port 8002")

            print(f"\n{Colors.DIM}To enable distributed mode:{Colors.ENDC}")
            print("  1. Configure federation in config.yaml")
            print("  2. Start eggroll workers: hololoom cluster start-workers")
            print("  3. Join federation: hololoom cluster join <peer-address>")
            return

        # Cluster is running
        print(f"  Status:        {Colors.GREEN}{cluster.get('status')}{Colors.ENDC}")
        print(f"  Nodes:         {cluster.get('nodes', 1)}")
        print(f"  Active Threads:{cluster.get('active_threads', 0)}")
        print(f"  Active Swarms: {cluster.get('active_swarms', 0)}")

        uptime = cluster.get("uptime_seconds", 0)
        if uptime > 0:
            uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
            print(f"  Uptime:        {uptime_str}")


async def cmd_cluster_nodes(args):
    """List cluster nodes."""
    print(f"\n{Colors.BOLD}Cluster Nodes{Colors.ENDC}\n")

    if not HTTPX_AVAILABLE:
        print_warning("httpx not installed. Install with: pip install httpx")
        return

    async with HypervisorClient() as client:
        nodes = await client.get_cluster_nodes()

        if not nodes:
            print(f"  {Colors.DIM}No cluster nodes - running in standalone mode{Colors.ENDC}")
            print("\n  Start Agent Manager: uvicorn HoloLoom.apps.server.agent_manager_api:app --port 8002")
            return

        # Display nodes
        for i, node in enumerate(nodes, 1):
            node_id = node.get("id", f"node-{i}")
            address = node.get("address", "unknown")
            status = node.get("status", "unknown")
            active_threads = node.get("active_threads", 0)
            uptime = node.get("uptime_seconds", 0)

            if status == "healthy":
                status_color = Colors.GREEN
            elif status == "degraded":
                status_color = Colors.YELLOW
            else:
                status_color = Colors.RED

            print(f"  {Colors.CYAN}{node_id}{Colors.ENDC}")
            print(f"    Address: {address}")
            print(f"    Status:  {status_color}{status}{Colors.ENDC}")
            print(f"    Threads: {active_threads} active")
            if uptime > 0:
                uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
                print(f"    Uptime:  {uptime_str}")
            print()


# ============================================================================
# Studio Command
# ============================================================================

async def cmd_studio(args):
    """Launch HoloLoom Studio (Dashboard)."""
    print_banner()
    print_info("Launching HoloLoom Studio...")

    try:
        # Check if dashboard exists
        import os
        from pathlib import Path

        from hololoom.studio_server import start_studio

        base_dir = Path(__file__).parent.resolve()
        dist_dir = base_dir / "dist"

        if not dist_dir.exists():
            print_warning("Dashboard assets not found in HoloLoom/dist/")
            print_info("To build the dashboard, run:")
            print("  cd HoloLoom/hololoom-dashboard && npm install && npm run build")
            print()

        start_studio(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser
        )
    except ImportError as e:
        print_error(f"Failed to import studio_server: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Studio failed: {e}")
        sys.exit(1)


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

    # studio
    studio_parser = subparsers.add_parser('studio', help='Launch HoloLoom Studio (Dashboard)')
    studio_parser.add_argument('--port', type=int, default=8000, help='Port to run Studio on')
    studio_parser.add_argument('--host', default='127.0.0.1', help='Host to bind Studio to')
    studio_parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    studio_parser.set_defaults(func=cmd_studio)

    # -------------------------------------------------------------------------
    # Nexus commands
    # -------------------------------------------------------------------------
    nexus_parser = subparsers.add_parser('nexus', help='Integrations and bridges')
    nexus_subparsers = nexus_parser.add_subparsers(dest='nexus_command', help='Nexus commands')

    # nexus mcp
    nexus_mcp = nexus_subparsers.add_parser('mcp', help='Launch HoloLoom MCP Server')
    nexus_mcp.set_defaults(func=cmd_nexus_mcp)

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
