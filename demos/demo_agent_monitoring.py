#!/usr/bin/env python3
"""
Agent Monitoring Demo
=====================
Demonstrates the real-time agent monitoring system with multiple concurrent agents.

Shows:
- Multiple agents running across different projects
- WebSocket-based real-time updates
- Tree structure visualization
- Performance metrics tracking

Usage:
    # Terminal 1: Start the server
    PYTHONPATH=. uvicorn HoloLoom.apps.server.agentic_api:app --port 8000

    # Terminal 2: Run this demo
    PYTHONPATH=. python demos/demo_agent_monitoring.py
"""

import asyncio
import json
from datetime import datetime
import websockets
from typing import List, Dict, Any

from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard
from hololoom.agentic.monitoring import get_monitor, start_monitoring


# ============================================================================
# Demo Scenarios
# ============================================================================

DEMO_SCENARIOS = [
    {
        "project": "mythRL",
        "query": "Explain Thompson Sampling and its tradeoffs",
        "mode": ReasoningMode.RESEARCH,
        "description": "Research mode with multi-query exploration"
    },
    {
        "project": "squad",
        "query": "How do we integrate D3.js tree visualization?",
        "mode": ReasoningMode.VERIFY,
        "description": "Verification mode with answer validation"
    },
    {
        "project": "elle",
        "query": "Test AR adapter scene recognition",
        "mode": ReasoningMode.PLAN_EXECUTE,
        "description": "Plan & execute mode with sub-goal decomposition"
    }
]


def create_test_shards() -> List[MemoryShard]:
    """Create example memory shards for demo."""
    return [
        MemoryShard(
            id="ts_1",
            text="Thompson Sampling is a Bayesian approach to the exploration-exploitation dilemma in reinforcement learning.",
            episode="rl_concepts",
            entities=["Thompson Sampling", "Bayesian", "exploration", "exploitation"],
            motifs=["definition", "algorithm"]
        ),
        MemoryShard(
            id="ts_2",
            text="Thompson Sampling balances exploration and exploitation by sampling from posterior distributions.",
            episode="rl_concepts",
            entities=["Thompson Sampling", "posterior distributions", "bandit algorithms"],
            motifs=["mechanism"]
        ),
        MemoryShard(
            id="d3_1",
            text="D3.js is a JavaScript library for manipulating documents based on data.",
            episode="visualization",
            entities=["D3.js", "JavaScript", "visualization"],
            motifs=["definition", "tool"]
        ),
        MemoryShard(
            id="ar_1",
            text="AR (Augmented Reality) adapters overlay digital information onto the physical world.",
            episode="ar_guide",
            entities=["AR", "augmented reality", "scene recognition"],
            motifs=["definition", "technology"]
        )
    ]


# ============================================================================
# WebSocket Monitor Client
# ============================================================================

class MonitorClient:
    """WebSocket client for monitoring agent updates."""

    def __init__(self, url: str = "ws://localhost:8000/ws/monitor"):
        self.url = url
        self.websocket = None
        self.messages: List[Dict[str, Any]] = []

    async def connect(self):
        """Connect to monitoring WebSocket."""
        try:
            self.websocket = await websockets.connect(self.url)
            print(f"✅ Connected to monitoring WebSocket: {self.url}")
            return True
        except Exception as e:
            print(f"⚠️  Could not connect to WebSocket: {e}")
            print("   Make sure server is running: uvicorn HoloLoom.apps.server.agentic_api:app --port 8000")
            return False

    async def listen(self, duration: float = 30.0):
        """Listen for updates for specified duration."""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return

        print(f"\n📡 Listening for agent updates (duration: {duration}s)...\n")
        start_time = asyncio.get_event_loop().time()

        try:
            while asyncio.get_event_loop().time() - start_time < duration:
                try:
                    # Wait for message with timeout
                    message_str = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0
                    )
                    message = json.loads(message_str)
                    self.messages.append(message)
                    self._print_message(message)

                except asyncio.TimeoutError:
                    continue  # No message within timeout, continue listening
                except websockets.exceptions.ConnectionClosed:
                    print("❌ WebSocket connection closed")
                    break

        except KeyboardInterrupt:
            print("\n⏹️  Stopped listening")

    def _print_message(self, message: Dict[str, Any]):
        """Pretty-print monitoring message."""
        msg_type = message.get('type', 'unknown')
        timestamp = message.get('timestamp', datetime.now().isoformat())

        if msg_type == 'agent_started':
            agent_id = message['agent_id']
            project = message['project']
            query = message['query']
            mode = message['mode']
            print(f"🚀 [{timestamp}] Agent {agent_id} STARTED")
            print(f"   Project: {project}")
            print(f"   Mode: {mode}")
            print(f"   Query: {query[:60]}...")
            print()

        elif msg_type == 'agent_step':
            agent_id = message['agent_id']
            step = message['step']
            total = message['total_steps']
            step_type = message['step_type']
            confidence = message.get('confidence', 0.0)
            epistemic = message.get('epistemic')
            print(f"⚙️  [{timestamp}] Agent {agent_id} STEP {step}/{total}")
            print(f"   Type: {step_type}")
            print(f"   Confidence: {confidence:.2f}")
            if epistemic is not None:
                print(f"   Epistemic: {epistemic:.2f}")
            print()

        elif msg_type == 'agent_feed':
            agent_id = message['agent_id']
            line1 = message['line1']
            line2 = message.get('line2', '')
            print(f"💬 [{timestamp}] Agent {agent_id} FEED")
            print(f"   {line1}")
            if line2:
                print(f"   {line2}")
            print()

        elif msg_type == 'agent_status':
            agent_id = message['agent_id']
            status = message['status']
            print(f"📊 [{timestamp}] Agent {agent_id} STATUS → {status}")
            print()

        elif msg_type == 'agent_completed':
            agent_id = message['agent_id']
            duration = message['total_duration_ms']
            print(f"✅ [{timestamp}] Agent {agent_id} COMPLETED")
            print(f"   Duration: {duration:.0f}ms")
            print()

        elif msg_type == 'agent_failed':
            agent_id = message['agent_id']
            error = message['error']
            print(f"❌ [{timestamp}] Agent {agent_id} FAILED")
            print(f"   Error: {error}")
            print()

        elif msg_type == 'connected':
            print(f"🔗 [{timestamp}] {message['message']}")
            print(f"   Active agents: {message['active_agents']}")
            print(f"   Projects: {', '.join(message['projects']) if message['projects'] else 'None'}")
            print()

    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            print("🔌 WebSocket connection closed")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of received messages."""
        by_type = {}
        for msg in self.messages:
            msg_type = msg.get('type', 'unknown')
            by_type[msg_type] = by_type.get(msg_type, 0) + 1

        return {
            "total_messages": len(self.messages),
            "by_type": by_type,
            "agents_started": by_type.get('agent_started', 0),
            "agents_completed": by_type.get('agent_completed', 0),
            "agents_failed": by_type.get('agent_failed', 0),
            "steps_executed": by_type.get('agent_step', 0)
        }


# ============================================================================
# Demo Runner
# ============================================================================

async def run_agent(
    scenario: Dict[str, Any],
    orchestrator,
    delay: float = 0.0
):
    """Run a single agent scenario."""
    await asyncio.sleep(delay)  # Stagger agent starts

    print(f"\n🤖 Starting agent for {scenario['project']}: {scenario['description']}")

    try:
        result = await orchestrator.reason(
            query=Query(text=scenario['query']),
            mode=scenario['mode'],
            max_steps=3,  # Limit steps for demo
            project=scenario['project']
        )

        print(f"✅ Agent completed for {scenario['project']}")
        print(f"   Confidence: {result.spacetime.confidence:.2f}")
        print(f"   Steps: {result.total_queries}")
        if result.aggregated_epistemic_confidence is not None:
            print(f"   Epistemic: {result.aggregated_epistemic_confidence:.2f}")

    except Exception as e:
        print(f"❌ Agent failed for {scenario['project']}: {e}")


async def main():
    """
    Main demo: Run multiple agents concurrently while monitoring via WebSocket.
    """
    print("=" * 80)
    print("Agent Monitoring Demo")
    print("=" * 80)
    print()
    print("This demo simulates multiple agents running across different projects:")
    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"{i}. {scenario['project']}: {scenario['description']}")
    print()
    print("=" * 80)
    print()

    # Initialize monitoring
    print("🔧 Initializing monitoring system...")
    monitor = get_monitor()
    await start_monitoring()
    print("✅ Monitoring initialized")
    print()

    # Create orchestrator with monitoring
    print("🔧 Creating orchestrator...")
    config = Config.fast()
    shards = create_test_shards()
    orchestrator = await create_agentic_orchestrator(
        config,
        shards,
        enable_verification=True,
        enable_goal_tracking=True,
        monitor=monitor  # Pass monitor instance
    )
    print("✅ Orchestrator ready")
    print()

    # Create WebSocket monitor client
    client = MonitorClient()
    connected = await client.connect()

    if connected:
        # Run agents concurrently while monitoring
        async def monitor_and_run():
            # Start listening in background
            listen_task = asyncio.create_task(client.listen(duration=30.0))

            # Small delay to let WebSocket connection establish
            await asyncio.sleep(0.5)

            # Run all agents concurrently (stagger starts by 2 seconds each)
            agent_tasks = [
                run_agent(scenario, orchestrator, delay=i * 2.0)
                for i, scenario in enumerate(DEMO_SCENARIOS)
            ]

            # Wait for all agents to complete
            await asyncio.gather(*agent_tasks)

            # Let monitor catch final messages
            await asyncio.sleep(2.0)

            # Stop listening
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

        await monitor_and_run()

        # Print summary
        print("\n" + "=" * 80)
        print("Monitoring Summary")
        print("=" * 80)
        summary = client.get_summary()
        print(f"Total messages: {summary['total_messages']}")
        print(f"Agents started: {summary['agents_started']}")
        print(f"Agents completed: {summary['agents_completed']}")
        print(f"Agents failed: {summary['agents_failed']}")
        print(f"Steps executed: {summary['steps_executed']}")
        print()

        # Close WebSocket
        await client.close()

    else:
        print("⚠️  WebSocket not available. Running agents without monitoring...")
        # Run agents anyway for testing
        for scenario in DEMO_SCENARIOS:
            await run_agent(scenario, orchestrator)

    # Cleanup
    await orchestrator.close()
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
