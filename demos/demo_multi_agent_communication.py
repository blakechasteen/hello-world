"""
Demo: Multi-Agent Communication Protocol (Phase 7.1)
====================================================

Showcases HoloLoom's multi-agent coordination infrastructure:
1. Agent registration and discovery
2. Message passing (request/response)
3. Shared memory with namespaces
4. Multi-agent research workflows
5. Load balancing strategies

Author: Claude Code
Date: 2025-12-09
"""

import asyncio
import time
from typing import Dict, Any, Optional

# Phase 7.1 imports
from hololoom.agentic import (
    # Enums
    AgentCapability,
    MessageType,
    MessagePriority,
    AgentStatus,
    # Classes
    AgentMessage,
    AgentInfo,
    AgentRegistry,
    MessageBus,
    SharedMemory,
    MultiAgentCoordinator,
    # Convenience functions
    create_coordinator,
    create_agent_message,
    create_research_agent,
    create_synthesis_agent,
    create_code_agent,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# Demo 1: Agent Registration and Discovery
# =============================================================================

async def demo_agent_registration():
    """Demonstrate agent registration and capability-based discovery."""
    print_header("Demo 1: Agent Registration and Discovery")

    # Create registry
    registry = AgentRegistry()

    # Register agents with different capabilities
    print_subheader("Registering Agents")

    # Research specialist
    research_agent = AgentInfo(
        id="research-001",
        name="Deep Research Bot",
        capabilities={
            AgentCapability.RESEARCH,
            AgentCapability.ANALYSIS,
            AgentCapability.VERIFICATION
        }
    )
    await registry.register(research_agent)
    print(f"  Registered: {research_agent.name} ({research_agent.id})")
    print(f"    Capabilities: {[c.value for c in research_agent.capabilities]}")

    # Code specialist
    code_agent = AgentInfo(
        id="code-001",
        name="Code Review Bot",
        capabilities={
            AgentCapability.CODE_REVIEW,
            AgentCapability.CODE_GENERATION,
            AgentCapability.DEBUGGING
        }
    )
    await registry.register(code_agent)
    print(f"  Registered: {code_agent.name} ({code_agent.id})")
    print(f"    Capabilities: {[c.value for c in code_agent.capabilities]}")

    # Synthesis specialist
    synthesis_agent = AgentInfo(
        id="synth-001",
        name="Synthesis Bot",
        capabilities={
            AgentCapability.SYNTHESIS,
            AgentCapability.SUMMARIZATION,
            AgentCapability.WRITING
        }
    )
    await registry.register(synthesis_agent)
    print(f"  Registered: {synthesis_agent.name} ({synthesis_agent.id})")
    print(f"    Capabilities: {[c.value for c in synthesis_agent.capabilities]}")

    # Discover agents by capability (sync methods)
    print_subheader("Finding Agents by Capability")

    # Find research-capable agents
    research_agents = registry.find_by_capability(AgentCapability.RESEARCH)
    print(f"  RESEARCH capable: {[a.name for a in research_agents]}")

    # Find code review agents
    code_agents = registry.find_by_capability(AgentCapability.CODE_REVIEW)
    print(f"  CODE_REVIEW capable: {[a.name for a in code_agents]}")

    # Find synthesis agents
    synth_agents = registry.find_by_capability(AgentCapability.SYNTHESIS)
    print(f"  SYNTHESIS capable: {[a.name for a in synth_agents]}")

    # List all agents
    print_subheader("All Registered Agents")
    for agent in registry.get_all_agents():
        print(f"  [{agent.id}] {agent.name} - Status: {agent.status.value}")

    print(f"\n  Total agents: {len(registry.get_all_agents())}")

    return registry


# =============================================================================
# Demo 2: Message Passing
# =============================================================================

async def demo_message_passing():
    """Demonstrate async message passing between agents."""
    print_header("Demo 2: Message Passing (Request/Response)")

    # Create registry and bus
    registry = AgentRegistry()
    bus = MessageBus(registry)

    # Register agents
    agent_a = AgentInfo(
        id="agent-a",
        name="Coordinator",
        capabilities={AgentCapability.COORDINATION, AgentCapability.PLANNING}
    )
    agent_b = AgentInfo(
        id="agent-b",
        name="Worker",
        capabilities={AgentCapability.RESEARCH, AgentCapability.ANALYSIS}
    )
    await registry.register(agent_a)
    await registry.register(agent_b)

    # Track received messages
    received_messages = []

    # Define message handler for agent-b
    async def worker_handler(msg: AgentMessage) -> Optional[AgentMessage]:
        """Handler that simulates processing and RETURNS response.

        NOTE: The request() method expects handlers to RETURN the response,
        not send it via bus.send(). The request() method wraps the handler
        and captures the returned response.
        """
        print(f"\n  Worker received: {msg.content.get('task', 'unknown')}")
        received_messages.append(msg)

        # Simulate processing
        await asyncio.sleep(0.1)

        # Create and RETURN response (not send!)
        response = msg.create_response(
            content={
                "result": "Analysis complete",
                "findings": ["Finding A", "Finding B", "Finding C"],
                "confidence": 0.92
            },
            from_agent="agent-b"
        )
        print(f"  Worker returning response: {response.content.get('result')}")
        return response  # IMPORTANT: Return, don't send!

    # Subscribe worker to messages (sync method)
    bus.subscribe("agent-b", worker_handler)

    print_subheader("Sending Request from Coordinator to Worker")

    # Create and send request
    request = create_agent_message(
        from_agent="agent-a",
        to_agent="agent-b",
        msg_type=MessageType.REQUEST,
        content={
            "task": "analyze",
            "query": "What is Thompson Sampling?",
            "context": {"domain": "machine_learning"}
        },
        priority=MessagePriority.NORMAL
    )

    print(f"  Coordinator sending request...")
    print(f"    Message ID: {request.id}")
    print(f"    Task: {request.content.get('task')}")
    print(f"    Query: {request.content.get('query')}")

    # Send and wait for response
    try:
        response = await bus.request(request, timeout=5.0)

        print_subheader("Response Received")
        print(f"  Result: {response.content.get('result')}")
        print(f"  Findings: {response.content.get('findings')}")
        print(f"  Confidence: {response.content.get('confidence')}")

    except TimeoutError:
        print("  Request timed out!")

    print(f"\n  Messages exchanged: {len(received_messages)}")


# =============================================================================
# Demo 3: Shared Memory
# =============================================================================

async def demo_shared_memory():
    """Demonstrate shared memory with namespaces and TTL."""
    print_header("Demo 3: Shared Memory with Namespaces")

    memory = SharedMemory()

    print_subheader("Storing Context in Shared Memory")

    # Store research context in 'research' namespace
    await memory.put(
        key="query_context",
        value={
            "query": "Explain Thompson Sampling",
            "domain": "machine_learning",
            "complexity": "moderate"
        },
        owner="coordinator",  # Required: agent that owns this entry
        namespace="research"
    )
    print("  Stored 'query_context' in 'research' namespace (owner: coordinator)")

    # Store findings in 'findings' namespace
    await memory.put(
        key="agent_a_findings",
        value=["Thompson Sampling is Bayesian", "Uses Beta distributions"],
        owner="agent-a",  # Each agent owns their findings
        namespace="findings"
    )
    print("  Stored 'agent_a_findings' in 'findings' namespace (owner: agent-a)")

    await memory.put(
        key="agent_b_findings",
        value=["Balances exploration/exploitation", "Optimal in many cases"],
        owner="agent-b",  # Each agent owns their findings
        namespace="findings"
    )
    print("  Stored 'agent_b_findings' in 'findings' namespace (owner: agent-b)")

    # Store temporary data with TTL
    await memory.put(
        key="temp_cache",
        value={"cached": True, "data": [1, 2, 3]},
        owner="cache-service",
        namespace="temp",
        ttl_seconds=60  # Expires in 60 seconds (int, not float)
    )
    print("  Stored 'temp_cache' with 60s TTL in 'temp' namespace")

    print_subheader("Retrieving from Shared Memory")

    # Retrieve research context
    # NOTE: get() returns the value directly, not a SharedMemoryEntry
    value = await memory.get("query_context", namespace="research")
    if value:
        print(f"  query_context: {value}")

    # Retrieve all findings
    # NOTE: get_namespace() returns Dict[str, value], not Dict[str, SharedMemoryEntry]
    findings_values = await memory.get_namespace("findings")
    print(f"  Findings namespace has {len(findings_values)} entries:")
    for key, value in findings_values.items():
        print(f"    - {key}: {value}")

    print_subheader("Namespace Isolation")

    # Try to get research data from wrong namespace
    wrong_ns = await memory.get("query_context", namespace="findings")
    print(f"  'query_context' in 'findings' namespace: {wrong_ns}")  # Should be None

    # Get statistics (sync method)
    print_subheader("Memory Statistics")
    stats = memory.get_statistics()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Namespaces: {stats['namespaces']}")


# =============================================================================
# Demo 4: Multi-Agent Research Workflow
# =============================================================================

async def demo_multi_agent_workflow():
    """Demonstrate a complete multi-agent research workflow."""
    print_header("Demo 4: Multi-Agent Research Workflow")

    # Create coordinator
    coordinator = create_coordinator()

    # Results storage (must be defined before handlers)
    results: Dict[str, Any] = {}

    # Define handlers FIRST (required for register_agent)
    async def research_handler(msg: AgentMessage) -> Optional[AgentMessage]:
        """Simulate research agent processing. Returns response."""
        await asyncio.sleep(0.1)  # Simulate work
        results["research"] = {
            "sources": ["Paper A", "Paper B", "Documentation"],
            "key_points": [
                "Thompson Sampling is a Bayesian approach",
                "It maintains Beta(alpha, beta) distributions",
                "Balances exploration and exploitation"
            ],
            "confidence": 0.89
        }
        # Return response (for request/response pattern)
        return msg.create_response(
            content=results["research"],
            from_agent="research-specialist"
        )

    async def synthesis_handler(msg: AgentMessage) -> Optional[AgentMessage]:
        """Simulate synthesis agent processing. Returns response."""
        await asyncio.sleep(0.1)  # Simulate work
        results["synthesis"] = {
            "summary": "Thompson Sampling uses Bayesian probability to balance exploring new options vs exploiting known good ones.",
            "length": "concise",
            "quality": 0.92
        }
        return msg.create_response(
            content=results["synthesis"],
            from_agent="synthesis-specialist"
        )

    async def code_handler(msg: AgentMessage) -> Optional[AgentMessage]:
        """Simulate code agent processing. Returns response."""
        await asyncio.sleep(0.1)  # Simulate work
        results["code"] = {
            "implementation": "def thompson_sample(alpha, beta): return np.random.beta(alpha, beta)",
            "language": "python",
            "tested": True
        }
        return msg.create_response(
            content=results["code"],
            from_agent="code-specialist"
        )

    # Register agents WITH handlers
    print_subheader("Setting Up Agent Team")

    research = AgentInfo(
        id="research-specialist",
        name="Research Specialist",
        capabilities={
            AgentCapability.RESEARCH,
            AgentCapability.VERIFICATION,
            AgentCapability.ANALYSIS
        }
    )
    await coordinator.register_agent(research, research_handler)
    print(f"  Registered: {research.name}")

    synthesis = AgentInfo(
        id="synthesis-specialist",
        name="Synthesis Specialist",
        capabilities={
            AgentCapability.SYNTHESIS,
            AgentCapability.SUMMARIZATION,
            AgentCapability.WRITING
        }
    )
    await coordinator.register_agent(synthesis, synthesis_handler)
    print(f"  Registered: {synthesis.name}")

    code = AgentInfo(
        id="code-specialist",
        name="Code Specialist",
        capabilities={
            AgentCapability.CODE_GENERATION,
            AgentCapability.CODE_REVIEW,
            AgentCapability.DEBUGGING
        }
    )
    await coordinator.register_agent(code, code_handler)
    print(f"  Registered: {code.name}")

    # Store shared context
    print_subheader("Sharing Context with All Agents")
    await coordinator.memory.put(
        key="shared_query",
        value={
            "topic": "Thompson Sampling",
            "goal": "Comprehensive explanation with code",
            "audience": "intermediate developers"
        },
        owner="coordinator",  # Required: who owns this entry
        namespace="workflow"
    )
    print("  Stored shared query context")

    # Delegate tasks to agents
    # NOTE: delegate_task signature is:
    #   delegate_task(task, capability, from_agent, context, strategy, timeout)
    print_subheader("Delegating Tasks")

    # Research phase
    print("\n  [Phase 1: Research]")
    research_response = await coordinator.delegate_task(
        task={"action": "research", "topic": "Thompson Sampling"},
        capability=AgentCapability.RESEARCH,
        timeout=5.0
    )
    if research_response:
        print(f"    Sources found: {len(results.get('research', {}).get('sources', []))}")
        print(f"    Key points: {len(results.get('research', {}).get('key_points', []))}")
        print(f"    Confidence: {results.get('research', {}).get('confidence', 0):.2f}")

    # Synthesis phase
    print("\n  [Phase 2: Synthesis]")
    synthesis_response = await coordinator.delegate_task(
        task={"action": "synthesize", "input": results.get("research", {})},
        capability=AgentCapability.SYNTHESIS,
        timeout=5.0
    )
    if synthesis_response:
        summary = results.get("synthesis", {}).get("summary", "")
        print(f"    Summary: {summary[:80]}...")
        print(f"    Quality: {results.get('synthesis', {}).get('quality', 0):.2f}")

    # Code generation phase
    print("\n  [Phase 3: Code Generation]")
    code_response = await coordinator.delegate_task(
        task={"action": "generate_code", "topic": "Thompson Sampling"},
        capability=AgentCapability.CODE_GENERATION,
        timeout=5.0
    )
    if code_response:
        print(f"    Language: {results.get('code', {}).get('language', 'unknown')}")
        print(f"    Tested: {results.get('code', {}).get('tested', False)}")
        impl = results.get("code", {}).get("implementation", "")
        print(f"    Code: {impl[:60]}...")

    # Get workflow statistics (sync method)
    # NOTE: coordinator.get_statistics() returns nested dict:
    # {
    #     "registry": {"total_agents", "available_agents", "total_requests", ...},
    #     "memory": {"total_entries", "namespaces", "total_access_count", ...},
    #     "bus": {"total_messages", "subscribed_agents", ...}
    # }
    print_subheader("Workflow Statistics")
    stats = coordinator.get_statistics()
    print(f"  Total agents: {stats['registry']['total_agents']}")
    print(f"  Available agents: {stats['registry']['available_agents']}")
    print(f"  Total requests: {stats['registry']['total_requests']}")
    print(f"  Memory entries: {stats['memory']['total_entries']}")
    print(f"  Memory namespaces: {stats['memory']['namespaces']}")
    print(f"  Total messages: {stats['bus']['total_messages']}")


# =============================================================================
# Demo 5: Load Balancing Strategies
# =============================================================================

async def demo_load_balancing():
    """Demonstrate different agent selection strategies."""
    print_header("Demo 5: Load Balancing Strategies")

    registry = AgentRegistry()

    # Create multiple agents with same capability but different performance
    print_subheader("Registering Agents with Different Performance Profiles")

    # Fast but less reliable
    fast_agent = AgentInfo(
        id="fast-001",
        name="Fast Agent",
        capabilities={AgentCapability.RESEARCH}
    )
    # Simulate performance history
    for _ in range(10):
        fast_agent.update_metrics(50.0, success=True)  # 50ms, always succeeds
    for _ in range(3):
        fast_agent.update_metrics(50.0, success=False)  # Some failures
    await registry.register(fast_agent)
    print(f"  {fast_agent.name}: avg={fast_agent.avg_response_time_ms:.0f}ms, success={fast_agent.success_rate:.1%}")

    # Slower but more reliable
    reliable_agent = AgentInfo(
        id="reliable-001",
        name="Reliable Agent",
        capabilities={AgentCapability.RESEARCH}
    )
    for _ in range(15):
        reliable_agent.update_metrics(150.0, success=True)  # 150ms, always succeeds
    await registry.register(reliable_agent)
    print(f"  {reliable_agent.name}: avg={reliable_agent.avg_response_time_ms:.0f}ms, success={reliable_agent.success_rate:.1%}")

    # Balanced
    balanced_agent = AgentInfo(
        id="balanced-001",
        name="Balanced Agent",
        capabilities={AgentCapability.RESEARCH}
    )
    for _ in range(10):
        balanced_agent.update_metrics(100.0, success=True)  # 100ms
    for _ in range(1):
        balanced_agent.update_metrics(100.0, success=False)  # Rare failure
    await registry.register(balanced_agent)
    print(f"  {balanced_agent.name}: avg={balanced_agent.avg_response_time_ms:.0f}ms, success={balanced_agent.success_rate:.1%}")

    print_subheader("Agent Selection by Strategy")

    # Strategy: Fastest (sync method)
    best_fastest = registry.get_best_agent(AgentCapability.RESEARCH, strategy="fastest")
    if best_fastest:
        print(f"  FASTEST strategy: {best_fastest.name} ({best_fastest.avg_response_time_ms:.0f}ms)")

    # Strategy: Most reliable
    best_reliable = registry.get_best_agent(AgentCapability.RESEARCH, strategy="reliable")
    if best_reliable:
        print(f"  RELIABLE strategy: {best_reliable.name} ({best_reliable.success_rate:.1%})")

    # Strategy: Balanced (score = success_rate / (1 + response_time/1000))
    best_balanced = registry.get_best_agent(AgentCapability.RESEARCH, strategy="balanced")
    if best_balanced:
        score = best_balanced.success_rate / (1 + best_balanced.avg_response_time_ms / 1000)
        print(f"  BALANCED strategy: {best_balanced.name} (score={score:.3f})")

    # Strategy: Random
    print("\n  RANDOM strategy (5 selections):")
    for i in range(5):
        random_agent = registry.get_best_agent(AgentCapability.RESEARCH, strategy="random")
        if random_agent:
            print(f"    Selection {i+1}: {random_agent.name}")


# =============================================================================
# Demo 6: Using Convenience Functions
# =============================================================================

async def demo_convenience_functions():
    """Demonstrate the convenience functions for quick setup."""
    print_header("Demo 6: Convenience Functions")

    print_subheader("Pre-built Agent Templates")

    # Create research agent
    research = create_research_agent()
    print(f"  Research Agent: {research.name}")
    print(f"    ID: {research.id}")
    print(f"    Capabilities: {[c.value for c in research.capabilities]}")

    # Create synthesis agent
    synthesis = create_synthesis_agent()
    print(f"\n  Synthesis Agent: {synthesis.name}")
    print(f"    ID: {synthesis.id}")
    print(f"    Capabilities: {[c.value for c in synthesis.capabilities]}")

    # Create code agent
    code = create_code_agent()
    print(f"\n  Code Agent: {code.name}")
    print(f"    ID: {code.id}")
    print(f"    Capabilities: {[c.value for c in code.capabilities]}")

    print_subheader("Quick Message Creation")

    msg = create_agent_message(
        from_agent="coordinator",
        to_agent="worker",
        msg_type=MessageType.BROADCAST,
        content={"announcement": "System ready"},
        priority=MessagePriority.HIGH
    )
    print(f"  Message ID: {msg.id}")
    print(f"  Type: {msg.type.value}")  # NOTE: attribute is 'type', not 'msg_type'
    print(f"  Priority: {msg.priority.value}")
    print(f"  Content: {msg.content}")

    print_subheader("One-Line Coordinator Setup")

    coordinator = create_coordinator()
    print(f"  Coordinator created with:")
    print(f"    - AgentRegistry (for agent discovery)")
    print(f"    - MessageBus (for communication)")
    print(f"    - SharedMemory (for context sharing)")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  HoloLoom Phase 7.1: Multi-Agent Communication Protocol")
    print("=" * 60)

    # Demo 1: Registration and Discovery
    await demo_agent_registration()

    # Demo 2: Message Passing
    await demo_message_passing()

    # Demo 3: Shared Memory
    await demo_shared_memory()

    # Demo 4: Multi-Agent Workflow
    await demo_multi_agent_workflow()

    # Demo 5: Load Balancing
    await demo_load_balancing()

    # Demo 6: Convenience Functions
    await demo_convenience_functions()

    # Summary
    print_header("Summary")
    print("""
  Phase 7.1 Multi-Agent Communication Protocol provides:

  1. Agent Registry
     - Capability-based discovery
     - Status tracking (IDLE, BUSY, OFFLINE)
     - Performance metrics (latency, success rate)

  2. Message Bus
     - Async pub/sub messaging
     - Request/response with timeout
     - Priority-based routing
     - Message expiration

  3. Shared Memory
     - Namespace isolation
     - TTL-based expiration
     - Cross-agent context sharing

  4. Load Balancing
     - Fastest (minimize latency)
     - Reliable (maximize success rate)
     - Balanced (optimize both)
     - Random (even distribution)

  5. Coordinator
     - High-level task delegation
     - Automatic agent selection
     - Statistics and monitoring

  Next: Phase 7.2 will add Task Delegation System with
  automatic query-to-agent routing and ensemble decisions.
""")


if __name__ == "__main__":
    asyncio.run(main())
