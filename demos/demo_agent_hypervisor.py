#!/usr/bin/env python3
"""
HoloLoom Agent Hypervisor Demo

Demonstrates the "Kubernetes for AI Agents" architecture:
- 3 specialized agents (Researcher, Verifier, Synthesizer)
- 2 model routing (Claude for reasoning, Ollama for fast tasks)
- Enterprise governance (budgets, audit trails, safety)
- Verifiable memory (Thompson Sampling learning)

This showcases HoloLoom's unique value: production-grade agent infrastructure
that no other framework provides.

Usage:
    PYTHONPATH=. python demos/demo_agent_hypervisor.py
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import random

# ANSI colors for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_banner():
    """Print the HoloLoom Agent Hypervisor banner."""
    banner = """
    +===============================================================+
    |                                                               |
    |     H O L O L O O M                                           |
    |                                                               |
    |              AGENT HYPERVISOR DEMONSTRATION                   |
    |              "Kubernetes for AI Agents"                       |
    |                                                               |
    +===============================================================+
    """
    print(f"{Colors.CYAN}{banner}{Colors.ENDC}")


def print_section(title: str, emoji: str = ""):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{emoji} {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'-'*60}{Colors.ENDC}")


def print_step(text: str, status: str = "OK"):
    """Print a step with status indicator."""
    if status == "OK":
        icon = f"{Colors.GREEN}[OK]{Colors.ENDC}"
    elif status == "WARN":
        icon = f"{Colors.YELLOW}[!]{Colors.ENDC}"
    elif status == "ERROR":
        icon = f"{Colors.RED}[X]{Colors.ENDC}"
    else:
        icon = f"{Colors.CYAN}[-]{Colors.ENDC}"
    print(f"  {icon} {text}")


# ============================================================================
# SIMULATED INFRASTRUCTURE (Production would use real components)
# ============================================================================

class ModelProvider(Enum):
    """Model providers for multi-model routing."""
    CLAUDE = "anthropic/claude-3-5-sonnet"
    OLLAMA = "ollama/llama3.2:3b"
    GPT4V = "openai/gpt-4v"


@dataclass
class AgentBudget:
    """Budget constraints for an agent."""
    max_tokens: int = 10000
    max_time_seconds: int = 60
    tokens_used: int = 0
    time_used: float = 0.0

    def consume(self, tokens: int, time_s: float):
        self.tokens_used += tokens
        self.time_used += time_s

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def time_remaining(self) -> float:
        return max(0, self.max_time_seconds - self.time_used)

    @property
    def budget_exceeded(self) -> bool:
        return self.tokens_used >= self.max_tokens or self.time_used >= self.max_time_seconds


@dataclass
class AuditEntry:
    """A single audit trail entry."""
    timestamp: float
    agent_id: str
    action: str
    input_summary: str
    output_summary: str
    model_used: str
    tokens_used: int
    latency_ms: float
    risk_level: str
    success: bool


@dataclass
class ThompsonSamplerState:
    """Thompson Sampling state for a single agent."""
    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0   # Failures + 1

    def update(self, success: bool, confidence: float = 1.0):
        if success:
            self.alpha += confidence
        else:
            self.beta += (1 - confidence)

    @property
    def expected_reward(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        # Beta distribution sample (simplified)
        return random.betavariate(self.alpha, self.beta)


class SimulatedAgent:
    """A simulated agent for the demo."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        preferred_model: ModelProvider,
        budget: AgentBudget = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.preferred_model = preferred_model
        self.budget = budget or AgentBudget()
        self.thompson = ThompsonSamplerState()

    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task (simulated)."""
        # Simulate processing time
        latency_ms = random.uniform(100, 500)
        await asyncio.sleep(latency_ms / 1000)

        # Simulate token usage
        tokens = random.randint(500, 2000)

        # Consume budget
        self.budget.consume(tokens, latency_ms / 1000)

        # Simulate success/quality
        confidence = random.uniform(0.7, 0.95)
        success = random.random() < 0.9  # 90% success rate

        # Update Thompson Sampling
        self.thompson.update(success, confidence)

        return {
            "agent_id": self.agent_id,
            "task": task,
            "model_used": self.preferred_model.value,
            "tokens_used": tokens,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "success": success,
            "response": f"[{self.name}] Completed: {task[:50]}..."
        }


class SimulatedAuditTrail:
    """Simulated audit trail for the demo."""

    def __init__(self):
        self.entries: List[AuditEntry] = []

    def log(self, entry: AuditEntry):
        self.entries.append(entry)

    def get_summary(self) -> Dict[str, Any]:
        if not self.entries:
            return {}
        return {
            "total_entries": len(self.entries),
            "total_tokens": sum(e.tokens_used for e in self.entries),
            "avg_latency_ms": sum(e.latency_ms for e in self.entries) / len(self.entries),
            "success_rate": sum(1 for e in self.entries if e.success) / len(self.entries),
            "by_agent": {
                agent_id: len([e for e in self.entries if e.agent_id == agent_id])
                for agent_id in set(e.agent_id for e in self.entries)
            }
        }


class SimulatedSafetyGuardrails:
    """Simulated safety guardrails."""

    def __init__(self):
        self.blocked_count = 0
        self.allowed_count = 0

    def evaluate(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action safety."""
        # Simple pattern matching for demo
        dangerous_patterns = ["rm -rf", "DROP TABLE", "system("]
        risk_level = "LOW"

        for pattern in dangerous_patterns:
            if pattern in str(context):
                risk_level = "CRITICAL"
                self.blocked_count += 1
                return {
                    "allowed": False,
                    "risk_level": risk_level,
                    "reason": f"Blocked: Detected dangerous pattern '{pattern}'"
                }

        self.allowed_count += 1
        return {
            "allowed": True,
            "risk_level": risk_level,
            "reason": "Action permitted"
        }


class AgentHypervisor:
    """
    The Agent Hypervisor - "Kubernetes for AI Agents"

    Orchestrates multi-agent, multi-model workflows with:
    - Budget enforcement
    - Audit trails
    - Safety guardrails
    - Thompson Sampling learning
    """

    def __init__(self):
        # Infrastructure
        self.audit_trail = SimulatedAuditTrail()
        self.guardrails = SimulatedSafetyGuardrails()

        # Agents (3 specialized agents)
        self.agents: Dict[str, SimulatedAgent] = {}

        # Global metrics
        self.total_tokens = 0
        self.total_requests = 0
        self.start_time = time.time()

    def register_agent(self, agent: SimulatedAgent):
        """Register an agent with the hypervisor."""
        self.agents[agent.agent_id] = agent
        print_step(f"Registered agent: {agent.name} ({agent.role})")

    async def execute_workflow(
        self,
        workflow_name: str,
        query: str,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a multi-agent workflow."""
        print_section(f"Executing Workflow: {workflow_name}", "")

        results = []
        context = {"query": query, "previous_results": []}

        for i, step in enumerate(steps, 1):
            agent_id = step["agent_id"]
            task = step["task"]

            agent = self.agents.get(agent_id)
            if not agent:
                print_step(f"Step {i}: Agent not found: {agent_id}", "ERROR")
                continue

            # Safety check
            safety = self.guardrails.evaluate(task, context)
            if not safety["allowed"]:
                print_step(f"Step {i}: BLOCKED by guardrails - {safety['reason']}", "WARN")
                continue

            # Budget check
            if agent.budget.budget_exceeded:
                print_step(f"Step {i}: Agent {agent.name} budget exceeded", "WARN")
                continue

            # Execute
            print(f"\n  {Colors.CYAN}Step {i}/{len(steps)}: {agent.name}{Colors.ENDC}")
            print(f"  {Colors.DIM}Task: {task[:60]}...{Colors.ENDC}")
            print(f"  {Colors.DIM}Model: {agent.preferred_model.value}{Colors.ENDC}")

            result = await agent.execute(task, context)
            results.append(result)

            # Log to audit trail
            self.audit_trail.log(AuditEntry(
                timestamp=time.time(),
                agent_id=agent_id,
                action=task,
                input_summary=query[:100],
                output_summary=result["response"][:100],
                model_used=result["model_used"],
                tokens_used=result["tokens_used"],
                latency_ms=result["latency_ms"],
                risk_level=safety["risk_level"],
                success=result["success"]
            ))

            # Update context
            context["previous_results"].append(result)

            # Update global metrics
            self.total_tokens += result["tokens_used"]
            self.total_requests += 1

            # Print result
            status = "OK" if result["success"] else "WARN"
            print_step(
                f"Completed in {result['latency_ms']:.0f}ms, "
                f"{result['tokens_used']} tokens, "
                f"confidence: {result['confidence']:.2f}",
                status
            )

        return {
            "workflow": workflow_name,
            "query": query,
            "steps_completed": len(results),
            "total_tokens": sum(r["tokens_used"] for r in results),
            "total_latency_ms": sum(r["latency_ms"] for r in results),
            "avg_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0,
            "all_successful": all(r["success"] for r in results),
            "results": results
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get hypervisor metrics."""
        runtime = time.time() - self.start_time
        return {
            "runtime_seconds": runtime,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "requests_per_second": self.total_requests / runtime if runtime > 0 else 0,
            "audit_summary": self.audit_trail.get_summary(),
            "guardrails": {
                "blocked": self.guardrails.blocked_count,
                "allowed": self.guardrails.allowed_count
            },
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "tokens_used": agent.budget.tokens_used,
                    "tokens_remaining": agent.budget.tokens_remaining,
                    "thompson_expected_reward": agent.thompson.expected_reward
                }
                for agent_id, agent in self.agents.items()
            }
        }


async def main():
    """Run the Agent Hypervisor demonstration."""
    print_banner()

    # ========================================================================
    # 1. Initialize the Hypervisor
    # ========================================================================
    print_section("1. Initializing Agent Hypervisor", "")
    hypervisor = AgentHypervisor()
    print_step("Hypervisor initialized")
    print_step("Audit trail: ENABLED")
    print_step("Safety guardrails: ENABLED")
    print_step("Thompson Sampling: ENABLED")

    # ========================================================================
    # 2. Register Specialized Agents (3 agents)
    # ========================================================================
    print_section("2. Registering Specialized Agents", "")

    # Agent 1: Researcher (Claude for deep reasoning)
    researcher = SimulatedAgent(
        agent_id="agent-researcher",
        name="Researcher",
        role="Deep research and analysis",
        preferred_model=ModelProvider.CLAUDE,
        budget=AgentBudget(max_tokens=20000, max_time_seconds=120)
    )
    hypervisor.register_agent(researcher)

    # Agent 2: Verifier (Claude for accuracy)
    verifier = SimulatedAgent(
        agent_id="agent-verifier",
        name="Verifier",
        role="Fact-checking and verification",
        preferred_model=ModelProvider.CLAUDE,
        budget=AgentBudget(max_tokens=10000, max_time_seconds=60)
    )
    hypervisor.register_agent(verifier)

    # Agent 3: Synthesizer (Ollama for fast summarization)
    synthesizer = SimulatedAgent(
        agent_id="agent-synthesizer",
        name="Synthesizer",
        role="Synthesis and summarization",
        preferred_model=ModelProvider.OLLAMA,
        budget=AgentBudget(max_tokens=5000, max_time_seconds=30)
    )
    hypervisor.register_agent(synthesizer)

    print(f"\n  {Colors.GREEN}3 agents registered:{Colors.ENDC}")
    print(f"    - Researcher  (Claude)  -> Deep analysis")
    print(f"    - Verifier    (Claude)  -> Fact-checking")
    print(f"    - Synthesizer (Ollama)  -> Fast synthesis")

    # ========================================================================
    # 3. Execute Multi-Agent Workflow
    # ========================================================================
    print_section("3. Executing Research Pipeline", "")

    query = "What are the tradeoffs of Thompson Sampling vs Upper Confidence Bound (UCB) in multi-armed bandit problems?"

    print(f"\n  {Colors.BOLD}Query:{Colors.ENDC}")
    print(f"  {Colors.DIM}{query}{Colors.ENDC}")

    # Define workflow steps
    workflow_steps = [
        {
            "agent_id": "agent-researcher",
            "task": f"Research and analyze: {query}"
        },
        {
            "agent_id": "agent-verifier",
            "task": "Verify the accuracy of the research findings and check for any factual errors"
        },
        {
            "agent_id": "agent-synthesizer",
            "task": "Synthesize the verified research into a concise, actionable summary"
        }
    ]

    result = await hypervisor.execute_workflow(
        workflow_name="Research Pipeline",
        query=query,
        steps=workflow_steps
    )

    # ========================================================================
    # 4. Show Workflow Results
    # ========================================================================
    print_section("4. Workflow Results", "")

    print(f"\n  {Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"    Steps completed: {result['steps_completed']}/{len(workflow_steps)}")
    print(f"    Total tokens: {result['total_tokens']:,}")
    print(f"    Total latency: {result['total_latency_ms']:.0f}ms")
    print(f"    Avg confidence: {result['avg_confidence']:.2f}")
    print(f"    All successful: {'Yes' if result['all_successful'] else 'No'}")

    # ========================================================================
    # 5. Show Agent Metrics (Thompson Sampling Learning)
    # ========================================================================
    print_section("5. Agent Learning (Thompson Sampling)", "")

    metrics = hypervisor.get_metrics()

    print(f"\n  {Colors.BOLD}Per-Agent Statistics:{Colors.ENDC}")
    for agent_id, agent_metrics in metrics["agents"].items():
        expected_reward = agent_metrics["thompson_expected_reward"]
        bar_length = int(expected_reward * 30)
        bar = "#" * bar_length + "." * (30 - bar_length)
        print(f"    {agent_metrics['name']:12s}: {bar} {expected_reward:.3f}")
        print(f"      {Colors.DIM}Tokens: {agent_metrics['tokens_used']:,} / {agent_metrics['tokens_used'] + agent_metrics['tokens_remaining']:,}{Colors.ENDC}")

    # ========================================================================
    # 6. Show Governance Metrics (Audit Trail)
    # ========================================================================
    print_section("6. Enterprise Governance", "")

    audit_summary = metrics["audit_summary"]
    guardrails = metrics["guardrails"]

    print(f"\n  {Colors.BOLD}Audit Trail:{Colors.ENDC}")
    print(f"    Total entries: {audit_summary.get('total_entries', 0)}")
    print(f"    Total tokens logged: {audit_summary.get('total_tokens', 0):,}")
    print(f"    Avg latency: {audit_summary.get('avg_latency_ms', 0):.0f}ms")
    print(f"    Success rate: {audit_summary.get('success_rate', 0):.1%}")

    print(f"\n  {Colors.BOLD}Safety Guardrails:{Colors.ENDC}")
    print(f"    Actions allowed: {guardrails['allowed']}")
    print(f"    Actions blocked: {guardrails['blocked']}")

    # ========================================================================
    # 7. Demonstrate Safety Guardrails
    # ========================================================================
    print_section("7. Safety Guardrails Demo", "")

    print(f"\n  Testing dangerous action detection...")

    # Try a dangerous action
    dangerous_task = "Execute: rm -rf / to clean up system"
    safety_result = hypervisor.guardrails.evaluate(dangerous_task, {"code": dangerous_task})

    if not safety_result["allowed"]:
        print_step(f"BLOCKED: {safety_result['reason']}", "WARN")
    else:
        print_step(f"Allowed (unexpected)", "ERROR")

    # Try a safe action
    safe_task = "Summarize the research findings"
    safety_result = hypervisor.guardrails.evaluate(safe_task, {"task": safe_task})

    if safety_result["allowed"]:
        print_step(f"ALLOWED: '{safe_task}' - Risk: {safety_result['risk_level']}", "OK")

    # ========================================================================
    # 8. Final Summary
    # ========================================================================
    print_section("8. Hypervisor Summary", "")

    print(f"""
  {Colors.BOLD}{Colors.GREEN}Agent Hypervisor Demo Complete!{Colors.ENDC}

  {Colors.BOLD}What We Demonstrated:{Colors.ENDC}

  {Colors.CYAN}1. Multi-Agent Orchestration{Colors.ENDC}
     - 3 specialized agents (Researcher, Verifier, Synthesizer)
     - Coordinated workflow execution
     - Automatic task routing

  {Colors.CYAN}2. Multi-Model Routing{Colors.ENDC}
     - Claude for deep reasoning (Researcher, Verifier)
     - Ollama for fast tasks (Synthesizer)
     - Capability-based model selection

  {Colors.CYAN}3. Enterprise Governance{Colors.ENDC}
     - Token budget enforcement per agent
     - Complete audit trail with provenance
     - Safety guardrails with action blocking

  {Colors.CYAN}4. Verifiable Memory (Thompson Sampling){Colors.ENDC}
     - Each agent learns from outcomes
     - Expected reward updates on success/failure
     - Adaptive quality over time

  {Colors.BOLD}This is what makes HoloLoom different:{Colors.ENDC}

  {Colors.YELLOW}+--------------------------------------------------------+
  |  CrewAI, LangGraph, AutoGPT are frameworks.            |
  |  HoloLoom is infrastructure.                           |
  |                                                        |
  |  "Kubernetes for AI Agents"                            |
  +--------------------------------------------------------+{Colors.ENDC}

  Runtime: {metrics['runtime_seconds']:.1f}s
  Total tokens: {metrics['total_tokens']:,}
  Total requests: {metrics['total_requests']}
    """)


if __name__ == "__main__":
    asyncio.run(main())
