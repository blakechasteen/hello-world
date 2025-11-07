"""
Repository Context Manager Demo - For mythRL Agents
====================================================
Shows how to use repository context management with mythRL's agentic system,
not just Claude Code agents.

Use case: Deploy mythRL agents that need controlled access to different
codebases for specialized tasks.

Example Agents:
1. Code Review Agent - Needs access to all code
2. Documentation Agent - Needs access to docs only
3. Security Audit Agent - Needs access to security-critical code
4. Business Intelligence Agent - Needs access to business data (cos/)
5. ML Research Agent - Needs access to ML code (HoloLoom/policy, HoloLoom/embedding)

Author: HoloLoom Team
Date: 2025-11-04
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.memory.repository_context import (
    create_repo_manager,
    AccessLevel,
    RepositoryContextManager,
    AgentQueryContext
)
from HoloLoom.agentic import AgenticOrchestrator, ReasoningMode
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query


# ============================================================================
# mythRL Agent Wrapper with Repository Context
# ============================================================================

class RepositoryAwareAgent:
    """
    mythRL agent with repository context awareness.

    Wraps AgenticOrchestrator with repository access control.
    """

    def __init__(
        self,
        agent_id: str,
        query_context: AgentQueryContext,
        reasoning_mode: ReasoningMode = ReasoningMode.DIRECT
    ):
        """
        Initialize repository-aware agent.

        Args:
            agent_id: Agent identifier
            query_context: Repository query context
            reasoning_mode: mythRL reasoning mode
        """
        self.agent_id = agent_id
        self.query_context = query_context
        self.reasoning_mode = reasoning_mode

        # Will be initialized on demand
        self._orchestrator = None

    async def initialize(self, config: Config):
        """Initialize mythRL orchestrator."""
        # In production, this would create AgenticOrchestrator
        # For demo, we'll simulate it
        self._orchestrator = None  # Placeholder
        print(f"✓ {self.agent_id} initialized with {self.reasoning_mode.value} mode")

    async def query_with_context(self, query: str, limit: int = 5) -> dict:
        """
        Query using repository context + mythRL reasoning.

        Pipeline:
        1. Retrieve relevant code from accessible repositories
        2. Use mythRL reasoning to analyze/synthesize
        3. Return results with citations

        Args:
            query: User query
            limit: Max results

        Returns:
            Dict with results, reasoning steps, citations
        """
        # Step 1: Retrieve code context
        print(f"\n🔍 [{self.agent_id}] Retrieving context for: '{query}'")

        code_results = await self.query_context.query(query, limit=limit)

        if not code_results:
            print(f"⚠️  [{self.agent_id}] No accessible code found")
            return {
                'query': query,
                'accessible_repos': [r.name for r in self.query_context.list_repositories()],
                'results': [],
                'reasoning': 'No code context available'
            }

        print(f"✓ [{self.agent_id}] Found {len(code_results)} relevant code snippets")

        # Step 2: Format code context
        context_snippets = []
        for i, res in enumerate(code_results, 1):
            snippet = {
                'source': res['context']['file_path'],
                'repository': res['context']['repository'],
                'score': res['score'],
                'text': res['text'][:500],  # Truncate for demo
                'entity': res['context'].get('entity_name', 'N/A')
            }
            context_snippets.append(snippet)

            print(f"  {i}. [{res['score']:.3f}] {res['context']['file_path']}")
            print(f"     Entity: {res['context'].get('entity_name', 'N/A')}")

        # Step 3: mythRL reasoning (simulated for demo)
        # In production, this would be:
        # result = await self._orchestrator.reason(
        #     query=query,
        #     mode=self.reasoning_mode,
        #     context=context_snippets
        # )

        print(f"🤔 [{self.agent_id}] Analyzing with {self.reasoning_mode.value} reasoning...")

        # Simulated reasoning result
        reasoning_result = {
            'query': query,
            'agent_id': self.agent_id,
            'reasoning_mode': self.reasoning_mode.value,
            'accessible_repos': [r.name for r in self.query_context.list_repositories()],
            'code_context': context_snippets,
            'reasoning_steps': [
                'Retrieved code context from accessible repositories',
                f'Analyzed {len(context_snippets)} relevant code snippets',
                'Synthesized answer based on code understanding'
            ],
            'answer': self._simulate_answer(query, context_snippets)
        }

        return reasoning_result

    def _simulate_answer(self, query: str, snippets: list) -> str:
        """Simulate mythRL answer generation."""
        # In production, this would be actual mythRL reasoning
        repos = {s['repository'] for s in snippets}
        files = {s['source'] for s in snippets}

        return (
            f"Based on analysis of {len(snippets)} code snippets from "
            f"{len(repos)} repositories ({', '.join(repos)}), "
            f"I found relevant information in {len(files)} files. "
            f"[Simulated reasoning result - in production, this would be "
            f"actual mythRL {self.reasoning_mode.value} mode output]"
        )


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_scenario_1_code_review_agent():
    """
    Scenario 1: Code Review Agent
    Needs access to all code (HoloLoom + squad) but NOT business docs.
    """
    print("\n" + "="*60)
    print("SCENARIO 1: Code Review Agent")
    print("="*60)

    # Create repository manager
    repo_mgr = await create_repo_manager()

    # Register repositories
    await repo_mgr.add_repository(
        name="HoloLoom",
        path=str(Path(__file__).parent.parent / "HoloLoom"),
        tags={"python", "ml", "core"},
        access_level=AccessLevel.PUBLIC,
        description="Core HoloLoom system",
        auto_index=False  # Skip indexing for demo speed
    )

    await repo_mgr.add_repository(
        name="squad",
        path=str(Path(__file__).parent.parent / "squad"),
        tags={"typescript", "vscode", "frontend"},
        access_level=AccessLevel.PUBLIC,
        description="VS Code extension",
        auto_index=False
    )

    await repo_mgr.add_repository(
        name="cos",
        path=str(Path(__file__).parent.parent / "cos"),
        tags={"business", "private"},
        access_level=AccessLevel.PRIVATE,
        description="Business planning (confidential)",
        auto_index=False
    )

    # Create Code Review Agent
    code_review_context = repo_mgr.create_agent_context(
        agent_id="code_reviewer",
        allowed_repos={"HoloLoom", "squad"},  # Code only
        blocked_repos={"cos"},  # No business docs
        allowed_tags={"python", "typescript"}
    )

    agent = RepositoryAwareAgent(
        agent_id="code_reviewer",
        query_context=code_review_context,
        reasoning_mode=ReasoningMode.VERIFY  # Verify code patterns
    )

    await agent.initialize(Config.fused())

    # Test queries
    print("\n📝 Test Query 1: Cross-repository question")
    result = await agent.query_with_context(
        "How do the frontend and backend communicate?",
        limit=5
    )

    print(f"\n✓ Answer: {result['answer'][:200]}...")

    print("\n📝 Test Query 2: Blocked repository (should fail)")
    result = await agent.query_with_context(
        "What are the business revenue projections?",  # From cos/
        limit=5
    )

    print(f"\n⚠️  Result: {len(result['code_context'])} snippets (should be 0 - blocked)")


async def demo_scenario_2_security_audit_agent():
    """
    Scenario 2: Security Audit Agent
    Needs access to security-critical code only.
    """
    print("\n" + "="*60)
    print("SCENARIO 2: Security Audit Agent")
    print("="*60)

    repo_mgr = await create_repo_manager()

    # Register with security tags
    await repo_mgr.add_repository(
        name="HoloLoom",
        path=str(Path(__file__).parent.parent / "HoloLoom"),
        tags={"python", "ml", "security"},
        access_level=AccessLevel.INTERNAL,
        auto_index=False
    )

    # Create Security Audit Agent
    security_context = repo_mgr.create_agent_context(
        agent_id="security_auditor",
        allowed_repos={"HoloLoom"},
        allowed_tags={"security", "alignment", "guardrails"},
        access_level=AccessLevel.INTERNAL
    )

    agent = RepositoryAwareAgent(
        agent_id="security_auditor",
        query_context=security_context,
        reasoning_mode=ReasoningMode.RESEARCH  # Deep analysis
    )

    await agent.initialize(Config.fused())

    # Test query
    print("\n🔒 Test Query: Security-focused question")
    result = await agent.query_with_context(
        "How does the alignment framework prevent deception?",
        limit=5
    )

    print(f"\n✓ Found security-related code in: {result['accessible_repos']}")


async def demo_scenario_3_ml_research_agent():
    """
    Scenario 3: ML Research Agent
    Needs access to ML-related code only (embeddings, policy, etc.).
    """
    print("\n" + "="*60)
    print("SCENARIO 3: ML Research Agent")
    print("="*60)

    repo_mgr = await create_repo_manager()

    await repo_mgr.add_repository(
        name="HoloLoom",
        path=str(Path(__file__).parent.parent / "HoloLoom"),
        tags={"python", "ml", "embeddings", "policy", "bandits"},
        access_level=AccessLevel.PUBLIC,
        auto_index=False
    )

    # Create ML Research Agent - only ML tags
    ml_context = repo_mgr.create_agent_context(
        agent_id="ml_researcher",
        allowed_repos={"HoloLoom"},
        allowed_tags={"ml", "embeddings", "policy", "bandits", "thompson_sampling"},
        blocked_tags={"business", "deployment", "infrastructure"}
    )

    agent = RepositoryAwareAgent(
        agent_id="ml_researcher",
        query_context=ml_context,
        reasoning_mode=ReasoningMode.RESEARCH
    )

    await agent.initialize(Config.fused())

    # Test query
    print("\n🧠 Test Query: ML research question")
    result = await agent.query_with_context(
        "How does Thompson Sampling balance exploration and exploitation?",
        limit=5
    )

    print(f"\n✓ Analyzed {len(result['code_context'])} ML code snippets")


async def demo_scenario_4_business_intelligence_agent():
    """
    Scenario 4: Business Intelligence Agent
    Needs access to business data (cos/) but NOT code.
    """
    print("\n" + "="*60)
    print("SCENARIO 4: Business Intelligence Agent")
    print("="*60)

    repo_mgr = await create_repo_manager()

    # Register business repo
    await repo_mgr.add_repository(
        name="cos",
        path=str(Path(__file__).parent.parent / "cos"),
        tags={"business", "finance", "planning"},
        access_level=AccessLevel.PRIVATE,
        auto_index=False
    )

    # Register code repo (blocked for this agent)
    await repo_mgr.add_repository(
        name="HoloLoom",
        path=str(Path(__file__).parent.parent / "HoloLoom"),
        tags={"python", "ml"},
        access_level=AccessLevel.PUBLIC,
        auto_index=False
    )

    # Create Business Intelligence Agent
    bi_context = repo_mgr.create_agent_context(
        agent_id="business_analyst",
        allowed_repos={"cos"},  # Only business data
        blocked_repos={"HoloLoom", "squad"},  # No code access
        allowed_tags={"business", "finance"},
        access_level=AccessLevel.PRIVATE  # Can access private repos
    )

    agent = RepositoryAwareAgent(
        agent_id="business_analyst",
        query_context=bi_context,
        reasoning_mode=ReasoningMode.ANALYTICAL
    )

    await agent.initialize(Config.fused())

    # Test queries
    print("\n💼 Test Query 1: Business question (should work)")
    result = await agent.query_with_context(
        "What are the revenue projections for Q1?",
        limit=5
    )

    print(f"\n✓ Accessible repos: {result['accessible_repos']}")

    print("\n💼 Test Query 2: Code question (should fail)")
    result = await agent.query_with_context(
        "How does the Thompson Sampling algorithm work?",  # From HoloLoom
        limit=5
    )

    print(f"\n⚠️  Result: {len(result['code_context'])} snippets (should be 0 - blocked)")


async def demo_scenario_5_parallel_agent_swarm():
    """
    Scenario 5: Parallel Agent Swarm
    Deploy multiple agents simultaneously with different access.
    """
    print("\n" + "="*60)
    print("SCENARIO 5: Parallel Agent Swarm (5 Agents)")
    print("="*60)

    repo_mgr = await create_repo_manager()

    # Register all repos
    await repo_mgr.add_repository(
        name="HoloLoom",
        path=str(Path(__file__).parent.parent / "HoloLoom"),
        tags={"python", "ml"},
        access_level=AccessLevel.PUBLIC,
        auto_index=False
    )

    await repo_mgr.add_repository(
        name="squad",
        path=str(Path(__file__).parent.parent / "squad"),
        tags={"typescript", "vscode"},
        access_level=AccessLevel.PUBLIC,
        auto_index=False
    )

    await repo_mgr.add_repository(
        name="cos",
        path=str(Path(__file__).parent.parent / "cos"),
        tags={"business"},
        access_level=AccessLevel.PRIVATE,
        auto_index=False
    )

    # Create 5 agents with different access
    agents = {
        "frontend": RepositoryAwareAgent(
            agent_id="frontend_dev",
            query_context=repo_mgr.create_agent_context(
                agent_id="frontend_dev",
                allowed_repos={"squad"},
                allowed_tags={"typescript"}
            ),
            reasoning_mode=ReasoningMode.DIRECT
        ),

        "backend": RepositoryAwareAgent(
            agent_id="backend_dev",
            query_context=repo_mgr.create_agent_context(
                agent_id="backend_dev",
                allowed_repos={"HoloLoom"},
                allowed_tags={"python"}
            ),
            reasoning_mode=ReasoningMode.DIRECT
        ),

        "security": RepositoryAwareAgent(
            agent_id="security_auditor",
            query_context=repo_mgr.create_agent_context(
                agent_id="security_auditor",
                allowed_repos={"HoloLoom", "squad"},
                blocked_repos={"cos"},
                access_level=AccessLevel.INTERNAL
            ),
            reasoning_mode=ReasoningMode.VERIFY
        ),

        "ml_research": RepositoryAwareAgent(
            agent_id="ml_researcher",
            query_context=repo_mgr.create_agent_context(
                agent_id="ml_researcher",
                allowed_repos={"HoloLoom"},
                allowed_tags={"ml", "embeddings", "policy"}
            ),
            reasoning_mode=ReasoningMode.RESEARCH
        ),

        "business": RepositoryAwareAgent(
            agent_id="business_analyst",
            query_context=repo_mgr.create_agent_context(
                agent_id="business_analyst",
                allowed_repos={"cos"},
                blocked_repos={"HoloLoom", "squad"},
                access_level=AccessLevel.PRIVATE
            ),
            reasoning_mode=ReasoningMode.ANALYTICAL
        )
    }

    # Initialize all agents
    config = Config.fused()
    for agent in agents.values():
        await agent.initialize(config)

    # Deploy in parallel (simulated)
    print("\n🚀 Deploying 5 agents in parallel...")

    queries = {
        "frontend": "How does the VS Code extension connect to HoloLoom?",
        "backend": "How does the RAG system work?",
        "security": "How does the alignment framework work?",
        "ml_research": "How does Thompson Sampling work?",
        "business": "What are the revenue projections?"
    }

    # Run queries in parallel
    tasks = [
        agents[name].query_with_context(query, limit=3)
        for name, query in queries.items()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Display results
    print("\n📊 Results Summary:")
    for i, (name, result) in enumerate(zip(queries.keys(), results)):
        if isinstance(result, Exception):
            print(f"\n❌ {name}: Error - {result}")
        else:
            print(f"\n✅ {name}:")
            print(f"   Query: {queries[name][:60]}...")
            print(f"   Repos: {', '.join(result['accessible_repos'])}")
            print(f"   Snippets: {len(result['code_context'])}")
            print(f"   Mode: {result['reasoning_mode']}")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all demo scenarios."""
    print("\n" + "="*70)
    print("Repository Context Manager - mythRL Agent Integration Demo")
    print("="*70)

    print("\nThis demo shows how mythRL agents can use repository context")
    print("management for controlled code access across multiple repositories.")

    try:
        # Run all scenarios
        await demo_scenario_1_code_review_agent()
        await demo_scenario_2_security_audit_agent()
        await demo_scenario_3_ml_research_agent()
        await demo_scenario_4_business_intelligence_agent()
        await demo_scenario_5_parallel_agent_swarm()

        print("\n" + "="*70)
        print("✅ All demos completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
