"""
HoloLoom Agent System

Specialized agent bots with working memory and learning:
- Domain-specific configuration (budget, architecture, code review, etc.)
- Trinity working memory (semantic + graph + computational)
- Pattern learning (learns from success)
- Persistence (survives restarts)

Quick Start:
    ```python
    from HoloLoom.agents import create_agent
    from HoloLoom.memory.graph import KG
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    # Shared knowledge
    kg = KG()
    emb = MatryoshkaEmbeddings()

    # Create budget advisor
    async with create_agent('budget', kg, emb) as agent:
        # Pin persistent context
        agent.pin("company budget policy 2024", weight=0.7)

        # Query
        result = await agent.query(Query(text="Q4 marketing budget?"))

        # View working memory
        print(agent.get_working_memory_summary())

        # View learning stats
        print(agent.get_learning_stats())
    ```

Available Profiles:
    - budget: Financial planning and cost analysis
    - architecture: Software design evaluation
    - code_review: Code quality and best practices
    - research: Exploratory analysis and synthesis
    - planning: Task decomposition and sequencing
    - general: General-purpose assistant
"""

from HoloLoom.agents.types import (
    AgentProfile,
    AgentDomain,
    AgentStats,
    WorkingMemoryState,
    WorkingMemorySnapshot,
    LearnedPattern
)

from HoloLoom.agents.profiles import (
    BUDGET_ADVISOR,
    ARCHITECTURE_REVIEWER,
    CODE_REVIEWER,
    RESEARCH_ASSISTANT,
    PLANNING_AGENT,
    GENERAL_AGENT,
    AGENT_PROFILES,
    PROFILES,
    get_profile,
    list_profiles
)

from HoloLoom.agents.working_memory import AgentWorkingMemory
from HoloLoom.agents.learner import WorkingMemoryLearner
from HoloLoom.agents.orchestrator import AgentOrchestrator, create_agent


__all__ = [
    # Types
    'AgentProfile',
    'AgentDomain',
    'AgentStats',
    'WorkingMemoryState',
    'WorkingMemorySnapshot',
    'LearnedPattern',

    # Profiles
    'BUDGET_ADVISOR',
    'ARCHITECTURE_REVIEWER',
    'CODE_REVIEWER',
    'RESEARCH_ASSISTANT',
    'PLANNING_AGENT',
    'GENERAL_AGENT',
    'AGENT_PROFILES',
    'PROFILES',
    'get_profile',
    'list_profiles',

    # Core classes
    'AgentWorkingMemory',
    'WorkingMemoryLearner',
    'AgentOrchestrator',

    # Factory
    'create_agent'
]
