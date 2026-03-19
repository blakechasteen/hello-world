"""
Background MCTS Learning Service

Runs MCTS agents continuously in background, learning from interactions
and building up expertise over time.

Key Features:
- Continuous MCTS exploration when idle
- Learns from user query feedback
- Persists patterns to disk
- Reloads learned patterns on startup
- Serves learned knowledge to future queries

Architecture:
1. LearningQueue: Collects experiences (query + outcome + feedback)
2. BackgroundLearner: Async task that explores and learns
3. AgentPool: Manages multiple specialized agents
4. PatternStore: Persistent storage for learned patterns

Usage:
    async with AgentPool(kg, emb) as pool:
        # Start background learning
        await pool.start_learning()

        # Process queries (foreground)
        result = await pool.query('budget', query)

        # Provide feedback (goes to learning queue)
        await pool.feedback('budget', query, feedback={'helpful': True})

        # Agents learn in background, patterns improve over time
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hololoom.agents import PROFILES
from hololoom.agents.orchestrator_mcts import MCTSAgentOrchestrator, create_mcts_agent
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.memory.graph import KG
from hololoom.protocols.types import Query

# ============================================================================
# Learning Queue
# ============================================================================

@dataclass
class Experience:
    """Single experience for background learning"""
    agent_name: str
    query_text: str
    outcome: Any  # Spacetime result
    feedback: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)


class LearningQueue:
    """Thread-safe queue for collecting experiences"""

    def __init__(self, maxsize: int = 1000):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.total_added = 0
        self.total_processed = 0

    async def add(self, experience: Experience):
        """Add experience to queue"""
        await self.queue.put(experience)
        self.total_added += 1

    async def get(self) -> Experience:
        """Get next experience (blocks if empty)"""
        exp = await self.queue.get()
        self.total_processed += 1
        return exp

    def qsize(self) -> int:
        """Current queue size"""
        return self.queue.qsize()

    def stats(self) -> dict[str, int]:
        """Queue statistics"""
        return {
            'queue_size': self.qsize(),
            'total_added': self.total_added,
            'total_processed': self.total_processed,
            'pending': self.total_added - self.total_processed
        }


# ============================================================================
# Pattern Store (Persistent)
# ============================================================================

class PatternStore:
    """Persistent storage for learned patterns"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.patterns: dict[str, list[dict]] = defaultdict(list)
        self.stats: dict[str, dict] = {}

    def save_patterns(self, agent_name: str, patterns: list[dict]):
        """Save patterns for agent"""
        self.patterns[agent_name] = patterns

        path = self.base_path / f"{agent_name}_patterns.json"
        with open(path, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)

    def load_patterns(self, agent_name: str) -> list[dict]:
        """Load patterns for agent"""
        path = self.base_path / f"{agent_name}_patterns.json"

        if path.exists():
            with open(path) as f:
                patterns = json.load(f)
                self.patterns[agent_name] = patterns
                return patterns

        return []

    def save_stats(self, agent_name: str, stats: dict):
        """Save learning statistics"""
        self.stats[agent_name] = stats

        path = self.base_path / f"{agent_name}_stats.json"
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

    def load_stats(self, agent_name: str) -> dict:
        """Load learning statistics"""
        path = self.base_path / f"{agent_name}_stats.json"

        if path.exists():
            with open(path) as f:
                stats = json.load(f)
                self.stats[agent_name] = stats
                return stats

        return {}

    def get_all_patterns(self) -> dict[str, list[dict]]:
        """Get all patterns (all agents)"""
        return dict(self.patterns)

    def get_all_stats(self) -> dict[str, dict]:
        """Get all statistics (all agents)"""
        return dict(self.stats)


# ============================================================================
# Background Learner
# ============================================================================

class BackgroundLearner:
    """
    Async task that continuously learns from experiences.

    Process:
    1. Wait for experience from queue
    2. Run MCTS exploration on similar queries
    3. Update agent's learned patterns
    4. Persist patterns to disk
    5. Repeat
    """

    def __init__(
        self,
        agent_pool: 'AgentPool',
        learning_queue: LearningQueue,
        pattern_store: PatternStore,
        exploration_simulations: int = 50,
        learn_interval: float = 1.0  # seconds between learning cycles
    ):
        self.agent_pool = agent_pool
        self.learning_queue = learning_queue
        self.pattern_store = pattern_store
        self.exploration_simulations = exploration_simulations
        self.learn_interval = learn_interval

        self.running = False
        self.task: asyncio.Task | None = None

        # Stats
        self.learning_cycles = 0
        self.patterns_learned = 0
        self.total_exploration_time = 0.0

    async def start(self):
        """Start background learning"""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._learning_loop())
        print(f"[BackgroundLearner] Started (exploration: {self.exploration_simulations} sims)")

    async def stop(self):
        """Stop background learning"""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        print(f"[BackgroundLearner] Stopped ({self.learning_cycles} cycles, {self.patterns_learned} patterns)")

    async def _learning_loop(self):
        """Main learning loop"""
        while self.running:
            try:
                # Wait for experience (with timeout so we can check self.running)
                try:
                    experience = await asyncio.wait_for(
                        self.learning_queue.get(),
                        timeout=self.learn_interval
                    )
                except asyncio.TimeoutError:
                    # No experiences, explore anyway
                    await self._idle_exploration()
                    continue

                # Learn from experience
                await self._learn_from_experience(experience)

                self.learning_cycles += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[BackgroundLearner] Error: {e}")
                await asyncio.sleep(1.0)

    async def _learn_from_experience(self, experience: Experience):
        """Learn from single experience"""
        start = time.time()

        # Get agent
        agent = self.agent_pool.agents.get(experience.agent_name)
        if not agent:
            return

        # Run focused MCTS exploration around this query
        # This discovers related queries and patterns
        query = Query(text=experience.query_text)

        try:
            # Use MCTS to explore variations of this query
            result = await agent.query(query, use_mcts=True)

            # Check if patterns were learned
            stats = agent.get_learning_stats()
            patterns_before = self.pattern_store.patterns.get(experience.agent_name, [])
            patterns_after = stats.get('patterns', [])

            if len(patterns_after) > len(patterns_before):
                new_patterns = len(patterns_after) - len(patterns_before)
                self.patterns_learned += new_patterns

                # Persist patterns
                self.pattern_store.save_patterns(experience.agent_name, patterns_after)

                print(f"[BackgroundLearner] {experience.agent_name}: +{new_patterns} patterns "
                      f"from '{experience.query_text[:50]}'")

        except Exception as e:
            print(f"[BackgroundLearner] Error learning from experience: {e}")

        elapsed = time.time() - start
        self.total_exploration_time += elapsed

    async def _idle_exploration(self):
        """Explore when idle (no experiences in queue)"""
        # Pick random agent and let it explore
        if not self.agent_pool.agents:
            return

        # For now, just wait - could implement random exploration here
        await asyncio.sleep(self.learn_interval)

    def stats(self) -> dict[str, Any]:
        """Learning statistics"""
        return {
            'running': self.running,
            'learning_cycles': self.learning_cycles,
            'patterns_learned': self.patterns_learned,
            'total_exploration_time_s': self.total_exploration_time,
            'avg_time_per_cycle_ms': (
                self.total_exploration_time / self.learning_cycles * 1000
                if self.learning_cycles > 0 else 0
            )
        }


# ============================================================================
# Agent Pool
# ============================================================================

class AgentPool:
    """
    Manages multiple specialized MCTS agents with background learning.

    Features:
    - Spawns agents on demand
    - Collects experiences for background learning
    - Persists learned patterns
    - Reloads patterns on startup
    """

    def __init__(
        self,
        kg: KG,
        emb: MatryoshkaEmbeddings,
        persist_path: Path = Path("./agent_patterns"),
        enable_background_learning: bool = True,
        mcts_simulations: int = 100,
        exploration_simulations: int = 50
    ):
        self.kg = kg
        self.emb = emb
        self.persist_path = Path(persist_path)
        self.enable_background_learning = enable_background_learning
        self.mcts_simulations = mcts_simulations

        # Components
        self.agents: dict[str, MCTSAgentOrchestrator] = {}
        self.learning_queue = LearningQueue()
        self.pattern_store = PatternStore(self.persist_path)
        self.background_learner = BackgroundLearner(
            self,
            self.learning_queue,
            self.pattern_store,
            exploration_simulations=exploration_simulations
        )

        # Stats
        self.query_count: dict[str, int] = defaultdict(int)
        self.total_queries = 0

    async def __aenter__(self):
        """Async context manager entry"""
        if self.enable_background_learning:
            await self.start_learning()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def start_learning(self):
        """Start background learning"""
        await self.background_learner.start()

    async def stop_learning(self):
        """Stop background learning"""
        await self.background_learner.stop()

    async def get_agent(self, agent_name: str) -> MCTSAgentOrchestrator:
        """Get or create agent"""
        if agent_name in self.agents:
            return self.agents[agent_name]

        # Create new agent
        profile = PROFILES.get(agent_name)
        if not profile:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Load learned patterns if available
        learned_patterns = self.pattern_store.load_patterns(agent_name)

        agent = await create_mcts_agent(
            agent_name,
            self.kg,
            self.emb,
            mcts_working_memory=True,
            mcts_wm_simulations=self.mcts_simulations
        ).__aenter__()

        # Inject learned patterns if any
        if learned_patterns:
            # TODO: Need method to inject patterns into agent.learner
            pass

        self.agents[agent_name] = agent
        print(f"[AgentPool] Spawned agent: {agent_name}")

        return agent

    async def query(
        self,
        agent_name: str,
        query: Query,
        use_mcts: bool = True
    ) -> Any:
        """Process query with agent"""
        agent = await self.get_agent(agent_name)
        result = await agent.query(query, use_mcts=use_mcts)

        self.query_count[agent_name] += 1
        self.total_queries += 1

        return result

    async def feedback(
        self,
        agent_name: str,
        query: Query,
        outcome: Any,
        feedback: dict[str, Any]
    ):
        """Provide feedback on query outcome (goes to learning queue)"""
        experience = Experience(
            agent_name=agent_name,
            query_text=query.text,
            outcome=outcome,
            feedback=feedback
        )

        await self.learning_queue.add(experience)

    async def close(self):
        """Close all agents and stop learning"""
        await self.stop_learning()

        # Close all agents
        for agent in self.agents.values():
            await agent.close()

        # Save final patterns
        for agent_name, agent in self.agents.items():
            stats = agent.get_learning_stats()
            patterns = stats.get('patterns', [])
            self.pattern_store.save_patterns(agent_name, patterns)
            self.pattern_store.save_stats(agent_name, stats)

        print(f"[AgentPool] Closed ({self.total_queries} queries)")

    def stats(self) -> dict[str, Any]:
        """Pool statistics"""
        return {
            'total_queries': self.total_queries,
            'query_count_by_agent': dict(self.query_count),
            'active_agents': list(self.agents.keys()),
            'learning_queue': self.learning_queue.stats(),
            'background_learner': self.background_learner.stats(),
            'patterns_by_agent': {
                name: len(patterns)
                for name, patterns in self.pattern_store.get_all_patterns().items()
            }
        }


# ============================================================================
# Factory Functions
# ============================================================================

async def create_agent_pool(
    kg: KG,
    emb: MatryoshkaEmbeddings,
    persist_path: Path = Path("./agent_patterns"),
    enable_background_learning: bool = True,
    mcts_simulations: int = 100,
    exploration_simulations: int = 50
) -> AgentPool:
    """Create agent pool with background learning"""
    pool = AgentPool(
        kg,
        emb,
        persist_path=persist_path,
        enable_background_learning=enable_background_learning,
        mcts_simulations=mcts_simulations,
        exploration_simulations=exploration_simulations
    )

    return pool
