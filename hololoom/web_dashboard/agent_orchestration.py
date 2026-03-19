"""
Agent-Centric Orchestration System

"Agents All The Way" - Agents are first-class persistent entities that can:
- Participate in conversations (reactive)
- Work in background (proactive)
- Share breakthroughs with each other
- Learn from each other's experiences
- Have persistent identity and memory
- Execute prioritized task queues

Key shift: Conversations don't own agents - agents participate in conversations.

Architecture:
    Agent Pool (persistent)
        ├─ Budget Agent
        │   ├─ Active in 3 conversations
        │   ├─ Running 2 background tasks
        │   └─ Has 15 learned patterns
        ├─ Architecture Agent
        │   ├─ Active in 1 conversation
        │   └─ Running 1 background task
        └─ Research Agent
            ├─ Active in 5 conversations
            ├─ Running 3 background tasks
            └─ Has 42 learned patterns

    Task Queue (smart prioritization)
        ├─ CRITICAL: Error recovery
        ├─ HIGH: User queries
        ├─ NORMAL: Background learning
        └─ LOW: Cleanup, maintenance
"""

import asyncio
import heapq
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hololoom.agents.background_learner import LearningQueue
from hololoom.agents.mcts_breakthrough import BreakthroughDetector, FeedForwardBroadcaster
from hololoom.agents.orchestrator_mcts import create_mcts_agent
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.memory.graph import KG

# ============================================================================
# Task Types and Priorities
# ============================================================================

class TaskPriority(Enum):
    """Explicit task priorities"""
    CRITICAL = 1  # System errors, user-blocking issues
    HIGH = 2      # User queries, real-time requests
    NORMAL = 3    # Background learning, pattern updates
    LOW = 4       # Maintenance, cleanup, logging


class TaskType(Enum):
    """Types of agent tasks"""
    USER_QUERY = "user_query"                # Respond to user query
    BACKGROUND_LEARNING = "background_learning"  # Learn from experience
    BREAKTHROUGH_EXPLORATION = "breakthrough_exploration"  # Explore breakthrough
    PATTERN_VALIDATION = "pattern_validation"  # Validate learned patterns
    CROSS_AGENT_SYNC = "cross_agent_sync"    # Sync knowledge between agents
    MAINTENANCE = "maintenance"              # Cleanup, optimize
    MONITORING = "monitoring"                # Health checks


@dataclass
class AgentTask:
    """Task for an agent to execute"""
    task_id: str
    priority: TaskPriority
    task_type: TaskType

    # Target agent
    agent_name: str

    # Task execution
    task_fn: Callable  # Async function to execute
    context: dict[str, Any] = field(default_factory=dict)

    # Scheduling
    created_at: float = field(default_factory=time.time)
    scheduled_for: float | None = None  # Delay execution
    timeout_seconds: float = 60.0

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Task IDs

    # Results
    status: str = "pending"  # pending, running, completed, failed
    result: Any | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None

    def __lt__(self, other):
        """Priority queue comparison"""
        # Lower priority number = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value

        # Tie-break with creation time (FIFO within priority)
        return self.created_at < other.created_at

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if task dependencies are satisfied"""
        return all(dep_id in completed_tasks for dep_id in self.depends_on)


# ============================================================================
# Persistent Agent
# ============================================================================

@dataclass
class PersistentAgent:
    """
    First-class persistent agent.

    Lives independently of conversations - can participate in many activities.
    """
    agent_id: str
    agent_name: str  # budget, architecture, research, etc.
    agent_instance: Any  # MCTSAgentOrchestrator

    # Activity tracking
    active_conversations: set[str] = field(default_factory=set)
    active_background_tasks: set[str] = field(default_factory=set)
    total_queries_processed: int = 0
    total_breakthroughs: int = 0

    # Learning
    patterns_learned: int = 0
    experiences_collected: int = 0

    # Health
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    error_count: int = 0
    success_rate: float = 1.0

    def update_activity(self):
        """Update activity timestamp"""
        self.last_active = time.time()

    def record_success(self):
        """Record successful task execution"""
        self.total_queries_processed += 1
        self.success_rate = (self.success_rate * 0.95 + 0.05)  # Moving average

    def record_error(self):
        """Record error"""
        self.error_count += 1
        self.success_rate = (self.success_rate * 0.95)  # Penalty

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'active_conversations': len(self.active_conversations),
            'active_background_tasks': len(self.active_background_tasks),
            'total_queries': self.total_queries_processed,
            'total_breakthroughs': self.total_breakthroughs,
            'patterns_learned': self.patterns_learned,
            'experiences_collected': self.experiences_collected,
            'uptime_seconds': time.time() - self.created_at,
            'idle_seconds': time.time() - self.last_active,
            'error_count': self.error_count,
            'success_rate': self.success_rate
        }


# ============================================================================
# Agent Orchestration System
# ============================================================================

class AgentOrchestrationSystem:
    """
    Manages persistent agents and their task queues.

    Agents All The Way:
    - Agents are long-lived, not per-conversation
    - Agents have prioritized task queues
    - Agents share breakthroughs
    - Agents learn from each other
    """

    def __init__(
        self,
        kg: KG,
        emb: MatryoshkaEmbeddings,
        enable_breakthrough_sharing: bool = True
    ):
        self.kg = kg
        self.emb = emb
        self.enable_breakthrough_sharing = enable_breakthrough_sharing

        # Agent pool (persistent)
        self.agents: dict[str, PersistentAgent] = {}

        # Task queue (priority queue)
        self.task_queue: list[AgentTask] = []
        self.completed_tasks: set[str] = set()
        self.running_tasks: dict[str, AgentTask] = {}

        # Breakthrough system (shared)
        self.breakthrough_detector = BreakthroughDetector(
            reward_improvement_threshold=1.8,
            confidence_jump_threshold=0.18,
            max_breakthrough_memory=1000  # Large for multi-agent
        )

        self.breakthrough_broadcaster = FeedForwardBroadcaster()

        # Learning queue (shared)
        self.learning_queue = LearningQueue(maxsize=5000)

        # Task processor
        self.processor_task: asyncio.Task | None = None
        self.running = False

        # Statistics
        self.total_tasks_queued = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0

    async def start(self):
        """Start orchestration system"""
        if self.running:
            return

        self.running = True
        self.processor_task = asyncio.create_task(self._process_tasks())

    async def stop(self):
        """Stop orchestration system"""
        self.running = False

        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        # Cleanup agents
        for agent in self.agents.values():
            if hasattr(agent.agent_instance, '__aexit__'):
                await agent.agent_instance.__aexit__(None, None, None)

    async def get_or_create_agent(self, agent_name: str) -> PersistentAgent:
        """
        Get existing agent or create new one.

        Agents are persistent - reused across conversations.
        """
        if agent_name in self.agents:
            return self.agents[agent_name]

        # Create new agent
        agent_instance = await create_mcts_agent(
            agent_name,
            self.kg,
            self.emb,
            mcts_working_memory=True,
            mcts_wm_simulations=50,
            breakthrough_detector=self.breakthrough_detector
        ).__aenter__()

        # Register with broadcaster
        if self.enable_breakthrough_sharing and hasattr(agent_instance, 'working_memory'):
            if hasattr(agent_instance.working_memory, 'mcts_engine'):
                self.breakthrough_broadcaster.register_listener(
                    agent_instance.working_memory.mcts_engine
                )

        # Create persistent agent
        persistent_agent = PersistentAgent(
            agent_id=str(uuid.uuid4()),
            agent_name=agent_name,
            agent_instance=agent_instance
        )

        self.agents[agent_name] = persistent_agent

        return persistent_agent

    async def queue_task(
        self,
        agent_name: str,
        task_fn: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_type: TaskType = TaskType.USER_QUERY,
        context: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        scheduled_for: float | None = None
    ) -> str:
        """
        Queue task for agent execution.

        Returns:
            task_id
        """
        task_id = f"task_{self.total_tasks_queued}"

        task = AgentTask(
            task_id=task_id,
            priority=priority,
            task_type=task_type,
            agent_name=agent_name,
            task_fn=task_fn,
            context=context or {},
            depends_on=depends_on or [],
            scheduled_for=scheduled_for
        )

        heapq.heappush(self.task_queue, task)
        self.total_tasks_queued += 1

        return task_id

    async def _process_tasks(self):
        """Background task processor"""
        while self.running:
            try:
                await self._process_queue()
                await asyncio.sleep(0.1)  # Check every 100ms
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[AgentOrchestration] Error in task processor: {e}")
                await asyncio.sleep(1.0)

    async def _process_queue(self):
        """Process tasks from queue"""
        if not self.task_queue:
            return

        # Check next task
        next_task = self.task_queue[0]

        # Check scheduling
        if next_task.scheduled_for:
            if time.time() < next_task.scheduled_for:
                return  # Not ready yet

        # Check dependencies
        if not next_task.is_ready(self.completed_tasks):
            return  # Dependencies not satisfied

        # Pop task
        task = heapq.heappop(self.task_queue)

        # Execute task
        await self._execute_task(task)

    async def _execute_task(self, task: AgentTask):
        """Execute single task"""
        task.status = "running"
        task.started_at = time.time()
        self.running_tasks[task.task_id] = task

        # Get agent
        agent = await self.get_or_create_agent(task.agent_name)
        agent.update_activity()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                task.task_fn(agent, task.context),
                timeout=task.timeout_seconds
            )

            # Success
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()

            self.completed_tasks.add(task.task_id)
            self.total_tasks_completed += 1

            agent.record_success()

        except asyncio.TimeoutError:
            # Timeout
            task.status = "failed"
            task.error = f"Timeout after {task.timeout_seconds}s"
            task.completed_at = time.time()

            self.total_tasks_failed += 1
            agent.record_error()

        except Exception as e:
            # Error
            task.status = "failed"
            task.error = str(e)
            task.completed_at = time.time()

            self.total_tasks_failed += 1
            agent.record_error()

            print(f"[AgentOrchestration] Task {task.task_id} failed: {e}")

        finally:
            # Remove from running
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

    def get_agent_stats(self, agent_name: str) -> dict[str, Any] | None:
        """Get statistics for specific agent"""
        if agent_name not in self.agents:
            return None

        return self.agents[agent_name].get_stats()

    def get_global_stats(self) -> dict[str, Any]:
        """Get global orchestration statistics"""
        return {
            'active_agents': len(self.agents),
            'queue_size': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'total_tasks_queued': self.total_tasks_queued,
            'total_tasks_completed': self.total_tasks_completed,
            'total_tasks_failed': self.total_tasks_failed,
            'success_rate': (
                self.total_tasks_completed /
                (self.total_tasks_completed + self.total_tasks_failed)
                if (self.total_tasks_completed + self.total_tasks_failed) > 0
                else 1.0
            ),
            'agents_by_name': {
                name: agent.get_stats()
                for name, agent in self.agents.items()
            },
            'breakthrough_detector': self.breakthrough_detector.get_stats(),
            'broadcaster': self.breakthrough_broadcaster.get_stats(),
            'learning_queue': self.learning_queue.stats()
        }


# ============================================================================
# Helper Functions
# ============================================================================

async def create_agent_orchestration_system(
    kg: KG,
    emb: MatryoshkaEmbeddings,
    enable_breakthrough_sharing: bool = True
) -> AgentOrchestrationSystem:
    """Create and start agent orchestration system"""
    system = AgentOrchestrationSystem(
        kg=kg,
        emb=emb,
        enable_breakthrough_sharing=enable_breakthrough_sharing
    )

    await system.start()
    return system
