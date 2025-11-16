"""
Adversarial Agent Orchestration - Integration

Extends AgentOrchestrationSystem with adversarial negotiation capabilities.

Architecture:
    AgentOrchestrationSystem (base)
        ↓
    AdversarialOrchestrationSystem (extended)
        ├─ Creative Agent Proposals
        ├─ QC Agent Reviews
        ├─ Negotiated Strategy Execution
        └─ Learning from Outcomes

Key Innovation: Tasks can optionally go through adversarial negotiation
before execution, creating productive tension between exploration and safety.

Usage:
    system = await create_adversarial_orchestration_system(kg, emb)

    # Normal task (no negotiation)
    await system.queue_task(agent_name='budget', task_fn=..., priority=HIGH)

    # Adversarial task (negotiated)
    await system.queue_negotiated_task(
        agent_name='budget',
        task_fn=...,
        priority=HIGH,
        enable_negotiation=True  # Creative vs QC negotiation
    )
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from HoloLoom.web_dashboard.agent_orchestration import (
    AgentOrchestrationSystem,
    PersistentAgent,
    TaskPriority,
    TaskType,
    AgentTask
)
from HoloLoom.agents.adversarial_agents import (
    CreativeAgent,
    QualityControlAgent,
    AdversarialNegotiationSystem
)
from HoloLoom.memory.graph import KG
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings


# ============================================================================
# Extended Task with Negotiation
# ============================================================================

@dataclass
class NegotiatedTask(AgentTask):
    """
    Task that goes through adversarial negotiation before execution.

    Extends AgentTask with negotiation metadata.
    """
    # Negotiation settings
    enable_negotiation: bool = False
    creativity_level: float = 0.8  # 0-1
    strictness_level: float = 0.8  # 0-1

    # Negotiation results
    negotiation_outcome: Optional[str] = None  # 'creative_win', 'qc_win', 'compromise'
    final_strategy: Optional[Dict[str, Any]] = None
    negotiation_rounds: int = 0


# ============================================================================
# Adversarial Orchestration System
# ============================================================================

class AdversarialOrchestrationSystem(AgentOrchestrationSystem):
    """
    Extended orchestration system with adversarial negotiation.

    Allows tasks to optionally go through Creative vs QC negotiation
    before execution, creating optimal balance between exploration and safety.
    """

    def __init__(
        self,
        kg: KG,
        emb: MatryoshkaEmbeddings,
        enable_breakthrough_sharing: bool = True,
        default_creativity: float = 0.8,
        default_strictness: float = 0.8
    ):
        super().__init__(kg, emb, enable_breakthrough_sharing)

        self.default_creativity = default_creativity
        self.default_strictness = default_strictness

        # Adversarial agents (one set per agent type)
        self.creative_agents: Dict[str, CreativeAgent] = {}
        self.qc_agents: Dict[str, QualityControlAgent] = {}
        self.negotiation_systems: Dict[str, AdversarialNegotiationSystem] = {}

        # Statistics
        self.total_negotiated_tasks = 0
        self.creative_wins = 0
        self.qc_wins = 0
        self.compromises = 0

    def _get_or_create_negotiation_system(
        self,
        agent_name: str,
        creativity_level: float,
        strictness_level: float
    ) -> AdversarialNegotiationSystem:
        """Get or create adversarial negotiation system for agent"""
        key = f"{agent_name}_{creativity_level}_{strictness_level}"

        if key not in self.negotiation_systems:
            # Create creative agent
            creative = CreativeAgent(
                creativity_level=creativity_level,
                novelty_threshold=0.6
            )

            # Create QC agent
            qc = QualityControlAgent(
                strictness_level=strictness_level,
                min_confidence=0.7,
                max_risk="MEDIUM"
            )

            # Create negotiation system
            negotiation = AdversarialNegotiationSystem(
                creative_agent=creative,
                qc_agent=qc,
                negotiation_rounds=3
            )

            self.negotiation_systems[key] = negotiation
            self.creative_agents[key] = creative
            self.qc_agents[key] = qc

        return self.negotiation_systems[key]

    async def queue_negotiated_task(
        self,
        agent_name: str,
        task_fn: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_type: TaskType = TaskType.USER_QUERY,
        context: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        scheduled_for: Optional[float] = None,
        enable_negotiation: bool = True,
        creativity_level: Optional[float] = None,
        strictness_level: Optional[float] = None
    ) -> str:
        """
        Queue task with optional adversarial negotiation.

        If enable_negotiation=True, task strategy will be negotiated between
        Creative Agent (exploration) and QC Agent (safety) before execution.

        Args:
            agent_name: Target agent
            task_fn: Async function to execute
            priority: Task priority (CRITICAL/HIGH/NORMAL/LOW)
            task_type: Type of task
            context: Task context
            depends_on: Task dependencies
            scheduled_for: Delayed execution time
            enable_negotiation: Whether to negotiate strategy
            creativity_level: How creative (0-1, default 0.8)
            strictness_level: How strict QC (0-1, default 0.8)

        Returns:
            task_id
        """
        # Use defaults if not specified
        creativity = creativity_level if creativity_level is not None else self.default_creativity
        strictness = strictness_level if strictness_level is not None else self.default_strictness

        # Create negotiated task
        task_id = f"negotiated_task_{self.total_tasks_queued}"

        task = NegotiatedTask(
            task_id=task_id,
            priority=priority,
            task_type=task_type,
            agent_name=agent_name,
            task_fn=task_fn,
            context=context or {},
            depends_on=depends_on or [],
            scheduled_for=scheduled_for,
            enable_negotiation=enable_negotiation,
            creativity_level=creativity,
            strictness_level=strictness
        )

        # If negotiation enabled, negotiate strategy before queueing
        if enable_negotiation:
            negotiation_system = self._get_or_create_negotiation_system(
                agent_name,
                creativity,
                strictness
            )

            # Negotiate strategy
            negotiation_result = negotiation_system.negotiate_strategy(context or {})

            # Store negotiation results
            task.negotiation_outcome = negotiation_result['outcome']
            task.final_strategy = negotiation_result['final_strategy']
            task.negotiation_rounds = negotiation_result['negotiation_rounds']

            # Update task context with negotiated strategy
            task.context['negotiated_strategy'] = negotiation_result['final_strategy']

            # Track statistics
            self.total_negotiated_tasks += 1
            if negotiation_result['outcome'] == 'creative_win':
                self.creative_wins += 1
            elif negotiation_result['outcome'] == 'qc_win':
                self.qc_wins += 1
            else:
                self.compromises += 1

        # Queue task
        import heapq
        heapq.heappush(self.task_queue, task)
        self.total_tasks_queued += 1

        return task_id

    async def _execute_task(self, task: AgentTask):
        """
        Execute task (override to handle negotiated tasks).

        If task is NegotiatedTask with negotiation results, those results
        can influence execution.
        """
        # Check if negotiated task
        is_negotiated = isinstance(task, NegotiatedTask) and task.enable_negotiation

        if is_negotiated:
            print(f"[AdversarialOrchestration] Executing negotiated task: {task.task_id}")
            print(f"  Outcome: {task.negotiation_outcome}")
            print(f"  Strategy: {task.final_strategy['strategy']}")
            print(f"  Rounds: {task.negotiation_rounds}")

        # Execute task normally (base class handles it)
        await super()._execute_task(task)

        # Record outcome to adversarial system
        if is_negotiated and task.status == "completed":
            # Task succeeded - record outcome
            negotiation_system = self._get_or_create_negotiation_system(
                task.agent_name,
                task.creativity_level,
                task.strictness_level
            )

            # Check if breakthrough occurred
            breakthrough = task.result and hasattr(task.result, 'confidence') and task.result.confidence > 0.9
            quality_maintained = task.result and not hasattr(task.result, 'error')

            negotiation_system.record_outcome(
                outcome_type=task.negotiation_outcome,
                breakthrough=breakthrough,
                quality_maintained=quality_maintained
            )

    def get_negotiation_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get adversarial negotiation statistics"""
        if agent_name:
            # Stats for specific agent
            agent_systems = {
                key: system
                for key, system in self.negotiation_systems.items()
                if key.startswith(f"{agent_name}_")
            }

            if not agent_systems:
                return {}

            # Aggregate stats
            total_negotiations = sum(s.total_negotiations for s in agent_systems.values())
            creative_wins = sum(s.creative_wins for s in agent_systems.values())
            qc_wins = sum(s.qc_wins for s in agent_systems.values())
            compromises = sum(s.true_compromises for s in agent_systems.values())

            return {
                'agent_name': agent_name,
                'total_negotiations': total_negotiations,
                'creative_wins': creative_wins,
                'qc_wins': qc_wins,
                'compromises': compromises,
                'creative_win_rate': creative_wins / total_negotiations if total_negotiations > 0 else 0,
                'qc_win_rate': qc_wins / total_negotiations if total_negotiations > 0 else 0,
                'compromise_rate': compromises / total_negotiations if total_negotiations > 0 else 0
            }
        else:
            # Global stats
            return {
                'total_negotiated_tasks': self.total_negotiated_tasks,
                'creative_wins': self.creative_wins,
                'qc_wins': self.qc_wins,
                'compromises': self.compromises,
                'creative_win_rate': self.creative_wins / self.total_negotiated_tasks if self.total_negotiated_tasks > 0 else 0,
                'qc_win_rate': self.qc_wins / self.total_negotiated_tasks if self.total_negotiated_tasks > 0 else 0,
                'compromise_rate': self.compromises / self.total_negotiated_tasks if self.total_negotiated_tasks > 0 else 0,
                'negotiation_systems': len(self.negotiation_systems),
                'agents_by_name': {
                    name: self.get_negotiation_stats(name)
                    for name in set(
                        key.split('_')[0]
                        for key in self.negotiation_systems.keys()
                    )
                }
            }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics (override to include negotiation stats)"""
        base_stats = super().get_global_stats()

        # Add negotiation stats
        base_stats['adversarial_negotiation'] = self.get_negotiation_stats()

        return base_stats


# ============================================================================
# Helper Functions
# ============================================================================

async def create_adversarial_orchestration_system(
    kg: KG,
    emb: MatryoshkaEmbeddings,
    enable_breakthrough_sharing: bool = True,
    default_creativity: float = 0.8,
    default_strictness: float = 0.8
) -> AdversarialOrchestrationSystem:
    """
    Create and start adversarial orchestration system.

    Args:
        kg: Knowledge graph
        emb: Embeddings
        enable_breakthrough_sharing: Enable breakthrough feed-forward
        default_creativity: Default creativity level (0-1)
        default_strictness: Default QC strictness (0-1)

    Returns:
        Running AdversarialOrchestrationSystem
    """
    system = AdversarialOrchestrationSystem(
        kg=kg,
        emb=emb,
        enable_breakthrough_sharing=enable_breakthrough_sharing,
        default_creativity=default_creativity,
        default_strictness=default_strictness
    )

    await system.start()
    return system


# ============================================================================
# Usage Example
# ============================================================================

async def example_adversarial_orchestration():
    """
    Example: Queue tasks with adversarial negotiation

    Demonstrates:
    1. Normal task (no negotiation)
    2. Negotiated task (creative vs QC)
    3. High creativity task
    4. High strictness task
    """
    from HoloLoom.memory.graph import KG
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    from HoloLoom.documentation.types import Query

    # Create system
    kg = KG()
    emb = MatryoshkaEmbeddings(model_name='all-MiniLM-L6-v2', scales=[96, 192, 384])

    system = await create_adversarial_orchestration_system(
        kg,
        emb,
        default_creativity=0.8,
        default_strictness=0.8
    )

    # Example 1: Normal task (no negotiation)
    async def simple_query(agent: PersistentAgent, context: Dict):
        query = Query(text=context['query_text'])
        result = await agent.agent_instance.query(query, use_mcts=True)
        return result

    task_id_1 = await system.queue_task(
        agent_name='budget',
        task_fn=simple_query,
        priority=TaskPriority.HIGH,
        task_type=TaskType.USER_QUERY,
        context={'query_text': 'What is Q4 revenue?'}
    )

    print(f"Queued normal task: {task_id_1}")

    # Example 2: Negotiated task (balanced)
    task_id_2 = await system.queue_negotiated_task(
        agent_name='budget',
        task_fn=simple_query,
        priority=TaskPriority.HIGH,
        task_type=TaskType.USER_QUERY,
        context={'query_text': 'Explore new revenue patterns'},
        enable_negotiation=True,
        creativity_level=0.8,  # Balanced
        strictness_level=0.8   # Balanced
    )

    print(f"Queued negotiated task (balanced): {task_id_2}")

    # Example 3: High creativity task
    task_id_3 = await system.queue_negotiated_task(
        agent_name='research',
        task_fn=simple_query,
        priority=TaskPriority.NORMAL,
        task_type=TaskType.BREAKTHROUGH_EXPLORATION,
        context={'query_text': 'Find breakthrough patterns'},
        enable_negotiation=True,
        creativity_level=0.95,  # Very creative
        strictness_level=0.6    # Relaxed QC
    )

    print(f"Queued high-creativity task: {task_id_3}")

    # Example 4: High strictness task
    task_id_4 = await system.queue_negotiated_task(
        agent_name='architecture',
        task_fn=simple_query,
        priority=TaskPriority.HIGH,
        task_type=TaskType.USER_QUERY,
        context={'query_text': 'System architecture validation'},
        enable_negotiation=True,
        creativity_level=0.6,   # Conservative
        strictness_level=0.95   # Very strict QC
    )

    print(f"Queued high-strictness task: {task_id_4}")

    # Wait for tasks to complete
    await asyncio.sleep(5.0)

    # Get statistics
    stats = system.get_global_stats()

    print("\n=== Global Statistics ===")
    print(f"Total tasks: {stats['total_tasks_queued']}")
    print(f"Completed: {stats['total_tasks_completed']}")
    print(f"Failed: {stats['total_tasks_failed']}")

    print("\n=== Adversarial Negotiation Statistics ===")
    neg_stats = stats['adversarial_negotiation']
    print(f"Total negotiated: {neg_stats['total_negotiated_tasks']}")
    print(f"Creative wins: {neg_stats['creative_wins']} ({neg_stats['creative_win_rate']:.1%})")
    print(f"QC wins: {neg_stats['qc_wins']} ({neg_stats['qc_win_rate']:.1%})")
    print(f"Compromises: {neg_stats['compromises']} ({neg_stats['compromise_rate']:.1%})")

    # Cleanup
    await system.stop()


if __name__ == "__main__":
    asyncio.run(example_adversarial_orchestration())
