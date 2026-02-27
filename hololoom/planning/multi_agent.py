"""
Multi-Agent Coordination for Layer 2 Planning

Implements collaborative and competitive multi-agent planning with:
- Negotiation protocols (Contract Net, Monotonic Concession)
- Coalition formation with stability checking
- Task allocation optimization
- Conflict resolution
- Joint plan execution

Research:
- Wooldridge (2009): "Introduction to Multiagent Systems"
- Sandholm (1999): Contract Net Protocol
- Rahwan (2009): Coalition formation
- Durfee (2001): Distributed problem solving
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable
from enum import Enum
import logging
from collections import defaultdict

# Import Layer 2 core
from HoloLoom.planning.planner import HierarchicalPlanner, Plan, Goal, Action

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Core Types
# ============================================================================

class AgentType(Enum):
    """Agent cooperation style."""
    COOPERATIVE = "cooperative"          # Maximize joint utility
    COMPETITIVE = "competitive"          # Zero-sum game
    SELF_INTERESTED = "self_interested"  # Independent goals, negotiate


class NegotiationProtocol(Enum):
    """Negotiation mechanism."""
    CONTRACT_NET = "contract_net"                # Bidding (manager-bidder)
    MONOTONIC_CONCESSION = "monotonic_concession"  # Gradual concessions
    MEDIATION = "mediation"                      # Third-party mediator
    AUCTION = "auction"                          # Auction bidding


class MessageType(Enum):
    """Agent communication messages."""
    PROPOSE = "propose"          # Propose plan/action
    ACCEPT = "accept"           # Accept proposal
    REJECT = "reject"           # Reject proposal
    COUNTER = "counter"         # Counter-proposal
    COMMIT = "commit"           # Commit to action
    INFORM = "inform"           # Share information
    REQUEST = "request"         # Request action/info


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Task:
    """Task to be allocated to agent(s)."""
    task_id: str
    goal: Goal
    required_capabilities: Set[str]
    deadline: Optional[float] = None
    priority: float = 1.0
    difficulty: float = 1.0


@dataclass
class Capability:
    """Agent capability."""
    name: str
    proficiency: float  # 0-1, how good at this
    cost_multiplier: float = 1.0  # Cost factor for this capability


@dataclass
class Proposal:
    """Plan proposal from an agent."""
    agent_id: str
    task_id: str
    plan: Plan
    cost: float
    expected_duration: float
    confidence: float  # 0-1, confidence in success
    required_resources: Dict[str, float] = field(default_factory=dict)
    utility: float = 0.0  # Expected utility for proposer

    def __repr__(self):
        return (f"Proposal(agent={self.agent_id}, task={self.task_id}, "
                f"cost={self.cost:.2f}, conf={self.confidence:.2f})")


@dataclass
class Agreement:
    """Negotiated agreement between agents."""
    agents: List[str]
    task_id: str
    joint_plan: Plan
    resource_allocation: Dict[str, Dict[str, float]]  # agent -> resources
    utility_distribution: Dict[str, float]  # agent -> utility
    commitments: Dict[str, List[Action]]  # agent -> committed actions

    def __repr__(self):
        return (f"Agreement({len(self.agents)} agents, "
                f"task={self.task_id}, utilities={self.utility_distribution})")


@dataclass
class Message:
    """Communication message between agents."""
    sender: str
    receiver: str
    msg_type: MessageType
    content: any  # Proposal, Agreement, etc.
    timestamp: float = 0.0


@dataclass
class Coalition:
    """Coalition of cooperating agents."""
    members: Set[str]
    tasks: List[Task]
    value: float  # Coalition value (Shapley value)
    is_stable: bool = False

    def __repr__(self):
        return f"Coalition({len(self.members)} agents, value={self.value:.2f})"


# ============================================================================
# Agent Class
# ============================================================================

class Agent:
    """Multi-agent planning agent with negotiation capabilities."""

    def __init__(self,
                 agent_id: str,
                 agent_type: AgentType,
                 planner: HierarchicalPlanner,
                 capabilities: List[Capability],
                 resources: Optional[Dict[str, float]] = None):
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier
            agent_type: Cooperation style
            planner: Hierarchical planner from Layer 2 core
            capabilities: What this agent can do
            resources: Resources controlled by this agent
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.planner = planner
        self.capabilities = {c.name: c for c in capabilities}
        self.resources = resources or {}

        # Communication
        self.inbox: List[Message] = []
        self.commitments: Dict[str, Agreement] = {}  # task_id -> agreement

        logger.info(f"Initialized {agent_type.value} agent {agent_id} "
                   f"with {len(capabilities)} capabilities")

    # ------------------------------------------------------------------------
    # Task Evaluation
    # ------------------------------------------------------------------------

    def can_perform(self, task: Task) -> bool:
        """Check if agent has required capabilities."""
        return task.required_capabilities.issubset(self.capabilities.keys())

    def estimate_cost(self, task: Task) -> float:
        """Estimate cost to complete task."""
        if not self.can_perform(task):
            return float('inf')

        # Base cost from task difficulty
        base_cost = task.difficulty

        # Adjust for capability proficiency
        capability_factor = 1.0
        for cap_name in task.required_capabilities:
            cap = self.capabilities[cap_name]
            # Lower proficiency → higher cost
            capability_factor *= cap.cost_multiplier / (cap.proficiency + 0.1)

        return base_cost * capability_factor

    def estimate_utility(self, task: Task) -> float:
        """Estimate utility of completing task."""
        # For now, simple utility based on task priority
        # In real system, would consider agent's goals
        return task.priority

    # ------------------------------------------------------------------------
    # Proposal Generation
    # ------------------------------------------------------------------------

    def propose_plan(self, task: Task, current_state: Dict) -> Optional[Proposal]:
        """
        Generate plan proposal for task.

        Args:
            task: Task to plan for
            current_state: Current world state

        Returns:
            Proposal or None if unable to plan
        """
        if not self.can_perform(task):
            logger.debug(f"Agent {self.agent_id} cannot perform task {task.task_id}")
            return None

        # Generate plan using Layer 2 planner
        try:
            plan = self.planner.plan(task.goal, current_state)
            if not plan:
                return None

            # Calculate metrics
            cost = self.estimate_cost(task)
            utility = self.estimate_utility(task)
            confidence = self._calculate_confidence(task)
            duration = self._estimate_duration(plan)

            proposal = Proposal(
                agent_id=self.agent_id,
                task_id=task.task_id,
                plan=plan,
                cost=cost,
                expected_duration=duration,
                confidence=confidence,
                utility=utility
            )

            logger.info(f"Agent {self.agent_id} proposed plan for {task.task_id}: "
                       f"cost={cost:.2f}, conf={confidence:.2f}")
            return proposal

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to plan for {task.task_id}: {e}")
            return None

    def _calculate_confidence(self, task: Task) -> float:
        """Calculate confidence in successfully completing task."""
        # Based on capability proficiency
        if not task.required_capabilities:
            return 0.5

        proficiencies = [
            self.capabilities[cap].proficiency
            for cap in task.required_capabilities
        ]
        return sum(proficiencies) / len(proficiencies)

    def _estimate_duration(self, plan: Plan) -> float:
        """Estimate plan execution duration."""
        # Simple: count actions
        # Real system would use action duration models
        return len(plan.actions) * 1.0

    # ------------------------------------------------------------------------
    # Negotiation
    # ------------------------------------------------------------------------

    def evaluate_proposal(self, proposal: Proposal) -> float:
        """
        Evaluate another agent's proposal.

        Returns utility score (higher = better).
        """
        if self.agent_type == AgentType.COOPERATIVE:
            # Cooperative: care about joint efficiency
            utility = proposal.confidence / (proposal.cost + 1.0)
        elif self.agent_type == AgentType.COMPETITIVE:
            # Competitive: want high cost for others
            utility = proposal.cost
        else:  # SELF_INTERESTED
            # Self-interested: care about own utility
            utility = proposal.utility

        return utility

    def send_message(self, receiver: str, msg_type: MessageType,
                     content: any) -> Message:
        """Send message to another agent."""
        msg = Message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            content=content
        )
        logger.debug(f"{self.agent_id} → {receiver}: {msg_type.value}")
        return msg

    def receive_message(self, message: Message):
        """Receive message from another agent."""
        self.inbox.append(message)
        logger.debug(f"{self.agent_id} ← {message.sender}: {message.msg_type.value}")

    def process_messages(self) -> List[Message]:
        """Process all messages in inbox. Returns responses."""
        responses = []
        for msg in self.inbox:
            response = self._handle_message(msg)
            if response:
                responses.append(response)

        self.inbox.clear()
        return responses

    def _handle_message(self, msg: Message) -> Optional[Message]:
        """Handle single message. Returns response if any."""
        if msg.msg_type == MessageType.PROPOSE:
            proposal = msg.content
            utility = self.evaluate_proposal(proposal)

            # Decide: accept, reject, or counter
            if utility > 0.7:  # Threshold
                return self.send_message(
                    msg.sender, MessageType.ACCEPT, proposal
                )
            else:
                return self.send_message(
                    msg.sender, MessageType.REJECT, proposal
                )

        elif msg.msg_type == MessageType.REQUEST:
            # Handle information requests
            # TODO: Implement information sharing
            pass

        return None

    # ------------------------------------------------------------------------
    # Commitment
    # ------------------------------------------------------------------------

    def commit_to(self, agreement: Agreement):
        """Commit to an agreement."""
        self.commitments[agreement.task_id] = agreement
        logger.info(f"Agent {self.agent_id} committed to task {agreement.task_id}")

    def is_committed(self, task_id: str) -> bool:
        """Check if already committed to task."""
        return task_id in self.commitments


# ============================================================================
# Multi-Agent Coordinator
# ============================================================================

class MultiAgentCoordinator:
    """Coordinates multiple planning agents."""

    def __init__(self,
                 agents: List[Agent],
                 protocol: NegotiationProtocol = NegotiationProtocol.CONTRACT_NET):
        """
        Initialize coordinator.

        Args:
            agents: List of agents to coordinate
            protocol: Negotiation protocol to use
        """
        self.agents = {a.agent_id: a for a in agents}
        self.protocol = protocol

        logger.info(f"Initialized MultiAgentCoordinator with {len(agents)} agents "
                   f"using {protocol.value} protocol")

    # ------------------------------------------------------------------------
    # Task Allocation
    # ------------------------------------------------------------------------

    def allocate_tasks(self, tasks: List[Task],
                      current_state: Dict) -> Dict[str, str]:
        """
        Allocate tasks to agents.

        Args:
            tasks: Tasks to allocate
            current_state: Current world state

        Returns:
            task_id -> agent_id mapping
        """
        if self.protocol == NegotiationProtocol.CONTRACT_NET:
            return self._allocate_contract_net(tasks, current_state)
        elif self.protocol == NegotiationProtocol.AUCTION:
            return self._allocate_auction(tasks, current_state)
        else:
            # Default: greedy assignment
            return self._allocate_greedy(tasks, current_state)

    def _allocate_contract_net(self, tasks: List[Task],
                               current_state: Dict) -> Dict[str, str]:
        """
        Contract Net Protocol (Smith 1980).

        Process:
        1. Manager announces task
        2. Agents submit bids (proposals)
        3. Manager selects best bid
        4. Winner commits to task
        """
        allocation = {}

        for task in tasks:
            logger.info(f"Allocating task {task.task_id} via Contract Net")

            # Step 1: Announce task (implicit - all agents know)

            # Step 2: Collect bids
            bids = []
            for agent in self.agents.values():
                if agent.is_committed(task.task_id):
                    continue  # Skip if already committed

                proposal = agent.propose_plan(task, current_state)
                if proposal:
                    bids.append(proposal)

            if not bids:
                logger.warning(f"No bids for task {task.task_id}")
                continue

            # Step 3: Select best bid
            # Lower cost, higher confidence = better
            best_bid = min(bids, key=lambda p: p.cost / (p.confidence + 0.1))

            # Step 4: Award task
            winner = self.agents[best_bid.agent_id]
            allocation[task.task_id] = winner.agent_id

            # Create agreement
            agreement = Agreement(
                agents=[winner.agent_id],
                task_id=task.task_id,
                joint_plan=best_bid.plan,
                resource_allocation={winner.agent_id: best_bid.required_resources},
                utility_distribution={winner.agent_id: best_bid.utility},
                commitments={winner.agent_id: best_bid.plan.actions}
            )
            winner.commit_to(agreement)

            logger.info(f"Task {task.task_id} → Agent {winner.agent_id} "
                       f"(cost={best_bid.cost:.2f}, conf={best_bid.confidence:.2f})")

        return allocation

    def _allocate_greedy(self, tasks: List[Task],
                        current_state: Dict) -> Dict[str, str]:
        """Greedy task allocation: best agent for each task."""
        allocation = {}

        for task in tasks:
            best_agent = None
            best_cost = float('inf')

            for agent in self.agents.values():
                if not agent.can_perform(task):
                    continue
                cost = agent.estimate_cost(task)
                if cost < best_cost:
                    best_cost = cost
                    best_agent = agent

            if best_agent:
                allocation[task.task_id] = best_agent.agent_id
                logger.info(f"Task {task.task_id} → Agent {best_agent.agent_id}")

        return allocation

    def _allocate_auction(self, tasks: List[Task],
                         current_state: Dict) -> Dict[str, str]:
        """Auction-based allocation (similar to Contract Net but with bidding rounds)."""
        # For now, same as Contract Net
        # TODO: Implement multi-round bidding
        return self._allocate_contract_net(tasks, current_state)

    # ------------------------------------------------------------------------
    # Coalition Formation
    # ------------------------------------------------------------------------

    def form_coalition(self, task: Task,
                      current_state: Dict) -> Optional[Coalition]:
        """
        Form coalition of agents to achieve task.

        Uses coalition formation algorithm:
        1. Identify capable agents
        2. Try coalitions of increasing size
        3. Calculate coalition value
        4. Check stability

        Returns:
            Coalition if formed, None otherwise
        """
        # Step 1: Find capable agents
        capable_agents = [
            agent for agent in self.agents.values()
            if agent.can_perform(task)
        ]

        if not capable_agents:
            return None

        # Step 2: Try different coalition sizes
        for size in range(1, len(capable_agents) + 1):
            from itertools import combinations

            for members in combinations(capable_agents, size):
                member_ids = {agent.agent_id for agent in members}

                # Calculate coalition value
                value = self._calculate_coalition_value(
                    member_ids, task, current_state
                )

                # Check stability (no agent wants to leave)
                stable = self._check_coalition_stability(
                    member_ids, task, value, current_state
                )

                if stable and value > 0:
                    coalition = Coalition(
                        members=member_ids,
                        tasks=[task],
                        value=value,
                        is_stable=stable
                    )
                    logger.info(f"Formed stable coalition: {coalition}")
                    return coalition

        return None

    def _calculate_coalition_value(self, members: Set[str],
                                   task: Task,
                                   current_state: Dict) -> float:
        """
        Calculate coalition value (Shapley value).

        For simplicity, using average utility across members.
        Real implementation would use proper Shapley value calculation.
        """
        total_utility = 0.0
        total_cost = 0.0

        for agent_id in members:
            agent = self.agents[agent_id]
            utility = agent.estimate_utility(task)
            cost = agent.estimate_cost(task)
            total_utility += utility
            total_cost += cost

        # Coalition value = total utility - avg cost
        avg_cost = total_cost / len(members) if members else 0
        return total_utility - avg_cost

    def _check_coalition_stability(self, members: Set[str],
                                   task: Task, value: float,
                                   current_state: Dict) -> bool:
        """
        Check if coalition is stable (core stability).

        Stable if: no subset can do better alone.
        """
        # For simplicity, check if each member benefits
        for agent_id in members:
            agent = self.agents[agent_id]
            individual_utility = agent.estimate_utility(task)
            individual_cost = agent.estimate_cost(task)
            individual_value = individual_utility - individual_cost

            # Share of coalition value (equal split)
            coalition_share = value / len(members)

            # Unstable if agent better off alone
            if individual_value > coalition_share:
                return False

        return True

    # ------------------------------------------------------------------------
    # Conflict Resolution
    # ------------------------------------------------------------------------

    def resolve_conflicts(self, plans: List[Tuple[str, Plan]]) -> List[Tuple[str, Plan]]:
        """
        Resolve conflicts between agent plans.

        Conflict types:
        - Resource conflicts: Two agents need same resource
        - Timing conflicts: Actions interfere temporally
        - Goal conflicts: Conflicting objectives

        Args:
            plans: List of (agent_id, plan) tuples

        Returns:
            Conflict-free plans
        """
        # Detect conflicts
        conflicts = self._detect_conflicts(plans)

        if not conflicts:
            return plans

        logger.info(f"Detected {len(conflicts)} conflicts, resolving...")

        # Resolve conflicts
        resolved_plans = []
        for agent_id, plan in plans:
            # For each conflict involving this agent, modify plan
            modified_plan = plan
            for conflict in conflicts:
                if agent_id in conflict['agents']:
                    modified_plan = self._resolve_conflict(
                        conflict, agent_id, modified_plan
                    )
            resolved_plans.append((agent_id, modified_plan))

        logger.info("Conflicts resolved")
        return resolved_plans

    def _detect_conflicts(self, plans: List[Tuple[str, Plan]]) -> List[Dict]:
        """Detect conflicts between plans."""
        conflicts = []

        # Check all pairs of plans
        for i, (agent1, plan1) in enumerate(plans):
            for j, (agent2, plan2) in enumerate(plans[i+1:], start=i+1):
                # Check resource conflicts
                # TODO: Implement proper resource conflict detection
                # For now, placeholder
                pass

        return conflicts

    def _resolve_conflict(self, conflict: Dict, agent_id: str,
                         plan: Plan) -> Plan:
        """Resolve a specific conflict for an agent's plan."""
        # TODO: Implement conflict resolution strategies
        # - Priority-based: Higher priority agent gets resource
        # - Negotiation: Agents negotiate resolution
        # - Temporal: Shift actions in time

        # For now, return original plan
        return plan

    # ------------------------------------------------------------------------
    # Joint Execution
    # ------------------------------------------------------------------------

    def execute_joint_plan(self, agreement: Agreement,
                          executor: Callable) -> Dict:
        """
        Execute coordinated plan.

        Args:
            agreement: Agreed-upon plan
            executor: Function to execute actions

        Returns:
            Execution results
        """
        logger.info(f"Executing joint plan for task {agreement.task_id}")

        results = {
            'task_id': agreement.task_id,
            'agents': agreement.agents,
            'actions_executed': 0,
            'success': False
        }

        # Execute actions in plan order
        for action in agreement.joint_plan.actions:
            # Find responsible agent
            responsible_agent = None
            for agent_id, actions in agreement.commitments.items():
                if action in actions:
                    responsible_agent = agent_id
                    break

            if not responsible_agent:
                logger.error(f"No agent responsible for action: {action}")
                continue

            # Execute action
            try:
                executor(action, responsible_agent)
                results['actions_executed'] += 1
            except Exception as e:
                logger.error(f"Action failed: {e}")
                results['success'] = False
                return results

        results['success'] = True
        logger.info(f"Joint plan completed successfully")
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def create_agent(agent_id: str,
                agent_type: AgentType,
                dag: 'CausalDAG',  # From Layer 1
                capabilities: List[str],
                proficiency: float = 0.8) -> Agent:
    """
    Convenient agent factory.

    Args:
        agent_id: Unique ID
        agent_type: Cooperation style
        dag: Causal DAG for planning
        capabilities: List of capability names
        proficiency: Default proficiency level

    Returns:
        Configured Agent
    """
    from HoloLoom.planning.planner import HierarchicalPlanner

    planner = HierarchicalPlanner(dag)
    caps = [
        Capability(name=cap, proficiency=proficiency)
        for cap in capabilities
    ]

    return Agent(
        agent_id=agent_id,
        agent_type=agent_type,
        planner=planner,
        capabilities=caps
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'Agent',
    'AgentType',
    'MultiAgentCoordinator',
    'NegotiationProtocol',
    'Task',
    'Capability',
    'Proposal',
    'Agreement',
    'Coalition',
    'Message',
    'MessageType',
    'create_agent',
]
