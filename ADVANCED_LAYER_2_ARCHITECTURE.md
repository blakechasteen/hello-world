# Advanced Layer 2: Production-Grade Planning Architecture

**Status:** 🚀 MOONSHOT IN PROGRESS
**Scope:** Advanced planning capabilities for real-world deployment
**Foundation:** Built on Layer 2 core (HTN + Causal Reasoning)

---

## Overview

Advanced Layer 2 extends the core hierarchical planner with production-critical features:

1. **Multi-Agent Coordination** - Multiple planners collaborating/competing
2. **Resource Constraints** - Finite resources, allocation optimization
3. **Continuous Replanning** - Execution monitoring and dynamic adaptation
4. **POMDP Planning** - Partial observability, belief states, information gathering

These features transform Layer 2 from a single-agent, fully-observable, resource-unlimited toy into a **production-ready planning system**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     ADVANCED LAYER 2                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Multi-Agent Coordinator                                │     │
│  │  - Negotiation protocols                                │     │
│  │  - Coalition formation                                  │     │
│  │  - Conflict resolution                                  │     │
│  │  - Joint plan execution                                 │     │
│  └────────────────────────────────────────────────────────┘     │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Resource-Aware Planner                                 │     │
│  │  - Resource tracking (time, money, materials)          │     │
│  │  - Allocation optimization                              │     │
│  │  - Budget constraints                                   │     │
│  │  - Feasibility checking                                 │     │
│  └────────────────────────────────────────────────────────┘     │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Replanning Engine                                      │     │
│  │  - Execution monitoring                                 │     │
│  │  - Failure detection                                    │     │
│  │  - Dynamic replanning                                   │     │
│  │  - Plan repair vs full replan                          │     │
│  └────────────────────────────────────────────────────────┘     │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  POMDP Planner                                          │     │
│  │  - Belief state tracking                                │     │
│  │  - Information-gathering actions                        │     │
│  │  - Contingent planning                                  │     │
│  │  - Value of information                                 │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                            ▲
                            │ uses
┌──────────────────────────────────────────────────────────────────┐
│                      LAYER 2 CORE                                 │
│  - HTN Planner                                                    │
│  - Causal Chain Finder                                            │
│  - Goal Decomposition                                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Multi-Agent Coordination

### Motivation

Real-world scenarios often involve multiple agents:
- **Collaborative:** Multiple robots assembling a product
- **Competitive:** Auction bidding, resource competition
- **Mixed:** Supply chain with cooperating firms and competing vendors

### Core Concepts

**1. Agent Types:**
- **Cooperative:** Share goals, maximize joint utility
- **Competitive:** Conflicting goals, zero-sum games
- **Self-Interested:** Independent goals, negotiate for mutual benefit

**2. Coordination Mechanisms:**
- **Negotiation:** Agents propose, counter-propose, agree on plans
- **Coalition Formation:** Agents group to achieve shared goals
- **Task Allocation:** Distribute tasks based on capabilities
- **Conflict Resolution:** Handle plan conflicts (resource, timing)

**3. Communication:**
- **Message Passing:** Agents exchange proposals, commitments
- **Shared State:** Common knowledge about world state
- **Commitments:** Agents commit to actions, trust mechanisms

### Implementation Design

```python
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum

class AgentType(Enum):
    COOPERATIVE = "cooperative"      # Maximize joint utility
    COMPETITIVE = "competitive"      # Zero-sum game
    SELF_INTERESTED = "self_interested"  # Independent goals

class NegotiationProtocol(Enum):
    CONTRACT_NET = "contract_net"    # Bidding protocol
    MONOTONIC_CONCESSION = "monotonic_concession"  # Gradual concessions
    MEDIATION = "mediation"          # Third-party mediator

@dataclass
class Agent:
    """Multi-agent planning agent."""
    agent_id: str
    agent_type: AgentType
    planner: HierarchicalPlanner  # From Layer 2 core
    goals: List[Goal]
    capabilities: Set[str]  # What actions this agent can perform
    resources: Dict[str, float]  # Resources this agent controls

    def propose_plan(self, task: Task) -> Optional[Proposal]:
        """Propose a plan for a task."""
        pass

    def evaluate_proposal(self, proposal: Proposal) -> float:
        """Evaluate another agent's proposal (utility)."""
        pass

    def negotiate(self, other_agent: 'Agent', task: Task) -> Optional[Agreement]:
        """Negotiate with another agent."""
        pass

@dataclass
class Proposal:
    """Plan proposal from an agent."""
    agent_id: str
    task: Task
    plan: Plan
    cost: float
    utility: float  # Expected utility for proposer
    required_resources: Dict[str, float]

@dataclass
class Agreement:
    """Negotiated agreement between agents."""
    agents: List[str]
    task: Task
    joint_plan: Plan
    resource_allocation: Dict[str, Dict[str, float]]  # agent_id -> resources
    utility_distribution: Dict[str, float]  # agent_id -> utility

class MultiAgentCoordinator:
    """Coordinates multiple planning agents."""

    def __init__(self, agents: List[Agent], protocol: NegotiationProtocol):
        self.agents = {a.agent_id: a for a in agents}
        self.protocol = protocol

    def allocate_tasks(self, tasks: List[Task]) -> Dict[str, Task]:
        """
        Allocate tasks to agents.

        Strategies:
        - CONTRACT_NET: Agents bid on tasks
        - OPTIMAL_ASSIGNMENT: Minimize total cost (Hungarian algorithm)
        - CAPABILITY_MATCH: Assign based on capabilities
        """
        pass

    def form_coalition(self, task: Task) -> Optional[List[Agent]]:
        """
        Form coalition of agents to achieve task.

        Uses coalition formation algorithm:
        1. Identify agents who can contribute
        2. Calculate coalition values (Shapley value)
        3. Form stable coalition (core stability)
        """
        pass

    def resolve_conflicts(self, plans: List[Plan]) -> List[Plan]:
        """
        Resolve conflicts between agent plans.

        Conflict types:
        - Resource conflicts: Two agents need same resource
        - Timing conflicts: Actions interfere temporally
        - Goal conflicts: Conflicting objectives
        """
        pass

    def execute_joint_plan(self, agreement: Agreement) -> ExecutionResult:
        """Execute coordinated plan with monitoring."""
        pass
```

### Research Alignment

- **Wooldridge (2009):** "Introduction to Multiagent Systems"
- **Sandholm (1999):** "Contract net protocol"
- **Rahwan (2009):** "Coalition formation with spatial/temporal constraints"
- **Durfee (2001):** "Distributed problem solving and planning"

---

## Feature 2: Resource Constraints

### Motivation

Real-world planning must respect finite resources:
- **Time:** Deadlines, task durations
- **Money:** Budgets, costs
- **Materials:** Consumable resources
- **Energy:** Battery life, fuel
- **Personnel:** Staff availability

### Core Concepts

**1. Resource Types:**
- **Consumable:** Used up by actions (fuel, materials)
- **Reusable:** Temporarily held (tools, rooms)
- **Producible:** Generated by actions (products, data)

**2. Constraints:**
- **Budget:** Total cost ≤ budget
- **Deadline:** Total time ≤ deadline
- **Capacity:** Resource usage ≤ capacity at all times
- **Precedence:** Some actions must precede others

**3. Optimization:**
- **Minimize Cost:** Cheapest plan
- **Minimize Time:** Fastest plan
- **Multi-Objective:** Pareto-optimal tradeoffs

### Implementation Design

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum

class ResourceType(Enum):
    CONSUMABLE = "consumable"    # Used up (fuel, materials)
    REUSABLE = "reusable"        # Held temporarily (tools, rooms)
    PRODUCIBLE = "producible"    # Generated (products)

@dataclass
class Resource:
    """Resource definition."""
    name: str
    resource_type: ResourceType
    initial_amount: float
    capacity: Optional[float] = None  # Max at any time
    cost_per_unit: float = 0.0

@dataclass
class ResourceRequirement:
    """Resources required by an action."""
    resource: str
    amount: float
    when: str = "start"  # "start", "end", "duration"

@dataclass
class ResourceState:
    """State of resources at a point in time."""
    time: float
    available: Dict[str, float]  # resource_name -> amount
    allocated: Dict[str, float]  # resource_name -> amount held

class ResourceTracker:
    """Tracks resource usage over time."""

    def __init__(self, resources: List[Resource]):
        self.resources = {r.name: r for r in resources}
        self.timeline: List[ResourceState] = []

    def check_feasibility(self, plan: Plan) -> bool:
        """Check if plan is feasible given resources."""
        pass

    def compute_resource_timeline(self, plan: Plan) -> List[ResourceState]:
        """Compute resource usage over time."""
        pass

    def find_violations(self, plan: Plan) -> List[ResourceViolation]:
        """Find resource constraint violations."""
        pass

class ResourceAwarePlanner:
    """Planner with resource constraints."""

    def __init__(self, base_planner: HierarchicalPlanner,
                 resources: List[Resource],
                 constraints: Dict[str, float]):
        self.base_planner = base_planner
        self.tracker = ResourceTracker(resources)
        self.constraints = constraints  # e.g., {"budget": 1000, "deadline": 100}

    def plan(self, goal: Goal, current_state: Dict) -> Optional[Plan]:
        """
        Generate resource-feasible plan.

        Algorithm:
        1. Generate candidate plan (base planner)
        2. Check resource feasibility
        3. If infeasible, repair or backtrack
        4. Optimize for resource usage
        """
        pass

    def optimize_resources(self, plan: Plan,
                          objective: str = "minimize_cost") -> Plan:
        """
        Optimize plan for resource usage.

        Objectives:
        - minimize_cost: Cheapest plan
        - minimize_time: Fastest plan
        - balance: Pareto frontier
        """
        pass

    def repair_plan(self, plan: Plan, violations: List[ResourceViolation]) -> Optional[Plan]:
        """Repair plan to fix resource violations."""
        pass
```

### Research Alignment

- **Ghallab et al. (2004):** "Automated Planning" (resource constraints chapter)
- **Coles et al. (2009):** "COLIN: Planning with continuous linear numeric change"
- **Fox & Long (2003):** "PDDL2.1: Numeric planning"

---

## Feature 3: Continuous Replanning

### Motivation

Plans fail in the real world:
- **Unexpected events:** Robot obstacle, equipment failure
- **Inaccurate models:** Estimated costs wrong
- **Dynamic environments:** Goals change, new information
- **Execution errors:** Actions don't achieve expected effects

**Solution:** Monitor execution and dynamically replan.

### Core Concepts

**1. Execution Monitoring:**
- **State Tracking:** Monitor actual vs expected state
- **Failure Detection:** Detect when plan fails
- **Progress Tracking:** Measure plan progress

**2. Replanning Strategies:**
- **Full Replan:** Generate entirely new plan
- **Plan Repair:** Fix broken plan minimally
- **Plan Continuation:** Continue from current state
- **Opportunistic:** Exploit unexpected opportunities

**3. When to Replan:**
- **Failure-Driven:** Replan when action fails
- **Proactive:** Replan when better options emerge
- **Periodic:** Replan every N steps
- **Threshold-Based:** Replan when divergence > threshold

### Implementation Design

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum

class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"      # Partial success
    DELAYED = "delayed"      # Taking longer than expected

class ReplanTrigger(Enum):
    FAILURE = "failure"               # Action failed
    OPPORTUNITY = "opportunity"       # Better option available
    DIVERGENCE = "divergence"         # State diverged too much
    TIMEOUT = "timeout"               # Deadline approaching
    NEW_GOAL = "new_goal"            # Goal changed

@dataclass
class ExecutionResult:
    """Result of executing an action."""
    action: Action
    status: ExecutionStatus
    actual_state: Dict[str, Any]
    expected_state: Dict[str, Any]
    cost: float
    duration: float

@dataclass
class ExecutionTrace:
    """Trace of plan execution."""
    plan: Plan
    results: List[ExecutionResult]
    current_step: int
    current_state: Dict[str, Any]
    total_cost: float
    total_time: float

class ExecutionMonitor:
    """Monitors plan execution and detects failures."""

    def __init__(self, plan: Plan, executor: Callable):
        self.plan = plan
        self.executor = executor  # Function that executes actions
        self.trace = ExecutionTrace(plan, [], 0, {}, 0.0, 0.0)

    def execute_step(self, action: Action) -> ExecutionResult:
        """Execute single action and monitor result."""
        pass

    def check_divergence(self) -> float:
        """Measure state divergence from expected."""
        pass

    def detect_failure(self, result: ExecutionResult) -> bool:
        """Detect if action failed."""
        pass

    def should_replan(self) -> Optional[ReplanTrigger]:
        """Decide if replanning is needed."""
        pass

class ReplanningEngine:
    """Continuous replanning engine."""

    def __init__(self, planner: HierarchicalPlanner,
                 monitor: ExecutionMonitor,
                 replan_strategy: str = "repair"):
        self.planner = planner
        self.monitor = monitor
        self.replan_strategy = replan_strategy

    def execute_with_monitoring(self, plan: Plan) -> ExecutionTrace:
        """
        Execute plan with continuous monitoring and replanning.

        Algorithm:
        1. Execute action
        2. Monitor result
        3. Check if replanning needed
        4. If needed, replan (repair or full)
        5. Continue with new plan
        6. Repeat until goal achieved or failure
        """
        pass

    def replan(self, trigger: ReplanTrigger,
               current_state: Dict, goal: Goal) -> Plan:
        """
        Generate new plan based on trigger.

        Strategies:
        - REPAIR: Minimal changes to current plan
        - FULL: Generate entirely new plan
        - CONTINUATION: Plan from current state
        - OPPORTUNISTIC: Exploit new opportunities
        """
        pass

    def repair_plan(self, plan: Plan, failure_step: int,
                    current_state: Dict) -> Optional[Plan]:
        """
        Repair broken plan.

        Algorithm:
        1. Identify failed action
        2. Find alternative actions
        3. Splice into plan
        4. Check feasibility
        """
        pass

class AdaptivePlanner:
    """Combines planning and replanning."""

    def __init__(self, base_planner: HierarchicalPlanner):
        self.base_planner = base_planner
        self.execution_history: List[ExecutionTrace] = []

    def plan_and_execute(self, goal: Goal, current_state: Dict,
                        max_replans: int = 10) -> ExecutionTrace:
        """
        Plan, execute, monitor, and replan as needed.

        Returns complete execution trace.
        """
        pass

    def learn_from_failures(self):
        """Learn from execution failures to improve future planning."""
        pass
```

### Research Alignment

- **Ghallab et al. (2016):** "Acting and Planning" (execution monitoring)
- **van der Krogt & de Weerdt (2005):** "Plan repair in temporal planning"
- **Fox et al. (2006):** "EUROPA: Planning and scheduling for space missions"

---

## Feature 4: POMDP Planning

### Motivation

Real world is **partially observable**:
- **Sensor Limitations:** Can't see everything
- **Hidden State:** Internal state of other agents
- **Uncertainty:** Noisy observations
- **Incomplete Info:** Missing data

**Solution:** Plan under uncertainty using belief states.

### Core Concepts

**1. Belief State:**
- **Definition:** Probability distribution over possible states
- **Update:** Bayesian update after observations
- **Maintenance:** Track belief over time

**2. Information-Gathering Actions:**
- **Observe:** Gather information (reduce uncertainty)
- **Test:** Active sensing (e.g., "check if door locked")
- **Ask:** Query other agents

**3. Value of Information:**
- **Trade-off:** Cost of gathering info vs benefit
- **Optimal Stopping:** When to stop gathering and act
- **Contingent Plans:** Plan branches based on observations

### Implementation Design

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Callable
import numpy as np

@dataclass
class BeliefState:
    """Probability distribution over states."""
    states: List[Dict[str, Any]]  # Possible states
    probabilities: np.ndarray     # P(state)

    def update(self, observation: Dict, observation_model: Callable) -> 'BeliefState':
        """Bayesian update after observation."""
        pass

    def entropy(self) -> float:
        """Measure uncertainty (Shannon entropy)."""
        pass

    def most_likely_state(self) -> Dict[str, Any]:
        """Return most probable state."""
        pass

@dataclass
class ObservationAction:
    """Action that gathers information."""
    variable: str           # What to observe
    cost: float            # Cost of observation
    accuracy: float        # Observation accuracy (0-1)

@dataclass
class ContingentPlan:
    """Plan with branches based on observations."""
    root_action: Action
    branches: Dict[str, 'ContingentPlan']  # observation -> subplan
    depth: int
    expected_cost: float

class ObservationModel:
    """Models P(observation | state)."""

    def __init__(self):
        self.accuracy: Dict[str, float] = {}  # variable -> accuracy

    def sample_observation(self, true_state: Dict, variable: str) -> Any:
        """Sample observation given true state."""
        pass

    def likelihood(self, observation: Any, variable: str, state: Dict) -> float:
        """P(observation | state, variable)."""
        pass

class POMDPPlanner:
    """Planner for partially observable domains."""

    def __init__(self, base_planner: HierarchicalPlanner,
                 observation_model: ObservationModel):
        self.base_planner = base_planner
        self.observation_model = observation_model

    def plan(self, goal: Goal, belief: BeliefState,
             max_depth: int = 5) -> ContingentPlan:
        """
        Generate contingent plan under partial observability.

        Algorithm:
        1. Check if information-gathering worthwhile
        2. If yes, observe and branch
        3. Otherwise, act on most likely state
        4. Recursively plan for each branch

        Returns tree of conditional plans.
        """
        pass

    def value_of_information(self, belief: BeliefState,
                            variable: str, goal: Goal) -> float:
        """
        Calculate expected value of observing variable.

        VOI = E[V(belief after observation)] - V(current belief)

        If VOI > cost of observation, it's worth gathering info.
        """
        pass

    def select_observation(self, belief: BeliefState,
                          available_observations: List[str]) -> Optional[str]:
        """
        Select best variable to observe.

        Criteria:
        - Maximizes VOI / cost ratio
        - Reduces entropy most
        - Enables goal achievement
        """
        pass

    def execute_contingent_plan(self, plan: ContingentPlan,
                                belief: BeliefState) -> ExecutionTrace:
        """
        Execute contingent plan.

        Algorithm:
        1. Execute root action
        2. Make observation (if branch point)
        3. Update belief
        4. Select branch based on observation
        5. Recurse
        """
        pass

class BeliefSpacePlanner:
    """Plans in belief space (alternative to contingent planning)."""

    def __init__(self, base_planner: HierarchicalPlanner):
        self.base_planner = base_planner

    def plan_in_belief_space(self, goal: Goal, initial_belief: BeliefState) -> Plan:
        """
        Plan directly in belief space.

        State = belief (not physical state)
        Actions transform beliefs
        Goal = belief satisfying goal condition
        """
        pass
```

### Research Alignment

- **Kaelbling et al. (1998):** "Planning and acting in partially observable domains"
- **Cassandra (1998):** "POMDP solver"
- **Pineau et al. (2003):** "Point-based value iteration"
- **Silver & Veness (2010):** "Monte-Carlo planning in POMDPs"

---

## Integration Architecture

### How Features Compose

**1. Multi-Agent + Resources:**
```python
# Agents negotiate resource allocation
coordinator = MultiAgentCoordinator(agents, protocol="contract_net")
resource_tracker = ResourceTracker(shared_resources)

# Allocate tasks considering resource constraints
allocation = coordinator.allocate_tasks_with_resources(
    tasks, resource_tracker
)
```

**2. Resources + Replanning:**
```python
# Replan when resource constraints violated
monitor = ExecutionMonitor(plan, executor)
tracker = ResourceTracker(resources)

if tracker.find_violations(monitor.trace):
    new_plan = replanner.replan_with_resources(
        trigger=ReplanTrigger.DIVERGENCE,
        current_state=monitor.current_state,
        resource_state=tracker.current_state
    )
```

**3. POMDP + Multi-Agent:**
```python
# Agents plan under partial observability
agents = [
    Agent(
        planner=POMDPPlanner(base_planner, obs_model),
        belief=initial_belief
    )
    for base_planner, initial_belief in agent_configs
]

# Coordinate with belief sharing
coordinator.coordinate_with_beliefs(agents)
```

**4. All Together:**
```python
# Production-grade multi-agent system with full capabilities
system = AdvancedPlanningSystem(
    agents=agents,                    # Multi-agent
    resources=resources,              # Resource constraints
    observation_model=obs_model,      # POMDP
    monitoring=True,                  # Replanning
    coordination_protocol="contract_net"
)

result = system.plan_and_execute(
    tasks=tasks,
    constraints=constraints,
    max_replans=10
)
```

---

## Testing & Validation

### Test Scenarios

**1. Multi-Agent:**
- **Cooperative:** Multi-robot warehouse
- **Competitive:** Auction bidding
- **Negotiation:** Supply chain contracts

**2. Resources:**
- **Budget-constrained:** Project planning
- **Deadline-driven:** Manufacturing
- **Multi-resource:** Mission planning

**3. Replanning:**
- **Failure recovery:** Robot navigation
- **Dynamic goals:** Emergency response
- **Opportunistic:** Stock trading

**4. POMDP:**
- **Hidden state:** Poker, diagnosis
- **Noisy observations:** Robot localization
- **Active sensing:** Medical testing

---

## Deliverables (Option A)

**Code (est. 3,000+ lines):**
- `HoloLoom/planning/multi_agent.py` (~800 lines)
- `HoloLoom/planning/resources.py` (~600 lines)
- `HoloLoom/planning/replanning.py` (~700 lines)
- `HoloLoom/planning/pomdp.py` (~900 lines)

**Demos (est. 1,200+ lines):**
- `demos/demo_multi_agent_warehouse.py` (~300 lines)
- `demos/demo_resource_planning.py` (~300 lines)
- `demos/demo_replanning.py` (~300 lines)
- `demos/demo_pomdp_diagnosis.py` (~300 lines)

**Tests:**
- Unit tests for each component
- Integration tests
- End-to-end scenarios

**Documentation:**
- Architecture document (this file)
- API reference
- Tutorial notebooks

---

## Timeline (Estimated)

- **Multi-Agent:** 6-8 hours
- **Resources:** 4-6 hours
- **Replanning:** 5-7 hours
- **POMDP:** 8-10 hours

**Total:** 23-31 hours of development

---

## Success Criteria

**Option A Complete When:**
1. ✅ All 4 features implemented
2. ✅ All demos running
3. ✅ Tests passing
4. ✅ Integrated with Layer 2 core
5. ✅ Documentation complete
6. ✅ Committed and pushed

**Quality Bar:**
- Production-ready code
- Comprehensive error handling
- Graceful degradation
- Research-aligned algorithms
- Clear documentation

---

**Let's build this!** 🚀

Starting with **Multi-Agent Coordination**...
