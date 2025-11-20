"""
Hierarchical MCTS Planner

Multi-step query planning using hierarchical Monte Carlo Tree Search.

Scales:
- MACRO: Goal decomposition (What high-level goals to achieve?)
- MESO: Action selection (What queries to run?)
- MICRO: Parameter tuning (What context to use?)

Example:
    Goal: "Create Q4 budget report"
    Macro: [gather_data, analyze, format, review]
    Meso: ["Q4 revenue?", "Q4 expenses?", "Calculate margin", ...]
    Micro: [focus=[financial_data], activate=[Q4], ...]
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

from HoloLoom.agents.mcts_core import MCTSStateSpace, MCTSEngine, HierarchicalMCTS
from HoloLoom.agents.orchestrator_mcts import MCTSAgentOrchestrator
from HoloLoom.Documentation.types import Query


# ============================================================================
# Planning State Representations
# ============================================================================

@dataclass(frozen=True)
class MacroGoal:
    """High-level goal"""
    description: str
    domain: str  # "budget", "architecture", etc.
    estimated_steps: int
    priority: float = 1.0


@dataclass(frozen=True)
class MesoAction:
    """Mid-level action (query to run)"""
    query_text: str
    goal: MacroGoal
    dependencies: tuple = ()  # Query IDs that must complete first (tuple for frozen)
    estimated_confidence: float = 0.7


@dataclass(frozen=True)
class MicroParameters:
    """Low-level parameters (working memory config)"""
    focus_direction: Optional[tuple] = None  # Tuple for frozen (convert from numpy)
    activate_nodes: tuple = ()  # Tuple for frozen
    tension_threads: tuple = ()  # Tuple for frozen
    mcts_simulations: int = 100


@dataclass
class PlanningState:
    """Complete planning state across all scales"""
    # Current position
    current_goal: Optional[MacroGoal] = None
    current_action: Optional[MesoAction] = None
    current_parameters: Optional[MicroParameters] = None

    # History
    completed_goals: List[MacroGoal] = None
    completed_actions: List[MesoAction] = None
    results: Dict[str, Any] = None

    # Target
    target_goal: str = ""  # Overall objective

    def __post_init__(self):
        if self.completed_goals is None:
            self.completed_goals = []
        if self.completed_actions is None:
            self.completed_actions = []
        if self.results is None:
            self.results = {}

    def is_complete(self) -> bool:
        """Check if planning is complete"""
        return len(self.completed_goals) >= 3  # Arbitrary completion criterion


# ============================================================================
# Macro State Space - Goal Decomposition
# ============================================================================

class MacroStateSpace(MCTSStateSpace):
    """
    Macro-level state space for goal decomposition.

    Actions: Select next high-level goal
    Reward: Progress toward overall objective
    """

    def __init__(self, agent: MCTSAgentOrchestrator, target_goal: str):
        self.agent = agent
        self.target_goal = target_goal

    def get_legal_actions(self, state: PlanningState) -> List[MacroGoal]:
        """Get possible next goals"""
        # Domain-specific goal generation
        if "budget" in self.target_goal.lower():
            goals = [
                MacroGoal("gather_financial_data", "budget", 3, 1.0),
                MacroGoal("analyze_spending", "budget", 2, 0.9),
                MacroGoal("forecast_trends", "budget", 2, 0.7),
                MacroGoal("prepare_report", "budget", 1, 0.8)
            ]
        elif "architecture" in self.target_goal.lower():
            goals = [
                MacroGoal("analyze_structure", "architecture", 3, 1.0),
                MacroGoal("identify_patterns", "architecture", 2, 0.9),
                MacroGoal("evaluate_tradeoffs", "architecture", 2, 0.8),
                MacroGoal("recommend_improvements", "architecture", 1, 0.7)
            ]
        else:
            goals = [
                MacroGoal("explore_domain", "general", 2, 1.0),
                MacroGoal("gather_info", "general", 2, 0.9),
                MacroGoal("synthesize", "general", 1, 0.7)
            ]

        # Filter out completed goals
        completed_descriptions = [g.description for g in state.completed_goals]
        return [g for g in goals if g.description not in completed_descriptions]

    def apply_action(self, state: PlanningState, action: MacroGoal) -> PlanningState:
        """Select a goal"""
        return PlanningState(
            current_goal=action,
            current_action=state.current_action,
            current_parameters=state.current_parameters,
            completed_goals=state.completed_goals.copy(),
            completed_actions=state.completed_actions.copy(),
            results=state.results.copy(),
            target_goal=state.target_goal
        )

    async def evaluate(self, state: PlanningState) -> float:
        """Evaluate progress toward target goal"""
        if state.current_goal is None:
            return 0.0

        # Reward = goal priority × completion progress
        progress = len(state.completed_goals) / 3.0  # Normalize
        priority = state.current_goal.priority

        # Bonus for goal relevance
        goal_words = set(state.current_goal.description.lower().split('_'))
        target_words = set(state.target_goal.lower().split())
        relevance = len(goal_words & target_words) / max(len(target_words), 1)

        return 0.4 * progress + 0.3 * priority + 0.3 * relevance

    def is_terminal(self, state: PlanningState) -> bool:
        return state.is_complete()

    def copy_state(self, state: PlanningState) -> PlanningState:
        return PlanningState(
            current_goal=state.current_goal,
            current_action=state.current_action,
            current_parameters=state.current_parameters,
            completed_goals=state.completed_goals.copy(),
            completed_actions=state.completed_actions.copy(),
            results=state.results.copy(),
            target_goal=state.target_goal
        )


# ============================================================================
# Meso State Space - Query Selection
# ============================================================================

class MesoStateSpace(MCTSStateSpace):
    """
    Meso-level state space for query selection.

    Actions: Choose next query to run
    Reward: Information gain + goal progress
    """

    def __init__(self, agent: MCTSAgentOrchestrator, goal: MacroGoal):
        self.agent = agent
        self.goal = goal

    def get_legal_actions(self, state: PlanningState) -> List[MesoAction]:
        """Get possible queries for current goal"""
        # Generate queries based on goal
        if self.goal.description == "gather_financial_data":
            queries = [
                MesoAction("What is Q4 revenue?", self.goal),
                MesoAction("What are Q4 expenses?", self.goal),
                MesoAction("What is Q4 profit margin?", self.goal)
            ]
        elif self.goal.description == "analyze_spending":
            queries = [
                MesoAction("Which categories had highest spending?", self.goal),
                MesoAction("How does spending compare to Q3?", self.goal),
                MesoAction("What are spending trends?", self.goal)
            ]
        else:
            # Generic queries
            queries = [
                MesoAction(f"Explore {self.goal.description}", self.goal),
                MesoAction(f"Analyze {self.goal.description}", self.goal)
            ]

        # Filter out completed
        completed_texts = [a.query_text for a in state.completed_actions]
        return [q for q in queries if q.query_text not in completed_texts]

    def apply_action(self, state: PlanningState, action: MesoAction) -> PlanningState:
        """Select a query"""
        return PlanningState(
            current_goal=state.current_goal,
            current_action=action,
            current_parameters=state.current_parameters,
            completed_goals=state.completed_goals.copy(),
            completed_actions=state.completed_actions.copy(),
            results=state.results.copy(),
            target_goal=state.target_goal
        )

    async def evaluate(self, state: PlanningState) -> float:
        """Evaluate query quality"""
        if state.current_action is None:
            return 0.0

        # Simulate query execution (simplified)
        # In reality, would run through agent
        estimated_confidence = state.current_action.estimated_confidence

        # Novelty bonus (prefer unexplored queries)
        novelty = 1.0 / (len(state.completed_actions) + 1)

        return 0.7 * estimated_confidence + 0.3 * novelty

    def is_terminal(self, state: PlanningState) -> bool:
        return len(state.completed_actions) >= self.goal.estimated_steps

    def copy_state(self, state: PlanningState) -> PlanningState:
        return PlanningState(
            current_goal=state.current_goal,
            current_action=state.current_action,
            current_parameters=state.current_parameters,
            completed_goals=state.completed_goals.copy(),
            completed_actions=state.completed_actions.copy(),
            results=state.results.copy(),
            target_goal=state.target_goal
        )


# ============================================================================
# Micro State Space - Parameter Optimization
# ============================================================================

class MicroStateSpace(MCTSStateSpace):
    """
    Micro-level state space for parameter optimization.

    Actions: Adjust working memory parameters
    Reward: Expected query quality
    """

    def __init__(self, agent: MCTSAgentOrchestrator, action: MesoAction):
        self.agent = agent
        self.action = action

    def get_legal_actions(self, state: PlanningState) -> List[MicroParameters]:
        """Get possible parameter configurations"""
        params = []

        # Default
        params.append(MicroParameters())

        # High exploration
        params.append(MicroParameters(mcts_simulations=200))

        # Fast mode
        params.append(MicroParameters(mcts_simulations=50))

        # Domain-focused (if we had access to working memory state)
        # params.append(MicroParameters(activate_nodes=['budget', 'Q4']))

        return params

    def apply_action(self, state: PlanningState, action: MicroParameters) -> PlanningState:
        """Select parameters"""
        return PlanningState(
            current_goal=state.current_goal,
            current_action=state.current_action,
            current_parameters=action,
            completed_goals=state.completed_goals.copy(),
            completed_actions=state.completed_actions.copy(),
            results=state.results.copy(),
            target_goal=state.target_goal
        )

    async def evaluate(self, state: PlanningState) -> float:
        """Evaluate parameter configuration"""
        if state.current_parameters is None:
            return 0.5

        # Heuristic evaluation
        # Higher simulations = better but slower
        sim_quality = min(state.current_parameters.mcts_simulations / 200.0, 1.0)

        return 0.7 + 0.3 * sim_quality

    def is_terminal(self, state: PlanningState) -> bool:
        return state.current_parameters is not None

    def copy_state(self, state: PlanningState) -> PlanningState:
        return PlanningState(
            current_goal=state.current_goal,
            current_action=state.current_action,
            current_parameters=state.current_parameters,
            completed_goals=state.completed_goals.copy(),
            completed_actions=state.completed_actions.copy(),
            results=state.results.copy(),
            target_goal=state.target_goal
        )


# ============================================================================
# Hierarchical MCTS Planner
# ============================================================================

class HierarchicalMCTSPlanner:
    """
    Hierarchical MCTS planner for multi-step queries.

    Uses 3-level hierarchy:
    - Macro: What goals? (50 simulations)
    - Meso: What queries? (100 simulations)
    - Micro: What parameters? (200 simulations)
    """

    def __init__(
        self,
        agent: MCTSAgentOrchestrator,
        macro_budget: int = 50,
        meso_budget: int = 100,
        micro_budget: int = 200
    ):
        self.agent = agent
        self.macro_budget = macro_budget
        self.meso_budget = meso_budget
        self.micro_budget = micro_budget

    async def plan(self, target_goal: str) -> List[Dict[str, Any]]:
        """
        Create hierarchical plan for achieving target goal.

        Returns:
            List of planned queries with parameters
        """
        initial_state = PlanningState(target_goal=target_goal)

        # MACRO: Plan goals
        macro_space = MacroStateSpace(self.agent, target_goal)
        macro_engine = MCTSEngine(macro_space, max_depth=5)

        _, macro_root = await macro_engine.search(
            initial_state,
            n_simulations=self.macro_budget
        )

        # Extract goal sequence
        goals = self._extract_goals(macro_root)

        # MESO + MICRO: Plan each goal
        complete_plan = []

        for goal in goals:
            # MESO: Plan queries for this goal
            meso_space = MesoStateSpace(self.agent, goal)
            meso_engine = MCTSEngine(meso_space, max_depth=goal.estimated_steps)

            _, meso_root = await meso_engine.search(
                PlanningState(current_goal=goal, target_goal=target_goal),
                n_simulations=self.meso_budget
            )

            actions = self._extract_actions(meso_root)

            # MICRO: Optimize parameters for each query
            for action in actions:
                micro_space = MicroStateSpace(self.agent, action)
                micro_engine = MCTSEngine(micro_space, max_depth=3)

                _, micro_root = await micro_engine.search(
                    PlanningState(current_goal=goal, current_action=action, target_goal=target_goal),
                    n_simulations=self.micro_budget
                )

                # Extract best parameters
                if micro_root.children:
                    best = micro_root.most_visited_child()
                    params = best.state.current_parameters if best else MicroParameters()
                else:
                    params = MicroParameters()

                complete_plan.append({
                    'goal': goal.description,
                    'query': action.query_text,
                    'parameters': params,
                    'estimated_confidence': action.estimated_confidence
                })

        return complete_plan

    def _extract_goals(self, root) -> List[MacroGoal]:
        """Extract goal sequence from macro tree"""
        goals = []
        node = root

        while node.children:
            node = node.most_visited_child()
            if node and node.state.current_goal:
                goals.append(node.state.current_goal)

        return goals

    def _extract_actions(self, root) -> List[MesoAction]:
        """Extract action sequence from meso tree"""
        actions = []
        node = root

        while node.children:
            node = node.most_visited_child()
            if node and node.state.current_action:
                actions.append(node.state.current_action)

        return actions

    async def execute_plan(self, plan: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute planned queries with optimized parameters.

        Returns:
            List of query results
        """
        results = []

        for step in plan:
            query = Query(text=step['query'])
            params = step['parameters']

            # Configure agent with optimized parameters
            # (Would set mcts_simulations, etc.)

            # Execute query
            result = await self.agent.query(query, use_mcts=True)
            results.append(result)

        return results
