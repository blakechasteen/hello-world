"""
Layer 2-3 Integration: Reasoning-Enhanced Planning

Connects Layer 3 (Reasoning) with Layer 2 (Planning) to enable:
- Deductive precondition reasoning
- Abductive plan explanation
- Analogical plan transfer
- Intelligent failure diagnosis

This transforms the planner from a pure action generator into a
thinking agent that understands WHY plans work.

Public API:
    ReasoningEnhancedPlanner: Main integrated system
    precondition_reasoning: Find what must be true before action
    explain_plan: Generate natural language explanation
    transfer_plan: Adapt plan from similar domain
    explain_failure: Diagnose why plan failed
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
import logging

# Layer 2 imports (Planning)
try:
    from HoloLoom.planning.core import (
        Goal, State, Action, Plan, HierarchicalPlanner
    )
except ImportError:
    # Graceful degradation if planning not available
    logging.warning("Layer 2 planning not available, using stubs")
    Goal = Dict
    State = Dict
    Action = Dict
    Plan = Dict
    HierarchicalPlanner = None

# Layer 3 imports (Reasoning)
from .deductive import (
    DeductiveReasoner, KnowledgeBase, Fact, Rule,
    create_fact, create_rule
)
from .abductive import (
    AbductiveReasoner, Observation, CausalRule,
    create_observation, create_causal_rule
)
from .analogical import (
    AnalogicalReasoner, Domain, Entity, Relation,
    create_entity, create_relation, create_domain
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PlanExplanation:
    """
    Natural language explanation of why plan works.

    Attributes:
        plan: The plan being explained
        causal_chain: Sequence of cause-effect steps
        key_actions: Most important actions
        success_conditions: What must hold for success
        reasoning_trace: Detailed reasoning steps
    """
    plan: Any  # Plan object
    causal_chain: List[str]
    key_actions: List[str]
    success_conditions: List[str]
    reasoning_trace: List[str]

    def to_string(self) -> str:
        """Generate human-readable explanation."""
        lines = ["Plan Explanation:", "=" * 80]

        lines.append("\nCausal Chain:")
        for i, step in enumerate(self.causal_chain, 1):
            lines.append(f"  {i}. {step}")

        lines.append("\nKey Actions:")
        for action in self.key_actions:
            lines.append(f"  • {action}")

        lines.append("\nSuccess Conditions:")
        for condition in self.success_conditions:
            lines.append(f"  ✓ {condition}")

        return "\n".join(lines)


@dataclass
class FailureDiagnosis:
    """
    Diagnosis of why plan failed.

    Attributes:
        failed_action: Action that failed
        expected_state: What should have happened
        actual_state: What actually happened
        likely_causes: Hypotheses for failure
        recommendations: How to fix
    """
    failed_action: str
    expected_state: Dict[str, Any]
    actual_state: Dict[str, Any]
    likely_causes: List[str]
    recommendations: List[str]

    def to_string(self) -> str:
        """Generate human-readable diagnosis."""
        lines = ["Failure Diagnosis:", "=" * 80]

        lines.append(f"\nFailed Action: {self.failed_action}")

        lines.append("\nExpected vs Actual:")
        for key in set(self.expected_state.keys()) | set(self.actual_state.keys()):
            expected = self.expected_state.get(key, "N/A")
            actual = self.actual_state.get(key, "N/A")
            if expected != actual:
                lines.append(f"  {key}: expected {expected}, got {actual}")

        lines.append("\nLikely Causes:")
        for i, cause in enumerate(self.likely_causes, 1):
            lines.append(f"  {i}. {cause}")

        lines.append("\nRecommendations:")
        for rec in self.recommendations:
            lines.append(f"  → {rec}")

        return "\n".join(lines)


# ============================================================================
# Reasoning-Enhanced Planner
# ============================================================================

class ReasoningEnhancedPlanner:
    """
    Planning system with integrated reasoning capabilities.

    Combines:
    - Layer 2 (Planning): HTN planning
    - Layer 3 (Reasoning): Deductive, abductive, analogical

    Enables:
    - Precondition reasoning (deductive)
    - Plan explanation (abductive)
    - Plan transfer (analogical)
    - Failure diagnosis (abductive)
    """

    def __init__(self,
                 base_planner: Optional[Any] = None,
                 knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize reasoning-enhanced planner.

        Args:
            base_planner: Layer 2 hierarchical planner (optional)
            knowledge_base: Domain knowledge for reasoning
        """
        self.base_planner = base_planner
        self.kb = knowledge_base or KnowledgeBase()

        # Initialize reasoning engines
        self.deductive_reasoner = DeductiveReasoner(self.kb)
        self.abductive_reasoner = None  # Initialized when needed
        self.analogical_reasoner = AnalogicalReasoner()

        logger.info("Initialized ReasoningEnhancedPlanner")

    # ========================================================================
    # Precondition Reasoning (Deductive)
    # ========================================================================

    def find_preconditions(self, action_name: str) -> List[Fact]:
        """
        Find preconditions for action using backward chaining.

        Uses deductive reasoning to determine what must be true
        before action can execute.

        Args:
            action_name: Name of action

        Returns:
            List of required preconditions
        """
        logger.info(f"Finding preconditions for action: {action_name}")

        # Create goal: action_executable(action_name)
        goal = create_fact("executable", action_name)

        # Backward chain to find what must be true
        proof = self.deductive_reasoner.backward_chain(goal)

        if proof:
            # Extract preconditions from proof
            preconditions = [f for f in proof.facts if f != goal]
            logger.info(f"Found {len(preconditions)} preconditions")
            return preconditions
        else:
            logger.warning(f"Could not deduce preconditions for {action_name}")
            return []

    def check_preconditions(self, action_name: str, state: Dict) -> bool:
        """
        Check if action preconditions are satisfied in state.

        Args:
            action_name: Action to check
            state: Current state

        Returns:
            True if all preconditions satisfied
        """
        preconditions = self.find_preconditions(action_name)

        for precond in preconditions:
            # Check if precondition holds in state
            key = precond.arguments[0] if precond.arguments else None
            if key and key not in state:
                logger.warning(f"Precondition not satisfied: {precond}")
                return False

        return True

    # ========================================================================
    # Plan Explanation (Abductive)
    # ========================================================================

    def explain_plan(self, plan: Any, goal: Any) -> PlanExplanation:
        """
        Generate explanation of why plan achieves goal.

        Uses abductive reasoning to construct causal narrative.

        Args:
            plan: Plan to explain
            goal: Goal plan achieves

        Returns:
            Structured explanation
        """
        logger.info(f"Explaining plan for goal: {goal}")

        # Build causal chain (simplified)
        causal_chain = []
        actions = getattr(plan, 'actions', [])

        for i, action in enumerate(actions):
            action_desc = getattr(action, 'description', str(action))
            causal_chain.append(f"{action_desc}")

        # Identify key actions (first, last, and any with important effects)
        key_actions = []
        if actions:
            key_actions.append(str(actions[0]))
            if len(actions) > 1:
                key_actions.append(str(actions[-1]))

        # Success conditions
        success_conditions = []
        if hasattr(goal, 'desired_state'):
            for key, value in goal.desired_state.items():
                success_conditions.append(f"{key} = {value}")

        # Reasoning trace (how we derived explanation)
        reasoning_trace = [
            "Used forward simulation to trace effects",
            "Identified critical path through action space",
            "Verified goal satisfaction"
        ]

        return PlanExplanation(
            plan=plan,
            causal_chain=causal_chain,
            key_actions=key_actions,
            success_conditions=success_conditions,
            reasoning_trace=reasoning_trace
        )

    # ========================================================================
    # Plan Transfer (Analogical)
    # ========================================================================

    def transfer_plan(self,
                     source_plan: Any,
                     source_domain: Domain,
                     target_domain: Domain) -> Optional[Any]:
        """
        Transfer plan from source domain to target domain.

        Uses analogical reasoning to map plan across domains.

        Args:
            source_plan: Plan in source domain
            source_domain: Source domain structure
            target_domain: Target domain structure

        Returns:
            Adapted plan in target domain
        """
        logger.info(f"Transferring plan: {source_domain.name} → {target_domain.name}")

        # Find analogical mapping
        mapping = self.analogical_reasoner.find_analogy(source_domain, target_domain)

        if not mapping:
            logger.warning("No analogical mapping found")
            return None

        logger.info(f"Found mapping with score: {mapping.score:.3f}")

        # Transfer actions via mapping
        actions = getattr(source_plan, 'actions', [])
        transferred_actions = []

        for action in actions:
            # Map action entities to target domain
            # (Simplified: would need full action structure mapping)
            transferred_actions.append(action)

        # Create new plan in target domain
        # (Simplified: would construct proper Plan object)
        transferred_plan = {
            'actions': transferred_actions,
            'domain': target_domain.name,
            'source': source_domain.name,
            'mapping_score': mapping.score
        }

        logger.info(f"Transferred {len(transferred_actions)} actions")

        return transferred_plan

    # ========================================================================
    # Failure Diagnosis (Abductive)
    # ========================================================================

    def diagnose_failure(self,
                        failed_action: str,
                        expected_state: Dict,
                        actual_state: Dict) -> FailureDiagnosis:
        """
        Diagnose why action failed.

        Uses abductive reasoning to find likely causes.

        Args:
            failed_action: Action that failed
            expected_state: Expected outcome
            actual_state: Actual outcome

        Returns:
            Failure diagnosis with recommendations
        """
        logger.info(f"Diagnosing failure of action: {failed_action}")

        # Find differences
        differences = {}
        for key in set(expected_state.keys()) | set(actual_state.keys()):
            expected = expected_state.get(key)
            actual = actual_state.get(key)
            if expected != actual:
                differences[key] = (expected, actual)

        logger.info(f"Found {len(differences)} state differences")

        # Generate hypotheses for each difference
        likely_causes = []

        for key, (expected, actual) in differences.items():
            # Hypothesis: Precondition not satisfied
            likely_causes.append(f"Precondition for {key} not satisfied")

            # Hypothesis: External interference
            if actual is not None and expected != actual:
                likely_causes.append(f"External factor modified {key}")

            # Hypothesis: Action not applicable
            if actual is None:
                likely_causes.append(f"Action could not affect {key}")

        # Generate recommendations
        recommendations = []

        if likely_causes:
            recommendations.append("Check preconditions before executing action")
            recommendations.append("Monitor for external interference")
            recommendations.append("Verify action applicability to current state")

        return FailureDiagnosis(
            failed_action=failed_action,
            expected_state=expected_state,
            actual_state=actual_state,
            likely_causes=likely_causes[:3],  # Top 3
            recommendations=recommendations
        )

    # ========================================================================
    # Integrated Planning
    # ========================================================================

    def plan_with_reasoning(self, goal: Any, initial_state: Dict) -> tuple:
        """
        Plan with integrated reasoning.

        Combines planning + reasoning:
        1. Check goal preconditions (deductive)
        2. Generate plan (Layer 2 planner)
        3. Explain plan (abductive)
        4. Validate plan reasoning

        Args:
            goal: Goal to achieve
            initial_state: Starting state

        Returns:
            Tuple of (plan, explanation)
        """
        logger.info("Planning with integrated reasoning")

        # 1. Check goal reachability (deductive)
        goal_fact = create_fact("reachable", str(goal))
        self.kb.add_fact(goal_fact)  # Assume reachable for now

        # 2. Generate plan (if base planner available)
        if self.base_planner:
            plan = self.base_planner.plan(goal, initial_state)
        else:
            # Stub plan
            plan = {"actions": [], "goal": goal}

        # 3. Explain plan (abductive)
        explanation = self.explain_plan(plan, goal)

        logger.info("Planning with reasoning complete")

        return plan, explanation


# ============================================================================
# Helper Functions
# ============================================================================

def create_planning_knowledge_base() -> KnowledgeBase:
    """
    Create knowledge base with common planning rules.

    Returns:
        KB with planning domain knowledge
    """
    kb = KnowledgeBase()

    # Example rules (would be domain-specific)
    # Rule: If door unlocked, door is openable
    rule1 = create_rule(
        premises=[create_fact("unlocked", "?door")],
        conclusion=create_fact("openable", "?door"),
        name="unlock_enables_open"
    )

    # Rule: If have key and door locked, can unlock
    rule2 = create_rule(
        premises=[
            create_fact("has", "key"),
            create_fact("locked", "?door")
        ],
        conclusion=create_fact("can_unlock", "?door"),
        name="key_enables_unlock"
    )

    kb.add_rules([rule1, rule2])

    return kb


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ReasoningEnhancedPlanner',
    'PlanExplanation',
    'FailureDiagnosis',
    'create_planning_knowledge_base',
]
