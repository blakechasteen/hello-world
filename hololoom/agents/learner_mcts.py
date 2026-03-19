"""
MCTS Pattern Validator

Uses Monte Carlo Tree Search to validate learned patterns before applying them.

Key Innovation: Don't blindly apply patterns - simulate outcomes first!

State Space:
- State: (Query, Pattern, WorkingMemoryState)
- Actions: Apply pattern, Skip pattern, Modify pattern
- Reward: Expected confidence of outcome

This prevents bad patterns from degrading performance.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from hololoom.agents.learner import WorkingMemoryLearner
from hololoom.agents.mcts_core import MCTSEngine, MCTSStateSpace
from hololoom.agents.types import AgentProfile, LearnedPattern, WorkingMemoryState
from hololoom.protocols.types import Query

# ============================================================================
# Pattern Validation Actions
# ============================================================================

@dataclass
class PatternAction:
    """Action for pattern application"""
    type: str  # "apply", "skip", "modify_focus", "modify_activation", "modify_tension"
    pattern_id: str | None = None
    modification: dict[str, Any] | None = None


@dataclass
class PatternValidationState:
    """State for pattern validation"""
    query: Query
    base_state: WorkingMemoryState
    pattern: LearnedPattern | None = None
    applied_modifications: list[str] = None

    def __post_init__(self):
        if self.applied_modifications is None:
            self.applied_modifications = []


# ============================================================================
# Pattern Validation State Space
# ============================================================================

class PatternValidationStateSpace(MCTSStateSpace):
    """
    State space for validating pattern application.

    MCTS explores:
    - Should we apply this pattern?
    - Should we modify it first?
    - What modifications improve outcomes?
    """

    def __init__(
        self,
        patterns: dict[str, LearnedPattern],
        profile: AgentProfile,
        embedding_model
    ):
        self.patterns = patterns
        self.profile = profile
        self.emb = embedding_model

    def get_legal_actions(self, state: PatternValidationState) -> list[PatternAction]:
        """Get legal actions from validation state"""
        actions = []

        if state.pattern is None:
            # No pattern selected yet - can select or skip all
            for pattern_id, pattern in list(self.patterns.items())[:5]:  # Top 5 patterns
                actions.append(PatternAction(
                    type="apply",
                    pattern_id=pattern_id
                ))

            # Or skip patterns entirely
            actions.append(PatternAction(type="skip"))

        else:
            # Pattern selected - can modify before applying
            if "focus" not in state.applied_modifications:
                actions.append(PatternAction(
                    type="modify_focus",
                    pattern_id=state.pattern.pattern_id,
                    modification={'scale': 0.8}  # Reduce semantic shift
                ))

            if "activation" not in state.applied_modifications:
                actions.append(PatternAction(
                    type="modify_activation",
                    pattern_id=state.pattern.pattern_id,
                    modification={'threshold': 0.6}  # Lower activation threshold
                ))

            if "tension" not in state.applied_modifications:
                actions.append(PatternAction(
                    type="modify_tension",
                    pattern_id=state.pattern.pattern_id,
                    modification={'keep_top_k': 3}  # Only tension top 3 threads
                ))

            # Final action: commit to this pattern
            actions.append(PatternAction(type="commit"))

        return actions

    def apply_action(
        self,
        state: PatternValidationState,
        action: PatternAction
    ) -> PatternValidationState:
        """Apply action to validation state"""
        new_state = PatternValidationState(
            query=state.query,
            base_state=state.base_state.copy(),
            pattern=state.pattern,
            applied_modifications=state.applied_modifications.copy()
        )

        if action.type == "apply":
            # Select pattern
            new_state.pattern = self.patterns[action.pattern_id]

        elif action.type == "skip":
            # Skip all patterns
            new_state.pattern = None

        elif action.type == "modify_focus":
            # Modify semantic focus strength
            new_state.applied_modifications.append("focus")
            # Modification applied during evaluation

        elif action.type == "modify_activation":
            # Modify activation threshold
            new_state.applied_modifications.append("activation")

        elif action.type == "modify_tension":
            # Modify tension strategy
            new_state.applied_modifications.append("tension")

        elif action.type == "commit":
            # Commit to pattern (terminal)
            pass

        return new_state

    async def evaluate(self, state: PatternValidationState) -> float:
        """
        Evaluate expected outcome quality.

        Simulates applying the pattern and estimates confidence.
        """
        if state.pattern is None:
            # No pattern - baseline performance
            return 0.7  # Assume baseline confidence

        # Simulate pattern application
        simulated_state = state.base_state.copy()
        pattern = state.pattern

        # SEMANTIC: Apply focus shift (with modifications)
        focus_scale = 1.0
        if "focus" in state.applied_modifications:
            focus_scale = 0.8  # Reduced shift

        semantic_shift = pattern.semantic_region - simulated_state.focus_vector
        simulated_state.focus_vector += focus_scale * 0.3 * semantic_shift

        # Normalize
        norm = np.linalg.norm(simulated_state.focus_vector)
        if norm > 0:
            simulated_state.focus_vector /= norm

        # ACTIVATION: Apply typical activation pattern
        activation_threshold = 0.5
        if "activation" in state.applied_modifications:
            activation_threshold = 0.6

        for node_id, target_activation in pattern.typical_activation_pattern.items():
            if target_activation > activation_threshold:
                simulated_state.activation_map[node_id] = target_activation

        # TENSION: Apply critical threads
        if "tension" in state.applied_modifications:
            # Only keep top 3
            sorted_threads = sorted(
                pattern.critical_threads,
                key=lambda t: pattern.typical_activation_pattern.get(t, 0.0),
                reverse=True
            )[:3]
            threads_to_tension = sorted_threads
        else:
            threads_to_tension = pattern.critical_threads

        for thread_id in threads_to_tension:
            simulated_state.tensioned_threads.add(thread_id)
            simulated_state.tension_profile[thread_id] = 0.7

        # ESTIMATE OUTCOME QUALITY
        # Based on pattern's historical performance + semantic similarity

        # Historical confidence
        historical_confidence = pattern.avg_confidence

        # Semantic similarity to query
        query_vector = self.emb.encode_scales([state.query.text], size=len(simulated_state.focus_vector))[0]
        norm_product = np.linalg.norm(query_vector) * np.linalg.norm(simulated_state.focus_vector)
        if norm_product > 0:
            semantic_match = np.dot(query_vector, simulated_state.focus_vector) / norm_product
        else:
            semantic_match = 0.0

        # Success rate
        success_rate = pattern.success_rate()

        # Combined estimate
        estimated_confidence = (
            0.5 * historical_confidence +
            0.3 * semantic_match +
            0.2 * success_rate
        )

        # Penalty for modifications (uncertainty)
        modification_penalty = 0.05 * len(state.applied_modifications)
        estimated_confidence -= modification_penalty

        return max(0.0, min(1.0, estimated_confidence))

    def is_terminal(self, state: PatternValidationState) -> bool:
        """Terminal when committed to decision"""
        return False  # Use max_depth instead

    def copy_state(self, state: PatternValidationState) -> PatternValidationState:
        """Copy state"""
        return PatternValidationState(
            query=state.query,
            base_state=state.base_state.copy(),
            pattern=state.pattern,
            applied_modifications=state.applied_modifications.copy()
        )


# ============================================================================
# MCTS-Powered Learner
# ============================================================================

class MCTSLearner(WorkingMemoryLearner):
    """
    Learner with MCTS-based pattern validation.

    Before applying patterns, runs MCTS to:
    1. Decide which pattern to use (if any)
    2. Determine if modifications are needed
    3. Estimate expected outcome
    """

    def __init__(
        self,
        profile: AgentProfile,
        persist_dir,
        mcts_simulations: int = 50,
        validation_threshold: float = 0.75
    ):
        super().__init__(profile, persist_dir)

        self.mcts_simulations = mcts_simulations
        self.validation_threshold = validation_threshold

        # Statistics
        self.mcts_validations = 0
        self.patterns_accepted = 0
        self.patterns_rejected = 0
        self.avg_validation_time = 0.0

    async def validate_and_suggest(
        self,
        query: Query,
        current_state: WorkingMemoryState,
        embedding_model
    ) -> dict:
        """
        Validate patterns using MCTS and suggest best option.

        Returns:
            Dict with 'apply_pattern', 'pattern_id', 'modifications', 'expected_confidence'
        """
        import time
        start_time = time.time()

        if not self.patterns:
            return {'apply_pattern': False, 'reason': 'No patterns learned'}

        # Create state space
        state_space = PatternValidationStateSpace(
            patterns=self.patterns,
            profile=self.profile,
            embedding_model=embedding_model
        )

        # Initial state
        initial_state = PatternValidationState(
            query=query,
            base_state=current_state
        )

        # Run MCTS
        mcts = MCTSEngine(
            state_space=state_space,
            exploration_weight=1.0,  # Lower exploration (we want best, not diverse)
            max_depth=5
        )

        best_action, root = await mcts.search(
            initial_state=initial_state,
            n_simulations=self.mcts_simulations
        )

        # Extract best path
        path = []
        node = root
        while node.children:
            node = node.most_visited_child()
            if node and node.action:
                path.append(node.action)

        # Analyze result
        expected_confidence = node.value() if node else 0.7

        # Update statistics
        self.mcts_validations += 1
        self.avg_validation_time = (
            (self.avg_validation_time * (self.mcts_validations - 1) + (time.time() - start_time)) /
            self.mcts_validations
        )

        # Decision
        if expected_confidence >= self.validation_threshold:
            # Accept pattern
            pattern_action = next((a for a in path if a.type == "apply"), None)

            if pattern_action:
                self.patterns_accepted += 1

                # Extract modifications
                modifications = {
                    'focus_scale': 0.8 if any(a.type == "modify_focus" for a in path) else 1.0,
                    'activation_threshold': 0.6 if any(a.type == "modify_activation" for a in path) else 0.5,
                    'tension_top_k': 3 if any(a.type == "modify_tension" for a in path) else None
                }

                return {
                    'apply_pattern': True,
                    'pattern_id': pattern_action.pattern_id,
                    'pattern': self.patterns[pattern_action.pattern_id],
                    'modifications': modifications,
                    'expected_confidence': expected_confidence,
                    'mcts_simulations': self.mcts_simulations
                }

        # Reject patterns
        self.patterns_rejected += 1

        return {
            'apply_pattern': False,
            'reason': f'Expected confidence {expected_confidence:.2f} < threshold {self.validation_threshold}',
            'expected_confidence': expected_confidence
        }

    def get_mcts_statistics(self) -> dict:
        """Get MCTS validation statistics"""
        return {
            'total_validations': self.mcts_validations,
            'patterns_accepted': self.patterns_accepted,
            'patterns_rejected': self.patterns_rejected,
            'acceptance_rate': self.patterns_accepted / self.mcts_validations if self.mcts_validations > 0 else 0.0,
            'avg_validation_time': self.avg_validation_time
        }
