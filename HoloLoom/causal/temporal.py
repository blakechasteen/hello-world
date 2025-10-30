"""
Temporal Causal Networks

Causality over TIME - because real-world causes and effects don't happen instantly!

Key questions:
- How LONG after treatment does recovery occur?
- What's the TRAJECTORY of causal effects over time?
- Can we predict FUTURE states from interventions?

Example:
    treatment[t=0] → recovery[t=5]  # Recovery happens 5 days later

Instead of static:
    treatment → recovery  # When? Who knows!
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from .dag import CausalDAG, CausalNode, CausalEdge, NodeType

logger = logging.getLogger(__name__)


@dataclass
class TemporalEdge:
    """
    Time-lagged causal relationship.

    Examples:
        treatment[t] → recovery[t+5]  (lag=5)
        exercise[t] → fitness[t+30]   (lag=30 days)
    """
    source: str
    target: str
    lag: int  # Time steps from source to target
    strength: float = 1.0
    mechanism: str = ""
    confidence: float = 1.0

    def __repr__(self):
        return f"{self.source}[t] → {self.target}[t+{self.lag}] (strength={self.strength:.2f})"


@dataclass
class TemporalState:
    """State of all variables at a specific time."""
    time: int
    values: Dict[str, float]

    def __getitem__(self, var: str) -> float:
        return self.values[var]


class TemporalCausalDAG:
    """
    Temporal causal model - causality over time.

    Extends CausalDAG with time lags:
    - Static DAG: X → Y
    - Temporal DAG: X[t] → Y[t+lag]

    Key features:
    - Time-lagged edges
    - Trajectory prediction
    - Temporal intervention effects
    - Granger causality

    Usage:
        tcdag = TemporalCausalDAG(variables=['treatment', 'recovery'])

        # Treatment causes recovery after 5 days
        tcdag.add_temporal_edge(TemporalEdge(
            source='treatment',
            target='recovery',
            lag=5,
            strength=0.6
        ))

        # Predict future trajectory
        trajectory = tcdag.predict_trajectory(
            initial_state={'treatment': 1, 'recovery': 0},
            steps=10
        )
    """

    def __init__(self, variables: List[str], max_lag: int = 10):
        """
        Initialize temporal causal DAG.

        Args:
            variables: Variable names
            max_lag: Maximum time lag to consider
        """
        self.variables = variables
        self.max_lag = max_lag
        self.temporal_edges: List[TemporalEdge] = []

        # For each variable, track its temporal dependencies
        self.parents: Dict[str, List[Tuple[str, int]]] = {
            var: [] for var in variables
        }

    def add_temporal_edge(self, edge: TemporalEdge):
        """Add time-lagged causal edge."""
        if edge.source not in self.variables:
            raise ValueError(f"Unknown variable: {edge.source}")
        if edge.target not in self.variables:
            raise ValueError(f"Unknown variable: {edge.target}")
        if edge.lag < 0:
            raise ValueError(f"Lag must be non-negative, got {edge.lag}")
        if edge.lag > self.max_lag:
            raise ValueError(f"Lag {edge.lag} exceeds max_lag {self.max_lag}")

        self.temporal_edges.append(edge)
        self.parents[edge.target].append((edge.source, edge.lag))

        logger.info(f"Added edge: {edge}")

    def predict_step(
        self,
        current_state: TemporalState,
        history: List[TemporalState]
    ) -> Dict[str, float]:
        """
        Predict next time step from current state and history.

        Args:
            current_state: Current values
            history: Past states (for time-lagged effects)

        Returns:
            Predicted values at next time step
        """
        next_values = {}
        current_time = current_state.time

        for var in self.variables:
            # Sum up all temporal influences
            influences = []

            for source, lag in self.parents[var]:
                # Find state at t - lag
                required_time = current_time - lag + 1  # +1 for next step

                if required_time < 0:
                    # Not enough history
                    continue

                # Get source value at that time
                if required_time == current_time + 1:
                    # Depends on current values (lag=0)
                    source_value = current_state[source]
                else:
                    # Look in history
                    hist_idx = required_time
                    if hist_idx < len(history):
                        source_value = history[hist_idx][source]
                    else:
                        continue

                # Find edge strength
                edge = next((e for e in self.temporal_edges
                           if e.source == source and e.target == var and e.lag == lag), None)

                if edge:
                    influences.append(edge.strength * source_value)

            # Combine influences
            if influences:
                next_values[var] = sum(influences)
            else:
                # No parents: Keep current value (or drift to 0)
                next_values[var] = current_state[var] * 0.9  # Decay

        return next_values

    def predict_trajectory(
        self,
        initial_state: Dict[str, float],
        steps: int,
        interventions: Optional[Dict[int, Dict[str, float]]] = None
    ) -> List[TemporalState]:
        """
        Predict temporal trajectory starting from initial state.

        Args:
            initial_state: Starting values
            steps: Number of time steps to predict
            interventions: Optional interventions {time: {var: value}}

        Returns:
            List of states over time
        """
        if interventions is None:
            interventions = {}

        trajectory = [TemporalState(time=0, values=initial_state.copy())]

        for t in range(steps):
            current_state = trajectory[-1]

            # Predict next state
            next_values = self.predict_step(current_state, trajectory)

            # Apply interventions
            if t + 1 in interventions:
                for var, value in interventions[t + 1].items():
                    next_values[var] = value
                    logger.debug(f"Intervention at t={t+1}: {var} = {value}")

            # Create next state
            next_state = TemporalState(time=t + 1, values=next_values)
            trajectory.append(next_state)

        return trajectory

    def intervene_trajectory(
        self,
        intervention_time: int,
        intervention: Dict[str, float],
        initial_state: Dict[str, float],
        total_steps: int
    ) -> Tuple[List[TemporalState], List[TemporalState]]:
        """
        Compare factual and counterfactual trajectories.

        Args:
            intervention_time: When to intervene
            intervention: What to intervene on
            initial_state: Starting state
            total_steps: Total time steps

        Returns:
            (factual_trajectory, counterfactual_trajectory)
        """
        # Factual: No intervention
        factual = self.predict_trajectory(initial_state, total_steps)

        # Counterfactual: Intervention at specified time
        counterfactual = self.predict_trajectory(
            initial_state,
            total_steps,
            interventions={intervention_time: intervention}
        )

        return factual, counterfactual

    def compute_temporal_ate(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float,
        control_value: float,
        initial_state: Dict[str, float],
        observation_time: int
    ) -> float:
        """
        Compute temporal Average Treatment Effect.

        ATE at time t = E[Y[t] | do(X=1)] - E[Y[t] | do(X=0)]

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Treatment value
            control_value: Control value
            initial_state: Starting state
            observation_time: When to measure outcome

        Returns:
            Temporal ATE
        """
        # Treatment trajectory
        treatment_traj = self.predict_trajectory(
            initial_state,
            observation_time,
            interventions={0: {treatment: treatment_value}}
        )

        # Control trajectory
        control_traj = self.predict_trajectory(
            initial_state,
            observation_time,
            interventions={0: {treatment: control_value}}
        )

        # ATE = difference in outcome at observation time
        treatment_outcome = treatment_traj[observation_time][outcome]
        control_outcome = control_traj[observation_time][outcome]

        ate = treatment_outcome - control_outcome

        logger.info(
            f"Temporal ATE (t={observation_time}): {ate:.3f} "
            f"({treatment_outcome:.3f} - {control_outcome:.3f})"
        )

        return ate

    def granger_causality(
        self,
        source: str,
        target: str,
        max_lag_test: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Test Granger causality: Does source help predict target?

        Granger causality: X Granger-causes Y if past values of X
        improve prediction of Y beyond past values of Y alone.

        Args:
            source: Potential cause
            target: Effect
            max_lag_test: Maximum lag to test (default: self.max_lag)

        Returns:
            (granger_causes, max_strength)
        """
        if max_lag_test is None:
            max_lag_test = self.max_lag

        # Find all temporal edges from source to target
        relevant_edges = [
            e for e in self.temporal_edges
            if e.source == source and e.target == target and e.lag <= max_lag_test
        ]

        if not relevant_edges:
            return False, 0.0

        # Aggregate evidence
        max_strength = max(e.strength for e in relevant_edges)

        # Granger-causes if any edge has significant strength
        granger_causes = max_strength > 0.1

        logger.info(
            f"Granger causality: {source} → {target}: "
            f"{granger_causes} (strength={max_strength:.3f})"
        )

        return granger_causes, max_strength

    def find_optimal_intervention_timing(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float,
        initial_state: Dict[str, float],
        max_time: int = None
    ) -> Tuple[int, float]:
        """
        Find optimal time to intervene for maximum effect.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Treatment value
            initial_state: Starting state
            max_time: Max time to search

        Returns:
            (optimal_time, max_effect)
        """
        if max_time is None:
            max_time = self.max_lag

        best_time = 0
        max_effect = 0

        for intervention_time in range(max_time + 1):
            # Measure effect at various observation times
            total_effect = 0
            n_observations = 0

            for obs_time in range(intervention_time + 1, max_time + 1):
                # Treatment trajectory
                treatment_traj = self.predict_trajectory(
                    initial_state,
                    obs_time,
                    interventions={intervention_time: {treatment: treatment_value}}
                )

                # Control trajectory
                control_traj = self.predict_trajectory(
                    initial_state,
                    obs_time,
                    interventions={intervention_time: {treatment: 0}}
                )

                # Effect at this observation time
                effect = treatment_traj[obs_time][outcome] - control_traj[obs_time][outcome]
                total_effect += abs(effect)
                n_observations += 1

            # Average effect across observation times
            avg_effect = total_effect / n_observations if n_observations > 0 else 0

            if avg_effect > max_effect:
                max_effect = avg_effect
                best_time = intervention_time

        logger.info(
            f"Optimal intervention time: t={best_time} "
            f"(effect={max_effect:.3f})"
        )

        return best_time, max_effect

    def to_static_dag(self) -> CausalDAG:
        """
        Convert temporal DAG to static DAG (ignoring lags).

        Useful for visualization or applying static algorithms.
        """
        dag = CausalDAG()

        # Add nodes
        for var in self.variables:
            dag.add_node(CausalNode(var, NodeType.OBSERVABLE))

        # Add edges (collapse temporal edges)
        edge_map = {}
        for tedge in self.temporal_edges:
            key = (tedge.source, tedge.target)
            if key not in edge_map:
                edge_map[key] = tedge.strength
            else:
                # Aggregate multiple lags
                edge_map[key] = max(edge_map[key], tedge.strength)

        for (source, target), strength in edge_map.items():
            dag.add_edge(CausalEdge(
                source, target,
                strength=strength,
                mechanism="temporal (lags collapsed)"
            ))

        return dag

    def __repr__(self):
        return f"TemporalCausalDAG({len(self.variables)} vars, {len(self.temporal_edges)} temporal edges)"
