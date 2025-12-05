"""
Temporal Dynamics for NeuroHood

Predicts how relationships evolve over time using HoloLoom's TemporalCausalDAG.

Combines:
- Discrete time (Spring Dynamics simulation steps)
- Continuous physics (relationship strength, tension)
- Time-lagged causality (yesterday's conflict affects today's stress)

Enables:
- Long-term relationship predictions (weeks, months)
- Intervention timing optimization ("when should we mediate?")
- Early warning detection ("relationship at risk in 7 days")
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from HoloLoom.causal.temporal import TemporalCausalDAG, TimeWindow

from .relationship_scm import RelationshipSCM, RelationshipData, extract_relationship_data
from .. import NeuroHoodState, Relationship

logger = logging.getLogger(__name__)


@dataclass
class TemporalPrediction:
    """Prediction of relationship state at future time."""
    time_step: int  # Days in future
    predicted_strength: float
    predicted_conflict: float
    predicted_stress: float
    confidence: float  # Prediction confidence [0, 1]


class NeuroHoodTemporalEngine:
    """
    Temporal dynamics engine for relationship trajectories.

    Uses time-lagged causal relationships to predict how relationships
    evolve over weeks and months.

    Key insight: Past events influence future states with decay.
        conflict(t-1) -> stress(t) -> retaliation(t) -> conflict(t+1)

    This creates feedback loops and long-term dynamics.
    """

    def __init__(self, scm: RelationshipSCM):
        """
        Initialize temporal engine.

        Args:
            scm: Trained RelationshipSCM
        """
        self.scm = scm
        self.temporal_dag = self._create_temporal_dag()

    def _create_temporal_dag(self) -> TemporalCausalDAG:
        """
        Create temporal causal DAG with time lags.

        Structure:
            conflict(t-1) -> stress(t)
            stress(t-1) -> retaliation(t)
            retaliation(t-1) -> conflict(t)
            relationship_strength(t-1) -> communication_quality(t)
        """
        temporal_dag = TemporalCausalDAG(max_lag=3)  # Look back 3 days

        # Time-lagged relationships
        # Yesterday's conflict affects today's stress
        temporal_dag.add_temporal_edge(
            "conflict_intensity", "stress_level",
            lag=1, strength=0.7
        )

        # Yesterday's stress affects today's retaliation
        temporal_dag.add_temporal_edge(
            "stress_level", "retaliation_likelihood",
            lag=1, strength=0.6
        )

        # Yesterday's retaliation affects today's conflict
        temporal_dag.add_temporal_edge(
            "retaliation_likelihood", "conflict_intensity",
            lag=1, strength=0.5
        )

        # Relationship strength influences communication (slower decay)
        temporal_dag.add_temporal_edge(
            "relationship_strength", "communication_quality",
            lag=1, strength=0.4
        )

        logger.info("Created temporal causal DAG with 4 time-lagged edges")

        return temporal_dag

    def predict_trajectory(
        self,
        initial_state: RelationshipData,
        n_days: int = 30,
        interventions: Optional[Dict[int, Dict[str, float]]] = None
    ) -> List[TemporalPrediction]:
        """
        Predict relationship trajectory over time.

        Args:
            initial_state: Starting relationship state
            n_days: Number of days to predict
            interventions: Optional interventions at specific days
                Example: {7: {"communication_quality": 0.9}}  # Apologize on day 7

        Returns:
            List of TemporalPrediction objects (one per day)

        Example:
            # Predict relationship without intervention
            trajectory = engine.predict_trajectory(
                initial_state=current_relationship,
                n_days=30
            )

            # Check if relationship deteriorates
            final_strength = trajectory[-1].predicted_strength
            if final_strength < 0.3:
                print("WARNING: Relationship at risk!")

            # Predict with intervention
            trajectory_with_apology = engine.predict_trajectory(
                initial_state=current_relationship,
                n_days=30,
                interventions={7: {"communication_quality": 0.9}}
            )

            # Compare outcomes
            improvement = trajectory_with_apology[-1].predicted_strength - final_strength
            print(f"Apology on day 7 improves outcome by {improvement:+.2f}")
        """
        if interventions is None:
            interventions = {}

        predictions = []

        # Current state
        state = {
            "personality_compatibility": initial_state.personality_compatibility,
            "communication_quality": initial_state.communication_quality,
            "conflict_intensity": initial_state.conflict_intensity,
            "stress_level": initial_state.stress_level,
            "relationship_strength": initial_state.relationship_strength,
            "retaliation_likelihood": initial_state.retaliation_likelihood,
        }

        # Simulate forward in time
        for day in range(n_days):
            # Apply intervention if scheduled
            if day in interventions:
                state.update(interventions[day])
                logger.info(f"Applied intervention on day {day}: {interventions[day]}")

            # Predict next state using SCM
            state = self.scm.predict(state)

            # Apply temporal dynamics (feedback loops)
            # Conflict intensity gradually increases/decreases based on relationship
            conflict_feedback = (1.0 - state["relationship_strength"]) * 0.1
            state["conflict_intensity"] = min(1.0, max(0.0,
                state["conflict_intensity"] + conflict_feedback - 0.05  # Natural decay
            ))

            # Communication quality regresses toward personality compatibility
            state["communication_quality"] = (
                state["communication_quality"] * 0.8 +
                state["personality_compatibility"] * 0.2
            )

            # Clamp all values
            for key in state:
                state[key] = min(1.0, max(0.0, state[key]))

            # Store prediction
            predictions.append(TemporalPrediction(
                time_step=day + 1,
                predicted_strength=state["relationship_strength"],
                predicted_conflict=state["conflict_intensity"],
                predicted_stress=state["stress_level"],
                confidence=self._compute_confidence(day, n_days)
            ))

        return predictions

    def _compute_confidence(self, day: int, total_days: int) -> float:
        """
        Compute prediction confidence (decreases with time).

        Args:
            day: Current day
            total_days: Total prediction horizon

        Returns:
            Confidence [0, 1]
        """
        # Linear decay: 1.0 at day 0, 0.5 at last day
        return max(0.5, 1.0 - (day / total_days) * 0.5)

    def find_optimal_intervention_timing(
        self,
        initial_state: RelationshipData,
        intervention: Dict[str, float],
        n_days: int = 30
    ) -> Tuple[int, float]:
        """
        Find the best day to apply an intervention for maximum effect.

        Args:
            initial_state: Starting relationship state
            intervention: Intervention to apply (e.g., {"communication_quality": 0.9})
            n_days: Prediction horizon

        Returns:
            (best_day, final_strength) tuple

        Example:
            # When should Alice apologize for maximum effect?
            best_day, final_strength = engine.find_optimal_intervention_timing(
                initial_state=current_relationship,
                intervention={"communication_quality": 0.9},
                n_days=14
            )

            print(f"Best to apologize on day {best_day}")
            print(f"Final relationship strength: {final_strength:.2f}")
        """
        best_day = 0
        best_outcome = 0.0

        for day in range(n_days):
            # Predict trajectory with intervention on this day
            trajectory = self.predict_trajectory(
                initial_state,
                n_days=n_days,
                interventions={day: intervention}
            )

            final_strength = trajectory[-1].predicted_strength

            if final_strength > best_outcome:
                best_outcome = final_strength
                best_day = day

        logger.info(f"Optimal intervention timing: day {best_day} (outcome: {best_outcome:.2f})")

        return best_day, best_outcome

    def detect_early_warning(
        self,
        initial_state: RelationshipData,
        threshold: float = 0.3,
        n_days: int = 30
    ) -> Optional[int]:
        """
        Detect if relationship will cross warning threshold.

        Args:
            initial_state: Starting relationship state
            threshold: Warning threshold for relationship strength
            n_days: Prediction horizon

        Returns:
            Day when threshold will be crossed, or None if safe

        Example:
            warning_day = engine.detect_early_warning(
                initial_state=current_relationship,
                threshold=0.3,
                n_days=14
            )

            if warning_day:
                print(f"WARNING: Relationship at risk on day {warning_day}")
                print("Immediate intervention recommended!")
        """
        trajectory = self.predict_trajectory(initial_state, n_days=n_days)

        for prediction in trajectory:
            if prediction.predicted_strength < threshold:
                logger.warning(f"Early warning: Threshold crossed on day {prediction.time_step}")
                return prediction.time_step

        logger.info("No early warning detected - relationship stable")
        return None

    def format_trajectory(self, predictions: List[TemporalPrediction]) -> str:
        """Format trajectory predictions as ASCII chart."""
        lines = ["Relationship Trajectory Prediction:", ""]

        # ASCII chart
        lines.append("  Day | Strength | Conflict | Stress | Confidence")
        lines.append("  " + "-" * 55)

        for pred in predictions:
            bar_strength = "=" * int(pred.predicted_strength * 10)
            bar_conflict = "#" * int(pred.predicted_conflict * 10)

            lines.append(
                f"  {pred.time_step:3d} | {pred.predicted_strength:.2f} "
                f"[{bar_strength:10s}] | {pred.predicted_conflict:.2f} "
                f"[{bar_conflict:10s}] | {pred.predicted_stress:.2f} | "
                f"{pred.confidence:.1%}"
            )

        return "\n".join(lines)
