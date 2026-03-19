"""
Free Energy Bridge — Active Inference over Spring Dynamics
==========================================================

Wire 9 of the structured AI integration roadmap.

Reinterprets HoloLoom's spring dynamics energy landscape through the
lens of Karl Friston's Free Energy Principle. The key insight:

    F = -kx  (Hooke's Law)  IS  the gradient of quadratic free energy.
    Spring potential energy  IS  prediction error (surprise).
    The system minimizes free energy by relaxing toward equilibrium.

This bridge adds active inference capabilities on top of existing
spring dynamics WITHOUT changing any existing code. It's a pure
interpretation layer that computes:

1. **Variational Free Energy**: How "surprised" the system is by its
   current state vs. its predictions (= spring potential energy)
2. **Expected Free Energy**: Which action would minimize future surprise
   (= which node activation would reduce total spring energy most)
3. **Epistemic Value**: How much an action would reduce uncertainty
   (= information gain from activating a node)
4. **Pragmatic Value**: How much an action would achieve goals
   (= activation of target nodes)

Thompson Sampling stays for fast individual decisions.
Free energy works at the orchestration level — the meta-controller
that decides WHAT to attend to in the knowledge graph.

Based on:
- Friston (2010): "The free-energy principle: a unified brain theory?"
- VERSES AXIOM (2025): crushed DreamerV3 on Gameworld 10k
- Empirical result: active inference outperforms TS in short sessions

Author: Claude Code (Structured AI Wiring - Wire 9)
Date: 2026-03-18
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class FreeEnergyState:
    """Free energy decomposition of the current system state."""
    total_free_energy: float        # F = potential + complexity
    potential_energy: float         # Spring potential (prediction error / surprise)
    kinetic_energy: float           # Node velocity (rate of belief change)
    complexity: float               # KL(posterior || prior) — cost of updating beliefs
    surprise: float                 # -log P(observations | model) ≈ potential energy
    entropy: float                  # H(beliefs) — uncertainty in current state


@dataclass
class ActionValue:
    """Expected free energy for a candidate action (node activation)."""
    node_id: str
    expected_free_energy: float     # G = epistemic + pragmatic (lower is better)
    epistemic_value: float          # Information gain from this action
    pragmatic_value: float          # Goal-directed value
    energy_reduction: float         # How much this action reduces spring potential


# ============================================================================
# Free Energy Bridge
# ============================================================================

class FreeEnergyBridge:
    """
    Interprets spring dynamics through the free energy principle.

    Does NOT modify spring dynamics — reads their state and provides
    an active inference interpretation. This is a pure observation layer.

    The bridge answers: "Given the current spring energy landscape,
    which nodes should the system attend to next?"
    """

    def __init__(self, prior_precision: float = 1.0):
        """
        Args:
            prior_precision: How strongly the system trusts its priors.
                           Higher = more exploitation, lower = more exploration.
        """
        self.prior_precision = prior_precision
        logger.info(f"FreeEnergyBridge initialized (prior_precision={prior_precision})")

    def compute_free_energy(
        self,
        node_activations: dict[str, float],
        node_velocities: dict[str, float],
        edge_potentials: list[tuple[str, str, float]],
    ) -> FreeEnergyState:
        """
        Compute variational free energy from spring dynamics state.

        The free energy F decomposes as:
            F = E[surprise] + complexity
              = potential_energy + KL(q || p)

        Where:
            potential_energy = Σ ½k(Δa)² (spring potential = prediction error)
            complexity = KL(current activations || prior activations)
            surprise ≈ potential_energy (under quadratic approximation)

        Args:
            node_activations: {node_id: activation} current state
            node_velocities: {node_id: velocity} rates of change
            edge_potentials: [(src, dst, ½k(Δa)²)] pre-computed spring potentials

        Returns:
            FreeEnergyState with full decomposition
        """
        # Potential energy = sum of spring potentials (prediction error)
        potential = sum(e for _, _, e in edge_potentials)

        # Kinetic energy = sum of ½mv²
        kinetic = sum(0.5 * v ** 2 for v in node_velocities.values())

        # Complexity: KL divergence from uniform prior
        # Under active inference, complexity = cost of deviating from prior beliefs
        activations = np.array(list(node_activations.values()))
        if len(activations) > 0 and activations.sum() > 0:
            # Normalize to distribution
            q = activations / (activations.sum() + 1e-12)
            # Prior: uniform
            p = np.ones_like(q) / len(q)
            # KL(q || p) — cost of the current belief distribution
            kl = np.sum(np.where(q > 1e-12, q * np.log(q / p + 1e-12), 0.0))
            complexity = float(kl) * self.prior_precision
        else:
            complexity = 0.0

        # Entropy of current beliefs
        if len(activations) > 0 and activations.sum() > 0:
            q = activations / (activations.sum() + 1e-12)
            entropy = float(-np.sum(np.where(q > 1e-12, q * np.log(q + 1e-12), 0.0)))
        else:
            entropy = 0.0

        total = potential + complexity

        return FreeEnergyState(
            total_free_energy=total,
            potential_energy=potential,
            kinetic_energy=kinetic,
            complexity=complexity,
            surprise=potential,  # Under quadratic approximation
            entropy=entropy,
        )

    def rank_actions(
        self,
        candidates: list[str],
        node_activations: dict[str, float],
        neighbor_potentials: dict[str, float],
        target_nodes: list[str] | None = None,
    ) -> list[ActionValue]:
        """
        Rank candidate node activations by expected free energy.

        Expected Free Energy G = -epistemic_value - pragmatic_value
        Lower G = better action (reduces future surprise more).

        Active inference selects the action that minimizes G:
        - Epistemic value: how much uncertainty would be reduced
        - Pragmatic value: how much closer to goals

        Args:
            candidates: Node IDs to evaluate
            node_activations: Current activation state
            neighbor_potentials: {node_id: sum of spring potentials to neighbors}
            target_nodes: Optional goal nodes (pragmatic value targets)

        Returns:
            List of ActionValue sorted by expected_free_energy (ascending = best first)
        """
        target_set = set(target_nodes or [])
        actions = []

        for node_id in candidates:
            current_activation = node_activations.get(node_id, 0.0)

            # Epistemic value: activating this node would resolve spring tension
            # with its neighbors (reduce prediction error / gain information)
            spring_tension = neighbor_potentials.get(node_id, 0.0)
            epistemic = spring_tension  # Higher tension = more to learn

            # Pragmatic value: is this node a target or connected to targets?
            if node_id in target_set:
                pragmatic = 1.0
            else:
                pragmatic = 0.0

            # Energy reduction: how much would activating this node reduce
            # total spring potential? Nodes with high tension to activated
            # neighbors would reduce energy most.
            energy_reduction = spring_tension * (1.0 - current_activation)

            # Expected free energy: lower is better
            # G = -(epistemic + pragmatic)
            efg = -(epistemic + pragmatic)

            actions.append(ActionValue(
                node_id=node_id,
                expected_free_energy=efg,
                epistemic_value=epistemic,
                pragmatic_value=pragmatic,
                energy_reduction=energy_reduction,
            ))

        # Sort by expected free energy (ascending = best action first)
        actions.sort(key=lambda a: a.expected_free_energy)
        return actions

    def from_spring_dynamics(self, spring_dynamics) -> FreeEnergyState:
        """
        Compute free energy directly from a SpringDynamics instance.

        Args:
            spring_dynamics: HoloLoom SpringDynamics instance

        Returns:
            FreeEnergyState
        """
        node_activations = {
            nid: state.activation
            for nid, state in spring_dynamics.node_states.items()
        }
        node_velocities = {
            nid: state.velocity
            for nid, state in spring_dynamics.node_states.items()
        }

        # Compute edge potentials
        edge_potentials = []
        config = spring_dynamics.config
        for u, v, edge_data in spring_dynamics.graph.G.edges(data=True):
            state_u = spring_dynamics.node_states.get(u)
            state_v = spring_dynamics.node_states.get(v)
            if state_u and state_v:
                edge_type = edge_data.get('type', 'RELATED_TO')
                edge_weight = edge_data.get('weight', 1.0)
                k = config.get_edge_stiffness(edge_type, edge_weight)
                delta = state_v.activation - state_u.activation
                potential = 0.5 * k * delta ** 2
                edge_potentials.append((u, v, potential))

        return self.compute_free_energy(node_activations, node_velocities, edge_potentials)


# ============================================================================
# Factory
# ============================================================================

def create_free_energy_bridge(prior_precision: float = 1.0) -> FreeEnergyBridge:
    """Create a free energy bridge for active inference over spring dynamics."""
    return FreeEnergyBridge(prior_precision=prior_precision)
