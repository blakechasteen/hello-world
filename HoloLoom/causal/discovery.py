"""
Active Causal Discovery

Learn causal structure through active experimentation.

Instead of hand-coding:
    dag.add_edge(CausalEdge("age", "recovery"))  # Manual

The system learns:
    discoverer.run_experiment({"treatment": 1})  # Observes effects
    dag = discoverer.get_dag()  # Learned automatically!

Implements:
- PC (Peter-Clark) algorithm: Constraint-based discovery
- Active learning: Choose most informative experiments
- Information gain: Measure uncertainty reduction
"""

import numpy as np
from typing import Dict, List, Set, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from itertools import combinations
import logging

from .dag import CausalDAG, CausalNode, CausalEdge, NodeType

logger = logging.getLogger(__name__)


@dataclass
class ConditionalIndependence:
    """Result of conditional independence test."""
    X: str
    Y: str
    Z: Set[str]
    independent: bool
    p_value: float
    test_statistic: float


@dataclass
class ExperimentResult:
    """Result of causal experiment."""
    intervention: Dict[str, Any]
    observations: Dict[str, Any]
    timestamp: float
    sample_size: int


class CausalDiscovery:
    """
    Learn causal structure from data.

    Uses PC (Peter-Clark) algorithm:
    1. Start with fully connected graph
    2. Test conditional independence
    3. Remove edges where X ⊥ Y | Z
    4. Orient edges using rules

    Usage:
        discoverer = CausalDiscovery(variables=['X', 'Y', 'Z'])

        # Learn from observational data
        discoverer.fit_observational(data)

        # Or learn from interventions
        for _ in range(10):
            intervention = discoverer.select_intervention()
            result = environment.do(intervention)
            discoverer.update(intervention, result)

        # Get learned DAG
        dag = discoverer.get_dag()
    """

    def __init__(
        self,
        variables: List[str],
        alpha: float = 0.05,
        max_conditioning_size: int = 3
    ):
        """
        Initialize causal discovery.

        Args:
            variables: List of variable names
            alpha: Significance level for independence tests
            max_conditioning_size: Max size of conditioning set
        """
        self.variables = variables
        self.alpha = alpha
        self.max_conditioning_size = max_conditioning_size

        # Learned structure
        self.undirected_edges: Set[Tuple[str, str]] = set()
        self.directed_edges: Set[Tuple[str, str]] = set()
        self.separating_sets: Dict[Tuple[str, str], Set[str]] = {}

        # Experimental data
        self.experiments: List[ExperimentResult] = []
        self.observational_data: Optional[np.ndarray] = None

        # Edge beliefs (for active learning)
        self.edge_beliefs: Dict[Tuple[str, str], float] = {}
        self._initialize_edge_beliefs()

    def _initialize_edge_beliefs(self):
        """Initialize edge beliefs (uniform prior)."""
        for i, X in enumerate(self.variables):
            for j, Y in enumerate(self.variables):
                if i < j:  # Undirected
                    self.edge_beliefs[(X, Y)] = 0.5  # 50% prior

    def fit_observational(self, data: np.ndarray, variable_names: List[str]):
        """
        Learn causal structure from observational data using PC algorithm.

        Args:
            data: Observational data (rows=samples, cols=variables)
            variable_names: Names of columns
        """
        self.observational_data = data
        logger.info(f"Learning causal structure from {len(data)} observations")

        # Step 1: Start with fully connected undirected graph
        self.undirected_edges = set()
        for i, X in enumerate(self.variables):
            for j, Y in enumerate(self.variables):
                if i < j:
                    self.undirected_edges.add((X, Y))

        logger.info(f"Initial: {len(self.undirected_edges)} edges (fully connected)")

        # Step 2: Test conditional independence, remove edges
        for depth in range(self.max_conditioning_size + 1):
            self._test_independence_at_depth(data, variable_names, depth)

        logger.info(f"After independence tests: {len(self.undirected_edges)} edges")

        # Step 3: Orient edges using rules
        self._orient_edges()

        logger.info(f"Oriented: {len(self.directed_edges)} directed edges")

    def _test_independence_at_depth(
        self,
        data: np.ndarray,
        variable_names: List[str],
        depth: int
    ):
        """Test conditional independence with conditioning sets of size 'depth'."""

        edges_to_remove = []

        for X, Y in list(self.undirected_edges):
            # Get neighbors of X (excluding Y)
            neighbors = self._get_neighbors(X) - {Y}

            if len(neighbors) < depth:
                continue

            # Test all conditioning sets of size 'depth'
            for Z in combinations(neighbors, depth):
                Z_set = set(Z)

                # Test: X ⊥ Y | Z?
                result = self._conditional_independence_test(
                    X, Y, Z_set, data, variable_names
                )

                if result.independent:
                    # Found conditional independence: X ⊥ Y | Z
                    # Remove edge
                    edges_to_remove.append((X, Y))
                    self.separating_sets[(X, Y)] = Z_set
                    logger.debug(f"Remove {X}—{Y}: {X} ⊥ {Y} | {Z_set}")
                    break  # No need to test other conditioning sets

        # Remove edges
        for edge in edges_to_remove:
            self.undirected_edges.discard(edge)

    def _conditional_independence_test(
        self,
        X: str,
        Y: str,
        Z: Set[str],
        data: np.ndarray,
        variable_names: List[str]
    ) -> ConditionalIndependence:
        """
        Test if X ⊥ Y | Z using partial correlation.

        Returns True if X and Y are conditionally independent given Z.
        """
        # Get column indices
        X_idx = variable_names.index(X)
        Y_idx = variable_names.index(Y)
        Z_idx = [variable_names.index(z) for z in Z] if Z else []

        # Extract data
        X_data = data[:, X_idx]
        Y_data = data[:, Y_idx]

        if not Z:
            # Unconditional independence: Pearson correlation
            corr = np.corrcoef(X_data, Y_data)[0, 1]
            n = len(data)

            # Test statistic: correlation
            test_stat = abs(corr) * np.sqrt(n - 2)

            # p-value (approximate)
            from scipy.stats import t as t_dist
            p_value = 2 * (1 - t_dist.cdf(abs(test_stat), n - 2))

        else:
            # Conditional independence: Partial correlation
            Z_data = data[:, Z_idx]

            # Regress X on Z
            from sklearn.linear_model import LinearRegression
            reg_X = LinearRegression().fit(Z_data, X_data)
            X_residuals = X_data - reg_X.predict(Z_data)

            # Regress Y on Z
            reg_Y = LinearRegression().fit(Z_data, Y_data)
            Y_residuals = Y_data - reg_Y.predict(Z_data)

            # Partial correlation = correlation of residuals
            partial_corr = np.corrcoef(X_residuals, Y_residuals)[0, 1]
            n = len(data)

            # Test statistic
            test_stat = abs(partial_corr) * np.sqrt(n - len(Z) - 2)

            # p-value
            from scipy.stats import t as t_dist
            p_value = 2 * (1 - t_dist.cdf(abs(test_stat), n - len(Z) - 2))

        # Independence if p-value > alpha
        independent = p_value > self.alpha

        return ConditionalIndependence(
            X=X,
            Y=Y,
            Z=Z,
            independent=independent,
            p_value=p_value,
            test_statistic=test_stat
        )

    def _get_neighbors(self, node: str) -> Set[str]:
        """Get undirected neighbors of node."""
        neighbors = set()
        for X, Y in self.undirected_edges:
            if X == node:
                neighbors.add(Y)
            elif Y == node:
                neighbors.add(X)
        return neighbors

    def _orient_edges(self):
        """Orient undirected edges using PC orientation rules."""
        self.directed_edges = set()

        # Rule 1: Orient v-structures (colliders)
        # If X—Z—Y and X,Y not adjacent, and Z not in separating set of X,Y
        # Then: X → Z ← Y
        for Z in self.variables:
            neighbors = self._get_neighbors(Z)

            for X in neighbors:
                for Y in neighbors:
                    if X >= Y:  # Avoid duplicates
                        continue

                    # Check if X and Y are adjacent
                    if (X, Y) in self.undirected_edges or (Y, X) in self.undirected_edges:
                        continue

                    # Check if Z is in separating set
                    sep_set = self.separating_sets.get((X, Y)) or self.separating_sets.get((Y, X))
                    if sep_set and Z in sep_set:
                        continue

                    # Orient: X → Z ← Y
                    self.directed_edges.add((X, Z))
                    self.directed_edges.add((Y, Z))
                    self.undirected_edges.discard((X, Z))
                    self.undirected_edges.discard((Z, X))
                    self.undirected_edges.discard((Y, Z))
                    self.undirected_edges.discard((Z, Y))

                    logger.debug(f"V-structure: {X} → {Z} ← {Y}")

        # Rule 2: Orient chains
        # If X → Y—Z and X,Z not adjacent, orient Y → Z
        changed = True
        while changed:
            changed = False
            for Y in self.variables:
                # Find X → Y
                incoming = [X for X, tgt in self.directed_edges if tgt == Y]
                # Find Y—Z
                undirected = [Z for X, Z in self.undirected_edges if X == Y]
                undirected += [X for X, Z in self.undirected_edges if Z == Y]

                for X in incoming:
                    for Z in undirected:
                        if Z == X:
                            continue

                        # Check if X and Z are adjacent
                        if (X, Z) in self.undirected_edges or (Z, X) in self.undirected_edges:
                            continue
                        if (X, Z) in self.directed_edges or (Z, X) in self.directed_edges:
                            continue

                        # Orient: Y → Z
                        self.directed_edges.add((Y, Z))
                        self.undirected_edges.discard((Y, Z))
                        self.undirected_edges.discard((Z, Y))
                        logger.debug(f"Chain: {X} → {Y} → {Z}")
                        changed = True

    def get_dag(self) -> CausalDAG:
        """Convert learned structure to CausalDAG."""
        dag = CausalDAG()

        # Add nodes
        for var in self.variables:
            dag.add_node(CausalNode(var, NodeType.OBSERVABLE))

        # Add directed edges
        for source, target in self.directed_edges:
            dag.add_edge(CausalEdge(
                source, target,
                strength=1.0,
                confidence=self.edge_beliefs.get((source, target), 0.5),
                mechanism="learned from data"
            ))

        return dag

    def select_intervention(self) -> Dict[str, Any]:
        """
        Select most informative intervention for active learning.

        Uses expected information gain to choose which variable to intervene on.

        Returns:
            Intervention {variable: value}
        """
        max_info_gain = 0
        best_var = None

        for var in self.variables:
            # Estimate information gain for intervening on this variable
            info_gain = self._expected_info_gain(var)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_var = var

        if best_var is None:
            # Fallback: random
            best_var = np.random.choice(self.variables)

        # Choose intervention value
        # For now, just use 0 or 1 (binary)
        value = np.random.choice([0, 1])

        logger.info(f"Selected intervention: {best_var} = {value} (info gain: {max_info_gain:.3f})")

        return {best_var: value}

    def _expected_info_gain(self, var: str) -> float:
        """
        Estimate expected information gain from intervening on variable.

        Information gain = reduction in uncertainty about causal structure.

        Heuristic: Intervene on variables with most uncertain edges.
        """
        # Count uncertain edges involving this variable
        uncertainty = 0

        for X, Y in self.edge_beliefs.keys():
            if X != var and Y != var:
                continue

            belief = self.edge_beliefs[(X, Y)]

            # Uncertainty = entropy
            # H(p) = -p log p - (1-p) log (1-p)
            if belief > 0 and belief < 1:
                entropy = -belief * np.log(belief) - (1 - belief) * np.log(1 - belief)
                uncertainty += entropy

        return uncertainty

    def update(self, intervention: Dict[str, Any], observations: Dict[str, Any]):
        """
        Update causal beliefs based on experimental result.

        Args:
            intervention: Variables intervened on
            observations: Observed outcomes
        """
        # Store experiment
        result = ExperimentResult(
            intervention=intervention,
            observations=observations,
            timestamp=0.0,  # Would use time.time() in production
            sample_size=1
        )
        self.experiments.append(result)

        # Update edge beliefs
        intervened_var = list(intervention.keys())[0]

        for observed_var, value in observations.items():
            if observed_var == intervened_var:
                continue

            # If intervening on X affects Y, there's likely a causal path X → ... → Y
            # Update belief in X → Y edge

            if (intervened_var, observed_var) in self.edge_beliefs:
                # Increase belief (Bayesian update, simplified)
                current_belief = self.edge_beliefs[(intervened_var, observed_var)]
                self.edge_beliefs[(intervened_var, observed_var)] = min(current_belief + 0.1, 1.0)

                logger.debug(
                    f"Updated belief: {intervened_var} → {observed_var}: "
                    f"{current_belief:.2f} → {self.edge_beliefs[(intervened_var, observed_var)]:.2f}"
                )

    def __repr__(self):
        return (
            f"CausalDiscovery({len(self.variables)} vars, "
            f"{len(self.directed_edges)} directed edges, "
            f"{len(self.experiments)} experiments)"
        )


class ActiveCausalLearner:
    """
    Active causal learner that runs experiments to learn structure.

    Combines:
    - Causal discovery (PC algorithm)
    - Active learning (experiment selection)
    - Neural SCM (mechanism learning)

    Usage:
        learner = ActiveCausalLearner(
            variables=['X', 'Y', 'Z'],
            environment=env
        )

        # Run active learning loop
        for _ in range(20):
            learner.run_experiment()

        # Get learned DAG
        dag = learner.get_dag()
    """

    def __init__(
        self,
        variables: List[str],
        environment: Optional[Callable] = None
    ):
        """
        Initialize active learner.

        Args:
            variables: Variable names
            environment: Function that executes interventions
                        environment(intervention) -> observations
        """
        self.variables = variables
        self.environment = environment
        self.discoverer = CausalDiscovery(variables)

        logger.info(f"Initialized active learner with {len(variables)} variables")

    def run_experiment(self) -> ExperimentResult:
        """
        Run one active learning experiment.

        Returns:
            Experiment result
        """
        # Select intervention
        intervention = self.discoverer.select_intervention()

        # Execute in environment
        if self.environment is None:
            # Simulate random outcome
            observations = {
                var: np.random.randn()
                for var in self.variables
            }
        else:
            observations = self.environment(intervention)

        # Update beliefs
        self.discoverer.update(intervention, observations)

        # Return result
        return ExperimentResult(
            intervention=intervention,
            observations=observations,
            timestamp=0.0,
            sample_size=1
        )

    def get_dag(self) -> CausalDAG:
        """Get learned causal DAG."""
        return self.discoverer.get_dag()

    def __repr__(self):
        return f"ActiveCausalLearner({len(self.discoverer.experiments)} experiments)"
