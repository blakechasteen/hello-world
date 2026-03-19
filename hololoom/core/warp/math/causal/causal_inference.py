"""
Causal Inference - Do-Calculus and Structural Causal Models
============================================================

Mathematical foundations for causal reasoning and intervention analysis.

Classes:
    StructuralCausalModel: Complete SCM with equations and noise
    CausalGraph: DAG representation with d-separation
    CausalQuery: Query specification for causal effects

Key Concepts:
    do(X=x): Intervention that sets X to x (removes incoming edges)

    Do-Calculus Rules (Pearl):
        Rule 1: P(y|do(x),z,w) = P(y|do(x),w) if (Y ⊥ Z | X,W)_G_X̄
        Rule 2: P(y|do(x),do(z),w) = P(y|do(x),z,w) if (Y ⊥ Z | X,W)_G_X̄Z_
        Rule 3: P(y|do(x),do(z),w) = P(y|do(x),w) if (Y ⊥ Z | X,W)_G_X̄Z(W)

    Identifiability: Can we compute P(y|do(x)) from observational data?

Applications:
    - Alignment: Understanding AI decision causality
    - Reasoning: Counterfactual analysis
    - Learning: Causal feature selection
    - Interpretability: Why did the model do X?

Author: Claude Code
Date: December 2025 (Math Module Expansion)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class CausalQueryType(Enum):
    """Types of causal queries."""
    OBSERVATIONAL = "observational"    # P(Y|X)
    INTERVENTIONAL = "interventional"  # P(Y|do(X))
    COUNTERFACTUAL = "counterfactual"  # P(Y_x | X=x', Y=y')


@dataclass
class CausalQuery:
    """
    Specification of a causal query.

    Attributes:
        outcome: Target variable(s)
        treatment: Treatment/intervention variable(s)
        query_type: Type of causal query
        conditioning: Variables to condition on
        intervention_values: Values for do() operation
    """
    outcome: list[str]
    treatment: list[str]
    query_type: CausalQueryType = CausalQueryType.INTERVENTIONAL
    conditioning: list[str] = field(default_factory=list)
    intervention_values: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.query_type == CausalQueryType.INTERVENTIONAL:
            treatment_str = ", ".join(f"do({t})" for t in self.treatment)
            return f"P({', '.join(self.outcome)} | {treatment_str})"
        elif self.query_type == CausalQueryType.COUNTERFACTUAL:
            return f"P({self.outcome}_x | evidence)"
        else:
            return f"P({', '.join(self.outcome)} | {', '.join(self.treatment)})"


@dataclass
class IdentificationResult:
    """Result of causal effect identification."""
    identifiable: bool
    estimand: str | None = None  # Statistical estimand if identifiable
    adjustment_set: set[str] | None = None
    formula: str | None = None
    explanation: str = ""


class CausalGraph:
    """
    Directed Acyclic Graph for causal structure.

    Implements d-separation, ancestor/descendant queries,
    and graph manipulation for do-calculus.
    """

    def __init__(self):
        """Initialize empty causal graph."""
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            # Fallback: dict of adjacency lists
            self._parents: dict[str, set[str]] = {}
            self._children: dict[str, set[str]] = {}

    def add_edge(self, parent: str, child: str) -> None:
        """Add causal edge: parent → child."""
        if HAS_NETWORKX:
            self.graph.add_edge(parent, child)
        else:
            if child not in self._parents:
                self._parents[child] = set()
            if parent not in self._children:
                self._children[parent] = set()
            if parent not in self._parents:
                self._parents[parent] = set()
            if child not in self._children:
                self._children[child] = set()

            self._parents[child].add(parent)
            self._children[parent].add(child)

    def add_edges(self, edges: list[tuple[str, str]]) -> None:
        """Add multiple edges."""
        for parent, child in edges:
            self.add_edge(parent, child)

    @property
    def nodes(self) -> set[str]:
        """Get all nodes in graph."""
        if HAS_NETWORKX:
            return set(self.graph.nodes())
        else:
            return set(self._parents.keys()) | set(self._children.keys())

    def parents(self, node: str) -> set[str]:
        """Get direct parents of node."""
        if HAS_NETWORKX:
            return set(self.graph.predecessors(node))
        else:
            return self._parents.get(node, set())

    def children(self, node: str) -> set[str]:
        """Get direct children of node."""
        if HAS_NETWORKX:
            return set(self.graph.successors(node))
        else:
            return self._children.get(node, set())

    def ancestors(self, node: str) -> set[str]:
        """Get all ancestors of node."""
        if HAS_NETWORKX:
            return nx.ancestors(self.graph, node)
        else:
            result = set()
            queue = list(self.parents(node))
            while queue:
                current = queue.pop(0)
                if current not in result:
                    result.add(current)
                    queue.extend(self.parents(current))
            return result

    def descendants(self, node: str) -> set[str]:
        """Get all descendants of node."""
        if HAS_NETWORKX:
            return nx.descendants(self.graph, node)
        else:
            result = set()
            queue = list(self.children(node))
            while queue:
                current = queue.pop(0)
                if current not in result:
                    result.add(current)
                    queue.extend(self.children(current))
            return result

    def do(self, intervention_vars: set[str]) -> 'CausalGraph':
        """
        Create mutilated graph for do() operation.

        Removes all incoming edges to intervention variables.

        Args:
            intervention_vars: Variables being intervened on

        Returns:
            New CausalGraph with edges removed
        """
        new_graph = CausalGraph()

        for node in self.nodes:
            for child in self.children(node):
                # Keep edge unless child is being intervened
                if child not in intervention_vars:
                    new_graph.add_edge(node, child)

        return new_graph

    def d_separated(
        self,
        x: set[str],
        y: set[str],
        z: set[str]
    ) -> bool:
        """
        Check if X and Y are d-separated given Z.

        Uses Bayes-Ball algorithm.

        Args:
            x: Source variables
            y: Target variables
            z: Conditioning variables

        Returns:
            True if X ⊥ Y | Z in the graph
        """
        # Implement Bayes-Ball algorithm for d-separation
        # X and Y are d-separated given Z if no path connects X to Y
        # where the path is "active" (not blocked by Z)

        # Build set of ancestors of Z (needed for collider activation)
        z_ancestors = set()
        for z_node in z:
            if z_node in self.nodes:
                z_ancestors.update(self.ancestors(z_node))
        z_ancestors.update(z)

        # BFS from X, tracking direction of travel
        # State: (node, coming_from_child)
        visited = set()
        queue = [(x_node, False) for x_node in x if x_node in self.nodes]

        while queue:
            node, from_child = queue.pop(0)
            state = (node, from_child)

            if state in visited:
                continue
            visited.add(state)

            # Check if we reached Y
            if node in y:
                return False  # Found active path, NOT d-separated

            is_conditioned = node in z

            # Determine what directions we can travel
            if from_child:
                # Came from child (traveling upstream)
                if not is_conditioned:
                    # Can go to parents (chain/fork continuation)
                    for parent in self.parents(node):
                        queue.append((parent, False))
                    # Can go to other children (fork)
                    for child in self.children(node):
                        queue.append((child, True))
            else:
                # Came from parent (traveling downstream)
                if not is_conditioned:
                    # Can continue to children (chain)
                    for child in self.children(node):
                        queue.append((child, True))
                    # Can go to parents (fork structure)
                    for parent in self.parents(node):
                        queue.append((parent, False))

            # Collider case: node has multiple parents
            # Collider is activated if node or descendant is in Z
            if node in z_ancestors:
                # Collider is activated, can traverse
                for parent in self.parents(node):
                    queue.append((parent, False))

        return True  # No active path found, X and Y are d-separated

    def is_valid_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        adjustment: set[str]
    ) -> bool:
        """
        Check if adjustment set satisfies back-door criterion.

        Back-door criterion:
            1. No node in Z is a descendant of X
            2. Z blocks all back-door paths from X to Y
        """
        # Condition 1: No descendants of X in Z
        desc_x = self.descendants(treatment)
        if adjustment & desc_x:
            return False

        # Condition 2: Z blocks all back-door paths
        # Create graph with outgoing edges FROM treatment removed
        # This isolates back-door paths (paths into X)
        backdoor_graph = CausalGraph()
        for node in self.nodes:
            for child in self.children(node):
                # Keep edge unless it's an outgoing edge from treatment
                if node != treatment:
                    backdoor_graph.add_edge(node, child)

        # Check if adjustment blocks all paths in backdoor graph
        return backdoor_graph.d_separated({treatment}, {outcome}, adjustment)

    def find_adjustment_set(
        self,
        treatment: str,
        outcome: str
    ) -> set[str] | None:
        """
        Find a valid adjustment set using back-door criterion.

        Returns:
            Valid adjustment set or None if not identifiable
        """
        all_vars = self.nodes - {treatment, outcome}
        desc_x = self.descendants(treatment)

        # Candidate variables (not descendants of treatment)
        candidates = all_vars - desc_x

        # Try parents of treatment first (usually sufficient)
        parents_x = self.parents(treatment)
        if parents_x and self.is_valid_adjustment_set(treatment, outcome, parents_x):
            return parents_x

        # Try all candidates
        if self.is_valid_adjustment_set(treatment, outcome, candidates):
            return candidates

        # Try subsets (expensive for large graphs)
        from itertools import combinations

        for size in range(len(candidates) + 1):
            for subset in combinations(candidates, size):
                subset_set = set(subset)
                if self.is_valid_adjustment_set(treatment, outcome, subset_set):
                    return subset_set

        return None


@dataclass
class StructuralEquation:
    """
    A single structural equation: Y = f(parents, noise).

    Attributes:
        variable: Target variable
        parents: Parent variables
        function: Deterministic function f(parents)
        noise_dist: Noise distribution (callable returning sample)
    """
    variable: str
    parents: list[str]
    function: Callable[[dict[str, float]], float]
    noise_dist: Callable[[], float] = field(
        default_factory=lambda: lambda: np.random.normal(0, 0.1)
    )

    def evaluate(
        self,
        parent_values: dict[str, float],
        noise: float | None = None
    ) -> float:
        """
        Evaluate equation given parent values.

        Args:
            parent_values: Dict of parent variable values
            noise: Noise term (sampled if not provided)

        Returns:
            Variable value
        """
        deterministic = self.function(parent_values)
        noise_term = noise if noise is not None else self.noise_dist()
        return deterministic + noise_term


class StructuralCausalModel:
    """
    Complete Structural Causal Model (SCM).

    Components:
        - U: Exogenous (noise) variables
        - V: Endogenous (observed) variables
        - F: Structural equations V_i = f_i(Pa(V_i), U_i)
        - P(U): Distribution over noise

    Example (simple SCM):
        X = U_X
        Y = 2*X + U_Y
        Z = X + Y + U_Z

    Supports:
        - Forward sampling
        - Interventions do(X=x)
        - Counterfactual reasoning
    """

    def __init__(self):
        """Initialize empty SCM."""
        self.equations: dict[str, StructuralEquation] = {}
        self.graph = CausalGraph()
        self._topological_order: list[str] | None = None

    def add_equation(
        self,
        variable: str,
        parents: list[str],
        function: Callable[[dict[str, float]], float],
        noise_dist: Callable[[], float] | None = None
    ) -> None:
        """
        Add structural equation for a variable.

        Args:
            variable: Target variable name
            parents: List of parent variable names
            function: Deterministic function of parents
            noise_dist: Noise distribution (default: N(0, 0.1))
        """
        if noise_dist is None:
            noise_dist = lambda: np.random.normal(0, 0.1)

        self.equations[variable] = StructuralEquation(
            variable=variable,
            parents=parents,
            function=function,
            noise_dist=noise_dist
        )

        # Update causal graph
        for parent in parents:
            self.graph.add_edge(parent, variable)

        # Invalidate cached order
        self._topological_order = None

    @property
    def topological_order(self) -> list[str]:
        """Get variables in topological order."""
        if self._topological_order is None:
            self._topological_order = self._compute_topological_order()
        return self._topological_order

    def _compute_topological_order(self) -> list[str]:
        """Compute topological ordering of variables."""
        if HAS_NETWORKX:
            return list(nx.topological_sort(self.graph.graph))
        else:
            # Kahn's algorithm
            in_degree = {v: len(self.graph.parents(v)) for v in self.equations}
            queue = [v for v, d in in_degree.items() if d == 0]
            order = []

            while queue:
                v = queue.pop(0)
                order.append(v)
                for child in self.graph.children(v):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

            return order

    def sample(
        self,
        n_samples: int = 1,
        interventions: dict[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Sample from the SCM (possibly under intervention).

        Args:
            n_samples: Number of samples
            interventions: do(X=x) interventions (overrides equations)

        Returns:
            Dict mapping variable names to sample arrays
        """
        interventions = interventions or {}

        samples = {v: np.zeros(n_samples) for v in self.equations}

        for i in range(n_samples):
            values = {}

            for var in self.topological_order:
                if var in interventions:
                    # Intervention: set to fixed value
                    values[var] = interventions[var]
                else:
                    # Normal: evaluate equation
                    eq = self.equations[var]
                    parent_values = {p: values[p] for p in eq.parents}
                    values[var] = eq.evaluate(parent_values)

                samples[var][i] = values[var]

        return samples

    def intervene(
        self,
        interventions: dict[str, float],
        n_samples: int = 1000
    ) -> dict[str, np.ndarray]:
        """
        Perform intervention do(X=x) and sample.

        Args:
            interventions: Variable -> value mapping
            n_samples: Number of samples

        Returns:
            Samples from interventional distribution
        """
        return self.sample(n_samples, interventions)

    def counterfactual(
        self,
        evidence: dict[str, float],
        intervention: dict[str, float],
        query_var: str
    ) -> float:
        """
        Compute counterfactual: Y_x given evidence.

        "What would Y have been if we had done X=x,
         given that we observed evidence?"

        Steps:
            1. Abduction: Infer noise given evidence
            2. Intervention: Apply do(X=x)
            3. Prediction: Compute Y under intervention

        Args:
            evidence: Observed values
            intervention: Counterfactual intervention
            query_var: Variable to query

        Returns:
            Counterfactual value of query_var
        """
        # Step 1: Abduction (infer noise terms)
        noise_terms = self._abduct_noise(evidence)

        # Step 2 & 3: Intervene and compute
        values = {}

        for var in self.topological_order:
            if var in intervention:
                values[var] = intervention[var]
            else:
                eq = self.equations[var]
                parent_values = {p: values.get(p, evidence.get(p, 0.0))
                               for p in eq.parents}
                # Use inferred noise
                noise = noise_terms.get(var, 0.0)
                values[var] = eq.evaluate(parent_values, noise=noise)

        return values.get(query_var, 0.0)

    def _abduct_noise(self, evidence: dict[str, float]) -> dict[str, float]:
        """
        Infer noise terms given evidence (simplified).

        Full abduction requires solving inverse problem.
        This is a simplified version that assumes linear equations.
        """
        noise_terms = {}

        for var in self.topological_order:
            if var in evidence:
                eq = self.equations[var]
                parent_values = {
                    p: evidence.get(p, 0.0) for p in eq.parents
                }
                deterministic = eq.function(parent_values)
                noise_terms[var] = evidence[var] - deterministic

        return noise_terms


# Convenience functions

def do_intervention(
    scm: StructuralCausalModel,
    intervention: dict[str, float],
    n_samples: int = 1000
) -> dict[str, np.ndarray]:
    """Perform do() intervention on SCM."""
    return scm.intervene(intervention, n_samples)


def counterfactual(
    scm: StructuralCausalModel,
    evidence: dict[str, float],
    intervention: dict[str, float],
    query_var: str
) -> float:
    """Compute counterfactual query."""
    return scm.counterfactual(evidence, intervention, query_var)


def identify_effect(
    graph: CausalGraph,
    treatment: str,
    outcome: str
) -> IdentificationResult:
    """
    Attempt to identify causal effect P(Y|do(X)).

    Uses back-door criterion for identification.

    Args:
        graph: Causal DAG
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        IdentificationResult with estimand if identifiable
    """
    adjustment_set = graph.find_adjustment_set(treatment, outcome)

    if adjustment_set is not None:
        if adjustment_set:
            z_str = ", ".join(adjustment_set)
            estimand = f"∑_z P({outcome}|{treatment}, {z_str}) P({z_str})"
            formula = f"E[{outcome}|{treatment}, Z={z_str}]"
        else:
            estimand = f"P({outcome}|{treatment})"
            formula = f"E[{outcome}|{treatment}]"

        return IdentificationResult(
            identifiable=True,
            estimand=estimand,
            adjustment_set=adjustment_set,
            formula=formula,
            explanation="Identifiable via back-door adjustment"
        )
    else:
        return IdentificationResult(
            identifiable=False,
            explanation="No valid adjustment set found (may require front-door or IV)"
        )


def is_identifiable(
    graph: CausalGraph,
    treatment: str,
    outcome: str
) -> bool:
    """Check if causal effect is identifiable."""
    result = identify_effect(graph, treatment, outcome)
    return result.identifiable


def find_adjustment_set(
    graph: CausalGraph,
    treatment: str,
    outcome: str
) -> set[str] | None:
    """Find valid adjustment set for back-door criterion."""
    return graph.find_adjustment_set(treatment, outcome)


# HoloLoom Integration
class CausalReasoner:
    """
    Causal reasoning for HoloLoom.

    Applications:
        - Alignment: Why did the model choose action X?
        - Debugging: What would have happened if...?
        - Feature selection: Which features causally affect outcome?
    """

    def __init__(self, scm: StructuralCausalModel | None = None):
        """
        Args:
            scm: Structural causal model (or build one later)
        """
        self.scm = scm or StructuralCausalModel()

    def add_causal_mechanism(
        self,
        effect: str,
        causes: list[str],
        mechanism: Callable[[dict[str, float]], float]
    ) -> None:
        """
        Add causal mechanism: effect = f(causes).

        Args:
            effect: Effect variable name
            causes: List of cause variable names
            mechanism: Function computing effect from causes
        """
        self.scm.add_equation(effect, causes, mechanism)

    def what_if(
        self,
        intervention: dict[str, float],
        query: str,
        n_samples: int = 1000
    ) -> dict[str, float]:
        """
        Answer "what if" question via intervention.

        Args:
            intervention: do(X=x) settings
            query: Variable to query (or "all")
            n_samples: Samples for estimation

        Returns:
            Dict with mean and std of query variable(s)
        """
        samples = self.scm.intervene(intervention, n_samples)

        if query == "all":
            return {
                var: {"mean": np.mean(vals), "std": np.std(vals)}
                for var, vals in samples.items()
            }
        else:
            vals = samples.get(query, np.array([0.0]))
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    def would_have_been(
        self,
        observed: dict[str, float],
        counterfactual_action: dict[str, float],
        query: str
    ) -> float:
        """
        Answer counterfactual: "What would query have been if..."

        Args:
            observed: What was actually observed
            counterfactual_action: What we hypothetically did instead
            query: Variable to query

        Returns:
            Counterfactual value
        """
        return self.scm.counterfactual(observed, counterfactual_action, query)


class CausalDiscovery:
    """
    Discover causal structure from data.

    Methods (placeholders for extension):
        - PC algorithm
        - FCI algorithm
        - GES (Greedy Equivalence Search)
        - LiNGAM (for linear non-Gaussian models)
    """

    @staticmethod
    def from_correlation(
        data: np.ndarray,
        var_names: list[str],
        threshold: float = 0.3
    ) -> CausalGraph:
        """
        Simple structure learning from correlations.

        This is NOT proper causal discovery (correlation ≠ causation),
        but provides a starting point.

        For real causal discovery, use dedicated libraries like:
        - causal-learn
        - tigramite
        - lingam
        """
        n_vars = len(var_names)
        graph = CausalGraph()

        # Compute correlation matrix
        corr = np.corrcoef(data.T)

        # Add edges for high correlations
        # Direction is arbitrary (this is the limitation)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(corr[i, j]) > threshold:
                    # Arbitrary direction (would need more info)
                    graph.add_edge(var_names[i], var_names[j])

        return graph


if __name__ == "__main__":
    print("Causal Inference Module Demo")
    print("=" * 50)

    # Create a simple SCM:
    # Z → X → Y
    # Z → Y (confounder)

    scm = StructuralCausalModel()

    # Z = noise (exogenous)
    scm.add_equation("Z", [], lambda _: 0.0,
                     noise_dist=lambda: np.random.normal(0, 1))

    # X = 0.5*Z + noise
    scm.add_equation("X", ["Z"], lambda p: 0.5 * p["Z"],
                     noise_dist=lambda: np.random.normal(0, 0.5))

    # Y = 2*X + Z + noise
    scm.add_equation("Y", ["X", "Z"], lambda p: 2 * p["X"] + p["Z"],
                     noise_dist=lambda: np.random.normal(0, 0.5))

    print("\nSCM Structure: Z → X → Y, Z → Y (confounder)")

    # Sample observationally
    obs_samples = scm.sample(1000)
    print(f"\nObservational correlation X-Y: {np.corrcoef(obs_samples['X'], obs_samples['Y'])[0,1]:.3f}")

    # Interventional: do(X=2)
    int_samples = scm.intervene({"X": 2.0}, 1000)
    print(f"E[Y | do(X=2)]: {np.mean(int_samples['Y']):.3f} (true: ~4.0)")

    # Counterfactual
    observed = {"Z": 1.0, "X": 1.5, "Y": 4.0}
    cf_value = scm.counterfactual(observed, {"X": 0.0}, "Y")
    print(f"\nCounterfactual: If X had been 0 instead of 1.5, Y would have been {cf_value:.3f}")

    # Identification
    print("\n--- Effect Identification ---")
    result = identify_effect(scm.graph, "X", "Y")
    print(f"P(Y|do(X)) identifiable: {result.identifiable}")
    print(f"Adjustment set: {result.adjustment_set}")
    print(f"Estimand: {result.estimand}")

    # HoloLoom integration
    print("\n--- CausalReasoner (HoloLoom) ---")
    reasoner = CausalReasoner(scm)

    what_if_result = reasoner.what_if({"X": 5.0}, "Y")
    print(f"What if X=5? E[Y] = {what_if_result['mean']:.2f} ± {what_if_result['std']:.2f}")

    print("\n" + "=" * 50)
    print("Causal inference module ready!")
