# HoloLoom Math Module

**Status**: Production Ready (December 2025)
**Total Modules**: 52 Python files across 9 subdirectories
**Test Coverage**: 125 comprehensive tests (39 new + 86 existing)

Advanced mathematical foundations for HoloLoom's decision-making and learning systems.

## Module Organization

```
HoloLoom/warp/math/
├── advanced/         # Differential privacy, topological data analysis
├── algebra/          # Abstract algebraic structures
├── analysis/         # Real, complex, functional analysis
├── causal/           # Causal inference, do-calculus, SCMs
├── decision/         # Game theory, mechanism design, regret minimization
├── extensions/       # Multivariable calculus, topology
├── geometry/         # Differential geometry, manifolds
├── graph/            # Spectral clustering, graph algorithms
├── information/      # Information theory, entropy, mutual information
└── logic/            # Formal logic, type theory
```

## Key Modules

### Decision Theory (`decision/`)

#### game_theory.py
Nash equilibria, mechanism design, and auction theory.

```python
from HoloLoom.warp.math.decision.game_theory import (
    NormalFormGame, NashEquilibrium, MechanismDesign,
    AuctionTheory, CooperativeGame, EvolutionaryGame
)

# Create Prisoner's Dilemma
game = NormalFormGame.prisoners_dilemma()

# Find pure Nash equilibria
equilibria = NashEquilibrium.find_pure(game)  # [(1, 1)] = Defect-Defect

# Find mixed Nash equilibrium
strategies = NashEquilibrium.find_mixed_2player(game)

# Check auction truthfulness
is_truthful, _ = MechanismDesign.is_truthful(
    MechanismDesign.vcg_auction,
    test_cases=[np.array([10.0, 8.0, 5.0])]
)  # True (VCG is truthful)

# Compute Shapley values
coop_game = CooperativeGame.glove_game(n_left=1, n_right=2)
shapley = coop_game.shapley_value()
```

**Features**:
- Pure Nash equilibrium finding (enumeration)
- Mixed Nash equilibrium (support enumeration algorithm)
- IESDS (Iterated Elimination of Strictly Dominated Strategies)
- VCG auction (incentive-compatible mechanism design)
- Cooperative games (Shapley value, core)
- Evolutionary games (replicator dynamics, ESS)

#### correlated_equilibria.py
Linear programming approach to correlated equilibria (generalization of Nash).

```python
from HoloLoom.warp.math.decision.correlated_equilibria import (
    CorrelatedEquilibrium, find_correlated_equilibrium,
    find_social_optimal, find_egalitarian, CHICKEN_GAME
)

# Chicken game example
ce = find_correlated_equilibrium(CHICKEN_GAME)
print(f"Expected utilities: {ce.expected_utilities}")

# Social welfare maximizing CE
social_ce = find_social_optimal(CHICKEN_GAME)

# Egalitarian CE (maximize minimum utility)
egal_ce = find_egalitarian(CHICKEN_GAME)
```

**Features**:
- LP-based correlated equilibrium computation
- Social welfare and egalitarian optimization
- Standard game matrices (Chicken, Prisoner's Dilemma, Battle of Sexes)

#### regret_minimization.py
Online learning algorithms for repeated games.

```python
from HoloLoom.warp.math.decision.regret_minimization import (
    RegretMinimizer, ExternalRegret, InternalRegret,
    regret_matching, regret_matching_plus
)

# Regret Matching+
rm = regret_matching_plus(n_actions=3)
for t in range(1000):
    action = rm.get_action()
    loss = get_loss(action)
    rm.update(loss)

# External regret bound: O(√T)
print(f"Average regret: {rm.average_regret}")
```

**Features**:
- Regret Matching and Regret Matching+ algorithms
- External and internal (swap) regret tracking
- Convergence to coarse correlated equilibrium

#### information_theory.py
Shannon information theory for decision-making.

```python
from HoloLoom.warp.math.decision.information_theory import (
    entropy, conditional_entropy, mutual_information,
    kl_divergence, jensen_shannon_divergence
)

# Entropy H(X)
p = np.array([0.5, 0.25, 0.25])
H = entropy(p)  # 1.5 bits

# Mutual information I(X;Y)
joint_pxy = np.array([[0.1, 0.2], [0.3, 0.4]])
I = mutual_information(joint_pxy)

# KL divergence D_KL(P || Q)
q = np.array([0.33, 0.33, 0.34])
D = kl_divergence(p, q)
```

**Features**:
- Shannon entropy (bits/nats)
- Conditional entropy H(X|Y)
- Mutual information I(X;Y)
- KL and Jensen-Shannon divergence

### Extensions (`extensions/`)

#### multivariable_calculus.py
Vector calculus with central finite differences for numerical accuracy.

```python
from HoloLoom.warp.math.extensions.multivariable_calculus import (
    VectorField, ScalarField, GradientCurlDiv
)

# Create rotation field F = (-y, x, 0)
field = VectorField.rotation_z()

# Compute curl: curl(F) = (0, 0, 2)
curl = field.curl(np.array([1.0, 1.0, 0.0]))

# Create scalar field f = x² + y² + z²
f = ScalarField.distance_squared()

# Gradient: ∇f = 2(x, y, z)
grad = f.gradient(np.array([1.0, 2.0, 3.0]))

# Vector calculus identities
identities = GradientCurlDiv.get_identities()
# curl(∇f) = 0, div(curl F) = 0, etc.
```

**Features**:
- Gradient, divergence, curl, Laplacian
- Central finite differences (O(h²) accuracy)
- Line integrals and circulation
- Conservative field detection
- Vector calculus identities (Stokes, Green, Divergence theorems)

### Analysis (`analysis/`)

#### functional_analysis.py
Operator theory and function spaces.

```python
from HoloLoom.warp.math.analysis.functional_analysis import (
    BanachSpace, HilbertSpace, LinearOperator
)
```

#### probability_theory.py
Probability distributions and stochastic processes.

#### optimization.py
Convex optimization and gradient descent methods.

### Graph Theory (`graph/`)

#### spectral_clustering.py
Graph Laplacian-based clustering using spectral methods.

```python
from HoloLoom.warp.math.graph.spectral_clustering import (
    SpectralClustering, LaplacianType, SpectralEmbedding,
    compute_laplacian, fiedler_vector
)

# Create adjacency matrix
A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0.01, 0],
    [0, 0, 0.01, 0, 1],
    [0, 0, 0, 1, 0]
])

# Spectral clustering
sc = SpectralClustering(n_clusters=2, laplacian_type=LaplacianType.SYMMETRIC)
result = sc.fit(A)

print(f"Cluster labels: {result.labels}")
print(f"Inertia: {result.inertia:.4f}")

# Fiedler vector (second smallest eigenvector of Laplacian)
fiedler = fiedler_vector(A)
```

**Features**:
- Unnormalized, symmetric, and random-walk Laplacians
- K-means++ initialization for robust clustering
- Spectral embedding for graph visualization
- Fiedler vector for graph bisection

### Causal Inference (`causal/`)

#### causal_inference.py
Structural causal models and do-calculus.

```python
from HoloLoom.warp.math.causal.causal_inference import (
    CausalGraph, StructuralCausalModel,
    do_calculus, backdoor_adjustment, frontdoor_adjustment
)

# Build causal graph: X -> Y, Z -> X, Z -> Y
graph = CausalGraph()
graph.add_edge("Z", "X")
graph.add_edge("Z", "Y")
graph.add_edge("X", "Y")

# Check d-separation
separated = graph.d_separated({"X"}, {"Y"}, {"Z"})  # True

# Intervention: do(X = x)
mutilated = graph.do("X")  # Removes edges into X

# Valid adjustment set for backdoor criterion
valid = graph.is_valid_adjustment_set(
    treatment="X", outcome="Y", adjustment={"Z"}
)  # True
```

**Features**:
- Directed acyclic graphs (DAGs) for causal structure
- D-separation testing (Bayes-Ball algorithm)
- Interventions via do-calculus
- Backdoor and frontdoor adjustment criteria
- Counterfactual inference

### Advanced Methods (`advanced/`)

#### differential_privacy.py
Privacy-preserving computations with rigorous guarantees.

```python
from HoloLoom.warp.math.advanced.differential_privacy import (
    DifferentialPrivacy, PrivacyBudget,
    laplace_mechanism, gaussian_mechanism, exponential_mechanism
)

# Private mean with Laplace noise
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
sensitivity = 1.0  # max impact of one record
epsilon = 1.0

private_mean = laplace_mechanism(
    true_value=np.mean(data),
    sensitivity=sensitivity,
    epsilon=epsilon
)

# Compose privacy budgets
budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
budget.compose(epsilon=0.5)  # Sequential composition
print(f"Remaining: ε={budget.remaining_epsilon:.2f}")
```

**Features**:
- Laplace mechanism (ε-DP)
- Gaussian mechanism (ε,δ-DP)
- Exponential mechanism for discrete outputs
- Privacy budget tracking and composition
- Sensitivity computation

#### topological_analysis.py
Topological data analysis with persistent homology.

```python
from HoloLoom.warp.math.advanced.topological_analysis import (
    PersistentHomology, PersistenceDiagram, BettiNumbers,
    compute_persistence, bottleneck_distance
)

# Point cloud data
points = np.random.randn(100, 3)

# Compute persistent homology
ph = PersistentHomology()
diagram = ph.fit_transform(points, max_dimension=2)

print(f"H0 (connected components): {len(diagram.h0)}")
print(f"H1 (loops): {len(diagram.h1)}")
print(f"H2 (voids): {len(diagram.h2)}")

# Bottleneck distance between diagrams
distance = bottleneck_distance(diagram1, diagram2)
```

**Features**:
- Vietoris-Rips complex construction
- H0, H1, H2 persistence computation
- Bottleneck and Wasserstein distances
- Persistence landscapes
- Topological feature extraction for ML

### Algebra (`algebra/`)

#### abstract_algebra.py
Groups, rings, fields, and algebraic structures.

#### galois_theory.py
Field extensions and Galois groups.

### Geometry (`geometry/`)

Differential geometry, manifolds, and curvature.

### Logic (`logic/`)

Formal logic, type theory, and proof systems.

## Integration with HoloLoom

The math modules power HoloLoom's core systems:

| Math Module | HoloLoom System | Application |
|-------------|-----------------|-------------|
| `game_theory` | Thompson Sampling | Exploration/exploitation balance |
| `correlated_equilibria` | Multi-Agent Reasoning | Coordination between agents |
| `regret_minimization` | Adaptive Routing | Online learning for query routing |
| `information_theory` | Feature Selection | Mutual information for relevance |
| `spectral_clustering` | Memory Graph | Community detection in knowledge graphs |
| `causal_inference` | Agentic Reasoning | Causal explanation of decisions |
| `differential_privacy` | Alignment Framework | Privacy-preserving memory queries |
| `topological_analysis` | Embedding Space | Persistent features for representations |
| `probability_theory` | Policy Engine | Bayesian updates |
| `optimization` | Neural Networks | Gradient descent, Adam |
| `multivariable_calculus` | Warp Space | Tensor field operations |
| `functional_analysis` | Embedding Space | Hilbert space geometry |

## Testing

Run all math module tests:

```bash
# Core math tests (86 tests)
PYTHONPATH=. python -m pytest HoloLoom/tests/unit/test_game_theory_complete.py -v
PYTHONPATH=. python -m pytest HoloLoom/tests/unit/test_multivariable_calculus.py -v
PYTHONPATH=. python -m pytest HoloLoom/tests/unit/test_thompson_sampling_math.py -v

# Math expansion tests (39 tests)
PYTHONPATH=. python -m pytest HoloLoom/tests/unit/test_math_expansion.py -v
```

**Test Coverage** (125 tests total):

*Core Tests (86)*:
- `test_thompson_sampling_math.py`: 20 tests (Beta distributions, bandits)
- `test_game_theory_complete.py`: 33 tests (Nash, IESDS, auctions)
- `test_multivariable_calculus.py`: 33 tests (curl, gradient, identities)

*Math Expansion Tests (39)*:
- `test_math_expansion.py`: 39 tests across 7 new modules
  - Information Theory: entropy, mutual information, KL divergence (6 tests)
  - Correlated Equilibria: LP solutions, social welfare (4 tests)
  - Regret Minimization: Regret Matching+, convergence (5 tests)
  - Spectral Clustering: Laplacians, k-means++, embeddings (6 tests)
  - Causal Inference: d-separation, do-calculus, backdoor (7 tests)
  - Differential Privacy: Laplace/Gaussian mechanisms, composition (6 tests)
  - Topological Analysis: persistence, Betti numbers, distances (5 tests)

## Mathematical Foundations

### Nash Equilibrium
A strategy profile (s₁*, s₂*, ..., sₙ*) is a Nash equilibrium if:

```
∀i, sᵢ*: Uᵢ(sᵢ*, s₋ᵢ*) ≥ Uᵢ(sᵢ, s₋ᵢ*)
```

No player can unilaterally improve their payoff.

### VCG Mechanism (Vickrey-Clarke-Groves)
Payment for player i:

```
pᵢ = ∑ⱼ≠ᵢ vⱼ(ω*(θ₋ᵢ)) - ∑ⱼ≠ᵢ vⱼ(ω*(θ))
```

Truthful reporting is a dominant strategy.

### Vector Calculus Identities
```
curl(∇f) = 0           (gradient is irrotational)
div(curl F) = 0        (curl is solenoidal)
∇²f = div(∇f)          (Laplacian definition)
```

### Thompson Sampling
Prior: Beta(α, β)
Update: Success → α += 1, Failure → β += 1
Expected reward: E[X] = α / (α + β)

## Changelog

### December 2025 (Math Module Expansion)
- **Phase 4**: Added 7 new mathematical modules
  - `decision/information_theory.py`: Shannon entropy, mutual information, KL divergence
  - `decision/correlated_equilibria.py`: LP-based CE computation, social welfare
  - `decision/regret_minimization.py`: Regret Matching+, external/internal regret
  - `graph/spectral_clustering.py`: Graph Laplacians, k-means++, Fiedler vector
  - `causal/causal_inference.py`: D-separation (Bayes-Ball), do-calculus, backdoor criterion
  - `advanced/differential_privacy.py`: Laplace/Gaussian mechanisms, privacy composition
  - `advanced/topological_analysis.py`: Persistent homology, bottleneck distance

- **Phase 4**: Fixed 2 critical algorithm bugs
  - `is_valid_adjustment_set`: Now correctly checks backdoor criterion using graph with outgoing edges from treatment removed
  - `spectral_clustering`: Added k-means++ initialization for robust clustering

- **Test Coverage**: 39 new tests (all passing)
  - Comprehensive coverage of all 7 new modules
  - Mathematical correctness validated against known results

### December 2025 (Math Module Upgrade - Earlier)
- **Phase 1**: Fixed 5 critical bugs
  - NumPy 2.0 compatibility (`np.math.factorial` → `math.factorial`)
  - Central differences for curl (O(h) → O(h²))
  - Type hints (`any` → `Any`)
  - Division-by-zero guards
  - Strategy list order fix in `best_response()`

- **Phase 2**: Completed 3 core implementations
  - Mixed Nash equilibrium (support enumeration)
  - IESDS (dominated strategy elimination)
  - Truthfulness checking (incentive compatibility)

- **Phase 3**: Expanded test coverage (33% → 70%+)
  - 86 new comprehensive tests
  - All vector calculus identities validated
  - All game theory algorithms tested

## References

**Game Theory**:
- Nash, J. (1950). "Equilibrium points in n-person games"
- Aumann, R. (1974). "Subjectivity and correlation in randomized strategies" (Correlated equilibria)
- Hart, S. & Mas-Colell, A. (2000). "A Simple Adaptive Procedure Leading to Correlated Equilibrium" (Regret Matching)

**Mechanism Design**:
- Vickrey, W. (1961). "Counterspeculation, Auctions, and Competitive Sealed Tenders"
- Clarke, E. (1971). "Multipart pricing of public goods"
- Groves, T. (1973). "Incentives in Teams"
- Shapley, L. (1953). "A Value for n-Person Games"

**Information Theory**:
- Shannon, C. (1948). "A Mathematical Theory of Communication"
- Cover, T. & Thomas, J. (2006). "Elements of Information Theory"

**Graph Theory**:
- Ng, A., Jordan, M. & Weiss, Y. (2001). "On Spectral Clustering" (Spectral clustering)
- Fiedler, M. (1973). "Algebraic connectivity of graphs" (Fiedler vector)
- Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding"

**Causal Inference**:
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
- Pearl, J. (1995). "Causal diagrams for empirical research" (Backdoor criterion)
- Shachter, R. (1998). "Bayes-Ball: The Rational Pastime" (D-separation algorithm)

**Differential Privacy**:
- Dwork, C. & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy"
- Dwork, C. et al. (2006). "Calibrating Noise to Sensitivity in Private Data Analysis"

**Topological Data Analysis**:
- Edelsbrunner, H. & Harer, J. (2010). "Computational Topology"
- Carlsson, G. (2009). "Topology and Data"
- Cohen-Steiner, D. et al. (2007). "Stability of Persistence Diagrams" (Bottleneck distance)
