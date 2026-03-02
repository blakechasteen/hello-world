# Warp Math Quick Start Guide

Fast introduction to using HoloLoom's mathematical foundation.

---

## 🚀 Installation

No additional dependencies required beyond NumPy. All modules are pure Python implementations.

```bash
# Already included in HoloLoom
cd mythRL
python  # Ensure you can import
```

---

## 📦 Import Examples

### Analysis

```python
from HoloLoom.warp.math.analysis import (
    MetricSpace,           # Real analysis
    ComplexFunction,       # Complex analysis
    HilbertSpace,         # Functional analysis
    LebesgueMeasure,      # Measure theory
    FourierTransform,     # Fourier analysis
    BrownianMotion,       # Stochastic calculus
    RandomVariable,       # Probability theory
    RootFinder,           # Numerical analysis
)
```

### Algebra

```python
from HoloLoom.warp.math.algebra import (
    Group,                # Groups (cyclic, symmetric, dihedral)
    Ring,                 # Rings and ideals
    Field,                # Fields (finite, extensions)
    GaloisGroup,          # Galois theory
    Module,               # Module theory
    ChainComplex,         # Homological algebra
)
```

### Geometry & Physics

```python
from HoloLoom.warp.math.geometry import (
    SmoothManifold,       # Differential geometry
    RiemannianMetric,     # Riemannian geometry
    Geodesic,             # Geodesics on manifolds
    LagrangianMechanics,  # Classical mechanics
    HamiltonianMechanics, # Hamiltonian formulation
    SymplecticManifold,   # Symplectic geometry
)
```

### Decision & Information

```python
from HoloLoom.warp.math.decision import (
    Entropy,              # Shannon entropy
    MutualInformation,    # Information theory
    NormalFormGame,       # Game theory
    NashEquilibrium,      # Equilibrium solver
    NetworkFlows,         # Operations research
    DynamicProgramming,   # DP algorithms
)
```

### Logic & Foundations

```python
from HoloLoom.warp.math.logic import (
    PropositionalLogic,   # Boolean logic
    FirstOrderLogic,      # Predicate logic
    GodelTheorems,        # Incompleteness
    TuringMachine,        # Computability
    ComplexityClasses,    # P, NP, etc.
)
```

---

## 🎯 Common Use Cases

### 1. Feature Selection via Mutual Information

```python
from HoloLoom.warp.math.decision import MutualInformation
import numpy as np

# Your data
features = np.random.randn(1000, 10)  # 1000 samples, 10 features
target = (features[:, 0] + features[:, 3] > 0).astype(int)

# Compute MI for each feature
mi_scores = []
for i in range(features.shape[1]):
    mi = MutualInformation.from_samples(features[:, i], target)
    mi_scores.append(mi)

# Select top-k features
k = 5
top_features = np.argsort(mi_scores)[-k:]
print(f"Top {k} features: {top_features}")
```

### 2. Riemannian Optimization

```python
from HoloLoom.warp.math.geometry import RiemannianMetric, Geodesic
import numpy as np

# Define manifold (e.g., sphere)
metric = RiemannianMetric.sphere(radius=1.0)
geodesic = Geodesic(metric)

# Compute geodesic between two points
initial_point = np.array([np.pi/6, 0.0])
initial_velocity = np.array([0.0, 1.0])

path, velocities = geodesic.integrate(
    initial_point,
    initial_velocity,
    t_final=np.pi/2
)

print(f"Geodesic computed: {len(path)} points")
```

### 3. Multi-Agent Nash Equilibrium

```python
from HoloLoom.warp.math.decision import NormalFormGame, NashEquilibrium

# Define 2-player game (Prisoner's Dilemma)
game = NormalFormGame.prisoners_dilemma()

# Find Nash equilibria
equilibria = NashEquilibrium.find_pure(game)
print(f"Nash equilibria: {equilibria}")
# Output: [(1, 1)] - both defect

# Check if it's actually an equilibrium
is_nash = NashEquilibrium.verify_equilibrium(game, strategies)
```

### 4. Stochastic Differential Equations

```python
from HoloLoom.warp.math.analysis import SDESolver
import numpy as np

# Geometric Brownian motion: dS = μS dt + σS dW
mu, sigma = 0.05, 0.2  # Drift and volatility
S0 = 100.0

def drift(t, S):
    return mu * S

def diffusion(t, S):
    return sigma * S

# Simulate 100 paths
paths = SDESolver.euler_maruyama(
    drift=drift,
    diffusion=diffusion,
    X0=S0,
    T=1.0,
    n_steps=252
)

print(f"Final price: {paths[-1]:.2f}")
```

### 5. Network Flow Optimization

```python
from HoloLoom.warp.math.decision import NetworkFlows

# Create network
network = NetworkFlows(n_nodes=6)

# Add edges (source, sink, capacity)
network.add_edge(0, 1, capacity=10)
network.add_edge(0, 2, capacity=10)
network.add_edge(1, 3, capacity=4)
network.add_edge(1, 4, capacity=8)
network.add_edge(2, 4, capacity=9)
network.add_edge(3, 5, capacity=10)
network.add_edge(4, 5, capacity=10)

# Compute max flow
max_flow = network.max_flow(source=0, sink=5)
print(f"Maximum flow: {max_flow}")
```

### 6. Error Correction with Hamming Codes

```python
from HoloLoom.warp.math.decision import ErrorCorrection
import numpy as np

# Encode 4 bits to 7 bits
data = np.array([1, 0, 1, 1])
encoded = ErrorCorrection.hamming_7_4_encode(data)
print(f"Encoded: {encoded}")

# Introduce 1-bit error
received = encoded.copy()
received[2] ^= 1  # Flip bit 2

# Decode with error correction
decoded, error_corrected = ErrorCorrection.hamming_7_4_decode(received)
print(f"Decoded: {decoded}, Error corrected: {error_corrected}")
# Output: [1 0 1 1], True
```

### 7. Dynamic Programming (Knapsack)

```python
from HoloLoom.warp.math.decision import DynamicProgramming
import numpy as np

# Items: (value, weight)
values = np.array([60, 100, 120])
weights = np.array([10, 20, 30])
capacity = 50

# Solve 0-1 knapsack
max_value, items_selected = DynamicProgramming.knapsack_01(
    values, weights, capacity
)

print(f"Maximum value: {max_value}")
print(f"Items selected: {items_selected}")
# Output: Maximum value: 220, Items: [1, 2]
```

---

## 🧮 Mathematics Quick Reference

### Entropy

```python
from HoloLoom.warp.math.decision import Entropy

# Shannon entropy
probs = np.array([0.5, 0.3, 0.2])
h = Entropy.shannon(probs, base=2)  # In bits

# Cross-entropy
p = np.array([0.5, 0.5])
q = np.array([0.3, 0.7])
h_cross = Entropy.cross_entropy(p, q)

# From samples
samples = np.random.choice([0, 1, 2], size=1000, p=probs)
h_empirical = Entropy.from_samples(samples)
```

### Riemannian Metrics

```python
from HoloLoom.warp.math.geometry import RiemannianMetric

# Euclidean metric
euclidean = RiemannianMetric.euclidean(dim=3)

# Sphere metric (S²)
sphere = RiemannianMetric.sphere(radius=1.0)

# Hyperbolic metric (Poincaré disk)
hyperbolic = RiemannianMetric.hyperbolic(dim=2)

# Compute inner product
point = np.array([0.5, 0.0])
v = np.array([1.0, 0.0])
w = np.array([0.0, 1.0])
inner = hyperbolic.inner_product(point, v, w)
```

### Manifolds

```python
from HoloLoom.warp.math.geometry import SmoothManifold

# Circle S¹
S1 = SmoothManifold.circle()

# Sphere S²
S2 = SmoothManifold.sphere()

# Torus T² = S¹ × S¹
T2 = SmoothManifold.torus()
```

### Lagrangian Mechanics

```python
from HoloLoom.warp.math.geometry import LagrangianMechanics
import numpy as np

# Simple harmonic oscillator
system = LagrangianMechanics.simple_harmonic_oscillator(m=1.0, k=1.0)

# Evaluate action
q = np.array([1.0])
q_dot = np.array([0.0])
t = 0.0

L = system.L(q, q_dot, t)
print(f"Lagrangian: {L}")
```

### Game Theory

```python
from HoloLoom.warp.math.decision import (
    NormalFormGame,
    Strategy,
    NashEquilibrium
)

# Create game
game = NormalFormGame.matching_pennies()  # Zero-sum game

# Find pure Nash equilibria
pure_eq = NashEquilibrium.find_pure(game)
print(f"Pure equilibria: {pure_eq}")  # [] (none exist)

# Matching pennies has only mixed Nash: (0.5, 0.5) for both players
```

---

## 🎓 Advanced Topics

### Gödel's Incompleteness

```python
from HoloLoom.warp.math.logic import GodelTheorems

# First incompleteness theorem
print(GodelTheorems.first_incompleteness())
# Explains existence of true but unprovable statements

# Second incompleteness theorem
print(GodelTheorems.second_incompleteness())
# System cannot prove own consistency
```

### Halting Problem

```python
from HoloLoom.warp.math.logic import HaltingProblem

# Undecidability proof
print(HaltingProblem.undecidability_proof())
# Diagonalization argument

# Semi-decidability
print(HaltingProblem.semi_decidable())
```

### Turing Machines

```python
from HoloLoom.warp.math.logic import TuringMachine

# Binary increment TM
tm = TuringMachine.binary_increment()

# Run on input "101" (binary 5)
accepted, trace = tm.run("101")

# Check final tape
final_tape = ''.join(trace[-1].tape).strip('_')
print(f"Result: {final_tape}")  # "110" (binary 6)
```

### Complexity Classes

```python
from HoloLoom.warp.math.logic import ComplexityClasses

# P vs NP problem
print(ComplexityClasses.P_vs_NP())

# Class hierarchy
hierarchy = ComplexityClasses.complexity_hierarchy()
print(hierarchy["Relations"])
# P ⊆ NP ⊆ PSPACE ⊆ EXPTIME
```

---

## 📚 Module Organization

```
HoloLoom/warp/math/
├── analysis/          # Real, complex, functional, measure, fourier, stochastic
├── algebra/           # Groups, rings, fields, Galois, modules, homology
├── geometry/          # Differential, Riemannian, mathematical physics
├── decision/          # Information theory, game theory, operations research
└── logic/             # Mathematical logic, computability theory
```

---

## 🔗 Cross-Module Examples

### Combining Geometry + Physics

```python
from HoloLoom.warp.math.geometry import (
    RiemannianMetric,
    HamiltonianMechanics
)

# Define configuration space metric
metric = RiemannianMetric.sphere(radius=1.0)

# Define Hamiltonian dynamics
H = HamiltonianMechanics.simple_harmonic_oscillator(m=1.0, k=1.0)

# Initial conditions
q0 = np.array([1.0])
p0 = np.array([0.0])

# Integrate Hamilton's equations
q_traj, p_traj = H.integrate(q0, p0, t_final=2*np.pi)

# Compute energy conservation
E_initial = H.energy(q_traj[0], p_traj[0], 0.0)
E_final = H.energy(q_traj[-1], p_traj[-1], 2*np.pi)
print(f"Energy conserved: {np.abs(E_final - E_initial) < 1e-3}")
```

### Information Theory + Probability

```python
from HoloLoom.warp.math.analysis import RandomVariable
from HoloLoom.warp.math.decision import Entropy, MutualInformation

# Create random variables
samples = np.random.randn(1000)
rv = RandomVariable(samples)

# Compute entropy (continuous approximation)
h = Entropy.from_samples((samples * 10).astype(int), base=2)

# For two correlated variables
x = np.random.randn(1000)
y = x + 0.5 * np.random.randn(1000)

# Mutual information
mi = MutualInformation.from_samples(
    (x * 10).astype(int),
    (y * 10).astype(int)
)
print(f"Mutual information: {mi:.3f} bits")
```

---

## 🐛 Troubleshooting

### Import Errors

```python
# If you get "No module named 'HoloLoom'"
import sys
sys.path.insert(0, '/path/to/mythRL')  # Adjust to your path

# Then imports should work
from HoloLoom.warp.math.analysis import MetricSpace
```

### Numerical Issues

```python
# For numerical stability, some methods have tolerance parameters
from HoloLoom.warp.math.analysis import SequenceAnalyzer

sequence = [1/n for n in range(1, 101)]
is_convergent = SequenceAnalyzer.is_convergent(
    sequence,
    tolerance=1e-3  # Adjust as needed
)
```

---

## 📖 Further Reading

- **COMPLETE_WARP_MATH_FOUNDATION.md**: Comprehensive documentation
- **MATH_SPRINT_COMPLETE.md**: Session summary and achievements
- Individual module docstrings: Every class and method documented

---

## 🎯 Quick Tips

1. **Start Simple**: Try the examples above first
2. **Read Docstrings**: Every class/method has detailed docs
3. **Check Examples**: Each module has test cases at bottom
4. **Combine Modules**: Most powerful when used together
5. **Numerical Care**: Watch for edge cases (division by zero, etc.)

---

## 🚀 Next Steps

1. Explore individual modules in detail
2. Adapt examples to your use case
3. Combine mathematics with HoloLoom's neural components
4. Build applications on this foundation

---

**Happy Computing!** 🎉

---

*For questions or issues, refer to the comprehensive documentation in COMPLETE_WARP_MATH_FOUNDATION.md*
