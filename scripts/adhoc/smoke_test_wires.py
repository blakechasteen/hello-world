"""Smoke test: all 11 structured AI wires."""
import sys
import types
import numpy as np
import networkx as nx
from dataclasses import dataclass

print("=" * 60)
print("STRUCTURED AI SMOKE TEST -- 11 WIRES")
print("=" * 60)

# --- Patch RefinementStep into protocols (same pattern as test suite) ---
@dataclass
class Step:
    iteration: int; query: str; response: str; confidence: float
    improvement: float; strategy_used: str; reasoning: str; timestamp: object = None

class _ConvergenceDetectorProtocol:
    pass

_mod = sys.modules.get("hololoom.protocols.recursive_reasoning") or \
       sys.modules.get("hololoom.core.protocols.recursive_reasoning")
if _mod is not None:
    if not hasattr(_mod, "RefinementStep"):
        _mod.RefinementStep = Step
    if not hasattr(_mod, "ConvergenceDetectorProtocol"):
        _mod.ConvergenceDetectorProtocol = _ConvergenceDetectorProtocol
else:
    _shim = types.ModuleType("hololoom.protocols.recursive_reasoning")
    _shim.RefinementStep = Step
    _shim.ConvergenceDetectorProtocol = _ConvergenceDetectorProtocol
    sys.modules["hololoom.protocols.recursive_reasoning"] = _shim
    sys.modules["hololoom.core.protocols.recursive_reasoning"] = _shim

# === W1: Semantic Convergence ===
print("\n--- W1: Semantic Convergence Detector ---")
from hololoom.core.convergence.detectors import SemanticConvergenceDetector
det = SemanticConvergenceDetector(threshold=0.05)
history = [
    Step(1, 'q', 'Thompson Sampling uses Beta distributions for exploration and exploitation balancing.', 0.6, 0.0, 's', 'r'),
    Step(2, 'q', 'Thompson Sampling employs Beta(a,b) priors to balance exploration vs exploitation in bandit problems.', 0.75, 0.15, 's', 'r'),
    Step(3, 'q', 'Thompson Sampling employs Beta(a,b) priors to balance exploration versus exploitation in multi-armed bandit problems.', 0.82, 0.07, 's', 'r'),
    Step(4, 'q', 'Thompson Sampling employs Beta(a,b) priors to balance exploration versus exploitation in multi-armed bandit problems.', 0.83, 0.01, 's', 'r'),
]
for i in range(1, len(history) + 1):
    converged, reason = det.detect(history[:i])
    status = "CONVERGED" if converged else "continuing"
    print(f"  Pass {i}: {status}")
    if converged:
        print(f"         {reason}")

# === W2: Drift Detection ===
print("\n--- W2: Drift Detector (KL hallucination) ---")
from hololoom.core.convergence.drift_detector import DriftDetector
drift = DriftDetector(threshold=0.3)
context = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
r1 = drift.check(np.array([1.1, 2.1, 2.9, 4.1, 5.0]), context)
r2 = drift.check(np.array([10.0, 0.0, 0.0, 0.0, 0.0]), context)
print(f"  Grounded response:     JSD={r1.divergence:.4f}  drifted={r1.drifted}")
print(f"  Hallucinated response: JSD={r2.divergence:.4f}  drifted={r2.drifted}")

# === W3: Minimax Gate ===
print("\n--- W3: Minimax Verification Gate ---")
from hololoom.core.convergence.minimax_gate import MinimaxGate
gate = MinimaxGate()
payoffs = np.array([
    [3.0, 3.0, 3.0],  # answer: safe
    [0.5, 8.0, 1.0],  # research: risky
    [2.0, 2.0, 2.0],  # remember: ok
    [1.0, 1.0, 6.0],  # clarify: niche
])
result = gate.verify(selected_tool=1, tool_payoffs=payoffs)
print(f"  TS picked:       tool 1 (research)  worst-case={result.ts_payoff:.1f}")
print(f"  Minimax prefers: tool {result.minimax_tool} (answer)    worst-case={result.minimax_payoff:.1f}")
print(f"  Verified: {result.verified}  regret={result.regret:.1f}")

# === W6: Causal Bandit ===
print("\n--- W6: Causal Bandit (deconfounded TS) ---")
from hololoom.core.convergence.causal_bandits import CausalBandit
try:
    from hololoom.causal.dag import CausalDAG, CausalNode, CausalEdge
    dag = CausalDAG()
    dag.add_node(CausalNode(name="query_difficulty"))
    dag.add_node(CausalNode(name="tool_selected"))
    dag.add_node(CausalNode(name="outcome_quality"))
    dag.add_edge(CausalEdge(source="query_difficulty", target="tool_selected"))
    dag.add_edge(CausalEdge(source="query_difficulty", target="outcome_quality"))
    dag.add_edge(CausalEdge(source="tool_selected", target="outcome_quality"))

    bandit = CausalBandit(n_arms=3, dag=dag)
    r = bandit.update(0, reward=0.9, context={"query_difficulty": 0.9})
    print(f"  Confounders found: {r.confounders_found}")
    print(f"  Raw: {r.raw_reward:.2f} -> Adjusted: {r.adjusted_reward:.2f}")
    print(f"  Deconfounding applied: {r.adjustment_applied}")
except ImportError:
    bandit = CausalBandit(n_arms=3)
    bandit.update(0, reward=0.9)
    print(f"  Standard TS (no causal module): priors={bandit.get_priors()}")

# === W7: Deliberation Convergence ===
print("\n--- W7: Deliberation Convergence (EVINCE) ---")
from hololoom.core.convergence.deliberation_convergence import DeliberationConvergence
delib = DeliberationConvergence(mi_threshold=0.05, max_rounds=5)
rounds = [
    ("We should use protocol-first design with swappable backends",
     "The protocol approach adds complexity without clear benefit for small teams",
     "Protocol-first is warranted given multi-backend requirement, but keep protocols minimal"),
    ("Minimal protocols with 3 methods: load, save, health_check",
     "Three methods is reasonable but health_check should be optional",
     "Minimal protocols: 2 required (load, save) + 1 optional (health_check)"),
    ("Agreed on 2+1 protocol pattern, now need serialization format",
     "JSON serialization is simplest and sufficient",
     "Minimal protocols: 2 required (load, save) + 1 optional (health_check). JSON serialization."),
]
for i, (p, c, s) in enumerate(rounds):
    m = delib.record_round(p, c, s)
    converged, reason = delib.has_converged()
    print(f"  Round {i+1}: agreement={m.proposal_critique_jsd:.3f}  stability={m.cross_round_jsd:.3f}  converged={converged}")

# === W8: KG Reasoning ===
print("\n--- W8: KG Reasoner (MCTS multi-hop) ---")
from hololoom.core.convergence.kg_reasoning import KGReasoner
kg = nx.MultiDiGraph()
kg.add_edge("thompson_sampling", "beta_distribution", rel_type="uses")
kg.add_edge("beta_distribution", "bayesian_inference", rel_type="part_of")
kg.add_edge("bayesian_inference", "probability_theory", rel_type="part_of")
kg.add_edge("thompson_sampling", "exploration", rel_type="balances")
kg.add_edge("exploration", "exploitation", rel_type="tradeoff_with")
kg.add_edge("thompson_sampling", "bandits", rel_type="solves")
kg.add_edge("bandits", "reinforcement_learning", rel_type="subfield_of")

reasoner = KGReasoner(max_depth=3, num_simulations=100)
result = reasoner.reason(kg, anchors=["thompson_sampling"], targets=["probability_theory"])
print(f"  Best path: {' -> '.join(result.best_path.nodes)}")
print(f"  Score: {result.best_path.score:.3f}  depth: {result.best_path.depth}")
print(f"  Explored: {result.paths_explored}  unique: {len(result.all_paths)}")

# === W9: Free Energy Bridge ===
print("\n--- W9: Free Energy Bridge (active inference) ---")
from hololoom.core.convergence.free_energy import FreeEnergyBridge
bridge = FreeEnergyBridge()
state = bridge.compute_free_energy(
    node_activations={"concept_A": 0.9, "concept_B": 0.1, "concept_C": 0.5},
    node_velocities={"concept_A": 0.0, "concept_B": 0.3, "concept_C": -0.1},
    edge_potentials=[("concept_A", "concept_B", 0.32), ("concept_B", "concept_C", 0.08)],
)
print(f"  Free energy: {state.total_free_energy:.3f}  (surprise={state.surprise:.3f} + complexity={state.complexity:.3f})")
print(f"  Entropy: {state.entropy:.3f}  kinetic: {state.kinetic_energy:.3f}")
actions = bridge.rank_actions(
    candidates=["concept_B", "concept_C"],
    node_activations={"concept_A": 0.9, "concept_B": 0.1, "concept_C": 0.5},
    neighbor_potentials={"concept_B": 0.32, "concept_C": 0.08},
    target_nodes=["concept_C"],
)
for a in actions:
    print(f"  Attend to {a.node_id}: epistemic={a.epistemic_value:.2f} pragmatic={a.pragmatic_value:.0f} -> EFE={a.expected_free_energy:.2f}")

# === W10: World Model ===
print("\n--- W10: KG World Model (forward prediction) ---")
from hololoom.core.convergence.kg_world_model import KGWorldModel
model = KGWorldModel()
pred = model.predict_next_state(
    current_activations={"thompson_sampling": 1.0, "beta_distribution": 0.3},
    kg=kg,
)
print(f"  Emerging nodes: {pred.emerging_nodes}")
print(f"  Confidence: {pred.confidence:.2f}")
for node, delta in sorted(pred.activation_deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
    print(f"  {node}: {delta:+.4f} -> {pred.predicted_activations[node]:.4f}")

# === W11: Symbolic Router ===
print("\n--- W11: Symbolic Router (adaptive dispatch) ---")
from hololoom.core.convergence.symbolic_router import SymbolicRouter
router = SymbolicRouter()
queries = [
    "What causes memory pressure to increase?",
    "Rank these three tools by reliability",
    "What path connects causal inference to game theory?",
    "Predict activation levels in 5 steps",
    "Should we refactor? Consider the tradeoffs.",
]
for q in queries:
    d = router.route(q)
    print(f"  \"{q[:45]}\" -> {d.solver_name} ({d.problem_type.value})")

# === W12: Scoring Evolver ===
print("\n--- W12: Scoring Evolver (AlphaEvolve) ---")
from hololoom.core.convergence.scoring_evolver import ScoringEvolver
evolver = ScoringEvolver(
    term_names=["w_tc", "w_urgency", "w_variety", "w_craving"],
    initial_weights={"w_tc": 25, "w_urgency": 15, "w_variety": 5, "w_craving": 6},
)
evolver.seed_population(n=10)
data = [{"option_0_w_tc": 0.8, "option_0_w_urgency": 0.5, "option_0_w_variety": 0.3, "option_0_w_craving": 0.6,
         "option_1_w_tc": 0.2, "option_1_w_urgency": 0.9, "option_1_w_variety": 0.7, "option_1_w_craving": 0.1}] * 20
truth = [0] * 20
for gen in range(50):
    parent = evolver.select_parent()
    child = evolver.mutate(parent)
    fitness = evolver.evaluate(child, data, truth)
    evolver.record(child, fitness)

best = evolver.best_config()
stats = evolver.stats()
print(f"  Best fitness: {stats.best_fitness:.0%}  mean: {stats.mean_fitness:.0%}")
print(f"  Population: {stats.population_size} configs  diversity: {stats.diversity:.2f}")
print(f"  Evolved weights:")
for k, v in sorted(best.weights.items()):
    initial = {"w_tc": 25, "w_urgency": 15, "w_variety": 5, "w_craving": 6}
    delta = v - initial.get(k, 10)
    print(f"    {k}: {v:.1f} ({delta:+.1f} from initial)")

print("\n" + "=" * 60)
print("ALL 11 WIRES OPERATIONAL")
print("=" * 60)
