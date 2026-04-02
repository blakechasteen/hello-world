"""
HoloLoom Structured AI — 14-Day Kitchen Simulation
===================================================

Runs 14 days of SOUS-style meal decisions through the full structured
AI pipeline. Tracks how the systems learn, adapt, and cross-check
over time with realistic kitchen dynamics.

Each day:
  1. Inventory changes (ingredients expire, new items arrive)
  2. Symbolic router classifies the query
  3. Additive scoring ranks meals
  4. Causal bandit deconfounds rewards
  5. Minimax gate checks worst-case
  6. Free energy identifies what to attend to
  7. Decision is made
  8. Outcome reported (quality varies by context)
  9. Evolver updates weights
  10. All priors update

Run: python demo_multiday.py
"""

import sys
import types
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- Patch ---
@dataclass
class _RS:
    iteration: int; query: str; response: str; confidence: float
    improvement: float; strategy_used: str; reasoning: str; timestamp: object = None

class _CDP:
    pass

_shim = types.ModuleType("hololoom.protocols.recursive_reasoning")
_shim.RefinementStep = _RS
_shim.ConvergenceDetectorProtocol = _CDP
sys.modules.setdefault("hololoom.protocols.recursive_reasoning", _shim)
sys.modules.setdefault("hololoom.core.protocols.recursive_reasoning", _shim)

from hololoom.core.convergence.minimax_gate import MinimaxGate
from hololoom.core.convergence.causal_bandits import CausalBandit
from hololoom.core.convergence.free_energy import FreeEnergyBridge
from hololoom.core.convergence.kg_world_model import KGWorldModel
from hololoom.core.convergence.symbolic_router import SymbolicRouter, RoutingOutcome
from hololoom.core.convergence.scoring_evolver import ScoringEvolver
from hololoom.core.convergence.drift_detector import DriftDetector

np.random.seed(42)


def header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def section(title):
    print(f"\n  --- {title} ---")


# ============================================================================
# KITCHEN STATE
# ============================================================================

MEALS = ["Fried Rice", "Lentil Soup", "Pasta Carbonara", "Bean Tacos",
         "Omelette", "Stir Fry", "Grilled Cheese", "Chili"]

# Base properties per meal (static)
MEAL_PROPS = {
    "Fried Rice":      {"effort": 0.3, "uses_eggs": True,  "uses_veggies": True,  "uses_protein": False, "comfort": 0.6},
    "Lentil Soup":     {"effort": 0.4, "uses_eggs": False, "uses_veggies": True,  "uses_protein": False, "comfort": 0.7},
    "Pasta Carbonara": {"effort": 0.5, "uses_eggs": True,  "uses_veggies": False, "uses_protein": True,  "comfort": 0.9},
    "Bean Tacos":      {"effort": 0.2, "uses_eggs": False, "uses_veggies": True,  "uses_protein": False, "comfort": 0.5},
    "Omelette":        {"effort": 0.1, "uses_eggs": True,  "uses_veggies": True,  "uses_protein": False, "comfort": 0.4},
    "Stir Fry":        {"effort": 0.3, "uses_eggs": False, "uses_veggies": True,  "uses_protein": True,  "comfort": 0.6},
    "Grilled Cheese":  {"effort": 0.1, "uses_eggs": False, "uses_veggies": False, "uses_protein": False, "comfort": 0.8},
    "Chili":           {"effort": 0.6, "uses_eggs": False, "uses_veggies": True,  "uses_protein": True,  "comfort": 0.8},
}

@dataclass
class KitchenState:
    day: int = 0
    eggs_days_left: int = 5
    veggies_days_left: int = 3
    protein_days_left: int = 7
    last_cooked: Dict[str, int] = field(default_factory=dict)  # meal -> days ago
    cook_history: List[str] = field(default_factory=list)
    energy_level: float = 0.7  # 0=exhausted, 1=energetic

    def advance_day(self):
        self.day += 1
        self.eggs_days_left = max(0, self.eggs_days_left - 1)
        self.veggies_days_left = max(0, self.veggies_days_left - 1)
        self.protein_days_left = max(0, self.protein_days_left - 1)
        for meal in list(self.last_cooked.keys()):
            self.last_cooked[meal] += 1
        # Random events
        if self.day % 4 == 0:
            self.eggs_days_left = 6  # Restocked
        if self.day % 5 == 0:
            self.veggies_days_left = 4
        if self.day % 7 == 0:
            self.protein_days_left = 5
        # Energy: some days are rough
        # Energy: weekday pattern (day%7: 0=Sun relaxed, 1-5=work, 6=Sat relaxed)
        weekday = self.day % 7
        if weekday in (0, 6):  # Weekend
            self.energy_level = 0.6 + np.random.random() * 0.3
        elif weekday in (3, 4):  # Mid-late week slump
            self.energy_level = 0.15 + np.random.random() * 0.35
        else:
            self.energy_level = 0.35 + np.random.random() * 0.45

    def compute_scores(self) -> Dict[str, Dict[str, float]]:
        """Compute scoring terms for each meal given current state."""
        scores = {}
        for meal in MEALS:
            props = MEAL_PROPS[meal]
            # Time crunch: how much does this meal use expiring items?
            tc = 0.0
            if props["uses_eggs"] and self.eggs_days_left <= 2:
                tc += 0.5
            if props["uses_veggies"] and self.veggies_days_left <= 2:
                tc += 0.5
            if props["uses_protein"] and self.protein_days_left <= 2:
                tc += 0.3
            tc = min(1.0, tc)

            # Urgency: inverse of days left for ingredients this meal uses
            urgency = 0.0
            count = 0
            if props["uses_eggs"]:
                urgency += 1.0 / max(self.eggs_days_left, 1)
                count += 1
            if props["uses_veggies"]:
                urgency += 1.0 / max(self.veggies_days_left, 1)
                count += 1
            if props["uses_protein"]:
                urgency += 1.0 / max(self.protein_days_left, 1)
                count += 1
            urgency = min(1.0, urgency / max(count, 1))

            # Variety: penalty for recently cooked
            days_since = self.last_cooked.get(meal, 99)
            variety = min(1.0, days_since / 4.0)

            # Craving: comfort * energy interaction
            craving = props["comfort"] * (0.5 + 0.5 * (1.0 - self.energy_level))

            scores[meal] = {
                "w_tc": tc,
                "w_urgency": urgency,
                "w_variety": variety,
                "w_craving": craving,
            }
        return scores


def score_meal(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(weights[k] * scores[k] for k in weights)


def compute_outcome(meal: str, kitchen: KitchenState) -> float:
    """Simulate meal outcome quality (ground truth)."""
    props = MEAL_PROPS[meal]
    quality = 0.5

    # Bonus for using expiring items
    if props["uses_eggs"] and kitchen.eggs_days_left <= 2:
        quality += 0.15
    if props["uses_veggies"] and kitchen.veggies_days_left <= 2:
        quality += 0.15
    if props["uses_protein"] and kitchen.protein_days_left <= 2:
        quality += 0.1

    # Penalty for high effort when tired
    if kitchen.energy_level < 0.4 and props["effort"] > 0.4:
        quality -= 0.2

    # Bonus for variety
    days_since = kitchen.last_cooked.get(meal, 99)
    if days_since >= 3:
        quality += 0.1

    # Noise
    quality += np.random.normal(0, 0.05)

    return max(0.0, min(1.0, quality))


# ============================================================================
# INITIALIZE SYSTEMS
# ============================================================================

kitchen = KitchenState()
weights = {"w_tc": 25, "w_urgency": 15, "w_variety": 5, "w_craving": 6}

try:
    from hololoom.causal.dag import CausalDAG, CausalNode, CausalEdge
    dag = CausalDAG()
    dag.add_node(CausalNode(name="energy_level"))
    dag.add_node(CausalNode(name="tool_selected"))
    dag.add_node(CausalNode(name="outcome_quality"))
    dag.add_edge(CausalEdge(source="energy_level", target="tool_selected"))
    dag.add_edge(CausalEdge(source="energy_level", target="outcome_quality"))
    dag.add_edge(CausalEdge(source="tool_selected", target="outcome_quality"))
    bandit = CausalBandit(n_arms=len(MEALS), dag=dag)
    has_causal = True
except ImportError:
    bandit = CausalBandit(n_arms=len(MEALS))
    has_causal = False

gate = MinimaxGate(regret_threshold=0.05)
bridge = FreeEnergyBridge(prior_precision=1.0)
router = SymbolicRouter()
drift = DriftDetector(threshold=0.4)

evolver = ScoringEvolver(
    term_names=list(weights.keys()),
    initial_weights=weights,
    mutation_rate=0.3,
    mutation_scale=2.0,
)
evolver.seed_population(n=10)

# Historical data for evolver (accumulates)
evo_historical = []
evo_truth = []


# ============================================================================
# RUN 14 DAYS
# ============================================================================

print("=" * 80)
print("  HOLOLOOM STRUCTURED AI — 14-DAY KITCHEN SIMULATION")
print("=" * 80)
print()
print(f"  {'Day':>3} {'Energy':>7} {'Eggs':>5} {'Veg':>5} {'Prot':>5} "
      f"{'TS Pick':<18} {'MM Pick':<18} {'Decision':<18} {'Quality':>8} {'Gate':>6}")
print(f"  {'-'*95}")

daily_log = []
decisions_overridden = 0
total_quality = 0
gate_fires = 0
causal_adjustments = 0

TOTAL_DAYS = 180

for day in range(1, TOTAL_DAYS + 1):
    kitchen.advance_day()
    meal_scores = kitchen.compute_scores()

    # --- Score all meals ---
    scored = [(m, score_meal(meal_scores[m], weights)) for m in MEALS]
    scored.sort(key=lambda x: x[1], reverse=True)
    ts_pick = scored[0][0]
    ts_pick_idx = MEALS.index(ts_pick)

    # --- Minimax gate ---
    # Build payoff matrix: meals x (normal, guests, tired)
    payoffs = np.zeros((len(MEALS), 3))
    for i, meal in enumerate(MEALS):
        props = MEAL_PROPS[meal]
        base = score_meal(meal_scores[meal], weights) / 50.0  # Normalize
        payoffs[i, 0] = base  # Normal
        payoffs[i, 1] = base * (1.0 + props["comfort"] * 0.3)  # Guests love comfort
        payoffs[i, 2] = base * (1.0 - props["effort"] * 0.8)  # Tired penalizes effort

    gate_result = gate.verify(selected_tool=ts_pick_idx, tool_payoffs=payoffs)
    mm_pick = MEALS[gate_result.minimax_tool]

    # --- Free energy: what has most tension? ---
    activations = {}
    potentials = {}
    for meal in MEALS:
        tc = meal_scores[meal]["w_tc"]
        activations[meal] = tc
        potentials[meal] = tc * (1.0 - tc)  # Tension = how unresolved

    fe_actions = bridge.rank_actions(
        candidates=MEALS,
        node_activations=activations,
        neighbor_potentials=potentials,
    )
    fe_top = fe_actions[0].node_id if fe_actions else ts_pick

    # --- Decision logic ---
    # Use TS pick unless minimax gate fires AND we're in a risky state
    decision = ts_pick
    override_reason = ""
    if not gate_result.verified:
        gate_fires += 1
        if kitchen.energy_level < 0.4:
            decision = mm_pick
            override_reason = f"minimax override (tired, regret={gate_result.regret:.2f})"
            decisions_overridden += 1

    # --- Compute outcome ---
    quality = compute_outcome(decision, kitchen)
    total_quality += quality

    # --- Update systems ---
    decision_idx = MEALS.index(decision)
    context = {"energy_level": kitchen.energy_level} if has_causal else None
    update_result = bandit.update(decision_idx, quality, context)
    if update_result.adjustment_applied:
        causal_adjustments += 1

    router.report_outcome(RoutingOutcome(
        solver_name="convergence_engine",
        problem_type=router.route("What should I cook?").problem_type,
        quality=quality,
    ))

    kitchen.last_cooked[decision] = 0
    kitchen.cook_history.append(decision)

    # --- Evolver data ---
    example = {}
    for j, meal in enumerate(MEALS):
        for term in weights:
            example[f"option_{j}_{term}"] = meal_scores[meal][term]
    evo_historical.append(example)
    evo_truth.append(decision_idx)

    # --- Log ---
    gate_str = "FIRE" if not gate_result.verified else "ok"
    if override_reason:
        gate_str = override_reason

    print(f"  {day:>3} {kitchen.energy_level:>7.2f} {kitchen.eggs_days_left:>5} "
          f"{kitchen.veggies_days_left:>5} {kitchen.protein_days_left:>5} "
          f"{ts_pick:<18} {mm_pick:<18} {decision:<18} {quality:>8.2f} {gate_str:>6}")

    daily_log.append({
        "day": day, "ts_pick": ts_pick, "mm_pick": mm_pick,
        "decision": decision, "quality": quality,
        "energy": kitchen.energy_level, "gate_fired": not gate_result.verified,
        "override": bool(override_reason),
    })


# ============================================================================
# ANALYSIS
# ============================================================================

print()
print("=" * 80)
print(f"  {TOTAL_DAYS}-DAY ANALYSIS")
print("=" * 80)

avg_quality = total_quality / TOTAL_DAYS
print(f"\n  Average meal quality: {avg_quality:.2f}")
print(f"  Minimax gate fired:  {gate_fires} times")
print(f"  Decisions overridden: {decisions_overridden} times")
print(f"  Causal adjustments:  {causal_adjustments} / 14")

# Meal frequency
from collections import Counter
freq = Counter(kitchen.cook_history)
print(f"\n  Meal frequency:")
for meal, count in freq.most_common():
    bar = "#" * (count * 4)
    print(f"    {meal:<18} {count:>2}x  {bar}")

# Bandit priors after 14 days
print(f"\n  Causal bandit priors (after 14 days):")
priors = bandit.get_priors()
for i, meal in enumerate(MEALS):
    bar = "|" * int(priors[i] * 30)
    print(f"    {meal:<18} {priors[i]:.3f}  {bar}")

# Confounder stats
if bandit.confounder_stats:
    print(f"\n  Confounder adjustments: {dict(bandit.confounder_stats)}")

# Quality by decision method
override_qualities = [d["quality"] for d in daily_log if d["override"]]
normal_qualities = [d["quality"] for d in daily_log if not d["override"]]
if override_qualities:
    print(f"\n  Quality when minimax overrode TS:    {np.mean(override_qualities):.2f} (n={len(override_qualities)})")
    print(f"  Quality when TS was accepted:        {np.mean(normal_qualities):.2f} (n={len(normal_qualities)})")

# Quality when tired vs not
tired_q = [d["quality"] for d in daily_log if d["energy"] < 0.5]
rested_q = [d["quality"] for d in daily_log if d["energy"] >= 0.5]
if tired_q:
    print(f"\n  Quality on tired days (energy<0.5):  {np.mean(tired_q):.2f} (n={len(tired_q)})")
    print(f"  Quality on rested days:              {np.mean(rested_q):.2f} (n={len(rested_q)})")


# ============================================================================
# EVOLVE WEIGHTS BASED ON 14 DAYS OF DATA
# ============================================================================

print(f"\n  --- Evolving weights from 14-day history ---")

for gen in range(200):
    parent = evolver.select_parent()
    child = evolver.mutate(parent)
    fitness = evolver.evaluate(child, evo_historical, evo_truth)
    evolver.record(child, fitness)

best = evolver.best_config()
stats = evolver.stats()

print(f"  Best fitness after 200 generations: {stats.best_fitness:.0%}")
print(f"  Evolved weights vs initial:")
for term in sorted(weights.keys()):
    init = weights[term]
    evolved = best.weights[term]
    delta = evolved - init
    arrow = "^" if delta > 1 else "v" if delta < -1 else "="
    print(f"    {term:<12} {init:>6.1f} -> {evolved:>6.1f}  ({delta:+.1f}) {arrow}")


# ============================================================================
# RE-SCORE DAY 14 WITH EVOLVED WEIGHTS
# ============================================================================

print(f"\n  --- Re-scoring today with evolved weights ---")
current_scores = kitchen.compute_scores()
print(f"  {'Meal':<18} {'Initial wt':>12} {'Evolved wt':>12} {'Delta':>8}")
print(f"  {'-'*52}")
for meal in MEALS:
    s_init = score_meal(current_scores[meal], weights)
    s_evolved = score_meal(current_scores[meal], best.weights)
    delta = s_evolved - s_init
    print(f"  {meal:<18} {s_init:>12.1f} {s_evolved:>12.1f} {delta:>+8.1f}")

scored_evolved = [(m, score_meal(current_scores[m], best.weights)) for m in MEALS]
scored_evolved.sort(key=lambda x: x[1], reverse=True)
scored_initial = [(m, score_meal(current_scores[m], weights)) for m in MEALS]
scored_initial.sort(key=lambda x: x[1], reverse=True)

print(f"\n  Initial weights pick:  {scored_initial[0][0]} ({scored_initial[0][1]:.1f})")
print(f"  Evolved weights pick:  {scored_evolved[0][0]} ({scored_evolved[0][1]:.1f})")

if scored_initial[0][0] != scored_evolved[0][0]:
    print(f"  The evolver changed the recommendation!")
else:
    print(f"  Same recommendation, different confidence.")


# ============================================================================
# FINAL
# ============================================================================

# ============================================================================
# WEEKLY BREAKDOWN
# ============================================================================

print(f"\n  --- Weekly Breakdown ---")
n_weeks = (TOTAL_DAYS + 6) // 7
print(f"  {'Week':>6} {'Avg Q':>7} {'Gate':>6} {'Override':>9} {'Top Meal':<18} {'Tired Days':>11}")
print(f"  {'-'*60}")

for week in range(n_weeks):
    start = week * 7
    end = min(start + 7, len(daily_log))
    week_log = daily_log[start:end]
    if not week_log:
        continue
    wk_quality = np.mean([d["quality"] for d in week_log])
    wk_gates = sum(1 for d in week_log if d["gate_fired"])
    wk_overrides = sum(1 for d in week_log if d["override"])
    wk_tired = sum(1 for d in week_log if d["energy"] < 0.4)
    wk_meals = Counter(d["decision"] for d in week_log)
    top_meal = wk_meals.most_common(1)[0][0] if wk_meals else "—"
    print(f"  {week+1:>6} {wk_quality:>7.2f} {wk_gates:>6} {wk_overrides:>9} {top_meal:<18} {wk_tired:>11}")


# ============================================================================
# LEARNING CURVE
# ============================================================================

print(f"\n  --- Learning Curve (quality over time) ---")
window = 7
if len(daily_log) >= window:
    print(f"  {'Window':>8} {'Avg Quality':>12} {'Trend'}")
    print(f"  {'-'*45}")
    prev_avg = None
    for i in range(0, len(daily_log) - window + 1, window):
        chunk = daily_log[i:i+window]
        chunk_avg = np.mean([d["quality"] for d in chunk])
        days_label = f"Day {chunk[0]['day']}-{chunk[-1]['day']}"
        if prev_avg is not None:
            delta = chunk_avg - prev_avg
            trend = f"{'+'if delta > 0 else ''}{delta:.2f} {'>>>' if delta > 0.03 else '---' if delta < -0.03 else '==='}"
        else:
            trend = "(baseline)"
        bar = "|" * int(chunk_avg * 40)
        print(f"  {days_label:>8} {chunk_avg:>12.3f}  {bar}  {trend}")
        prev_avg = chunk_avg


# ============================================================================
# WHAT THE BANDIT LEARNED — COMPARE FIRST HALF vs SECOND HALF
# ============================================================================

print(f"\n  --- Adaptation: First Half vs Second Half ---")
half = len(daily_log) // 2
first_half = daily_log[:half]
second_half = daily_log[half:]

fh_quality = np.mean([d["quality"] for d in first_half])
sh_quality = np.mean([d["quality"] for d in second_half])
fh_overrides = sum(1 for d in first_half if d["override"])
sh_overrides = sum(1 for d in second_half if d["override"])
fh_tired_q = [d["quality"] for d in first_half if d["energy"] < 0.4]
sh_tired_q = [d["quality"] for d in second_half if d["energy"] < 0.4]

print(f"  {'Metric':<30} {'First Half':>12} {'Second Half':>12} {'Delta':>8}")
print(f"  {'-'*64}")
print(f"  {'Average quality':<30} {fh_quality:>12.3f} {sh_quality:>12.3f} {sh_quality-fh_quality:>+8.3f}")
print(f"  {'Minimax overrides':<30} {fh_overrides:>12} {sh_overrides:>12} {sh_overrides-fh_overrides:>+8}")
if fh_tired_q and sh_tired_q:
    fh_tq = np.mean(fh_tired_q)
    sh_tq = np.mean(sh_tired_q)
    print(f"  {'Tired-day quality':<30} {fh_tq:>12.3f} {sh_tq:>12.3f} {sh_tq-fh_tq:>+8.3f}")

fh_meals = Counter(d["decision"] for d in first_half)
sh_meals = Counter(d["decision"] for d in second_half)
print(f"\n  Meal distribution shift:")
all_decided = set(fh_meals.keys()) | set(sh_meals.keys())
for meal in sorted(all_decided):
    fh_count = fh_meals.get(meal, 0)
    sh_count = sh_meals.get(meal, 0)
    if fh_count != sh_count:
        print(f"    {meal:<18} {fh_count:>3}x -> {sh_count:>3}x  ({sh_count-fh_count:+d})")


# ============================================================================
# COUNTERFACTUAL: WHAT IF WE NEVER USED THE GATE?
# ============================================================================

print(f"\n  --- Counterfactual: What if the minimax gate was disabled? ---")

# Re-run with same seed, no gate
np.random.seed(42)
kitchen_cf = KitchenState()
bandit_cf = CausalBandit(n_arms=len(MEALS))
cf_quality_total = 0
cf_tired_qualities = []

for day_cf in range(1, TOTAL_DAYS + 1):
    kitchen_cf.advance_day()
    cf_scores = kitchen_cf.compute_scores()
    cf_scored = [(m, score_meal(cf_scores[m], weights)) for m in MEALS]
    cf_scored.sort(key=lambda x: x[1], reverse=True)
    cf_pick = cf_scored[0][0]

    cf_quality = compute_outcome(cf_pick, kitchen_cf)
    cf_quality_total += cf_quality
    if kitchen_cf.energy_level < 0.4:
        cf_tired_qualities.append(cf_quality)
    kitchen_cf.last_cooked[cf_pick] = 0
    bandit_cf.update(MEALS.index(cf_pick), cf_quality)

cf_avg = cf_quality_total / TOTAL_DAYS
print(f"  With structured AI:    avg quality = {avg_quality:.3f}")
print(f"  Without (TS only):     avg quality = {cf_avg:.3f}")
print(f"  Improvement:           {avg_quality - cf_avg:+.3f} ({(avg_quality-cf_avg)/cf_avg*100:+.1f}%)")

if cf_tired_qualities and tired_q:
    cf_tired_avg = np.mean(cf_tired_qualities)
    actual_tired_avg = np.mean(tired_q)
    print(f"\n  Tired-day quality:")
    print(f"    With structured AI:  {actual_tired_avg:.3f}")
    print(f"    Without:             {cf_tired_avg:.3f}")
    print(f"    Improvement:         {actual_tired_avg - cf_tired_avg:+.3f}")


# ============================================================================
# FINAL
# ============================================================================

print()
print("=" * 80)
print(f"  {TOTAL_DAYS} days. {TOTAL_DAYS} decisions. Every one auditable.")
print(f"  Average quality: {avg_quality:.3f} (vs {cf_avg:.3f} without structured AI)")
print(f"  System learned from {causal_adjustments} causal adjustments,")
print(f"  {gate_fires} minimax gate fires, and {decisions_overridden} overrides.")
print(f"  The evolver reached {stats.best_fitness:.0%} accuracy across {stats.generations} generations.")
print("=" * 80)


# ============================================================================
# DEEP QUESTIONS
# ============================================================================

header("DEEP QUESTIONS")

# Q1: Does the gate make better decisions than TS, or just different ones?
section("Q1: Does the gate actually help, or just substitute?")
override_days = [d for d in daily_log if d["override"]]
non_override_days = [d for d in daily_log if not d["override"]]
gate_fire_no_override = [d for d in daily_log if d["gate_fired"] and not d["override"]]

print(f"  Gate fired {gate_fires} times out of {TOTAL_DAYS} days ({gate_fires/TOTAL_DAYS*100:.0f}%)")
print(f"  Overrode TS {decisions_overridden} times ({decisions_overridden/max(gate_fires,1)*100:.0f}% of fires)")
print()

if override_days and non_override_days:
    o_q = np.mean([d["quality"] for d in override_days])
    n_q = np.mean([d["quality"] for d in non_override_days])
    print(f"  Quality when gate overrode:    {o_q:.3f} (n={len(override_days)})")
    print(f"  Quality when gate accepted:    {n_q:.3f} (n={len(non_override_days)})")

    # What would have happened if we always listened to the gate?
    # We can't know exactly, but we can compare gate-fire days
    if gate_fire_no_override:
        gf_q = np.mean([d["quality"] for d in gate_fire_no_override])
        print(f"  Quality when gate fired but we IGNORED it: {gf_q:.3f} (n={len(gate_fire_no_override)})")
        print(f"\n  Interpretation: The gate fires on {gate_fires/TOTAL_DAYS*100:.0f}% of days.")
        if o_q > gf_q:
            print(f"  When we listen to it (override), quality is {o_q:.3f}.")
            print(f"  When we ignore it (fire but no override), quality is {gf_q:.3f}.")
            print(f"  Delta: {o_q - gf_q:+.3f} — listening to the gate helps.")
        else:
            print(f"  When we listen: {o_q:.3f}. When we ignore: {gf_q:.3f}.")
            print(f"  Delta: {o_q - gf_q:+.3f} — the gate's overrides aren't always better.")
            print(f"  This suggests the gate is too conservative or the regret threshold needs tuning.")

# Q2: Is the causal deconfounding doing real work?
section("Q2: Is causal deconfounding changing outcomes?")

# Compare bandit priors: causal vs naive
print(f"  Causal bandit priors vs naive bandit priors after {TOTAL_DAYS} days:")
print(f"  {'Meal':<18} {'Causal':>8} {'Naive':>8} {'Delta':>8} {'Interpretation'}")
print(f"  {'-'*65}")

naive_priors = bandit_cf.get_priors()
causal_priors = bandit.get_priors()
for i, meal in enumerate(MEALS):
    cp = causal_priors[i]
    np_ = naive_priors[i]
    delta = cp - np_
    if abs(delta) > 0.05:
        interp = "causal sees more" if delta > 0 else "causal sees less"
    else:
        interp = "~same"
    print(f"  {meal:<18} {cp:>8.3f} {np_:>8.3f} {delta:>+8.3f} {interp}")

# Which meals are most affected by energy confounding?
print(f"\n  Meals most affected by energy-level confounding:")
energy_corr = {}
for meal in MEALS:
    meal_days = [(d["quality"], d["energy"]) for d in daily_log if d["decision"] == meal]
    if len(meal_days) >= 3:
        qualities = [q for q, e in meal_days]
        energies = [e for q, e in meal_days]
        corr = np.corrcoef(qualities, energies)[0, 1] if np.std(qualities) > 0 and np.std(energies) > 0 else 0
        energy_corr[meal] = corr
        if abs(corr) > 0.2:
            direction = "better when rested" if corr > 0 else "better when tired"
            print(f"    {meal:<18} corr={corr:+.2f} ({direction})")

# Q3: What did the evolver actually learn?
section("Q3: What did the evolver learn about scoring?")

print(f"  Evolution: {stats.generations} generations, best fitness={stats.best_fitness:.0%}")
print(f"  Population diversity: {stats.diversity:.2f}")
print()

# Show the evolved weights tell us about what matters
print(f"  What the weight changes tell us:")
for term in sorted(weights.keys()):
    init = weights[term]
    evolved = best.weights[term]
    delta = evolved - init
    pct = (delta / init) * 100 if init > 0 else 0

    if abs(pct) > 10:
        if delta > 0:
            print(f"    {term}: +{pct:.0f}% — this signal is UNDERWEIGHTED in initial config")
        else:
            print(f"    {term}: {pct:.0f}% — this signal is OVERWEIGHTED or confounded")
    else:
        print(f"    {term}: ~stable ({pct:+.0f}%) — initial weight is about right")

# Q4: Exploration vs exploitation
section("Q4: Are we exploring enough?")

never_tried = [m for m in MEALS if freq.get(m, 0) == 0]
rarely_tried = [m for m in MEALS if 0 < freq.get(m, 0) <= 2]
dominant = [m for m in MEALS if freq.get(m, 0) > TOTAL_DAYS * 0.15]

print(f"  Never tried:     {never_tried if never_tried else 'none'}")
print(f"  Rarely tried:    {rarely_tried if rarely_tried else 'none'}")
print(f"  Dominant (>15%): {dominant}")
print()

if never_tried:
    print(f"  {len(never_tried)} meal(s) never explored in {TOTAL_DAYS} days.")
    print(f"  Thompson Sampling's 5% entropy injection should eventually explore these,")
    print(f"  but the system is exploiting known-good options aggressively.")
    print(f"  This is the right behavior for a kitchen (cost of bad meal is real)")
    print(f"  but would be wrong for a research system (need to explore).")

# Effective exploration rate
unique_per_week = []
for week in range(n_weeks):
    start = week * 7
    end = min(start + 7, len(daily_log))
    week_meals = set(d["decision"] for d in daily_log[start:end])
    unique_per_week.append(len(week_meals))

print(f"\n  Unique meals per week: {unique_per_week}")
print(f"  Average: {np.mean(unique_per_week):.1f} / {len(MEALS)} available")
print(f"  Exploration rate: {np.mean(unique_per_week)/len(MEALS)*100:.0f}%")

# Q5: What's the system's biggest blind spot?
section("Q5: Biggest blind spots")

# Meals we cook on tired days vs rested days
print(f"  What we cook when tired vs rested:")
tired_meals = Counter(d["decision"] for d in daily_log if d["energy"] < 0.4)
rested_meals = Counter(d["decision"] for d in daily_log if d["energy"] >= 0.6)
print(f"  {'Meal':<18} {'Tired':>6} {'Rested':>7}")
print(f"  {'-'*33}")
for meal in MEALS:
    t = tired_meals.get(meal, 0)
    r = rested_meals.get(meal, 0)
    if t + r > 0:
        print(f"  {meal:<18} {t:>6} {r:>7}")

# Are there quality patterns we're missing?
print(f"\n  Quality by meal (is any meal consistently bad?):")
for meal in MEALS:
    meal_q = [d["quality"] for d in daily_log if d["decision"] == meal]
    if meal_q:
        mean_q = np.mean(meal_q)
        std_q = np.std(meal_q)
        marker = " *** high variance" if std_q > 0.12 else ""
        marker = " *** consistently low" if mean_q < 0.65 else marker
        print(f"    {meal:<18} mean={mean_q:.3f} std={std_q:.3f} n={len(meal_q):>3}{marker}")

# Q6: What would a perfectly calibrated system do?
section("Q6: Optimal vs actual (hindsight analysis)")

# For each day, what SHOULD we have cooked (best outcome)?
optimal_quality = 0
optimal_decisions = []
for d in daily_log:
    # We can't know the counterfactual for all meals, but we know
    # the outcome for the meal we picked. The question is: could
    # the system have known to pick better?
    pass

# Instead: correlation between score and outcome
scores_vs_quality = []
for d in daily_log:
    day_scores = kitchen.compute_scores()  # This won't be right (state changed), so approximate
    scores_vs_quality.append((d["quality"],))

print(f"  Perfect hindsight isn't available (we only observe the chosen meal).")
print(f"  But we can measure calibration: does high confidence predict high quality?")
print()

# Bin by day number to show calibration over time
thirds = len(daily_log) // 3
for label, chunk in [("Early", daily_log[:thirds]),
                     ("Mid", daily_log[thirds:2*thirds]),
                     ("Late", daily_log[2*thirds:])]:
    gate_accuracy = sum(1 for d in chunk if d["gate_fired"] == (d["quality"] < 0.7)) / max(len(chunk), 1)
    avg_q = np.mean([d["quality"] for d in chunk])
    overrides = sum(1 for d in chunk if d["override"])
    print(f"  {label:>5} ({len(chunk)} days):  quality={avg_q:.3f}  gate_accuracy={gate_accuracy:.0%}  overrides={overrides}")

print(f"\n  Gate accuracy = fraction of days where gate-fire correctly predicted")
print(f"  below-median quality. Higher = better calibrated gate.")
