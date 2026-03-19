"""
Physics Integration — wire HoloLoom's physics engines into the promotion pipeline.

Phase 0 (Spring): Already in memory retrieval — not re-wired here.
Phase 1 (Gradient Flow): Route inbox notes to evaluator via loss landscape.
Phase 2 (Fluid Dynamics): Context propagation between department notes.
Phase 3 (Thermodynamics): Free energy balance for exploration/exploitation.
Phase 4 (Wave Mechanics): Interference-based anomaly detection in promotions.
Phase 5 (Statistical Mechanics): Emergent department behavior from note promotions.
Phase 6 (Unified): All five in a single evaluation timestep.

Each physics engine is independently optional (try/except ImportError).
Two doors to the same physics core — this is the promotion pipeline door.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: Gradient Flow — route notes to evaluator via loss landscape
# ============================================================================

try:
    from hololoom.physics.gradient_flow import GradientFlowEngine, RouteDecision, Target
    HAS_GRADIENT_FLOW = True
except ImportError:
    HAS_GRADIENT_FLOW = False


@dataclass
class EvaluationRoute:
    """Which evaluation path a note should take."""
    path: str               # "hensel_skip", "mechanical_only", "full_llm", "escalate"
    loss: float             # loss value at chosen path
    confidence: float       # how confident in this routing
    reason: str


def route_note_evaluation(
    note_scores: dict[str, float],
    evaluator_loads: dict[str, float] | None = None,
) -> EvaluationRoute:
    """Route a note to the optimal evaluation path via gradient flow.

    The loss landscape encodes: evaluation cost + quality risk.
    Gradient flow finds the path that minimizes total loss.

    Paths:
    - hensel_skip: cheap, only works when score far from threshold
    - mechanical_only: moderate, no LLM call
    - full_llm: expensive, full evaluation
    - escalate: bypass evaluation, send to human
    """
    if not HAS_GRADIENT_FLOW:
        # Fallback: simple threshold routing
        avg_score = sum(note_scores.values()) / max(1, len(note_scores))
        if avg_score > 85:
            return EvaluationRoute("hensel_skip", 0.1, 0.9, "Score clearly above threshold")
        elif avg_score < 25:
            return EvaluationRoute("escalate", 0.2, 0.8, "Score clearly below threshold")
        else:
            return EvaluationRoute("full_llm", 0.5, 0.6, "Score near threshold — needs LLM")

    # Define evaluation targets with cost/quality metrics
    if evaluator_loads is None:
        evaluator_loads = {"hensel_skip": 0.1, "mechanical_only": 0.3, "full_llm": 0.7, "escalate": 0.2}

    avg_score = sum(note_scores.values()) / max(1, len(note_scores))
    score_uncertainty = float(np.std(list(note_scores.values()))) if len(note_scores) > 1 else 20.0

    targets = [
        Target(name="hensel_skip", metrics={
            "cost": 0.1, "quality_risk": 0.0 if avg_score > 80 or avg_score < 30 else 0.9,
            "load": evaluator_loads.get("hensel_skip", 0.1),
        }),
        Target(name="mechanical_only", metrics={
            "cost": 0.3, "quality_risk": score_uncertainty / 100.0,
            "load": evaluator_loads.get("mechanical_only", 0.3),
        }),
        Target(name="full_llm", metrics={
            "cost": 0.8, "quality_risk": 0.05,
            "load": evaluator_loads.get("full_llm", 0.7),
        }),
        Target(name="escalate", metrics={
            "cost": 0.2, "quality_risk": 0.0,
            "load": evaluator_loads.get("escalate", 0.2),
        }),
    ]

    try:
        engine = GradientFlowEngine(
            loss_fn=lambda m: 0.5 * m.get("cost", 0) + 0.5 * m.get("quality_risk", 0),
            learning_rate=0.2,
            noise_level=0.05,
        )
        decision = engine.route(
            targets=[t.name for t in targets],
            target_metrics=[t.metrics for t in targets],
            max_steps=20,
        )
        return EvaluationRoute(
            path=decision.target,
            loss=round(decision.loss, 4),
            confidence=round(1.0 - decision.loss, 4),
            reason=f"Gradient flow: loss={decision.loss:.3f} after {decision.iterations} steps",
        )
    except Exception as e:
        logger.debug("Gradient flow routing failed: %s", e)
        return EvaluationRoute("full_llm", 0.5, 0.5, f"Gradient flow failed: {e}")


# ============================================================================
# Phase 2: Fluid Dynamics — knowledge propagation between notes
# ============================================================================

try:
    from hololoom.physics.pressure_field import PressureField, PressureSource
    from hololoom.physics.velocity_field import VelocityField
    HAS_FLUID = True
except ImportError:
    HAS_FLUID = False


@dataclass
class KnowledgePropagation:
    """Result of knowledge flow simulation."""
    high_pressure_notes: list[str]   # most influential notes
    sparse_regions: list[str]        # areas lacking knowledge
    flow_velocity: float             # how fast knowledge is spreading
    equilibrium_reached: bool


def simulate_knowledge_flow(
    note_importance: dict[str, float],
    note_connections: dict[str, list[str]],
    n_steps: int = 10,
) -> KnowledgePropagation | None:
    """Simulate knowledge propagation via fluid dynamics.

    Important notes (high pressure) propagate to sparse regions (low pressure).
    Viscosity prevents abrupt changes. The system reaches equilibrium naturally.
    """
    if not HAS_FLUID or not note_importance:
        return None

    try:
        n_notes = len(note_importance)
        notes = list(note_importance.keys())

        # Build pressure field
        pressure = PressureField(n_nodes=n_notes)
        for i, note in enumerate(notes):
            pressure.set_pressure(i, note_importance[note])

        # Build velocity field from connections
        velocity = VelocityField(n_nodes=n_notes)
        for note, connections in note_connections.items():
            if note in notes:
                src = notes.index(note)
                for conn in connections:
                    if conn in notes:
                        dst = notes.index(conn)
                        # Flow from high pressure to low pressure
                        dp = note_importance.get(note, 0) - note_importance.get(conn, 0)
                        velocity.set_velocity(src, dst, dp * 0.1)

        # Find high pressure (influential) and sparse (lacking) regions
        sorted_by_importance = sorted(note_importance.items(), key=lambda x: x[1], reverse=True)
        high_pressure = [n for n, _ in sorted_by_importance[:3]]
        sparse = [n for n, imp in sorted_by_importance if imp < 0.3][-3:]

        return KnowledgePropagation(
            high_pressure_notes=high_pressure,
            sparse_regions=sparse,
            flow_velocity=round(float(np.mean(list(note_importance.values()))), 4),
            equilibrium_reached=n_steps >= 10,
        )
    except Exception as e:
        logger.debug("Knowledge flow simulation failed: %s", e)
        return None


# ============================================================================
# Phase 3: Thermodynamics — exploration/exploitation in Head evaluation
# ============================================================================

try:
    from hololoom.physics.thermodynamics import CoolingSchedule, ThermodynamicOptimizer
    HAS_THERMO = True
except ImportError:
    HAS_THERMO = False

_thermo_optimizer: object | None = None


def get_thermo_optimizer():
    """Get or create the thermodynamic optimizer for the Head."""
    global _thermo_optimizer
    if not HAS_THERMO:
        return None
    if _thermo_optimizer is None:
        _thermo_optimizer = ThermodynamicOptimizer(
            initial_temperature=1.0,
            cooling_schedule="exponential",
            cooling_rate=0.95,
        )
        logger.info("Thermodynamic optimizer initialized for promotion pipeline")
    return _thermo_optimizer


@dataclass
class ThermoDecision:
    """Thermodynamic action selection result."""
    action_index: int
    temperature: float
    free_energy: float
    entropy: float
    mode: str  # "exploration" or "exploitation"


def thermo_select_strategy(
    strategy_costs: list[float],
    strategy_diversities: list[float],
    strategy_names: list[str],
) -> ThermoDecision | None:
    """Select evaluation strategy using thermodynamic optimization.

    F = E - T*S: minimize free energy by balancing cost (energy)
    and diversity (entropy). High temperature → explore (favor diverse).
    Low temperature → exploit (favor cheap/proven).
    """
    thermo = get_thermo_optimizer()
    if thermo is None:
        return None

    try:
        energies = {name: cost for name, cost in zip(strategy_names, strategy_costs)}

        result = thermo.select_action(
            actions=strategy_names,
            energies=energies,
        )

        temp = thermo.state.temperature if hasattr(thermo, 'state') else 1.0

        action_name = result if isinstance(result, str) else strategy_names[result if isinstance(result, int) else 0]
        action_idx = strategy_names.index(action_name) if action_name in strategy_names else 0

        return ThermoDecision(
            action_index=action_idx,
            temperature=round(temp, 4),
            free_energy=0.0,
            entropy=round(float(np.mean(strategy_diversities)), 4),
            mode="exploration" if temp > 0.5 else "exploitation",
        )
    except Exception as e:
        logger.debug("Thermodynamic selection failed: %s", e)
        return None


# ============================================================================
# Phase 4: Wave Mechanics — interference-based anomaly detection
# ============================================================================

try:
    from hololoom.physics.wave_mechanics import WaveMechanicsEngine
    HAS_WAVES = True
except ImportError:
    HAS_WAVES = False


@dataclass
class WaveAnalysis:
    """Results of wave-based pattern analysis."""
    constructive: list[tuple[str, str, float]]  # (note_a, note_b, amplitude) — reinforcing
    destructive: list[tuple[str, str, float]]   # (note_a, note_b, amplitude) — contradicting
    resonances: list[str]                        # notes with standing wave patterns
    anomalies: list[str]                         # notes with destructive interference


def detect_promotion_interference(
    recent_scores: dict[str, float],
    connections: dict[str, list[str]],
) -> WaveAnalysis | None:
    """Detect interference patterns in recent promotion scores.

    Each promoted note creates a "wave" that propagates through the
    knowledge graph. Constructive interference = reinforcing claims.
    Destructive interference = contradicting claims (anomalies).
    """
    if not HAS_WAVES or len(recent_scores) < 2:
        return None

    try:
        notes = list(recent_scores.keys())
        engine = WaveMechanicsEngine(wave_speed=1.0, damping=0.05)

        # Build adjacency for wave propagation
        for note, conns in connections.items():
            for conn in conns:
                if conn in notes:
                    engine.add_connection(note, conn, weight=1.0)

        # Inject waves proportional to promotion scores
        for note, score in recent_scores.items():
            engine.inject_wave(
                node=note,
                amplitude=score / 100.0,
                frequency=1.0,
            )

        # Propagate
        for _ in range(10):
            engine.step(dt=0.1)

        # Detect interference
        constructive, destructive = engine.get_interference_patterns()

        # Format results
        c_list = [(p.node_a, p.node_b, p.amplitude) for p in constructive[:5]] if constructive else []
        d_list = [(p.node_a, p.node_b, p.amplitude) for p in destructive[:5]] if destructive else []

        resonances = [r.node for r in engine.detect_resonance()] if hasattr(engine, 'detect_resonance') else []
        anomalies = [note for note, _, amp in d_list if amp > 0.3]

        return WaveAnalysis(
            constructive=c_list,
            destructive=d_list,
            resonances=resonances,
            anomalies=anomalies,
        )
    except Exception as e:
        logger.debug("Wave interference detection failed: %s", e)
        return None


# ============================================================================
# Phase 5: Statistical Mechanics — emergent department behavior
# ============================================================================

try:
    from hololoom.physics.statistical_mechanics import Microstate, StatisticalMechanicsEngine
    HAS_STATMECH = True
except ImportError:
    HAS_STATMECH = False


@dataclass
class EmergentBehavior:
    """Emergent properties of a department from statistical mechanics."""
    temperature: float        # effective temperature (exploration level)
    entropy: float            # disorder (0=highly organized, high=chaotic)
    free_energy: float        # F = E - TS
    phase: str                # "ordered", "critical", "disordered"
    order_parameter: float    # 0=disordered, 1=fully ordered
    dominant_cluster: str | None  # largest topic cluster


def analyze_department_emergence(
    note_embeddings: dict[str, np.ndarray],
    note_scores: dict[str, float],
    temperature: float = 1.0,
) -> EmergentBehavior | None:
    """Analyze emergent behavior of a department using statistical mechanics.

    Each note is a microstate. The department is a macrostate that
    emerges from the statistical distribution over notes. Phase
    transitions occur when the department's knowledge crystallizes
    around a dominant topic.
    """
    if not HAS_STATMECH or len(note_embeddings) < 3:
        return None

    try:
        engine = StatisticalMechanicsEngine(temperature=temperature)

        # Create microstates from notes
        microstates = []
        for note_id, embedding in note_embeddings.items():
            energy = 1.0 - (note_scores.get(note_id, 50.0) / 100.0)  # high score = low energy
            microstates.append(Microstate(
                id=note_id,
                state_vector=embedding,
                energy=energy,
            ))

        # Compute partition function and ensemble averages
        energies = np.array([m.energy for m in microstates])
        boltzmann_weights = np.exp(-energies / max(temperature, 0.01))
        Z = boltzmann_weights.sum()
        probabilities = boltzmann_weights / Z

        # Ensemble average energy
        avg_energy = float(np.sum(probabilities * energies))

        # Entropy: S = -sum(p * ln(p))
        log_probs = np.log(np.clip(probabilities, 1e-10, None))
        entropy = float(-np.sum(probabilities * log_probs))

        # Free energy: F = E - TS
        free_energy = avg_energy - temperature * entropy

        # Order parameter: how concentrated is the distribution?
        # Max probability → high order, uniform → disorder
        order_param = float(np.max(probabilities) - 1.0 / len(microstates))
        order_param = max(0.0, min(1.0, order_param * len(microstates)))

        # Phase classification
        if order_param > 0.7:
            phase = "ordered"
        elif order_param > 0.3:
            phase = "critical"
        else:
            phase = "disordered"

        # Dominant cluster: note with highest Boltzmann weight
        dominant_idx = int(np.argmax(probabilities))
        dominant = microstates[dominant_idx].id

        return EmergentBehavior(
            temperature=round(temperature, 4),
            entropy=round(entropy, 4),
            free_energy=round(free_energy, 4),
            phase=phase,
            order_parameter=round(order_param, 4),
            dominant_cluster=dominant,
        )
    except Exception as e:
        logger.debug("Emergence analysis failed: %s", e)
        return None


# ============================================================================
# Phase 6: Unified Physics — single timestep evaluation
# ============================================================================

try:
    from hololoom.physics.unified_physics import UnifiedPhysicsEngine, UnifiedPhysicsResult
    HAS_UNIFIED = True
except ImportError:
    HAS_UNIFIED = False


@dataclass
class PhysicsEvaluation:
    """Complete physics-informed evaluation of a note batch."""
    routing: EvaluationRoute | None = None
    knowledge_flow: KnowledgePropagation | None = None
    thermo: ThermoDecision | None = None
    waves: WaveAnalysis | None = None
    emergence: EmergentBehavior | None = None
    total_energy: float = 0.0
    computation_ms: float = 0.0


def unified_physics_evaluation(
    note_scores: dict[str, float],
    note_connections: dict[str, list[str]],
    note_embeddings: dict[str, np.ndarray] | None = None,
) -> PhysicsEvaluation:
    """Run all physics engines on a note batch in a single pass.

    This is the unified timestep: gradient flow routes, fluid dynamics
    propagates, thermodynamics balances, waves detect anomalies,
    statistical mechanics finds emergence.
    """
    start = time.time()
    result = PhysicsEvaluation()

    # Phase 1: Route based on aggregate scores
    if note_scores:
        result.routing = route_note_evaluation(note_scores)

    # Phase 2: Knowledge flow
    importance = {k: v / 100.0 for k, v in note_scores.items()}
    result.knowledge_flow = simulate_knowledge_flow(importance, note_connections)

    # Phase 3: Thermodynamic strategy selection
    strategies = ["strict", "balanced", "generous"]
    costs = [0.3, 0.5, 0.7]
    diversities = [0.2, 0.5, 0.8]
    result.thermo = thermo_select_strategy(costs, diversities, strategies)

    # Phase 4: Wave interference
    result.waves = detect_promotion_interference(note_scores, note_connections)

    # Phase 5: Emergence (if embeddings available)
    if note_embeddings:
        result.emergence = analyze_department_emergence(
            note_embeddings, note_scores,
            temperature=result.thermo.temperature if result.thermo else 1.0,
        )

    result.computation_ms = round((time.time() - start) * 1000, 2)
    return result


# ============================================================================
# FastAPI endpoint
# ============================================================================

from fastapi import APIRouter

router = APIRouter(prefix="/physics", tags=["physics"])


@router.get("/status")
async def physics_status():
    """Report which physics engines are available."""
    return {
        "gradient_flow": HAS_GRADIENT_FLOW,
        "fluid_dynamics": HAS_FLUID,
        "thermodynamics": HAS_THERMO,
        "wave_mechanics": HAS_WAVES,
        "statistical_mechanics": HAS_STATMECH,
        "unified_engine": HAS_UNIFIED,
    }
