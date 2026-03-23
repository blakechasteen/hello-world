"""
PhysicsCoupler — runs the 6-phase coupled ring with time-stepping.

The ring: Spring -> Gradient -> Fluid -> Thermo -> Wave -> StatMech -> (back to Spring)
Each phase reads from PhysicsState, runs its engine, writes back.
The loop closes: StatMech's order parameter feeds Spring's damping.

Graceful degradation: if any phase is unavailable, the coupler
continues with the remaining phases. The ring is partial but still coupled.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

from .meta_thermo import MetaThermodynamics
from .protocol import PhysicsPhase, PhysicsState

logger = logging.getLogger(__name__)


def _discover_phases() -> list[PhysicsPhase]:
    """Try to import each phase adapter. Skip unavailable ones."""
    phases: list[PhysicsPhase] = []
    phase_classes = []

    try:
        from .spring_phase import SpringPhase
        phase_classes.append(SpringPhase)
    except ImportError:
        logger.warning("SpringPhase unavailable")

    try:
        from .gradient_phase import GradientPhase
        phase_classes.append(GradientPhase)
    except ImportError:
        logger.warning("GradientPhase unavailable")

    try:
        from .fluid_phase import FluidPhase
        phase_classes.append(FluidPhase)
    except ImportError:
        logger.warning("FluidPhase unavailable")

    try:
        from .thermo_phase import ThermoPhase
        phase_classes.append(ThermoPhase)
    except ImportError:
        logger.warning("ThermoPhase unavailable")

    try:
        from .wave_phase import WavePhase
        phase_classes.append(WavePhase)
    except ImportError:
        logger.warning("WavePhase unavailable")

    try:
        from .statmech_phase import StatMechPhase
        phase_classes.append(StatMechPhase)
    except ImportError:
        logger.warning("StatMechPhase unavailable")

    for cls in phase_classes:
        try:
            phase = cls()
            if phase.available():
                phases.append(phase)
                logger.info(f"Phase {phase.name} initialized")
            else:
                logger.warning(f"Phase {cls.__name__} not available")
        except Exception as e:
            logger.warning(f"Phase {cls.__name__} init failed: {e}")

    return phases


class PhysicsCoupler:
    """Runs the coupled physics ring.

    Usage:
        coupler = PhysicsCoupler()
        for _ in range(100):
            state = coupler.step()
        print(state.free_energy)  # Should trend down
    """

    def __init__(
        self,
        phases: list[PhysicsPhase] | None = None,
        initial_state: PhysicsState | None = None,
        history_size: int = 100,
    ):
        self._phases = phases if phases is not None else _discover_phases()
        self._state = initial_state or PhysicsState()
        self._history: deque[PhysicsState] = deque(maxlen=history_size)
        self._meta_thermo = MetaThermodynamics()
        self._step_times_ms: deque[float] = deque(maxlen=100)

        logger.info(
            f"PhysicsCoupler initialized with {len(self._phases)} phases: "
            f"{[p.name for p in self._phases]}"
        )

    @property
    def state(self) -> PhysicsState:
        return self._state

    @property
    def phases(self) -> list[PhysicsPhase]:
        return list(self._phases)

    @property
    def meta_thermo(self) -> MetaThermodynamics:
        return self._meta_thermo

    def step(self, context: dict | None = None) -> PhysicsState:
        """Run one full cycle through all available phases.

        Args:
            context: Optional per-step context (node IDs, query info, etc.)

        Returns:
            Updated PhysicsState after full ring traversal.
        """
        t0 = time.perf_counter()
        state = self._state
        state.timestep += 1

        # Inject context into state if provided
        if context:
            for node_id in context.get("node_ids", []):
                if node_id not in state.activations:
                    state.activations[node_id] = 0.1  # Seed activation

        # Run each phase in ring order
        for phase in self._phases:
            try:
                state = phase.step(state)
            except Exception as e:
                logger.warning(f"Phase {phase.name} failed at step {state.timestep}: {e}")
                # Continue with state as-is (graceful degradation)

        # Meta-thermodynamics: system observes itself, learns cooling schedule
        self._meta_thermo.observe(state)

        # Apply learned cooling schedule
        if state.timestep % 10 == 0:
            self._meta_thermo.select_schedule()
        state = self._meta_thermo.apply_schedule(state)

        # Compute derived quantities
        state.free_energy = state.energy - state.temperature * state.entropy

        # Cap per-node dicts to prevent unbounded growth
        _cap_dict(state.activations, 1000)
        _cap_dict(state.velocities, 1000)
        _cap_dict(state.amplitudes, 1000)
        _cap_dict(state.pressures, 1000)

        # Record
        self._state = state
        self._history.append(state.copy())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._step_times_ms.append(elapsed_ms)

        return state

    def step_n(self, n: int, context: dict | None = None) -> PhysicsState:
        """Run n full cycles."""
        for _ in range(n):
            self.step(context)
        return self._state

    def energy_trend(self, window: int = 20) -> float:
        """Compute free energy trend over recent history.

        Returns negative if energy is decreasing (good — finding minimum).
        """
        if len(self._history) < 2:
            return 0.0
        recent = list(self._history)[-window:]
        if len(recent) < 2:
            return 0.0
        return recent[-1].free_energy - recent[0].free_energy

    def order_trend(self, window: int = 20) -> float:
        """Compute order parameter trend. Positive = structure emerging."""
        if len(self._history) < 2:
            return 0.0
        recent = list(self._history)[-window:]
        if len(recent) < 2:
            return 0.0
        return recent[-1].order_parameter - recent[0].order_parameter

    def avg_step_ms(self) -> float:
        """Average step time in milliseconds."""
        if not self._step_times_ms:
            return 0.0
        return sum(self._step_times_ms) / len(self._step_times_ms)

    def status(self) -> dict:
        """Full status for monitoring/viz."""
        return {
            "n_phases": len(self._phases),
            "phases": [p.name for p in self._phases],
            "state": self._state.to_dict(),
            "energy_trend": round(self.energy_trend(), 4),
            "order_trend": round(self.order_trend(), 4),
            "avg_step_ms": round(self.avg_step_ms(), 3),
            "total_steps": self._state.timestep,
            "meta_thermo": self._meta_thermo.stats(),
        }


def _cap_dict(d: dict, max_size: int) -> None:
    """Remove lowest-value entries if dict exceeds max_size."""
    if len(d) <= max_size:
        return
    sorted_keys = sorted(d, key=lambda k: abs(d[k]))
    for key in sorted_keys[: len(d) - max_size]:
        del d[key]


# Module-level singleton (lazy)
_coupler: Optional[PhysicsCoupler] = None


def get_coupler() -> PhysicsCoupler:
    """Get or create the singleton PhysicsCoupler."""
    global _coupler
    if _coupler is None:
        _coupler = PhysicsCoupler()
    return _coupler
