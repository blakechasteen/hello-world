"""
RitualLayer — the first concrete runtime layer.
=================================================

Composes RitualRegistry + RitualTrigger + RitualJournal into a
single lifecycle unit managed by the HoloLoomRuntime.

On start:
  1. Get bus from runtime (if available)
  2. Create RitualTrigger wired to bus + registry
  3. Register default ritual bindings
  4. Validate DAG
  5. Start trigger
  If any step fails → DEGRADED, log, continue

On stop:
  1. Stop trigger
  2. Close journal
"""

import logging
from typing import Any

from .layer import LayerState
from ..ritual.hooks import HookEvent
from ..ritual.journal import RitualJournal
from ..ritual.registry import RitualRegistry
from ..ritual.types import HookBinding, Priority

logger = logging.getLogger(__name__)


def _default_bindings() -> list[HookBinding]:
    """Register the built-in ritual bindings."""
    bindings = []

    # Coding: pre_code_health_check
    try:
        from ..ritual.rituals.coding import (
            PRE_CODE_HEALTH_CHECK,
            PRE_CODE_HEALTH_CHECK_BINDING_CONFIG,
        )
        bindings.append(HookBinding(
            ritual=PRE_CODE_HEALTH_CHECK,
            trigger=PRE_CODE_HEALTH_CHECK_BINDING_CONFIG["trigger"],
            condition=PRE_CODE_HEALTH_CHECK_BINDING_CONFIG["condition"],
            priority=Priority.BLOCKING,
        ))
    except ImportError:
        logger.debug("Coding rituals unavailable")

    # Session: dawn_prime + dusk_consolidate
    try:
        from ..ritual.rituals.session import (
            DAWN_PRIME,
            DAWN_PRIME_BINDING_CONFIG,
            DUSK_CONSOLIDATE,
            DUSK_CONSOLIDATE_BINDING_CONFIG,
        )
        bindings.append(HookBinding(
            ritual=DAWN_PRIME,
            trigger=DAWN_PRIME_BINDING_CONFIG["trigger"],
            condition=DAWN_PRIME_BINDING_CONFIG["condition"],
            priority=Priority.BLOCKING,
        ))
        bindings.append(HookBinding(
            ritual=DUSK_CONSOLIDATE,
            trigger=DUSK_CONSOLIDATE_BINDING_CONFIG["trigger"],
            condition=DUSK_CONSOLIDATE_BINDING_CONFIG["condition"],
            priority=Priority.DEFERRED,
        ))
    except ImportError:
        logger.debug("Session rituals unavailable")

    # Memory: memory_write_guard
    try:
        from ..ritual.rituals.memory import (
            MEMORY_WRITE_GUARD,
            MEMORY_WRITE_GUARD_BINDING_CONFIG,
        )
        bindings.append(HookBinding(
            ritual=MEMORY_WRITE_GUARD,
            trigger=MEMORY_WRITE_GUARD_BINDING_CONFIG["trigger"],
            condition=MEMORY_WRITE_GUARD_BINDING_CONFIG["condition"],
            priority=Priority.BLOCKING,
        ))
    except ImportError:
        logger.debug("Memory rituals unavailable")

    # Meta: ritual_health_audit
    try:
        from ..ritual.meta import (
            RITUAL_HEALTH_AUDIT,
            RITUAL_HEALTH_AUDIT_BINDING_CONFIG,
        )
        bindings.append(HookBinding(
            ritual=RITUAL_HEALTH_AUDIT,
            trigger=RITUAL_HEALTH_AUDIT_BINDING_CONFIG["trigger"],
            condition=RITUAL_HEALTH_AUDIT_BINDING_CONFIG["condition"],
            priority=Priority.DEFERRED,
        ))
    except ImportError:
        logger.debug("Meta rituals unavailable")

    return bindings


class RitualLayer:
    """
    Runtime layer that manages the Ritual Grammar system.

    Composes: RitualRegistry + RitualTrigger + RitualJournal
    """
    name = "ritual"

    # Cosmic event → breathing rhythm modifiers
    _COSMIC_MODIFIERS: dict[str, dict[str, float]] = {
        "full_moon": {"breathing_rate": 1.2},
        "new_moon": {"breathing_rate": 0.8},
        "summer_solstice": {"pressure_threshold": 0.9},
        "winter_solstice": {"pressure_threshold": 0.9},
        "vernal_equinox": {"pressure_threshold": 0.9},
        "autumnal_equinox": {"pressure_threshold": 0.9},
    }

    def __init__(
        self,
        registry: RitualRegistry | None = None,
        journal: RitualJournal | None = None,
        bindings: list[HookBinding] | None = None,
        register_defaults: bool = True,
        enable_cosmic: bool = True,
    ):
        self._journal = journal
        self._registry = registry or RitualRegistry(journal=journal)
        self._custom_bindings = bindings or []
        self._register_defaults = register_defaults
        self._enable_cosmic = enable_cosmic
        self._trigger: Any = None  # RitualTrigger, set on start
        self._runtime: Any = None
        self._cosmic_events_fired: list[str] = []
        self._state = LayerState.INIT

    async def start(self, runtime: Any) -> None:
        """Start the ritual layer."""
        try:
            # Register default bindings
            if self._register_defaults:
                for binding in _default_bindings():
                    self._registry.register(binding)

            # Register custom bindings
            for binding in self._custom_bindings:
                self._registry.register(binding)

            # Validate DAG
            self._registry.validate_dag()

            # Wire to bus if available
            if runtime.bus is not None:
                try:
                    from ..bus.adapters.trigger_adapters import RitualTrigger
                    self._trigger = RitualTrigger(
                        bus=runtime.bus,
                        registry=self._registry,
                    )
                    await self._trigger.start()
                except Exception as e:
                    logger.warning("RitualTrigger failed to start: %s", e)

            # Store journal reference from runtime if not provided
            if self._journal is None and hasattr(runtime, 'journal'):
                self._journal = runtime.journal
                # Late-wire journal to registry
                if self._journal is not None and self._registry._journal is None:
                    self._registry._journal = self._journal

            # Subscribe to cosmic cron signals
            self._runtime = runtime
            if self._enable_cosmic and runtime.bus is not None:
                try:
                    from ..bus.signal import SignalKind
                    runtime.bus.on(SignalKind.CRON, self._handle_cosmic)
                except Exception as e:
                    logger.debug("Cosmic subscription failed: %s", e)

            self._state = LayerState.RUNNING
            logger.info(
                "RitualLayer started: %d rituals registered, cosmic=%s",
                len(self._registry._rituals), self._enable_cosmic,
            )

        except Exception as e:
            self._state = LayerState.DEGRADED
            logger.warning("RitualLayer degraded: %s", e)

    async def stop(self) -> None:
        """Stop the ritual layer."""
        if self._trigger is not None:
            try:
                await self._trigger.stop()
            except Exception as e:
                logger.warning("RitualTrigger stop failed: %s", e)

        if self._journal is not None:
            try:
                self._journal.close()
            except Exception as e:
                logger.warning("RitualJournal close failed: %s", e)

        self._state = LayerState.STOPPED

    async def _handle_cosmic(self, signal: Any) -> None:
        """Handle CRON 'cosmic_check' — run RitualScheduler, apply tension."""
        if getattr(signal, "target", "") != "cosmic_check":
            return

        try:
            from hololoom.domain_harness.strands.ritual_scheduler import (
                RitualScheduler,
            )

            scheduler = RitualScheduler()
            fired = scheduler.check_and_execute()

            for ritual in fired:
                rtype = getattr(ritual, "ritual_type", None)
                if rtype is not None:
                    name = rtype.value if hasattr(rtype, "value") else str(rtype)
                    self._cosmic_events_fired.append(name)
                    self._apply_tension_modifier(name)

        except ImportError:
            pass  # domain_harness unavailable — degrade silently
        except Exception as e:
            logger.warning("Cosmic check failed: %s", e)

    def _apply_tension_modifier(self, ritual_name: str) -> None:
        """Apply cosmic tension modifier to ChronoTrigger breathing rhythm."""
        mods = self._COSMIC_MODIFIERS.get(ritual_name, {})
        if not mods or self._runtime is None:
            return

        # Find ChronoTrigger — might be on a BreathingPipeline or direct attribute
        chrono = getattr(self._runtime, "_chrono", None)
        if chrono is None:
            return

        breathing = getattr(chrono, "breathing", None)
        if breathing is None:
            return

        for key, multiplier in mods.items():
            current = getattr(breathing, key, None)
            if current is not None:
                new_val = current * multiplier
                setattr(breathing, key, new_val)
                logger.info(
                    "Cosmic modifier: %s.%s *= %.2f (%.3f -> %.3f)",
                    ritual_name, key, multiplier, current, new_val,
                )

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "registered_rituals": len(self._registry._rituals),
            "bindings_by_event": {
                e.name: len(b)
                for e, b in self._registry._bindings.items()
                if b
            },
            "trigger_active": self._trigger is not None,
            "journal_available": self._journal is not None,
            "cosmic_events": list(self._cosmic_events_fired),
        }

    def state_snapshot(self) -> dict[str, Any]:
        return {"state": self._state.value}

    @property
    def registry(self) -> RitualRegistry:
        return self._registry

    @property
    def state(self) -> LayerState:
        return self._state
