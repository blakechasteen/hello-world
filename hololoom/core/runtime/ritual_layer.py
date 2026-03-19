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

    def __init__(
        self,
        registry: RitualRegistry | None = None,
        journal: RitualJournal | None = None,
        bindings: list[HookBinding] | None = None,
        register_defaults: bool = True,
    ):
        self._journal = journal
        self._registry = registry or RitualRegistry(journal=journal)
        self._custom_bindings = bindings or []
        self._register_defaults = register_defaults
        self._trigger: Any = None  # RitualTrigger, set on start
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

            self._state = LayerState.RUNNING
            logger.info(
                "RitualLayer started: %d rituals registered",
                len(self._registry._rituals),
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
        }

    def state_snapshot(self) -> dict[str, Any]:
        return {"state": self._state.value}

    @property
    def registry(self) -> RitualRegistry:
        return self._registry

    @property
    def state(self) -> LayerState:
        return self._state
