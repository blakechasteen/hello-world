"""
Jenny Lifecycle Manager
========================
Panel lifecycle management for Jenny Generative UI runtime.

Manages the state machine:
    NASCENT → STABLE → DISSOLVING → ARCHIVED
          ↘           ↙
       (timeout/superseded)

Philosophy:
- Panels are disposable by default (NASCENT dissolves after timeout)
- User can pin panels to make them STABLE
- All transitions logged to SpecLedger for provenance
- Memory pressure can force dissolution

References:
- protocols/jenny.py (JennyLifecycleProtocol)
- jenny_spec.py (JennySpec, LifecycleStage, DissolutionTrigger)
- spec_ledger.py (SpecLedger for archival)

Author: HoloLoom Team
Date: 2025-12-01 (Jenny MVP Week 2)
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from .jenny_spec import (
    JennySpec,
    LifecycleStage,
    DissolutionTrigger,
)
from .spec_ledger import SpecLedger

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class PanelNotFoundError(Exception):
    """Raised when a panel is not found in the lifecycle manager."""
    def __init__(self, spec_id: str):
        self.spec_id = spec_id
        super().__init__(f"Panel not found: {spec_id}")


class InvalidTransitionError(Exception):
    """Raised when an invalid lifecycle transition is attempted."""
    def __init__(self, spec_id: str, current_stage: LifecycleStage, target_stage: LifecycleStage):
        self.spec_id = spec_id
        self.current_stage = current_stage
        self.target_stage = target_stage
        super().__init__(
            f"Invalid transition for {spec_id}: {current_stage.value} → {target_stage.value}"
        )


# ============================================================================
# Panel State Tracking
# ============================================================================

@dataclass
class PanelState:
    """
    Internal state tracking for a panel.

    Tracks metadata beyond what's in JennySpec:
    - Spawn timestamp for staleness detection
    - Session ID for multi-session support
    - Transition history for debugging
    """
    spec: JennySpec
    session_id: str
    spawned_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    transition_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Record initial spawn in transition history."""
        self.transition_history.append({
            "from": None,
            "to": self.spec.lifecycle.value,
            "timestamp": self.spawned_at.isoformat(),
            "trigger": None,
        })

    def record_transition(
        self,
        new_stage: LifecycleStage,
        trigger: Optional[DissolutionTrigger] = None
    ) -> None:
        """Record a lifecycle transition."""
        self.transition_history.append({
            "from": self.spec.lifecycle.value,
            "to": new_stage.value,
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger.value if trigger else None,
        })

    @property
    def age_seconds(self) -> float:
        """Get age in seconds since spawn."""
        return (datetime.now() - self.spawned_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds since last access."""
        return (datetime.now() - self.last_accessed).total_seconds()


# ============================================================================
# Lifecycle Manager Base Class
# ============================================================================

class JennyLifecycleManagerBase(ABC):
    """
    Abstract base class for Jenny lifecycle managers.

    Provides common functionality for lifecycle management:
    - State tracking
    - Transition validation
    - Statistics
    """

    VERSION = "1.0.0"

    # Valid state transitions
    VALID_TRANSITIONS = {
        LifecycleStage.NASCENT: {LifecycleStage.STABLE, LifecycleStage.DISSOLVING},
        LifecycleStage.STABLE: {LifecycleStage.DISSOLVING},
        LifecycleStage.DISSOLVING: {LifecycleStage.ARCHIVED},
        LifecycleStage.ARCHIVED: set(),  # Terminal state
        LifecycleStage.SYSTEM: set(),    # No transitions for SYSTEM
    }

    @abstractmethod
    async def spawn(self, spec: JennySpec) -> JennySpec:
        """Spawn a new panel (transition to NASCENT)."""
        ...

    @abstractmethod
    async def pin(self, spec_id: str) -> JennySpec:
        """Pin a panel (transition NASCENT → STABLE)."""
        ...

    @abstractmethod
    async def dissolve(
        self,
        spec_id: str,
        trigger: DissolutionTrigger
    ) -> JennySpec:
        """Dissolve a panel (transition to DISSOLVING)."""
        ...

    @abstractmethod
    async def archive(self, spec_id: str) -> JennySpec:
        """Archive a panel (transition DISSOLVING → ARCHIVED)."""
        ...

    @abstractmethod
    async def get_active_panels(
        self,
        session_id: Optional[str] = None
    ) -> List[JennySpec]:
        """Get all active panels (NASCENT or STABLE)."""
        ...

    @abstractmethod
    async def get_panel(self, spec_id: str) -> Optional[JennySpec]:
        """Get a panel by ID."""
        ...

    @abstractmethod
    async def cleanup_stale(self, max_age_seconds: int = 300) -> List[str]:
        """Clean up stale NASCENT panels."""
        ...

    @abstractmethod
    async def handle_memory_pressure(
        self,
        target_panel_count: int
    ) -> List[str]:
        """Handle memory pressure by dissolving low-priority panels."""
        ...

    def _validate_transition(
        self,
        spec: JennySpec,
        target_stage: LifecycleStage
    ) -> None:
        """
        Validate a lifecycle transition.

        Raises:
            InvalidTransitionError: If transition is not valid
        """
        valid_targets = self.VALID_TRANSITIONS.get(spec.lifecycle, set())
        if target_stage not in valid_targets:
            raise InvalidTransitionError(spec.spec_id, spec.lifecycle, target_stage)


# ============================================================================
# In-Memory Lifecycle Manager
# ============================================================================

class InMemoryLifecycleManager(JennyLifecycleManagerBase):
    """
    In-memory lifecycle manager for single-session use.

    Features:
    - Fast panel tracking (O(1) lookups)
    - Session-based filtering
    - Automatic staleness cleanup
    - Memory pressure handling
    - Optional SpecLedger integration for archival

    Usage:
        manager = InMemoryLifecycleManager(spec_ledger=ledger)

        # Spawn panel
        spec = await manager.spawn(my_spec)

        # Pin to make stable
        spec = await manager.pin(spec.spec_id)

        # Dissolve when done
        spec = await manager.dissolve(spec.spec_id, DissolutionTrigger.MANUAL)

        # Archive after animation completes
        spec = await manager.archive(spec.spec_id)
    """

    def __init__(
        self,
        spec_ledger: Optional[SpecLedger] = None,
        default_session_id: str = "default",
        max_active_panels: int = 100,
    ):
        """
        Initialize the lifecycle manager.

        Args:
            spec_ledger: Optional SpecLedger for archival logging
            default_session_id: Default session ID for panels
            max_active_panels: Maximum active panels before memory pressure
        """
        # Panel state storage
        self._panels: Dict[str, PanelState] = {}

        # Indexes for fast queries
        self._by_session: Dict[str, Set[str]] = defaultdict(set)
        self._by_lifecycle: Dict[LifecycleStage, Set[str]] = defaultdict(set)

        # Configuration
        self.spec_ledger = spec_ledger
        self.default_session_id = default_session_id
        self.max_active_panels = max_active_panels

        # Statistics
        self._total_spawned = 0
        self._total_pinned = 0
        self._total_dissolved = 0
        self._total_archived = 0

        logger.info(
            f"InMemoryLifecycleManager initialized "
            f"(max_active={max_active_panels}, ledger={'enabled' if spec_ledger else 'disabled'})"
        )

    # ========================================================================
    # Core Lifecycle Methods
    # ========================================================================

    async def spawn(self, spec: JennySpec) -> JennySpec:
        """
        Spawn a new panel (transition to NASCENT).

        Args:
            spec: Panel specification to spawn

        Returns:
            Updated spec with spawn metadata
        """
        # Ensure spec is in NASCENT state
        if spec.lifecycle != LifecycleStage.NASCENT:
            spec = spec.with_lifecycle(LifecycleStage.NASCENT)

        # Create panel state
        session_id = spec.metadata.get("session_id", self.default_session_id)
        state = PanelState(
            spec=spec,
            session_id=session_id,
        )

        # Store state
        self._panels[spec.spec_id] = state
        self._by_session[session_id].add(spec.spec_id)
        self._by_lifecycle[LifecycleStage.NASCENT].add(spec.spec_id)

        # Update statistics
        self._total_spawned += 1

        # Log to ledger if available
        if self.spec_ledger:
            await self.spec_ledger.log_spec(
                spec=spec,
                spacetime_id=spec.spacetime_id,
                session_id=session_id,
            )

        logger.debug(f"Spawned panel: {spec.spec_id} (session: {session_id})")

        # Check memory pressure
        active_count = len(self._by_lifecycle[LifecycleStage.NASCENT]) + \
                       len(self._by_lifecycle[LifecycleStage.STABLE])
        if active_count > self.max_active_panels:
            logger.warning(f"Memory pressure: {active_count} active panels > {self.max_active_panels}")
            await self.handle_memory_pressure(self.max_active_panels)

        return spec

    async def pin(self, spec_id: str) -> JennySpec:
        """
        Pin a panel (transition NASCENT → STABLE).

        Args:
            spec_id: ID of panel to pin

        Returns:
            Updated spec with STABLE lifecycle

        Raises:
            PanelNotFoundError: If panel doesn't exist
            InvalidTransitionError: If panel not in NASCENT state
        """
        state = self._panels.get(spec_id)
        if not state:
            raise PanelNotFoundError(spec_id)

        # Validate transition
        self._validate_transition(state.spec, LifecycleStage.STABLE)

        # Record transition
        state.record_transition(LifecycleStage.STABLE)

        # Update spec (immutable update)
        new_spec = state.spec.with_lifecycle(LifecycleStage.STABLE)
        state.spec = new_spec
        state.last_accessed = datetime.now()

        # Update indexes
        self._by_lifecycle[LifecycleStage.NASCENT].discard(spec_id)
        self._by_lifecycle[LifecycleStage.STABLE].add(spec_id)

        # Update statistics
        self._total_pinned += 1

        # Log to ledger
        if self.spec_ledger:
            await self.spec_ledger.log_lifecycle_transition(
                spec_id=spec_id,
                new_stage=LifecycleStage.STABLE,
            )

        logger.debug(f"Pinned panel: {spec_id}")
        return new_spec

    async def dissolve(
        self,
        spec_id: str,
        trigger: DissolutionTrigger
    ) -> JennySpec:
        """
        Dissolve a panel (transition to DISSOLVING).

        Args:
            spec_id: ID of panel to dissolve
            trigger: What caused dissolution

        Returns:
            Updated spec with DISSOLVING lifecycle

        Raises:
            PanelNotFoundError: If panel doesn't exist
        """
        state = self._panels.get(spec_id)
        if not state:
            raise PanelNotFoundError(spec_id)

        # SYSTEM panels cannot be dissolved
        if state.spec.lifecycle == LifecycleStage.SYSTEM:
            logger.warning(f"Cannot dissolve SYSTEM panel: {spec_id}")
            return state.spec

        # Validate transition (allow from NASCENT or STABLE)
        if state.spec.lifecycle not in (LifecycleStage.NASCENT, LifecycleStage.STABLE):
            raise InvalidTransitionError(spec_id, state.spec.lifecycle, LifecycleStage.DISSOLVING)

        # Record transition
        state.record_transition(LifecycleStage.DISSOLVING, trigger)

        # Update spec
        new_spec = state.spec.with_lifecycle(LifecycleStage.DISSOLVING, trigger)
        state.spec = new_spec

        # Update indexes
        self._by_lifecycle[LifecycleStage.NASCENT].discard(spec_id)
        self._by_lifecycle[LifecycleStage.STABLE].discard(spec_id)
        self._by_lifecycle[LifecycleStage.DISSOLVING].add(spec_id)

        # Update statistics
        self._total_dissolved += 1

        # Log to ledger
        if self.spec_ledger:
            await self.spec_ledger.log_lifecycle_transition(
                spec_id=spec_id,
                new_stage=LifecycleStage.DISSOLVING,
                trigger=trigger,
            )

        logger.debug(f"Dissolved panel: {spec_id} (trigger: {trigger.value})")
        return new_spec

    async def archive(self, spec_id: str) -> JennySpec:
        """
        Archive a panel (transition DISSOLVING → ARCHIVED).

        Called after dissolution animation completes.

        Args:
            spec_id: ID of panel to archive

        Returns:
            Archived spec
        """
        state = self._panels.get(spec_id)
        if not state:
            raise PanelNotFoundError(spec_id)

        # Validate transition
        if state.spec.lifecycle != LifecycleStage.DISSOLVING:
            raise InvalidTransitionError(spec_id, state.spec.lifecycle, LifecycleStage.ARCHIVED)

        # Record transition
        state.record_transition(LifecycleStage.ARCHIVED)

        # Update spec
        new_spec = state.spec.with_lifecycle(LifecycleStage.ARCHIVED)
        state.spec = new_spec

        # Update indexes
        self._by_lifecycle[LifecycleStage.DISSOLVING].discard(spec_id)
        self._by_lifecycle[LifecycleStage.ARCHIVED].add(spec_id)

        # Remove from active storage (keep archived reference in ledger)
        session_id = state.session_id
        self._by_session[session_id].discard(spec_id)
        del self._panels[spec_id]

        # Update statistics
        self._total_archived += 1

        # Log to ledger
        if self.spec_ledger:
            await self.spec_ledger.log_lifecycle_transition(
                spec_id=spec_id,
                new_stage=LifecycleStage.ARCHIVED,
            )

        logger.debug(f"Archived panel: {spec_id}")
        return new_spec

    # ========================================================================
    # Query Methods
    # ========================================================================

    async def get_active_panels(
        self,
        session_id: Optional[str] = None
    ) -> List[JennySpec]:
        """
        Get all active panels (NASCENT or STABLE).

        Args:
            session_id: Optional session filter

        Returns:
            List of active panels
        """
        # Get active spec IDs
        active_ids = (
            self._by_lifecycle[LifecycleStage.NASCENT] |
            self._by_lifecycle[LifecycleStage.STABLE]
        )

        # Filter by session if specified
        if session_id:
            session_ids = self._by_session.get(session_id, set())
            active_ids = active_ids & session_ids

        # Get specs and update access time
        result = []
        for spec_id in active_ids:
            state = self._panels.get(spec_id)
            if state:
                state.last_accessed = datetime.now()
                result.append(state.spec)

        return result

    async def get_panel(self, spec_id: str) -> Optional[JennySpec]:
        """
        Get a panel by ID (any lifecycle stage).

        Args:
            spec_id: Panel ID

        Returns:
            Panel if found, None otherwise
        """
        state = self._panels.get(spec_id)
        if state:
            state.last_accessed = datetime.now()
            return state.spec

        # Check ledger for archived panels
        if self.spec_ledger:
            return await self.spec_ledger.get_spec(spec_id)

        return None

    # ========================================================================
    # Cleanup Methods
    # ========================================================================

    async def cleanup_stale(self, max_age_seconds: int = 300) -> List[str]:
        """
        Clean up stale NASCENT panels.

        Dissolves unpinned panels that have been NASCENT
        longer than max_age_seconds.

        Args:
            max_age_seconds: Maximum age before auto-dissolution

        Returns:
            List of dissolved panel IDs
        """
        dissolved_ids = []

        # Find stale NASCENT panels
        nascent_ids = list(self._by_lifecycle[LifecycleStage.NASCENT])

        for spec_id in nascent_ids:
            state = self._panels.get(spec_id)
            if state and state.age_seconds > max_age_seconds:
                try:
                    await self.dissolve(spec_id, DissolutionTrigger.TIMEOUT)
                    dissolved_ids.append(spec_id)
                except Exception as e:
                    logger.warning(f"Failed to dissolve stale panel {spec_id}: {e}")

        if dissolved_ids:
            logger.info(f"Cleaned up {len(dissolved_ids)} stale panels")

        return dissolved_ids

    async def handle_memory_pressure(
        self,
        target_panel_count: int
    ) -> List[str]:
        """
        Handle memory pressure by dissolving low-priority panels.

        Prioritization for dissolution:
        1. NASCENT panels (never pinned)
        2. Oldest STABLE panels
        3. Never dissolve SYSTEM panels

        Args:
            target_panel_count: Target number of panels to keep

        Returns:
            List of dissolved panel IDs
        """
        dissolved_ids = []

        # Count active panels
        active_count = (
            len(self._by_lifecycle[LifecycleStage.NASCENT]) +
            len(self._by_lifecycle[LifecycleStage.STABLE])
        )

        if active_count <= target_panel_count:
            return dissolved_ids

        to_dissolve = active_count - target_panel_count

        # Priority 1: Dissolve NASCENT panels (oldest first)
        nascent_panels = [
            (spec_id, self._panels[spec_id])
            for spec_id in self._by_lifecycle[LifecycleStage.NASCENT]
            if spec_id in self._panels
        ]
        nascent_panels.sort(key=lambda x: x[1].spawned_at)

        for spec_id, _ in nascent_panels:
            if len(dissolved_ids) >= to_dissolve:
                break
            try:
                await self.dissolve(spec_id, DissolutionTrigger.MEMORY)
                dissolved_ids.append(spec_id)
            except Exception as e:
                logger.warning(f"Failed to dissolve {spec_id}: {e}")

        # Priority 2: Dissolve STABLE panels (oldest first) if still over limit
        if len(dissolved_ids) < to_dissolve:
            stable_panels = [
                (spec_id, self._panels[spec_id])
                for spec_id in self._by_lifecycle[LifecycleStage.STABLE]
                if spec_id in self._panels
            ]
            stable_panels.sort(key=lambda x: x[1].spawned_at)

            for spec_id, state in stable_panels:
                # Skip SYSTEM panels
                if state.spec.lifecycle == LifecycleStage.SYSTEM:
                    continue

                if len(dissolved_ids) >= to_dissolve:
                    break
                try:
                    await self.dissolve(spec_id, DissolutionTrigger.MEMORY)
                    dissolved_ids.append(spec_id)
                except Exception as e:
                    logger.warning(f"Failed to dissolve {spec_id}: {e}")

        if dissolved_ids:
            logger.warning(
                f"Memory pressure: dissolved {len(dissolved_ids)} panels "
                f"(target: {target_panel_count}, was: {active_count})"
            )

        return dissolved_ids

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get lifecycle manager statistics.

        Returns:
            Statistics dict
        """
        return {
            "total_spawned": self._total_spawned,
            "total_pinned": self._total_pinned,
            "total_dissolved": self._total_dissolved,
            "total_archived": self._total_archived,
            "active_nascent": len(self._by_lifecycle[LifecycleStage.NASCENT]),
            "active_stable": len(self._by_lifecycle[LifecycleStage.STABLE]),
            "dissolving": len(self._by_lifecycle[LifecycleStage.DISSOLVING]),
            "total_panels": len(self._panels),
            "sessions": len(self._by_session),
            "max_active_panels": self.max_active_panels,
            "ledger_enabled": self.spec_ledger is not None,
        }

    async def get_session_panels(self, session_id: str) -> List[JennySpec]:
        """
        Get all panels for a specific session.

        Args:
            session_id: Session ID

        Returns:
            List of panels in session
        """
        spec_ids = self._by_session.get(session_id, set())
        return [
            self._panels[spec_id].spec
            for spec_id in spec_ids
            if spec_id in self._panels
        ]

    async def dissolve_all_session(
        self,
        session_id: str,
        trigger: DissolutionTrigger = DissolutionTrigger.ORPHAN
    ) -> List[str]:
        """
        Dissolve all panels in a session.

        Useful for session cleanup.

        Args:
            session_id: Session ID to clean up
            trigger: Dissolution trigger reason

        Returns:
            List of dissolved panel IDs
        """
        dissolved_ids = []
        spec_ids = list(self._by_session.get(session_id, set()))

        for spec_id in spec_ids:
            state = self._panels.get(spec_id)
            if state and state.spec.lifecycle in (LifecycleStage.NASCENT, LifecycleStage.STABLE):
                try:
                    await self.dissolve(spec_id, trigger)
                    dissolved_ids.append(spec_id)
                except Exception as e:
                    logger.warning(f"Failed to dissolve {spec_id}: {e}")

        logger.info(f"Dissolved {len(dissolved_ids)} panels from session {session_id}")
        return dissolved_ids


# ============================================================================
# Background Cleanup Task
# ============================================================================

async def run_cleanup_loop(
    manager: InMemoryLifecycleManager,
    interval_seconds: int = 60,
    max_age_seconds: int = 300,
) -> None:
    """
    Run periodic cleanup task.

    Args:
        manager: Lifecycle manager to clean up
        interval_seconds: How often to run cleanup
        max_age_seconds: Maximum age for NASCENT panels
    """
    logger.info(f"Starting cleanup loop (interval={interval_seconds}s, max_age={max_age_seconds}s)")

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            dissolved = await manager.cleanup_stale(max_age_seconds)
            if dissolved:
                logger.debug(f"Cleanup dissolved {len(dissolved)} stale panels")
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_lifecycle_manager(
    spec_ledger: Optional[SpecLedger] = None,
    default_session_id: str = "default",
    max_active_panels: int = 100,
) -> InMemoryLifecycleManager:
    """
    Create an in-memory lifecycle manager.

    Args:
        spec_ledger: Optional SpecLedger for archival
        default_session_id: Default session ID
        max_active_panels: Maximum active panels

    Returns:
        Configured lifecycle manager
    """
    return InMemoryLifecycleManager(
        spec_ledger=spec_ledger,
        default_session_id=default_session_id,
        max_active_panels=max_active_panels,
    )


def get_default_lifecycle_manager() -> InMemoryLifecycleManager:
    """
    Get a default lifecycle manager (no ledger).

    Returns:
        Default lifecycle manager
    """
    return InMemoryLifecycleManager()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Exceptions
    'PanelNotFoundError',
    'InvalidTransitionError',
    # State tracking
    'PanelState',
    # Base class
    'JennyLifecycleManagerBase',
    # Implementation
    'InMemoryLifecycleManager',
    # Background task
    'run_cleanup_loop',
    # Factory functions
    'create_lifecycle_manager',
    'get_default_lifecycle_manager',
]
