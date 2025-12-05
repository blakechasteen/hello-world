"""
Skills Protocol for Zero-G App Docking
=======================================

**Created**: November 22, 2025
**Purpose**: Define protocol for skills to dock with Zero-G platform

This protocol enables skills to function as "apps" in Zero-G's App Orbit Layer:
- Skills are shuttles docking to the orbital platform
- Shared WarpSpace and YarnGraph semantic layer
- Event bus for cross-skill communication
- Launch System integration (Preflight → Orbit)

## Architecture

```
Zero-G Platform (Orbital Infrastructure)
├── Loom Core
│   ├── WarpSpace (semantic retrieval)
│   ├── YarnGraph (knowledge graph)
│   └── ResonanceShed (multimodal fusion)
│
├── Launch System
│   ├── Preflight (safety checks) ← skill_tester, skill_security_analyzer
│   ├── Countdown (preparation)
│   ├── Lift-Off (execution)
│   └── Orbit (stable operation) ← continuous_learning_capture
│
└── App Orbit Layer
    ├── Docking Protocol ← THIS FILE
    ├── Event Bus (pub/sub)
    └── Apps (Skills)
        ├── Meta-Skills (self-improvement)
        ├── Domain Skills (task-focused)
        └── Agentic Skills (HoloLoom YAML)
```
"""

from typing import Protocol, Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class SkillLifecycleState(Enum):
    """Skill lifecycle states (mirrors Zero-G launch sequence)."""
    PREFLIGHT = "preflight"      # Safety checks, validation
    COUNTDOWN = "countdown"       # Preparation
    LIFT_OFF = "lift_off"        # Execution starting
    ORBIT = "orbit"              # Stable operation
    EVA = "eva"                  # Manual intervention
    REENTRY = "reentry"          # Shutdown
    LANDED = "landed"            # Complete


class EventType(Enum):
    """Event types for inter-skill communication."""
    SKILL_STARTED = "skill_started"
    SKILL_COMPLETED = "skill_completed"
    SKILL_FAILED = "skill_failed"
    PATTERN_DETECTED = "pattern_detected"
    GAP_IDENTIFIED = "gap_identified"
    SECURITY_ALERT = "security_alert"
    QUALITY_WARNING = "quality_warning"
    CUSTOM = "custom"


class DockingStatus(Enum):
    """Status of skill docking to Zero-G."""
    UNDOCKED = "undocked"
    DOCKING = "docking"
    DOCKED = "docked"
    UNDOCKING = "undocking"
    REJECTED = "rejected"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SkillManifest:
    """
    Manifest for a skill docking to Zero-G.

    This is the "spacecraft specification" - what the skill needs
    and what it provides.
    """
    # Identity
    name: str
    version: str
    category: str  # "meta", "domain", "agentic"

    # Capabilities
    description: str
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)

    # Requirements
    requires_warp_space: bool = False      # Semantic retrieval
    requires_yarn_graph: bool = False      # Knowledge graph
    requires_event_bus: bool = False       # Pub/sub
    requires_mission_control: bool = False  # Monitoring
    requires_tools: List[str] = field(default_factory=list)  # External tools

    # Metadata
    author: str = ""
    created: str = ""
    homepage: str = ""


@dataclass
class SkillEvent:
    """
    Event emitted by a skill.

    Enhanced with workflow support for event chains and correlation.
    """
    event_type: EventType
    skill_name: str
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Event routing
    event_id: str = ""               # Unique event ID (auto-generated)
    topic: str = ""                  # Routing topic (auto-generated if not provided)

    # Workflow support
    correlation_id: Optional[str] = None  # Links events in same workflow
    causation_id: Optional[str] = None    # Parent event ID (what caused this)
    sequence: int = 0                     # Order in event chain


@dataclass
class PreflightCheckResult:
    """Result of preflight safety checks."""
    passed: bool
    checks_run: int
    checks_passed: int
    checks_failed: int
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillExecutionContext:
    """
    Execution context provided to a skill.

    This is what Zero-G provides to the skill during execution:
    - Access to shared infrastructure (WarpSpace, YarnGraph)
    - Event bus for communication
    - Mission Control for monitoring
    """
    # Shared infrastructure
    warp_space: Optional[Any] = None    # Semantic retrieval
    yarn_graph: Optional[Any] = None    # Knowledge graph
    event_bus: Optional[Any] = None     # Pub/sub
    mission_control: Optional[Any] = None  # Monitoring

    # Execution state
    session_id: str = ""
    skill_name: str = ""
    lifecycle_state: SkillLifecycleState = SkillLifecycleState.PREFLIGHT

    # Metadata
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillExecutionResult:
    """Result of skill execution."""
    success: bool
    output: Any
    confidence: float = 0.0

    # Provenance
    lifecycle_states: List[SkillLifecycleState] = field(default_factory=list)
    events_emitted: List[SkillEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Performance
    execution_time_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Protocols
# ============================================================================

class DockableSkill(Protocol):
    """
    Protocol for skills that can dock with Zero-G.

    Any skill implementing this protocol can be loaded into
    Zero-G's App Orbit Layer.
    """

    @property
    def manifest(self) -> SkillManifest:
        """Get skill manifest."""
        ...

    async def preflight(self, context: SkillExecutionContext) -> PreflightCheckResult:
        """
        Run preflight safety checks.

        This is analogous to T-10 to T-0 countdown checks.
        Validates all preconditions are met before execution.

        Args:
            context: Execution context

        Returns:
            Preflight check result
        """
        ...

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: SkillExecutionContext
    ) -> SkillExecutionResult:
        """
        Execute the skill (Lift-Off → Orbit).

        Args:
            parameters: Skill-specific parameters
            context: Execution context

        Returns:
            Execution result
        """
        ...

    async def shutdown(self, context: SkillExecutionContext) -> None:
        """
        Graceful shutdown (Reentry → Landed).

        Args:
            context: Execution context
        """
        ...


class EventBus(Protocol):
    """
    Event bus for inter-skill communication.

    Skills can:
    - Emit events (fire-and-forget)
    - Subscribe to event types
    - React to events from other skills
    """

    async def emit(self, event: SkillEvent) -> None:
        """Emit an event."""
        ...

    async def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[SkillEvent], None]
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function

        Returns:
            Subscription ID
        """
        ...

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events."""
        ...


class MissionControl(Protocol):
    """
    Mission Control for monitoring skill execution.

    Provides:
    - Real-time monitoring
    - Health checks
    - Rollback capability
    - Performance metrics
    """

    async def register_skill(self, manifest: SkillManifest) -> str:
        """Register a skill for monitoring."""
        ...

    async def update_state(
        self,
        skill_name: str,
        state: SkillLifecycleState
    ) -> None:
        """Update skill lifecycle state."""
        ...

    async def report_metrics(
        self,
        skill_name: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Report performance metrics."""
        ...

    async def health_check(self, skill_name: str) -> Dict[str, Any]:
        """Get skill health status."""
        ...


# ============================================================================
# Docking Manager
# ============================================================================

class SkillDockingManager:
    """
    Manages skill docking to Zero-G platform.

    Responsibilities:
    - Validate skill manifests
    - Run preflight checks
    - Provide execution context
    - Manage lifecycle
    - Monitor health
    """

    def __init__(
        self,
        warp_space: Optional[Any] = None,
        yarn_graph: Optional[Any] = None,
        event_bus: Optional[EventBus] = None,
        mission_control: Optional[MissionControl] = None
    ):
        self.warp_space = warp_space
        self.yarn_graph = yarn_graph
        self.event_bus = event_bus
        self.mission_control = mission_control

        self.docked_skills: Dict[str, DockableSkill] = {}
        self.skill_states: Dict[str, DockingStatus] = {}

    async def dock_skill(
        self,
        skill: DockableSkill
    ) -> tuple[bool, str]:
        """
        Dock a skill to Zero-G platform.

        Args:
            skill: Skill to dock

        Returns:
            (success, message)
        """
        manifest = skill.manifest
        skill_name = manifest.name

        logger.info(f"Docking skill: {skill_name}")
        self.skill_states[skill_name] = DockingStatus.DOCKING

        try:
            # Validate manifest
            if not self._validate_manifest(manifest):
                self.skill_states[skill_name] = DockingStatus.REJECTED
                return False, f"Invalid manifest for {skill_name}"

            # Check requirements
            missing = self._check_requirements(manifest)
            if missing:
                self.skill_states[skill_name] = DockingStatus.REJECTED
                return False, f"Missing requirements: {', '.join(missing)}"

            # Run preflight checks
            context = self._create_context(skill_name)
            preflight_result = await skill.preflight(context)

            if not preflight_result.passed:
                self.skill_states[skill_name] = DockingStatus.REJECTED
                failures = ", ".join(preflight_result.failures)
                return False, f"Preflight failed: {failures}"

            # Register with Mission Control
            if self.mission_control:
                await self.mission_control.register_skill(manifest)

            # Dock
            self.docked_skills[skill_name] = skill
            self.skill_states[skill_name] = DockingStatus.DOCKED

            logger.info(f"Successfully docked: {skill_name}")
            return True, f"Skill {skill_name} docked successfully"

        except Exception as e:
            self.skill_states[skill_name] = DockingStatus.REJECTED
            logger.error(f"Error docking {skill_name}: {e}")
            return False, f"Docking error: {e}"

    async def undock_skill(self, skill_name: str) -> tuple[bool, str]:
        """
        Undock a skill from Zero-G platform.

        Args:
            skill_name: Name of skill to undock

        Returns:
            (success, message)
        """
        if skill_name not in self.docked_skills:
            return False, f"Skill {skill_name} not docked"

        logger.info(f"Undocking skill: {skill_name}")
        self.skill_states[skill_name] = DockingStatus.UNDOCKING

        try:
            skill = self.docked_skills[skill_name]
            context = self._create_context(skill_name)
            await skill.shutdown(context)

            del self.docked_skills[skill_name]
            self.skill_states[skill_name] = DockingStatus.UNDOCKED

            logger.info(f"Successfully undocked: {skill_name}")
            return True, f"Skill {skill_name} undocked successfully"

        except Exception as e:
            logger.error(f"Error undocking {skill_name}: {e}")
            return False, f"Undocking error: {e}"

    async def execute_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any]
    ) -> SkillExecutionResult:
        """
        Execute a docked skill.

        Args:
            skill_name: Name of skill to execute
            parameters: Skill parameters

        Returns:
            Execution result
        """
        if skill_name not in self.docked_skills:
            return SkillExecutionResult(
                success=False,
                output=None,
                errors=[f"Skill {skill_name} not docked"]
            )

        skill = self.docked_skills[skill_name]
        context = self._create_context(skill_name)

        try:
            # Update state: Countdown
            context.lifecycle_state = SkillLifecycleState.COUNTDOWN
            if self.mission_control:
                await self.mission_control.update_state(
                    skill_name,
                    SkillLifecycleState.COUNTDOWN
                )

            # Update state: Lift-Off
            context.lifecycle_state = SkillLifecycleState.LIFT_OFF
            if self.mission_control:
                await self.mission_control.update_state(
                    skill_name,
                    SkillLifecycleState.LIFT_OFF
                )

            # Execute
            result = await skill.execute(parameters, context)

            # Update state: Orbit (if successful)
            if result.success:
                context.lifecycle_state = SkillLifecycleState.ORBIT
                if self.mission_control:
                    await self.mission_control.update_state(
                        skill_name,
                        SkillLifecycleState.ORBIT
                    )

            return result

        except Exception as e:
            logger.error(f"Error executing {skill_name}: {e}")
            return SkillExecutionResult(
                success=False,
                output=None,
                errors=[str(e)]
            )

    def _validate_manifest(self, manifest: SkillManifest) -> bool:
        """Validate skill manifest."""
        required_fields = ["name", "version", "category", "description"]
        for field in required_fields:
            if not getattr(manifest, field, None):
                logger.error(f"Missing required field: {field}")
                return False
        return True

    def _check_requirements(self, manifest: SkillManifest) -> List[str]:
        """Check if all requirements are met."""
        missing = []

        if manifest.requires_warp_space and not self.warp_space:
            missing.append("WarpSpace")

        if manifest.requires_yarn_graph and not self.yarn_graph:
            missing.append("YarnGraph")

        if manifest.requires_event_bus and not self.event_bus:
            missing.append("EventBus")

        if manifest.requires_mission_control and not self.mission_control:
            missing.append("MissionControl")

        return missing

    def _create_context(self, skill_name: str) -> SkillExecutionContext:
        """Create execution context for a skill."""
        return SkillExecutionContext(
            warp_space=self.warp_space,
            yarn_graph=self.yarn_graph,
            event_bus=self.event_bus,
            mission_control=self.mission_control,
            session_id=f"{skill_name}-{datetime.now().isoformat()}",
            skill_name=skill_name,
            lifecycle_state=SkillLifecycleState.PREFLIGHT,
            timestamp=datetime.now().isoformat()
        )

    def get_docked_skills(self) -> List[str]:
        """Get list of docked skills."""
        return list(self.docked_skills.keys())

    def get_skill_status(self, skill_name: str) -> Optional[DockingStatus]:
        """Get skill docking status."""
        return self.skill_states.get(skill_name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'SkillLifecycleState',
    'EventType',
    'DockingStatus',

    # Data Structures
    'SkillManifest',
    'SkillEvent',
    'PreflightCheckResult',
    'SkillExecutionContext',
    'SkillExecutionResult',

    # Protocols
    'DockableSkill',
    'EventBus',
    'MissionControl',

    # Manager
    'SkillDockingManager',
]
