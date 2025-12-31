"""
Ritual Phase Base Class
=======================

Created: December 30, 2025
Purpose: Base class for ritual phase skills implementing DockableSkill protocol

Each ritual phase (AWAKEN, PLAN, IMPLEMENT, REVIEW, REFLECT) implements this
protocol for consistent lifecycle management, preflight checks, and execution.

Follows "software for agents" paradigm - phases are composable primitives
that can be discovered and invoked by agents autonomously.
"""

from typing import Protocol, Dict, List, Any, Optional, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import logging

# Import from skills protocol
import sys
sys.path.insert(0, "c:/Users/blake/OneDrive/Documents/mythRL/.claude/skills")
from protocol import (
    SkillManifest,
    SkillExecutionContext,
    SkillExecutionResult,
    PreflightCheckResult,
    SkillLifecycleState,
    EventType,
    SkillEvent,
)

# Import ritual-specific types
from ..ritual_events import (
    RitualPhase,
    RitualEventType,
    DecisionType,
    RitualContext,
    RitualDecision,
    create_ritual_event,
    phase_started,
    phase_completed,
    decision_required,
    decision_made,
    PHASE_DECISION_OPTIONS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Agent Capability Enum (for registry integration)
# ============================================================================

class AgentCapability:
    """Capabilities that ritual phase agents can provide.

    Enables O(1) lookup in AgentRegistry for capability-based routing.
    """
    # Ritual-specific capabilities
    CONTEXT_RESTORATION = "context_restoration"      # AWAKEN
    PLANNING = "planning"                            # PLAN
    CODE_ASSISTANCE = "code_assistance"              # IMPLEMENT
    QUALITY_ASSURANCE = "quality_assurance"          # REVIEW
    KNOWLEDGE_CONSOLIDATION = "knowledge_consolidation"  # REFLECT

    # Cross-cutting capabilities
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_STORAGE = "memory_storage"
    DECISION_SUPPORT = "decision_support"


# ============================================================================
# Phase-Specific Data Structures
# ============================================================================

@dataclass
class PhasePreflightResult(PreflightCheckResult):
    """
    Extended preflight result with ritual-specific fields.

    Adds memory_context for informed execution decisions.
    """
    # Memory context gathered during preflight
    memory_context: Dict[str, Any] = field(default_factory=dict)

    # Ritual context reference
    ritual_context: Optional[RitualContext] = None

    # Estimated execution time (ms)
    estimated_duration_ms: float = 0.0


@dataclass
class PhaseResult:
    """
    Result of ritual phase execution.

    Extends SkillExecutionResult concept with decision point support
    for semi-automatic workflow with human guidance.
    """
    # Core result
    success: bool
    output: Any
    confidence: float = 0.0

    # Memory operations performed
    memories_stored: List[str] = field(default_factory=list)
    memories_retrieved: List[str] = field(default_factory=list)

    # Decision point support (semi-automatic workflow)
    decision_needed: bool = False
    decision_options: List[str] = field(default_factory=list)
    suggested_decision: str = ""
    decision_reasoning: str = ""

    # Phase-specific metadata
    phase: Optional[RitualPhase] = None
    duration_ms: float = 0.0
    events_emitted: List[str] = field(default_factory=list)

    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Full metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_skill_result(self) -> SkillExecutionResult:
        """Convert to standard SkillExecutionResult for protocol compliance."""
        return SkillExecutionResult(
            success=self.success,
            output=self.output,
            confidence=self.confidence,
            lifecycle_states=[],  # Managed by orchestrator
            events_emitted=[],    # Converted separately
            errors=self.errors,
            warnings=self.warnings,
            execution_time_ms=self.duration_ms,
            metadata={
                "phase": self.phase.value if self.phase else None,
                "decision_needed": self.decision_needed,
                "decision_options": self.decision_options,
                "suggested_decision": self.suggested_decision,
                "memories_stored": self.memories_stored,
                "memories_retrieved": self.memories_retrieved,
                **self.metadata
            }
        )


@dataclass
class PhaseExecutionContext:
    """
    Extended execution context for ritual phases.

    Wraps SkillExecutionContext with ritual-specific additions.
    """
    # Base context (from protocol)
    skill_context: SkillExecutionContext

    # Ritual context (shared across phases)
    ritual_context: RitualContext

    # Preflight results (for informed execution)
    preflight_result: Optional[PhasePreflightResult] = None

    # Handoff context from previous phase (MI-filtered)
    handoff_context: str = ""
    handoff_tokens: int = 0

    # Auto-decision mode (for agent invocation)
    auto_decisions: bool = False

    # Feature name being worked on
    feature_name: str = ""

    @property
    def warp_space(self):
        return self.skill_context.warp_space

    @property
    def yarn_graph(self):
        return self.skill_context.yarn_graph

    @property
    def event_bus(self):
        return self.skill_context.event_bus

    @property
    def mission_control(self):
        return self.skill_context.mission_control


# ============================================================================
# Ritual Phase Protocol
# ============================================================================

@runtime_checkable
class RitualPhaseSkill(Protocol):
    """
    Protocol for ritual phase skills.

    Extends DockableSkill protocol with ritual-specific methods.
    Phases implementing this protocol can be:
    - Invoked by humans via /ritual-<phase> commands
    - Invoked by agents via capability registry
    - Orchestrated by the /ritual coordinator

    Each phase follows the lifecycle:
    PREFLIGHT → COUNTDOWN → LIFT_OFF → ORBIT → REENTRY → LANDED
    """

    @property
    def manifest(self) -> SkillManifest:
        """Skill manifest for docking."""
        ...

    @property
    def phase(self) -> RitualPhase:
        """Which ritual phase this skill implements."""
        ...

    @property
    def capabilities(self) -> List[str]:
        """Agent capabilities this phase provides."""
        ...

    async def preflight(
        self,
        context: PhaseExecutionContext
    ) -> PhasePreflightResult:
        """
        Run preflight checks before phase execution.

        Checks:
        - Memory system available
        - Required context present
        - No conflicting rituals in progress

        Returns PhasePreflightResult with memory context for informed decisions.
        """
        ...

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: PhaseExecutionContext
    ) -> PhaseResult:
        """
        Execute the phase.

        Uses MemoryInformedDecisionEngine to enhance parameters
        with relevant memories before execution.

        Returns PhaseResult with decision_needed if human guidance required.
        """
        ...

    async def shutdown(self, context: PhaseExecutionContext) -> None:
        """Graceful cleanup after phase completion."""
        ...

    def get_decision_options(self) -> List[str]:
        """Get available decision options for this phase."""
        ...


# ============================================================================
# Abstract Base Implementation
# ============================================================================

class AbstractRitualPhase(ABC):
    """
    Abstract base class for ritual phases.

    Provides common functionality:
    - Manifest generation
    - Preflight check framework
    - Event emission helpers
    - Memory operation wrappers
    - Decision point handling

    Subclasses implement phase-specific logic.
    """

    def __init__(
        self,
        phase: RitualPhase,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        capabilities: Optional[List[str]] = None,
    ):
        self._phase = phase
        self._name = name
        self._version = version
        self._description = description
        self._capabilities = capabilities or []
        self._manifest: Optional[SkillManifest] = None

    @property
    def manifest(self) -> SkillManifest:
        """Generate skill manifest."""
        if self._manifest is None:
            self._manifest = SkillManifest(
                name=self._name,
                version=self._version,
                category="domain",
                description=self._description,
                tags=["ritual", self._phase.value, "workflow", "memory"],
                capabilities=self._capabilities,
                requires_warp_space=True,   # For semantic retrieval
                requires_yarn_graph=True,    # For knowledge graph
                requires_event_bus=True,     # For pub/sub
                requires_mission_control=False,  # Optional monitoring
                requires_tools=["holoLoom-memory"],  # MCP tools
            )
        return self._manifest

    @property
    def phase(self) -> RitualPhase:
        """Which ritual phase this skill implements."""
        return self._phase

    @property
    def capabilities(self) -> List[str]:
        """Agent capabilities this phase provides."""
        return self._capabilities

    # ========================================================================
    # Preflight Framework
    # ========================================================================

    async def preflight(
        self,
        context: PhaseExecutionContext
    ) -> PhasePreflightResult:
        """
        Run preflight checks.

        Default implementation checks:
        1. Memory system available
        2. No conflicting rituals
        3. Calls phase-specific preflight checks
        """
        checks = []
        failures = []
        warnings = []
        memory_context = {}

        # Check 1: Memory system available
        if context.warp_space is not None:
            checks.append("memory_available")
        else:
            warnings.append("WarpSpace not available - limited functionality")

        # Check 2: No conflicting rituals (would come from context)
        active_rituals = context.ritual_context.metadata.get("active_rituals", [])
        current_ritual = context.ritual_context.ritual_id
        conflicts = [r for r in active_rituals if r != current_ritual]

        if not conflicts:
            checks.append("no_conflicts")
        else:
            failures.append(f"Conflicting ritual in progress: {conflicts[0]}")

        # Check 3: Phase-specific preflight
        phase_result = await self._phase_specific_preflight(context)
        checks.extend(phase_result.get("checks", []))
        failures.extend(phase_result.get("failures", []))
        warnings.extend(phase_result.get("warnings", []))
        memory_context.update(phase_result.get("memory_context", {}))

        return PhasePreflightResult(
            passed=len(failures) == 0,
            checks_run=len(checks) + len(failures),
            checks_passed=len(checks),
            checks_failed=len(failures),
            failures=failures,
            warnings=warnings,
            details={"phase": self._phase.value},
            memory_context=memory_context,
            ritual_context=context.ritual_context,
            estimated_duration_ms=self._estimate_duration(context),
        )

    @abstractmethod
    async def _phase_specific_preflight(
        self,
        context: PhaseExecutionContext
    ) -> Dict[str, Any]:
        """
        Phase-specific preflight checks.

        Returns dict with:
        - checks: List of passed check names
        - failures: List of failure messages
        - warnings: List of warning messages
        - memory_context: Dict of pre-fetched memory data
        """
        ...

    def _estimate_duration(self, context: PhaseExecutionContext) -> float:
        """Estimate execution duration in ms. Override for phase-specific estimates."""
        return 5000.0  # Default 5 seconds

    # ========================================================================
    # Execution Framework
    # ========================================================================

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: PhaseExecutionContext
    ) -> PhaseResult:
        """
        Execute the phase.

        Wraps phase-specific execution with:
        1. Event emission (phase started)
        2. Memory-informed parameter enhancement
        3. Phase-specific execution
        4. Decision point handling
        5. Event emission (phase completed/decision required)
        """
        start_time = datetime.now()
        events_emitted = []

        try:
            # Emit phase started event
            if context.event_bus:
                event = phase_started(context.ritual_context, self._phase)
                await context.event_bus.emit(SkillEvent(
                    event_type=EventType.CUSTOM,
                    skill_name=self._name,
                    timestamp=datetime.now().isoformat(),
                    payload=event,
                    correlation_id=context.ritual_context.ritual_id,
                ))
                events_emitted.append(f"phase.started.{self._phase.value}")

            # Enhance parameters with memory context
            enhanced_params = await self._enhance_with_memory(
                parameters,
                context
            )

            # Execute phase-specific logic
            result = await self._phase_specific_execute(enhanced_params, context)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() * 1000
            result.duration_ms = duration
            result.phase = self._phase
            result.events_emitted = events_emitted

            # Handle decision point
            if result.decision_needed and not context.auto_decisions:
                # Emit decision required event
                if context.event_bus:
                    event = decision_required(
                        context.ritual_context,
                        result.decision_options
                    )
                    await context.event_bus.emit(SkillEvent(
                        event_type=EventType.CUSTOM,
                        skill_name=self._name,
                        timestamp=datetime.now().isoformat(),
                        payload=event,
                        correlation_id=context.ritual_context.ritual_id,
                    ))
                    events_emitted.append("decision.required")
            else:
                # Auto-accept suggested decision or emit completion
                if result.decision_needed and context.auto_decisions:
                    # Record auto-decision
                    context.ritual_context.record_decision(
                        self._phase.value,
                        result.suggested_decision,
                        {"auto": True, "reasoning": result.decision_reasoning}
                    )

                # Emit phase completed
                if context.event_bus:
                    event = phase_completed(
                        context.ritual_context,
                        self._phase,
                        result.success,
                        result.confidence
                    )
                    await context.event_bus.emit(SkillEvent(
                        event_type=EventType.CUSTOM,
                        skill_name=self._name,
                        timestamp=datetime.now().isoformat(),
                        payload=event,
                        correlation_id=context.ritual_context.ritual_id,
                    ))
                    events_emitted.append(f"phase.completed.{self._phase.value}")

            return result

        except Exception as e:
            logger.error(f"Phase {self._phase.value} execution failed: {e}")
            duration = (datetime.now() - start_time).total_seconds() * 1000

            return PhaseResult(
                success=False,
                output=None,
                confidence=0.0,
                phase=self._phase,
                duration_ms=duration,
                errors=[str(e)],
                events_emitted=events_emitted,
            )

    @abstractmethod
    async def _phase_specific_execute(
        self,
        parameters: Dict[str, Any],
        context: PhaseExecutionContext
    ) -> PhaseResult:
        """
        Phase-specific execution logic.

        Implement in subclass.
        """
        ...

    async def _enhance_with_memory(
        self,
        parameters: Dict[str, Any],
        context: PhaseExecutionContext
    ) -> Dict[str, Any]:
        """
        Enhance parameters with memory context.

        Uses preflight memory_context and handoff context.
        Override for phase-specific enhancement.
        """
        enhanced = parameters.copy()

        # Add preflight memory context
        if context.preflight_result and context.preflight_result.memory_context:
            enhanced["_memory_context"] = context.preflight_result.memory_context

        # Add handoff context from previous phase
        if context.handoff_context:
            enhanced["_handoff_context"] = context.handoff_context
            enhanced["_handoff_tokens"] = context.handoff_tokens

        # Add feature name
        enhanced["_feature_name"] = context.feature_name or context.ritual_context.feature_name

        return enhanced

    # ========================================================================
    # Shutdown
    # ========================================================================

    async def shutdown(self, context: PhaseExecutionContext) -> None:
        """
        Graceful cleanup after phase completion.

        Override for phase-specific cleanup.
        """
        logger.info(f"Phase {self._phase.value} shutdown complete")

    # ========================================================================
    # Decision Helpers
    # ========================================================================

    def get_decision_options(self) -> List[str]:
        """Get available decision options for this phase."""
        return PHASE_DECISION_OPTIONS.get(self._phase, ["Proceed", "Skip"])

    def get_default_decision(self) -> str:
        """Get default decision for this phase."""
        options = self.get_decision_options()
        return options[0] if options else "Proceed"

    # ========================================================================
    # Memory Operation Helpers
    # ========================================================================

    async def recall_memories(
        self,
        query: str,
        context: PhaseExecutionContext,
        strategy: str = "fused",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recall memories using HoloLoom MCP tools.

        Wrapper for holoLoom-memory:recall_memories.
        """
        # This would integrate with actual MCP tool call
        # For now, return structure that matches expected format
        logger.info(f"Recalling memories: {query[:50]}... (strategy={strategy})")
        return []

    async def store_memory(
        self,
        content: str,
        context: PhaseExecutionContext,
        tags: Optional[List[str]] = None,
        importance: float = 0.5
    ) -> Optional[str]:
        """
        Store memory using HoloLoom MCP tools.

        Wrapper for holoLoom-memory:store_memory.
        """
        logger.info(f"Storing memory: {content[:50]}... (importance={importance})")
        return None  # Would return memory ID


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Capability enum
    'AgentCapability',

    # Data structures
    'PhasePreflightResult',
    'PhaseResult',
    'PhaseExecutionContext',

    # Protocol
    'RitualPhaseSkill',

    # Base class
    'AbstractRitualPhase',
]
