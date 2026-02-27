"""
Ritual Agent Coordinator
========================

Created: December 30, 2025
Purpose: Autonomous coordinator for ritual phases

Mirrors hololoom/agentic/multi_agent.py:MultiAgentCoordinator pattern.
Implements the 3-stage pattern:
1. Capability Detection (which phases needed?)
2. Routing (which agent for each phase? - Thompson Sampling)
3. Execution (run phases with context handoff)

Can be invoked by:
- Humans via /ritual commands
- Other agents via delegate_ritual()
- Automated triggers (e.g., session start/end)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
from enum import Enum
import asyncio
import logging

# Import ritual components
from .ritual_events import (
    RitualPhase,
    RitualContext,
    RitualEventType,
    create_ritual_event,
    ritual_started,
    ritual_completed,
    phase_started,
    phase_completed,
)
from .agent_registration import (
    RitualAgentRegistry,
    RitualAgentInfo,
    AgentCapability,
    DEFAULT_RITUAL_AGENTS,
)
from .thompson_router import (
    RitualPhaseRouter,
    SelectionStrategy,
)
from .message_bus import (
    RitualMessageBus,
    PhaseMessage,
    MessageType,
    MessagePriority,
)
from .context_handoff import (
    RitualContextHandoff,
    ContextItem,
    HandoffStrategy,
    HandoffResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Coordinator State
# ============================================================================

class CoordinatorState(Enum):
    """State of the ritual coordinator."""
    IDLE = "idle"               # No active rituals
    EXECUTING = "executing"     # Ritual in progress
    PAUSED = "paused"           # Ritual paused (user decision pending)
    CANCELLED = "cancelled"     # Ritual cancelled
    COMPLETED = "completed"     # Ritual completed


# ============================================================================
# Execution Result
# ============================================================================

@dataclass
class RitualExecutionResult:
    """
    Result of ritual execution.

    Contains comprehensive execution metadata for analysis
    and learning.
    """
    success: bool
    phases_completed: List[str]
    phases_skipped: List[str]
    total_duration_seconds: float
    decisions_made: List[Dict[str, Any]]
    memories_stored: List[str]
    final_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Learning metrics
    avg_confidence: float = 0.0
    context_tokens_saved: int = 0
    handoff_efficiency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "phases_completed": self.phases_completed,
            "phases_skipped": self.phases_skipped,
            "total_duration_seconds": self.total_duration_seconds,
            "decisions_made": self.decisions_made,
            "memories_stored": self.memories_stored,
            "final_output": str(self.final_output)[:500] if self.final_output else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "avg_confidence": self.avg_confidence,
            "context_tokens_saved": self.context_tokens_saved,
            "handoff_efficiency": self.handoff_efficiency,
            "metadata": self.metadata,
        }


@dataclass
class PhaseExecutionResult:
    """Result of single phase execution."""
    success: bool
    output: Any
    confidence: float = 0.0
    decision_needed: bool = False
    decision_options: List[str] = field(default_factory=list)
    suggested_decision: str = ""
    duration_ms: float = 0.0
    memories_stored: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Phase to Capability Mapping
# ============================================================================

PHASE_CAPABILITY_MAP: Dict[str, AgentCapability] = {
    "awaken": AgentCapability.CONTEXT_RESTORATION,
    "plan": AgentCapability.PLANNING,
    "implement": AgentCapability.CODE_ASSISTANCE,
    "review": AgentCapability.QUALITY_ASSURANCE,
    "reflect": AgentCapability.KNOWLEDGE_CONSOLIDATION,
}

DEFAULT_PHASE_ORDER = ["awaken", "plan", "implement", "review", "reflect"]


# ============================================================================
# Ritual Agent Coordinator
# ============================================================================

class RitualAgentCoordinator:
    """
    Autonomous coordinator for ritual phases.

    Implements the 3-stage pattern from MultiAgentCoordinator:
    1. Capability Detection (which phases needed?)
    2. Routing (which agent for each phase? - Thompson Sampling)
    3. Execution (run phases with context handoff)

    Can be invoked by:
    - Humans via /ritual commands
    - Other agents via delegate_ritual()
    - Automated triggers (e.g., session start/end)

    Usage:
        coordinator = RitualAgentCoordinator()

        # Full ritual with auto-decisions
        result = await coordinator.delegate_ritual(
            feature_name="Add Thompson Sampling",
            context_type="new_feature",
            auto_decisions=True
        )

        # Specific phases only
        result = await coordinator.delegate_ritual(
            feature_name="Fix bug",
            required_phases=["awaken", "review", "reflect"],
            skip_phases=["plan", "implement"]
        )
    """

    def __init__(
        self,
        registry: Optional[RitualAgentRegistry] = None,
        router: Optional[RitualPhaseRouter] = None,
        message_bus: Optional[RitualMessageBus] = None,
        handoff: Optional[RitualContextHandoff] = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
        handoff_strategy: HandoffStrategy = HandoffStrategy.BALANCED,
        default_token_budget: int = 2000,
    ):
        """
        Initialize the ritual coordinator.

        Args:
            registry: Agent registry (created if None)
            router: Phase router (created if None)
            message_bus: Message bus (created if None)
            handoff: Context handoff (created if None)
            selection_strategy: Strategy for phase selection
            handoff_strategy: Strategy for context handoff
            default_token_budget: Default token budget for handoffs
        """
        # Create or use provided components
        self.registry = registry or self._create_default_registry()
        self.router = router or RitualPhaseRouter(
            registry=self.registry,
            strategy=selection_strategy
        )
        self.message_bus = message_bus or RitualMessageBus()
        self.handoff = handoff or RitualContextHandoff(strategy=handoff_strategy)

        # Configuration
        self.default_token_budget = default_token_budget
        self.selection_strategy = selection_strategy
        self.handoff_strategy = handoff_strategy

        # State tracking
        self._active_rituals: Dict[str, RitualContext] = {}
        self._state = CoordinatorState.IDLE

        # Phase handlers (can be extended)
        self._phase_handlers: Dict[str, Callable] = {}

        # Subscribe coordinator to message bus
        self._setup_message_handlers()

    def _create_default_registry(self) -> RitualAgentRegistry:
        """Create and populate default agent registry."""
        registry = RitualAgentRegistry()
        for agent in DEFAULT_RITUAL_AGENTS:
            registry.register(agent)
        return registry

    def _setup_message_handlers(self) -> None:
        """Setup message bus handlers for coordinator."""
        # Subscribe to control messages
        async def handle_control(msg: PhaseMessage) -> Optional[PhaseMessage]:
            action = msg.payload.get("action", "")
            if action == "pause":
                self._state = CoordinatorState.PAUSED
            elif action == "resume":
                self._state = CoordinatorState.EXECUTING
            elif action == "cancel":
                self._state = CoordinatorState.CANCELLED
            return None

        self.message_bus.subscribe("coordinator", handle_control)

    # ========================================================================
    # Main Delegation Methods
    # ========================================================================

    async def delegate_ritual(
        self,
        feature_name: str,
        context_type: str = "default",
        required_phases: Optional[List[str]] = None,
        skip_phases: Optional[List[str]] = None,
        auto_decisions: bool = False,
        token_budget: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> RitualExecutionResult:
        """
        Delegate a complete ritual to the agent system.

        For autonomous agent invocation:
        - Detects which phases are needed
        - Routes to best agents via Thompson Sampling
        - Executes with MI-aware context handoff
        - Updates Thompson priors based on outcomes

        Args:
            feature_name: Feature being worked on
            context_type: Type for Thompson Sampling context
            required_phases: Force these phases (None = auto-detect)
            skip_phases: Skip these phases
            auto_decisions: Auto-accept suggested decisions
            token_budget: Token budget for context handoff
            timeout_seconds: Optional timeout for entire ritual

        Returns:
            RitualExecutionResult with comprehensive metrics
        """
        # Create ritual context
        ritual_id = f"ritual-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        context = RitualContext(
            ritual_id=ritual_id,
            feature_name=feature_name,
        )
        self._active_rituals[ritual_id] = context
        self._state = CoordinatorState.EXECUTING

        start_time = datetime.now()
        phases_completed: List[str] = []
        phases_skipped: List[str] = []
        context_items: List[ContextItem] = []
        total_tokens_saved = 0
        confidence_sum = 0.0

        # Determine phase order
        phase_order = required_phases or DEFAULT_PHASE_ORDER.copy()
        skip_set = set(skip_phases or [])

        # Emit ritual started event
        await self.message_bus.broadcast(
            from_phase="coordinator",
            payload=ritual_started(context),
            correlation_id=ritual_id
        )

        try:
            for phase_name in phase_order:
                # Check for cancellation
                if self._state == CoordinatorState.CANCELLED:
                    break

                # Skip if in skip set
                if phase_name in skip_set:
                    phases_skipped.append(phase_name)
                    continue

                # Handle pause state
                while self._state == CoordinatorState.PAUSED:
                    await asyncio.sleep(0.1)
                    if self._state == CoordinatorState.CANCELLED:
                        break

                # Get capability for this phase
                capability = self._phase_to_capability(phase_name)
                if capability is None:
                    logger.warning(f"Unknown phase: {phase_name}")
                    phases_skipped.append(phase_name)
                    continue

                # Route to best agent via Thompson Sampling
                agents = self.router.select_phase(
                    required_capability=capability,
                    context_type=context_type,
                    top_k=1
                )

                if not agents:
                    logger.warning(f"No agent found for phase: {phase_name}")
                    phases_skipped.append(phase_name)
                    continue

                agent = agents[0]

                # Prepare context handoff from previous phases
                handoff_result = self.handoff.prepare_handoff(
                    context_items=context_items,
                    from_phase=phases_completed[-1] if phases_completed else "orchestrator",
                    to_phase=phase_name,
                    target_capability=capability.value,
                    token_budget=token_budget or self.default_token_budget
                )

                # Track token savings
                if handoff_result.original_count > 0:
                    original_tokens = sum(
                        item.token_estimate() for item in context_items
                    )
                    used_tokens = handoff_result.get_total_tokens()
                    total_tokens_saved += (original_tokens - used_tokens)

                # Execute phase
                phase_result = await self._execute_phase(
                    agent=agent,
                    phase_name=phase_name,
                    context=context,
                    handoff_context=handoff_result.get_context_text(),
                    handoff_tokens=handoff_result.get_total_tokens(),
                    auto_decisions=auto_decisions
                )

                # Update Thompson prior based on outcome
                self.router.update(
                    agent_id=agent.id,
                    context_type=context_type,
                    success=phase_result.success,
                    confidence=phase_result.confidence
                )

                # Update agent statistics
                agent.total_invocations += 1
                if phase_result.success:
                    agent.successful_invocations += 1

                # Collect output for next phase handoff
                if phase_result.output:
                    context_items.append(ContextItem(
                        content=str(phase_result.output),
                        source_phase=phase_name,
                        importance=phase_result.confidence,
                        metadata={
                            "agent_id": agent.id,
                            "duration_ms": phase_result.duration_ms,
                        }
                    ))

                # Track metrics
                phases_completed.append(phase_name)
                confidence_sum += phase_result.confidence

                # Store memories
                context.memories_stored.extend(phase_result.memories_stored)

                # Handle decision points
                if phase_result.decision_needed and not auto_decisions:
                    # Pause for user decision
                    self._state = CoordinatorState.PAUSED

                    # Would emit decision_required event here
                    # For now, auto-accept suggestion
                    decision = phase_result.suggested_decision
                    context.record_decision(
                        phase_name,
                        decision,
                        {"auto": False, "options": phase_result.decision_options}
                    )

                    self._state = CoordinatorState.EXECUTING

            # Calculate final metrics
            duration = (datetime.now() - start_time).total_seconds()
            avg_confidence = (
                confidence_sum / len(phases_completed)
                if phases_completed else 0.0
            )

            # Calculate handoff efficiency
            total_original = sum(item.token_estimate() for item in context_items)
            handoff_efficiency = (
                total_tokens_saved / total_original
                if total_original > 0 else 0.0
            )

            # Emit ritual completed event
            await self.message_bus.broadcast(
                from_phase="coordinator",
                payload=ritual_completed(context),
                correlation_id=ritual_id
            )

            self._state = CoordinatorState.COMPLETED

            return RitualExecutionResult(
                success=len(phases_completed) > 0 and self._state != CoordinatorState.CANCELLED,
                phases_completed=phases_completed,
                phases_skipped=phases_skipped,
                total_duration_seconds=duration,
                decisions_made=context.decisions_made,
                memories_stored=context.memories_stored,
                final_output=context_items[-1].content if context_items else None,
                avg_confidence=avg_confidence,
                context_tokens_saved=total_tokens_saved,
                handoff_efficiency=handoff_efficiency,
                metadata={
                    "ritual_id": ritual_id,
                    "context_type": context_type,
                    "feature_name": feature_name,
                    "selection_strategy": self.selection_strategy.value,
                    "handoff_strategy": self.handoff_strategy.value,
                }
            )

        except Exception as e:
            logger.error(f"Ritual execution failed: {e}")
            duration = (datetime.now() - start_time).total_seconds()

            return RitualExecutionResult(
                success=False,
                phases_completed=phases_completed,
                phases_skipped=phases_skipped,
                total_duration_seconds=duration,
                decisions_made=context.decisions_made,
                memories_stored=context.memories_stored,
                final_output=None,
                errors=[str(e)],
                metadata={"ritual_id": ritual_id}
            )

        finally:
            # Cleanup
            del self._active_rituals[ritual_id]
            if self._state != CoordinatorState.CANCELLED:
                self._state = CoordinatorState.IDLE

    async def _execute_phase(
        self,
        agent: RitualAgentInfo,
        phase_name: str,
        context: RitualContext,
        handoff_context: str,
        handoff_tokens: int,
        auto_decisions: bool,
    ) -> PhaseExecutionResult:
        """
        Execute a single phase via message bus.

        Args:
            agent: Agent to execute phase
            phase_name: Phase name
            context: Ritual context
            handoff_context: Filtered context from previous phases
            handoff_tokens: Token count of handoff context
            auto_decisions: Whether to auto-accept decisions

        Returns:
            PhaseExecutionResult with execution details
        """
        start_time = datetime.now()

        try:
            # Emit phase started
            context.current_phase = RitualPhase(phase_name)

            # Send request to phase agent via message bus
            message_id = await self.message_bus.request_phase(
                from_phase="coordinator",
                to_phase=agent.id,
                payload={
                    "phase": phase_name,
                    "feature_name": context.feature_name,
                    "handoff_context": handoff_context,
                    "handoff_tokens": handoff_tokens,
                    "auto_decisions": auto_decisions,
                    "ritual_id": context.ritual_id,
                },
                correlation_id=context.ritual_id,
                priority=MessagePriority.NORMAL,
            )

            # Check if there's a registered handler for this phase
            if phase_name in self._phase_handlers:
                handler = self._phase_handlers[phase_name]
                result = await handler(
                    agent=agent,
                    context=context,
                    handoff_context=handoff_context,
                )
                return result

            # Default execution (simulated for now)
            # In production, this would await actual phase execution
            duration = (datetime.now() - start_time).total_seconds() * 1000

            return PhaseExecutionResult(
                success=True,
                output=f"Phase {phase_name} completed for {context.feature_name}",
                confidence=0.8,
                decision_needed=False,
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Phase {phase_name} execution failed: {e}")
            duration = (datetime.now() - start_time).total_seconds() * 1000

            return PhaseExecutionResult(
                success=False,
                output=None,
                confidence=0.0,
                duration_ms=duration,
                errors=[str(e)],
            )

    def _phase_to_capability(self, phase_name: str) -> Optional[AgentCapability]:
        """Map phase name to capability enum."""
        return PHASE_CAPABILITY_MAP.get(phase_name.lower())

    # ========================================================================
    # Phase Handler Registration
    # ========================================================================

    def register_phase_handler(
        self,
        phase_name: str,
        handler: Callable[..., Awaitable[PhaseExecutionResult]]
    ) -> None:
        """
        Register a custom handler for a phase.

        Args:
            phase_name: Phase name (awaken, plan, etc.)
            handler: Async handler function
        """
        self._phase_handlers[phase_name] = handler
        logger.debug(f"Registered handler for phase: {phase_name}")

    def unregister_phase_handler(self, phase_name: str) -> bool:
        """
        Unregister a phase handler.

        Args:
            phase_name: Phase name

        Returns:
            True if handler was found and removed
        """
        if phase_name in self._phase_handlers:
            del self._phase_handlers[phase_name]
            return True
        return False

    # ========================================================================
    # Control Methods
    # ========================================================================

    async def pause_ritual(self, ritual_id: Optional[str] = None) -> bool:
        """
        Pause a running ritual.

        Args:
            ritual_id: Specific ritual to pause (None = current)

        Returns:
            True if ritual was paused
        """
        if self._state == CoordinatorState.EXECUTING:
            await self.message_bus.control(
                to_phase="",  # Broadcast
                action="pause",
                correlation_id=ritual_id
            )
            self._state = CoordinatorState.PAUSED
            return True
        return False

    async def resume_ritual(self, ritual_id: Optional[str] = None) -> bool:
        """
        Resume a paused ritual.

        Args:
            ritual_id: Specific ritual to resume (None = current)

        Returns:
            True if ritual was resumed
        """
        if self._state == CoordinatorState.PAUSED:
            await self.message_bus.control(
                to_phase="",  # Broadcast
                action="resume",
                correlation_id=ritual_id
            )
            self._state = CoordinatorState.EXECUTING
            return True
        return False

    async def cancel_ritual(self, ritual_id: Optional[str] = None) -> bool:
        """
        Cancel a running ritual.

        Args:
            ritual_id: Specific ritual to cancel (None = current)

        Returns:
            True if ritual was cancelled
        """
        if self._state in (CoordinatorState.EXECUTING, CoordinatorState.PAUSED):
            await self.message_bus.control(
                to_phase="",  # Broadcast
                action="cancel",
                correlation_id=ritual_id
            )
            self._state = CoordinatorState.CANCELLED
            return True
        return False

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_state(self) -> CoordinatorState:
        """Get current coordinator state."""
        return self._state

    def get_active_rituals(self) -> Dict[str, RitualContext]:
        """Get all active rituals."""
        return self._active_rituals.copy()

    def get_phase_recommendations(
        self,
        context_type: str = "default"
    ) -> Dict[str, Dict[str, float]]:
        """
        Get Thompson Sampling recommendations for all phases.

        Args:
            context_type: Context type for recommendations

        Returns:
            Dict of agent_id → recommendation metrics
        """
        return self.router.get_recommendations(context_type)

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "state": self._state.value,
            "active_rituals": len(self._active_rituals),
            "registered_agents": len(self.registry._agents),
            "registered_handlers": len(self._phase_handlers),
            "message_bus_stats": self.message_bus.get_statistics(),
            "selection_strategy": self.selection_strategy.value,
            "handoff_strategy": self.handoff_strategy.value,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_coordinator(
    selection_strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
    handoff_strategy: HandoffStrategy = HandoffStrategy.BALANCED,
    token_budget: int = 2000,
) -> RitualAgentCoordinator:
    """
    Create a fully configured ritual coordinator.

    Args:
        selection_strategy: Strategy for phase selection
        handoff_strategy: Strategy for context handoff
        token_budget: Default token budget for handoffs

    Returns:
        Configured RitualAgentCoordinator
    """
    return RitualAgentCoordinator(
        selection_strategy=selection_strategy,
        handoff_strategy=handoff_strategy,
        default_token_budget=token_budget,
    )


def create_minimal_coordinator() -> RitualAgentCoordinator:
    """
    Create a minimal coordinator with defaults.

    Returns:
        RitualAgentCoordinator with default settings
    """
    return RitualAgentCoordinator()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # State enum
    'CoordinatorState',

    # Result types
    'RitualExecutionResult',
    'PhaseExecutionResult',

    # Constants
    'PHASE_CAPABILITY_MAP',
    'DEFAULT_PHASE_ORDER',

    # Main class
    'RitualAgentCoordinator',

    # Factory functions
    'create_coordinator',
    'create_minimal_coordinator',
]
