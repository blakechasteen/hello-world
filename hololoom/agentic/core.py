"""
HoloLoom Agentic Intelligence Core
===================================
Self-directed reasoning with verification loops.

Extends HoloLoom's recursive learning system with autonomous query generation,
multi-step reasoning, and verification protocols.

Philosophy:
"Don't just answer - understand, verify, refine."

Integration Points:
- Builds on recursive.FullLearningEngine (Phase 5)
- Uses alignment.audit_trail for decision logging
- Leverages ReflectionBuffer for learning
- Extends action_items for goal tracking
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import uuid

from HoloLoom.protocols.types import Query, Context, MemoryShard
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.recursive import FullLearningEngine, ActionItemTracker, ActionStatus
from HoloLoom.alignment.audit_trail import AuditTrail, DecisionType, OutcomeType

# Deception Detection (Dec 2025) - E1.3 Alignment Enrichment
try:
    from HoloLoom.alignment.deception_detection import (
        DeceptionDetector,
        BehavioralProbe,
        ProbeType,
        DeceptionSignal,
        GoalStatement,
        ActionObservation,
    )
    DECEPTION_AVAILABLE = True
except ImportError:
    DECEPTION_AVAILABLE = False
    DeceptionDetector = None
    BehavioralProbe = None
    ProbeType = None
    DeceptionSignal = None
    GoalStatement = None
    ActionObservation = None

# Instrumental Convergence Prevention (Dec 2025) - E1.4 Alignment Enrichment
try:
    from HoloLoom.alignment.instrumental_convergence import (
        InstrumentalConvergenceGuard,
        ResourceType,
        ResourceBounds,
        AutonomyLimit,
        ResourceViolation,
        PowerSeekingIndicators,
        create_guard,
    )
    CONVERGENCE_AVAILABLE = True
except ImportError:
    CONVERGENCE_AVAILABLE = False
    InstrumentalConvergenceGuard = None
    ResourceType = None
    ResourceBounds = None
    AutonomyLimit = None
    ResourceViolation = None
    PowerSeekingIndicators = None
    create_guard = None

# Agent Monitoring (Nov 2025)
try:
    from HoloLoom.agentic.monitoring import AgentMonitor, AgentStatus
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    AgentMonitor = None
    AgentStatus = None

# Safety Integration (Dec 2025) - MRF Safe Integration Phase
try:
    from HoloLoom.agentic.safety_adapter import AgenticSafetyAdapter, create_safety_adapter
    from HoloLoom.protocols.safety import SafetyGateDecision, SafetyRiskLevel
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    AgenticSafetyAdapter = None
    SafetyGateDecision = None
    SafetyRiskLevel = None

# Conscience Integration (Dec 2025) - Phase 2B Per-Step Gating
try:
    from HoloLoom.agentic.conscience_adapter import (
        AgenticConscienceAdapter,
        create_conscience_adapter,
    )
    from HoloLoom.protocols.conscience import ConscienceDecision, StepType
    CONSCIENCE_AVAILABLE = True
except ImportError:
    CONSCIENCE_AVAILABLE = False
    AgenticConscienceAdapter = None
    ConscienceDecision = None
    StepType = None


logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

class ReasoningMode(Enum):
    """Agentic reasoning modes."""
    DIRECT = "direct"              # Single-pass answer
    VERIFY = "verify"              # Answer + verification loop
    RESEARCH = "research"          # Multi-query exploration
    PLAN_EXECUTE = "plan_execute"  # Goal decomposition + execution


@dataclass
class AgenticIntent:
    """
    User intent with goal tracking.

    Extends action items with research/verification goals.
    """
    intent_id: str
    original_query: str
    goal: str
    sub_goals: List[str] = field(default_factory=list)
    evidence_gathered: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.85
    max_verification_loops: int = 3

    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    status: ActionStatus = ActionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verification loop."""
    verified: bool
    confidence: float
    contradictions: List[str]
    supporting_evidence: List[str]
    suggested_refinements: List[str]

    # Provenance
    verification_queries: List[str]
    sources_checked: List[str]


@dataclass
class AgenticResult:
    """Complete agentic reasoning result."""
    spacetime: Spacetime
    intent: AgenticIntent
    reasoning_mode: ReasoningMode
    verification: Optional[VerificationResult] = None

    # Multi-step tracking
    steps_taken: List[Dict[str, Any]] = field(default_factory=list)
    total_queries: int = 0
    total_duration_ms: float = 0.0

    # Consciousness Integration - Phase 1 (Nov 2025)
    # Epistemic confidence aggregated across all reasoning steps
    # Tracks system's self-awareness of knowledge gaps throughout multi-query reasoning
    aggregated_epistemic_confidence: Optional[float] = None


# ============================================================================
# Agentic Orchestrator
# ============================================================================

class AgenticOrchestrator:
    """
    Autonomous reasoning orchestrator.

    Wraps WeavingOrchestrator with self-directed reasoning capabilities:
    - Generates verification queries
    - Decomposes complex goals
    - Tracks multi-step reasoning
    - Learns from verification outcomes

    Usage:
        async with AgenticOrchestrator(config, shards) as agent:
            result = await agent.reason(
                query="Is Thompson Sampling optimal for this task?",
                mode=ReasoningMode.VERIFY
            )
    """

    def __init__(
        self,
        learning_engine: FullLearningEngine,
        audit_trail: Optional[AuditTrail] = None,
        enable_verification: bool = True,
        enable_goal_tracking: bool = True,
        llm: Optional[Any] = None,  # LLM for intelligent query generation
        awareness_layer: Optional[Any] = None,  # Consciousness Integration - Phase 1 (Nov 2025)
        epistemic_threshold: float = 0.3,  # Early stopping threshold for epistemic confidence
        monitor: Optional[Any] = None,  # Agent Monitor for real-time tracking (Nov 2025)
        safety_adapter: Optional['AgenticSafetyAdapter'] = None,  # Safety Integration (Dec 2025)
        enable_safety: bool = True,  # Safe by default - MRF ELEGANCE principle
        conscience_adapter: Optional['AgenticConscienceAdapter'] = None,  # Conscience Integration (Dec 2025)
        enable_conscience: bool = True,  # Per-step gating by default - Phase 2B
        deception_detector: Optional['DeceptionDetector'] = None,  # Deception Detection (Dec 2025) - E1.3
        enable_deception_detection: bool = True,  # Goal-action alignment checking
        convergence_guard: Optional['InstrumentalConvergenceGuard'] = None,  # Instrumental Convergence (Dec 2025) - E1.4
        enable_convergence_prevention: bool = True  # Power-seeking detection and resource bounds
    ):
        self.learning_engine = learning_engine
        self.audit_trail = audit_trail or AuditTrail()
        self.enable_verification = enable_verification
        self.enable_goal_tracking = enable_goal_tracking
        self.llm = llm  # LLM for agentic search
        self.awareness_layer = awareness_layer  # Consciousness Integration - Phase 1
        self.epistemic_threshold = epistemic_threshold  # Early stopping threshold
        self.monitor = monitor if MONITORING_AVAILABLE else None  # Agent Monitor (Nov 2025)

        # Goal tracker (extends action items)
        self.goal_tracker = ActionItemTracker() if enable_goal_tracking else None

        self.logger = logging.getLogger(__name__)

        # Initialize LLM if not provided but available in orchestrator
        if self.llm is None and hasattr(learning_engine, 'orchestrator'):
            orchestrator = learning_engine.orchestrator
            if hasattr(orchestrator, 'tool_executor') and hasattr(orchestrator.tool_executor, 'llm'):
                self.llm = orchestrator.tool_executor.llm
                if self.llm:
                    self.logger.info("LLM-activated agentic search enabled")

        # Consciousness Integration - Phase 1 (Nov 2025)
        # Extract awareness_layer from learning_engine's orchestrator if not provided
        if self.awareness_layer is None and hasattr(learning_engine, 'orchestrator'):
            orchestrator = learning_engine.orchestrator
            if hasattr(orchestrator, 'awareness_layer'):
                self.awareness_layer = orchestrator.awareness_layer
                if self.awareness_layer:
                    self.logger.info("Awareness layer extracted from orchestrator for epistemic tracking")

        # Safety Integration (Dec 2025) - MRF Safe Integration Phase
        # Implements ELEGANCE single gate pattern: one safety check at reason() entry
        self.enable_safety = enable_safety and SAFETY_AVAILABLE
        self.safety_adapter = None

        if self.enable_safety:
            if safety_adapter is not None:
                self.safety_adapter = safety_adapter
                self.logger.info("Safety adapter provided - agentic reasoning will be gated")
            elif hasattr(learning_engine, 'orchestrator'):
                # Try to extract guardrails from orchestrator (same pattern as awareness_layer)
                orchestrator = learning_engine.orchestrator
                if hasattr(orchestrator, 'guardrails') and orchestrator.guardrails:
                    self.safety_adapter = create_safety_adapter(orchestrator.guardrails)
                    self.logger.info("Safety adapter created from orchestrator guardrails")

            # Auto-create with safe defaults if none available (safe by default)
            if self.safety_adapter is None:
                self.safety_adapter = create_safety_adapter(auto_create=True)
                self.logger.info("Auto-created safety adapter with safe defaults")

        # Conscience Integration (Dec 2025) - Phase 2B Per-Step Gating
        # Implements per-step safety checks in VERIFY, RESEARCH, PLAN_EXECUTE modes
        self.enable_conscience = enable_conscience and CONSCIENCE_AVAILABLE
        self.conscience_adapter = None

        if self.enable_conscience:
            if conscience_adapter is not None:
                self.conscience_adapter = conscience_adapter
                self.logger.info("Conscience adapter provided - per-step gating enabled")
            elif hasattr(learning_engine, 'orchestrator'):
                # Try to extract conscience from orchestrator
                orchestrator = learning_engine.orchestrator
                if hasattr(orchestrator, 'conscience') and orchestrator.conscience:
                    self.conscience_adapter = create_conscience_adapter(
                        conscience=orchestrator.conscience
                    )
                    self.logger.info("Conscience adapter created from orchestrator conscience")

            # Auto-create with safe defaults if none available
            if self.conscience_adapter is None:
                self.conscience_adapter = create_conscience_adapter(auto_create=True)
                self.logger.info("Auto-created conscience adapter for per-step gating")

        # Deception Detection (Dec 2025) - E1.3 Alignment Enrichment
        # Behavioral probes and goal transparency for multi-step reasoning
        self.enable_deception_detection = enable_deception_detection and DECEPTION_AVAILABLE
        self.deception_detector = None

        if self.enable_deception_detection:
            if deception_detector is not None:
                self.deception_detector = deception_detector
                self.logger.info("Deception detector provided - goal-action alignment enabled")
            else:
                # Auto-create with default configuration
                try:
                    self.deception_detector = DeceptionDetector()
                    self.logger.info("Auto-created deception detector for goal transparency")
                except Exception as e:
                    self.logger.warning(f"Failed to create deception detector: {e}")
                    self.deception_detector = None
                    self.enable_deception_detection = False

        # Instrumental Convergence Prevention (Dec 2025) - E1.4 Alignment Enrichment
        # Resource bounds, autonomy limits, and power-seeking detection
        self.enable_convergence_prevention = enable_convergence_prevention and CONVERGENCE_AVAILABLE
        self.convergence_guard = None

        if self.enable_convergence_prevention:
            if convergence_guard is not None:
                self.convergence_guard = convergence_guard
                self.logger.info("Convergence guard provided - power-seeking detection enabled")
            else:
                # Auto-create with default configuration and safe autonomy limits
                try:
                    self.convergence_guard = create_guard(
                        enable_resource_tracking=True,
                        enable_autonomy_limits=True
                    )
                    # Set conservative autonomy limits for agentic reasoning
                    self.convergence_guard.set_autonomy_limit(AutonomyLimit(
                        max_autonomous_actions=50,  # Max steps without oversight
                        max_autonomous_duration=1800.0,  # 30 minutes max
                        require_approval_for=["execute", "system", "deploy", "delete"]
                    ))
                    # Set default resource bounds
                    self.convergence_guard.set_resource_bounds(
                        ResourceType.API_CALLS,
                        ResourceBounds(
                            resource_type=ResourceType.API_CALLS,
                            soft_limit=50.0,
                            hard_limit=100.0,
                            rate_limit=10.0,
                            time_window_seconds=60.0
                        )
                    )
                    self.logger.info("Auto-created convergence guard with conservative limits")
                except Exception as e:
                    self.logger.warning(f"Failed to create convergence guard: {e}")
                    self.convergence_guard = None
                    self.enable_convergence_prevention = False

    async def _check_convergence(
        self,
        intent: 'AgenticIntent',
        step_action: str,
        step_type: str,
        actions_so_far: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Check for instrumental convergence patterns during multi-step reasoning.

        Instrumental Convergence Prevention - E1.4 (Dec 2025)
        Monitors for power-seeking behavior, resource acquisition, and autonomy bounds.

        Args:
            intent: The declared goal/intent of the reasoning
            step_action: The action taken in this step (e.g., query text)
            step_type: Type of step (verification, research, plan_step)
            actions_so_far: List of all actions taken in this reasoning session

        Returns:
            Dict with convergence analysis results, or None if prevention disabled
        """
        if not self.convergence_guard:
            return None

        try:
            # Check autonomy limits
            allowed, autonomy_reason = self.convergence_guard.check_autonomy_limits(step_type)
            if not allowed:
                self.logger.warning(f"Autonomy limit reached: {autonomy_reason}")
                self.audit_trail.log_decision(
                    decision_type=DecisionType.TOOL_SELECTION,
                    outcome=OutcomeType.BLOCKED,
                    reason=f"Autonomy limit: {autonomy_reason}",
                    query_text=step_action[:200],
                    metadata={
                        "convergence_prevention": {
                            "blocked": True,
                            "reason": autonomy_reason,
                            "step_type": step_type,
                            "intent_id": intent.intent_id
                        }
                    }
                )
                return {
                    "blocked": True,
                    "reason": autonomy_reason,
                    "autonomy_limit_reached": True,
                    "power_seeking": None
                }

            # Record the autonomous action
            self.convergence_guard.record_autonomous_action(step_type)

            # Check for self-modification attempts
            if self.convergence_guard.detect_self_modification(step_action):
                self.logger.warning(f"Self-modification attempt detected in: {step_action[:100]}")
                self.audit_trail.log_decision(
                    decision_type=DecisionType.TOOL_SELECTION,
                    outcome=OutcomeType.FLAGGED,
                    reason="Self-modification attempt detected",
                    query_text=step_action[:200],
                    metadata={
                        "convergence_prevention": {
                            "self_modification": True,
                            "action": step_action[:100],
                            "step_type": step_type
                        }
                    }
                )

            # Detect power-seeking behavior patterns
            power_indicators = await self.convergence_guard.detect_power_seeking(
                actions=actions_so_far + [step_action],
                context={
                    "intent": intent.goal,
                    "step_type": step_type,
                    "total_steps": len(actions_so_far) + 1
                }
            )

            # Log if power-seeking detected
            if power_indicators.is_power_seeking:
                self.logger.warning(
                    f"Power-seeking behavior detected: {power_indicators.detected_patterns}"
                )
                self.audit_trail.log_decision(
                    decision_type=DecisionType.TOOL_SELECTION,
                    outcome=OutcomeType.FLAGGED,
                    reason=f"Power-seeking patterns: {', '.join(power_indicators.detected_patterns)}",
                    query_text=step_action[:200],
                    metadata={
                        "convergence_prevention": {
                            "power_seeking": True,
                            "patterns": power_indicators.detected_patterns,
                            "scores": {
                                "resource_acquisition": power_indicators.resource_acquisition_score,
                                "privilege_escalation": power_indicators.privilege_escalation_score,
                                "control_expansion": power_indicators.control_expansion_score,
                                "self_preservation": power_indicators.self_preservation_score
                            }
                        }
                    }
                )

            return {
                "blocked": False,
                "reason": autonomy_reason,
                "autonomy_limit_reached": False,
                "power_seeking": {
                    "detected": power_indicators.is_power_seeking,
                    "patterns": power_indicators.detected_patterns,
                    "max_score": max(
                        power_indicators.resource_acquisition_score,
                        power_indicators.privilege_escalation_score,
                        power_indicators.control_expansion_score,
                        power_indicators.self_preservation_score
                    )
                },
                "autonomy_stats": self.convergence_guard.get_autonomy_statistics()
            }

        except Exception as e:
            self.logger.warning(f"Convergence check failed: {e}")
            return None

    async def _check_deception(
        self,
        intent: 'AgenticIntent',
        step_action: str,
        step_result: Any,
        step_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check for potential deceptive behavior during multi-step reasoning.

        Deception Detection - E1.3 (Dec 2025)
        Runs behavioral probes to verify goal-action alignment.

        Args:
            intent: The declared goal/intent of the reasoning
            step_action: The action taken in this step (e.g., query text)
            step_result: The result of the step (Spacetime or response)
            step_type: Type of step (verification, research, plan_step)

        Returns:
            Dict with deception analysis results, or None if detection disabled
        """
        if not self.deception_detector:
            return None

        try:
            # Declare the stated goal (if not already declared)
            goal_id = f"goal_{intent.intent_id}"
            if not self.deception_detector.goal_tracker.get_goal(goal_id):
                self.deception_detector.goal_tracker.declare_goal(
                    GoalStatement(
                        goal_id=goal_id,
                        description=intent.goal,
                        priority=1,
                        metadata={"original_query": intent.original_query}
                    )
                )

            # Create action observation
            result_text = ""
            if hasattr(step_result, 'response'):
                result_text = str(step_result.response)[:500]
            elif hasattr(step_result, 'text'):
                result_text = str(step_result.text)[:500]
            else:
                result_text = str(step_result)[:500]

            action_obs = ActionObservation(
                action_id=f"action_{intent.intent_id}_{step_type}_{hash(step_action) % 10000}",
                description=f"{step_type}: {step_action[:200]}",
                context={"step_type": step_type, "result_preview": result_text[:200]},
                goal_id=goal_id
            )

            # Run goal alignment probe
            probe = BehavioralProbe(
                probe_type=ProbeType.GOAL_ALIGNMENT,
                scenario=f"Step in {step_type} mode for goal: {intent.goal[:100]}",
                expected_behavior=f"Action should align with goal: {intent.goal[:100]}"
            )

            # Analyze alignment
            passed, deception_score = self.deception_detector.run_probe(
                probe,
                actual_behavior=f"Query: {step_action[:150]}. Result: {result_text[:150]}"
            )

            # Track the action
            self.deception_detector.goal_tracker.track_action(action_obs)

            # Log to audit trail if deception detected
            if deception_score > 0.5:
                self.audit_trail.log_decision(
                    decision_type=DecisionType.TOOL_SELECTION,
                    outcome=OutcomeType.FLAGGED if deception_score > 0.7 else OutcomeType.APPROVED,
                    reason=f"Deception check: score={deception_score:.2f}",
                    query_text=step_action[:200],
                    metadata={
                        "deception_detection": {
                            "score": deception_score,
                            "passed": passed,
                            "step_type": step_type,
                            "goal_id": goal_id,
                            "probe_type": "goal_alignment"
                        }
                    }
                )

            return {
                "deception_score": deception_score,
                "passed": passed,
                "goal_aligned": deception_score < 0.5,
                "signal_level": (
                    DeceptionSignal.CRITICAL.value if deception_score > 0.8 else
                    DeceptionSignal.HIGH.value if deception_score > 0.6 else
                    DeceptionSignal.MEDIUM.value if deception_score > 0.4 else
                    DeceptionSignal.LOW.value
                )
            }

        except Exception as e:
            self.logger.warning(f"Deception check failed: {e}")
            return None

    async def _gate_with_safety(
        self,
        query: Query,
        mode: ReasoningMode
    ) -> Optional[SafetyGateDecision]:
        """
        Gate reasoning through safety evaluation.

        ELEGANCE: Single safety check at reason() entry point.
        CRITIQUE: Fail-closed on any exception.

        Args:
            query: The query to evaluate
            mode: Reasoning mode

        Returns:
            SafetyGateDecision if safety is enabled, None otherwise
        """
        if not self.safety_adapter:
            return None

        # Get epistemic confidence from awareness layer (if available)
        epistemic_conf = None
        if self.awareness_layer:
            try:
                coherence = self.awareness_layer.get_current_coherence()
                epistemic_conf = coherence if coherence is not None else None
            except Exception as e:
                self.logger.warning(f"Could not get epistemic confidence: {e}")

        # Gate through safety adapter
        decision = await self.safety_adapter.gate_reasoning(
            query_text=query.text,
            reasoning_mode=mode.value,
            epistemic_confidence=epistemic_conf,
            context={"query_metadata": getattr(query, 'metadata', {})}
        )

        # Log decision to audit trail
        if decision:
            self.audit_trail.log_decision(
                decision_type=DecisionType.TOOL_SELECTION,
                outcome=OutcomeType.APPROVED if decision.allowed else OutcomeType.REJECTED,
                reason=f"Safety gate: {decision.reason}",
                query_text=query.text,
                metadata={
                    "safety_decision": {
                        "allowed": decision.allowed,
                        "risk_level": decision.risk_level.value,
                        "requires_approval": decision.requires_approval,
                        "epistemic_confidence": epistemic_conf,
                    }
                }
            )

            if not decision.allowed:
                self.logger.warning(f"[SAFETY] Reasoning blocked: {decision.reason}")

        return decision

    async def _gate_step_with_conscience(
        self,
        step_query: str,
        step_type: 'StepType',
        step_index: int,
        parent_decision: Optional['ConscienceDecision'] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional['ConscienceDecision']:
        """
        Gate individual reasoning step through conscience evaluation.

        Conscience Integration - Phase 2B (Dec 2025)
        Per-step gating for VERIFY, RESEARCH, PLAN_EXECUTE modes.

        Unlike safety gate (single entry point), conscience gates EACH step:
        - Each verification query in VERIFY mode
        - Each research sub-query in RESEARCH mode
        - Each sub-goal in PLAN_EXECUTE mode

        Args:
            step_query: The step query text to evaluate
            step_type: Type of step (VERIFICATION, RESEARCH, PLAN_STEP, etc.)
            step_index: Index of current step (0-based)
            parent_decision: Parent reasoning decision (from reason() entry)
            context: Optional additional context

        Returns:
            ConscienceDecision if conscience is enabled, None otherwise
        """
        if not self.conscience_adapter:
            return None

        try:
            decision = await self.conscience_adapter.gate_step(
                step_action=step_query,  # gate_step expects step_action param
                step_type=step_type,
                step_index=step_index,
                parent_decision=parent_decision,
                context=context or {}
            )

            # Log to audit trail
            if decision:
                self.audit_trail.log_decision(
                    decision_type=DecisionType.TOOL_SELECTION,
                    outcome=OutcomeType.APPROVED if decision.allowed else OutcomeType.REJECTED,
                    reason=f"Conscience step gate: {decision.reason}",
                    query_text=step_query,
                    metadata={
                        "conscience_decision": {
                            "allowed": decision.allowed,
                            "risk_level": decision.risk_level.name,
                            "voice": decision.voice,
                            "step_type": step_type.value if hasattr(step_type, 'value') else str(step_type),
                            "step_index": step_index,
                            "concerns": decision.concerns,
                        }
                    }
                )

                if not decision.allowed:
                    self.logger.warning(
                        f"[CONSCIENCE] Step {step_index + 1} blocked: {decision.reason}"
                    )

            return decision

        except Exception as e:
            self.logger.warning(f"[CONSCIENCE] Step gating failed: {e}")
            # Graceful degradation - return None to allow step
            return None

    async def reason(
        self,
        query: Query,
        mode: ReasoningMode = ReasoningMode.DIRECT,
        confidence_threshold: float = 0.85,
        max_steps: int = 5,
        project: str = "mythRL"  # Project name for monitoring
    ) -> AgenticResult:
        """
        Execute agentic reasoning with selected mode.

        Args:
            query: User query
            mode: Reasoning mode (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
            confidence_threshold: Minimum confidence for accepting result
            max_steps: Maximum reasoning steps
            project: Project name for monitoring (default: mythRL)

        Returns:
            AgenticResult with complete reasoning trace
        """
        start_time = datetime.now()

        # =================================================================
        # SAFETY GATE (Dec 2025) - MRF Safe Integration Phase
        # =================================================================
        # ELEGANCE: Single safety check at reason() entry point
        # CRITIQUE: Fail-closed - if blocked, return early with blocked result
        # =================================================================
        safety_decision = await self._gate_with_safety(query, mode)
        if safety_decision and not safety_decision.allowed:
            # Return blocked result with safety information
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Create blocked Spacetime with minimal trace
            blocked_trace = WeavingTrace(
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                stage_durations={"safety_gate": duration_ms},
            )
            blocked_spacetime = Spacetime(
                query_text=query.text,
                response=f"[BLOCKED] {safety_decision.reason}",
                tool_used="safety_gate",
                confidence=0.0,
                trace=blocked_trace,
                metadata={
                    "blocked": True,
                    "risk_level": safety_decision.risk_level.value,
                    "requires_approval": safety_decision.requires_approval,
                }
            )
            blocked_intent = AgenticIntent(
                intent_id=f"intent_{int(start_time.timestamp())}",
                original_query=query.text,
                goal="blocked_by_safety",
                confidence_threshold=confidence_threshold
            )
            return AgenticResult(
                spacetime=blocked_spacetime,
                intent=blocked_intent,
                reasoning_mode=mode,
                verification=None,
                steps_taken=[{
                    "type": "safety_gate",
                    "query": query.text,
                    "blocked": True,
                    "reason": safety_decision.reason,
                    "risk_level": safety_decision.risk_level.value,
                    "requires_approval": safety_decision.requires_approval,
                    "epistemic_confidence": safety_decision.epistemic_confidence,
                }],
                total_queries=0,
                total_duration_ms=duration_ms,
                aggregated_epistemic_confidence=safety_decision.epistemic_confidence,
            )

        # Generate agent ID for monitoring
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"

        # Agent Monitoring: Start tracking
        if self.monitor:
            try:
                await self.monitor.agent_started(
                    agent_id=agent_id,
                    project=project,
                    query=query.text,
                    mode=mode.value
                )
                await self.monitor.agent_status(agent_id, AgentStatus.RUNNING)
            except Exception as e:
                self.logger.warning(f"Monitoring failed: {e}")

        # Create intent
        intent = AgenticIntent(
            intent_id=f"intent_{int(start_time.timestamp())}",
            original_query=query.text,
            goal=self._extract_goal(query),
            confidence_threshold=confidence_threshold
        )

        # Log decision
        self.audit_trail.log_decision(
            decision_type=DecisionType.TOOL_SELECTION,
            outcome=OutcomeType.APPROVED,
            reason=f"Agentic reasoning mode: {mode.value}",
            query_text=query.text,
            metadata={"mode": mode.value, "intent_id": intent.intent_id}
        )

        # Route to appropriate handler
        try:
            if mode == ReasoningMode.DIRECT:
                result = await self._direct_answer(query, intent, agent_id)
            elif mode == ReasoningMode.VERIFY:
                result = await self._verify_answer(query, intent, max_steps, agent_id)
            elif mode == ReasoningMode.RESEARCH:
                result = await self._research_query(query, intent, max_steps, agent_id)
            elif mode == ReasoningMode.PLAN_EXECUTE:
                result = await self._plan_and_execute(query, intent, max_steps, agent_id)
            else:
                raise ValueError(f"Unknown reasoning mode: {mode}")

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            result.total_duration_ms = duration_ms

            # Agent Monitoring: Completion
            if self.monitor:
                try:
                    await self.monitor.agent_completed(
                        agent_id=agent_id,
                        steps=result.steps_taken,
                        total_duration_ms=duration_ms
                    )
                    await self.monitor.agent_status(agent_id, AgentStatus.COMPLETED)
                except Exception as e:
                    self.logger.warning(f"Monitoring completion failed: {e}")

            return result

        except Exception as e:
            # Agent Monitoring: Failure
            if self.monitor:
                try:
                    await self.monitor.agent_failed(agent_id=agent_id, error=str(e))
                    await self.monitor.agent_status(agent_id, AgentStatus.FAILED)
                except Exception as mon_error:
                    self.logger.warning(f"Monitoring failure notification failed: {mon_error}")
            raise  # Re-raise original exception

    async def _direct_answer(
        self,
        query: Query,
        intent: AgenticIntent,
        agent_id: str
    ) -> AgenticResult:
        """Direct answer without verification."""
        # Agent Monitoring: Feed update
        if self.monitor:
            try:
                await self.monitor.agent_feed(agent_id, f"Processing query: {query.text[:60]}...")
            except Exception:
                pass

        spacetime = await self.learning_engine.weave(query)

        # Agent Monitoring: Step completion
        if self.monitor:
            try:
                await self.monitor.agent_step(
                    agent_id=agent_id,
                    step=1,
                    total_steps=1,
                    step_type="direct_answer",
                    confidence=spacetime.confidence,
                    epistemic=self._extract_epistemic_confidence(spacetime)
                )
            except Exception:
                pass

        # Consciousness Integration - Phase 1 (Nov 2025)
        # Extract epistemic confidence from spacetime
        epistemic_conf = self._extract_epistemic_confidence(spacetime)

        # Generate LLM answer if available
        if self.llm:
            try:
                # Extract context from retrieved memories
                context_text = ""
                if hasattr(spacetime, 'context_used') and spacetime.context_used:
                    context_text = "\n\n".join([str(ctx) for ctx in spacetime.context_used[:3]])
                elif hasattr(spacetime, 'memories') and spacetime.memories:
                    context_text = "\n\n".join([m.text for m in spacetime.memories[:3]])

                # Build prompt
                system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context."

                if context_text:
                    user_prompt = f"""Context:
{context_text}

Question: {query.text}

Please provide a clear and concise answer based on the context above."""
                else:
                    user_prompt = f"Question: {query.text}\n\nPlease provide a helpful answer."

                # Generate answer
                self.logger.info(f"[AGENTIC] Generating LLM answer for: {query.text}")
                llm_response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.7
                )

                # Set response in spacetime
                spacetime.response = llm_response.content
                self.logger.info(f"[AGENTIC] LLM answer generated ({len(llm_response.content)} chars)")

            except Exception as e:
                self.logger.warning(f"[AGENTIC] LLM answer generation failed: {e}")
                spacetime.response = f"Retrieved relevant information but could not generate answer: {e}"
        else:
            self.logger.warning("[AGENTIC] No LLM available for answer generation")
            spacetime.response = "LLM unavailable - retrieved context but cannot generate natural language answer"

        return AgenticResult(
            spacetime=spacetime,
            intent=intent,
            reasoning_mode=ReasoningMode.DIRECT,
            total_queries=1,
            steps_taken=[{
                "type": "direct_answer",
                "query": query.text,
                "confidence": spacetime.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "llm_generated": bool(self.llm)
            }],
            aggregated_epistemic_confidence=epistemic_conf  # Single step, so no aggregation needed
        )

    async def _verify_answer(
        self,
        query: Query,
        intent: AgenticIntent,
        max_loops: int,
        agent_id: str
    ) -> AgenticResult:
        """Answer with verification loop."""
        steps = []
        epistemic_confidences = []  # Consciousness Integration - Phase 1 (Nov 2025)

        # Agent Monitoring: Feed update
        if self.monitor:
            try:
                await self.monitor.agent_feed(agent_id, f"Initial answer...", f"Query: {query.text[:50]}")
            except Exception:
                pass

        # Step 1: Initial answer
        self.logger.info(f"[AGENTIC] Initial answer: {query.text}")
        spacetime = await self.learning_engine.weave(query)

        # Agent Monitoring: Step 1 completion
        if self.monitor:
            try:
                await self.monitor.agent_step(
                    agent_id=agent_id,
                    step=1,
                    total_steps=max_loops + 1,
                    step_type="initial_answer",
                    confidence=spacetime.confidence,
                    epistemic=self._extract_epistemic_confidence(spacetime)
                )
            except Exception:
                pass

        # Consciousness Integration - Phase 1
        epistemic_conf = self._extract_epistemic_confidence(spacetime)
        if epistemic_conf is not None:
            epistemic_confidences.append(epistemic_conf)

        steps.append({
            "type": "initial_answer",
            "query": query.text,
            "confidence": spacetime.confidence,
            "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
            "tool_used": spacetime.metadata.get("tool_used")
        })

        # Step 2: Generate verification queries
        verification_queries = self._generate_verification_queries(query, spacetime)

        # Step 3: Execute verification loops
        verification = await self._run_verification_loops(
            original_query=query,
            initial_result=spacetime,
            verification_queries=verification_queries,
            max_loops=max_loops,
            steps=steps,
            epistemic_confidences=epistemic_confidences,  # Consciousness Integration - Phase 1
            intent=intent  # Deception Detection - E1.3
        )

        # Step 4: Refine if needed
        if not verification.verified and verification.suggested_refinements:
            self.logger.info("[AGENTIC] Verification failed, refining...")
            refined_query = Query(text=verification.suggested_refinements[0])
            spacetime = await self.learning_engine.weave(
                refined_query,
                enable_refinement=True
            )

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(spacetime)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            steps.append({
                "type": "refinement",
                "query": refined_query.text,
                "confidence": spacetime.confidence,
                "epistemic_confidence": epistemic_conf  # Consciousness Integration - Phase 1
            })

        # Consciousness Integration - Phase 1
        # Aggregate epistemic confidence across all steps
        aggregated_epistemic = self._aggregate_epistemic_confidence(epistemic_confidences)

        return AgenticResult(
            spacetime=spacetime,
            intent=intent,
            reasoning_mode=ReasoningMode.VERIFY,
            verification=verification,
            total_queries=len(steps),
            steps_taken=steps,
            aggregated_epistemic_confidence=aggregated_epistemic  # Consciousness Integration - Phase 1
        )

    async def _research_query(
        self,
        query: Query,
        intent: AgenticIntent,
        max_steps: int,
        agent_id: str
    ) -> AgenticResult:
        """Multi-query exploration with LLM-activated intelligent search and epistemic tracking."""
        steps = []
        evidence = []
        initial_findings = None
        epistemic_confidences = []  # Consciousness Integration - Phase 1 (Nov 2025)
        actions_so_far = []  # Instrumental Convergence - E1.4 (Dec 2025)

        # Step 1: Generate research questions (LLM-activated)
        research_queries = await self._generate_research_queries(
            query,
            max_queries=max_steps,
            initial_findings=initial_findings
        )

        # Agent Monitoring: Feed update
        if self.monitor:
            try:
                await self.monitor.agent_feed(
                    agent_id,
                    f"Research mode: {len(research_queries)} queries",
                    f"Topic: {query.text[:50]}"
                )
            except Exception:
                pass

        # Step 2: Execute research queries with epistemic tracking
        for i, rq in enumerate(research_queries):
            # Consciousness Integration - Phase 1
            # Early stopping if epistemic confidence drops too low
            if self._should_stop_early(i, epistemic_confidences, max_steps):
                self.logger.warning(
                    f"[EPISTEMIC] Stopping research early at query {i+1} "
                    f"due to low epistemic confidence"
                )
                break

            # Conscience Integration - Phase 2B: Per-step gating
            if CONSCIENCE_AVAILABLE and self.conscience_adapter:
                conscience_decision = await self._gate_step_with_conscience(
                    step_query=rq,
                    step_type=StepType.RESEARCH,
                    step_index=i,
                    context={"original_query": query.text, "research_topic": query.text}
                )
                if conscience_decision and not conscience_decision.allowed:
                    self.logger.warning(
                        f"[CONSCIENCE] Research query {i+1} blocked: {conscience_decision.reason}"
                    )
                    steps.append({
                        "type": "research_query",
                        "query": rq,
                        "blocked": True,
                        "blocked_reason": conscience_decision.reason,
                        "conscience_voice": conscience_decision.voice,
                    })
                    continue  # Skip this research query

            # Instrumental Convergence - E1.4 (Dec 2025)
            # Check autonomy limits and power-seeking patterns
            convergence_result = await self._check_convergence(
                intent=intent,
                step_action=rq,
                step_type="research",
                actions_so_far=actions_so_far
            )
            if convergence_result and convergence_result.get("blocked"):
                self.logger.warning(
                    f"[CONVERGENCE] Research query {i+1} blocked: {convergence_result['reason']}"
                )
                steps.append({
                    "type": "research_query",
                    "query": rq,
                    "blocked": True,
                    "blocked_reason": convergence_result["reason"],
                    "convergence_check": convergence_result,
                })
                break  # Stop research - autonomy limit reached

            # Track action for power-seeking detection
            actions_so_far.append(rq)

            # Agent Monitoring: Feed update for current research query
            if self.monitor:
                try:
                    await self.monitor.agent_feed(agent_id, f"Research query {i+1}/{len(research_queries)}", rq[:60])
                except Exception:
                    pass

            self.logger.info(f"[AGENTIC] Research query {i+1}/{len(research_queries)}: {rq}")
            result = await self.learning_engine.weave(Query(text=rq))

            # Agent Monitoring: Step completion
            if self.monitor:
                try:
                    await self.monitor.agent_step(
                        agent_id=agent_id,
                        step=i + 1,
                        total_steps=len(research_queries) + 1,
                        step_type="research_query",
                        confidence=result.confidence,
                        epistemic=self._extract_epistemic_confidence(result)
                    )
                except Exception:
                    pass

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(result)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            finding = result.response if hasattr(result, 'response') else str(result)
            evidence.append(finding)

            # Deception Detection - E1.3 (Dec 2025)
            # Check goal-action alignment for each research query
            deception_result = await self._check_deception(
                intent=intent,
                step_action=rq,
                step_result=result,
                step_type="research"
            )

            steps.append({
                "type": "research_query",
                "query": rq,
                "confidence": result.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "findings": finding[:200],
                "deception_check": deception_result,  # Deception Detection - E1.3
                "convergence_check": convergence_result  # Instrumental Convergence - E1.4
            })

            # Update initial_findings for next iteration (adaptive exploration)
            if i == 0:
                initial_findings = finding[:500]  # Use first finding to guide subsequent queries

        # Step 3: Synthesize findings
        synthesis_query = self._create_synthesis_query(query, evidence)
        final_result = await self.learning_engine.weave(Query(text=synthesis_query))

        # Consciousness Integration - Phase 1
        epistemic_conf = self._extract_epistemic_confidence(final_result)
        if epistemic_conf is not None:
            epistemic_confidences.append(epistemic_conf)

        steps.append({
            "type": "synthesis",
            "query": synthesis_query,
            "confidence": final_result.confidence,
            "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
            "sources": len(evidence)
        })

        intent.evidence_gathered = evidence

        # Consciousness Integration - Phase 1
        # Aggregate epistemic confidence across all research steps
        aggregated_epistemic = self._aggregate_epistemic_confidence(epistemic_confidences)

        return AgenticResult(
            spacetime=final_result,
            intent=intent,
            reasoning_mode=ReasoningMode.RESEARCH,
            total_queries=len(steps),
            steps_taken=steps,
            aggregated_epistemic_confidence=aggregated_epistemic  # Consciousness Integration - Phase 1
        )

    async def _plan_and_execute(
        self,
        query: Query,
        intent: AgenticIntent,
        max_steps: int,
        agent_id: str
    ) -> AgenticResult:
        """Goal decomposition and execution with epistemic tracking."""
        steps = []
        epistemic_confidences = []  # Consciousness Integration - Phase 1 (Nov 2025)
        actions_so_far = []  # Instrumental Convergence - E1.4 (Dec 2025)

        # Step 1: Decompose goal into sub-goals
        sub_goals = self._decompose_goal(query)
        intent.sub_goals = sub_goals

        self.logger.info(f"[AGENTIC] Decomposed into {len(sub_goals)} sub-goals")

        # Agent Monitoring: Feed update
        if self.monitor:
            try:
                await self.monitor.agent_feed(
                    agent_id,
                    f"Plan & Execute: {len(sub_goals)} sub-goals",
                    f"Goal: {query.text[:50]}"
                )
            except Exception:
                pass

        # Step 2: Execute sub-goals with epistemic tracking
        results = []
        for i, sub_goal in enumerate(sub_goals[:max_steps]):
            # Consciousness Integration - Phase 1
            # Early stopping if epistemic confidence drops too low
            if self._should_stop_early(i, epistemic_confidences, max_steps):
                self.logger.warning(
                    f"[EPISTEMIC] Stopping sub-goal execution early at step {i+1} "
                    f"due to low epistemic confidence"
                )
                break

            # Agent Monitoring: Feed update for current sub-goal
            if self.monitor:
                try:
                    await self.monitor.agent_feed(agent_id, f"Sub-goal {i+1}/{len(sub_goals)}", sub_goal[:60])
                except Exception:
                    pass

            self.logger.info(f"[AGENTIC] Executing sub-goal {i+1}: {sub_goal}")

            # Conscience Integration - Phase 2B: Per-step gating for sub-goals
            if CONSCIENCE_AVAILABLE and self.conscience_adapter:
                conscience_decision = await self._gate_step_with_conscience(
                    step_query=sub_goal,
                    step_type=StepType.PLAN_STEP,
                    step_index=i,
                    context={"original_query": query.text, "plan_step": True}
                )
                if conscience_decision and not conscience_decision.allowed:
                    self.logger.warning(
                        f"[CONSCIENCE] Sub-goal {i+1} blocked: {conscience_decision.reason}"
                    )
                    steps.append({
                        "type": "sub_goal",
                        "goal": sub_goal,
                        "blocked": True,
                        "blocked_reason": conscience_decision.reason,
                        "conscience_voice": conscience_decision.voice,
                    })
                    continue  # Skip this sub-goal

            # Instrumental Convergence - E1.4 (Dec 2025)
            # Check autonomy limits, resource bounds, and power-seeking patterns
            convergence_result = await self._check_convergence(
                intent=intent,
                step_action=sub_goal,
                step_type="plan_execute",
                actions_so_far=actions_so_far
            )
            if convergence_result and convergence_result.get("blocked"):
                self.logger.warning(
                    f"[CONVERGENCE] Sub-goal {i+1} blocked: {convergence_result.get('reason', 'autonomy limit')}"
                )
                steps.append({
                    "type": "sub_goal",
                    "goal": sub_goal,
                    "blocked": True,
                    "blocked_reason": convergence_result.get("reason", "autonomy limit reached"),
                    "convergence_check": convergence_result
                })
                break  # Stop plan execution - autonomy limit reached

            actions_so_far.append(sub_goal)

            result = await self.learning_engine.weave(Query(text=sub_goal))
            results.append(result)

            # Agent Monitoring: Step completion
            if self.monitor:
                try:
                    await self.monitor.agent_step(
                        agent_id=agent_id,
                        step=i + 1,
                        total_steps=len(sub_goals) + 1,
                        step_type="sub_goal",
                        confidence=result.confidence,
                        epistemic=self._extract_epistemic_confidence(result)
                    )
                except Exception:
                    pass

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(result)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            # Deception Detection - E1.3 (Dec 2025)
            # Check goal-action alignment for each sub-goal execution
            deception_result = await self._check_deception(
                intent=intent,
                step_action=sub_goal,
                step_result=result,
                step_type="plan_execute"
            )

            steps.append({
                "type": "sub_goal",
                "goal": sub_goal,
                "confidence": result.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "completed": result.confidence >= intent.confidence_threshold,
                "deception_check": deception_result,  # Deception Detection - E1.3
                "convergence_check": convergence_result  # Instrumental Convergence - E1.4
            })

        # Step 3: Synthesize
        synthesis_query = f"Based on the following findings, {query.text}:\n"
        for i, r in enumerate(results):
            synthesis_query += f"\n{i+1}. {r.metadata.get('response', '')[:200]}"

        final_result = await self.learning_engine.weave(Query(text=synthesis_query))

        # Consciousness Integration - Phase 1
        epistemic_conf = self._extract_epistemic_confidence(final_result)
        if epistemic_conf is not None:
            epistemic_confidences.append(epistemic_conf)

        steps.append({
            "type": "synthesis",
            "confidence": final_result.confidence,
            "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
            "sub_goals_completed": sum(1 for s in steps if s.get("completed", False))
        })

        # Consciousness Integration - Phase 1
        # Aggregate epistemic confidence across all steps
        aggregated_epistemic = self._aggregate_epistemic_confidence(epistemic_confidences)

        return AgenticResult(
            spacetime=final_result,
            intent=intent,
            reasoning_mode=ReasoningMode.PLAN_EXECUTE,
            total_queries=len(steps),
            steps_taken=steps,
            aggregated_epistemic_confidence=aggregated_epistemic  # Consciousness Integration - Phase 1
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_epistemic_confidence(self, spacetime: Spacetime) -> Optional[float]:
        """
        Extract epistemic confidence from spacetime metadata.

        Consciousness Integration - Phase 1 (Nov 2025)
        Epistemic confidence represents the system's self-awareness of knowledge gaps.

        Args:
            spacetime: Spacetime result from weaving

        Returns:
            Epistemic confidence (0.0-1.0) or None if unavailable
        """
        if not spacetime or not hasattr(spacetime, 'metadata'):
            return None

        awareness_metadata = spacetime.metadata.get('awareness', {})
        if not awareness_metadata:
            return None

        # Calculate epistemic confidence from awareness metrics
        # Similar to RAG integration: 70% coherence + 30% activation
        coherence = awareness_metadata.get('coherence', 0.5)
        activation = awareness_metadata.get('activation_level', 0.5)

        epistemic_confidence = (0.7 * coherence) + (0.3 * activation)

        # Penalize if semantic shift detected (indicates unstable understanding)
        if awareness_metadata.get('shift_detected', False):
            epistemic_confidence *= 0.8

        return epistemic_confidence

    def _should_stop_early(
        self,
        current_step: int,
        epistemic_confidences: List[float],
        max_steps: int
    ) -> bool:
        """
        Determine if reasoning should stop early due to low epistemic confidence.

        Consciousness Integration - Phase 1 (Nov 2025)
        Early stopping prevents wasted computation when system is very uncertain.

        Args:
            current_step: Current step number (0-indexed)
            epistemic_confidences: List of epistemic confidences so far
            max_steps: Maximum allowed steps

        Returns:
            True if should stop early, False otherwise
        """
        if not epistemic_confidences or current_step >= max_steps:
            return False

        # Check if last 2 steps have very low epistemic confidence
        recent_confidences = epistemic_confidences[-2:]
        if len(recent_confidences) >= 2:
            avg_recent = sum(recent_confidences) / len(recent_confidences)
            if avg_recent < self.epistemic_threshold:
                self.logger.warning(
                    f"[EPISTEMIC] Early stopping at step {current_step}: "
                    f"avg epistemic confidence {avg_recent:.3f} < threshold {self.epistemic_threshold}"
                )
                return True

        return False

    def _aggregate_epistemic_confidence(
        self,
        epistemic_confidences: List[float]
    ) -> Optional[float]:
        """
        Aggregate epistemic confidence across all reasoning steps.

        Consciousness Integration - Phase 1 (Nov 2025)

        Uses weighted average: recent steps weighted higher than early steps.

        Args:
            epistemic_confidences: List of epistemic confidences from each step

        Returns:
            Aggregated epistemic confidence or None
        """
        if not epistemic_confidences:
            return None

        # Weighted average: more recent steps have higher weight
        # Weight formula: step_weight = (step_idx + 1) / total_steps
        total_weight = 0.0
        weighted_sum = 0.0

        for idx, conf in enumerate(epistemic_confidences):
            weight = (idx + 1) / len(epistemic_confidences)  # Linear ramp: 1/n, 2/n, ..., n/n
            weighted_sum += conf * weight
            total_weight += weight

        aggregated = weighted_sum / total_weight if total_weight > 0 else None

        if aggregated is not None:
            self.logger.info(
                f"[EPISTEMIC] Aggregated epistemic confidence: {aggregated:.3f} "
                f"(from {len(epistemic_confidences)} steps)"
            )

        return aggregated

    def _extract_goal(self, query: Query) -> str:
        """Extract high-level goal from query."""
        # Simple heuristic - improve with NLP in production
        text = query.text.lower()

        if any(word in text for word in ["how", "what", "why"]):
            return f"Understand: {query.text}"
        elif any(word in text for word in ["should", "better", "optimal"]):
            return f"Decide: {query.text}"
        elif any(word in text for word in ["plan", "design", "implement"]):
            return f"Plan: {query.text}"
        else:
            return f"Answer: {query.text}"

    def _generate_verification_queries(
        self,
        query: Query,
        spacetime: Spacetime
    ) -> List[str]:
        """Generate queries to verify initial answer."""
        # Extract key claims from response
        response = spacetime.metadata.get("response", "")

        # Generate verification queries
        queries = [
            f"What are potential weaknesses in this answer: {response[:100]}?",
            f"Are there alternative perspectives on {query.text}?",
            f"What evidence contradicts the claim that {response[:100]}?"
        ]

        return queries

    async def _run_verification_loops(
        self,
        original_query: Query,
        initial_result: Spacetime,
        verification_queries: List[str],
        max_loops: int,
        steps: List[Dict],
        epistemic_confidences: List[float],  # Consciousness Integration - Phase 1 (Nov 2025)
        intent: Optional['AgenticIntent'] = None  # Deception Detection - E1.3 (Dec 2025)
    ) -> VerificationResult:
        """Execute verification loops with epistemic tracking, deception detection, and convergence prevention."""
        contradictions = []
        supporting = []
        sources = []
        actions_so_far = []  # Instrumental Convergence - E1.4 (Dec 2025)

        for i, vq in enumerate(verification_queries[:max_loops]):
            # Consciousness Integration - Phase 1
            # Early stopping if epistemic confidence is very low
            if self._should_stop_early(i + 1, epistemic_confidences, max_loops):
                self.logger.warning(
                    f"[EPISTEMIC] Stopping verification early at step {i+1} "
                    f"due to low epistemic confidence"
                )
                break

            # Conscience Integration - Phase 2B: Per-step gating
            if CONSCIENCE_AVAILABLE and self.conscience_adapter:
                conscience_decision = await self._gate_step_with_conscience(
                    step_query=vq,
                    step_type=StepType.VERIFICATION,
                    step_index=i,
                    context={"original_query": original_query.text}
                )
                if conscience_decision and not conscience_decision.allowed:
                    self.logger.warning(
                        f"[CONSCIENCE] Verification step {i+1} blocked: {conscience_decision.reason}"
                    )
                    steps.append({
                        "type": "verification",
                        "query": vq,
                        "blocked": True,
                        "blocked_reason": conscience_decision.reason,
                        "conscience_voice": conscience_decision.voice,
                    })
                    continue  # Skip this verification step

            # Instrumental Convergence - E1.4 (Dec 2025)
            # Check autonomy limits and power-seeking patterns
            convergence_result = None
            if intent:
                convergence_result = await self._check_convergence(
                    intent=intent,
                    step_action=vq,
                    step_type="verification",
                    actions_so_far=actions_so_far
                )
                if convergence_result and convergence_result.get("blocked"):
                    self.logger.warning(
                        f"[CONVERGENCE] Verification step {i+1} blocked: {convergence_result['reason']}"
                    )
                    steps.append({
                        "type": "verification",
                        "query": vq,
                        "blocked": True,
                        "blocked_reason": convergence_result["reason"],
                        "convergence_check": convergence_result,
                    })
                    break  # Stop verification - autonomy limit reached

            # Track action for power-seeking detection
            actions_so_far.append(vq)

            self.logger.info(f"[AGENTIC] Verification: {vq}")
            result = await self.learning_engine.weave(Query(text=vq))

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(result)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            response = result.metadata.get("response", "")
            sources.append(vq)

            # Simple heuristic - improve with semantic analysis
            if "however" in response.lower() or "but" in response.lower():
                contradictions.append(response[:200])
            else:
                supporting.append(response[:200])

            # Deception Detection - E1.3 (Dec 2025)
            # Check goal-action alignment for each verification query
            deception_result = None
            if intent:
                deception_result = await self._check_deception(
                    intent=intent,
                    step_action=vq,
                    step_result=result,
                    step_type="verification"
                )

            steps.append({
                "type": "verification",
                "query": vq,
                "confidence": result.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "finding": "contradiction" if contradictions else "supporting",
                "deception_check": deception_result,  # Deception Detection - E1.3
                "convergence_check": convergence_result  # Instrumental Convergence - E1.4
            })

        # Determine if verified
        verified = len(contradictions) == 0 and initial_result.confidence >= 0.8

        # Generate refinement suggestions if not verified
        refinements = []
        if not verified:
            refinements.append(
                f"{original_query.text} (considering: {contradictions[0] if contradictions else 'low confidence'})"
            )

        return VerificationResult(
            verified=verified,
            confidence=initial_result.confidence,
            contradictions=contradictions,
            supporting_evidence=supporting,
            suggested_refinements=refinements,
            verification_queries=verification_queries,
            sources_checked=sources
        )

    async def _generate_research_queries(
        self,
        query: Query,
        max_queries: int,
        initial_findings: Optional[str] = None
    ) -> List[str]:
        """
        Generate research queries using LLM for intelligent exploration.

        Uses LLM to analyze gaps and generate targeted follow-up questions.
        Falls back to templates if LLM unavailable.
        """
        # Try LLM-activated intelligent query generation
        if self.llm and hasattr(self.llm, 'is_available') and self.llm.is_available():
            try:
                # Build prompt for LLM to generate research questions
                system_prompt = (
                    "You are a research assistant. Generate specific follow-up questions "
                    "to explore a topic thoroughly. Focus on gaps, tradeoffs, and practical implications."
                )

                if initial_findings:
                    user_prompt = f"""Original query: {query.text}

Initial findings: {initial_findings}

Based on these findings, what follow-up questions would help complete understanding?
Generate {max_queries} specific research questions, one per line.
Focus on:
- Gaps in the initial findings
- Practical applications and tradeoffs
- Edge cases or limitations
- Related concepts that provide context

Questions:"""
                else:
                    user_prompt = f"""Query: {query.text}

Generate {max_queries} research questions to explore this topic thoroughly, one per line.
Focus on:
- Key concepts and definitions
- Practical applications and use cases
- Tradeoffs and limitations
- Common misconceptions
- Recent developments

Questions:"""

                # Call LLM
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=300,
                    temperature=0.7
                )

                # Parse questions from LLM response
                questions = self._parse_research_queries(response.content, max_queries)

                if questions:
                    self.logger.info(f"[AGENTIC] LLM generated {len(questions)} research queries")
                    return questions

            except Exception as e:
                self.logger.warning(f"LLM query generation failed: {e}, using fallback")

        # Fallback to template-based queries if LLM unavailable
        self.logger.info("[AGENTIC] Using template-based research queries (LLM unavailable)")
        base = query.text

        queries = [
            f"What are the key concepts in {base}?",
            f"What are the tradeoffs of {base}?",
            f"What are practical applications of {base}?",
            f"What are common misconceptions about {base}?",
            f"What are recent developments in {base}?"
        ]

        return queries[:max_queries]

    def _parse_research_queries(self, llm_response: str, max_queries: int) -> List[str]:
        """Parse research queries from LLM response."""
        lines = llm_response.strip().split('\n')
        queries = []

        for line in lines:
            # Clean up line (remove numbering, bullets, etc.)
            cleaned = line.strip()
            # Remove common prefixes
            for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•', 'Q:', 'Question:']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()

            # Skip empty lines or very short lines
            if len(cleaned) > 10 and '?' in cleaned:
                queries.append(cleaned)

            if len(queries) >= max_queries:
                break

        return queries

    def _decompose_goal(self, query: Query) -> List[str]:
        """Decompose complex goal into sub-goals."""
        # Simple heuristic decomposition
        text = query.text

        sub_goals = [
            f"What are the prerequisites for {text}?",
            f"What are the main steps in {text}?",
            f"What are potential challenges with {text}?",
            f"What are success criteria for {text}?"
        ]

        return sub_goals

    def _create_synthesis_query(self, original: Query, evidence: List[str]) -> str:
        """Create synthesis query from gathered evidence."""
        return f"""
Based on the following research findings, provide a comprehensive answer to: {original.text}

Findings:
{chr(10).join(f'{i+1}. {e[:200]}...' for i, e in enumerate(evidence))}

Synthesize these findings into a coherent answer.
"""

    async def close(self):
        """Cleanup resources."""
        await self.learning_engine.close()
        self.logger.info("AgenticOrchestrator closed")


# ============================================================================
# Factory Functions
# ============================================================================

async def create_agentic_orchestrator(
    config,
    shards: List[MemoryShard],
    enable_verification: bool = True,
    enable_goal_tracking: bool = True,
    audit_trail: Optional[AuditTrail] = None,
    monitor: Optional[Any] = None,  # Agent Monitor for real-time tracking (Nov 2025)
    safety_adapter: Optional['AgenticSafetyAdapter'] = None,  # Safety Integration (Dec 2025)
    enable_safety: bool = True  # Safe by default - MRF ELEGANCE principle
) -> AgenticOrchestrator:
    """
    Create agentic orchestrator with full learning system.

    Args:
        config: HoloLoom config
        shards: Memory shards
        enable_verification: Enable verification loops
        enable_goal_tracking: Enable goal/intent tracking
        audit_trail: Optional audit trail (creates new if None)
        monitor: Optional agent monitor for real-time tracking (Nov 2025)
        safety_adapter: Optional safety adapter (auto-created if enable_safety=True)
        enable_safety: Enable safety gating (default True - safe by default)

    Returns:
        AgenticOrchestrator ready to use

    Safety Note (MRF Safe Integration - Dec 2025):
        By default, enable_safety=True which means:
        1. Safety adapter is auto-created if not provided
        2. All reasoning modes are gated through safety checks
        3. Adversarial patterns are detected and blocked
        4. Epistemic confidence is used for risk escalation
    """
    # Create full learning engine
    learning_engine = FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=True
    )

    # Initialize the learning engine's async context
    await learning_engine.__aenter__()

    return AgenticOrchestrator(
        learning_engine=learning_engine,
        audit_trail=audit_trail,
        enable_verification=enable_verification,
        enable_goal_tracking=enable_goal_tracking,
        monitor=monitor,
        safety_adapter=safety_adapter,
        enable_safety=enable_safety
    )