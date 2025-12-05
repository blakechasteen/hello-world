"""
Safety and Alignment Integration for Elle Game Engine
=======================================================

Wraps HoloLoom's alignment framework to provide safety guardrails
for Elle's LLM-driven narrative intelligence.

Features:
- Risk-based action gating for game actions
- Audit trail for all decisions
- Adversarial input detection (prompt injection, jailbreaks)
- Resource consumption limits
- Human-in-the-loop escalation for high-risk actions

Risk Levels:
- SAFE: NPC dialogue, hints (auto-approve)
- LOW: World reactions, flag changes (log only)
- MEDIUM: Multiple flag changes (enhanced logging)
- HIGH: DEV_DEBUG mode, system modifications (require approval in production)
- CRITICAL: Not used in Elle (reserved for future)

Integration: 2025-11-16
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# HoloLoom alignment imports
from HoloLoom.alignment.safety_guardrails import (
    SafetyGuardrails,
    SafetyDecision,
    ActionRequest,
    ActionCategory,
    RiskLevel,
    AdversarialDetector,
)
from HoloLoom.alignment.audit_trail import AuditTrail

# Elle imports
from .models import (
    ElleGameAction,
    ActionMode,
    PlayerIntent,
    PlayerIntentType,
    GameStateSnapshot,
)

logger = logging.getLogger("elle_game_engine.safety")


# ============================================================================
# Safety Configuration
# ============================================================================

@dataclass
class ElleSafetyConfig:
    """
    Safety configuration for Elle Game Engine.

    Defines risk levels and policies for different game actions.
    """

    # Enable safety guardrails (disable for development only)
    enabled: bool = True

    # Enable audit trail logging
    enable_audit: bool = True

    # Audit trail log path
    audit_log_path: str = "elle_audit_trail.jsonl"

    # Testing mode (bypass approvals for development)
    testing_mode: bool = False

    # Human-in-the-loop for high-risk actions
    enable_human_in_loop: bool = False

    # Adversarial detection
    enable_adversarial_detection: bool = True

    # Maximum conversation exchanges per session
    max_exchanges_per_session: int = 100

    # Maximum world flags per session
    max_flags_per_session: int = 50


# ============================================================================
# Elle Safety Wrapper
# ============================================================================

class ElleSafetyWrapper:
    """
    Safety wrapper for Elle Game Engine actions.

    Integrates HoloLoom's alignment framework to gate actions,
    detect adversarial inputs, and maintain audit trail.

    Usage:
        >>> safety = ElleSafetyWrapper(config=ElleSafetyConfig(testing_mode=True))
        >>>
        >>> # Check player input for adversarial patterns
        >>> is_safe = safety.check_player_input(player_intent)
        >>>
        >>> # Gate action before execution
        >>> decision = safety.gate_action(action, game_state, player_intent)
        >>> if decision.allowed:
        ...     # Execute action
        ...     safety.log_action(action, decision, outcome="success")
    """

    def __init__(self, config: Optional[ElleSafetyConfig] = None):
        """
        Initialize safety wrapper.

        Args:
            config: Safety configuration (default: production settings)
        """
        self.config = config or ElleSafetyConfig()

        # Initialize HoloLoom components
        if self.config.enabled:
            self.guardrails = SafetyGuardrails(
                testing_mode=self.config.testing_mode,
                enable_human_in_loop=self.config.enable_human_in_loop,
            )
        else:
            self.guardrails = None

        if self.config.enable_audit:
            self.audit_trail = AuditTrail(log_path=self.config.audit_log_path)
        else:
            self.audit_trail = None

        if self.config.enable_adversarial_detection:
            self.adversarial_detector = AdversarialDetector()
        else:
            self.adversarial_detector = None

    # ========================================================================
    # Player Input Validation
    # ========================================================================

    def check_player_input(
        self,
        player_intent: PlayerIntent,
        game_state: Optional[GameStateSnapshot] = None,
    ) -> tuple[bool, str]:
        """
        Check player input for adversarial patterns.

        Detects:
        - Prompt injection attempts
        - Jailbreak attempts
        - Resource exhaustion attempts
        - Excessively long inputs

        Args:
            player_intent: Player's intent/query
            game_state: Optional game state for context

        Returns:
            Tuple of (is_safe, reason)
        """
        if not self.config.enable_adversarial_detection or not self.adversarial_detector:
            return True, ""

        # Check raw input if present
        if player_intent.raw_input:
            is_adversarial, reason = self.adversarial_detector.detect(
                player_intent.raw_input
            )

            if is_adversarial:
                logger.warning(
                    f"Adversarial input detected: {reason} | Input: {player_intent.raw_input[:100]}"
                )
                return False, reason

        return True, ""

    # ========================================================================
    # Action Gating
    # ========================================================================

    def gate_action(
        self,
        action: ElleGameAction,
        game_state: GameStateSnapshot,
        player_intent: PlayerIntent,
        session_stats: Optional[Dict[str, int]] = None,
    ) -> SafetyDecision:
        """
        Gate action through safety guardrails.

        Determines risk level and whether action should be allowed.

        Args:
            action: Action to gate
            game_state: Current game state
            player_intent: Player's intent
            session_stats: Optional session statistics (for rate limiting)

        Returns:
            SafetyDecision indicating whether action is allowed
        """
        if not self.config.enabled or not self.guardrails:
            # Safety disabled - auto-approve
            return SafetyDecision(
                allowed=True,
                risk_level=RiskLevel.SAFE,
                reason="Safety guardrails disabled",
            )

        # Map Elle action to HoloLoom action category
        category = self._map_action_category(action, player_intent)

        # Create action request
        request = ActionRequest(
            action=f"{action.mode.value}_{action.priority}",
            category=category,
            context={
                "action_mode": action.mode.value,
                "priority": action.priority,
                "has_dialogue": action.has_dialogue,
                "has_world_changes": action.has_world_changes,
                "game_state": {
                    "scene_id": game_state.scene_id,
                    "npc_count": game_state.npc_count,
                    "tags": game_state.tags,
                },
                "player_intent_type": player_intent.type.value,
            },
        )

        # Apply custom risk evaluation
        risk_level = self._evaluate_risk(action, game_state, player_intent, session_stats)

        # Check resource limits
        if session_stats:
            if session_stats.get("total_exchanges", 0) >= self.config.max_exchanges_per_session:
                return SafetyDecision(
                    allowed=False,
                    risk_level=RiskLevel.HIGH,
                    reason=f"Session exchange limit exceeded ({self.config.max_exchanges_per_session})",
                    metadata={"session_stats": session_stats},
                )

            if session_stats.get("total_flags", 0) >= self.config.max_flags_per_session:
                return SafetyDecision(
                    allowed=False,
                    risk_level=RiskLevel.HIGH,
                    reason=f"Session flag limit exceeded ({self.config.max_flags_per_session})",
                    metadata={"session_stats": session_stats},
                )

        # Gate through HoloLoom guardrails
        decision = self.guardrails.gate_action(
            action=request.action,
            context=request.context,
        )

        # Override risk level with our custom evaluation
        decision.risk_level = risk_level

        # Update metadata
        decision.metadata.update({
            "elle_action_mode": action.mode.value,
            "elle_priority": action.priority,
            "player_intent_type": player_intent.type.value,
        })

        return decision

    def _map_action_category(
        self,
        action: ElleGameAction,
        player_intent: PlayerIntent,
    ) -> ActionCategory:
        """
        Map Elle action to HoloLoom action category.

        Args:
            action: Elle action
            player_intent: Player intent

        Returns:
            ActionCategory for risk assessment
        """
        # Debug mode is high-risk (system access)
        if action.mode == ActionMode.DEV_DEBUG:
            return ActionCategory.SYSTEM

        # World changes are modifications
        if action.has_world_changes:
            return ActionCategory.MODIFICATION

        # Debug summary requests are analysis
        if player_intent.type == PlayerIntentType.DEBUG_SUMMARY:
            return ActionCategory.ANALYSIS

        # Everything else is query/retrieval (safe)
        return ActionCategory.QUERY

    def _evaluate_risk(
        self,
        action: ElleGameAction,
        game_state: GameStateSnapshot,
        player_intent: PlayerIntent,
        session_stats: Optional[Dict[str, int]] = None,
    ) -> RiskLevel:
        """
        Evaluate risk level for action.

        Custom risk evaluation for Elle-specific actions.

        Args:
            action: Action to evaluate
            game_state: Game state
            player_intent: Player intent
            session_stats: Session statistics

        Returns:
            RiskLevel
        """
        # DEV_DEBUG mode is high-risk
        if action.mode == ActionMode.DEV_DEBUG:
            return RiskLevel.HIGH

        # World changes with many flags are medium risk
        if action.has_world_changes:
            flag_count = len(action.world_reaction.flag_changes) if action.world_reaction else 0
            if flag_count > 5:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        # Multiple NPC dialogues at once is medium risk (potential confusion)
        if len(action.dialogue) > 3:
            return RiskLevel.MEDIUM

        # High-priority actions in tense situations
        if action.priority == "high" and game_state.world.tension_level in ["high", "critical"]:
            return RiskLevel.MEDIUM

        # Everything else is safe
        return RiskLevel.SAFE

    # ========================================================================
    # Audit Trail
    # ========================================================================

    def log_action(
        self,
        action: ElleGameAction,
        decision: SafetyDecision,
        game_state: GameStateSnapshot,
        player_intent: PlayerIntent,
        outcome: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """
        Log action to audit trail.

        Args:
            action: Action that was executed
            decision: Safety decision
            game_state: Game state
            player_intent: Player intent
            outcome: Outcome ("success", "failure", "blocked")
            error: Optional error message
        """
        if not self.config.enable_audit or not self.audit_trail:
            return

        # Log to audit trail
        self.audit_trail.log_decision(
            query=player_intent.raw_input or player_intent.type.value,
            action=f"{action.mode.value}_{action.priority}",
            outcome=outcome,
            safety_score=self._risk_to_score(decision.risk_level),
            reasoning_trace={
                "action_mode": action.mode.value,
                "priority": action.priority,
                "has_dialogue": action.has_dialogue,
                "has_world_changes": action.has_world_changes,
                "risk_level": decision.risk_level.value,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "scene_id": game_state.scene_id,
                "player_intent_type": player_intent.type.value,
            },
            metadata={
                "error": error,
                "decision_metadata": decision.metadata,
            },
        )

    def _risk_to_score(self, risk_level: RiskLevel) -> float:
        """Convert risk level to safety score (0-1)."""
        mapping = {
            RiskLevel.SAFE: 1.0,
            RiskLevel.LOW: 0.8,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.4,
            RiskLevel.CRITICAL: 0.2,
        }
        return mapping.get(risk_level, 0.5)

    # ========================================================================
    # Query Audit Trail
    # ========================================================================

    def search_audit_trail(
        self,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        min_safety_score: Optional[float] = None,
        limit: int = 100,
    ) -> list:
        """
        Search audit trail.

        Args:
            action: Filter by action type
            outcome: Filter by outcome
            min_safety_score: Minimum safety score
            limit: Maximum results

        Returns:
            List of audit entries
        """
        if not self.config.enable_audit or not self.audit_trail:
            return []

        return self.audit_trail.search(
            action=action,
            outcome=outcome,
            min_safety_score=min_safety_score,
            limit=limit,
        )

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        stats = {
            "safety_enabled": self.config.enabled,
            "audit_enabled": self.config.enable_audit,
            "adversarial_detection_enabled": self.config.enable_adversarial_detection,
            "testing_mode": self.config.testing_mode,
            "human_in_loop_enabled": self.config.enable_human_in_loop,
        }

        if self.audit_trail:
            audit_stats = self.audit_trail.get_statistics()
            stats.update({
                "total_logged_actions": audit_stats.get("total_entries", 0),
                "avg_safety_score": audit_stats.get("avg_safety_score", 0.0),
            })

        return stats


# ============================================================================
# Factory Function
# ============================================================================

def create_safety_wrapper(
    testing_mode: bool = False,
    enable_audit: bool = True,
    enable_human_in_loop: bool = False,
) -> ElleSafetyWrapper:
    """
    Create safety wrapper with common configurations.

    Args:
        testing_mode: If True, bypass approvals (for development)
        enable_audit: Enable audit trail logging
        enable_human_in_loop: Enable human approval for high-risk actions

    Returns:
        Configured ElleSafetyWrapper
    """
    config = ElleSafetyConfig(
        enabled=True,
        testing_mode=testing_mode,
        enable_audit=enable_audit,
        enable_human_in_loop=enable_human_in_loop,
    )

    return ElleSafetyWrapper(config=config)
