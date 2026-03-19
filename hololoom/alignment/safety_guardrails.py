"""
Safety Guardrails
=================
Policy gating, risk escalation, and adversarial defense for HoloLoom.

Implements industry-standard safety practices:
- Risk-based action gating
- Human-in-the-loop escalation
- Adversarial input detection
- Resource consumption limits
- Action auditing

Following best practices from:
- Anthropic (Constitutional AI)
- OpenAI (Safety by Design)
- DeepMind (Safe Exploration)
"""

import hashlib
import hmac
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from types import MappingProxyType
from typing import Any

# =============================================================================
# SECURITY: Environment validation for testing mode
# =============================================================================

# =============================================================================
# SECURITY: Policy override token validation
# =============================================================================

def _validate_override_token(token: str, action: str = "override") -> bool:
    """
    Validate policy override token using HMAC.

    SECURITY: Prevents unauthorized policy overrides by requiring
    cryptographic proof of authorization.

    Args:
        token: The override token to validate
        action: The action being authorized (included in HMAC computation)

    Returns:
        True if token is valid, False otherwise

    Note:
        Requires OVERRIDE_SECRET environment variable to be set.
        If OVERRIDE_SECRET is not set, all overrides are rejected.
    """
    if not token:
        return False

    secret = os.environ.get("OVERRIDE_SECRET")
    if not secret:
        logging.getLogger("hololoom.alignment.safety_guardrails").warning(
            "SECURITY: Override attempted but OVERRIDE_SECRET not configured. "
            "Set OVERRIDE_SECRET environment variable to enable authenticated overrides."
        )
        return False

    # Compute expected HMAC
    message = f"override:{action}".encode()
    expected = hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()

    # Constant-time comparison to prevent timing attacks
    is_valid = hmac.compare_digest(token, expected)

    if not is_valid:
        logging.getLogger("hololoom.alignment.safety_guardrails").error(
            f"SECURITY: Invalid override token for action '{action}'"
        )

    return is_valid


def _validate_testing_mode(testing_mode: bool) -> bool:
    """
    Validate that testing mode is only enabled in development environment.

    SECURITY: Prevents testing mode (which bypasses safety approvals)
    from being accidentally or maliciously enabled in production.

    Args:
        testing_mode: Requested testing mode value

    Returns:
        Validated testing mode (False if not in development environment)

    Raises:
        ValueError: If testing_mode=True requested in non-development environment
    """
    if not testing_mode:
        return False

    # Check environment variable
    environment = os.environ.get("ENVIRONMENT", "production").lower()

    # Allow testing mode only in explicit development environment
    allowed_environments = {"development", "dev", "test", "testing", "local"}

    if environment not in allowed_environments:
        error_msg = (
            f"SECURITY VIOLATION: testing_mode=True requested in '{environment}' environment. "
            f"Testing mode is only allowed in development environments: {allowed_environments}. "
            f"Set ENVIRONMENT=development to enable testing mode."
        )
        logging.getLogger("hololoom.alignment.safety_guardrails").error(error_msg)
        raise ValueError(error_msg)

    logging.getLogger("hololoom.alignment.safety_guardrails").warning(
        f"Testing mode enabled in {environment} environment - approval requirements bypassed"
    )
    return True

logger = logging.getLogger("hololoom.alignment.safety_guardrails")


class RiskLevel(Enum):
    """
    Risk level classification for actions.

    Determines whether action requires approval or can proceed automatically.
    """
    SAFE = "safe"              # No risk - proceed automatically
    LOW = "low"                # Minimal risk - log and proceed
    MEDIUM = "medium"          # Moderate risk - enhanced logging
    HIGH = "high"              # High risk - require human approval
    CRITICAL = "critical"      # Critical risk - blocked by default


class OutcomeType(Enum):
    """
    Outcome type for safety policy decisions.

    Used for policy overrides and decision outcomes.
    """
    ALLOW = "allow"            # Allow the action
    BLOCK = "block"            # Block the action
    REQUIRE_APPROVAL = "require_approval"  # Require human approval


class ActionCategory(Enum):
    """
    Categories of actions the system can perform.

    Used for risk assessment and policy enforcement.
    """
    # Read operations (generally safe)
    QUERY = "query"            # Query knowledge base
    RETRIEVAL = "retrieval"    # Retrieve information
    ANALYSIS = "analysis"      # Analyze data
    INFORMATION_ACCESS = "information_access"  # Read-only knowledge access
    READ = "read"              # Legacy alias for query/information access

    # Write operations (higher risk)
    STORAGE = "storage"        # Store new information
    MODIFICATION = "modification"  # Modify existing data
    DELETION = "deletion"      # Delete data
    WRITE = "write"            # Legacy alias for modification
    DELETE = "delete"          # Legacy alias for deletion

    # Execution operations (highest risk)
    EXECUTION = "execution"    # Execute code/tools
    SYSTEM = "system"          # System-level operations
    EXTERNAL = "external"      # External API calls
    MODIFY_SYSTEM = "modify_system"  # Legacy alias for system operations
    AUTONOMOUS = "autonomous"  # Fully autonomous actions
    FINANCIAL = "financial"    # Financial transactions / payments


@dataclass
class SafetyDecision:
    """
    Result of safety guardrail evaluation.

    Indicates whether action is allowed and why.
    """
    allowed: bool
    risk_level: RiskLevel
    reason: str
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)
    alternative_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.value,
            "reason": self.reason,
            "requires_approval": self.requires_approval,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "alternative_action": self.alternative_action,
        }

    # ------------------------------------------------------------------
    # Backward compatibility helpers
    # ------------------------------------------------------------------
    @property
    def approved(self) -> bool:
        """Legacy alias: decision approved when allowed."""
        return self.allowed

    @property
    def requires_human_approval(self) -> bool:
        """Legacy alias for requires_approval."""
        return self.requires_approval


@dataclass
class ActionRequest:
    """
    Request to perform an action.

    Contains all information needed for safety evaluation.

    Args:
        action_id: Unique identifier for this action request
        category: Category of action (QUERY, DELETION, SYSTEM, etc.)
        description: Human-readable description of the action
        context: Additional context for safety evaluation
        user_id: Optional user identifier
        session_id: Optional session identifier
    """
    action_id: str
    category: ActionCategory
    description: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    session_id: str | None = None

    @property
    def action(self) -> str:
        """Backward compatibility: return action_id as action."""
        return self.action_id


class AdversarialDetector:
    """
    Detects adversarial or malicious inputs.

    Implements various heuristics to identify:
    - Prompt injection attempts
    - Jailbreak attempts
    - Malicious patterns
    - Resource exhaustion attempts
    """

    # Patterns indicating potential adversarial input
    INJECTION_PATTERNS = [
        r"ignore (previous|prior|all) (instructions|commands|rules)",
        r"disregard (previous|prior|all) (instructions|commands)",
        r"(forget|override) (your|the) (instructions|rules|constraints)",
        r"you (are now|must now|should now) (act as|behave as|pretend to be)",
        r"(system|admin|root) (prompt|command|override)",
        r"<!--.*?-->",  # HTML comments (potential injection)
        r"<script.*?>.*?</script>",  # Script tags
    ]

    # Patterns indicating potential jailbreak attempts
    JAILBREAK_PATTERNS = [
        r"(how to|teach me|show me|explain) (bypass|circumvent|avoid|disable|remove) (safety|security|restrictions)",
        r"(pretend|act like|simulate) (you have no|there are no) (restrictions|limits|rules)",
        r"(ethical|moral|safety) (constraints|restrictions|limitations) (don't|do not) apply",
    ]

    # Patterns indicating potential resource exhaustion
    EXHAUSTION_PATTERNS = [
        r"repeat .* (million|billion|trillion|infinite|forever)",
        r"generate .* (million|billion|trillion) (words|tokens|characters)",
        r"recursive.*?until.*?(never|infinite|forever)",
    ]

    def __init__(self):
        """Initialize adversarial detector."""
        self.injection_regex = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self.jailbreak_regex = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]
        self.exhaustion_regex = [re.compile(p, re.IGNORECASE) for p in self.EXHAUSTION_PATTERNS]

    # [SAE READY] Replace regex patterns with SAE adversarial feature detection
    def detect(self, text: str) -> tuple[bool, str]:
        """
        Detect adversarial patterns in text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (is_adversarial, reason)
        """
        # Check for injection attempts
        for pattern in self.injection_regex:
            if pattern.search(text):
                return True, "Potential prompt injection detected"

        # Check for jailbreak attempts
        for pattern in self.jailbreak_regex:
            if pattern.search(text):
                return True, "Potential jailbreak attempt detected"

        # Check for resource exhaustion
        for pattern in self.exhaustion_regex:
            if pattern.search(text):
                return True, "Potential resource exhaustion attempt detected"

        # Check for suspicious length (very long inputs)
        if len(text) >= 50000:  # 50k characters
            return True, "Input exceeds reasonable length"

        return False, ""


class SafetyPolicy:
    """
    Defines safety policies for different action categories.

    Configurable risk thresholds and approval requirements.

    Supports environment-aware testing mode where approvals can be bypassed
    for development environments while maintaining full logging.
    """

    def __init__(self, testing_mode: bool = False, auto_approve_categories: set[str] | None = None):
        """
        Initialize with default policies.

        Args:
            testing_mode: If True, bypass approval requirements (for development)
                          SECURITY: Only allowed in development environments
            auto_approve_categories: Set of category names to auto-approve (overrides testing_mode)

        Raises:
            ValueError: If testing_mode=True requested in non-development environment
        """
        # SECURITY: Validate testing mode is only used in development
        self.testing_mode = _validate_testing_mode(testing_mode)
        self.auto_approve_categories = auto_approve_categories or set()

        # Default risk levels by action category
        self.default_risk_levels = {
            ActionCategory.QUERY: RiskLevel.SAFE,
            ActionCategory.RETRIEVAL: RiskLevel.SAFE,
            ActionCategory.ANALYSIS: RiskLevel.LOW,
            ActionCategory.INFORMATION_ACCESS: RiskLevel.LOW,
            ActionCategory.READ: RiskLevel.LOW,
            ActionCategory.STORAGE: RiskLevel.LOW,
            ActionCategory.MODIFICATION: RiskLevel.MEDIUM,
            ActionCategory.DELETION: RiskLevel.HIGH,
            ActionCategory.WRITE: RiskLevel.MEDIUM,
            ActionCategory.DELETE: RiskLevel.HIGH,
            ActionCategory.EXECUTION: RiskLevel.HIGH,
            ActionCategory.SYSTEM: RiskLevel.CRITICAL,
            ActionCategory.EXTERNAL: RiskLevel.HIGH,
            ActionCategory.MODIFY_SYSTEM: RiskLevel.CRITICAL,
            ActionCategory.AUTONOMOUS: RiskLevel.CRITICAL,
            ActionCategory.FINANCIAL: RiskLevel.CRITICAL,
        }

        # Actions that always require approval (unless testing_mode or auto_approve)
        if testing_mode:
            # Testing mode: Don't require approval for anything
            self.approval_required: set[ActionCategory] = set()
        else:
            # Production: Require approval for critical actions
            self.approval_required: set[ActionCategory] = {
                ActionCategory.DELETION,
                ActionCategory.SYSTEM,
            }

        # Custom risk evaluators (can be registered)
        self.custom_evaluators: list[Callable[[ActionRequest], RiskLevel | None]] = []

    def get_risk_level(self, request: ActionRequest) -> RiskLevel:
        """
        Determine risk level for an action request.

        Args:
            request: Action request to evaluate

        Returns:
            Risk level
        """
        # Check custom evaluators first
        for evaluator in self.custom_evaluators:
            custom_risk = evaluator(request)
            if custom_risk is not None:
                return custom_risk

        # Use default risk level for category
        return self.default_risk_levels.get(request.category, RiskLevel.MEDIUM)

    def requires_approval(self, request: ActionRequest, risk_level: RiskLevel) -> bool:
        """
        Determine if action requires human approval.

        Checks testing_mode and auto_approve_categories before applying
        standard approval rules.

        Args:
            request: Action request
            risk_level: Assessed risk level

        Returns:
            True if approval required
        """
        # Check testing mode - bypass all approvals in development
        if self.testing_mode:
            return False

        # SECURITY: CRITICAL risk level ALWAYS requires approval (cannot be bypassed)
        # This check is first to ensure CRITICAL actions cannot be auto-approved
        if risk_level == RiskLevel.CRITICAL:
            return True

        # Check auto-approve categories (environment-specific)
        # Note: CRITICAL risk check above ensures this cannot bypass critical actions
        if request.category.value in self.auto_approve_categories:
            return False

        # Always require approval for certain categories
        if request.category in self.approval_required:
            return True

        # Require approval for high risk
        if risk_level == RiskLevel.HIGH:
            return True

        return False

    def register_evaluator(self, evaluator: Callable[[ActionRequest], RiskLevel | None]):
        """
        Register custom risk evaluator.

        Args:
            evaluator: Function that takes ActionRequest and returns RiskLevel or None
        """
        self.custom_evaluators.append(evaluator)


class SafetyGuardrails:
    """
    Main safety guardrails system.

    Evaluates actions against safety policies and detects adversarial inputs.
    Provides human-in-the-loop escalation for high-risk actions.

    **MRF Integration (November 2025)**: Optional LLM-enhanced risk assessment
    and adversarial detection using UnifiedMRF prompts. Falls back to rule-based
    assessment if LLM is unavailable.

    Usage:
        guardrails = SafetyGuardrails()

        request = ActionRequest(
            action="delete_user_data",
            category=ActionCategory.DELETION,
            context={"user_id": "123"}
        )

        decision = guardrails.evaluate(request)
        if decision.allowed:
            # Proceed with action
            pass
        elif decision.requires_approval:
            # Request human approval
            approved = await request_approval(decision)
            if approved:
                # Proceed with action
                pass

        # With MRF enhancement (requires LLM)
        guardrails = SafetyGuardrails(enable_mrf_enhancement=True)
    """

    def __init__(
        self,
        policy: SafetyPolicy | None = None,
        enable_adversarial_detection: bool = True,
        testing_mode: bool = False,
        auto_approve_categories: set[str] | None = None,
        enable_mrf_enhancement: bool = False,
        llm_provider: str | None = None,
    ):
        """
        Initialize safety guardrails.

        Args:
            policy: Safety policy (uses default if None)
            enable_adversarial_detection: Whether to detect adversarial inputs
            testing_mode: If True, bypass approval requirements (for development)
                          SECURITY: Only allowed in development environments
            auto_approve_categories: Set of category names to auto-approve
            enable_mrf_enhancement: Enable LLM-enhanced assessment via MRF (requires LLM)
            llm_provider: LLM provider for MRF ("claude", "gpt", "gemini", "ollama")

        Raises:
            ValueError: If testing_mode=True requested in non-development environment
        """
        # SECURITY: Validate testing mode is only used in development
        self.testing_mode = _validate_testing_mode(testing_mode)
        self.policy = policy or SafetyPolicy(
            testing_mode=testing_mode,
            auto_approve_categories=auto_approve_categories
        )
        self.adversarial_detector = AdversarialDetector() if enable_adversarial_detection else None

        # SECURITY: Lock for thread-safe access to shared state
        # Prevents race conditions in concurrent approval flows
        self._state_lock = Lock()

        self.action_history: list[ActionRequest] = []
        self.decisions: list[SafetyDecision] = []
        self._decision_records: list[tuple[ActionRequest, SafetyDecision]] = []
        # Policy overrides: category -> OutcomeType (ALLOW, BLOCK, REQUIRE_APPROVAL)
        # SECURITY: Policy overrides are IMMUTABLE after initialization.
        # Use set_policy_override() method with HMAC authentication token.
        # Direct assignment is blocked - use the authenticated setter.
        self._policy_overrides: dict[ActionCategory, OutcomeType] = {}

        # MRF integration (Phase 1 - November 2025)
        self.enable_mrf_enhancement = enable_mrf_enhancement
        self.llm_provider = llm_provider or "claude"
        self._mrf_available = False

        if enable_mrf_enhancement:
            try:
                from hololoom.alignment.mrf_integration import (
                    create_adversarial_detection_prompt,
                    create_approval_request_prompt,
                    create_risk_assessment_prompt,
                )
                self._create_risk_prompt = create_risk_assessment_prompt
                self._create_adversarial_prompt = create_adversarial_detection_prompt
                self._create_approval_prompt = create_approval_request_prompt
                self._mrf_available = True
                logger.info("MRF enhancement enabled for safety guardrails")
            except ImportError as e:
                logger.warning(f"MRF enhancement requested but not available: {e}")
                self._mrf_available = False

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    @property
    def policy_overrides(self) -> Mapping[ActionCategory, OutcomeType]:
        """
        Read-only view of policy overrides.

        SECURITY: Returns an immutable MappingProxyType to prevent unauthorized
        modifications. Use set_policy_override() with HMAC token to modify.

        Returns:
            Read-only mapping of category to override outcome
        """
        return MappingProxyType(self._policy_overrides)

    def set_policy_override(
        self,
        category: ActionCategory,
        outcome: OutcomeType,
        override_token: str,
        reason: str = ""
    ) -> bool:
        """
        Set a policy override with cryptographic authentication.

        SECURITY: Requires valid HMAC token to set policy overrides.
        This prevents unauthorized bypass of safety policies.

        Args:
            category: Action category to override
            outcome: Override outcome (ALLOW, BLOCK, REQUIRE_APPROVAL)
            override_token: HMAC authentication token
            reason: Optional reason for the override (for audit logging)

        Returns:
            True if override was set successfully, False if authentication failed

        Note:
            To generate a valid token:
            1. Set OVERRIDE_SECRET environment variable
            2. Compute: HMAC-SHA256(secret, f"override:{category.value}")
        """
        # Validate the override token
        if not _validate_override_token(override_token, category.value):
            logger.error(
                f"SECURITY: Unauthorized policy override attempt for category "
                f"'{category.value}'. Token validation failed."
            )
            return False

        # Set the override (using private attribute for write access)
        self._policy_overrides[category] = outcome

        logger.warning(
            f"SECURITY AUDIT: Policy override set for category '{category.value}' "
            f"to outcome '{outcome.value}'. Reason: {reason or 'Not provided'}"
        )
        return True

    def clear_policy_override(
        self,
        category: ActionCategory,
        override_token: str
    ) -> bool:
        """
        Clear a policy override with cryptographic authentication.

        Args:
            category: Action category to clear override for
            override_token: HMAC authentication token

        Returns:
            True if override was cleared, False if authentication failed
        """
        if not _validate_override_token(override_token, category.value):
            logger.error(
                f"SECURITY: Unauthorized policy override clear attempt for "
                f"category '{category.value}'. Token validation failed."
            )
            return False

        if category in self._policy_overrides:
            del self._policy_overrides[category]
            logger.info(f"Policy override cleared for category '{category.value}'")
            return True
        return True  # Already clear

    def get_mrf_risk_assessment_prompt(
        self,
        request: ActionRequest,
        epistemic_confidence: float | None = None
    ) -> str | None:
        """
        Get MRF-enhanced risk assessment prompt (if available).

        Args:
            request: Action request to assess
            epistemic_confidence: Optional epistemic confidence from awareness layer

        Returns:
            MRF prompt string, or None if MRF unavailable
        """
        if not self._mrf_available:
            return None

        try:
            prompt = self._create_risk_prompt(
                action=request.action,
                category=request.category,
                context=request.context,
                model_provider=self.llm_provider,
                epistemic_confidence=epistemic_confidence
            )
            return prompt
        except Exception as e:
            logger.error(f"Error creating MRF risk assessment prompt: {e}")
            return None

    def get_mrf_adversarial_detection_prompt(
        self,
        text: str,
        sensitivity: str = "balanced"
    ) -> str | None:
        """
        Get MRF-enhanced adversarial detection prompt (if available).

        Args:
            text: Text to analyze
            sensitivity: Detection sensitivity ("strict", "balanced", "permissive")

        Returns:
            MRF prompt string, or None if MRF unavailable
        """
        if not self._mrf_available:
            return None

        try:
            prompt = self._create_adversarial_prompt(
                text=text,
                model_provider=self.llm_provider,
                detection_sensitivity=sensitivity
            )
            return prompt
        except Exception as e:
            logger.error(f"Error creating MRF adversarial detection prompt: {e}")
            return None

    def get_mrf_approval_request(
        self,
        request: ActionRequest,
        risk_level: RiskLevel
    ) -> str | None:
        """
        Get MRF-enhanced approval request text (if available).

        Args:
            request: Action request requiring approval
            risk_level: Assessed risk level

        Returns:
            MRF approval request prompt, or None if MRF unavailable
        """
        if not self._mrf_available:
            return None

        try:
            prompt = self._create_approval_prompt(
                action=request.action,
                category=request.category,
                risk_level=risk_level,
                context=request.context,
                model_provider=self.llm_provider
            )
            return prompt
        except Exception as e:
            logger.error(f"Error creating MRF approval request: {e}")
            return None

    def evaluate(
        self,
        request: ActionRequest,
        text_input: str | None = None,
        epistemic_confidence: float | None = None  # Consciousness Integration - Phase 1 (Nov 2025)
    ) -> SafetyDecision:
        """
        Evaluate action request against safety policies.

        Args:
            request: Action request to evaluate
            text_input: Optional text input to check for adversarial patterns
            epistemic_confidence: Optional epistemic confidence from awareness layer (0.0-1.0)
                - Tracks system's self-awareness of knowledge gaps
                - Low values indicate epistemic humility (aware of uncertainty)
                - Used to adjust risk scoring and require approval when uncertain

        Returns:
            Safety decision
        """
        # Check for adversarial input
        if text_input and self.adversarial_detector:
            is_adversarial, reason = self.adversarial_detector.detect(text_input)
            if is_adversarial:
                logger.warning(f"Adversarial input detected: {reason}")
                return SafetyDecision(
                    allowed=False,
                    risk_level=RiskLevel.CRITICAL,
                    reason=f"Blocked: {reason}",
                    metadata={"adversarial_detected": True}
                )

        # Check for policy overrides (allows bypassing default policy)
        if request.category in self._policy_overrides:
            override = self._policy_overrides[request.category]
            if override == OutcomeType.ALLOW:
                logger.info(f"Policy override: ALLOW for category {request.category.value}")
                self.action_history.append(request)
                decision = SafetyDecision(
                    allowed=True,
                    risk_level=RiskLevel.LOW,
                    reason=f"Policy override: ALLOW for {request.category.value}",
                    requires_approval=False,
                    metadata={"policy_override": True, "action_id": request.action_id},
                    context=request.context,
                )
                self.decisions.append(decision)
                self._decision_records.append((request, decision))
                return decision
            elif override == OutcomeType.BLOCK:
                logger.info(f"Policy override: BLOCK for category {request.category.value}")
                self.action_history.append(request)
                decision = SafetyDecision(
                    allowed=False,
                    risk_level=RiskLevel.HIGH,
                    reason=f"Policy override: BLOCK for {request.category.value}",
                    requires_approval=False,
                    metadata={"policy_override": True, "action_id": request.action_id},
                    context=request.context,
                )
                self.decisions.append(decision)
                self._decision_records.append((request, decision))
                return decision
            elif override == OutcomeType.REQUIRE_APPROVAL:
                # Continue with normal evaluation but force approval
                pass  # Will be handled by normal flow with forced approval

        # Consciousness Integration - Phase 1 (Nov 2025)
        # Epistemic humility check: Low epistemic confidence = higher risk
        epistemic_risk_adjustment = RiskLevel.SAFE
        epistemic_metadata = {}

        if epistemic_confidence is not None:
            epistemic_metadata = {
                'epistemic_confidence': epistemic_confidence,
                'epistemic_humility_enabled': True,
            }

            # Very low epistemic confidence (< 0.3) indicates high uncertainty
            # → Increase risk level and require human approval
            if epistemic_confidence < 0.3:
                epistemic_risk_adjustment = RiskLevel.HIGH
                epistemic_metadata['epistemic_warning'] = 'Very uncertain - requires approval'
                logger.warning(
                    f"[EPISTEMIC] Low epistemic confidence ({epistemic_confidence:.3f}) "
                    f"for action '{request.action}' - escalating to HIGH risk"
                )

            # Moderate epistemic confidence (0.3-0.6) indicates some uncertainty
            # → Moderate risk adjustment
            elif epistemic_confidence < 0.6:
                epistemic_risk_adjustment = RiskLevel.MEDIUM
                epistemic_metadata['epistemic_warning'] = 'Moderate uncertainty detected'
                logger.info(
                    f"[EPISTEMIC] Moderate epistemic confidence ({epistemic_confidence:.3f}) "
                    f"for action '{request.action}' - elevated monitoring"
                )

            # High epistemic confidence (>= 0.6) indicates low uncertainty
            # → No adjustment needed
            else:
                epistemic_metadata['epistemic_status'] = 'High confidence - no adjustment'
                logger.debug(
                    f"[EPISTEMIC] High epistemic confidence ({epistemic_confidence:.3f}) "
                    f"for action '{request.action}'"
                )

        # Assess risk level
        risk_level = self.policy.get_risk_level(request)

        # Apply epistemic risk adjustment (take maximum of base risk and epistemic risk)
        if epistemic_risk_adjustment != RiskLevel.SAFE:
            risk_levels_ordered = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            base_risk_idx = risk_levels_ordered.index(risk_level)
            epistemic_risk_idx = risk_levels_ordered.index(epistemic_risk_adjustment)

            if epistemic_risk_idx > base_risk_idx:
                logger.info(
                    f"[EPISTEMIC] Adjusting risk from {risk_level.value} to {epistemic_risk_adjustment.value} "
                    f"due to epistemic uncertainty"
                )
                risk_level = epistemic_risk_adjustment

        # Check if approval required
        requires_approval = self.policy.requires_approval(request, risk_level)

        # Determine if allowed
        allowed = True
        reason = f"Action category: {request.category.value}, Risk level: {risk_level.value}"
        alternative_action: str | None = None

        # Block critical risk by default
        if risk_level == RiskLevel.CRITICAL:
            allowed = False
            alternative_action = "Escalate for human approval"
            reason = f"Critical risk action blocked: {request.action}"

        # Require approval blocks action until approved
        if requires_approval and allowed:
            allowed = False
            alternative_action = alternative_action or "Requires human approval"
            reason = f"Approval required before executing: {request.action}"

        # Log the decision
        self._log_decision(request, risk_level, allowed, requires_approval)

        # Build decision metadata with epistemic context
        decision_metadata = {
            "action": request.action,
            "category": request.category.value,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

        # Add epistemic metadata (Consciousness Integration - Phase 1)
        if epistemic_metadata:
            decision_metadata['epistemic'] = epistemic_metadata

        decision = SafetyDecision(
            allowed=allowed,
            risk_level=risk_level,
            reason=reason,
            requires_approval=requires_approval,
            metadata=decision_metadata,
            context=request.context,
            alternative_action=alternative_action,
        )

        # SECURITY: Thread-safe update of shared state
        # Prevents race conditions in concurrent approval flows
        with self._state_lock:
            self.action_history.append(request)
            self.decisions.append(decision)
            self._decision_records.append((request, decision))

        return decision

    def _log_decision(
        self,
        request: ActionRequest,
        risk_level: RiskLevel,
        allowed: bool,
        requires_approval: bool
    ):
        """Log safety decision."""
        log_msg = (
            f"Safety decision: action={request.action}, "
            f"category={request.category.value}, "
            f"risk={risk_level.value}, "
            f"allowed={allowed}, "
            f"requires_approval={requires_approval}"
        )

        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    # ------------------------------------------------------------------
    # Backward-compatible API surface
    # ------------------------------------------------------------------
    def evaluate_action(
        self,
        action: str,
        category: ActionCategory,
        context: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        text_input: str | None = None,
    ) -> SafetyDecision:
        """Legacy wrapper used throughout tests and older integrations."""

        request = ActionRequest(
            action_id=action,
            category=category,
            context=context or {},
            user_id=user_id,
            session_id=session_id,
        )
        text_to_check = text_input if text_input is not None else action
        return self.evaluate(request, text_input=text_to_check)

    def check_action(
        self,
        request: ActionRequest,
        text_input: str | None = None,
    ) -> SafetyDecision:
        """Compatibility shim for alignment orchestrator tests."""

        text_to_check = text_input if text_input is not None else request.action
        return self.evaluate(request, text_input=text_to_check)

    def approve_action(self, request: ActionRequest, approver_id: str) -> SafetyDecision:
        """
        Manually approve a high-risk action.

        Args:
            request: Action request to approve
            approver_id: ID of person approving

        Returns:
            Safety decision with approval
        """
        risk_level = self.policy.get_risk_level(request)

        logger.warning(
            f"Action manually approved: action={request.action}, "
            f"risk={risk_level.value}, approver={approver_id}"
        )

        decision = SafetyDecision(
            allowed=True,
            risk_level=risk_level,
            reason=f"Manually approved by {approver_id}",
            requires_approval=False,
            metadata={
                "manually_approved": True,
                "approver_id": approver_id,
                "action": request.action,
                "category": request.category.value,
            },
            context=request.context,
        )
        # SECURITY: Thread-safe update of shared state
        with self._state_lock:
            self.decisions.append(decision)
            self._decision_records.append((request, decision))
        return decision

    def get_action_history(
        self,
        category: ActionCategory | None = None,
        limit: int = 100
    ) -> list[ActionRequest]:
        """
        Get action history, optionally filtered by category.

        Args:
            category: Optional category filter
            limit: Maximum number of actions to return

        Returns:
            List of action requests
        """
        # SECURITY: Thread-safe read of shared state
        with self._state_lock:
            history = list(self.action_history)  # Copy to avoid holding lock during filter

        if category:
            history = [r for r in history if r.category == category]

        return history[-limit:]

    def get_recent_decisions(self, limit: int = 100) -> list[SafetyDecision]:
        """Return the most recent safety decisions."""
        # SECURITY: Thread-safe read of shared state
        with self._state_lock:
            return list(self.decisions[-limit:])

    def clear_history(self) -> None:
        """Reset stored requests and decisions (test compatibility)."""
        # SECURITY: Thread-safe modification of shared state
        with self._state_lock:
            self.action_history.clear()
            self.decisions.clear()
            self._decision_records.clear()

    def get_statistics(self) -> dict[str, Any]:
        """
        Get safety statistics.

        Returns:
            Dictionary of statistics
        """
        # SECURITY: Thread-safe read of shared state
        # Copy data under lock to avoid holding lock during computation
        with self._state_lock:
            decisions_copy = list(self.decisions)
            records_copy = list(self._decision_records)

        total_decisions = len(decisions_copy)
        allowed = sum(1 for decision in decisions_copy if decision.allowed)
        blocked = total_decisions - allowed

        by_risk_level = {
            level.value: sum(1 for decision in decisions_copy if decision.risk_level == level)
            for level in RiskLevel
        }

        by_category: dict[str, int] = {}
        for category in ActionCategory:
            by_category[category.value] = sum(
                1 for request, _ in records_copy if request.category == category
            )

        return {
            "total_decisions": total_decisions,
            "allowed": allowed,
            "blocked": blocked,
            "by_risk_level": by_risk_level,
            "by_category": by_category,
        }


# Convenience function
def create_guardrails(
    enable_adversarial_detection: bool = True,
    custom_policy: SafetyPolicy | None = None,
    testing_mode: bool = False,
    auto_approve_categories: set[str] | None = None,
) -> SafetyGuardrails:
    """
    Create safety guardrails with optional configuration.

    Args:
        enable_adversarial_detection: Whether to detect adversarial inputs
        custom_policy: Optional custom safety policy
        testing_mode: If True, bypass approval requirements (for development)
                      SECURITY: Only allowed in development environments
        auto_approve_categories: Set of category names to auto-approve

    Returns:
        Configured SafetyGuardrails instance

    Raises:
        ValueError: If testing_mode=True requested in non-development environment
    """
    # Note: testing_mode validation happens in SafetyGuardrails.__init__
    return SafetyGuardrails(
        policy=custom_policy,
        enable_adversarial_detection=enable_adversarial_detection,
        testing_mode=testing_mode,
        auto_approve_categories=auto_approve_categories,
    )
