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

import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("HoloLoom.alignment.safety_guardrails")


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


class ActionCategory(Enum):
    """
    Categories of actions the system can perform.

    Used for risk assessment and policy enforcement.
    """
    # Read operations (generally safe)
    QUERY = "query"            # Query knowledge base
    RETRIEVAL = "retrieval"    # Retrieve information
    ANALYSIS = "analysis"      # Analyze data

    # Write operations (higher risk)
    STORAGE = "storage"        # Store new information
    MODIFICATION = "modification"  # Modify existing data
    DELETION = "deletion"      # Delete data

    # Execution operations (highest risk)
    EXECUTION = "execution"    # Execute code/tools
    SYSTEM = "system"          # System-level operations
    EXTERNAL = "external"      # External API calls


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.value,
            "reason": self.reason,
            "requires_approval": self.requires_approval,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ActionRequest:
    """
    Request to perform an action.

    Contains all information needed for safety evaluation.
    """
    action: str
    category: ActionCategory
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


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
        if len(text) > 50000:  # 50k characters
            return True, "Input exceeds reasonable length"

        return False, ""


class SafetyPolicy:
    """
    Defines safety policies for different action categories.

    Configurable risk thresholds and approval requirements.

    Supports environment-aware testing mode where approvals can be bypassed
    for development environments while maintaining full logging.
    """

    def __init__(self, testing_mode: bool = False, auto_approve_categories: Optional[Set[str]] = None):
        """
        Initialize with default policies.

        Args:
            testing_mode: If True, bypass approval requirements (for development)
            auto_approve_categories: Set of category names to auto-approve (overrides testing_mode)
        """
        self.testing_mode = testing_mode
        self.auto_approve_categories = auto_approve_categories or set()

        # Default risk levels by action category
        self.default_risk_levels = {
            ActionCategory.QUERY: RiskLevel.SAFE,
            ActionCategory.RETRIEVAL: RiskLevel.SAFE,
            ActionCategory.ANALYSIS: RiskLevel.LOW,
            ActionCategory.STORAGE: RiskLevel.LOW,
            ActionCategory.MODIFICATION: RiskLevel.MEDIUM,
            ActionCategory.DELETION: RiskLevel.HIGH,
            ActionCategory.EXECUTION: RiskLevel.HIGH,
            ActionCategory.SYSTEM: RiskLevel.CRITICAL,
            ActionCategory.EXTERNAL: RiskLevel.HIGH,
        }

        # Actions that always require approval (unless testing_mode or auto_approve)
        if testing_mode:
            # Testing mode: Don't require approval for anything
            self.approval_required: Set[ActionCategory] = set()
        else:
            # Production: Require approval for critical actions
            self.approval_required: Set[ActionCategory] = {
                ActionCategory.DELETION,
                ActionCategory.SYSTEM,
            }

        # Custom risk evaluators (can be registered)
        self.custom_evaluators: List[Callable[[ActionRequest], Optional[RiskLevel]]] = []

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

        # Check auto-approve categories (environment-specific)
        if request.category.value in self.auto_approve_categories:
            return False

        # Always require approval for certain categories
        if request.category in self.approval_required:
            return True

        # Require approval for high/critical risk
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return True

        return False

    def register_evaluator(self, evaluator: Callable[[ActionRequest], Optional[RiskLevel]]):
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
    """

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        enable_adversarial_detection: bool = True,
        testing_mode: bool = False,
        auto_approve_categories: Optional[Set[str]] = None,
    ):
        """
        Initialize safety guardrails.

        Args:
            policy: Safety policy (uses default if None)
            enable_adversarial_detection: Whether to detect adversarial inputs
            testing_mode: If True, bypass approval requirements (for development)
            auto_approve_categories: Set of category names to auto-approve
        """
        self.testing_mode = testing_mode
        self.policy = policy or SafetyPolicy(
            testing_mode=testing_mode,
            auto_approve_categories=auto_approve_categories
        )
        self.adversarial_detector = AdversarialDetector() if enable_adversarial_detection else None
        self.action_history: List[ActionRequest] = []
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

    def evaluate(self, request: ActionRequest, text_input: Optional[str] = None) -> SafetyDecision:
        """
        Evaluate action request against safety policies.

        Args:
            request: Action request to evaluate
            text_input: Optional text input to check for adversarial patterns

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

        # Assess risk level
        risk_level = self.policy.get_risk_level(request)

        # Check if approval required
        requires_approval = self.policy.requires_approval(request, risk_level)

        # Determine if allowed
        allowed = True
        reason = f"Action category: {request.category.value}, Risk level: {risk_level.value}"

        # Block critical risk by default
        if risk_level == RiskLevel.CRITICAL:
            allowed = False
            reason = f"Critical risk action blocked: {request.action}"

        # Log the decision
        self._log_decision(request, risk_level, allowed, requires_approval)

        # Store in history
        self.action_history.append(request)

        return SafetyDecision(
            allowed=allowed,
            risk_level=risk_level,
            reason=reason,
            requires_approval=requires_approval,
            metadata={
                "action": request.action,
                "category": request.category.value,
                "user_id": request.user_id,
                "session_id": request.session_id,
            }
        )

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

        return SafetyDecision(
            allowed=True,
            risk_level=risk_level,
            reason=f"Manually approved by {approver_id}",
            requires_approval=False,
            metadata={
                "manually_approved": True,
                "approver_id": approver_id,
                "action": request.action,
                "category": request.category.value,
            }
        )

    def get_action_history(
        self,
        category: Optional[ActionCategory] = None,
        limit: int = 100
    ) -> List[ActionRequest]:
        """
        Get action history, optionally filtered by category.

        Args:
            category: Optional category filter
            limit: Maximum number of actions to return

        Returns:
            List of action requests
        """
        history = self.action_history

        if category:
            history = [r for r in history if r.category == category]

        return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get safety statistics.

        Returns:
            Dictionary of statistics
        """
        total_actions = len(self.action_history)

        by_category = {}
        for category in ActionCategory:
            count = sum(1 for r in self.action_history if r.category == category)
            by_category[category.value] = count

        return {
            "total_actions": total_actions,
            "by_category": by_category,
        }


# Convenience function
def create_guardrails(
    enable_adversarial_detection: bool = True,
    custom_policy: Optional[SafetyPolicy] = None,
    testing_mode: bool = False,
    auto_approve_categories: Optional[Set[str]] = None,
) -> SafetyGuardrails:
    """
    Create safety guardrails with optional configuration.

    Args:
        enable_adversarial_detection: Whether to detect adversarial inputs
        custom_policy: Optional custom safety policy
        testing_mode: If True, bypass approval requirements (for development)
        auto_approve_categories: Set of category names to auto-approve

    Returns:
        Configured SafetyGuardrails instance
    """
    return SafetyGuardrails(
        policy=custom_policy,
        enable_adversarial_detection=enable_adversarial_detection,
        testing_mode=testing_mode,
        auto_approve_categories=auto_approve_categories,
    )