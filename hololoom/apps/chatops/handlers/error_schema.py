"""
Error Schema - Structured error types with remediation guidance.

Provides award-winning error handling with:
- AI-powered suggestion learning (Thompson Sampling)
- Contextual help integration
- Recovery wizards for complex errors
- Multi-modal error presentation (terminal, web, voice)

Philosophy: Errors should guide users to success, not just report failure.

Created: 2025-12-15
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCategory(Enum):
    """Error classification for appropriate handling."""
    AUTH_REQUIRED = "auth_required"      # User needs to authenticate
    PERMISSION_DENIED = "permission"     # Lacks required permissions
    COOLDOWN_ACTIVE = "cooldown"         # Rate limit in effect
    COMMAND_NOT_FOUND = "not_found"      # Unknown command
    INVALID_ARGS = "invalid_args"        # Bad arguments provided
    INTERNAL_ERROR = "internal"          # System error
    DEPENDENCY_MISSING = "dependency"    # Optional component unavailable
    VALIDATION_ERROR = "validation"      # Input validation failed
    TIMEOUT_ERROR = "timeout"            # Operation timed out
    RESOURCE_ERROR = "resource"          # Resource unavailable


class ErrorSeverity(Enum):
    """Severity for UI presentation."""
    INFO = "info"           # Informational (blue)
    WARNING = "warning"     # User action needed (yellow)
    ERROR = "error"         # Operation failed (red)
    CRITICAL = "critical"   # System issue (red + alert)


@dataclass
class ChatOpsError(Exception):
    """
    Structured error with remediation guidance.

    Follows award-winning UX principles:
    - Anticipatory Design: Predict user needs with suggestions
    - Progressive Disclosure: Simple message first, details on demand
    - Learning Systems: Thompson Sampling improves suggestions over time
    """
    message: str
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.ERROR

    # Remediation
    suggestion: str = ""                           # Primary fix suggestion
    alternatives: list[str] = field(default_factory=list)  # Other options
    docs_url: str | None = None                 # Link to documentation

    # Context
    command: str = ""                              # Which command failed
    retry_after_seconds: float | None = None    # For cooldowns
    required_permission: str | None = None      # For auth errors

    # Technical details (for logging, not user display)
    details: dict[str, Any] = field(default_factory=dict)

    # Tracking for learning
    error_id: str = field(default_factory=lambda: f"err-{int(time.time() * 1000)}-{random.randint(1000, 9999)}")
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Initialize Exception base class with message
        super().__init__(self.message)

    def to_user_message(self, verbose: bool = False) -> str:
        """
        Format for user display with remediation hints.

        Args:
            verbose: Include technical details if True

        Returns:
            Formatted error message string
        """
        # Severity icons
        icons = {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨"
        }

        lines = [f"{icons.get(self.severity, '❌')} {self.message}"]

        if self.suggestion:
            lines.append(f"💡 {self.suggestion}")

        for alt in self.alternatives[:3]:  # Max 3 alternatives
            lines.append(f"   → {alt}")

        if self.docs_url:
            lines.append(f"📚 See: {self.docs_url}")

        if self.retry_after_seconds:
            lines.append(f"⏱️ Retry in {self.retry_after_seconds:.0f}s")

        if verbose and self.details:
            lines.append(f"🔧 Details: {self.details}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "error": True,
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "suggestion": self.suggestion,
            "alternatives": self.alternatives,
            "docs_url": self.docs_url,
            "retry_after_seconds": self.retry_after_seconds,
            "command": self.command,
            "timestamp": self.timestamp
        }

    def to_voice_message(self) -> str:
        """Format for text-to-speech output."""
        msg = self.message
        if self.suggestion:
            msg += f". {self.suggestion}"
        if self.retry_after_seconds:
            msg += f". You can retry in {int(self.retry_after_seconds)} seconds."
        return msg

    def to_html(self) -> str:
        """Format as HTML for web dashboards."""
        severity_colors = {
            ErrorSeverity.INFO: "#3b82f6",      # Blue
            ErrorSeverity.WARNING: "#f59e0b",   # Amber
            ErrorSeverity.ERROR: "#ef4444",     # Red
            ErrorSeverity.CRITICAL: "#dc2626"   # Dark Red
        }
        color = severity_colors.get(self.severity, "#ef4444")

        html = f'''
        <div class="chatops-error chatops-error--{self.severity.value}"
             style="border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; background: #fef2f2;">
            <div class="chatops-error__message" style="font-weight: 600; color: {color};">
                {self.message}
            </div>
        '''

        if self.suggestion:
            html += f'''
            <div class="chatops-error__suggestion" style="margin-top: 0.5rem; color: #4b5563;">
                💡 {self.suggestion}
            </div>
            '''

        if self.alternatives:
            html += '<ul class="chatops-error__alternatives" style="margin-top: 0.5rem; padding-left: 1.5rem;">'
            for alt in self.alternatives[:3]:
                html += f'<li style="color: #6b7280;">{alt}</li>'
            html += '</ul>'

        if self.docs_url:
            html += f'''
            <div class="chatops-error__docs" style="margin-top: 0.5rem;">
                <a href="{self.docs_url}" style="color: #2563eb;">📚 View Documentation</a>
            </div>
            '''

        html += '</div>'
        return html


# =============================================================================
# Error Factory Functions
# =============================================================================

def auth_required_error(command: str) -> ChatOpsError:
    """Create structured error for authentication requirement."""
    return ChatOpsError(
        message=f"Command !{command} requires authentication",
        category=ErrorCategory.AUTH_REQUIRED,
        severity=ErrorSeverity.WARNING,
        command=command,
        suggestion="Authenticate first with !login or set up Matrix SSO",
        alternatives=[
            "Use !auth status to check your authentication state",
            "Contact your admin if you believe you should have access",
            "Run !help auth for authentication options"
        ],
        docs_url="https://hololoom.dev/docs/auth"
    )


def cooldown_error(command: str, remaining: float) -> ChatOpsError:
    """Create structured error for rate limiting."""
    return ChatOpsError(
        message=f"Cooldown active for !{command}",
        category=ErrorCategory.COOLDOWN_ACTIVE,
        severity=ErrorSeverity.INFO,
        command=command,
        retry_after_seconds=remaining,
        suggestion=f"Wait {remaining:.1f}s before trying again",
        alternatives=[
            "This cooldown prevents API abuse",
            "Premium users have reduced cooldowns",
            "Use !stats to see your rate limit status"
        ]
    )


def command_not_found_error(command: str, suggestions: list[str]) -> ChatOpsError:
    """Create structured error for unknown command with fuzzy suggestions."""
    suggestion = (
        f"Did you mean: {', '.join(f'!{s}' for s in suggestions[:3])}?"
        if suggestions else "Use !help to see available commands"
    )

    return ChatOpsError(
        message=f"Unknown command: !{command}",
        category=ErrorCategory.COMMAND_NOT_FOUND,
        severity=ErrorSeverity.WARNING,
        command=command,
        suggestion=suggestion,
        alternatives=[
            "Use !help to see all available commands",
            "Use !help <command> for detailed usage",
            "Commands are case-sensitive"
        ]
    )


def invalid_args_error(command: str, usage: str, problem: str) -> ChatOpsError:
    """Create structured error for invalid arguments."""
    return ChatOpsError(
        message=f"Invalid arguments for !{command}: {problem}",
        category=ErrorCategory.INVALID_ARGS,
        severity=ErrorSeverity.WARNING,
        command=command,
        suggestion=f"Usage: {usage}",
        alternatives=[
            f"Use !help {command} for detailed usage",
            "Check argument types and formatting",
            "Wrap arguments with spaces in quotes"
        ]
    )


def permission_denied_error(command: str, required_permission: str) -> ChatOpsError:
    """Create structured error for insufficient permissions."""
    return ChatOpsError(
        message=f"Permission denied for !{command}",
        category=ErrorCategory.PERMISSION_DENIED,
        severity=ErrorSeverity.WARNING,
        command=command,
        required_permission=required_permission,
        suggestion=f"This command requires '{required_permission}' permission",
        alternatives=[
            "Contact your administrator for access",
            "Use !whoami to see your current permissions",
            "Some commands require room admin status"
        ]
    )


def internal_error(command: str, error: Exception) -> ChatOpsError:
    """Create structured error for unexpected internal errors."""
    return ChatOpsError(
        message=f"Internal error executing !{command}",
        category=ErrorCategory.INTERNAL_ERROR,
        severity=ErrorSeverity.ERROR,
        command=command,
        suggestion="This may be a temporary issue. Try again in a moment.",
        alternatives=[
            "Check !stats for system status",
            "Report persistent issues with !feedback",
            "View logs with !audit log"
        ],
        details={
            "original_error": str(error),
            "error_type": type(error).__name__
        }
    )


def dependency_missing_error(
    command: str,
    dependency: str,
    install_hint: str | None = None
) -> ChatOpsError:
    """Create structured error for missing optional dependency."""
    suggestion = f"The '{dependency}' component is not available"
    if install_hint:
        suggestion += f". {install_hint}"

    return ChatOpsError(
        message=f"!{command} requires '{dependency}' which is not installed",
        category=ErrorCategory.DEPENDENCY_MISSING,
        severity=ErrorSeverity.WARNING,
        command=command,
        suggestion=suggestion,
        alternatives=[
            "This feature requires optional dependencies",
            "Check the installation guide for setup instructions",
            "Some features degrade gracefully without dependencies"
        ]
    )


def timeout_error(command: str, timeout_seconds: float) -> ChatOpsError:
    """Create structured error for operation timeout."""
    return ChatOpsError(
        message=f"!{command} timed out after {timeout_seconds:.1f}s",
        category=ErrorCategory.TIMEOUT_ERROR,
        severity=ErrorSeverity.WARNING,
        command=command,
        suggestion="The operation took too long. Try a simpler query or retry later.",
        alternatives=[
            "Complex queries may need more time",
            "Check system load with !stats",
            "Consider breaking into smaller operations"
        ],
        details={"timeout_seconds": timeout_seconds}
    )


def validation_error(command: str, field: str, reason: str) -> ChatOpsError:
    """Create structured error for input validation failure."""
    return ChatOpsError(
        message=f"Validation failed for !{command}: {field} - {reason}",
        category=ErrorCategory.VALIDATION_ERROR,
        severity=ErrorSeverity.WARNING,
        command=command,
        suggestion=f"Please check the '{field}' parameter",
        alternatives=[
            f"Use !help {command} for valid input formats",
            "Check for typos or formatting issues",
            "Some fields have specific requirements"
        ],
        details={"field": field, "reason": reason}
    )


# =============================================================================
# Thompson Sampling Suggestion Learner
# =============================================================================

@dataclass
class SuggestionOutcome:
    """Tracks outcome of a suggestion for learning."""
    error_category: str
    suggestion_id: str
    recovered: bool
    recovery_time_seconds: float | None = None
    user_feedback: str | None = None


class ErrorSuggestionLearner:
    """
    Learns optimal error suggestions using Thompson Sampling.

    Tracks which suggestions lead to successful recovery (user
    successfully completes action after error) vs abandonment.

    Innovation: Uses Bayesian learning to continuously improve
    which suggestions are shown first for each error type.
    """

    def __init__(self, quality_threshold: float = 0.7):
        # Thompson Sampling priors: (alpha, beta) per suggestion
        # Key: f"{error_category}:{suggestion_id}"
        self._suggestion_priors: dict[str, tuple[float, float]] = {}
        self._recovery_tracker: dict[str, dict[str, Any]] = {}
        self._quality_threshold = quality_threshold

    def get_ranked_suggestions(
        self,
        error: ChatOpsError,
        base_suggestions: list[str]
    ) -> list[tuple[str, float]]:
        """
        Return suggestions ranked by Thompson-sampled expected value.

        Each suggestion is sampled from Beta(α, β) to balance:
        - Exploitation: Show suggestions that historically help
        - Exploration: Try new suggestions to learn their value

        Returns:
            List of (suggestion, confidence) tuples, highest confidence first
        """
        ranked = []

        for i, suggestion in enumerate(base_suggestions):
            key = f"{error.category.value}:{i}"
            alpha, beta = self._suggestion_priors.get(key, (1.0, 1.0))

            # Thompson sample from Beta distribution
            try:
                import numpy as np
                sampled_value = np.random.beta(alpha, beta)
            except ImportError:
                # Fallback: use mean
                sampled_value = alpha / (alpha + beta)

            confidence = alpha / (alpha + beta)
            ranked.append((suggestion, sampled_value, confidence))

        # Sort by sampled value (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)

        return [(s, c) for s, _, c in ranked]

    def record_error(self, error: ChatOpsError) -> None:
        """Record that an error occurred for recovery tracking."""
        self._recovery_tracker[error.error_id] = {
            "category": error.category.value,
            "timestamp": error.timestamp,
            "suggestions_shown": list(range(len(error.alternatives) + 1))  # +1 for primary
        }

    def record_outcome(
        self,
        error_id: str,
        recovered: bool,
        suggestion_index: int = 0
    ) -> None:
        """
        Update Thompson priors based on recovery outcome.

        Args:
            error_id: ID of the error
            recovered: Whether user successfully recovered
            suggestion_index: Which suggestion was followed (0=primary)
        """
        if error_id not in self._recovery_tracker:
            return

        tracker = self._recovery_tracker[error_id]
        category = tracker["category"]
        key = f"{category}:{suggestion_index}"

        alpha, beta = self._suggestion_priors.get(key, (1.0, 1.0))

        if recovered:
            alpha += 1.0  # Success: increase alpha
        else:
            beta += 1.0   # Failure: increase beta

        self._suggestion_priors[key] = (alpha, beta)

        # Cleanup old tracker entry
        del self._recovery_tracker[error_id]

    def get_suggestion_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics about suggestion effectiveness."""
        stats = {}
        for key, (alpha, beta) in self._suggestion_priors.items():
            expected_value = alpha / (alpha + beta)
            total_observations = alpha + beta - 2  # Subtract initial priors
            stats[key] = {
                "expected_value": expected_value,
                "alpha": alpha,
                "beta": beta,
                "observations": total_observations,
                "confidence": min(total_observations / 20, 1.0)  # Confident after 20 obs
            }
        return stats

    def save_state(self, filepath: str) -> None:
        """Save learned priors to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "priors": self._suggestion_priors,
                "quality_threshold": self._quality_threshold
            }, f, indent=2)

    def load_state(self, filepath: str) -> None:
        """Load learned priors from file."""
        import json
        try:
            with open(filepath) as f:
                data = json.load(f)
                self._suggestion_priors = {
                    k: tuple(v) for k, v in data.get("priors", {}).items()
                }
                self._quality_threshold = data.get("quality_threshold", 0.7)
        except FileNotFoundError:
            pass  # Start with fresh priors


# =============================================================================
# Recovery Wizard
# =============================================================================

@dataclass
class WizardStep:
    """Single step in a recovery wizard."""
    title: str
    action: str
    success_criteria: str
    if_failed: str


@dataclass
class WizardSession:
    """Active recovery wizard session."""
    error: ChatOpsError
    steps: list[WizardStep]
    current_step: int = 0
    session_id: str = field(default_factory=lambda: f"wiz-{int(time.time() * 1000)}")
    started_at: float = field(default_factory=time.time)

    def get_current_step(self) -> WizardStep | None:
        """Get current wizard step."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance(self) -> bool:
        """Advance to next step. Returns False if wizard complete."""
        self.current_step += 1
        return self.current_step < len(self.steps)

    def to_message(self) -> str:
        """Format current step as user message."""
        step = self.get_current_step()
        if not step:
            return "Recovery wizard complete! Your issue should be resolved."

        return f"""
**Recovery Step {self.current_step + 1}/{len(self.steps)}: {step.title}**

Action: `{step.action}`

Expected: {step.success_criteria}
If that doesn't work: {step.if_failed}
"""


class ErrorRecoveryWizard:
    """
    Step-by-step recovery guidance for complex errors.

    Provides guided workflows for common error scenarios,
    helping users resolve issues systematically.
    """

    # Pre-defined recovery flows
    RECOVERY_FLOWS: dict[ErrorCategory, list[WizardStep]] = {
        ErrorCategory.AUTH_REQUIRED: [
            WizardStep(
                title="Check Authentication Status",
                action="!auth status",
                success_criteria="Shows 'Authenticated as: ...'",
                if_failed="Proceed to step 2"
            ),
            WizardStep(
                title="Log In",
                action="!login",
                success_criteria="Shows 'Successfully authenticated'",
                if_failed="Check Matrix homeserver settings"
            ),
            WizardStep(
                title="Verify Permissions",
                action="!whoami",
                success_criteria="Shows your user role",
                if_failed="Contact admin for permissions"
            ),
            WizardStep(
                title="Retry Command",
                action="{original_command}",
                success_criteria="Command succeeds",
                if_failed="Report to support with error ID"
            )
        ],
        ErrorCategory.COOLDOWN_ACTIVE: [
            WizardStep(
                title="Check Rate Limit Status",
                action="!stats ratelimit",
                success_criteria="Shows remaining cooldown time",
                if_failed="Wait a moment and retry"
            ),
            WizardStep(
                title="Wait for Cooldown",
                action="(wait for displayed time)",
                success_criteria="Cooldown expires",
                if_failed="Cooldowns reset every minute"
            ),
            WizardStep(
                title="Retry Command",
                action="{original_command}",
                success_criteria="Command succeeds",
                if_failed="Check if you hit another rate limit"
            )
        ],
        ErrorCategory.DEPENDENCY_MISSING: [
            WizardStep(
                title="Check System Status",
                action="!stats dependencies",
                success_criteria="Shows which dependencies are available",
                if_failed="System status may be degraded"
            ),
            WizardStep(
                title="View Installation Guide",
                action="!help install",
                success_criteria="Shows installation instructions",
                if_failed="Check documentation online"
            ),
            WizardStep(
                title="Use Alternative Command",
                action="!help alternatives",
                success_criteria="Shows alternative commands available",
                if_failed="Some features require specific setup"
            )
        ]
    }

    _active_sessions: dict[str, WizardSession] = {}

    @classmethod
    def start_recovery(cls, error: ChatOpsError) -> WizardSession | None:
        """Start a guided recovery session for an error."""
        flow = cls.RECOVERY_FLOWS.get(error.category, [])
        if not flow:
            return None

        # Substitute original command placeholder
        steps = []
        for step in flow:
            action = step.action.replace("{original_command}", f"!{error.command}")
            steps.append(WizardStep(
                title=step.title,
                action=action,
                success_criteria=step.success_criteria,
                if_failed=step.if_failed
            ))

        session = WizardSession(error=error, steps=steps)
        cls._active_sessions[session.session_id] = session
        return session

    @classmethod
    def get_session(cls, session_id: str) -> WizardSession | None:
        """Get an active recovery session."""
        return cls._active_sessions.get(session_id)

    @classmethod
    def advance_session(cls, session_id: str) -> WizardSession | None:
        """Advance a session to the next step."""
        session = cls._active_sessions.get(session_id)
        if session:
            if not session.advance():
                # Wizard complete
                del cls._active_sessions[session_id]
                return None
        return session

    @classmethod
    def cancel_session(cls, session_id: str) -> bool:
        """Cancel an active recovery session."""
        if session_id in cls._active_sessions:
            del cls._active_sessions[session_id]
            return True
        return False


# =============================================================================
# Global Learner Instance
# =============================================================================

_global_learner: ErrorSuggestionLearner | None = None


def get_suggestion_learner() -> ErrorSuggestionLearner:
    """Get or create the global suggestion learner."""
    global _global_learner
    if _global_learner is None:
        _global_learner = ErrorSuggestionLearner()
    return _global_learner


def enhance_error_with_learning(error: ChatOpsError) -> ChatOpsError:
    """
    Enhance an error with learned suggestions.

    Uses Thompson Sampling to reorder suggestions based on
    historical recovery success rates.
    """
    learner = get_suggestion_learner()

    # Get all suggestions
    all_suggestions = [error.suggestion] + error.alternatives

    # Rank by Thompson Sampling
    ranked = learner.get_ranked_suggestions(error, all_suggestions)

    # Update error with ranked suggestions
    if ranked:
        error.suggestion = ranked[0][0]
        error.alternatives = [s for s, _ in ranked[1:]]

    # Record for tracking
    learner.record_error(error)

    return error
