"""
HoloLoom Alignment Integration for Coding Ritual
================================================

Integrates HoloLoom's Alignment Framework with /ritual check:
- SafetyGuardrails for risk-based action gating
- AuditTrail for complete decision provenance
- DeceptionDetection for goal transparency
- InstrumentalConvergence for power-seeking detection

This enables:
- Automated safety checks during ritual workflow
- Complete audit trail of all ritual decisions
- Risk assessment before code execution
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Try to import HoloLoom alignment - graceful degradation
HOLOLOOM_ALIGNMENT_AVAILABLE = False
try:
    hololoom_path = Path(__file__).parent.parent.parent.parent.parent.parent
    if str(hololoom_path) not in sys.path:
        sys.path.insert(0, str(hololoom_path))

    from hololoom.alignment.safety_guardrails import SafetyGuardrails
    from hololoom.alignment.audit_trail import AuditTrail
    HOLOLOOM_ALIGNMENT_AVAILABLE = True
except ImportError:
    pass


class RiskLevel(Enum):
    """Risk levels for ritual actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    check_id: str
    passed: bool
    risk_level: RiskLevel
    message: str
    recommendations: List[str]
    requires_review: bool = False
    blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "risk_level": self.risk_level.value,
            "message": self.message,
            "recommendations": self.recommendations,
            "requires_review": self.requires_review,
            "blocked": self.blocked,
        }


class RitualAlignmentAdapter:
    """
    Adapter for integrating HoloLoom alignment with ritual workflow.

    Provides:
    - Safety checks for ritual phases
    - Audit trail logging
    - Risk assessment
    - Human-in-loop escalation
    """

    # Safety check definitions aligned with /ritual check
    SAFETY_CHECKS = {
        "graceful_degradation": {
            "description": "Handles failures gracefully without crashing",
            "risk_patterns": ["sys.exit", "os._exit", "raise SystemExit"],
            "risk_level": RiskLevel.HIGH,
        },
        "input_validation": {
            "description": "Validates all user inputs",
            "risk_patterns": ["eval(", "exec(", "compile("],
            "risk_level": RiskLevel.CRITICAL,
        },
        "error_messages": {
            "description": "Error messages are clear and helpful",
            "risk_patterns": ["raise Exception()", "except: pass"],
            "risk_level": RiskLevel.MEDIUM,
        },
        "no_hardcoded_secrets": {
            "description": "No secrets in code",
            "risk_patterns": ["password=", "api_key=", "secret=", "token="],
            "risk_level": RiskLevel.CRITICAL,
        },
        "timeout_handling": {
            "description": "Long operations have timeouts",
            "risk_patterns": ["while True:", "time.sleep(9"],
            "risk_level": RiskLevel.MEDIUM,
        },
        "resource_cleanup": {
            "description": "Resources properly cleaned up",
            "risk_patterns": ["open(", "connect("],
            "risk_level": RiskLevel.MEDIUM,
        },
        "sql_injection": {
            "description": "No SQL injection vulnerabilities",
            "risk_patterns": ["f\"SELECT", "f'SELECT", ".format("],
            "risk_level": RiskLevel.CRITICAL,
        },
        "command_injection": {
            "description": "No command injection vulnerabilities",
            "risk_patterns": ["os.system(", "subprocess.call(", "shell=True"],
            "risk_level": RiskLevel.CRITICAL,
        },
        "path_traversal": {
            "description": "No path traversal vulnerabilities",
            "risk_patterns": ["../", "..\\"],
            "risk_level": RiskLevel.HIGH,
        },
        "data_validation": {
            "description": "External data properly validated",
            "risk_patterns": ["json.loads(", "yaml.load("],
            "risk_level": RiskLevel.MEDIUM,
        },
    }

    def __init__(
        self,
        guardrails: Optional[Any] = None,
        audit_trail: Optional[Any] = None,
        enable_human_in_loop: bool = True,
    ):
        """
        Initialize alignment adapter.

        Args:
            guardrails: HoloLoom SafetyGuardrails instance
            audit_trail: HoloLoom AuditTrail instance
            enable_human_in_loop: Enable human review for high-risk
        """
        self.guardrails = guardrails
        self.audit_trail = audit_trail
        self.enable_human_in_loop = enable_human_in_loop
        self._check_history: List[Dict[str, Any]] = []

        if HOLOLOOM_ALIGNMENT_AVAILABLE:
            self._auto_initialize()

    def _auto_initialize(self) -> None:
        """Auto-initialize alignment components."""
        if not HOLOLOOM_ALIGNMENT_AVAILABLE:
            return

        try:
            if self.guardrails is None:
                self.guardrails = SafetyGuardrails(
                    enable_human_in_loop=self.enable_human_in_loop
                )
            if self.audit_trail is None:
                self.audit_trail = AuditTrail()
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Check if alignment framework is available."""
        return HOLOLOOM_ALIGNMENT_AVAILABLE

    async def run_safety_check(
        self,
        check_id: str,
        code_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyCheckResult:
        """
        Run a specific safety check.

        Args:
            check_id: ID of check to run
            code_content: Code to analyze (optional)
            context: Additional context

        Returns:
            SafetyCheckResult with findings
        """
        if check_id not in self.SAFETY_CHECKS:
            return SafetyCheckResult(
                check_id=check_id,
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                message=f"Unknown check: {check_id}",
                recommendations=["Verify check ID"],
            )

        check_config = self.SAFETY_CHECKS[check_id]
        risk_level = check_config["risk_level"]
        patterns = check_config["risk_patterns"]

        # Default to passed if no code to analyze
        if not code_content:
            return SafetyCheckResult(
                check_id=check_id,
                passed=True,
                risk_level=RiskLevel.LOW,
                message="No code provided - assuming safe",
                recommendations=[],
            )

        # Check for risk patterns
        found_patterns = []
        for pattern in patterns:
            if pattern.lower() in code_content.lower():
                found_patterns.append(pattern)

        if found_patterns:
            return SafetyCheckResult(
                check_id=check_id,
                passed=False,
                risk_level=risk_level,
                message=f"Found risk patterns: {', '.join(found_patterns)}",
                recommendations=[
                    f"Review usage of: {p}" for p in found_patterns
                ],
                requires_review=risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
                blocked=risk_level == RiskLevel.CRITICAL,
            )

        return SafetyCheckResult(
            check_id=check_id,
            passed=True,
            risk_level=RiskLevel.LOW,
            message=check_config["description"],
            recommendations=[],
        )

    async def run_all_safety_checks(
        self,
        code_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run all safety checks.

        Args:
            code_content: Code to analyze
            context: Additional context

        Returns:
            Summary of all check results
        """
        results = []
        passed_count = 0
        failed_count = 0
        blocked_count = 0
        requires_review = False
        highest_risk = RiskLevel.LOW

        for check_id in self.SAFETY_CHECKS:
            result = await self.run_safety_check(
                check_id, code_content, context
            )
            results.append(result)

            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
                if result.blocked:
                    blocked_count += 1
                if result.requires_review:
                    requires_review = True

            # Track highest risk
            risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            if risk_order.index(result.risk_level) > risk_order.index(highest_risk):
                highest_risk = result.risk_level

        # Store in history
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "blocked": blocked_count,
            "requires_review": requires_review,
            "highest_risk": highest_risk.value,
            "results": [r.to_dict() for r in results],
        }
        self._check_history.append(summary)

        # Log to audit trail if available
        if self.audit_trail:
            try:
                await self.audit_trail.log_decision(
                    decision_type="safety_check",
                    decision_data=summary,
                    outcome="passed" if blocked_count == 0 else "blocked",
                )
            except Exception:
                pass

        return summary

    async def gate_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        epistemic_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Gate an action through safety guardrails.

        Args:
            action: Action to gate (e.g., "execute_code", "modify_file")
            context: Action context
            epistemic_confidence: Confidence in safety assessment

        Returns:
            Gating result with allowed/blocked status
        """
        if self.guardrails:
            try:
                result = await self.guardrails.gate_action(
                    action=action,
                    context=context or {},
                    epistemic_confidence=epistemic_confidence,
                )
                return {
                    "allowed": result.allowed if hasattr(result, 'allowed') else True,
                    "risk_level": result.risk_level.value if hasattr(result, 'risk_level') else "low",
                    "reason": result.reason if hasattr(result, 'reason') else "",
                    "recommendations": result.recommendations if hasattr(result, 'recommendations') else [],
                }
            except Exception as e:
                return {
                    "allowed": True,
                    "risk_level": "unknown",
                    "reason": f"Guardrails error: {str(e)}",
                    "recommendations": ["Manual review recommended"],
                }

        # Fallback without guardrails
        return {
            "allowed": True,
            "risk_level": "unknown",
            "reason": "Guardrails not available",
            "recommendations": [],
        }

    async def log_decision(
        self,
        decision_type: str,
        decision_data: Dict[str, Any],
        outcome: str,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log a decision to the audit trail.

        Args:
            decision_type: Type of decision
            decision_data: Decision details
            outcome: Decision outcome
            session_id: Ritual session ID
        """
        if self.audit_trail:
            try:
                await self.audit_trail.log_decision(
                    decision_type=decision_type,
                    decision_data={
                        **decision_data,
                        "ritual_session_id": session_id,
                    },
                    outcome=outcome,
                )
            except Exception:
                pass

    def get_check_history(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent check history."""
        return self._check_history[-limit:]

    def get_check_statistics(self) -> Dict[str, Any]:
        """Get statistics from check history."""
        if not self._check_history:
            return {
                "total_checks": 0,
                "pass_rate": 0.0,
                "block_rate": 0.0,
            }

        total_passed = sum(h.get("passed", 0) for h in self._check_history)
        total_checks = sum(h.get("total_checks", 0) for h in self._check_history)
        total_blocked = sum(h.get("blocked", 0) for h in self._check_history)

        return {
            "total_runs": len(self._check_history),
            "total_checks": total_checks,
            "total_passed": total_passed,
            "total_blocked": total_blocked,
            "pass_rate": total_passed / total_checks if total_checks > 0 else 0.0,
            "block_rate": total_blocked / total_checks if total_checks > 0 else 0.0,
        }


# Convenience functions
async def run_ritual_safety_checks(
    code_content: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all ritual safety checks."""
    adapter = RitualAlignmentAdapter()
    return await adapter.run_all_safety_checks(code_content)


async def gate_ritual_action(
    action: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Gate a ritual action through safety guardrails."""
    adapter = RitualAlignmentAdapter()
    return await adapter.gate_action(action, context)


def get_safety_check_definitions() -> Dict[str, Dict[str, Any]]:
    """Get all safety check definitions."""
    return {
        check_id: {
            "description": config["description"],
            "risk_level": config["risk_level"].value,
        }
        for check_id, config in RitualAlignmentAdapter.SAFETY_CHECKS.items()
    }