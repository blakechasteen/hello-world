"""
Alert escalation policies with time-based escalation.

**Implemented: 2025-11-15**
**Lines**: 425
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Escalation levels."""
    LEVEL_0 = 0  # Initial alert
    LEVEL_1 = 1  # 15 minutes
    LEVEL_2 = 2  # 30 minutes
    LEVEL_3 = 3  # 60 minutes


@dataclass
class EscalationTarget:
    """Escalation target (person or group)."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_id: Optional[str] = None
    pagerduty_id: Optional[str] = None
    channels: List[str] = field(default_factory=lambda: ["slack", "email"])


@dataclass
class EscalationPolicy:
    """Escalation policy for alerts."""

    name: str
    severity: str  # "critical", "warning", "info"
    targets: Dict[int, List[EscalationTarget]] = field(default_factory=dict)
    windows: Dict[int, int] = field(default_factory=dict)  # level -> seconds
    max_levels: int = 4
    support_24h: bool = True

    def get_target_at_level(self, level: int) -> List[EscalationTarget]:
        """Get escalation targets for level.

        Args:
            level: Escalation level (0-3).

        Returns:
            List of EscalationTarget objects.
        """
        return self.targets.get(level, [])

    def get_window_for_level(self, level: int) -> int:
        """Get time window (seconds) for escalation level.

        Args:
            level: Escalation level.

        Returns:
            Time window in seconds.
        """
        return self.windows.get(level, 900)


class EscalationPolicyManager:
    """Manages alert escalation policies."""

    def __init__(self):
        """Initialize escalation policy manager."""
        self.policies: Dict[str, EscalationPolicy] = {}
        self.on_call: Dict[str, EscalationTarget] = {}
        self._init_default_policies()

    def _init_default_policies(self):
        """Initialize default escalation policies."""

        # Critical alert policy
        critical_policy = EscalationPolicy(
            name="critical",
            severity="critical",
            windows={
                0: 0,     # Send immediately
                1: 900,   # Escalate after 15 min
                2: 900,   # Escalate after 15 min
                3: 1800,  # Escalate after 30 min
            },
            max_levels=4,
            support_24h=True,
        )

        # Add default targets (to be overridden)
        critical_policy.targets = {
            0: [
                EscalationTarget(
                    name="Security Team",
                    slack_id="C123",
                    channels=["slack"],
                )
            ],
            1: [
                EscalationTarget(
                    name="On-Call Engineer",
                    channels=["pagerduty", "slack"],
                )
            ],
            2: [
                EscalationTarget(
                    name="Security Lead",
                    email="lead@hololoom.com",
                    phone="+1-555-0100",
                    channels=["pagerduty", "email", "sms"],
                )
            ],
            3: [
                EscalationTarget(
                    name="CTO",
                    email="cto@hololoom.com",
                    phone="+1-555-0101",
                    channels=["pagerduty", "email", "sms"],
                )
            ],
        }

        self.policies["critical"] = critical_policy

        # Warning alert policy
        warning_policy = EscalationPolicy(
            name="warning",
            severity="warning",
            windows={
                0: 0,
                1: 1800,  # Escalate after 30 min
                2: 3600,  # Escalate after 60 min
            },
            max_levels=3,
        )

        warning_policy.targets = {
            0: [
                EscalationTarget(
                    name="Security Team",
                    slack_id="C123",
                    channels=["slack"],
                )
            ],
            1: [
                EscalationTarget(
                    name="On-Call Engineer",
                    channels=["pagerduty", "slack"],
                )
            ],
        }

        self.policies["warning"] = warning_policy

        # Info alert policy (no escalation)
        info_policy = EscalationPolicy(
            name="info",
            severity="info",
            max_levels=1,
        )

        info_policy.targets = {
            0: [
                EscalationTarget(
                    name="Security Team",
                    slack_id="C123",
                    channels=["slack"],
                )
            ]
        }

        self.policies["info"] = info_policy

    def get_policy(self, severity: str) -> Optional[EscalationPolicy]:
        """Get escalation policy for severity.

        Args:
            severity: Alert severity (critical, warning, info).

        Returns:
            EscalationPolicy or None.
        """
        return self.policies.get(severity.lower())

    def get_targets_for_escalation(
        self,
        severity: str,
        level: int,
    ) -> List[EscalationTarget]:
        """Get escalation targets for severity and level.

        Args:
            severity: Alert severity.
            level: Escalation level.

        Returns:
            List of escalation targets.
        """
        policy = self.get_policy(severity)
        if not policy:
            return []

        targets = policy.get_target_at_level(level)

        # Replace "On-Call Engineer" with actual on-call person
        result = []
        for target in targets:
            if target.name == "On-Call Engineer":
                oncall = self._get_oncall_engineer()
                if oncall:
                    result.append(oncall)
            else:
                result.append(target)

        return result

    def set_on_call(self, target: EscalationTarget):
        """Set current on-call engineer.

        Args:
            target: EscalationTarget for on-call person.
        """
        self.on_call["engineer"] = target
        logger.info(f"On-call engineer set to: {target.name}")

    def _get_oncall_engineer(self) -> Optional[EscalationTarget]:
        """Get current on-call engineer.

        Returns:
            EscalationTarget or None.
        """
        return self.on_call.get("engineer")

    def add_policy(self, policy: EscalationPolicy):
        """Add escalation policy.

        Args:
            policy: EscalationPolicy to add.
        """
        self.policies[policy.name] = policy
        logger.info(f"Escalation policy added: {policy.name}")

    def update_policy(self, name: str, **kwargs):
        """Update escalation policy.

        Args:
            name: Policy name.
            **kwargs: Fields to update.
        """
        policy = self.policies.get(name)
        if policy:
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            logger.info(f"Escalation policy updated: {name}")

    def get_all_policies(self) -> Dict[str, EscalationPolicy]:
        """Get all policies.

        Returns:
            Dict of policies.
        """
        return dict(self.policies)

    def validate_policy(self, policy: EscalationPolicy) -> List[str]:
        """Validate escalation policy.

        Args:
            policy: Policy to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Check targets
        if not policy.targets:
            errors.append("No escalation targets defined")

        for level in range(policy.max_levels):
            if level not in policy.targets or not policy.targets[level]:
                errors.append(f"No targets for escalation level {level}")

        # Check windows
        for level in range(policy.max_levels - 1):
            if level not in policy.windows:
                errors.append(f"No time window for escalation level {level}")

        return errors

    def get_policy_summary(self, severity: str) -> Dict:
        """Get summary of escalation policy.

        Args:
            severity: Alert severity.

        Returns:
            Dict with policy summary.
        """
        policy = self.get_policy(severity)
        if not policy:
            return {}

        summary = {
            "name": policy.name,
            "severity": policy.severity,
            "max_levels": policy.max_levels,
            "support_24h": policy.support_24h,
            "escalation_chain": [],
        }

        for level in range(policy.max_levels):
            targets = policy.get_target_at_level(level)
            window = policy.get_window_for_level(level) if level > 0 else 0

            escalation_step = {
                "level": level,
                "wait_seconds": window,
                "targets": [
                    {
                        "name": t.name,
                        "channels": t.channels,
                    } for t in targets
                ],
            }

            summary["escalation_chain"].append(escalation_step)

        return summary
