"""
Brute Force Response Playbook

Automated response to brute force authentication attacks.

**Implemented: 2025-11-16**
**Severity**: WARNING
**Triggers**: security.attack.brute_force
"""

from ..core import SecurityEvent, PlaybookResult, PlaybookMetadata, PlaybookSeverity
from ..actions import ActionExecutor


# Playbook metadata
METADATA = PlaybookMetadata(
    name="Brute Force Response",
    description="Automated response to brute force authentication attacks",
    severity=PlaybookSeverity.WARNING,
    triggers=["security.attack.brute_force"],
    version="1.0.0",
    author="HoloLoom Security Team",
    timeout_seconds=90.0,
    tags=["attack", "authentication", "brute_force"],
)


async def execute(event: SecurityEvent, actions: ActionExecutor) -> PlaybookResult:
    """
    Brute Force Response Playbook

    Steps:
    1. Block source IP (30 minutes)
    2. If user_id known, lock account temporarily
    3. Require MFA for affected user
    4. Alert security team (WARNING)
    5. Adjust rate limits (stricter for auth endpoints)
    6. Collect failed auth logs

    Args:
        event: Security event
        actions: Action executor

    Returns:
        PlaybookResult
    """
    actions_taken = []
    actions_failed = []

    # Step 1: Block source IP (30 minutes)
    if event.source_ip:
        result = await actions.block_ip(
            event.source_ip,
            duration=1800,  # 30 minutes
            reason="brute_force_attempt",
        )
        if result.success:
            actions_taken.append("block_ip")
        else:
            actions_failed.append("block_ip")

    # Step 2: Lock affected account temporarily
    if event.user_id:
        result = await actions.lock_account(
            event.user_id,
            reason="brute_force_target",
        )
        if result.success:
            actions_taken.append("lock_account")
        else:
            actions_failed.append("lock_account")

        # Step 3: Require MFA when account is unlocked
        result = await actions.require_mfa(event.user_id)
        if result.success:
            actions_taken.append("require_mfa")
        else:
            actions_failed.append("require_mfa")

    # Step 4: Alert security team
    result = await actions.send_alert(
        severity="warning",
        channels=["slack"],
        title="Brute Force Attack Detected",
        message=f"""
Brute force authentication attack detected.

**Source IP**: {event.source_ip}
**Target User**: {event.user_id or "Multiple accounts"}
**Attempts**: {event.metadata.get('attempt_count', 'Unknown')}
**Event ID**: {event.event_id}

The source IP has been blocked for 30 minutes.
        """.strip(),
    )
    if result.success:
        actions_taken.append("send_alert")
    else:
        actions_failed.append("send_alert")

    # Step 5: Adjust rate limits for auth endpoints
    rate_limit_adjustments = {
        "/api/auth/login": 3,  # 3 attempts per minute
        "/api/auth/token": 5,
    }

    result = await actions.adjust_rate_limits(rate_limit_adjustments)
    if result.success:
        actions_taken.append("adjust_rate_limits")
    else:
        actions_failed.append("adjust_rate_limits")

    # Step 6: Collect failed auth logs
    result = await actions.collect_logs(
        source="auth",
        time_range_minutes=60,
    )
    if result.success:
        actions_taken.append("collect_logs")
    else:
        actions_failed.append("collect_logs")

    return PlaybookResult(
        playbook_name=METADATA.name,
        execution_id="",
        status="success" if not actions_failed else "partial_success",
        actions_taken=actions_taken,
        actions_failed=actions_failed,
        metadata={
            "blocked_ip": event.source_ip,
            "affected_user": event.user_id,
            "attempt_count": event.metadata.get("attempt_count"),
        },
    )
