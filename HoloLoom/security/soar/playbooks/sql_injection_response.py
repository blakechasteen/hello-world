"""
SQL Injection Response Playbook

Automated response to SQL injection attacks.

**Implemented: 2025-11-16**
**Severity**: CRITICAL
**Triggers**: security.attack.sql_injection
"""

from ..core import SecurityEvent, PlaybookResult, PlaybookMetadata, PlaybookSeverity
from ..actions import ActionExecutor


# Playbook metadata
METADATA = PlaybookMetadata(
    name="SQL Injection Response",
    description="Automated response to SQL injection attacks",
    severity=PlaybookSeverity.CRITICAL,
    triggers=["security.attack.sql_injection"],
    version="1.0.0",
    author="HoloLoom Security Team",
    timeout_seconds=120.0,
    tags=["attack", "injection", "database"],
)


async def execute(event: SecurityEvent, actions: ActionExecutor) -> PlaybookResult:
    """
    SQL Injection Response Playbook

    Steps:
    1. Block source IP immediately
    2. Revoke active sessions from IP
    3. Alert security team (CRITICAL)
    4. Collect forensics (logs, DB queries, payload)
    5. Create incident ticket
    6. Update WAF rules to block similar patterns

    Args:
        event: Security event
        actions: Action executor

    Returns:
        PlaybookResult
    """
    actions_taken = []
    actions_failed = []

    # Step 1: Block source IP (1 hour)
    if event.source_ip:
        result = await actions.block_ip(
            event.source_ip,
            duration=3600,  # 1 hour
            reason="sql_injection_attempt",
        )
        if result.success:
            actions_taken.append("block_ip")
        else:
            actions_failed.append("block_ip")

        # Step 2: Revoke active sessions from IP
        result = await actions.revoke_sessions(source_ip=event.source_ip)
        if result.success:
            actions_taken.append("revoke_sessions")
        else:
            actions_failed.append("revoke_sessions")

    # Step 3: Alert security team
    result = await actions.send_alert(
        severity="critical",
        channels=["slack", "pagerduty"],
        title="SQL Injection Attack Detected",
        message=f"""
SQL injection attack detected and blocked.

**Source IP**: {event.source_ip}
**User**: {event.user_id or "Unknown"}
**Payload**: {event.payload[:200] if event.payload else "N/A"}
**Event ID**: {event.event_id}

The source IP has been blocked for 1 hour and active sessions revoked.
        """.strip(),
    )
    if result.success:
        actions_taken.append("send_alert")
    else:
        actions_failed.append("send_alert")

    # Step 4: Collect forensics
    forensics = await actions.collect_forensics(event)
    actions_taken.append("collect_forensics")

    # Step 5: Create incident ticket
    result = await actions.create_incident(
        title=f"SQL Injection from {event.source_ip}",
        severity="critical",
        forensics=forensics,
    )
    incident_id = None
    if result.success:
        actions_taken.append("create_incident")
        incident_id = result.metadata.get("incident_id")
    else:
        actions_failed.append("create_incident")

    # Step 6: Update WAF rules (block similar payloads)
    if event.payload:
        # Extract SQL keywords from payload
        waf_rules = [
            {
                "pattern": event.payload[:100],  # First 100 chars
                "action": "block",
                "reason": "sql_injection_pattern",
            }
        ]

        result = await actions.update_waf_rules(waf_rules)
        if result.success:
            actions_taken.append("update_waf_rules")
        else:
            actions_failed.append("update_waf_rules")

    return PlaybookResult(
        playbook_name=METADATA.name,
        execution_id="",  # Will be set by engine
        status="success" if not actions_failed else "partial_success",
        actions_taken=actions_taken,
        actions_failed=actions_failed,
        incident_id=incident_id,
        metadata={
            "blocked_ip": event.source_ip,
            "payload_snippet": event.payload[:200] if event.payload else None,
        },
    )
