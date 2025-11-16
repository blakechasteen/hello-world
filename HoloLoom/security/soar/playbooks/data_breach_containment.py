"""
Data Breach Containment Playbook

Automated response to data breach incidents.

**Implemented**: 2025-11-16**
**Severity**: CRITICAL
**Triggers**: security.incident.breach_detected, security.data.data_leak
"""

from ..core import SecurityEvent, PlaybookResult, PlaybookMetadata, PlaybookSeverity
from ..actions import ActionExecutor


# Playbook metadata
METADATA = PlaybookMetadata(
    name="Data Breach Containment",
    description="Automated response to data breach incidents",
    severity=PlaybookSeverity.CRITICAL,
    triggers=["security.incident.breach_detected", "security.data.data_leak"],
    version="1.0.0",
    author="HoloLoom Security Team",
    requires_approval=True,  # Requires manual approval (high impact)
    timeout_seconds=300.0,
    tags=["incident", "breach", "data_leak"],
)


async def execute(event: SecurityEvent, actions: ActionExecutor) -> PlaybookResult:
    """
    Data Breach Containment Playbook

    Steps:
    1. Snapshot system state immediately
    2. Block source IP (if known)
    3. Revoke all tokens/sessions for affected users
    4. Quarantine affected resources (databases, files)
    5. Preserve evidence (logs, network captures, memory dumps)
    6. Alert security + legal + executive teams (CRITICAL)
    7. Create incident ticket
    8. Trigger comprehensive backup
    9. Escalate to legal for compliance requirements

    Args:
        event: Security event
        actions: Action executor

    Returns:
        PlaybookResult
    """
    actions_taken = []
    actions_failed = []

    # Extract breach details
    affected_users = event.metadata.get("affected_users", [])
    affected_resources = event.metadata.get("affected_resources", [])
    data_type = event.metadata.get("data_type", "unknown")

    # Step 1: Snapshot system state immediately
    result = await actions.snapshot_state(
        components=["db", "cache", "config", "logs"],
    )
    if result.success:
        actions_taken.append("snapshot_state")
        snapshot_id = result.metadata.get("snapshot_id")
    else:
        actions_failed.append("snapshot_state")
        snapshot_id = None

    # Step 2: Block source IP (if known)
    if event.source_ip:
        result = await actions.block_ip(
            event.source_ip,
            duration=86400,  # 24 hours
            reason="data_breach_source",
        )
        if result.success:
            actions_taken.append("block_ip")
        else:
            actions_failed.append("block_ip")

    # Step 3: Revoke all tokens/sessions for affected users
    for user_id in affected_users:
        result = await actions.revoke_sessions(user_id=user_id)
        if result.success:
            actions_taken.append(f"revoke_sessions:{user_id}")
        else:
            actions_failed.append(f"revoke_sessions:{user_id}")

    # Step 4: Quarantine affected resources
    for resource in affected_resources:
        result = await actions.quarantine_resource(
            resource_id=resource.get("id"),
            resource_type=resource.get("type", "unknown"),
        )
        if result.success:
            actions_taken.append(f"quarantine_resource:{resource.get('id')}")
        else:
            actions_failed.append(f"quarantine_resource:{resource.get('id')}")

    # Step 5: Preserve evidence
    result = await actions.preserve_evidence(
        event_id=event.event_id,
        evidence_types=["logs", "network", "memory", "disk"],
    )
    if result.success:
        actions_taken.append("preserve_evidence")
        evidence_id = result.metadata.get("evidence_id")
    else:
        actions_failed.append("preserve_evidence")
        evidence_id = None

    # Also collect comprehensive logs
    for log_source in ["app", "auth", "api", "db"]:
        result = await actions.collect_logs(
            source=log_source,
            time_range_minutes=240,  # Last 4 hours
        )
        if result.success:
            actions_taken.append(f"collect_logs:{log_source}")

    # Step 6: Alert security + legal + executive teams
    result = await actions.send_alert(
        severity="critical",
        channels=["slack", "pagerduty", "email", "sms"],
        title="🚨 DATA BREACH DETECTED - IMMEDIATE ACTION REQUIRED",
        message=f"""
CRITICAL: Data breach detected and containment initiated.

**Data Type**: {data_type}
**Affected Users**: {len(affected_users)} users
**Affected Resources**: {len(affected_resources)} resources
**Source IP**: {event.source_ip or "Unknown"}
**Event ID**: {event.event_id}

**Containment Actions Taken**:
- System state snapshot: {snapshot_id}
- Source IP blocked (24 hours)
- User sessions revoked: {len(affected_users)}
- Resources quarantined: {len(affected_resources)}
- Evidence preserved: {evidence_id}

**IMMEDIATE ACTIONS REQUIRED**:
1. Legal team: Review compliance requirements (GDPR, CCPA, etc.)
2. Security team: Full forensics investigation
3. Executive team: Customer communication planning
4. PR team: Media response preparation

**Runbook**: See incident ticket for full details.
        """.strip(),
    )
    if result.success:
        actions_taken.append("send_alert")
    else:
        actions_failed.append("send_alert")

    # Step 7: Create incident ticket
    forensics = {
        "event_id": event.event_id,
        "timestamp": event.timestamp.isoformat(),
        "source_ip": event.source_ip,
        "affected_users": affected_users,
        "affected_user_count": len(affected_users),
        "affected_resources": affected_resources,
        "data_type": data_type,
        "snapshot_id": snapshot_id,
        "evidence_id": evidence_id,
    }

    result = await actions.create_incident(
        title=f"Data Breach - {data_type} ({len(affected_users)} users)",
        severity="critical",
        forensics=forensics,
    )

    incident_id = None
    if result.success:
        actions_taken.append("create_incident")
        incident_id = result.metadata.get("incident_id")
    else:
        actions_failed.append("create_incident")

    # Step 8: Trigger comprehensive backup
    result = await actions.trigger_backup(
        components=["db", "cache", "config", "logs", "files"],
    )
    if result.success:
        actions_taken.append("trigger_backup")
    else:
        actions_failed.append("trigger_backup")

    # Step 9: Escalate to legal
    if incident_id:
        result = await actions.escalate(
            incident_id=incident_id,
            target_team="legal",
            message=f"Data breach affecting {len(affected_users)} users. Review compliance requirements (breach notification laws, GDPR, CCPA).",
        )
        if result.success:
            actions_taken.append("escalate:legal")

    return PlaybookResult(
        playbook_name=METADATA.name,
        execution_id="",
        status="success" if not actions_failed else "partial_success",
        actions_taken=actions_taken,
        actions_failed=actions_failed,
        incident_id=incident_id,
        metadata={
            "affected_user_count": len(affected_users),
            "affected_resource_count": len(affected_resources),
            "data_type": data_type,
            "snapshot_id": snapshot_id,
            "evidence_id": evidence_id,
            "requires_legal_review": True,
        },
    )
