"""
Anomaly Investigation Playbook

Automated investigation of security anomalies.

**Implemented: 2025-11-16**
**Severity**: WARNING
**Triggers**: security.incident.anomaly_detected
"""

from ..core import SecurityEvent, PlaybookResult, PlaybookMetadata, PlaybookSeverity
from ..actions import ActionExecutor


# Playbook metadata
METADATA = PlaybookMetadata(
    name="Anomaly Investigation",
    description="Automated investigation of security anomalies",
    severity=PlaybookSeverity.WARNING,
    triggers=["security.incident.anomaly_detected"],
    version="1.0.0",
    author="HoloLoom Security Team",
    timeout_seconds=180.0,
    tags=["investigation", "anomaly", "monitoring"],
)


async def execute(event: SecurityEvent, actions: ActionExecutor) -> PlaybookResult:
    """
    Anomaly Investigation Playbook

    Steps:
    1. Collect logs from all sources
    2. Snapshot system state
    3. Throttle suspected user/IP (cautious approach)
    4. Alert security team (WARNING)
    5. Preserve evidence
    6. Create investigation ticket

    Args:
        event: Security event
        actions: Action executor

    Returns:
        PlaybookResult
    """
    actions_taken = []
    actions_failed = []

    # Extract anomaly details
    anomaly_type = event.metadata.get("anomaly_type", "unknown")
    anomaly_score = event.metadata.get("anomaly_score", 0.0)
    confidence = event.metadata.get("confidence", 0.0)

    # Step 1: Collect logs from all sources
    for log_source in ["app", "auth", "api", "db"]:
        result = await actions.collect_logs(
            source=log_source,
            time_range_minutes=120,  # Last 2 hours
        )
        if result.success:
            actions_taken.append(f"collect_logs:{log_source}")
        else:
            actions_failed.append(f"collect_logs:{log_source}")

    # Step 2: Snapshot system state
    result = await actions.snapshot_state(
        components=["db", "cache", "config"],
    )
    if result.success:
        actions_taken.append("snapshot_state")
        snapshot_id = result.metadata.get("snapshot_id")
    else:
        actions_failed.append("snapshot_state")
        snapshot_id = None

    # Step 3: Throttle suspected user/IP (cautious, not full block)
    # Only throttle if confidence is high enough
    if confidence > 0.7:
        if event.user_id:
            result = await actions.throttle_user(
                user_id=event.user_id,
                max_requests=30,  # Reduced from normal 60
                window_seconds=60,
            )
            if result.success:
                actions_taken.append("throttle_user")
            else:
                actions_failed.append("throttle_user")

        if event.source_ip:
            result = await actions.throttle_user(
                user_id=f"ip:{event.source_ip}",
                max_requests=20,  # Reduced
                window_seconds=60,
            )
            if result.success:
                actions_taken.append("throttle_ip")
            else:
                actions_failed.append("throttle_ip")

    # Step 4: Alert security team
    alert_severity = "warning" if confidence > 0.7 else "info"

    result = await actions.send_alert(
        severity=alert_severity,
        channels=["slack"],
        title=f"Security Anomaly Detected: {anomaly_type}",
        message=f"""
Security anomaly detected and investigation initiated.

**Anomaly Type**: {anomaly_type}
**Anomaly Score**: {anomaly_score:.2f}
**Confidence**: {confidence:.1%}
**Source IP**: {event.source_ip or "Unknown"}
**User**: {event.user_id or "Unknown"}
**Event ID**: {event.event_id}

**Investigation Actions**:
- Logs collected (2-hour window)
- System state snapshot: {snapshot_id}
{"- User/IP throttled (cautious approach)" if confidence > 0.7 else "- No throttling (low confidence)"}

**Next Steps**:
- Security team should review logs and snapshots
- Monitor for escalation
- Adjust anomaly detection thresholds if false positive
        """.strip(),
    )
    if result.success:
        actions_taken.append("send_alert")
    else:
        actions_failed.append("send_alert")

    # Step 5: Preserve evidence (if high confidence)
    if confidence > 0.8:
        result = await actions.preserve_evidence(
            event_id=event.event_id,
            evidence_types=["logs", "network"],
        )
        if result.success:
            actions_taken.append("preserve_evidence")
            evidence_id = result.metadata.get("evidence_id")
        else:
            actions_failed.append("preserve_evidence")
            evidence_id = None
    else:
        evidence_id = None

    # Step 6: Create investigation ticket
    result = await actions.create_incident(
        title=f"Anomaly Investigation - {anomaly_type}",
        severity="warning" if confidence > 0.7 else "info",
        forensics={
            "anomaly_type": anomaly_type,
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "snapshot_id": snapshot_id,
            "evidence_id": evidence_id,
        },
    )

    incident_id = None
    if result.success:
        actions_taken.append("create_incident")
        incident_id = result.metadata.get("incident_id")
    else:
        actions_failed.append("create_incident")

    return PlaybookResult(
        playbook_name=METADATA.name,
        execution_id="",
        status="success" if not actions_failed else "partial_success",
        actions_taken=actions_taken,
        actions_failed=actions_failed,
        incident_id=incident_id,
        metadata={
            "anomaly_type": anomaly_type,
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "snapshot_id": snapshot_id,
            "evidence_id": evidence_id,
            "throttled": confidence > 0.7,
        },
    )
