"""
DDoS Mitigation Playbook

Automated response to Distributed Denial of Service attacks.

**Implemented: 2025-11-16**
**Severity**: CRITICAL
**Triggers**: security.attack.dos, security.attack.ddos
"""

from ..core import SecurityEvent, PlaybookResult, PlaybookMetadata, PlaybookSeverity
from ..actions import ActionExecutor


# Playbook metadata
METADATA = PlaybookMetadata(
    name="DDoS Mitigation",
    description="Automated response to DDoS attacks",
    severity=PlaybookSeverity.CRITICAL,
    triggers=["security.attack.dos", "security.attack.ddos"],
    version="1.0.0",
    author="HoloLoom Security Team",
    timeout_seconds=180.0,
    tags=["attack", "ddos", "availability"],
)


async def execute(event: SecurityEvent, actions: ActionExecutor) -> PlaybookResult:
    """
    DDoS Mitigation Playbook

    Steps:
    1. Block source IP (if single-source DoS)
    2. Enable maintenance mode (reduce load)
    3. Adjust rate limits globally (stricter)
    4. Update WAF rules (block attack patterns)
    5. Alert security + SRE teams (CRITICAL)
    6. Trigger backup (preserve state)
    7. Escalate to SRE for CDN/infrastructure changes

    Args:
        event: Security event
        actions: Action executor

    Returns:
        PlaybookResult
    """
    actions_taken = []
    actions_failed = []

    # Determine if single-source or distributed
    is_distributed = event.metadata.get("is_distributed", False)
    source_count = event.metadata.get("source_count", 1)

    # Step 1: Block source IP (if single-source)
    if event.source_ip and not is_distributed:
        result = await actions.block_ip(
            event.source_ip,
            duration=7200,  # 2 hours
            reason="dos_attack",
        )
        if result.success:
            actions_taken.append("block_ip")
        else:
            actions_failed.append("block_ip")

    # Step 2: Enable maintenance mode (reduce load)
    result = await actions.enable_maintenance_mode(duration_minutes=15)
    if result.success:
        actions_taken.append("enable_maintenance_mode")
    else:
        actions_failed.append("enable_maintenance_mode")

    # Step 3: Adjust rate limits globally (stricter)
    rate_limit_adjustments = {
        "/api/*": 10,  # 10 requests per minute (global)
        "/health": 120,  # Keep health checks functional
    }

    result = await actions.adjust_rate_limits(rate_limit_adjustments)
    if result.success:
        actions_taken.append("adjust_rate_limits")
    else:
        actions_failed.append("adjust_rate_limits")

    # Step 4: Update WAF rules (block attack patterns)
    attack_signature = event.metadata.get("attack_signature")
    if attack_signature:
        waf_rules = [
            {
                "pattern": attack_signature,
                "action": "block",
                "reason": "ddos_attack_pattern",
            },
            {
                "pattern": "user_agent:bot",  # Block common bot user agents
                "action": "challenge",
                "reason": "ddos_prevention",
            },
        ]

        result = await actions.update_waf_rules(waf_rules)
        if result.success:
            actions_taken.append("update_waf_rules")
        else:
            actions_failed.append("update_waf_rules")

    # Step 5: Alert security + SRE teams
    result = await actions.send_alert(
        severity="critical",
        channels=["slack", "pagerduty", "email"],
        title="DDoS Attack Detected - System Under Load",
        message=f"""
DDoS attack detected. System is under heavy load.

**Type**: {"Distributed" if is_distributed else "Single-source"}
**Source Count**: {source_count}
**Source IP**: {event.source_ip or "Multiple sources"}
**Traffic Rate**: {event.metadata.get('requests_per_second', 'Unknown')} req/s
**Event ID**: {event.event_id}

**Actions Taken**:
- Maintenance mode enabled (15 minutes)
- Rate limits reduced globally
- WAF rules updated

**Next Steps**:
- SRE team should review CDN/infrastructure
- Consider enabling advanced DDoS protection (Cloudflare, AWS Shield)
        """.strip(),
    )
    if result.success:
        actions_taken.append("send_alert")
    else:
        actions_failed.append("send_alert")

    # Step 6: Trigger backup (preserve state before potential crash)
    result = await actions.trigger_backup(
        components=["db", "cache", "config"],
    )
    if result.success:
        actions_taken.append("trigger_backup")
    else:
        actions_failed.append("trigger_backup")

    # Step 7: Create incident and escalate
    result = await actions.create_incident(
        title=f"DDoS Attack - {source_count} sources",
        severity="critical",
        forensics={
            "is_distributed": is_distributed,
            "source_count": source_count,
            "requests_per_second": event.metadata.get("requests_per_second"),
            "attack_signature": attack_signature,
        },
    )

    incident_id = None
    if result.success:
        actions_taken.append("create_incident")
        incident_id = result.metadata.get("incident_id")

        # Escalate to SRE
        await actions.escalate(
            incident_id=incident_id,
            target_team="sre",
            message="DDoS attack in progress. Requires infrastructure changes (CDN, load balancing).",
        )
        actions_taken.append("escalate")
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
            "is_distributed": is_distributed,
            "source_count": source_count,
            "maintenance_mode_enabled": True,
        },
    )
