"""
Comprehensive Tests for SOAR System

Tests cover:
- Core engine functionality
- Action execution (unit tests)
- Playbook execution (integration tests)
- Dry-run mode
- Manual approval workflow
- Audit logging

**Implemented: 2025-11-16**
"""

import pytest
import asyncio
from pathlib import Path
import json
from datetime import datetime
from typing import List

from HoloLoom.security.soar import (
    SOAREngine,
    SecurityEvent,
    PlaybookResult,
    PlaybookMetadata,
    PlaybookSeverity,
    PlaybookStatus,
    ExecutionMode,
    ActionExecutor,
    create_soar_engine,
    register_all_playbooks,
)

from HoloLoom.security.soar.testing import (
    PlaybookTester,
    PlaybookTestCase,
    create_test_event,
    ALL_TEST_CASES,
)


# ====================
# Core Engine Tests
# ====================

@pytest.mark.asyncio
async def test_soar_engine_initialization():
    """Test SOAR engine initialization"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)

    assert soar.mode == ExecutionMode.DRY_RUN
    assert soar.registry is not None
    assert soar.stats["total_executions"] == 0


@pytest.mark.asyncio
async def test_playbook_registration():
    """Test playbook registration"""
    soar = create_soar_engine()

    # Define simple playbook
    metadata = PlaybookMetadata(
        name="Test Playbook",
        description="Test playbook",
        severity=PlaybookSeverity.INFO,
        triggers=["test.event"],
    )

    async def test_playbook(event: SecurityEvent, actions: ActionExecutor):
        return PlaybookResult(
            playbook_name="Test Playbook",
            execution_id="",
            status=PlaybookStatus.SUCCESS,
        )

    # Register
    soar.registry.register(metadata, test_playbook)

    # Verify registration
    assert "Test Playbook" in soar.registry.list_playbooks()
    assert soar.registry.get_metadata("Test Playbook") == metadata
    assert soar.registry.get_playbook("Test Playbook") == test_playbook


@pytest.mark.asyncio
async def test_trigger_matching():
    """Test event trigger matching"""
    soar = create_soar_engine()

    # Register playbook with specific trigger
    metadata = PlaybookMetadata(
        name="SQL Test",
        description="Test",
        severity=PlaybookSeverity.CRITICAL,
        triggers=["security.attack.sql_injection"],
    )

    async def sql_playbook(event: SecurityEvent, actions: ActionExecutor):
        return PlaybookResult(
            playbook_name="SQL Test",
            execution_id="",
            status=PlaybookStatus.SUCCESS,
        )

    soar.registry.register(metadata, sql_playbook)

    # Test exact match
    playbooks = soar.registry.get_playbooks_for_trigger("security.attack.sql_injection")
    assert "SQL Test" in playbooks

    # Test no match
    playbooks = soar.registry.get_playbooks_for_trigger("security.attack.xss")
    assert "SQL Test" not in playbooks


@pytest.mark.asyncio
async def test_event_processing():
    """Test security event processing"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)

    # Register test playbook
    metadata = PlaybookMetadata(
        name="Event Test",
        description="Test",
        severity=PlaybookSeverity.WARNING,
        triggers=["test.event"],
    )

    async def event_playbook(event: SecurityEvent, actions: ActionExecutor):
        assert event.event_type == "test.event"
        assert event.source_ip == "1.2.3.4"
        return PlaybookResult(
            playbook_name="Event Test",
            execution_id="",
            status=PlaybookStatus.SUCCESS,
            actions_taken=["test_action"],
        )

    soar.registry.register(metadata, event_playbook)

    # Process event
    event = SecurityEvent(
        event_type="test.event",
        source_ip="1.2.3.4",
    )

    results = await soar.process_event(event)

    assert len(results) == 1
    assert results[0].playbook_name == "Event Test"
    assert results[0].status == PlaybookStatus.SUCCESS
    assert "test_action" in results[0].actions_taken


# ====================
# Action Tests
# ====================

@pytest.mark.asyncio
async def test_action_executor_dry_run():
    """Test action executor in dry-run mode"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # Test block_ip in dry-run
    result = await actions.block_ip("1.2.3.4", duration=3600)

    assert result.success
    assert result.action_name == "block_ip"
    assert result.metadata["dry_run"] is True
    assert result.metadata["ip"] == "1.2.3.4"


@pytest.mark.asyncio
async def test_action_executor_logging():
    """Test action executor logging"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # Execute multiple actions
    await actions.block_ip("1.2.3.4")
    await actions.send_alert("warning", ["slack"], "Test alert")
    await actions.create_incident("Test incident", "warning")

    # Check action log
    log = actions.get_action_log()

    assert len(log) == 3
    assert log[0]["action_name"] == "block_ip"
    assert log[1]["action_name"] == "send_alert"
    assert log[2]["action_name"] == "create_incident"


@pytest.mark.asyncio
async def test_all_network_actions():
    """Test all network actions"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # block_ip
    result = await actions.block_ip("1.2.3.4", duration=3600)
    assert result.success

    # unblock_ip
    result = await actions.unblock_ip("1.2.3.4")
    assert result.success

    # throttle_user
    result = await actions.throttle_user("user123", max_requests=10, window_seconds=60)
    assert result.success


@pytest.mark.asyncio
async def test_all_auth_actions():
    """Test all authentication actions"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # revoke_token
    result = await actions.revoke_token("token123")
    assert result.success

    # revoke_sessions
    result = await actions.revoke_sessions(user_id="user123")
    assert result.success

    # require_mfa
    result = await actions.require_mfa("user123")
    assert result.success

    # lock_account
    result = await actions.lock_account("user123", reason="test")
    assert result.success


@pytest.mark.asyncio
async def test_all_alerting_actions():
    """Test all alerting actions"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # send_alert
    result = await actions.send_alert("critical", ["slack"], "Test alert")
    assert result.success

    # escalate
    result = await actions.escalate("INC-123", "security", "Escalation message")
    assert result.success

    # create_incident
    result = await actions.create_incident("Test incident", "critical")
    assert result.success
    assert "incident_id" in result.metadata


@pytest.mark.asyncio
async def test_all_forensics_actions():
    """Test all forensics actions"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # collect_logs
    result = await actions.collect_logs("app", time_range_minutes=60)
    assert result.success

    # snapshot_state
    result = await actions.snapshot_state(["db", "cache"])
    assert result.success
    assert "snapshot_id" in result.metadata

    # preserve_evidence
    result = await actions.preserve_evidence("event123", ["logs", "network"])
    assert result.success
    assert "evidence_id" in result.metadata


@pytest.mark.asyncio
async def test_all_response_actions():
    """Test all response actions"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # quarantine_resource
    result = await actions.quarantine_resource("file123", "file")
    assert result.success

    # rollback_config
    result = await actions.rollback_config("api", "SNAP-123")
    assert result.success

    # enable_maintenance_mode
    result = await actions.enable_maintenance_mode(duration_minutes=30)
    assert result.success


@pytest.mark.asyncio
async def test_all_integration_actions():
    """Test all integration actions"""
    actions = ActionExecutor(mode=ExecutionMode.DRY_RUN)

    # update_waf_rules
    result = await actions.update_waf_rules([{"pattern": "test", "action": "block"}])
    assert result.success

    # adjust_rate_limits
    result = await actions.adjust_rate_limits({"/api/test": 10})
    assert result.success

    # trigger_backup
    result = await actions.trigger_backup(["db", "config"])
    assert result.success
    assert "backup_id" in result.metadata


# ====================
# Playbook Tests
# ====================

@pytest.mark.asyncio
async def test_all_playbooks():
    """Test all pre-built playbooks"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)

    # Register all playbooks
    playbook_names = register_all_playbooks(soar)

    # Verify all registered
    assert len(playbook_names) == 5
    assert "SQL Injection Response" in playbook_names
    assert "Brute Force Response" in playbook_names
    assert "DDoS Mitigation" in playbook_names
    assert "Data Breach Containment" in playbook_names
    assert "Anomaly Investigation" in playbook_names


@pytest.mark.asyncio
async def test_sql_injection_playbook():
    """Test SQL injection playbook"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)
    register_all_playbooks(soar)

    event = SecurityEvent(
        event_type="security.attack.sql_injection",
        source_ip="1.2.3.4",
        payload="' OR '1'='1",
    )

    results = await soar.process_event(event)

    assert len(results) == 1
    result = results[0]

    assert result.status == PlaybookStatus.SUCCESS
    assert "block_ip" in result.actions_taken
    assert "send_alert" in result.actions_taken


@pytest.mark.asyncio
async def test_brute_force_playbook():
    """Test brute force playbook"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)
    register_all_playbooks(soar)

    event = SecurityEvent(
        event_type="security.attack.brute_force",
        source_ip="1.2.3.4",
        user_id="victim",
        metadata={"attempt_count": 50},
    )

    results = await soar.process_event(event)

    assert len(results) == 1
    result = results[0]

    assert result.status == PlaybookStatus.SUCCESS
    assert "block_ip" in result.actions_taken
    assert "lock_account" in result.actions_taken


@pytest.mark.asyncio
async def test_ddos_playbook():
    """Test DDoS mitigation playbook"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)
    register_all_playbooks(soar)

    event = SecurityEvent(
        event_type="security.attack.ddos",
        metadata={
            "is_distributed": True,
            "source_count": 200,
            "requests_per_second": 10000,
        },
    )

    results = await soar.process_event(event)

    assert len(results) == 1
    result = results[0]

    assert result.status == PlaybookStatus.SUCCESS
    assert "enable_maintenance_mode" in result.actions_taken
    assert "adjust_rate_limits" in result.actions_taken


# ====================
# Advanced Features Tests
# ====================

@pytest.mark.asyncio
async def test_manual_approval_workflow():
    """Test manual approval workflow"""
    soar = create_soar_engine(mode=ExecutionMode.PRODUCTION)

    # Register playbook requiring approval
    metadata = PlaybookMetadata(
        name="Approval Test",
        description="Test",
        severity=PlaybookSeverity.CRITICAL,
        triggers=["test.approval"],
        requires_approval=True,
    )

    async def approval_playbook(event: SecurityEvent, actions: ActionExecutor):
        return PlaybookResult(
            playbook_name="Approval Test",
            execution_id="",
            status=PlaybookStatus.SUCCESS,
        )

    soar.registry.register(metadata, approval_playbook)

    # Process event
    event = SecurityEvent(event_type="test.approval")
    results = await soar.process_event(event)

    # Should be pending approval
    assert len(results) == 1
    assert results[0].status == PlaybookStatus.PENDING
    assert len(soar.pending_approvals) == 1


@pytest.mark.asyncio
async def test_audit_logging(tmp_path):
    """Test audit logging"""
    audit_log_path = tmp_path / "audit.jsonl"

    soar = create_soar_engine(
        mode=ExecutionMode.DRY_RUN,
        enable_audit_log=True,
        audit_log_path=audit_log_path,
    )

    # Register simple playbook
    metadata = PlaybookMetadata(
        name="Audit Test",
        description="Test",
        severity=PlaybookSeverity.INFO,
        triggers=["test.audit"],
    )

    async def audit_playbook(event: SecurityEvent, actions: ActionExecutor):
        return PlaybookResult(
            playbook_name="Audit Test",
            execution_id="",
            status=PlaybookStatus.SUCCESS,
            actions_taken=["test_action"],
        )

    soar.registry.register(metadata, audit_playbook)

    # Execute playbook
    event = SecurityEvent(event_type="test.audit")
    await soar.process_event(event)

    # Check audit log
    assert audit_log_path.exists()

    with open(audit_log_path) as f:
        lines = f.readlines()
        assert len(lines) == 1

        log_entry = json.loads(lines[0])
        assert log_entry["playbook_name"] == "Audit Test"
        assert log_entry["status"] == "success"
        assert "test_action" in log_entry["actions_taken"]


@pytest.mark.asyncio
async def test_execution_statistics():
    """Test execution statistics"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)

    # Register test playbook
    metadata = PlaybookMetadata(
        name="Stats Test",
        description="Test",
        severity=PlaybookSeverity.INFO,
        triggers=["test.stats"],
    )

    async def stats_playbook(event: SecurityEvent, actions: ActionExecutor):
        return PlaybookResult(
            playbook_name="Stats Test",
            execution_id="",
            status=PlaybookStatus.SUCCESS,
        )

    soar.registry.register(metadata, stats_playbook)

    # Execute multiple times
    for i in range(3):
        event = SecurityEvent(event_type="test.stats")
        await soar.process_event(event)

    # Check statistics
    stats = soar.get_statistics()

    assert stats["total_executions"] == 3
    assert stats["dry_run_executions"] == 3


@pytest.mark.asyncio
async def test_playbook_timeout():
    """Test playbook timeout handling"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)

    # Register slow playbook with short timeout
    metadata = PlaybookMetadata(
        name="Timeout Test",
        description="Test",
        severity=PlaybookSeverity.INFO,
        triggers=["test.timeout"],
        timeout_seconds=1.0,  # 1 second timeout
    )

    async def slow_playbook(event: SecurityEvent, actions: ActionExecutor):
        await asyncio.sleep(5)  # Sleep longer than timeout
        return PlaybookResult(
            playbook_name="Timeout Test",
            execution_id="",
            status=PlaybookStatus.SUCCESS,
        )

    soar.registry.register(metadata, slow_playbook)

    # Execute playbook
    event = SecurityEvent(event_type="test.timeout")
    results = await soar.process_event(event)

    # Should timeout
    assert len(results) == 1
    assert results[0].status == PlaybookStatus.TIMEOUT


# ====================
# Testing Framework Tests
# ====================

@pytest.mark.asyncio
async def test_playbook_tester():
    """Test playbook testing framework"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)
    register_all_playbooks(soar)

    tester = PlaybookTester(soar)

    # Run single test
    test_case = PlaybookTestCase(
        name="Test Case",
        description="Test",
        event=create_test_event("security.attack.sql_injection"),
        expected_actions=["block_ip", "send_alert"],
    )

    result = await tester.run_test_case(test_case)

    assert result.passed
    assert result.playbook_name == "SQL Injection Response"


@pytest.mark.asyncio
async def test_full_test_suite():
    """Test full playbook test suite"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)
    register_all_playbooks(soar)

    tester = PlaybookTester(soar)

    # Run all test cases
    results = await tester.run_test_suite(ALL_TEST_CASES)

    assert results["total_tests"] == 5
    assert results["passed"] >= 4  # At least 4 should pass
    assert results["pass_rate"] >= 0.8  # At least 80% pass rate


# ====================
# Performance Tests
# ====================

@pytest.mark.asyncio
async def test_playbook_execution_performance():
    """Test playbook execution performance"""
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)
    register_all_playbooks(soar)

    event = SecurityEvent(
        event_type="security.attack.sql_injection",
        source_ip="1.2.3.4",
    )

    # Execute 10 times and measure
    start = datetime.utcnow()

    for _ in range(10):
        await soar.process_event(event)

    duration_ms = (datetime.utcnow() - start).total_seconds() * 1000

    # Should complete 10 executions in under 1 second
    assert duration_ms < 1000


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
