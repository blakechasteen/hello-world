"""
SOAR Playbook Testing Framework

Utilities for testing playbooks with dry-run mode, mock fixtures, and versioning.

**Implemented: 2025-11-16**
**Phase**: 4 - Security Pipeline
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from .core import (
    SOAREngine,
    SecurityEvent,
    PlaybookResult,
    PlaybookStatus,
    ExecutionMode,
    PlaybookSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class PlaybookTestCase:
    """Test case for playbook"""
    name: str
    description: str
    event: SecurityEvent
    expected_actions: List[str]  # Actions that should be taken
    expected_status: PlaybookStatus = PlaybookStatus.SUCCESS
    mode: ExecutionMode = ExecutionMode.DRY_RUN


@dataclass
class PlaybookTestResult:
    """Result of playbook test"""
    test_name: str
    playbook_name: str
    passed: bool
    execution_result: PlaybookResult
    assertions: Dict[str, bool] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_ms: float = 0.0


class PlaybookTester:
    """
    Test harness for SOAR playbooks

    Provides utilities for testing playbooks in isolation with mock fixtures.
    """

    def __init__(self, soar_engine: SOAREngine):
        """
        Initialize playbook tester

        Args:
            soar_engine: SOAR engine instance
        """
        self.soar_engine = soar_engine
        self.test_results: List[PlaybookTestResult] = []

    async def run_test_case(
        self,
        test_case: PlaybookTestCase,
    ) -> PlaybookTestResult:
        """
        Run a single test case

        Args:
            test_case: Test case to run

        Returns:
            PlaybookTestResult
        """
        start_time = datetime.utcnow()

        try:
            # Execute playbook
            results = await self.soar_engine.process_event(
                test_case.event,
                override_mode=test_case.mode,
            )

            if not results:
                return PlaybookTestResult(
                    test_name=test_case.name,
                    playbook_name="unknown",
                    passed=False,
                    execution_result=PlaybookResult(
                        playbook_name="unknown",
                        execution_id="",
                        status=PlaybookStatus.FAILED,
                        error_message="No playbooks matched event",
                    ),
                    error_message="No playbooks matched event",
                )

            # Use first result (most tests trigger one playbook)
            result = results[0]

            # Run assertions
            assertions = {}

            # Assert expected status
            status_ok = result.status == test_case.expected_status
            assertions["expected_status"] = status_ok

            # Assert expected actions were taken
            actions_ok = all(
                action in result.actions_taken
                for action in test_case.expected_actions
            )
            assertions["expected_actions"] = actions_ok

            # Assert no unexpected failures
            no_failures = len(result.actions_failed) == 0
            assertions["no_action_failures"] = no_failures

            # Overall pass/fail
            passed = all(assertions.values())

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            test_result = PlaybookTestResult(
                test_name=test_case.name,
                playbook_name=result.playbook_name,
                passed=passed,
                execution_result=result,
                assertions=assertions,
                duration_ms=duration_ms,
            )

            self.test_results.append(test_result)

            if passed:
                logger.info(
                    f"✅ Test passed: {test_case.name} ({duration_ms:.0f}ms)"
                )
            else:
                failed_assertions = [k for k, v in assertions.items() if not v]
                logger.error(
                    f"❌ Test failed: {test_case.name} - {failed_assertions}"
                )

            return test_result

        except Exception as e:
            logger.exception(f"Test error: {test_case.name}")

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            test_result = PlaybookTestResult(
                test_name=test_case.name,
                playbook_name="unknown",
                passed=False,
                execution_result=PlaybookResult(
                    playbook_name="unknown",
                    execution_id="",
                    status=PlaybookStatus.FAILED,
                    error_message=str(e),
                ),
                error_message=str(e),
                duration_ms=duration_ms,
            )

            self.test_results.append(test_result)
            return test_result

    async def run_test_suite(
        self,
        test_cases: List[PlaybookTestCase],
    ) -> Dict[str, Any]:
        """
        Run a suite of test cases

        Args:
            test_cases: List of test cases

        Returns:
            Test suite results
        """
        logger.info(f"Running test suite ({len(test_cases)} tests)...")

        results = []
        for test_case in test_cases:
            result = await self.run_test_case(test_case)
            results.append(result)

        # Compute statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        total_duration_ms = sum(r.duration_ms for r in results)

        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "total_duration_ms": total_duration_ms,
            "avg_duration_ms": total_duration_ms / total if total > 0 else 0.0,
            "results": [
                {
                    "test_name": r.test_name,
                    "playbook_name": r.playbook_name,
                    "passed": r.passed,
                    "assertions": r.assertions,
                    "duration_ms": r.duration_ms,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }

        logger.info(
            f"Test suite complete: {passed}/{total} passed "
            f"({summary['pass_rate']:.1%}) in {total_duration_ms:.0f}ms"
        )

        return summary

    def get_test_report(self) -> str:
        """
        Get human-readable test report

        Returns:
            Formatted test report
        """
        if not self.test_results:
            return "No tests run yet."

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        failed = total - passed

        report = []
        report.append("=" * 70)
        report.append("SOAR Playbook Test Report")
        report.append("=" * 70)
        report.append(f"\nTotal Tests: {total}")
        report.append(f"Passed: {passed} ✅")
        report.append(f"Failed: {failed} ❌")
        report.append(f"Pass Rate: {passed / total:.1%}")
        report.append("\n" + "=" * 70)
        report.append("Test Results")
        report.append("=" * 70)

        for result in self.test_results:
            status_icon = "✅" if result.passed else "❌"
            report.append(
                f"\n{status_icon} {result.test_name} ({result.duration_ms:.0f}ms)"
            )
            report.append(f"   Playbook: {result.playbook_name}")
            report.append(f"   Status: {result.execution_result.status.value}")
            report.append(
                f"   Actions: {len(result.execution_result.actions_taken)} taken, "
                f"{len(result.execution_result.actions_failed)} failed"
            )

            if not result.passed:
                failed_assertions = [k for k, v in result.assertions.items() if not v]
                report.append(f"   Failed Assertions: {', '.join(failed_assertions)}")

            if result.error_message:
                report.append(f"   Error: {result.error_message}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def create_test_event(
    event_type: str,
    source_ip: Optional[str] = "192.168.1.100",
    user_id: Optional[str] = "test_user",
    payload: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SecurityEvent:
    """
    Create a test security event

    Args:
        event_type: Event type (e.g., "security.attack.sql_injection")
        source_ip: Source IP
        user_id: User ID
        payload: Payload
        metadata: Additional metadata

    Returns:
        SecurityEvent
    """
    return SecurityEvent(
        event_type=event_type,
        severity=PlaybookSeverity.WARNING,
        source_ip=source_ip,
        user_id=user_id,
        payload=payload,
        metadata=metadata or {},
    )


# Pre-defined test cases for built-in playbooks
SQL_INJECTION_TEST = PlaybookTestCase(
    name="SQL Injection Response",
    description="Test SQL injection playbook",
    event=create_test_event(
        event_type="security.attack.sql_injection",
        source_ip="192.168.1.100",
        payload="' OR '1'='1",
    ),
    expected_actions=[
        "block_ip",
        "revoke_sessions",
        "send_alert",
        "collect_forensics",
        "create_incident",
    ],
)

BRUTE_FORCE_TEST = PlaybookTestCase(
    name="Brute Force Response",
    description="Test brute force playbook",
    event=create_test_event(
        event_type="security.attack.brute_force",
        source_ip="192.168.1.101",
        user_id="victim_user",
        metadata={"attempt_count": 25},
    ),
    expected_actions=[
        "block_ip",
        "lock_account",
        "require_mfa",
        "send_alert",
    ],
)

DDOS_TEST = PlaybookTestCase(
    name="DDoS Mitigation",
    description="Test DDoS playbook",
    event=create_test_event(
        event_type="security.attack.ddos",
        source_ip="192.168.1.102",
        metadata={
            "is_distributed": True,
            "source_count": 150,
            "requests_per_second": 5000,
        },
    ),
    expected_actions=[
        "enable_maintenance_mode",
        "adjust_rate_limits",
        "send_alert",
        "trigger_backup",
    ],
)

DATA_BREACH_TEST = PlaybookTestCase(
    name="Data Breach Containment",
    description="Test data breach playbook",
    event=create_test_event(
        event_type="security.incident.breach_detected",
        source_ip="192.168.1.103",
        metadata={
            "affected_users": ["user1", "user2", "user3"],
            "affected_resources": [
                {"id": "db_users", "type": "database"},
                {"id": "logs_2024", "type": "file"},
            ],
            "data_type": "PII",
        },
    ),
    expected_actions=[
        "snapshot_state",
        "block_ip",
        "preserve_evidence",
        "send_alert",
        "create_incident",
    ],
)

ANOMALY_TEST = PlaybookTestCase(
    name="Anomaly Investigation",
    description="Test anomaly playbook",
    event=create_test_event(
        event_type="security.incident.anomaly_detected",
        source_ip="192.168.1.104",
        user_id="anomalous_user",
        metadata={
            "anomaly_type": "unusual_access_pattern",
            "anomaly_score": 0.85,
            "confidence": 0.75,
        },
    ),
    expected_actions=[
        "snapshot_state",
        "send_alert",
        "create_incident",
    ],
)

ALL_TEST_CASES = [
    SQL_INJECTION_TEST,
    BRUTE_FORCE_TEST,
    DDOS_TEST,
    DATA_BREACH_TEST,
    ANOMALY_TEST,
]


async def run_all_playbook_tests() -> Dict[str, Any]:
    """
    Run all built-in playbook tests

    Returns:
        Test suite results
    """
    from .core import create_soar_engine
    from .playbooks import register_all_playbooks

    # Create SOAR engine in dry-run mode
    soar = create_soar_engine(mode=ExecutionMode.DRY_RUN)

    # Register all playbooks
    register_all_playbooks(soar)

    # Create tester
    tester = PlaybookTester(soar)

    # Run test suite
    results = await tester.run_test_suite(ALL_TEST_CASES)

    # Print report
    print(tester.get_test_report())

    return results
