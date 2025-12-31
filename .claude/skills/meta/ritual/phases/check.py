"""
/ritual check - Safety-First Checklist Phase
=============================================

Safety-first validation phase with comprehensive checklist:
- Graceful degradation
- Input validation
- Error messages
- Secrets handling
- Timeout handling
- Resource cleanup

Maps to: EVA (Extra-Vehicular Activity) lifecycle state
Emits: ritual.check.completed event

Can optionally integrate with:
- skill_tester for automated testing
- skill_security_analyzer for security scanning
- HoloLoom alignment framework for risk assessment
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
import uuid

# Optional HoloLoom integration
if TYPE_CHECKING:
    from hololoom import HoloLoomIntegration

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.session_manager import (
    SessionManager,
    SessionState,
    RitualPhase,
)
from events import (
    EVENT_BUS_AVAILABLE,
    RitualContext,
    RitualEventType,
    create_ritual_event,
    phase_completed,
)


class SafetyChecklist:
    """Standard safety checklist items for code review."""

    CHECKLIST_ITEMS = [
        {
            "id": "graceful_degradation",
            "category": "Reliability",
            "item": "Graceful degradation",
            "description": "System handles missing optional dependencies without crashing",
            "severity": "high",
        },
        {
            "id": "input_validation",
            "category": "Security",
            "item": "Input validation",
            "description": "All external inputs are validated and sanitized",
            "severity": "critical",
        },
        {
            "id": "error_messages",
            "category": "UX",
            "item": "Error messages clear",
            "description": "Error messages help users understand and fix issues",
            "severity": "medium",
        },
        {
            "id": "no_hardcoded_secrets",
            "category": "Security",
            "item": "No hardcoded secrets",
            "description": "No API keys, passwords, or tokens in code",
            "severity": "critical",
        },
        {
            "id": "timeout_handling",
            "category": "Reliability",
            "item": "Timeout handling",
            "description": "Long-running operations have appropriate timeouts",
            "severity": "high",
        },
        {
            "id": "resource_cleanup",
            "category": "Reliability",
            "item": "Resource cleanup",
            "description": "Files, connections, and resources are properly closed",
            "severity": "high",
        },
        {
            "id": "null_checks",
            "category": "Reliability",
            "item": "Null/None checks",
            "description": "Potential null values are handled appropriately",
            "severity": "high",
        },
        {
            "id": "logging_appropriate",
            "category": "Operations",
            "item": "Logging appropriate",
            "description": "Errors logged without exposing sensitive data",
            "severity": "medium",
        },
        {
            "id": "rate_limiting",
            "category": "Security",
            "item": "Rate limiting considered",
            "description": "External API calls have rate limiting if applicable",
            "severity": "medium",
        },
        {
            "id": "type_safety",
            "category": "Reliability",
            "item": "Type safety",
            "description": "Type hints used and consistent",
            "severity": "low",
        },
    ]

    @classmethod
    def get_all_items(cls) -> List[Dict[str, Any]]:
        """Get all checklist items."""
        return cls.CHECKLIST_ITEMS.copy()

    @classmethod
    def get_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """Get checklist items by category."""
        return [
            item for item in cls.CHECKLIST_ITEMS
            if item["category"].lower() == category.lower()
        ]

    @classmethod
    def get_by_severity(cls, severity: str) -> List[Dict[str, Any]]:
        """Get checklist items by severity."""
        return [
            item for item in cls.CHECKLIST_ITEMS
            if item["severity"].lower() == severity.lower()
        ]

    @classmethod
    def get_critical_items(cls) -> List[Dict[str, Any]]:
        """Get critical severity items."""
        return cls.get_by_severity("critical")


class CheckPhaseHandler:
    """
    Handles /ritual check command.

    Implements safety-first validation checklist:
    - Manual checklist review
    - Optional automated testing integration
    - Optional security scanning integration
    - Risk assessment based on check results
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[Any] = None,
        skill_tester: Optional[Any] = None,
        security_analyzer: Optional[Any] = None,
        hololoom_integration: Optional["HoloLoomIntegration"] = None,
    ):
        """
        Initialize check phase handler.

        Args:
            session_manager: SessionManager instance (created if not provided)
            event_bus: Optional EventBus for emitting events
            skill_tester: Optional skill_tester for automated testing
            security_analyzer: Optional skill_security_analyzer for scanning
            hololoom_integration: Optional HoloLoomIntegration for safety checks
        """
        self.session_manager = session_manager or SessionManager()
        self.event_bus = event_bus
        self.skill_tester = skill_tester
        self.security_analyzer = security_analyzer
        self.hololoom = hololoom_integration

    async def execute(
        self,
        check_results: Optional[Dict[str, bool]] = None,
        notes: Optional[Dict[str, str]] = None,
        run_automated: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the check phase.

        Args:
            check_results: Results of manual checks {check_id: passed}
            notes: Notes for each check {check_id: note}
            run_automated: Whether to run automated checks if available

        Returns:
            Structured result with check status and next steps
        """
        start_time = datetime.now()

        # Check for active session
        if not self.session_manager.has_active_session():
            return self._error_response(
                "No active session. Use /ritual open to start a session."
            )

        # Get current session
        session = self.session_manager.current_session

        # Validate phase - should be in BUILDING or CHECKING
        if session.phase not in [RitualPhase.BUILDING, RitualPhase.CHECKING]:
            if session.phase == RitualPhase.OPEN:
                return self._error_response(
                    "No builds completed yet. Use /ritual build first."
                )
            elif session.phase == RitualPhase.DESIGNING:
                return self._error_response(
                    "Still in design phase. Use /ritual build to start building."
                )
            else:
                return self._error_response(
                    f"Cannot check from phase '{session.phase.value}'. "
                    f"Use /ritual status to check current phase."
                )

        # Transition to checking phase if not already
        if session.phase == RitualPhase.BUILDING:
            self.session_manager.update_phase(RitualPhase.CHECKING)
            session = self.session_manager.current_session

        # If no check results provided, show checklist
        if check_results is None:
            return await self._show_checklist(
                session=session,
                run_automated=run_automated,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        # Process check results
        return await self._process_check_results(
            session=session,
            check_results=check_results,
            notes=notes or {},
            run_automated=run_automated,
            start_time=start_time,
        )

    async def _show_checklist(
        self,
        session: SessionState,
        run_automated: bool,
        execution_time_ms: float,
    ) -> Dict[str, Any]:
        """Show the safety checklist for review."""
        checklist_items = SafetyChecklist.get_all_items()

        # Categorize items
        categories = {}
        for item in checklist_items:
            cat = item["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "id": item["id"],
                "item": item["item"],
                "description": item["description"],
                "severity": item["severity"],
                "status": "pending",
            })

        # Check existing check results from session
        existing_checks = {
            c["check_id"]: c
            for c in session.checks
        }

        # Update status for existing checks
        for cat_items in categories.values():
            for item in cat_items:
                if item["id"] in existing_checks:
                    check = existing_checks[item["id"]]
                    item["status"] = "passed" if check.get("passed") else "failed"

        # Run automated checks if requested
        automated_results = {}
        if run_automated:
            automated_results = await self._run_automated_checks(session)

        # Count summary
        all_items = [item for items in categories.values() for item in items]
        passed = sum(1 for item in all_items if item["status"] == "passed")
        failed = sum(1 for item in all_items if item["status"] == "failed")
        pending = sum(1 for item in all_items if item["status"] == "pending")

        # Critical items
        critical_items = [
            item for item in all_items
            if item["severity"] == "critical"
        ]
        critical_passed = sum(1 for item in critical_items if item["status"] == "passed")

        return {
            "phase": "check",
            "session_id": session.session_id,
            "status": "checklist",
            "message": "Safety checklist for review",
            "next_steps": [
                "Review each checklist item",
                "Use /ritual check with results to submit",
                "All critical items must pass to proceed",
            ],
            "data": {
                "task": session.task,
                "checklist_by_category": categories,
                "summary": {
                    "total": len(all_items),
                    "passed": passed,
                    "failed": failed,
                    "pending": pending,
                    "critical_total": len(critical_items),
                    "critical_passed": critical_passed,
                },
                "automated_available": {
                    "skill_tester": self.skill_tester is not None,
                    "security_analyzer": self.security_analyzer is not None,
                },
                "automated_results": automated_results,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "correlation_id": session.correlation_id,
            },
        }

    async def _run_automated_checks(
        self,
        session: SessionState,
    ) -> Dict[str, Any]:
        """Run automated checks if skill_tester or security_analyzer available."""
        results = {}

        if self.skill_tester:
            try:
                test_result = await self.skill_tester.run_tests()
                results["skill_tester"] = {
                    "status": "completed",
                    "passed": test_result.get("passed", False),
                    "details": test_result,
                }
            except Exception as e:
                results["skill_tester"] = {
                    "status": "error",
                    "error": str(e),
                }

        if self.security_analyzer:
            try:
                scan_result = await self.security_analyzer.scan()
                results["security_analyzer"] = {
                    "status": "completed",
                    "passed": scan_result.get("issues_found", 0) == 0,
                    "details": scan_result,
                }
            except Exception as e:
                results["security_analyzer"] = {
                    "status": "error",
                    "error": str(e),
                }

        return results

    async def _process_check_results(
        self,
        session: SessionState,
        check_results: Dict[str, bool],
        notes: Dict[str, str],
        run_automated: bool,
        start_time: datetime,
    ) -> Dict[str, Any]:
        """Process submitted check results."""
        checklist_items = SafetyChecklist.get_all_items()
        valid_ids = {item["id"] for item in checklist_items}

        # Validate check IDs
        invalid_ids = set(check_results.keys()) - valid_ids
        if invalid_ids:
            return self._error_response(
                f"Invalid check IDs: {', '.join(invalid_ids)}"
            )

        # Create check records
        check_records = []
        for check_id, passed in check_results.items():
            item = next(i for i in checklist_items if i["id"] == check_id)
            check_record = {
                "check_id": check_id,
                "item": item["item"],
                "category": item["category"],
                "severity": item["severity"],
                "passed": passed,
                "note": notes.get(check_id, ""),
                "checked_at": datetime.now().isoformat(),
            }
            check_records.append(check_record)
            self.session_manager.add_check(check_record)

        # Run automated checks if requested
        automated_results = {}
        if run_automated:
            automated_results = await self._run_automated_checks(session)

        # Calculate results
        passed = sum(1 for r in check_records if r["passed"])
        failed = sum(1 for r in check_records if not r["passed"])

        # Check critical items
        critical_checks = [
            r for r in check_records
            if r["severity"] == "critical"
        ]
        critical_failed = [r for r in critical_checks if not r["passed"]]

        # Determine overall status
        all_critical_passed = len(critical_failed) == 0
        overall_passed = failed == 0

        # Emit event if EventBus available
        events_emitted = []
        if self.event_bus:
            event_id = await self._emit_check_event(
                session, passed, failed, all_critical_passed
            )
            if event_id:
                events_emitted.append(event_id)

        # Determine message and next steps
        if overall_passed:
            message = "All checks passed! Ready to close session."
            next_steps = [
                "Use /ritual close to complete the session",
                "Add lessons learned when closing",
            ]
        elif all_critical_passed:
            message = f"Critical checks passed. {failed} non-critical issues found."
            next_steps = [
                "Consider fixing non-critical issues",
                "Use /ritual close if acceptable",
                "Use /ritual check to re-check after fixes",
            ]
        else:
            failed_critical_names = [r["item"] for r in critical_failed]
            message = f"Critical checks failed: {', '.join(failed_critical_names)}"
            next_steps = [
                "Fix critical issues before proceeding",
                "Use /ritual check to re-check after fixes",
                "Critical items must pass to close session",
            ]

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "phase": "check",
            "session_id": session.session_id,
            "status": "success" if overall_passed else (
                "warning" if all_critical_passed else "failed"
            ),
            "message": message,
            "next_steps": next_steps,
            "data": {
                "task": session.task,
                "checks_submitted": len(check_records),
                "passed": passed,
                "failed": failed,
                "all_passed": overall_passed,
                "critical_passed": all_critical_passed,
                "critical_failed": [r["item"] for r in critical_failed],
                "check_records": check_records,
                "automated_results": automated_results,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "events_emitted": events_emitted,
                "correlation_id": session.correlation_id,
            },
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Build error response."""
        return {
            "phase": "check",
            "session_id": None,
            "status": "error",
            "message": message,
            "next_steps": [],
            "data": {},
            "metadata": {},
        }

    async def _emit_check_event(
        self,
        session: SessionState,
        passed: int,
        failed: int,
        critical_passed: bool,
    ) -> Optional[str]:
        """Emit ritual.check.completed event."""
        if not self.event_bus or not EVENT_BUS_AVAILABLE:
            return None

        try:
            # Create RitualContext from session state
            context = RitualContext(
                ritual_id=session.correlation_id or session.session_id,
                feature_name=session.task,
                started_at=session.started_at,
                current_phase="check",
            )

            # Create event using phase_completed factory
            event = phase_completed(context, phase_name="check")

            # Add check-specific payload
            event["payload"].update({
                "session_id": session.session_id,
                "checks_passed": passed,
                "checks_failed": failed,
                "critical_passed": critical_passed,
            })

            # Update topic to be specific
            event["topic"] = "ritual.check.completed"

            # Record event in session
            session.record_event(event.get("event_id", ""))

            # Emit via EventBus
            await self.event_bus.emit(event)
            return event.get("event_id")

        except Exception:
            # Event emission failed, but don't break the flow
            return None


# Convenience function for direct invocation
async def run_safety_check(
    check_results: Optional[Dict[str, bool]] = None,
    notes: Optional[Dict[str, str]] = None,
    run_automated: bool = False,
    event_bus: Optional[Any] = None,
    skill_tester: Optional[Any] = None,
    security_analyzer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run safety checklist.

    Args:
        check_results: Check results {check_id: passed}
        notes: Notes for checks {check_id: note}
        run_automated: Whether to run automated checks
        event_bus: Optional EventBus for events
        skill_tester: Optional skill_tester
        security_analyzer: Optional security analyzer

    Returns:
        Check phase result
    """
    handler = CheckPhaseHandler(
        event_bus=event_bus,
        skill_tester=skill_tester,
        security_analyzer=security_analyzer,
    )
    return await handler.execute(
        check_results=check_results,
        notes=notes,
        run_automated=run_automated,
    )


def format_check_response(result: Dict[str, Any]) -> str:
    """
    Format check response for display.

    Returns formatted string suitable for terminal output.
    """
    if result["status"] == "error":
        return f"Error: {result['message']}"

    lines = [
        "",
        "Safety Check Phase",
        "",
    ]

    # Show message
    status_icon = {
        "success": "[PASS]",
        "warning": "[WARN]",
        "failed": "[FAIL]",
        "checklist": "[INFO]",
    }.get(result["status"], "[?]")
    lines.append(f"{status_icon} {result['message']}")
    lines.append("")

    # Show summary if available
    summary = result["data"].get("summary")
    if summary:
        lines.extend([
            "**Summary**:",
            f"  Total: {summary['total']}",
            f"  Passed: {summary['passed']}",
            f"  Failed: {summary['failed']}",
            f"  Pending: {summary['pending']}",
            f"  Critical: {summary['critical_passed']}/{summary['critical_total']}",
            "",
        ])

    # Show checklist by category if available
    checklist_by_category = result["data"].get("checklist_by_category", {})
    if checklist_by_category:
        lines.append("**Checklist**:")
        for category, items in checklist_by_category.items():
            lines.append(f"\n  {category}:")
            for item in items:
                status_icon = {
                    "passed": "[x]",
                    "failed": "[!]",
                    "pending": "[ ]",
                }.get(item["status"], "[ ]")
                severity_tag = f"({item['severity']})" if item["severity"] in ["critical", "high"] else ""
                lines.append(f"    {status_icon} {item['item']} {severity_tag}")
        lines.append("")

    # Show check records if submitted
    check_records = result["data"].get("check_records", [])
    if check_records and result["status"] != "checklist":
        lines.append("**Results**:")
        for record in check_records:
            icon = "[x]" if record["passed"] else "[!]"
            lines.append(f"  {icon} {record['item']}")
            if record.get("note"):
                lines.append(f"      Note: {record['note']}")
        lines.append("")

    # Show critical failures prominently
    critical_failed = result["data"].get("critical_failed", [])
    if critical_failed:
        lines.extend([
            "**CRITICAL FAILURES**:",
        ])
        for item in critical_failed:
            lines.append(f"  [!] {item}")
        lines.append("")

    # Automated results
    automated = result["data"].get("automated_results", {})
    if automated:
        lines.append("**Automated Checks**:")
        for name, res in automated.items():
            status = res.get("status", "unknown")
            passed = res.get("passed", False)
            icon = "[x]" if passed else "[!]" if status == "completed" else "[?]"
            lines.append(f"  {icon} {name}: {status}")
        lines.append("")

    # Next steps
    lines.append("**Next Steps**:")
    for step in result["next_steps"]:
        lines.append(f"  - {step}")

    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    async def demo():
        # First open a session and complete builds
        from open import open_ritual_session
        from build import start_build_wave

        await open_ritual_session(
            task="Demo check task",
            complexity="fast",
            success_criteria=["Security checks pass"],
        )

        # Complete all waves
        await start_build_wave(wave=1, wave_status="pass")
        await start_build_wave(wave=2, wave_status="pass")
        await start_build_wave(wave=3, wave_status="pass")

        # Show checklist
        result = await run_safety_check()
        print(format_check_response(result))
        print("\n" + "=" * 50 + "\n")

        # Submit some check results
        result = await run_safety_check(
            check_results={
                "graceful_degradation": True,
                "input_validation": True,
                "error_messages": True,
                "no_hardcoded_secrets": True,
                "timeout_handling": True,
                "resource_cleanup": True,
                "null_checks": True,
                "logging_appropriate": True,
                "rate_limiting": False,  # Non-critical fail
                "type_safety": True,
            },
            notes={
                "rate_limiting": "Not applicable for this component",
            }
        )
        print(format_check_response(result))

    asyncio.run(demo())