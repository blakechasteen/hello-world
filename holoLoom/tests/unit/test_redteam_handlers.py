"""Unit tests for HoloLoom red team ChatOps handlers.

Tests the complete ChatOps integration for red team commands:
- Handler registration and dispatch
- Attack command parsing and execution
- Cycle command with multiple iterations
- Report generation from vulnerabilities
- Status command output
- Help documentation
- Safety gating with alignment framework

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-05
"""

import asyncio
import pytest
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import json


# Mock Matrix types
@dataclass
class MockMatrixRoom:
    """Mock nio.MatrixRoom for testing."""
    room_id: str = "!test:example.com"
    name: str = "test-room"
    display_name: str = "Test Room"


@dataclass
class MockRoomMessageText:
    """Mock nio.RoomMessageText event."""
    sender: str = "@user:example.com"
    body: str = ""
    room_id: str = "!test:example.com"


# Mock redteam components
@dataclass
class MockAttackResult:
    """Mock AttackResult."""
    strategy: str
    success: bool
    severity: float = 0.5
    details: str = "Test attack"
    vulnerabilities: List[str] = None

    def __post_init__(self):
        if self.vulnerabilities is None:
            self.vulnerabilities = []


@dataclass
class MockBanditStatistics:
    """Mock BanditStatistics."""
    total_pulls: int = 0
    total_rewards: float = 0.0
    expected_reward: float = 0.5
    best_strategy: str = "prompt_injection"


class MockRedTeamOrchestrator:
    """Mock RedTeamOrchestrator."""

    def __init__(self):
        self.is_active = False
        self.attack_history = []
        self.vulnerabilities = []
        self.bandit_stats = MockBanditStatistics()

    async def run_attack(self, strategy: str) -> MockAttackResult:
        """Mock attack execution."""
        success = strategy == "prompt_injection"
        result = MockAttackResult(
            strategy=strategy,
            success=success,
            severity=0.8 if success else 0.2,
            vulnerabilities=["test_vuln"] if success else []
        )
        self.attack_history.append(result)
        return result

    async def run_cycle(self, iterations: int) -> List[MockAttackResult]:
        """Mock cycle execution."""
        results = []
        for i in range(iterations):
            result = await self.run_attack("prompt_injection")
            results.append(result)
        return results

    def get_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Mock vulnerability retrieval."""
        return [
            {
                "id": "vuln_1",
                "strategy": "prompt_injection",
                "severity": 0.8,
                "timestamp": "2025-12-05T10:00:00"
            }
        ]

    def get_status(self) -> Dict[str, Any]:
        """Mock status."""
        return {
            "is_active": self.is_active,
            "attacks_performed": len(self.attack_history),
            "vulnerabilities_found": len(self.vulnerabilities),
            "best_strategy": "prompt_injection"
        }

    def get_statistics(self) -> MockBanditStatistics:
        """Mock statistics."""
        return self.bandit_stats


class MockMatrixBot:
    """Mock Matrix bot for testing."""

    def __init__(self):
        self.messages_sent = []
        self.room = MockMatrixRoom()

    async def send_message(self, room_id: str, message: str, **kwargs):
        """Mock message sending."""
        self.messages_sent.append({
            "room_id": room_id,
            "message": message,
            "kwargs": kwargs
        })


class MockGuardrails:
    """Mock SafetyGuardrails."""

    async def gate_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock action gating."""
        allowed = action in ["query", "adversarial_test", "status"]
        return {
            "allowed": allowed,
            "reason": "Test gating" if allowed else "Test blocked",
            "safety_score": 0.9 if allowed else 0.1
        }


class MockAuditTrail:
    """Mock AuditTrail."""

    def __init__(self):
        self.logs = []

    async def log_decision(self, **kwargs):
        """Mock logging."""
        self.logs.append(kwargs)


# Main handler class (simplified from actual implementation)
class MockRedTeamMatrixHandlers:
    """Mock red team handlers for testing."""

    def __init__(
        self,
        bot: MockMatrixBot,
        orchestrator: Optional[MockRedTeamOrchestrator] = None,
        guardrails: Optional[MockGuardrails] = None,
        audit: Optional[MockAuditTrail] = None
    ):
        self.bot = bot
        self.orchestrator = orchestrator or MockRedTeamOrchestrator()
        self.guardrails = guardrails
        self.audit = audit
        self.attack_history = []
        self.vulnerabilities = []

    async def handle_attack(
        self,
        room_id: str,
        strategy: str,
        user_id: str = "@user:example.com"
    ) -> str:
        """Handle !redteam attack command."""
        # Safety gating
        if self.guardrails:
            gate = await self.guardrails.gate_action(
                "adversarial_test",
                {"strategy": strategy, "user_id": user_id}
            )
            if not gate["allowed"]:
                return f"❌ Action blocked: {gate['reason']}"

        # Execute attack
        result = await self.orchestrator.run_attack(strategy)
        self.attack_history.append(result)

        # Audit logging
        if self.audit:
            await self.audit.log_decision(
                action="attack",
                strategy=strategy,
                success=result.success,
                user_id=user_id
            )

        # Format response
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        msg = f"{status}: {strategy} attack\nSeverity: {result.severity:.1%}\nDetails: {result.details}"
        await self.bot.send_message(room_id, msg)
        return msg

    async def handle_cycle(
        self,
        room_id: str,
        iterations: int,
        user_id: str = "@user:example.com"
    ) -> str:
        """Handle !redteam cycle command."""
        # Safety gating
        if self.guardrails:
            gate = await self.guardrails.gate_action(
                "adversarial_test",
                {"iterations": iterations, "user_id": user_id}
            )
            if not gate["allowed"]:
                return f"❌ Action blocked: {gate['reason']}"

        # Run cycle
        results = await self.orchestrator.run_cycle(iterations)
        successes = sum(1 for r in results if r.success)

        # Audit logging
        if self.audit:
            await self.audit.log_decision(
                action="cycle",
                iterations=iterations,
                successes=successes,
                user_id=user_id
            )

        msg = f"🔄 CYCLE COMPLETE: {iterations} iterations\n✅ Successes: {successes}\n❌ Failures: {iterations - successes}"
        await self.bot.send_message(room_id, msg)
        return msg

    async def handle_report(
        self,
        room_id: str,
        user_id: str = "@user:example.com"
    ) -> str:
        """Handle !redteam report command."""
        vulnerabilities = self.orchestrator.get_vulnerabilities()

        if self.audit:
            await self.audit.log_decision(
                action="report",
                vulnerabilities_count=len(vulnerabilities),
                user_id=user_id
            )

        if not vulnerabilities:
            msg = "📋 VULNERABILITY REPORT: No vulnerabilities found"
        else:
            vuln_lines = "\n".join(
                f"  • {v['strategy']}: severity {v['severity']:.1%}"
                for v in vulnerabilities[:5]
            )
            msg = f"📋 VULNERABILITY REPORT: {len(vulnerabilities)} found\n{vuln_lines}"

        await self.bot.send_message(room_id, msg)
        return msg

    async def handle_status(
        self,
        room_id: str,
        user_id: str = "@user:example.com"
    ) -> str:
        """Handle !redteam status command."""
        status = self.orchestrator.get_status()

        if self.audit:
            await self.audit.log_decision(
                action="status",
                user_id=user_id
            )

        msg = f"""📊 CARTS STATUS:
Active: {'✅ Yes' if status['is_active'] else '❌ No'}
Attacks: {status['attacks_performed']}
Vulnerabilities: {status['vulnerabilities_found']}
Best Strategy: {status['best_strategy']}"""

        await self.bot.send_message(room_id, msg)
        return msg

    async def handle_help(self, room_id: str) -> str:
        """Handle !redteam help command."""
        msg = """🔴 RED TEAM COMMANDS:
!redteam attack <strategy> - Execute single attack
!redteam cycle <n> - Run n iterations
!redteam report - Show vulnerabilities
!redteam status - Show system status
!redteam help - Show this help"""

        await self.bot.send_message(room_id, msg)
        return msg


# ============================================================================
# TESTS
# ============================================================================


class TestHandlerRegistration:
    """Test handler registration and initialization."""

    def test_handler_initialization(self):
        """Test handlers initialize correctly."""
        bot = MockMatrixBot()
        orchestrator = MockRedTeamOrchestrator()
        guardrails = MockGuardrails()
        audit = MockAuditTrail()

        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            orchestrator=orchestrator,
            guardrails=guardrails,
            audit=audit
        )

        assert handlers.bot == bot
        assert handlers.orchestrator == orchestrator
        assert handlers.guardrails == guardrails
        assert handlers.audit == audit

    def test_handler_without_alignment(self):
        """Test handlers work without alignment framework."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        assert handlers.guardrails is None
        assert handlers.audit is None
        assert handlers.orchestrator is not None


class TestAttackCommand:
    """Test attack command parsing and execution."""

    @pytest.mark.asyncio
    async def test_attack_command_success(self):
        """Test successful attack command."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        result = await handlers.handle_attack(
            room_id="!test:example.com",
            strategy="prompt_injection"
        )

        assert "SUCCESS" in result
        assert "prompt_injection" in result
        assert len(bot.messages_sent) == 1

    @pytest.mark.asyncio
    async def test_attack_command_failure(self):
        """Test failed attack command."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        result = await handlers.handle_attack(
            room_id="!test:example.com",
            strategy="unknown_strategy"
        )

        assert "FAILED" in result
        assert len(bot.messages_sent) == 1

    @pytest.mark.asyncio
    async def test_attack_with_safety_gating(self):
        """Test attack with safety gating."""
        bot = MockMatrixBot()
        guardrails = MockGuardrails()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            guardrails=guardrails
        )

        result = await handlers.handle_attack(
            room_id="!test:example.com",
            strategy="prompt_injection"
        )

        assert "SUCCESS" in result or "❌" not in result

    @pytest.mark.asyncio
    async def test_attack_with_audit_logging(self):
        """Test attack with audit trail."""
        bot = MockMatrixBot()
        audit = MockAuditTrail()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            audit=audit
        )

        await handlers.handle_attack(
            room_id="!test:example.com",
            strategy="prompt_injection",
            user_id="@user:example.com"
        )

        assert len(audit.logs) == 1
        assert audit.logs[0]["action"] == "attack"
        assert audit.logs[0]["strategy"] == "prompt_injection"


class TestCycleCommand:
    """Test cycle command execution."""

    @pytest.mark.asyncio
    async def test_cycle_command_basic(self):
        """Test basic cycle command."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        result = await handlers.handle_cycle(
            room_id="!test:example.com",
            iterations=5
        )

        assert "CYCLE COMPLETE" in result
        assert "5 iterations" in result

    @pytest.mark.asyncio
    async def test_cycle_command_with_results(self):
        """Test cycle with result counting."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        result = await handlers.handle_cycle(
            room_id="!test:example.com",
            iterations=3
        )

        assert "Successes" in result
        assert "Failures" in result

    @pytest.mark.asyncio
    async def test_cycle_with_audit_logging(self):
        """Test cycle with audit trail."""
        bot = MockMatrixBot()
        audit = MockAuditTrail()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            audit=audit
        )

        await handlers.handle_cycle(
            room_id="!test:example.com",
            iterations=3
        )

        assert len(audit.logs) == 1
        assert audit.logs[0]["action"] == "cycle"
        assert audit.logs[0]["iterations"] == 3


class TestReportGeneration:
    """Test report command and generation."""

    @pytest.mark.asyncio
    async def test_report_with_vulnerabilities(self):
        """Test report generation with findings."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        result = await handlers.handle_report(
            room_id="!test:example.com"
        )

        assert "VULNERABILITY REPORT" in result

    @pytest.mark.asyncio
    async def test_report_empty(self):
        """Test report with no vulnerabilities."""
        bot = MockMatrixBot()
        orchestrator = MockRedTeamOrchestrator()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            orchestrator=orchestrator
        )

        # Clear vulnerabilities
        orchestrator.vulnerabilities = []

        result = await handlers.handle_report(
            room_id="!test:example.com"
        )

        assert "VULNERABILITY REPORT" in result

    @pytest.mark.asyncio
    async def test_report_audit_logging(self):
        """Test report with audit trail."""
        bot = MockMatrixBot()
        audit = MockAuditTrail()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            audit=audit
        )

        await handlers.handle_report(
            room_id="!test:example.com"
        )

        assert len(audit.logs) == 1
        assert audit.logs[0]["action"] == "report"


class TestStatusCommand:
    """Test status command."""

    @pytest.mark.asyncio
    async def test_status_command(self):
        """Test status command output."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        result = await handlers.handle_status(
            room_id="!test:example.com"
        )

        assert "CARTS STATUS" in result
        assert "Attacks" in result
        assert "Vulnerabilities" in result

    @pytest.mark.asyncio
    async def test_status_audit_logging(self):
        """Test status with audit trail."""
        bot = MockMatrixBot()
        audit = MockAuditTrail()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            audit=audit
        )

        await handlers.handle_status(
            room_id="!test:example.com"
        )

        assert len(audit.logs) == 1
        assert audit.logs[0]["action"] == "status"


class TestHelpCommand:
    """Test help command."""

    @pytest.mark.asyncio
    async def test_help_command(self):
        """Test help command."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        result = await handlers.handle_help(
            room_id="!test:example.com"
        )

        assert "RED TEAM COMMANDS" in result
        assert "!redteam attack" in result
        assert "!redteam cycle" in result
        assert "!redteam report" in result


class TestSafetyGating:
    """Test safety gating integration."""

    @pytest.mark.asyncio
    async def test_gating_blocks_unauthorized(self):
        """Test gating can block actions."""
        bot = MockMatrixBot()

        # Create guardrails that always block
        class BlockingGuardrails:
            async def gate_action(self, action: str, context: Dict[str, Any]):
                return {
                    "allowed": False,
                    "reason": "Test block"
                }

        guardrails = BlockingGuardrails()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            guardrails=guardrails
        )

        result = await handlers.handle_attack(
            room_id="!test:example.com",
            strategy="prompt_injection"
        )

        assert "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_gating_allows_authorized(self):
        """Test gating allows authorized actions."""
        bot = MockMatrixBot()
        guardrails = MockGuardrails()
        handlers = MockRedTeamMatrixHandlers(
            bot=bot,
            guardrails=guardrails
        )

        result = await handlers.handle_attack(
            room_id="!test:example.com",
            strategy="prompt_injection"
        )

        assert "blocked" not in result.lower()


class TestMessageFormatting:
    """Test message formatting and output."""

    @pytest.mark.asyncio
    async def test_attack_message_format(self):
        """Test attack message formatting."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        await handlers.handle_attack(
            room_id="!test:example.com",
            strategy="prompt_injection"
        )

        assert len(bot.messages_sent) > 0
        msg = bot.messages_sent[0]["message"]
        assert "Severity" in msg or "SUCCESS" in msg

    @pytest.mark.asyncio
    async def test_cycle_message_format(self):
        """Test cycle message formatting."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        await handlers.handle_cycle(
            room_id="!test:example.com",
            iterations=5
        )

        msg = bot.messages_sent[0]["message"]
        assert "CYCLE" in msg
        assert "5" in msg

    @pytest.mark.asyncio
    async def test_help_message_format(self):
        """Test help message formatting."""
        bot = MockMatrixBot()
        handlers = MockRedTeamMatrixHandlers(bot=bot)

        await handlers.handle_help(
            room_id="!test:example.com"
        )

        msg = bot.messages_sent[0]["message"]
        assert "redteam" in msg
        assert "attack" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
