#!/usr/bin/env python3
"""
Red Team Matrix Bot Handlers
==============================
ChatOps command handlers for HoloLoom red teaming and adversarial testing.

Commands:
- !redteam attack <strategy> - Execute single attack with specified strategy
- !redteam cycle <n> - Run attack cycle with n iterations
- !redteam report - Generate vulnerability report from discovered issues
- !redteam status - Show CARTS system status and current state
- !redteam vulnerabilities - List all discovered vulnerabilities
- !redteam bandit - Show Thompson Sampling statistics and rewards
- !redteam learn - Trigger explicit learning from recent attacks
- !redteam help - Show detailed command help

Features:
- Adversarial testing with multiple attack strategies
- Thompson Sampling bandit learning (automatic strategy selection)
- Complete audit trail for red team operations
- Safety gating with ActionCategory.ADVERSARIAL_TEST
- Vulnerability discovery and reporting
- Real-time status monitoring
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    pass

# Import HoloLoom components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Red team components
try:
    from HoloLoom.redteam.orchestrator import RedTeamOrchestrator, RedTeamConfig
    from HoloLoom.redteam.executor import AttackExecutor, AttackResult, AttackStrategy
    from HoloLoom.redteam.bandit import RedTeamBandit, BanditStatistics
    REDTEAM_AVAILABLE = True
except ImportError:
    REDTEAM_AVAILABLE = False
    RedTeamOrchestrator = None
    RedTeamConfig = None
    AttackExecutor = None
    AttackResult = None
    AttackStrategy = None
    RedTeamBandit = None
    BanditStatistics = None

# Handler registry
from HoloLoom.chatops.handlers.handler_registry import (
    HandlerRegistry, HandlerCategory, chatops_handler
)

# Alignment framework imports (graceful degradation if unavailable)
try:
    from HoloLoom.alignment import create_guardrails, create_audit_trail
    from HoloLoom.alignment.safety_guardrails import ActionRequest, ActionCategory
    from HoloLoom.alignment.audit_trail import DecisionType, OutcomeType
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
    create_guardrails = None
    create_audit_trail = None

logger = logging.getLogger(__name__)


@dataclass
class AttackStats:
    """Statistics for a single attack."""
    strategy: str
    success: bool
    confidence: float
    reward: float
    vulnerability_found: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RedTeamMatrixHandlers:
    """
    Handler class for Matrix bot commands that use HoloLoom red teaming.

    Integrates Matrix chatops with:
    - Red team orchestrator (adversarial testing)
    - Attack executor (multiple strategies)
    - Thompson Sampling bandit (strategy learning)
    - Vulnerability discovery and reporting
    - Audit trail for red team operations
    """

    def __init__(self, bot, config: Optional[RedTeamConfig] = None, enable_alignment: bool = True):
        """
        Initialize handlers with HoloLoom red team orchestrator.

        Args:
            bot: MatrixBot instance
            config: Optional RedTeamConfig for customization
            enable_alignment: Enable SafetyGuardrails and AuditTrail integration
        """
        self.bot = bot

        if not REDTEAM_AVAILABLE:
            logger.warning("Red team components not available - handlers disabled")
            self.orchestrator = None
            self.guardrails = None
            self.audit = None
            return

        # Initialize red team orchestrator
        logger.info("Initializing HoloLoom Red Team Orchestrator...")

        # Use provided config or create default
        if config is None:
            config = RedTeamConfig()

        self.orchestrator = RedTeamOrchestrator(config)

        # Initialize alignment framework (SafetyGuardrails + AuditTrail)
        if enable_alignment and ALIGNMENT_AVAILABLE:
            self.guardrails = create_guardrails(enable_adversarial_detection=True)
            self.audit = create_audit_trail()
            logger.info("Alignment framework enabled (SafetyGuardrails + AuditTrail)")
        else:
            self.guardrails = None
            self.audit = None
            if enable_alignment and not ALIGNMENT_AVAILABLE:
                logger.warning("Alignment requested but not available - running without safety gating")

        # Track attack history and vulnerabilities
        self.attack_history: List[AttackStats] = []
        self.vulnerabilities: List[Dict[str, Any]] = []
        self.cycle_count = 0

        logger.info("Red team handlers initialized with Thompson Sampling bandit learning")

    # ========================================================================
    # Safety Gating
    # ========================================================================

    async def _gate_action(self, room, event, action: str, args: str) -> bool:
        """
        Gate action through SafetyGuardrails. Returns True if allowed.

        Red team actions are gated with ActionCategory.ADVERSARIAL_TEST
        for explicit tracking and oversight.

        Args:
            room: Matrix room
            event: Matrix event
            action: Action name (e.g., "attack", "cycle")
            args: User input text

        Returns:
            True if action is allowed, False if blocked
        """
        if not self.guardrails:
            return True  # No guardrails = allow all

        try:
            # Create action request with ADVERSARIAL_TEST category
            request = ActionRequest(
                action=action,
                category=ActionCategory.ADVERSARIAL_TEST,
                context={"args": args, "room": room.room_id, "red_team": True},
                user_id=event.sender
            )

            # Evaluate through guardrails
            decision = self.guardrails.evaluate(request, text_input=args)

            # Log to audit trail
            if self.audit:
                self.audit.log_decision(
                    decision_type=DecisionType.SAFETY_GATE,
                    outcome=OutcomeType.APPROVED if decision.allowed else OutcomeType.REJECTED,
                    reason=decision.reason,
                    query_text=args,
                    risk_level=decision.risk_level.value if hasattr(decision.risk_level, 'value') else str(decision.risk_level),
                    metadata={
                        "action": action,
                        "user": event.sender,
                        "room": room.room_id,
                        "red_team": True
                    }
                )

            if not decision.allowed:
                await self.bot.send_message(
                    room.room_id,
                    f"⚠️ **Red Team Action Blocked** ({decision.risk_level.value if hasattr(decision.risk_level, 'value') else decision.risk_level}): {decision.reason}",
                    markdown=True
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Safety gating error: {e}")
            # Fail open with warning (allow action but log)
            return True

    # ========================================================================
    # Core Commands
    # ========================================================================

    @chatops_handler(
        command="redteam",
        description="Red team and adversarial testing for HoloLoom",
        usage="!redteam <attack|cycle|report|status|vulnerabilities|bandit|learn|help> [args]",
        category=HandlerCategory.ADMIN,
        aliases=["rt", "adversarial"]
    )
    async def handle_redteam(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Main red team command dispatcher.

        Subcommands:
            attack <strategy> - Execute single attack
            cycle <n> - Run n attack iterations
            report - Generate vulnerability report
            status - Show CARTS system status
            vulnerabilities - List discovered vulnerabilities
            bandit - Show Thompson Sampling statistics
            learn - Trigger learning from recent attacks
            help - Show detailed help
        """
        if not self.orchestrator:
            await self.bot.send_message(
                room.room_id,
                "❌ **Red team components not available**. Check that HoloLoom.redteam is properly installed.",
                markdown=True
            )
            return

        if not args:
            await self._show_help(room)
            return

        # Parse subcommand
        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower()
        param = parts[1] if len(parts) > 1 else ""

        # Dispatch to subcommand handlers
        if subcommand == "attack":
            await self.handle_attack(room, event, param)
        elif subcommand == "cycle":
            await self.handle_cycle(room, event, param)
        elif subcommand == "report":
            await self.handle_report(room, event, param)
        elif subcommand == "status":
            await self.handle_status(room, event, param)
        elif subcommand == "vulnerabilities":
            await self.handle_vulnerabilities(room, event, param)
        elif subcommand == "bandit":
            await self.handle_bandit(room, event, param)
        elif subcommand == "learn":
            await self.handle_learn(room, event, param)
        elif subcommand == "help":
            await self._show_help(room)
        else:
            await self.bot.send_message(
                room.room_id,
                f"❓ **Unknown subcommand**: `{subcommand}`\n\nUse `!redteam help` for available commands.",
                markdown=True
            )

    async def handle_attack(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Execute single attack with specified strategy.

        Usage: !redteam attack <strategy>
        """
        # Safety gate check
        if not await self._gate_action(room, event, "attack", args):
            return

        if not args:
            await self.bot.send_message(
                room.room_id,
                "**Usage**: `!redteam attack <strategy>`\n\n"
                "Available strategies: prompt_injection, jailbreak, adversarial_input, "
                "resource_exhaustion, model_confusion, factual_probing",
                markdown=True
            )
            return

        strategy_name = args.strip()

        try:
            # Show typing indicator
            await self.bot.send_typing(room.room_id, typing=True)

            # Send processing message
            await self.bot.send_message(
                room.room_id,
                f"🎯 **Executing attack**: `{strategy_name}`...",
                markdown=True
            )

            # Execute attack
            start_time = datetime.now()
            result = await self.orchestrator.execute_attack(strategy_name)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Record in history
            attack_stat = AttackStats(
                strategy=strategy_name,
                success=result.success,
                confidence=result.confidence,
                reward=result.reward,
                vulnerability_found=result.vulnerability_category if result.success else None,
                duration_ms=duration_ms,
                timestamp=datetime.now().isoformat()
            )
            self.attack_history.append(attack_stat)

            # Track vulnerability if found
            if result.success and result.vulnerability_category:
                self.vulnerabilities.append({
                    "strategy": strategy_name,
                    "category": result.vulnerability_category,
                    "description": result.payload,
                    "severity": result.severity,
                    "timestamp": datetime.now().isoformat(),
                    "user": event.sender
                })

            # Log to audit trail
            if self.audit:
                self.audit.log_decision(
                    decision_type=DecisionType.ADVERSARIAL_TEST,
                    outcome=OutcomeType.APPROVED if result.success else OutcomeType.REJECTED,
                    reason=f"Attack {strategy_name}: {'found vulnerability' if result.success else 'no vulnerabilities found'}",
                    metadata={
                        "strategy": strategy_name,
                        "success": result.success,
                        "confidence": result.confidence,
                        "reward": result.reward,
                        "vulnerability": result.vulnerability_category,
                        "duration_ms": duration_ms
                    }
                )

            # Format response
            success_icon = "✅" if result.success else "❌"
            response = f"""
{success_icon} **Attack Result**

• **Strategy**: `{strategy_name}`
• **Status**: {"Vulnerability Found" if result.success else "No Vulnerabilities"}
• **Confidence**: {result.confidence:.1%}
• **Reward**: {result.reward:.2f}
• **Duration**: {duration_ms:.0f}ms
"""
            if result.success:
                response += f"• **Category**: `{result.vulnerability_category}`\n"
                response += f"• **Severity**: {result.severity}\n"

            response += f"\n**Payload**: `{result.payload[:100]}`..."

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Attack error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error**: {str(e)}"
            )
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    async def handle_cycle(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Run attack cycle with n iterations.

        Usage: !redteam cycle <n>
        """
        # Safety gate check
        if not await self._gate_action(room, event, "cycle", args):
            return

        if not args or not args.isdigit():
            await self.bot.send_message(
                room.room_id,
                "**Usage**: `!redteam cycle <n>`\n\nExample: `!redteam cycle 5`",
                markdown=True
            )
            return

        n_iterations = int(args)
        if n_iterations <= 0:
            await self.bot.send_message(room.room_id, "❌ **Error**: n must be positive integer")
            return

        try:
            # Show typing indicator
            await self.bot.send_typing(room.room_id, typing=True)

            # Send processing message
            await self.bot.send_message(
                room.room_id,
                f"🔄 **Starting attack cycle** with {n_iterations} iterations...",
                markdown=True
            )

            # Run cycle
            start_time = datetime.now()
            results = []

            for i in range(n_iterations):
                result = await self.orchestrator.execute_next_attack()
                results.append(result)

                # Record in history
                strategy = result.strategy if hasattr(result, 'strategy') else f"iteration_{i}"
                attack_stat = AttackStats(
                    strategy=strategy,
                    success=result.success,
                    confidence=result.confidence,
                    reward=result.reward,
                    vulnerability_found=result.vulnerability_category if result.success else None,
                    duration_ms=0.0,
                    timestamp=datetime.now().isoformat()
                )
                self.attack_history.append(attack_stat)

                # Track vulnerability if found
                if result.success and result.vulnerability_category:
                    self.vulnerabilities.append({
                        "strategy": strategy,
                        "category": result.vulnerability_category,
                        "description": result.payload,
                        "severity": result.severity,
                        "timestamp": datetime.now().isoformat(),
                        "user": event.sender
                    })

                # Small delay between iterations
                await asyncio.sleep(0.1)

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.cycle_count += 1

            # Calculate statistics
            successful = sum(1 for r in results if r.success)
            avg_confidence = sum(r.confidence for r in results) / len(results)
            avg_reward = sum(r.reward for r in results) / len(results)

            # Log to audit trail
            if self.audit:
                self.audit.log_decision(
                    decision_type=DecisionType.ADVERSARIAL_TEST,
                    outcome=OutcomeType.APPROVED,
                    reason=f"Completed attack cycle {self.cycle_count}: {successful} vulnerabilities found",
                    metadata={
                        "cycle": self.cycle_count,
                        "iterations": n_iterations,
                        "successful": successful,
                        "avg_confidence": avg_confidence,
                        "avg_reward": avg_reward,
                        "total_vulnerabilities": len(self.vulnerabilities)
                    }
                )

            # Format response
            response = f"""
✅ **Attack Cycle Complete** (Cycle #{self.cycle_count})

• **Iterations**: {n_iterations}
• **Successful**: {successful}/{n_iterations} ({successful/n_iterations:.1%})
• **Avg Confidence**: {avg_confidence:.1%}
• **Avg Reward**: {avg_reward:.2f}
• **Duration**: {duration_ms:.0f}ms
• **Total Vulnerabilities Found**: {len(self.vulnerabilities)}

💡 Use `!redteam report` to see full vulnerability report
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error**: {str(e)}"
            )
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    async def handle_report(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Generate vulnerability report from discovered issues.

        Usage: !redteam report
        """
        # Safety gate check
        if not await self._gate_action(room, event, "report", ""):
            return

        try:
            if not self.vulnerabilities:
                await self.bot.send_message(
                    room.room_id,
                    "📋 **No vulnerabilities found yet**. Run attacks with `!redteam attack` or `!redteam cycle`.",
                    markdown=True
                )
                return

            # Group vulnerabilities by category
            by_category: Dict[str, List] = {}
            for vuln in self.vulnerabilities:
                cat = vuln.get("category", "unknown")
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(vuln)

            # Build report
            response = """
📊 **Vulnerability Report**

"""
            response += f"**Total Vulnerabilities**: {len(self.vulnerabilities)}\n"
            response += f"**Categories**: {len(by_category)}\n"
            response += f"**Cycles Run**: {self.cycle_count}\n"
            response += f"**Generated**: {datetime.now().isoformat()}\n\n"

            response += "**Breakdown by Category**:\n\n"

            for category, vulns in sorted(by_category.items()):
                high_severity = sum(1 for v in vulns if v.get("severity") == "HIGH")
                medium_severity = sum(1 for v in vulns if v.get("severity") == "MEDIUM")
                low_severity = sum(1 for v in vulns if v.get("severity") == "LOW")

                response += f"• **{category}** ({len(vulns)} total)\n"
                response += f"  - HIGH: {high_severity} | MEDIUM: {medium_severity} | LOW: {low_severity}\n"

            # Show sample vulnerabilities
            response += "\n**Recent Vulnerabilities**:\n\n"
            for vuln in self.vulnerabilities[-5:]:
                response += f"• [{vuln.get('severity', 'UNKNOWN')}] `{vuln.get('category', 'unknown')}`\n"
                response += f"  Strategy: {vuln.get('strategy', 'unknown')}\n"
                response += f"  Time: {vuln.get('timestamp', 'unknown')[:10]}\n"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Report error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error**: {str(e)}"
            )

    async def handle_status(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show CARTS system status and current state.

        Usage: !redteam status
        """
        # Safety gate check
        if not await self._gate_action(room, event, "status", ""):
            return

        try:
            # Get current status from orchestrator
            status = self.orchestrator.get_status() if hasattr(self.orchestrator, 'get_status') else None

            # Calculate statistics from history
            total_attacks = len(self.attack_history)
            successful_attacks = sum(1 for a in self.attack_history if a.success)
            avg_confidence = sum(a.confidence for a in self.attack_history) / total_attacks if total_attacks > 0 else 0
            avg_reward = sum(a.reward for a in self.attack_history) / total_attacks if total_attacks > 0 else 0

            # Get bandit stats if available
            bandit_stats = self.orchestrator.bandit.get_statistics() if hasattr(self.orchestrator, 'bandit') else None

            response = """
⚙️ **CARTS System Status**

**Red Team Metrics:**
"""
            response += f"• **Total Attacks**: {total_attacks}\n"
            response += f"• **Successful**: {successful_attacks}/{total_attacks} ({successful_attacks/total_attacks:.1%} if total_attacks > 0 else '0%')\n"
            response += f"• **Avg Confidence**: {avg_confidence:.1%}\n"
            response += f"• **Avg Reward**: {avg_reward:.2f}\n"
            response += f"• **Cycles Run**: {self.cycle_count}\n"
            response += f"• **Vulnerabilities Found**: {len(self.vulnerabilities)}\n"

            if status:
                response += f"\n**Orchestrator Status**:\n"
                response += f"• **State**: {getattr(status, 'state', 'unknown')}\n"
                response += f"• **Current Attack**: {getattr(status, 'current_attack', 'none')}\n"
                response += f"• **Uptime**: {getattr(status, 'uptime_seconds', 0):.0f}s\n"

            if bandit_stats:
                response += f"\n**Bandit Learning**:\n"
                response += f"• **Strategies Evaluated**: {len(bandit_stats.alpha)}\n"
                response += f"• **Best Strategy**: {bandit_stats.best_strategy}\n"
                response += f"• **Best Reward**: {bandit_stats.best_reward:.2f}\n"

            response += f"\n**Last Updated**: {datetime.now().isoformat()}"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Status error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error**: {str(e)}"
            )

    async def handle_vulnerabilities(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        List all discovered vulnerabilities with details.

        Usage: !redteam vulnerabilities [category]
        """
        # Safety gate check
        if not await self._gate_action(room, event, "vulnerabilities", ""):
            return

        try:
            if not self.vulnerabilities:
                await self.bot.send_message(
                    room.room_id,
                    "📋 **No vulnerabilities found yet**.",
                    markdown=True
                )
                return

            # Filter by category if provided
            vulns = self.vulnerabilities
            if args:
                category = args.strip().lower()
                vulns = [v for v in vulns if v.get("category", "").lower() == category]

                if not vulns:
                    await self.bot.send_message(
                        room.room_id,
                        f"❌ **No vulnerabilities in category**: `{category}`",
                        markdown=True
                    )
                    return

            # Build response
            response = f"**🔓 Discovered Vulnerabilities** ({len(vulns)} total)\n\n"

            for i, vuln in enumerate(vulns[-10:], 1):  # Show last 10
                severity_icon = "🔴" if vuln.get("severity") == "HIGH" else "🟠" if vuln.get("severity") == "MEDIUM" else "🟡"
                response += f"{i}. {severity_icon} `{vuln.get('category', 'unknown')}`\n"
                response += f"   Strategy: {vuln.get('strategy', 'unknown')}\n"
                response += f"   Found: {vuln.get('timestamp', 'unknown')[:10]}\n\n"

            if len(vulns) > 10:
                response += f"\n📌 **Showing last 10 of {len(vulns)} vulnerabilities**"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Vulnerabilities error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error**: {str(e)}"
            )

    async def handle_bandit(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show Thompson Sampling statistics and strategy rewards.

        Usage: !redteam bandit
        """
        # Safety gate check
        if not await self._gate_action(room, event, "bandit", ""):
            return

        try:
            if not hasattr(self.orchestrator, 'bandit'):
                await self.bot.send_message(
                    room.room_id,
                    "⚠️ **Bandit learning not enabled** in this orchestrator.",
                    markdown=True
                )
                return

            # Get bandit statistics
            stats = self.orchestrator.bandit.get_statistics()

            response = """
📊 **Thompson Sampling Bandit Statistics**

"""
            response += f"• **Strategies Evaluated**: {len(stats.alpha)}\n"
            response += f"• **Total Pulls**: {stats.total_pulls}\n"
            response += f"• **Best Strategy**: `{stats.best_strategy}`\n"
            response += f"• **Best Reward**: {stats.best_reward:.2f}\n"
            response += f"• **Exploitation Rate**: {stats.exploitation_rate:.1%}\n\n"

            # Show per-strategy stats
            response += "**Per-Strategy Rewards**:\n\n"

            for strategy, (alpha, beta, reward) in sorted(
                zip(stats.alpha.keys(), stats.alpha.values()),
                key=lambda x: stats.rewards.get(x[0], 0),
                reverse=True
            ):
                trials = alpha.get(strategy, 0) + beta.get(strategy, 0) if isinstance(alpha, dict) and isinstance(beta, dict) else 0
                expected_reward = stats.rewards.get(strategy, reward)
                response += f"• **{strategy}**: {expected_reward:.2f} (trials: {trials})\n"

            response += f"\n**Last Updated**: {datetime.now().isoformat()}"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Bandit error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error**: {str(e)}"
            )

    async def handle_learn(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Trigger explicit learning from recent attacks.

        Usage: !redteam learn
        """
        # Safety gate check
        if not await self._gate_action(room, event, "learn", ""):
            return

        try:
            # Show typing indicator
            await self.bot.send_typing(room.room_id, typing=True)

            # Trigger learning
            await self.bot.send_message(
                room.room_id,
                "🧠 **Triggering learning from recent attacks...**",
                markdown=True
            )

            if hasattr(self.orchestrator, 'trigger_learning'):
                learning_result = await self.orchestrator.trigger_learning()
            else:
                learning_result = {"status": "no learning trigger available"}

            # Log to audit trail
            if self.audit:
                self.audit.log_decision(
                    decision_type=DecisionType.LEARNING,
                    outcome=OutcomeType.APPROVED,
                    reason="Explicit learning triggered from red team session",
                    metadata={
                        "attacks_analyzed": len(self.attack_history),
                        "vulnerabilities_found": len(self.vulnerabilities),
                        "learning_result": str(learning_result)
                    }
                )

            response = f"""
✅ **Learning Complete**

• **Attacks Analyzed**: {len(self.attack_history)}
• **Vulnerabilities Learned**: {len(self.vulnerabilities)}
• **Status**: {learning_result.get('status', 'complete')}

Thompson Sampling bandit has been updated with latest attack results.
Strategy selection will adapt based on historical success rates.
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Learn error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error**: {str(e)}"
            )
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    # ========================================================================
    # Help
    # ========================================================================

    async def _show_help(self, room: MatrixRoom):
        """Show detailed help for red team commands."""
        help_text = """
🎯 **HoloLoom Red Team Commands**

**Main Command**: `!redteam <subcommand> [args]`

**Subcommands**:

1. **attack <strategy>** - Execute single attack
   • `!redteam attack prompt_injection`
   • Strategies: prompt_injection, jailbreak, adversarial_input, resource_exhaustion, model_confusion, factual_probing

2. **cycle <n>** - Run n attack iterations with automatic strategy selection
   • `!redteam cycle 5` - Run 5 attacks
   • Uses Thompson Sampling to select best-performing strategies

3. **report** - Generate vulnerability report from discovered issues
   • `!redteam report`
   • Shows vulnerabilities grouped by category and severity

4. **status** - Show CARTS system status and metrics
   • `!redteam status`
   • Shows success rate, confidence, rewards, and bandit statistics

5. **vulnerabilities [category]** - List discovered vulnerabilities
   • `!redteam vulnerabilities`
   • `!redteam vulnerabilities prompt_injection` - Filter by category

6. **bandit** - Show Thompson Sampling statistics and strategy rewards
   • `!redteam bandit`
   • Shows per-strategy rewards and exploitation rates

7. **learn** - Trigger explicit learning from recent attacks
   • `!redteam learn`
   • Updates Thompson Sampling priors based on attack history

8. **help** - Show this help message
   • `!redteam help`

**Features**:
• ✅ Safety-gated with ActionCategory.ADVERSARIAL_TEST
• ✅ Thompson Sampling bandit for adaptive strategy selection
• ✅ Complete audit trail for all operations
• ✅ Vulnerability discovery and categorization
• ✅ Real-time status monitoring
• ✅ Graceful error handling

**Aliases**: `!rt`, `!adversarial`
"""

        await self.bot.send_message(room.room_id, help_text, markdown=True)

    # ========================================================================
    # Utilities
    # ========================================================================

    def register_all(self):
        """Register all handlers with the bot using the registry."""
        # Create registry and register this instance
        self._registry = HandlerRegistry()
        count = self._registry.register_instance(self)

        # Also register with bot's handler dict for backward compatibility
        for command in self._registry.get_all_commands():
            spec = self._registry.get_handler(command)
            if spec:
                self.bot.register_handler(command, spec.handler)
                # Register aliases too
                for alias in spec.aliases:
                    self.bot.register_handler(alias, spec.handler)

        logger.info(f"Registered {count} red team handlers via registry (with aliases)")

    def get_registry(self) -> HandlerRegistry:
        """Get the handler registry for introspection."""
        if not hasattr(self, '_registry'):
            self._registry = HandlerRegistry()
            self._registry.register_instance(self)
        return self._registry

    def generate_help_text(self) -> str:
        """Generate help text from registry metadata."""
        return self.get_registry().generate_help(prefix="!")

    async def shutdown(self):
        """Clean shutdown with proper lifecycle management."""
        logger.info("Shutting down red team orchestrator...")

        if self.orchestrator and hasattr(self.orchestrator, 'close'):
            await self.orchestrator.close()

        logger.info("Red team handlers shut down successfully")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
HoloLoom Red Team Matrix Bot Handlers
=====================================

This module provides command handlers that integrate
Matrix chatops with HoloLoom red teaming systems.

Commands available:
- !redteam attack <strategy> - Execute single attack
- !redteam cycle <n> - Run attack cycle with n iterations
- !redteam report - Generate vulnerability report
- !redteam status - Show CARTS system status
- !redteam vulnerabilities - List discovered vulnerabilities
- !redteam bandit - Show Thompson Sampling statistics
- !redteam learn - Trigger learning from recent attacks
- !redteam help - Show detailed help

Features:
- Adversarial testing with multiple attack strategies
- Thompson Sampling bandit for adaptive strategy selection
- Complete vulnerability discovery and reporting
- Safety gating with ActionCategory.ADVERSARIAL_TEST
- Real-time status monitoring
- Automatic learning and strategy optimization

To use:
    from HoloLoom.chatops.core.matrix_bot import MatrixBot
    from HoloLoom.chatops.handlers.redteam_handlers import RedTeamMatrixHandlers

    bot = MatrixBot(config)
    handlers = RedTeamMatrixHandlers(bot)
    handlers.register_all()
    await bot.start()
""")
