"""
Agentic Handlers for Matrix ChatOps
====================================

Matrix command handlers for HoloLoom Agentic Reasoning System.

Commands:
- !research <topic> - Multi-query exploration (RESEARCH mode)
- !verify <claim> - Cross-check verification (VERIFY mode)
- !plan <goal> - Goal decomposition (PLAN_EXECUTE mode)
- !reason <query> - Standard reasoning (DIRECT mode)

Usage:
    from HoloLoom.apps.chatops.handlers.agentic_handlers import register_agentic_handlers

    # In run_chatops.py:
    register_agentic_handlers(bot, orchestrator)

Created: 2025-12-05
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.agentic.core import AgenticOrchestrator, ReasoningMode

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# Agentic imports
try:
    from HoloLoom.agentic.core import AgenticOrchestrator, ReasoningMode
    from HoloLoom.protocols.types import Query
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False
    AgenticOrchestrator = None
    ReasoningMode = None
    Query = None

# Handler registry
try:
    from HoloLoom.apps.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_research(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: 'AgenticOrchestrator'
) -> str:
    """
    Handle !research command - Multi-query exploration.

    Uses RESEARCH mode for open-ended exploration from multiple angles.

    Args:
        room: Matrix room
        event: Message event
        args: Research topic
        orchestrator: AgenticOrchestrator instance

    Returns:
        Research findings with synthesized answer
    """
    if not args.strip():
        return "❌ Usage: !research <topic>\nExample: !research What are the tradeoffs of Thompson Sampling?"

    if not AGENTIC_AVAILABLE:
        return "❌ Agentic reasoning system not available"

    topic = args.strip()

    # Parse optional max_steps (--steps=N)
    max_steps = 5
    if topic.startswith("--steps="):
        parts = topic.split(" ", 1)
        try:
            max_steps = int(parts[0].replace("--steps=", ""))
        except ValueError:
            pass
        topic = parts[1] if len(parts) > 1 else ""

    if not topic:
        return "❌ Usage: !research [--steps=N] <topic>"

    try:
        query = Query(text=topic)
        result = await orchestrator.reason(
            query=query,
            mode=ReasoningMode.RESEARCH,
            max_steps=max_steps
        )

        # Format response
        response = f"🔬 **Research: {topic[:50]}{'...' if len(topic) > 50 else ''}**\n\n"

        # Main response
        if hasattr(result, 'spacetime') and result.spacetime:
            response += f"{result.spacetime.response}\n\n"
        elif hasattr(result, 'response'):
            response += f"{result.response}\n\n"

        # Confidence
        if hasattr(result, 'spacetime') and result.spacetime:
            response += f"**Confidence:** {result.spacetime.confidence:.2f}\n"

        # Steps taken
        if hasattr(result, 'steps_taken') and result.steps_taken:
            response += f"**Queries explored:** {len(result.steps_taken)}\n"
            for i, step in enumerate(result.steps_taken[:5], 1):
                step_text = step.get('query', step.get('text', str(step)))[:80]
                response += f"  {i}. {step_text}\n"
            if len(result.steps_taken) > 5:
                response += f"  _...and {len(result.steps_taken) - 5} more_\n"

        return response

    except Exception as e:
        logger.error(f"Error in research: {e}")
        return f"❌ Research failed: {str(e)}"


async def handle_verify(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: 'AgenticOrchestrator'
) -> str:
    """
    Handle !verify command - Cross-check verification.

    Uses VERIFY mode to check claims for accuracy and consistency.

    Args:
        room: Matrix room
        event: Message event
        args: Claim to verify
        orchestrator: AgenticOrchestrator instance

    Returns:
        Verification result with confidence
    """
    if not args.strip():
        return "❌ Usage: !verify <claim>\nExample: !verify Thompson Sampling always outperforms UCB"

    if not AGENTIC_AVAILABLE:
        return "❌ Agentic reasoning system not available"

    claim = args.strip()

    try:
        query = Query(text=claim)
        result = await orchestrator.reason(
            query=query,
            mode=ReasoningMode.VERIFY,
            confidence_threshold=0.85
        )

        # Format response
        response = f"✅ **Verification: {claim[:50]}{'...' if len(claim) > 50 else ''}**\n\n"

        # Verification result
        if hasattr(result, 'verification') and result.verification:
            verified = result.verification.get('verified', False)
            reason = result.verification.get('reason', 'No reason provided')

            if verified:
                response += f"**Status:** ✓ Verified\n"
            else:
                response += f"**Status:** ✗ Not Verified\n"

            response += f"**Reason:** {reason}\n\n"

        # Main response
        if hasattr(result, 'spacetime') and result.spacetime:
            response += f"**Analysis:**\n{result.spacetime.response}\n\n"
            response += f"**Confidence:** {result.spacetime.confidence:.2f}\n"

        # Sources if available
        if hasattr(result, 'sources') and result.sources:
            response += f"\n**Sources ({len(result.sources)}):**\n"
            for i, src in enumerate(result.sources[:3], 1):
                truncated = str(src)[:100] + "..." if len(str(src)) > 100 else str(src)
                response += f"  {i}. {truncated}\n"

        return response

    except Exception as e:
        logger.error(f"Error in verify: {e}")
        return f"❌ Verification failed: {str(e)}"


async def handle_plan(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: 'AgenticOrchestrator'
) -> str:
    """
    Handle !plan command - Goal decomposition.

    Uses PLAN_EXECUTE mode to break down goals into steps.

    Args:
        room: Matrix room
        event: Message event
        args: Goal to plan
        orchestrator: AgenticOrchestrator instance

    Returns:
        Decomposed plan with steps
    """
    if not args.strip():
        return "❌ Usage: !plan <goal>\nExample: !plan Implement a recommendation system"

    if not AGENTIC_AVAILABLE:
        return "❌ Agentic reasoning system not available"

    goal = args.strip()

    # Parse optional max_steps (--steps=N)
    max_steps = 5
    if goal.startswith("--steps="):
        parts = goal.split(" ", 1)
        try:
            max_steps = int(parts[0].replace("--steps=", ""))
        except ValueError:
            pass
        goal = parts[1] if len(parts) > 1 else ""

    if not goal:
        return "❌ Usage: !plan [--steps=N] <goal>"

    try:
        query = Query(text=goal)
        result = await orchestrator.reason(
            query=query,
            mode=ReasoningMode.PLAN_EXECUTE,
            max_steps=max_steps
        )

        # Format response
        response = f"📋 **Plan: {goal[:50]}{'...' if len(goal) > 50 else ''}**\n\n"

        # Main plan
        if hasattr(result, 'spacetime') and result.spacetime:
            response += f"{result.spacetime.response}\n\n"
            response += f"**Confidence:** {result.spacetime.confidence:.2f}\n"

        # Steps/execution trace
        if hasattr(result, 'steps_taken') and result.steps_taken:
            response += f"\n**Execution steps ({len(result.steps_taken)}):**\n"
            for i, step in enumerate(result.steps_taken, 1):
                step_type = step.get('type', 'step')
                step_desc = step.get('description', step.get('query', str(step)))[:80]
                status = step.get('status', '')
                status_icon = "✓" if status == 'completed' else "→"
                response += f"  {status_icon} {i}. [{step_type}] {step_desc}\n"

        return response

    except Exception as e:
        logger.error(f"Error in plan: {e}")
        return f"❌ Planning failed: {str(e)}"


async def handle_reason(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: 'AgenticOrchestrator'
) -> str:
    """
    Handle !reason command - Standard reasoning (DIRECT mode).

    Single-pass reasoning for quick answers.

    Args:
        room: Matrix room
        event: Message event
        args: Query to reason about
        orchestrator: AgenticOrchestrator instance

    Returns:
        Reasoned response
    """
    if not args.strip():
        return "❌ Usage: !reason <query>\nExample: !reason How does attention work in transformers?"

    if not AGENTIC_AVAILABLE:
        return "❌ Agentic reasoning system not available"

    query_text = args.strip()

    # Parse optional mode override (--mode=X)
    mode = ReasoningMode.DIRECT
    if query_text.startswith("--mode="):
        parts = query_text.split(" ", 1)
        mode_str = parts[0].replace("--mode=", "").upper()
        if hasattr(ReasoningMode, mode_str):
            mode = getattr(ReasoningMode, mode_str)
        query_text = parts[1] if len(parts) > 1 else ""

    if not query_text:
        return "❌ Usage: !reason [--mode=direct|verify|research|plan_execute] <query>"

    try:
        query = Query(text=query_text)
        result = await orchestrator.reason(
            query=query,
            mode=mode
        )

        # Format response
        mode_icons = {
            'DIRECT': '💭',
            'VERIFY': '✅',
            'RESEARCH': '🔬',
            'PLAN_EXECUTE': '📋'
        }
        icon = mode_icons.get(mode.name, '💭')

        response = f"{icon} **{mode.name.title()} Reasoning**\n\n"

        # Main response
        if hasattr(result, 'spacetime') and result.spacetime:
            response += f"{result.spacetime.response}\n\n"
            response += f"**Confidence:** {result.spacetime.confidence:.2f}\n"
            response += f"**Mode:** {mode.name}\n"

        # Metadata if available
        if hasattr(result, 'metadata') and result.metadata:
            if result.metadata.get('duration_ms'):
                response += f"_Reasoning took {result.metadata['duration_ms']:.0f}ms_\n"

        return response

    except Exception as e:
        logger.error(f"Error in reason: {e}")
        return f"❌ Reasoning failed: {str(e)}"


async def handle_agentic_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !agentic help command.

    Returns:
        Help message
    """
    return """🤖 **Agentic Reasoning Commands**

**Reasoning Modes:**
• `!reason <query>` - Quick single-pass reasoning (DIRECT)
• `!verify <claim>` - Cross-check verification with sources
• `!research <topic>` - Multi-query exploration from multiple angles
• `!plan <goal>` - Goal decomposition into actionable steps

**Options:**
• `!research --steps=10 <topic>` - Limit exploration steps
• `!plan --steps=8 <goal>` - Limit planning steps
• `!reason --mode=verify <query>` - Override reasoning mode

**Mode Descriptions:**
• `DIRECT` - Single-pass answer (fastest, ~150ms)
• `VERIFY` - Answer + verification (checks for contradictions)
• `RESEARCH` - Multi-query exploration (most thorough)
• `PLAN_EXECUTE` - Goal decomposition (multi-step tasks)

**Examples:**
```
!reason How does attention work in transformers?
!verify Thompson Sampling always outperforms UCB
!research What are the tradeoffs of different exploration strategies?
!plan Implement a recommendation system using collaborative filtering
!research --steps=10 Deep dive into Bayesian optimization
```
"""


# ============================================================================
# Registration
# ============================================================================

def register_agentic_handlers(
    bot,
    orchestrator: 'AgenticOrchestrator'
) -> None:
    """
    Register all Agentic Reasoning command handlers.

    Args:
        bot: Matrix bot instance
        orchestrator: AgenticOrchestrator instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, agentic handlers not registered")
        return

    if not AGENTIC_AVAILABLE:
        logger.warning("Agentic system not available, handlers not registered")
        return

    logger.info("Registering Agentic Reasoning command handlers")

    # Define command map
    command_map = {
        "research": lambda room, event, args: handle_research(room, event, args, orchestrator),
        "verify": lambda room, event, args: handle_verify(room, event, args, orchestrator),
        "plan": lambda room, event, args: handle_plan(room, event, args, orchestrator),
        "reason": lambda room, event, args: handle_reason(room, event, args, orchestrator),
    }

    # Register with bot
    for cmd, handler in command_map.items():
        bot.register_command(cmd, handler)

    # Register help
    bot.register_command("agentic", handle_agentic_help)
    bot.register_command("agentic help", handle_agentic_help)

    logger.info(f"Registered {len(command_map)} agentic reasoning commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class AgenticHandlers:
    """
    Decorator-based ChatOps handlers for Agentic Reasoning.

    Usage:
        from HoloLoom.apps.chatops.handlers.agentic_handlers import AgenticHandlers

        handlers = AgenticHandlers(orchestrator=agentic_orchestrator)
        registry.register_instance(handlers)
    """

    def __init__(self, orchestrator: 'AgenticOrchestrator'):
        """
        Initialize with AgenticOrchestrator instance.

        Args:
            orchestrator: AgenticOrchestrator instance for reasoning
        """
        self.orchestrator = orchestrator

    @HandlerRegistry.register(
        command="research",
        description="Multi-query exploration from multiple angles",
        usage="!research [--steps=N] <topic>",
        category=HandlerCategory.QUERY,
        aliases=["explore"]
    )
    async def handle_research(self, room, event, args: str) -> str:
        """Handle !research command."""
        return await handle_research(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="verify",
        description="Cross-check verification with sources",
        usage="!verify <claim>",
        category=HandlerCategory.QUERY,
        aliases=["check", "validate"]
    )
    async def handle_verify(self, room, event, args: str) -> str:
        """Handle !verify command."""
        return await handle_verify(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="plan",
        description="Goal decomposition into actionable steps",
        usage="!plan [--steps=N] <goal>",
        category=HandlerCategory.QUERY,
        aliases=["decompose"]
    )
    async def handle_plan(self, room, event, args: str) -> str:
        """Handle !plan command."""
        return await handle_plan(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="reason",
        description="Standard single-pass reasoning",
        usage="!reason [--mode=direct|verify|research|plan_execute] <query>",
        category=HandlerCategory.QUERY,
        aliases=["think"]
    )
    async def handle_reason(self, room, event, args: str) -> str:
        """Handle !reason command."""
        return await handle_reason(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="agentic help",
        description="Show agentic reasoning command help",
        usage="!agentic help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    )
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !agentic help command."""
        return await handle_agentic_help(room, event, args)
