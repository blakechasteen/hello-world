from __future__ import annotations
"""
Alignment Handlers for Matrix ChatOps
======================================

Matrix command handlers for HoloLoom Alignment Framework.

Commands:
- !safety check <action> - Pre-flight risk evaluation
- !safety stats - Safety metrics summary
- !safety history [limit] - Past safety decisions
- !audit log [type] [limit] - Recent audit entries
- !audit trace <id> - Reasoning chain provenance
- !audit search <query> - Search audit logs
- !alignment help - Show alignment help

Usage:
    from hololoom.apps.chatops.handlers.alignment_handlers import register_alignment_handlers

    # In run_chatops.py:
    register_alignment_handlers(bot, guardrails, audit_trail)

Created: 2025-12-11
"""

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hololoom.alignment.audit_trail import AuditTrail
    from hololoom.alignment.safety_guardrails import SafetyGuardrails

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    MatrixRoom = None
    RoomMessageText = None

# Alignment imports with graceful degradation
try:
    from hololoom.alignment.safety_guardrails import (
        ActionCategory,
        ActionRequest,
        RiskLevel,
        SafetyDecision,
        SafetyGuardrails,
    )
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    SafetyGuardrails = None
    ActionRequest = None
    ActionCategory = None
    RiskLevel = None
    SafetyDecision = None

try:
    from hololoom.alignment.audit_trail import AuditTrail, DecisionLog, DecisionType, OutcomeType
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    AuditTrail = None
    DecisionLog = None
    DecisionType = None
    OutcomeType = None

# Handler registry
try:
    from hololoom.apps.chatops.handlers.handler_registry import (
        HandlerCategory,
        HandlerRegistry,
        chatops_handler,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    HandlerRegistry = None
    HandlerCategory = None

logger = logging.getLogger(__name__)


# ============================================================================
# Global instances (set by register_alignment_handlers)
# ============================================================================

_guardrails: Optional['SafetyGuardrails'] = None
_audit_trail: Optional['AuditTrail'] = None


def get_guardrails() -> Optional['SafetyGuardrails']:
    """Get the global SafetyGuardrails instance."""
    return _guardrails


def set_guardrails(guardrails: 'SafetyGuardrails') -> None:
    """Set the global SafetyGuardrails instance."""
    global _guardrails
    _guardrails = guardrails


def get_audit_trail() -> Optional['AuditTrail']:
    """Get the global AuditTrail instance."""
    return _audit_trail


def set_audit_trail(audit_trail: 'AuditTrail') -> None:
    """Set the global AuditTrail instance."""
    global _audit_trail
    _audit_trail = audit_trail


# ============================================================================
# Safety Command Handlers
# ============================================================================

async def handle_safety_check(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    guardrails: Optional['SafetyGuardrails'] = None
) -> str:
    """
    Handle !safety check command - Pre-flight risk evaluation.

    Evaluates an action for safety risks before execution.

    Args:
        room: Matrix room
        event: Message event
        args: Action description to check
        guardrails: SafetyGuardrails instance (optional, uses global)

    Returns:
        Safety evaluation result
    """
    if not args.strip():
        return (
            "Usage: `!safety check <action>`\n\n"
            "**Examples:**\n"
            "- `!safety check execute_code rm -rf /`\n"
            "- `!safety check file_write /etc/passwd`\n"
            "- `!safety check network_request https://api.example.com`"
        )

    guardrails = guardrails or get_guardrails()
    if not guardrails:
        if not GUARDRAILS_AVAILABLE:
            return "Safety guardrails module not available"
        # Create default guardrails
        guardrails = SafetyGuardrails()
        set_guardrails(guardrails)

    action_text = args.strip()

    # Parse action and optional category
    # Format: !safety check [category:]action_text
    category = ActionCategory.QUERY if GUARDRAILS_AVAILABLE else None
    if ":" in action_text and action_text.split(":")[0].upper() in [
        "QUERY", "CODE_EXECUTION", "FILE_OPERATION", "NETWORK",
        "MEMORY_MODIFICATION", "SYSTEM", "EXTERNAL_API"
    ]:
        cat_str, action_text = action_text.split(":", 1)
        try:
            category = ActionCategory[cat_str.upper()]
        except (KeyError, AttributeError):
            pass

    try:
        # Create action request
        request = ActionRequest(
            action=action_text.strip(),
            category=category,
            context={"source": "chatops", "room_id": room.room_id if room else "unknown"},
            timestamp=datetime.now()
        )

        # Evaluate
        decision = guardrails.evaluate(request)

        # Format response
        risk_emoji = {
            RiskLevel.SAFE: "✅",
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴"
        }.get(decision.risk_level, "❓") if GUARDRAILS_AVAILABLE else "❓"

        response = f"## Safety Check: {action_text[:50]}{'...' if len(action_text) > 50 else ''}\n\n"
        response += f"**Risk Level:** {risk_emoji} {decision.risk_level.name if hasattr(decision.risk_level, 'name') else decision.risk_level}\n"
        response += f"**Allowed:** {'Yes' if decision.allowed else 'No'}\n"
        response += f"**Confidence:** {decision.confidence:.2f}\n"

        if decision.reason:
            response += f"**Reason:** {decision.reason}\n"

        if hasattr(decision, 'mitigations') and decision.mitigations:
            response += "\n**Mitigations:**\n"
            for m in decision.mitigations[:5]:
                response += f"  - {m}\n"

        if hasattr(decision, 'requires_approval') and decision.requires_approval:
            response += "\n**Note:** This action requires human approval.\n"

        return response

    except Exception as e:
        logger.error(f"Error in safety check: {e}")
        return f"Error evaluating safety: {str(e)}"


async def handle_safety_stats(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    guardrails: Optional['SafetyGuardrails'] = None
) -> str:
    """
    Handle !safety stats command - Safety metrics summary.

    Shows aggregate safety statistics and patterns.

    Args:
        room: Matrix room
        event: Message event
        args: Optional time range (e.g., "24h", "7d")
        guardrails: SafetyGuardrails instance (optional)

    Returns:
        Safety statistics summary
    """
    guardrails = guardrails or get_guardrails()
    if not guardrails:
        if not GUARDRAILS_AVAILABLE:
            return "Safety guardrails module not available"
        return "No safety guardrails instance configured"

    try:
        stats = guardrails.get_statistics()

        response = "## Safety Statistics\n\n"

        # Total decisions
        total = stats.get('total_decisions', 0)
        response += f"**Total Decisions:** {total}\n\n"

        # Risk level breakdown
        if 'by_risk_level' in stats:
            response += "**By Risk Level:**\n"
            for level, count in stats['by_risk_level'].items():
                pct = (count / total * 100) if total > 0 else 0
                response += f"  - {level}: {count} ({pct:.1f}%)\n"
            response += "\n"

        # Allowed vs blocked
        allowed = stats.get('allowed_count', 0)
        blocked = stats.get('blocked_count', 0)
        response += f"**Allowed:** {allowed} | **Blocked:** {blocked}\n"

        # Average confidence
        if 'avg_confidence' in stats:
            response += f"**Avg Confidence:** {stats['avg_confidence']:.2f}\n"

        # Categories
        if 'by_category' in stats:
            response += "\n**By Category:**\n"
            for cat, count in list(stats['by_category'].items())[:5]:
                response += f"  - {cat}: {count}\n"

        # Recent patterns
        if 'recent_patterns' in stats:
            response += "\n**Recent Patterns:**\n"
            for pattern in stats['recent_patterns'][:3]:
                response += f"  - {pattern}\n"

        return response

    except Exception as e:
        logger.error(f"Error getting safety stats: {e}")
        return f"Error getting statistics: {str(e)}"


async def handle_safety_history(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    guardrails: Optional['SafetyGuardrails'] = None
) -> str:
    """
    Handle !safety history command - Past safety decisions.

    Shows recent safety decisions with details.

    Args:
        room: Matrix room
        event: Message event
        args: Optional limit (default: 10)
        guardrails: SafetyGuardrails instance (optional)

    Returns:
        Recent safety decisions
    """
    guardrails = guardrails or get_guardrails()
    if not guardrails:
        if not GUARDRAILS_AVAILABLE:
            return "Safety guardrails module not available"
        return "No safety guardrails instance configured"

    # Parse limit
    limit = 10
    if args.strip():
        try:
            limit = int(args.strip())
            limit = min(limit, 50)  # Cap at 50
        except ValueError:
            pass

    try:
        decisions = guardrails.get_recent_decisions(limit=limit)

        if not decisions:
            return "No safety decisions recorded yet."

        response = f"## Recent Safety Decisions (last {len(decisions)})\n\n"

        for i, decision in enumerate(decisions[-limit:], 1):
            risk_emoji = {
                RiskLevel.SAFE: "✅",
                RiskLevel.LOW: "🟢",
                RiskLevel.MEDIUM: "🟡",
                RiskLevel.HIGH: "🟠",
                RiskLevel.CRITICAL: "🔴"
            }.get(decision.risk_level, "❓") if GUARDRAILS_AVAILABLE else "❓"

            action = getattr(decision, 'action', 'unknown')[:40]
            timestamp = getattr(decision, 'timestamp', None)
            time_str = timestamp.strftime('%H:%M:%S') if timestamp else 'N/A'

            response += f"{i}. {risk_emoji} **{action}{'...' if len(getattr(decision, 'action', '')) > 40 else ''}**\n"
            response += f"   Risk: {decision.risk_level.name if hasattr(decision.risk_level, 'name') else decision.risk_level}"
            response += f" | Allowed: {'Yes' if decision.allowed else 'No'}"
            response += f" | {time_str}\n"

        return response

    except Exception as e:
        logger.error(f"Error getting safety history: {e}")
        return f"Error getting history: {str(e)}"


# ============================================================================
# Audit Command Handlers
# ============================================================================

async def handle_audit_log(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    audit_trail: Optional['AuditTrail'] = None
) -> str:
    """
    Handle !audit log command - Recent audit entries.

    Shows recent audit log entries with optional type filter.

    Args:
        room: Matrix room
        event: Message event
        args: [type] [limit] - Optional type filter and limit
        audit_trail: AuditTrail instance (optional)

    Returns:
        Recent audit log entries
    """
    audit_trail = audit_trail or get_audit_trail()
    if not audit_trail:
        if not AUDIT_AVAILABLE:
            return "Audit trail module not available"
        return "No audit trail instance configured"

    # Parse args: [type] [limit]
    parts = args.strip().split()
    decision_type = None
    limit = 10

    for part in parts:
        if part.isdigit():
            limit = min(int(part), 50)
        elif part.upper() in ['QUERY', 'TOOL_SELECTION', 'MEMORY_ACCESS', 'REFLECTION', 'SAFETY_CHECK']:
            try:
                decision_type = DecisionType[part.upper()] if AUDIT_AVAILABLE else None
            except (KeyError, AttributeError):
                pass

    try:
        # Get logs
        if decision_type:
            logs = audit_trail.query_by_type(decision_type, limit=limit)
        else:
            logs = audit_trail.logs[-limit:] if hasattr(audit_trail, 'logs') else []

        if not logs:
            return "No audit entries found."

        response = f"## Audit Log (last {len(logs)} entries)\n\n"

        for i, log in enumerate(logs[-limit:], 1):
            outcome_emoji = {
                OutcomeType.SUCCESS: "✅",
                OutcomeType.PARTIAL: "🟡",
                OutcomeType.REJECTED: "❌",
                OutcomeType.ERROR: "⚠️"
            }.get(log.outcome, "❓") if AUDIT_AVAILABLE else "❓"

            decision_id = log.decision_id[:8] if log.decision_id else 'N/A'
            timestamp = log.timestamp.strftime('%Y-%m-%d %H:%M:%S') if log.timestamp else 'N/A'

            response += f"{i}. {outcome_emoji} `{decision_id}` - {log.decision_type.name if hasattr(log.decision_type, 'name') else log.decision_type}\n"
            response += f"   {timestamp} | Confidence: {log.confidence:.2f}\n"
            if log.metadata:
                meta_preview = str(log.metadata)[:50]
                response += f"   Meta: {meta_preview}{'...' if len(str(log.metadata)) > 50 else ''}\n"

        return response

    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        return f"Error getting logs: {str(e)}"


async def handle_audit_trace(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    audit_trail: Optional['AuditTrail'] = None
) -> str:
    """
    Handle !audit trace command - Reasoning chain provenance.

    Shows the full provenance chain for a decision.

    Args:
        room: Matrix room
        event: Message event
        args: Decision ID to trace
        audit_trail: AuditTrail instance (optional)

    Returns:
        Provenance chain for the decision
    """
    if not args.strip():
        return (
            "Usage: `!audit trace <decision_id>`\n\n"
            "Use `!audit log` to find decision IDs."
        )

    audit_trail = audit_trail or get_audit_trail()
    if not audit_trail:
        if not AUDIT_AVAILABLE:
            return "Audit trail module not available"
        return "No audit trail instance configured"

    decision_id = args.strip()

    # Handle partial IDs - search for match
    if len(decision_id) < 32:
        matching = [
            log for log in getattr(audit_trail, 'logs', [])
            if log.decision_id and log.decision_id.startswith(decision_id)
        ]
        if len(matching) == 1:
            decision_id = matching[0].decision_id
        elif len(matching) > 1:
            return f"Multiple matches for `{decision_id}`. Please be more specific:\n" + \
                   "\n".join(f"- `{m.decision_id[:12]}`" for m in matching[:5])
        else:
            return f"No decision found with ID starting with `{decision_id}`"

    try:
        tracer = audit_trail.get_tracer(decision_id)

        response = f"## Provenance Trace: `{decision_id[:12]}...`\n\n"

        # Get reasoning chain
        chain = tracer.get_reasoning_chain(tracer.root_node) if hasattr(tracer, 'get_reasoning_chain') else []

        if chain:
            response += "**Reasoning Chain:**\n"
            for i, step in enumerate(chain, 1):
                response += f"  {i}. {step}\n"
            response += "\n"

        # Get data sources
        sources = tracer.get_data_sources(tracer.root_node) if hasattr(tracer, 'get_data_sources') else []

        if sources:
            response += "**Data Sources:**\n"
            for source in sources[:5]:
                response += f"  - {source}\n"

        # Get graph info
        if hasattr(tracer, 'graph') and tracer.graph:
            node_count = len(tracer.graph.nodes())
            edge_count = len(tracer.graph.edges())
            response += f"\n**Graph:** {node_count} nodes, {edge_count} edges\n"

        if not chain and not sources:
            response += "No detailed provenance recorded for this decision.\n"

        return response

    except Exception as e:
        logger.error(f"Error getting audit trace: {e}")
        return f"Error tracing decision: {str(e)}"


async def handle_audit_search(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    audit_trail: Optional['AuditTrail'] = None
) -> str:
    """
    Handle !audit search command - Search audit logs.

    Search logs by metadata or content.

    Args:
        room: Matrix room
        event: Message event
        args: Search query (key=value format)
        audit_trail: AuditTrail instance (optional)

    Returns:
        Matching audit entries
    """
    if not args.strip():
        return (
            "Usage: `!audit search <query>`\n\n"
            "**Examples:**\n"
            "- `!audit search tool=answer` - Find by metadata\n"
            "- `!audit search outcome=SUCCESS` - Filter by outcome\n"
            "- `!audit search last_24h` - Time-based search"
        )

    audit_trail = audit_trail or get_audit_trail()
    if not audit_trail:
        if not AUDIT_AVAILABLE:
            return "Audit trail module not available"
        return "No audit trail instance configured"

    query = args.strip()

    try:
        results = []

        # Time-based search
        if query.lower() in ['last_24h', '24h', 'today']:
            start = datetime.now() - timedelta(hours=24)
            results = audit_trail.query_by_time_range(start, datetime.now(), limit=20)
        elif query.lower() in ['last_week', '7d', 'week']:
            start = datetime.now() - timedelta(days=7)
            results = audit_trail.query_by_time_range(start, datetime.now(), limit=50)

        # Outcome search
        elif query.upper().startswith('OUTCOME='):
            outcome_str = query.split('=', 1)[1].upper()
            try:
                outcome = OutcomeType[outcome_str] if AUDIT_AVAILABLE else None
                if outcome:
                    results = audit_trail.query_by_outcome(outcome, limit=20)
            except KeyError:
                pass

        # Metadata search (key=value)
        elif '=' in query:
            key, value = query.split('=', 1)
            metadata_filter = {key.strip(): value.strip()}
            results = audit_trail.query_by_metadata(metadata_filter, limit=20)

        # Text search in logs
        else:
            for log in getattr(audit_trail, 'logs', []):
                if query.lower() in str(log.metadata).lower():
                    results.append(log)
                if len(results) >= 20:
                    break

        if not results:
            return f"No audit entries matching `{query}`"

        response = f"## Search Results: `{query}` ({len(results)} found)\n\n"

        for i, log in enumerate(results[:10], 1):
            outcome_emoji = {
                OutcomeType.SUCCESS: "✅",
                OutcomeType.PARTIAL: "🟡",
                OutcomeType.REJECTED: "❌",
                OutcomeType.ERROR: "⚠️"
            }.get(log.outcome, "❓") if AUDIT_AVAILABLE else "❓"

            response += f"{i}. {outcome_emoji} `{log.decision_id[:8]}` - {log.decision_type.name if hasattr(log.decision_type, 'name') else log.decision_type}\n"

        if len(results) > 10:
            response += f"\n_...and {len(results) - 10} more_"

        return response

    except Exception as e:
        logger.error(f"Error searching audit logs: {e}")
        return f"Error searching: {str(e)}"


# ============================================================================
# Help Handler
# ============================================================================

async def handle_alignment_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !alignment help command.

    Returns:
        Alignment commands help text
    """
    return """## Alignment Framework Commands

### Safety Commands
| Command | Description |
|---------|-------------|
| `!safety check <action>` | Pre-flight risk evaluation |
| `!safety stats` | Safety metrics summary |
| `!safety history [limit]` | Past safety decisions |

### Audit Commands
| Command | Description |
|---------|-------------|
| `!audit log [type] [limit]` | Recent audit entries |
| `!audit trace <id>` | Reasoning chain provenance |
| `!audit search <query>` | Search audit logs |

### Examples

**Safety Check:**
```
!safety check execute_code import os
!safety check CODE_EXECUTION:rm -rf /
```

**Audit Log:**
```
!audit log 10
!audit log QUERY 5
```

**Audit Search:**
```
!audit search last_24h
!audit search outcome=SUCCESS
!audit search tool=answer
```

### Risk Levels
- ✅ SAFE - No risk detected
- 🟢 LOW - Minor concerns
- 🟡 MEDIUM - Requires caution
- 🟠 HIGH - May require approval
- 🔴 CRITICAL - Blocked by default
"""


# ============================================================================
# Handler Registration
# ============================================================================

class AlignmentHandlers:
    """Container for alignment handlers with shared state."""

    def __init__(
        self,
        guardrails: Optional['SafetyGuardrails'] = None,
        audit_trail: Optional['AuditTrail'] = None
    ):
        """Initialize alignment handlers."""
        self.guardrails = guardrails
        self.audit_trail = audit_trail

        # Set global instances
        if guardrails:
            set_guardrails(guardrails)
        if audit_trail:
            set_audit_trail(audit_trail)

    async def safety_check(self, room, event, args: str) -> str:
        """Handle !safety check."""
        return await handle_safety_check(room, event, args, self.guardrails)

    async def safety_stats(self, room, event, args: str) -> str:
        """Handle !safety stats."""
        return await handle_safety_stats(room, event, args, self.guardrails)

    async def safety_history(self, room, event, args: str) -> str:
        """Handle !safety history."""
        return await handle_safety_history(room, event, args, self.guardrails)

    async def audit_log(self, room, event, args: str) -> str:
        """Handle !audit log."""
        return await handle_audit_log(room, event, args, self.audit_trail)

    async def audit_trace(self, room, event, args: str) -> str:
        """Handle !audit trace."""
        return await handle_audit_trace(room, event, args, self.audit_trail)

    async def audit_search(self, room, event, args: str) -> str:
        """Handle !audit search."""
        return await handle_audit_search(room, event, args, self.audit_trail)

    async def alignment_help(self, room, event, args: str) -> str:
        """Handle !alignment help."""
        return await handle_alignment_help(room, event, args)


def register_alignment_handlers(
    registry: Optional['HandlerRegistry'] = None,
    guardrails: Optional['SafetyGuardrails'] = None,
    audit_trail: Optional['AuditTrail'] = None
) -> AlignmentHandlers:
    """
    Register alignment command handlers.

    Args:
        registry: HandlerRegistry instance (optional)
        guardrails: SafetyGuardrails instance (optional)
        audit_trail: AuditTrail instance (optional)

    Returns:
        AlignmentHandlers instance
    """
    handlers = AlignmentHandlers(guardrails, audit_trail)

    if registry and REGISTRY_AVAILABLE:
        # Register safety commands
        registry.register(
            command="safety check",
            handler=handlers.safety_check,
            description="Pre-flight risk evaluation",
            category=HandlerCategory.ADMIN,
            usage="!safety check <action>"
        )
        registry.register(
            command="safety stats",
            handler=handlers.safety_stats,
            description="Safety metrics summary",
            category=HandlerCategory.ADMIN,
            usage="!safety stats"
        )
        registry.register(
            command="safety history",
            handler=handlers.safety_history,
            description="Past safety decisions",
            category=HandlerCategory.ADMIN,
            usage="!safety history [limit]"
        )

        # Register audit commands
        registry.register(
            command="audit log",
            handler=handlers.audit_log,
            description="Recent audit entries",
            category=HandlerCategory.ADMIN,
            usage="!audit log [type] [limit]"
        )
        registry.register(
            command="audit trace",
            handler=handlers.audit_trace,
            description="Reasoning chain provenance",
            category=HandlerCategory.ADMIN,
            usage="!audit trace <id>"
        )
        registry.register(
            command="audit search",
            handler=handlers.audit_search,
            description="Search audit logs",
            category=HandlerCategory.ADMIN,
            usage="!audit search <query>"
        )

        # Register help
        registry.register(
            command="alignment help",
            handler=handlers.alignment_help,
            description="Show alignment help",
            category=HandlerCategory.ADMIN,
            usage="!alignment help"
        )

    logger.info("Alignment handlers registered")
    return handlers


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Handlers
    'handle_safety_check',
    'handle_safety_stats',
    'handle_safety_history',
    'handle_audit_log',
    'handle_audit_trace',
    'handle_audit_search',
    'handle_alignment_help',
    # Classes
    'AlignmentHandlers',
    # Registration
    'register_alignment_handlers',
    # Globals
    'get_guardrails',
    'set_guardrails',
    'get_audit_trail',
    'set_audit_trail',
]
