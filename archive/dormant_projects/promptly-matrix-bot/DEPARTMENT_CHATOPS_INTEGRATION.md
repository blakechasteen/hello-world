# Department ChatOps Integration

**Status**: Design Document
**Date**: November 9, 2025

This document describes how the Promptly Matrix Bot integrates with HoloLoom's departmental architecture to provide **conversational control and monitoring** of the agent swarm.

## Vision

Matrix chat becomes the **command center** for the departmental agent swarm:
- Query departments directly from chat
- Monitor department status and health
- Escalate decisions requiring human input
- View session state and roadmap progress
- Control permissions and routing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Matrix Chat Room                         │
│  "Hey @promptly, what's MasterWeaver's confidence on Q4 data?"  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Promptly Matrix Bot │
              │   (ChatOps Layer)    │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐     ┌───▼───┐     ┌────▼────┐
    │   Git   │     │Claude │     │ Depart  │
    │ Handler │     │  API  │     │  ments  │ ← NEW
    └─────────┘     └───────┘     └────┬────┘
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                    ┌────▼──┐      ┌───▼───┐     ┌───▼────┐
                    │Master │      │Verif. │     │ Orch.  │
                    │Weaver │      │       │     │        │
                    └───────┘      └───────┘     └────────┘
```

## Matrix Commands for Departments

### 1. Query Department Status

```
@promptly department status
```

Returns:
```
Department Status:

MasterWeaver     [●●●●●] Healthy  (Processing Q4 data, 89% confidence)
Verification     [●●●●○] Warning  (Detected 15% overconfidence)
Infrastructure   [●●●●●] Healthy  (Queries <100ms)
Execution        [●●●○○] Degraded (2 tasks pending)
Context          [●●●●●] Healthy  (60k/60k context budget)
Orchestration    [●●●●●] Healthy  (Session: q4_beekeeping_2025)
```

### 2. Query Specific Department

```
@promptly query MasterWeaver "What's your confidence on queen behavior extraction?"
```

Returns:
```
MasterWeaver Response:

Confidence: 0.89 (High)
Entities Extracted: 47 (queen behavior patterns)
Reasoning: Used domain ontology + cross-validated with hive inspection reports
Session: q4_beekeeping_2025

Verification Note: 15% overconfidence detected - recommend manual review
```

### 3. View Session State

```
@promptly session status
```

Returns:
```
Active Session: q4_beekeeping_2025

Roadmap Phase: Q4 Data Processing
├─ MasterWeaver:    [✓] Entity extraction complete
├─ Infrastructure:  [✓] Zero-copy queries operational
├─ Execution:       [▶] Deploying agent autonomy v1
└─ Verification:    [!] Confidence framework needed

Context Budget:
  Used: 187k / 300k tokens (62%)
  MasterWeaver: 45k/50k (90%)
  Context: 38k/60k (63%)
  Verification: 22k/30k (73%)
```

### 4. Request Department Action

```
@promptly run Verification "validate MasterWeaver confidence on task_id_123"
```

Returns:
```
Verification Task Started: task_abc456

Validating MasterWeaver output...
├─ Claimed confidence: 0.89
├─ Actual confidence: 0.74
├─ Discrepancy: -15%
└─ Recommendation: Rerun with confidence_threshold=0.9

Would you like me to request MasterWeaver rerun? (yes/no)
```

### 5. Escalate Decision

```
@promptly escalate "MasterWeaver confidence mismatch requires human review"
```

Returns:
```
Escalation Created: esc_789

Issue: MasterWeaver confidence mismatch
Severity: MEDIUM
Affected Departments: MasterWeaver, Verification
Human Review Required: YES

Assigned to: @blake
Waiting for decision...
```

### 6. View Department Permissions

```
@promptly permissions MasterWeaver
```

Returns:
```
MasterWeaver Permissions:

Allowed:
  [✓] Read Neo4j expertise nodes
  [✓] Write entity extractions
  [✓] Query beekeeping domain memory

Denied:
  [✗] Modify Verification outcomes
  [✗] Modify Infrastructure configs
  [✗] Execute deployment tasks

Recent Permission Requests (last 24h):
  - 2025-11-09 14:23: Attempted to write to Verification (DENIED)
```

### 7. Monitor Department Health

```
@promptly health MasterWeaver
```

Returns:
```
MasterWeaver Health Report:

Status: Healthy
Uptime: 7 days, 3 hours
Tasks Completed: 1,247
Success Rate: 94.2%
Avg Confidence: 0.87

Recent Issues:
  - 2025-11-08: Low confidence on 3 queen behavior queries (resolved)
  - 2025-11-07: Timeout on large transcript (resolved)

Performance:
  Avg Response Time: 156ms
  Context Usage: 45k/50k tokens (90%)
  Cache Hit Rate: 78%
```

### 8. View Department Dependencies

```
@promptly dependencies
```

Returns:
```
Department Dependency Graph:

Orchestration (no dependencies)
    │
    ├─► MasterWeaver → Infrastructure
    ├─► Verification → MasterWeaver, Infrastructure, Execution
    ├─► Infrastructure (no dependencies)
    ├─► Execution → Infrastructure
    └─► Context → Infrastructure, MasterWeaver

Dependency Health:
  All dependencies operational ✓
```

### 9. Compact Session Context

```
@promptly compact session
```

Returns:
```
Session Compaction Started: q4_beekeeping_2025

Before: 187k tokens used
After: 94k tokens used (50% reduction)

Compacted Artifacts:
  ├─ MasterWeaver: 45k → 18k (60% reduction)
  ├─ Context: 38k → 22k (42% reduction)
  └─ Verification: 22k → 15k (32% reduction)

Session state preserved. Resumable.
```

### 10. Request Department Re-run

```
@promptly rerun MasterWeaver task_123 "confidence_threshold=0.9"
```

Returns:
```
MasterWeaver Rerun Request: task_123_retry

Original Result:
  Confidence: 0.74
  Entities: 47

New Parameters:
  confidence_threshold: 0.9 (was 0.75)

Rerunning... (estimated 30s)

New Result:
  Confidence: 0.92
  Entities: 41 (filtered 6 low-confidence entities)

Verification: Confidence claim matches quality ✓
```

## Implementation

### 1. Department Bridge (`bot/department_bridge.py`)

```python
#!/usr/bin/env python3
"""
Department Bridge for Promptly Matrix Bot

Connects Matrix chat to HoloLoom departmental architecture via MCP.
"""

import logging
from typing import Dict, List, Optional, Any

# Import department registry
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mythRL"))

from HoloLoom.alignment.mcp_department_registry import (
    DEPARTMENT_REGISTRY,
    get_department,
    list_departments,
    check_permission,
    PermissionLevel
)

logger = logging.getLogger(__name__)


class DepartmentBridge:
    """Bridge between Matrix chat and HoloLoom departments"""

    def __init__(self):
        """Initialize department bridge"""
        self.departments = DEPARTMENT_REGISTRY
        logger.info(f"Department bridge initialized with {len(self.departments)} departments")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all departments"""
        status = {}
        for name, dept in self.departments.items():
            status[name] = {
                "role": dept.role,
                "health": "healthy",  # TODO: Implement health checks
                "tools": len(dept.tools),
                "context_budget": dept.context_budget,
                "context_used": 0,  # TODO: Implement context tracking
            }
        return status

    def query_department(self, department_name: str, query: str) -> str:
        """Query a specific department"""
        dept = get_department(department_name)
        if not dept:
            return f"Department '{department_name}' not found"

        # TODO: Implement actual MCP query
        return f"Query to {department_name}: {query}\n\n(MCP integration pending)"

    def get_permissions(self, department_name: str) -> Dict[str, Any]:
        """Get department permissions"""
        dept = get_department(department_name)
        if not dept:
            return {"error": f"Department '{department_name}' not found"}

        return {
            "department": department_name,
            "permissions": [p.value for p in dept.permissions],
            "tools": len(dept.tools),
            "dependencies": dept.dependencies
        }

    def get_dependencies(self) -> Dict[str, List[str]]:
        """Get department dependency graph"""
        deps = {}
        for name, dept in self.departments.items():
            deps[name] = dept.dependencies
        return deps

    def is_available(self) -> bool:
        """Check if department bridge is available"""
        return len(self.departments) > 0


# Example usage
if __name__ == "__main__":
    bridge = DepartmentBridge()

    print("=== Department Bridge Test ===\n")

    # Get status
    status = bridge.get_status()
    print(f"Departments available: {len(status)}\n")

    for name, info in status.items():
        print(f"{name}:")
        print(f"  Role: {info['role']}")
        print(f"  Tools: {info['tools']}")
        print()
```

### 2. Matrix Bot Integration (`bot/promptly_bot.py`)

Add to `__init__` method:

```python
# Initialize Department bridge (ChatOps Phase 3)
try:
    from .department_bridge import DepartmentBridge
    self.department_bridge = DepartmentBridge()
    if self.department_bridge.is_available():
        logger.info(f"Department bridge available ({len(self.department_bridge.departments)} departments)")
    else:
        logger.warning("Department bridge unavailable")
        self.department_bridge = None
except Exception as e:
    logger.warning(f"Department bridge init failed: {e}")
    self.department_bridge = None
```

Add command routing in `handle_command`:

```python
# Department commands (ChatOps Phase 3)
elif cmd_type == 'department-status':
    return await self.cmd_department_status(command, room)
elif cmd_type == 'department-query':
    return await self.cmd_department_query(command, room)
elif cmd_type == 'department-permissions':
    return await self.cmd_department_permissions(command, room)
```

### 3. Command Parser (`bot/command_parser.py`)

Add department command patterns:

```python
# Department commands
'department-status': r'(?:@promptly(?:bot)?\s+department\s+status|!department\s+status)',
'department-query': r'(?:@promptly(?:bot)?\s+query\s+(\w+)\s+"([^"]+)"|!query\s+(\w+)\s+"([^"]+)")',
'department-permissions': r'(?:@promptly(?:bot)?\s+permissions\s+(\w+)|!permissions\s+(\w+))',
'department-health': r'(?:@promptly(?:bot)?\s+health\s+(\w+)|!health\s+(\w+))',
'session-status': r'(?:@promptly(?:bot)?\s+session\s+status|!session\s+status)',
```

### 4. Department Command Methods (`bot/department_methods.py`)

```python
"""
Department command methods for Promptly Matrix Bot
"""

async def cmd_department_status(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle department status command"""
    if not self.department_bridge:
        return {
            "body": "Department bridge not available",
            "html": "<p>Department bridge not available</p>"
        }

    try:
        status = self.department_bridge.get_status()

        # Format status message
        body = "Department Status:\n\n"
        html = "<p><strong>Department Status:</strong></p><ul>"

        for name, info in status.items():
            health_icon = "●●●●●" if info["health"] == "healthy" else "●●●○○"
            body += f"{name:15} [{health_icon}] {info['health'].capitalize()}\n"
            html += f"<li><strong>{name}</strong>: {info['health']}</li>"

        html += "</ul>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Department status error: {e}")
        return {
            "body": f"Error getting department status: {e}",
            "html": f"<p>Error: <code>{e}</code></p>"
        }


async def cmd_department_query(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle department query command"""
    if not self.department_bridge:
        return {
            "body": "Department bridge not available",
            "html": "<p>Department bridge not available</p>"
        }

    department = command.get('department', '')
    query = command.get('query', '')

    if not department or not query:
        return {
            "body": 'Usage: @promptly query <department> "your question"',
            "html": '<p>Usage: <code>@promptly query &lt;department&gt; "your question"</code></p>'
        }

    await self.send_message(room.room_id, f"Querying {department}...")

    try:
        result = self.department_bridge.query_department(department, query)

        body = f"Query to {department}:\n\n{result}"
        html = f"<p><strong>Query to {department}:</strong></p><pre>{result}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Department query error: {e}")
        return {
            "body": f"Query failed: {e}",
            "html": f"<p>Query failed: <code>{e}</code></p>"
        }
```

## Integration Phases

### Phase 1: Read-Only Department Visibility (Week 1)
- ✅ Department bridge connects to registry
- ✅ Status command shows all departments
- ✅ Permissions command shows department capabilities
- ✅ Dependencies graph visible from chat

### Phase 2: Department Queries (Week 2)
- Query departments via MCP
- View department responses in chat
- Monitor department health
- Track context usage

### Phase 3: Department Control (Week 3)
- Request department actions
- Approve/deny permission requests
- Escalate decisions to humans
- Compact session context

### Phase 4: Full Integration (Week 4)
- Session management from chat
- Department re-runs
- Real-time monitoring
- Alert notifications

## Benefits

1. **Visibility**: See what departments are doing in real-time
2. **Control**: Issue commands and approvals from chat
3. **Debugging**: Monitor health and errors conversationally
4. **Collaboration**: Team members can all see department status
5. **Audit Trail**: Matrix chat logs all department interactions

## Example Workflow

```
User: @promptly department status
Bot: [Shows all departments healthy except Verification warning]

User: @promptly health Verification
Bot: [Shows 15% overconfidence detected in MasterWeaver]

User: @promptly query Verification "explain the confidence mismatch"
Bot: [Verification explains claimed 0.89 vs actual 0.74]

User: @promptly rerun MasterWeaver task_123 "confidence_threshold=0.9"
Bot: [MasterWeaver reruns with stricter params, now 0.92 confidence]

User: @promptly department status
Bot: [All departments now healthy ✓]
```

## Next Steps

1. **Create `bot/department_bridge.py`** - Connection to department registry
2. **Add command patterns** - Matrix command parsing
3. **Implement command methods** - Department operations
4. **Test with existing departments** - Validate integration
5. **Add MCP client** - Real department queries

---

**This makes ChatOps the human interface to the departmental agent swarm!**

Every department operation can be monitored, controlled, and debugged conversationally from Matrix chat. This is Conway's Law in action: your organizational structure (departments) is now directly accessible through your communication structure (Matrix chat).

*Generated: November 9, 2025*
