# Multi-Bot Department Architecture

**Status**: Design Document
**Date**: November 9, 2025

## The Vision

Each department is its own Matrix bot in a shared room, coordinated by the Orchestrator bot. Departments **collaborate conversationally** - just like human teams in Slack/Matrix, but with AI agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Matrix Room: #hololoom-departments                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  @orchestrator: MasterWeaver, please extract entities from Q4   │
│  @masterweaver: Starting extraction... (47 entities found,      │
│                 confidence 0.89)                                │
│  @verification: I'm detecting 15% overconfidence. Recommend     │
│                 rerun with threshold=0.9                        │
│  @orchestrator: Approved. MasterWeaver, please rerun.           │
│  @masterweaver: Rerunning... (41 entities, confidence 0.92)     │
│  @verification: ✓ Confidence matches quality. Approved.         │
│  @infrastructure: Storing 41 entities to Neo4j...               │
│  @infrastructure: ✓ Complete. Query response <100ms             │
│  @orchestrator: Task complete. Session state updated.           │
│                                                                 │
│  [blake]: Nice work team! What's next?                          │
│  @orchestrator: Next: Deploy agent autonomy v1 (Execution)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Bot Roster

Each department = its own bot with distinct personality and responsibilities:

### 1. @orchestrator (Orchestrator Department)
**Role**: Team coordinator, session manager, task router
**Personality**: Professional, decisive, big-picture thinker
**Responsibilities**:
- Routes tasks to appropriate departments
- Manages session state
- Escalates decisions requiring human input
- Updates roadmap status
- Monitors department health

**Example Messages**:
```
@orchestrator: Good morning team. Session q4_beekeeping_2025 resuming.
@orchestrator: MasterWeaver, please extract entities from inspection_20251015.txt
@orchestrator: Task routed to MasterWeaver (estimated 30s)
@orchestrator: @blake, I need your approval for production deployment
```

### 2. @masterweaver (MasterWeaver Department)
**Role**: Entity extraction, domain understanding
**Personality**: Meticulous, detail-oriented, domain expert
**Responsibilities**:
- Extract entities from multimodal input
- Validate entity consistency
- Query domain ontologies
- Report confidence levels

**Example Messages**:
```
@masterweaver: Starting entity extraction...
@masterweaver: Found 47 queen behavior patterns (confidence 0.89)
@masterweaver: Cross-validated with hive inspection reports ✓
@masterweaver: Reasoning: Used domain ontology + historical data
```

### 3. @verification (Verification Department)
**Role**: Quality assurance, confidence validation
**Personality**: Skeptical, rigorous, quality-focused
**Responsibilities**:
- Validate confidence claims
- Cross-check department outputs
- Request re-runs when needed
- Generate alignment reports

**Example Messages**:
```
@verification: Checking MasterWeaver's confidence claim...
@verification: ⚠️ Detected 15% overconfidence (claimed 0.89, actual 0.74)
@verification: Recommendation: Rerun with confidence_threshold=0.9
@verification: ✓ New result validated. Confidence matches quality.
```

### 4. @infrastructure (Infrastructure Department)
**Role**: Data systems, performance optimization
**Personality**: Pragmatic, efficiency-focused, systems thinker
**Responsibilities**:
- Manage Neo4j and Qdrant
- Optimize query performance
- Report system health
- Handle data persistence

**Example Messages**:
```
@infrastructure: Storing 41 entities to Neo4j...
@infrastructure: Query optimization applied: <100ms response time ✓
@infrastructure: System health: All services operational
@infrastructure: Context budget: 187k/300k tokens (62% used)
```

### 5. @execution (Execution Department)
**Role**: Task execution, code running
**Personality**: Action-oriented, results-driven
**Responsibilities**:
- Execute Claude Code tasks
- Monitor task status
- Report completion
- Handle deployments

**Example Messages**:
```
@execution: Task received: Deploy agent autonomy v1
@execution: Running pre-deployment checks...
@execution: ✓ Tests passed. Ready to deploy.
@execution: Deployment in progress... (estimated 2 min)
@execution: ✓ Deployed to staging. Monitoring...
```

### 6. @context (Context/HoloLoom Department)
**Role**: Multi-pass context enrichment
**Personality**: Thoughtful, connective, holistic
**Responsibilities**:
- Enrich context with multi-modal data
- Detect missing context
- Perform multi-pass graph traversal
- Provide context to other departments

**Example Messages**:
```
@context: Enriching query context... (pass 1/3)
@context: Found relevant beekeeping domain knowledge
@context: Cross-referencing with historical data...
@context: Context enrichment complete (confidence 0.91)
```

## Bot Implementation

### Bot Registry (`bot/department_bots.py`)

```python
#!/usr/bin/env python3
"""
Department Bots for Multi-Bot Architecture

Each department is a separate Matrix bot in a shared room.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DepartmentBot:
    """Configuration for a department bot"""
    name: str
    username: str  # Matrix username
    display_name: str
    role: str
    personality: str
    avatar_color: str  # Hex color for avatar


# Department bot configurations
DEPARTMENT_BOTS: Dict[str, DepartmentBot] = {
    "orchestrator": DepartmentBot(
        name="Orchestration",
        username="orchestrator",
        display_name="Orchestrator",
        role="Team coordinator, session manager",
        personality="Professional, decisive, big-picture",
        avatar_color="#1E88E5"  # Blue
    ),
    "masterweaver": DepartmentBot(
        name="MasterWeaver",
        username="masterweaver",
        display_name="MasterWeaver",
        role="Entity extraction, domain understanding",
        personality="Meticulous, detail-oriented, expert",
        avatar_color="#43A047"  # Green
    ),
    "verification": DepartmentBot(
        name="Verification",
        username="verification",
        display_name="Verification",
        role="Quality assurance, confidence validation",
        personality="Skeptical, rigorous, quality-focused",
        avatar_color="#E53935"  # Red
    ),
    "infrastructure": DepartmentBot(
        name="Infrastructure",
        username="infrastructure",
        display_name="Infrastructure",
        role="Data systems, performance optimization",
        personality="Pragmatic, efficiency-focused",
        avatar_color="#FB8C00"  # Orange
    ),
    "execution": DepartmentBot(
        name="Execution",
        username="execution",
        display_name="Execution",
        role="Task execution, code running",
        personality="Action-oriented, results-driven",
        avatar_color="#8E24AA"  # Purple
    ),
    "context": DepartmentBot(
        name="Context",
        username="context",
        display_name="Context",
        role="Multi-pass context enrichment",
        personality="Thoughtful, connective, holistic",
        avatar_color="#00ACC1"  # Cyan
    )
}


def get_bot_config(department_name: str) -> Optional[DepartmentBot]:
    """Get bot configuration for a department"""
    return DEPARTMENT_BOTS.get(department_name.lower())


def list_bots() -> Dict[str, DepartmentBot]:
    """List all department bots"""
    return DEPARTMENT_BOTS
```

### Multi-Bot Manager (`bot/multi_bot_manager.py`)

```python
#!/usr/bin/env python3
"""
Multi-Bot Manager

Manages multiple department bots in a shared Matrix room.
"""

import asyncio
import logging
from typing import Dict, Optional
from nio import AsyncClient, MatrixRoom, RoomMessageText

from .department_bots import DEPARTMENT_BOTS, get_bot_config

logger = logging.getLogger(__name__)


class DepartmentBotInstance:
    """Single department bot instance"""

    def __init__(self, config, homeserver: str, room_id: str):
        self.config = config
        self.homeserver = homeserver
        self.room_id = room_id
        self.client = None

    async def start(self):
        """Start the bot"""
        # Create Matrix client for this bot
        self.client = AsyncClient(
            self.homeserver,
            f"@{self.config.username}:matrix.org"
        )

        # Set display name and avatar
        # await self.client.set_displayname(self.config.display_name)

        # Register message callback
        self.client.add_event_callback(self.message_callback, RoomMessageText)

        logger.info(f"Started {self.config.display_name} bot")

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming messages"""
        # Check if bot is mentioned
        if f"@{self.config.username}" in event.body:
            # Bot was mentioned - respond based on department logic
            await self.handle_mention(room, event)

    async def handle_mention(self, room: MatrixRoom, event: RoomMessageText):
        """Handle when bot is mentioned"""
        # Department-specific logic here
        pass

    async def send_message(self, message: str):
        """Send message to the room"""
        if self.client:
            await self.client.room_send(
                room_id=self.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message
                }
            )


class MultiBotManager:
    """Manages all department bots"""

    def __init__(self, homeserver: str, room_id: str):
        self.homeserver = homeserver
        self.room_id = room_id
        self.bots: Dict[str, DepartmentBotInstance] = {}

    async def start_all_bots(self):
        """Start all department bots"""
        for name, config in DEPARTMENT_BOTS.items():
            bot = DepartmentBotInstance(config, self.homeserver, self.room_id)
            await bot.start()
            self.bots[name] = bot

        logger.info(f"Started {len(self.bots)} department bots")

    async def get_bot(self, department_name: str) -> Optional[DepartmentBotInstance]:
        """Get a specific bot instance"""
        return self.bots.get(department_name.lower())

    async def broadcast(self, message: str, from_department: str):
        """Broadcast message from a department"""
        bot = await self.get_bot(from_department)
        if bot:
            await bot.send_message(message)
```

## Example Conversations

### Scenario 1: Entity Extraction Task

```
@orchestrator: MasterWeaver, please extract entities from Q4 beekeeping data
@masterweaver: Starting extraction on 47 inspection reports...
@masterweaver: Progress: 23/47 complete (49%)
@context: I'm providing domain context from historical data
@masterweaver: Thanks Context! Using ontology cross-validation
@masterweaver: ✓ Complete. 1,203 entities extracted (confidence 0.89)
@verification: Checking MasterWeaver's output...
@verification: ⚠️ Detected 15% overconfidence. Recommend rerun.
@orchestrator: MasterWeaver, please rerun with threshold=0.9
@masterweaver: Rerunning with stricter parameters...
@masterweaver: ✓ Complete. 1,087 entities (confidence 0.92)
@verification: ✓ Validated. Confidence matches quality.
@infrastructure: Storing to Neo4j... Done. Queries optimized <100ms
@orchestrator: Task complete. Session state updated.
```

### Scenario 2: Deployment Decision

```
@execution: Ready to deploy agent autonomy v1 to staging
@verification: Have all tests passed?
@execution: Yes. 47/47 tests passing. Code coverage 94%.
@infrastructure: System health check: All services operational
@verification: ✓ Pre-deployment checks passed
@orchestrator: @blake, approval needed for staging deployment
[blake]: Approved
@orchestrator: Execution, you're cleared to deploy
@execution: Deploying to staging...
@execution: ✓ Deployed. Monitoring for issues...
@infrastructure: Performance looks good. Response times normal.
@orchestrator: Deployment successful. Logging to audit trail.
```

### Scenario 3: Error Escalation

```
@masterweaver: ⚠️ Error: Unable to extract entities from file_xyz
@masterweaver: Confidence dropped to 0.32. File may be corrupted.
@verification: Confirming low quality output from MasterWeaver
@infrastructure: Checking file integrity... File appears corrupted
@context: I'm missing required domain context for this file type
@orchestrator: Multiple issues detected. Escalating to human.
@orchestrator: @blake, we need your help with file_xyz
@orchestrator: Summary: Corrupted file + missing context
[blake]: Ah, that's a new data format. Let me provide a sample
@context: Thanks! Processing sample format...
@context: ✓ New format learned. Ready to retry.
@orchestrator: MasterWeaver, retry with new context
@masterweaver: ✓ Success! 89 entities extracted (confidence 0.88)
```

### Scenario 4: Session Management

```
[blake]: What's the status of Q4 project?
@orchestrator: Session: q4_beekeeping_2025
@orchestrator: Phase: Data Processing (Week 8/12)
@orchestrator: Status:
@orchestrator:   MasterWeaver: ✓ Complete (1,203 entities)
@orchestrator:   Infrastructure: ✓ Complete (queries <100ms)
@orchestrator:   Execution: ▶ In Progress (agent autonomy v1)
@orchestrator:   Verification: ! Blocked (confidence framework)
@orchestrator: Context budget: 187k/300k (62% used)
@orchestrator: Next milestone: Complete agent autonomy (3 days)
```

## Benefits of Multi-Bot Architecture

### 1. **Anthropomorphization**
Each department has personality and voice. This makes it easier for humans to understand what's happening and build trust with the system.

### 2. **Transparency**
All inter-department communication is visible in the chat. Humans can see exactly how departments collaborate and make decisions.

### 3. **Debuggability**
When something goes wrong, you can see the exact conversation that led to the error. Natural language debugging.

### 4. **Human Participation**
Humans can jump into the conversation at any time, provide guidance, approve decisions, or ask questions.

### 5. **Audit Trail**
Matrix chat logs provide complete audit trail of all department interactions, decisions, and escalations.

### 6. **Team Dynamics**
Departments can develop working relationships, learn from each other, and improve collaboration over time.

## Implementation Phases

### Phase 1: Single Room, Multiple Bots (Week 1)
- Deploy 6 department bots to shared room
- Basic message routing (mentions)
- Simple status updates

### Phase 2: Inter-Department Communication (Week 2)
- Departments respond to each other
- Orchestrator coordinates tasks
- Verification challenges outputs

### Phase 3: Human Integration (Week 3)
- Humans can mention specific bots
- Approval workflows
- Escalation protocols

### Phase 4: Advanced Collaboration (Week 4)
- Session management visible in chat
- Real-time debugging
- Learning from human feedback

## Matrix Room Setup

```yaml
Room: #hololoom-departments
Privacy: Invite-only
Purpose: Departmental agent coordination

Members:
  - @blake:matrix.org (admin, human)
  - @orchestrator:matrix.org (bot)
  - @masterweaver:matrix.org (bot)
  - @verification:matrix.org (bot)
  - @infrastructure:matrix.org (bot)
  - @execution:matrix.org (bot)
  - @context:matrix.org (bot)

Room Rules:
  1. Orchestrator coordinates all tasks
  2. Departments report status and confidence
  3. Verification validates outputs
  4. Humans can intervene at any time
  5. All decisions are logged and visible
```

## Advanced Features

### 1. **Department Threading**
```
@orchestrator: Starting Q4 data processing [Thread]
  ├─ @masterweaver: Extracting entities... [Reply]
  ├─ @context: Providing domain context... [Reply]
  └─ @verification: Validating outputs... [Reply]
```

### 2. **Status Reactions**
Bots use Matrix reactions for quick status:
- ✅ = Task complete
- ⏳ = In progress
- ⚠️ = Warning/issue
- ❌ = Failed
- 👀 = Monitoring

### 3. **Rich Media**
Departments can share:
- Code snippets (syntax highlighted)
- Graphs/charts (confidence over time)
- Tables (entity extraction results)
- Files (exported data)

### 4. **Voice Channels**
For real-time collaboration, departments could use Matrix voice channels for synchronous coordination.

## The Natural Conclusion

This architecture makes **Conway's Law visible**: your organizational structure (departments as bots) communicates through your communication structure (Matrix chat).

The result is a **collaborative AI team** that humans can observe, guide, and learn from. Every decision, every handoff, every escalation happens in natural language in a shared space.

**It's not just agents coordinating - it's a team having a conversation.**

---

*Generated: November 9, 2025*
*This is the ultimate expression of ChatOps + Departmental Architecture*
