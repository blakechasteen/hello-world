# Matrix.org Integration Architecture

**Promptly × Matrix: Full ChatOps Integration**

Turn any Matrix room into an AI reliability workspace with Promptly bot.

---

## Vision

**"Promptly in every Matrix room"**

Make AI reliability accessible via chat - no UI needed, just message `@promptly:matrix.org` in any Matrix room.

### What This Enables

1. **Chat-Native Workflow**
   - Message bot: "optimize my Q&A prompt"
   - Bot responds with optimized version
   - Iterate via conversation
   - No context switching

2. **Team Collaboration**
   - Shared prompt libraries in team rooms
   - Review/approve prompts via chat
   - Track experiments in threads
   - Async collaboration across timezones

3. **Federation Benefits**
   - Self-host bot on your infrastructure
   - Keep data in your homeserver
   - Bridge to Slack, Discord, Teams
   - True ownership

4. **Open Protocol**
   - Built on Matrix (open standard)
   - Interoperable with ecosystem
   - No vendor lock-in
   - Community extensions

---

## Architecture Overview

```
Matrix Client (Element, etc.)
    ↓ (Matrix Protocol)
Matrix Homeserver (synapse, dendrite, conduit)
    ↓ (Application Service API)
Promptly Matrix Bot
    ↓ (Internal API)
Promptly Core (HoloLoom + DSPy)
    ↓ (Results)
Matrix Room (formatted response)
```

### Key Components

1. **Promptly Matrix Bot** - Application Service bridging Matrix ↔ Promptly
2. **Command Parser** - Parse natural language commands
3. **Workflow Engine** - Execute multi-step processes
4. **State Manager** - Track conversation context
5. **Notification System** - Async updates (long-running tasks)

---

## Matrix Protocol Primer

### What is Matrix?

**Matrix** = Open standard for decentralized, encrypted communication
- Like email for chat (federated)
- End-to-end encrypted by default
- Real-time sync protocol
- Extensible with custom events

**Homeserver** = Your Matrix server (like email server)
- Stores your data
- Federates with other homeservers
- You control it (self-host or use matrix.org)

**Application Service** = Bot framework
- Registers namespaces (`@promptly_*`)
- Receives all messages in those namespaces
- Can impersonate users in namespace
- Bidirectional API with homeserver

### Why Matrix for Promptly?

**Perfect Fit**:
1. **Chat-Native AI** - Conversational interface for prompt engineering
2. **Team Collaboration** - Shared rooms, threads, reactions
3. **Privacy** - E2E encryption, self-hosted data
4. **Open Source** - Aligns with Promptly's open core model
5. **Federation** - Works across organizations
6. **Bridging** - Connects to Slack, Discord, Teams

---

## Use Cases

### 1. Solo Developer Workflow

**Scenario**: Developer optimizing prompts for their app

**Workflow**:
```
User: @promptly optimize
      Task: Answer customer support questions
      Examples: [paste 3 examples]

Bot:  Analyzing your task...
      Created signature: customer_support_qa
      Running DSPy optimization...
      [Progress bar: ████████░░ 80%]

Bot:  ✅ Optimization complete!

      Optimized prompt:
      "You are a helpful customer support agent..."

      Metrics:
      - Accuracy: 0.95 (+0.20 improvement)
      - Clarity: 0.92 (+0.15 improvement)
      - Avg response time: 850ms

      Try it: @promptly run customer_support_qa "How do I reset password?"
      Save it: @promptly save customer_support_qa
```

**Benefits**:
- No UI needed
- Immediate feedback
- Iterate via chat
- Save successful prompts

---

### 2. Team Code Review

**Scenario**: Team reviewing AI-generated code reviews

**Workflow**:
```
Alice: @promptly code-review [paste PR diff]

Bot:   Running code review workflow...
       [In thread]

       Step 1/5: Style analysis... ✅
       Step 2/5: Security scan... ⚠️  2 issues found
       Step 3/5: Performance check... ✅
       Step 4/5: Test coverage... ⚠️  Coverage: 65% (target: 80%)
       Step 5/5: Best practices... ✅

       Overall Score: 7.5/10

       Issues:
       1. SQL injection risk (line 42)
       2. Test coverage below threshold

       Suggestions:
       - Use parameterized queries
       - Add 3 test cases for error paths

       Approve? React with ✅ or ❌

Alice: [reacts with ❌]
Alice: @promptly fix issue 1

Bot:   Generating fix for SQL injection...

       Proposed change:
       ```python
       # Before
       query = f"SELECT * FROM users WHERE id={user_id}"

       # After
       query = "SELECT * FROM users WHERE id=%s"
       cursor.execute(query, (user_id,))
       ```

       Apply? React with ✅

Alice: [reacts with ✅]

Bot:   ✅ Fix applied to PR branch
       Re-running security scan...
       ✅ No security issues found

Bob:   Nice catch! @promptly approve

Bot:   ✅ Code review approved
       PR ready to merge
```

**Benefits**:
- Async review (Bob joined later)
- Threaded discussion
- Traceable decisions (reactions)
- Automated fixes

---

### 3. Enterprise Compliance

**Scenario**: Fortune 500 company with compliance requirements

**Workflow**:
```
ComplianceRoom (private, E2E encrypted):

Manager: @promptly_enterprise analyze-compliance
         Prompt: [paste customer-facing prompt]
         Policy: HIPAA

Bot:     Running compliance analysis...

         HIPAA Compliance Check:
         ✅ No PHI in prompt
         ✅ No external API calls
         ✅ Audit trail enabled
         ⚠️  Lacks explicit consent language

         Risk Level: MEDIUM

         Recommendation: Add consent disclaimer

         Suggested addition:
         "By using this service, you consent to..."

         Approve for production? (requires 2 approvals)

Manager: [reacts with ✅]

Compliance Officer: Reviewing...
                    @promptly show-audit-trail

Bot:                Audit Trail (last 30 days):
                    - 2024-11-01: Prompt v1 created by @alice
                    - 2024-11-05: Modified by @bob (added HIPAA check)
                    - 2024-11-07: Compliance review requested

                    All changes logged ✅

                    [reacts with ✅]

Bot:                ✅ Approved for production (2/2 approvals)
                    Deploying to production environment...
                    ✅ Deployed: prompt_v1_hipaa_compliant
```

**Benefits**:
- E2E encryption (sensitive data)
- Audit trail (compliance)
- Multi-approval workflow
- Traceable decisions

---

## Technical Design

### 1. Promptly Matrix Bot (Application Service)

**Technology Stack**:
- **Language**: Python 3.10+
- **Matrix SDK**: `matrix-nio` (async, E2E encryption support)
- **Framework**: FastAPI (for bot HTTP API)
- **Database**: PostgreSQL (state persistence)
- **Cache**: Redis (conversation context)

**Core Components**:

```python
# bot/promptly_matrix_bot.py

from nio import AsyncClient, MatrixRoom, RoomMessageText
from hololoom.promptly import DSPyHoloLoom
from hololoom.config import Config

class PromptlyMatrixBot:
    """Promptly bot for Matrix (Application Service)"""

    def __init__(self, homeserver_url: str, access_token: str):
        self.client = AsyncClient(homeserver_url, "@promptly:matrix.org")
        self.client.access_token = access_token
        self.promptly = DSPyHoloLoom(config=Config.fused())

        # Register callbacks
        self.client.add_event_callback(self.message_callback, RoomMessageText)

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming messages"""
        # Ignore own messages
        if event.sender == self.client.user_id:
            return

        # Check if bot is mentioned
        if not self.is_mentioned(event.body):
            return

        # Parse command
        command = self.parse_command(event.body)

        # Execute command
        response = await self.execute_command(command, room, event)

        # Send response
        await self.send_response(room.room_id, response, event.event_id)

    async def execute_command(self, command: dict, room: MatrixRoom, event: RoomMessageText):
        """Execute Promptly command"""
        cmd_type = command['type']

        if cmd_type == 'optimize':
            return await self.optimize_prompt(command, room)
        elif cmd_type == 'run':
            return await self.run_workflow(command, room)
        elif cmd_type == 'code-review':
            return await self.code_review(command, room)
        elif cmd_type == 'save':
            return await self.save_prompt(command, room)
        else:
            return {"error": f"Unknown command: {cmd_type}"}

    async def optimize_prompt(self, command: dict, room: MatrixRoom):
        """Optimize a prompt using DSPy"""
        # Extract task and examples
        task = command['task']
        examples = command['examples']

        # Send progress update
        await self.send_progress(room.room_id, "Analyzing task...")

        # Create signature
        signature = create_signature(task, inputs=["input"], outputs=["output"])

        await self.send_progress(room.room_id, "Running optimization...")

        # Optimize
        optimized = await self.promptly.optimize_from_memory(
            signature=signature,
            memory_query="optimization_examples"
        )

        # Format response
        return {
            "type": "optimization_result",
            "signature": signature,
            "optimized_program": optimized,
            "metrics": optimized.metrics
        }

    async def send_progress(self, room_id: str, message: str):
        """Send progress update to room"""
        await self.client.room_send(
            room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",  # Notice = bot status message
                "body": message
            }
        )

    async def send_response(self, room_id: str, response: dict, reply_to: str):
        """Send formatted response to room"""
        # Format as Matrix message
        formatted = self.format_response(response)

        # Send as threaded reply
        await self.client.room_send(
            room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": formatted['body'],
                "formatted_body": formatted['html'],
                "format": "org.matrix.custom.html",
                "m.relates_to": {
                    "m.in_reply_to": {
                        "event_id": reply_to
                    }
                }
            }
        )
```

---

### 2. Command Parser

**Natural Language → Structured Commands**

```python
# bot/command_parser.py

import re
from typing import Dict, List, Optional

class CommandParser:
    """Parse natural language commands for Promptly bot"""

    COMMANDS = {
        'optimize': r'@promptly optimize\s+Task:\s*(.+?)\s+Examples:\s*(\[.+\])',
        'run': r'@promptly run\s+(\w+)\s+"(.+)"',
        'code-review': r'@promptly code-review\s+(.+)',
        'save': r'@promptly save\s+(\w+)',
        'help': r'@promptly help',
    }

    def parse(self, message: str) -> Optional[Dict]:
        """Parse message into command"""
        for cmd_type, pattern in self.COMMANDS.items():
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return self.extract_command(cmd_type, match.groups())

        return None

    def extract_command(self, cmd_type: str, groups: tuple) -> Dict:
        """Extract command parameters"""
        if cmd_type == 'optimize':
            return {
                'type': 'optimize',
                'task': groups[0].strip(),
                'examples': self.parse_examples(groups[1])
            }
        elif cmd_type == 'run':
            return {
                'type': 'run',
                'workflow': groups[0],
                'input': groups[1]
            }
        elif cmd_type == 'code-review':
            return {
                'type': 'code-review',
                'code': groups[0]
            }
        elif cmd_type == 'save':
            return {
                'type': 'save',
                'name': groups[0]
            }
        elif cmd_type == 'help':
            return {'type': 'help'}

    def parse_examples(self, examples_str: str) -> List[Dict]:
        """Parse examples from string"""
        # Simple JSON parsing (can be improved)
        import json
        return json.loads(examples_str)
```

---

### 3. State Manager

**Track Conversation Context**

```python
# bot/state_manager.py

from typing import Dict, Optional
import redis
import json

class StateManager:
    """Manage conversation state across messages"""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour expiry

    def get_context(self, room_id: str, user_id: str) -> Optional[Dict]:
        """Get conversation context"""
        key = f"context:{room_id}:{user_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def set_context(self, room_id: str, user_id: str, context: Dict):
        """Set conversation context"""
        key = f"context:{room_id}:{user_id}"
        self.redis.setex(key, self.ttl, json.dumps(context))

    def append_message(self, room_id: str, user_id: str, message: str, response: str):
        """Append to conversation history"""
        context = self.get_context(room_id, user_id) or {"history": []}
        context['history'].append({
            "user": message,
            "bot": response,
            "timestamp": time.time()
        })
        self.set_context(room_id, user_id, context)

    def get_workflow_state(self, room_id: str, workflow_id: str) -> Optional[Dict]:
        """Get multi-step workflow state"""
        key = f"workflow:{room_id}:{workflow_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def set_workflow_state(self, room_id: str, workflow_id: str, state: Dict):
        """Set workflow state"""
        key = f"workflow:{room_id}:{workflow_id}"
        self.redis.setex(key, self.ttl * 24, json.dumps(state))  # 24h TTL
```

---

### 4. Response Formatter

**Format Results as Matrix Messages**

```python
# bot/response_formatter.py

from typing import Dict

class ResponseFormatter:
    """Format Promptly responses as Matrix messages"""

    def format_optimization_result(self, result: Dict) -> Dict:
        """Format optimization result"""
        # Plain text version
        body = f"""✅ Optimization complete!

Optimized prompt:
{result['optimized_program'].prompt}

Metrics:
- Accuracy: {result['metrics']['accuracy']:.2f}
- Clarity: {result['metrics']['clarity']:.2f}
- Avg response time: {result['metrics']['latency_ms']:.0f}ms

Try it: @promptly run {result['signature'].name} "your input"
Save it: @promptly save {result['signature'].name}
"""

        # HTML version (formatted)
        html = f"""<h3>✅ Optimization complete!</h3>

<p><strong>Optimized prompt:</strong></p>
<pre><code>{result['optimized_program'].prompt}</code></pre>

<p><strong>Metrics:</strong></p>
<ul>
<li>Accuracy: {result['metrics']['accuracy']:.2f}</li>
<li>Clarity: {result['metrics']['clarity']:.2f}</li>
<li>Avg response time: {result['metrics']['latency_ms']:.0f}ms</li>
</ul>

<p>
  <code>@promptly run {result['signature'].name} "your input"</code><br>
  <code>@promptly save {result['signature'].name}</code>
</p>
"""

        return {"body": body, "html": html}

    def format_code_review_result(self, result: Dict) -> Dict:
        """Format code review result"""
        # Build issue list
        issues_text = "\n".join([
            f"{i+1}. {issue['description']} (line {issue['line']})"
            for i, issue in enumerate(result['issues'])
        ])

        body = f"""Code Review Results

Overall Score: {result['score']}/10

Issues found: {len(result['issues'])}
{issues_text}

Suggestions:
{chr(10).join('- ' + s for s in result['suggestions'])}

Approve? React with ✅ or ❌
"""

        # HTML version with syntax highlighting
        html = f"""<h3>Code Review Results</h3>

<p><strong>Overall Score:</strong> {result['score']}/10</p>

<p><strong>Issues found:</strong> {len(result['issues'])}</p>
<ol>
{chr(10).join(f'<li>{issue["description"]} (line {issue["line"]})</li>' for issue in result['issues'])}
</ol>

<p><strong>Suggestions:</strong></p>
<ul>
{chr(10).join(f'<li>{s}</li>' for s in result['suggestions'])}
</ul>

<p>Approve? React with ✅ or ❌</p>
"""

        return {"body": body, "html": html}
```

---

### 5. Workflow Engine

**Multi-Step Workflows in Chat**

```python
# bot/workflow_engine.py

from typing import Dict, List, Callable
from enum import Enum

class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowStep:
    """Single workflow step"""
    def __init__(self, name: str, action: Callable, requires_approval: bool = False):
        self.name = name
        self.action = action
        self.requires_approval = requires_approval
        self.status = WorkflowStatus.PENDING
        self.result = None

class WorkflowEngine:
    """Execute multi-step workflows in Matrix rooms"""

    def __init__(self, bot, state_manager):
        self.bot = bot
        self.state = state_manager

    async def execute_workflow(
        self,
        workflow_id: str,
        steps: List[WorkflowStep],
        room_id: str,
        initial_input: Dict
    ):
        """Execute workflow with progress updates"""
        # Initialize workflow state
        self.state.set_workflow_state(room_id, workflow_id, {
            "status": WorkflowStatus.IN_PROGRESS.value,
            "current_step": 0,
            "steps": [s.name for s in steps]
        })

        context = initial_input

        for i, step in enumerate(steps):
            # Update progress
            await self.bot.send_progress(
                room_id,
                f"Step {i+1}/{len(steps)}: {step.name}..."
            )

            # Execute step
            try:
                step.status = WorkflowStatus.IN_PROGRESS
                step.result = await step.action(context)
                step.status = WorkflowStatus.COMPLETED

                # Update context for next step
                context.update(step.result)

                # Send step result
                await self.bot.send_response(
                    room_id,
                    self.format_step_result(step),
                    reply_to=None
                )

                # If requires approval, wait for reaction
                if step.requires_approval:
                    approved = await self.wait_for_approval(room_id)
                    if not approved:
                        await self.bot.send_message(
                            room_id,
                            "❌ Workflow cancelled by user"
                        )
                        return

            except Exception as e:
                step.status = WorkflowStatus.FAILED
                await self.bot.send_message(
                    room_id,
                    f"❌ Step {i+1} failed: {str(e)}"
                )
                return

        # Mark workflow complete
        self.state.set_workflow_state(room_id, workflow_id, {
            "status": WorkflowStatus.COMPLETED.value,
            "result": context
        })

        await self.bot.send_message(
            room_id,
            "✅ Workflow completed successfully!"
        )

    async def wait_for_approval(self, room_id: str) -> bool:
        """Wait for user approval (reaction)"""
        # Simplified - in production, use Matrix reaction events
        import asyncio
        await asyncio.sleep(1)  # Placeholder
        return True

    def format_step_result(self, step: WorkflowStep) -> Dict:
        """Format step result for display"""
        if step.status == WorkflowStatus.COMPLETED:
            return {
                "body": f"✅ {step.name} completed",
                "html": f"<p>✅ <strong>{step.name}</strong> completed</p>"
            }
        else:
            return {
                "body": f"⚠️ {step.name} in progress...",
                "html": f"<p>⚠️ <strong>{step.name}</strong> in progress...</p>"
            }
```

---

## Deployment Architecture

### Self-Hosted (Open Source)

```
User Infrastructure:
┌─────────────────────────────────────────────┐
│ Matrix Homeserver (synapse/dendrite)        │
│   ├─ Port 8008 (Client-Server API)         │
│   └─ Port 9000 (Application Service API)   │
└─────────────────────────────────────────────┘
            ↕ (HTTP/HTTPS)
┌─────────────────────────────────────────────┐
│ Promptly Matrix Bot                         │
│   ├─ Python 3.10+                           │
│   ├─ matrix-nio (Matrix SDK)                │
│   ├─ FastAPI (HTTP API)                     │
│   └─ PostgreSQL + Redis                     │
└─────────────────────────────────────────────┘
            ↕ (Internal)
┌─────────────────────────────────────────────┐
│ Promptly Core (HoloLoom + DSPy)             │
│   ├─ 6 Problem Solvers                      │
│   ├─ Memory System                          │
│   └─ Workflow Engine                        │
└─────────────────────────────────────────────┘
```

**Docker Compose Setup**:

```yaml
# docker-compose.yml

version: '3.8'

services:
  # Matrix Homeserver (Synapse)
  synapse:
    image: matrixdotorg/synapse:latest
    ports:
      - "8008:8008"
      - "9000:9000"
    volumes:
      - ./synapse:/data
    environment:
      - SYNAPSE_SERVER_NAME=matrix.example.com
      - SYNAPSE_REPORT_STATS=no

  # PostgreSQL (for Synapse + Promptly)
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=synapse
      - POSTGRES_PASSWORD=changeme
      - POSTGRES_DB=synapse
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # Redis (for Promptly state)
  redis:
    image: redis:7
    ports:
      - "6379:6379"

  # Promptly Matrix Bot
  promptly-bot:
    build: ./bot
    depends_on:
      - synapse
      - postgres
      - redis
    environment:
      - MATRIX_HOMESERVER_URL=http://synapse:8008
      - MATRIX_ACCESS_TOKEN=${MATRIX_BOT_TOKEN}
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://promptly:changeme@postgres:5432/promptly
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./bot:/app
      - ./HoloLoom:/HoloLoom

volumes:
  postgres-data:
  synapse-data:
```

---

### Hosted (Promptly Cloud)

```
Promptly Cloud Infrastructure:
┌──────────────────────────────────────────────┐
│ matrix.promptly.com (Hosted Homeserver)      │
│   ├─ High availability (3 replicas)         │
│   ├─ E2E encryption enabled                 │
│   └─ Federation enabled                      │
└──────────────────────────────────────────────┘
            ↕
┌──────────────────────────────────────────────┐
│ @promptly:matrix.promptly.com (Bot)          │
│   ├─ Auto-scaling (based on load)           │
│   ├─ Rate limiting                           │
│   └─ Multi-tenant isolation                  │
└──────────────────────────────────────────────┘
            ↕
┌──────────────────────────────────────────────┐
│ Promptly Core (HoloLoom + DSPy)              │
│   ├─ Kubernetes cluster                      │
│   ├─ GPU workers (for embeddings)            │
│   └─ Managed PostgreSQL + Redis              │
└──────────────────────────────────────────────┘
```

**Users can**:
- Use bot on matrix.promptly.com (hosted)
- OR self-host bot, federate with matrix.org
- OR run fully isolated (on-premise)

---

## Command Reference

### Core Commands

**1. Optimize Prompt**
```
@promptly optimize
Task: [describe task]
Examples: [
  {"input": "...", "output": "..."},
  {"input": "...", "output": "..."},
  {"input": "...", "output": "..."}
]
```

**2. Run Workflow**
```
@promptly run <workflow_name> "<input>"
```

**3. Code Review**
```
@promptly code-review [paste code or PR URL]
```

**4. Save Prompt**
```
@promptly save <name>
```

**5. List Saved Prompts**
```
@promptly list
```

**6. Schema Builder**
```
@promptly schema
Fields:
- name: string (required)
- age: number
- email: string (email format)
```

**7. Help**
```
@promptly help [command]
```

### Advanced Commands

**8. Multi-Pass Refinement**
```
@promptly refine
Strategy: elegance
Max iterations: 3
```

**9. Confidence Scoring**
```
@promptly verify "<statement>"
```

**10. Consistency Check**
```
@promptly consistency-check
Anchors: [key entities to preserve]
```

**11. Context Optimization**
```
@promptly optimize-context [long text]
Target tokens: 2000
```

---

## Integration with HoloLoom

### Memory Persistence

**Store conversation history in HoloLoom memory**:

```python
# bot/hololoom_integration.py

from hololoom import hololoom
from hololoom.documentation.types import MemoryShard

class HoloLoomIntegration:
    """Integrate Matrix bot with HoloLoom memory"""

    def __init__(self, config):
        self.loom = HoloLoom(config=config)

    async def store_conversation(self, room_id: str, messages: List[Dict]):
        """Store conversation in HoloLoom memory"""
        for msg in messages:
            shard = MemoryShard(
                content=f"User: {msg['user']}\nBot: {msg['bot']}",
                metadata={
                    "type": "conversation",
                    "room_id": room_id,
                    "timestamp": msg['timestamp']
                }
            )
            await self.loom.experience(shard)

    async def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from memory"""
        memories = await self.loom.recall(query)
        return "\n\n".join([m.content for m in memories])

    async def optimize_from_room_history(self, room_id: str, signature):
        """Optimize using conversation history as training data"""
        # Recall all conversations in this room
        memories = await self.loom.recall(f"room:{room_id}")

        # Convert to training examples
        examples = self.convert_to_examples(memories)

        # Optimize
        from hololoom.promptly import DSPyHoloLoom
        bridge = DSPyHoloLoom(config=self.loom.config)
        optimized = await bridge.optimize_from_memory(
            signature=signature,
            memory_query=f"room:{room_id}"
        )

        return optimized
```

---

## Roadmap

### Phase 1: Core Bot (Weeks 1-4)

**Week 1-2**: Basic Matrix integration
- [ ] Set up Matrix bot (Application Service)
- [ ] Implement command parser
- [ ] Basic commands (optimize, run, help)
- [ ] Response formatting
- [ ] Docker deployment

**Week 3-4**: HoloLoom integration
- [ ] Connect to Promptly Core
- [ ] Memory persistence
- [ ] Workflow execution
- [ ] State management

**Deliverables**:
- Working bot on matrix.org
- 3-5 core commands
- Self-hosting guide
- Basic documentation

---

### Phase 2: Team Features (Weeks 5-8)

**Week 5-6**: Collaboration
- [ ] Shared prompt libraries per room
- [ ] Approval workflows (reactions)
- [ ] Team roles (admin, reviewer, user)
- [ ] Room-based permissions

**Week 7-8**: Advanced workflows
- [ ] Multi-step workflows in threads
- [ ] Async notifications (long-running tasks)
- [ ] Progress bars and status updates
- [ ] Error handling and retries

**Deliverables**:
- Team collaboration features
- Approval workflows
- Advanced workflow engine

---

### Phase 3: Enterprise (Weeks 9-12)

**Week 9-10**: Compliance
- [ ] Audit trail (all commands logged)
- [ ] E2E encryption support
- [ ] RBAC (role-based access control)
- [ ] Compliance reports

**Week 11-12**: Production hardening
- [ ] Rate limiting
- [ ] Load balancing
- [ ] Monitoring and alerting
- [ ] High availability

**Deliverables**:
- Enterprise-ready bot
- Compliance features
- Production deployment guide

---

### Phase 4: Ecosystem (Weeks 13-16)

**Week 13-14**: Bridges
- [ ] Slack bridge (bidirectional)
- [ ] Discord bridge
- [ ] Microsoft Teams bridge
- [ ] Unified bot across platforms

**Week 15-16**: Extensions
- [ ] Custom command plugins
- [ ] Webhook integrations
- [ ] API for third-party tools
- [ ] Marketplace for extensions

**Deliverables**:
- Multi-platform support
- Plugin system
- Extension marketplace

---

## Business Model Integration

### Open Source (Free)

**What's included**:
- Core bot functionality
- Self-hosting tools
- Basic commands
- Community support

**Deploy yourself**:
```bash
git clone https://github.com/promptly/matrix-bot
docker-compose up -d
```

---

### Promptly Cloud ($49/user/month)

**What's included**:
- Hosted bot (@promptly:matrix.promptly.com)
- No infrastructure management
- Automatic updates
- Email support
- Team features (shared libraries, approvals)

**Getting started**:
1. Create account at promptly.com
2. Invite @promptly:matrix.promptly.com to your room
3. Start using immediately

---

### Promptly Enterprise (Custom pricing)

**What's included**:
- On-premise deployment
- SSO/SAML integration
- Advanced compliance (SOC2, HIPAA)
- Priority support (SLA)
- Custom integrations
- Dedicated account manager

**Contact**: enterprise@promptly.com

---

## Technical Challenges

### 1. E2E Encryption

**Challenge**: Matrix rooms can be E2E encrypted. Bot needs to decrypt messages.

**Solution**:
- Use `matrix-nio` (supports E2E encryption)
- Bot stores encryption keys in secure storage
- Supports `megolm` (Matrix E2E protocol)

**Code**:
```python
from nio import AsyncClient, encryption

client = AsyncClient(homeserver, user_id)

# Enable E2E encryption
client.encrypted = True

# Store keys in database
client.store_path = "./encryption_store"
```

---

### 2. Long-Running Tasks

**Challenge**: Optimization can take minutes. Matrix expects quick responses.

**Solution**:
- Send immediate acknowledgment ("Running optimization...")
- Execute task in background
- Send updates via additional messages
- Use threading for context

**Code**:
```python
async def optimize_prompt_async(self, room_id, command):
    """Long-running optimization with progress updates"""
    # Immediate ack
    await self.send_message(room_id, "Starting optimization...")

    # Background task
    task_id = str(uuid.uuid4())
    asyncio.create_task(self._optimize_background(room_id, command, task_id))

    return {"task_id": task_id}

async def _optimize_background(self, room_id, command, task_id):
    """Background optimization with progress updates"""
    try:
        # Progress: 25%
        await self.send_progress(room_id, "Analyzing examples... (25%)")

        # Progress: 50%
        await self.send_progress(room_id, "Running optimization... (50%)")

        # Progress: 75%
        await self.send_progress(room_id, "Evaluating results... (75%)")

        # Complete
        result = await self.promptly.optimize(command)
        await self.send_response(room_id, result)

    except Exception as e:
        await self.send_error(room_id, f"Optimization failed: {e}")
```

---

### 3. State Management

**Challenge**: Conversations span multiple messages. Need to track context.

**Solution**:
- Use Redis for session state (1h TTL)
- Store in Matrix room state events (persistent)
- Hybrid: Redis for active, room state for history

**Code**:
```python
# Store in room state
await client.room_put_state(
    room_id,
    event_type="com.promptly.context",
    content={"workflow_id": "abc123", "step": 2}
)

# Retrieve later
state = await client.room_get_state_event(
    room_id,
    event_type="com.promptly.context"
)
```

---

### 4. Rate Limiting

**Challenge**: Users could spam bot. Need rate limiting.

**Solution**:
- Per-user rate limits (10 commands/minute)
- Per-room rate limits (50 commands/minute)
- Enterprise: Custom limits

**Code**:
```python
from redis import Redis

class RateLimiter:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def check_limit(self, user_id: str, limit: int = 10) -> bool:
        """Check if user is within rate limit"""
        key = f"rate:{user_id}"
        count = self.redis.incr(key)

        if count == 1:
            # First request, set 1-minute expiry
            self.redis.expire(key, 60)

        return count <= limit
```

---

## Security Considerations

### 1. Authentication

- Bot uses Application Service token (shared secret with homeserver)
- Users authenticated via Matrix (no additional login)
- Per-room permissions managed via Matrix power levels

### 2. Authorization

- Commands checked against room power levels
- Admin commands require power level ≥ 50
- Sensitive operations require approval workflow

### 3. Data Privacy

- E2E encryption supported
- Bot can't read encrypted rooms without invite
- Self-hosting keeps all data on user infrastructure
- Hosted option: Data encrypted at rest, compliance certifications

### 4. Input Validation

- Sanitize all user inputs
- Prevent command injection
- Limit input sizes (prevent DoS)
- Validate examples before optimization

---

## Testing Strategy

### Unit Tests

```python
# tests/test_command_parser.py

import pytest
from bot.command_parser import CommandParser

def test_optimize_command():
    parser = CommandParser()
    cmd = parser.parse("""
        @promptly optimize
        Task: Answer questions
        Examples: [{"input": "test", "output": "result"}]
    """)

    assert cmd['type'] == 'optimize'
    assert cmd['task'] == "Answer questions"
    assert len(cmd['examples']) == 1

def test_run_command():
    parser = CommandParser()
    cmd = parser.parse('@promptly run qa_workflow "test input"')

    assert cmd['type'] == 'run'
    assert cmd['workflow'] == 'qa_workflow'
    assert cmd['input'] == 'test input'
```

### Integration Tests

```python
# tests/test_bot_integration.py

import pytest
from nio import AsyncClient
from bot.promptly_matrix_bot import PromptlyMatrixBot

@pytest.mark.asyncio
async def test_bot_responds_to_mention():
    """Test bot responds when mentioned"""
    bot = PromptlyMatrixBot(homeserver, token)

    # Simulate message
    response = await bot.handle_message(
        room_id="!test:matrix.org",
        sender="@alice:matrix.org",
        message="@promptly help"
    )

    assert response is not None
    assert "Available commands" in response['body']

@pytest.mark.asyncio
async def test_optimize_workflow():
    """Test end-to-end optimization"""
    bot = PromptlyMatrixBot(homeserver, token)

    response = await bot.handle_message(
        room_id="!test:matrix.org",
        sender="@alice:matrix.org",
        message="""
            @promptly optimize
            Task: QA
            Examples: [{"input": "Q1", "output": "A1"}]
        """
    )

    assert response['type'] == 'optimization_result'
    assert 'optimized_program' in response
```

---

## Documentation

### User Documentation

1. **Getting Started Guide**
   - How to invite bot to room
   - First command (@promptly help)
   - Example workflows

2. **Command Reference**
   - All commands with examples
   - Parameter descriptions
   - Error messages

3. **Best Practices**
   - Effective prompt engineering
   - Team collaboration patterns
   - Security recommendations

### Developer Documentation

1. **Self-Hosting Guide**
   - Docker setup
   - Configuration options
   - Troubleshooting

2. **API Reference**
   - Bot HTTP API
   - Webhook endpoints
   - Custom commands

3. **Contributing Guide**
   - Development setup
   - Testing
   - Pull request process

---

## Success Metrics

### Week 1-2 (MVP)
- [ ] Bot runs on matrix.org
- [ ] 5 core commands working
- [ ] 10+ test users
- [ ] Self-hosting guide complete

### Week 4 (Team Features)
- [ ] 50+ daily active users
- [ ] 5+ teams using collaboration features
- [ ] 100+ rooms with bot

### Week 8 (Enterprise)
- [ ] 2+ enterprise pilots
- [ ] E2E encryption working
- [ ] Compliance docs complete

### Week 12 (Ecosystem)
- [ ] 500+ daily active users
- [ ] Slack/Discord bridges working
- [ ] 10+ third-party integrations

---

## Next Steps

### Immediate (This Week)

1. **Set up Matrix homeserver** (Development)
   ```bash
   docker run -d -p 8008:8008 matrixdotorg/synapse:latest
   ```

2. **Create bot skeleton**
   ```bash
   mkdir promptly-matrix-bot
   cd promptly-matrix-bot
   pip install matrix-nio fastapi redis
   ```

3. **Implement basic command parser**
   - Parse @promptly mentions
   - Extract command type
   - Return structured data

4. **Test with Element client**
   - Create test room
   - Invite bot
   - Send test commands

### Short-Term (Week 1-2)

1. Connect to Promptly Core
2. Implement optimize command
3. Add response formatting
4. Deploy to matrix.org

### Medium-Term (Week 3-4)

1. Add team features
2. Implement workflows
3. Create self-hosting guide
4. Open source release

---

## Summary

**Matrix.org integration provides**:
- **Chat-native AI reliability** - No UI needed
- **Team collaboration** - Shared rooms, async workflows
- **True ownership** - Self-host, E2E encryption
- **Open protocol** - Interoperable, no lock-in
- **Perfect alignment** - Chat = natural interface for prompts

**Business model alignment**:
- **Open source**: Self-host bot for free
- **Promptly Cloud**: Hosted bot ($49/user/mo)
- **Enterprise**: On-premise, compliance, support

**This is the ChatOps layer Promptly needs** - making AI reliability accessible via conversation, not dashboards.

---

**Ready to build? Let's start with the bot skeleton!** 🚀
