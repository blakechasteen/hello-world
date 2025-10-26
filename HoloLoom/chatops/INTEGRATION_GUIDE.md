# HoloLoom ChatOps Integration Guide

Complete guide for deploying and using the full ChatOps system with all Phase 1-4 features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Feature Integration](#feature-integration)
4. [Usage Examples](#usage-examples)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tuning](#performance-tuning)

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install matrix-nio Pillow pytesseract psutil rank-bm25

# Optional but recommended
pip install sentence-transformers networkx scipy
```

### Minimal Setup

```python
#!/usr/bin/env python3
"""
Minimal ChatOps deployment with all features enabled.
"""
import asyncio
from HoloLoom.chatops.matrix_bot import MatrixBot, MatrixBotConfig
from HoloLoom.chatops.chatops_bridge import ChatOpsOrchestrator
from HoloLoom.chatops.multimodal_handler import MultiModalHandler
from HoloLoom.chatops.thread_handler import ThreadHandler
from HoloLoom.chatops.proactive_agent import ProactiveAgent
from HoloLoom.chatops.performance_optimizer import PerformanceOptimizer
from HoloLoom.chatops.custom_commands import CustomCommandManager
from HoloLoom.chatops.innovative_features import WorkflowEngine, IncidentManager, ChatDashboard
from HoloLoom.chatops.advanced_chatops import CodeReviewAssistant, ContextAwareAgent, KnowledgeMiner

async def main():
    # 1. Configure Matrix bot
    bot_config = MatrixBotConfig(
        homeserver="https://matrix.org",
        user_id="@yourbot:matrix.org",
        access_token="YOUR_ACCESS_TOKEN",
        command_prefix="!"
    )

    # 2. Initialize all components
    bot = MatrixBot(bot_config)
    orchestrator = ChatOpsOrchestrator()
    multimodal = MultiModalHandler()
    threads = ThreadHandler(max_depth=5)
    proactive = ProactiveAgent()
    optimizer = PerformanceOptimizer(cache_size=1000, cache_ttl=3600)
    commands = CustomCommandManager()

    # Phase 4 components
    workflow_engine = WorkflowEngine()
    incident_manager = IncidentManager()
    dashboard = ChatDashboard()
    code_reviewer = CodeReviewAssistant()
    context_agent = ContextAwareAgent()
    knowledge_miner = KnowledgeMiner()

    # 3. Register custom commands
    @commands.command(
        name="workflow",
        description="Execute a workflow",
        params=[{"name": "workflow_name", "type": "str", "required": True}],
        category="automation"
    )
    async def workflow_handler(ctx, workflow_name):
        execution = await workflow_engine.execute(workflow_name, {})
        return f"✅ Workflow '{workflow_name}' executed: {execution.status}"

    @commands.command(
        name="incident",
        description="Report an incident",
        params=[
            {"name": "title", "type": "str", "required": True},
            {"name": "severity", "type": "str", "choices": ["low", "medium", "high", "critical"]}
        ],
        category="operations"
    )
    async def incident_handler(ctx, title, severity="medium"):
        incident = await incident_manager.create_incident(title, severity, ctx.user_id)
        return f"🚨 Incident created: {incident.incident_id} (severity: {severity})"

    @commands.command(
        name="dashboard",
        description="Show real-time dashboard",
        params=[{"name": "dashboard_name", "type": "str", "required": True}],
        category="monitoring"
    )
    async def dashboard_handler(ctx, dashboard_name):
        # Fetch metrics (placeholder)
        metrics = {"cpu": 45.2, "memory": 67.8, "requests": 1234}
        rendered = await dashboard.render(dashboard_name, metrics)
        return f"```\n{rendered}\n```"

    @commands.command(
        name="review",
        description="Review a pull request",
        params=[{"name": "pr_number", "type": "int", "required": True}],
        category="development"
    )
    async def review_handler(ctx, pr_number):
        review = await code_reviewer.review_pr(pr_number, ctx.conversation_id)
        return review

    # 4. Integrate with bot message handler
    async def enhanced_message_handler(room, event, message_text):
        conversation_id = room.room_id

        # Thread tracking
        parent_id = getattr(event, 'relates_to', {}).get('event_id')
        thread_context = threads.process_message(
            message_id=event.event_id,
            text=message_text,
            sender=event.sender,
            conversation_id=conversation_id,
            parent_id=parent_id
        )

        # Context detection
        detected_context = context_agent.detect_context(message_text)
        if detected_context:
            context_msg = await context_agent.switch_context(
                detected_context,
                conversation_id
            )
            await bot.send_message(room.room_id, context_msg)

        # Check for custom commands
        if message_text.startswith("!"):
            command_parts = message_text[1:].split()
            command_name = command_parts[0]

            try:
                from HoloLoom.chatops.custom_commands import CommandContext
                ctx = CommandContext(
                    user_id=event.sender,
                    conversation_id=conversation_id,
                    message_id=event.event_id,
                    is_admin=(event.sender in bot_config.admin_users)
                )

                result = await commands.execute(command_name, ctx, *command_parts[1:])
                await bot.send_message(room.room_id, result)
                return
            except ValueError as e:
                # Command not found, fall through to regular processing
                pass

        # Regular orchestrator processing with caching
        @optimizer.cache(ttl=300)
        async def process_cached(query):
            return await orchestrator.handle_message(room, event, query)

        with optimizer.profile("message_processing"):
            response = await process_cached(message_text)

        # Proactive suggestions
        messages = [{"text": message_text, "sender": event.sender}]
        suggestions = await proactive.process_messages(messages, conversation_id)

        if suggestions:
            suggestions_text = "\n\n💡 **Suggestions:**\n" + "\n".join(
                f"• {s['suggestion']}" for s in suggestions
            )
            response += suggestions_text

        # Knowledge mining
        insights = knowledge_miner.mine(messages)

        await bot.send_message(room.room_id, response)

    # 5. Attach handler to bot
    bot.message_callback = enhanced_message_handler

    # 6. Handle multimodal events
    async def image_handler(room, event):
        result = await multimodal.process_image(event, room)
        await bot.send_message(room.room_id, f"📷 {result}")

    async def file_handler(room, event):
        result = await multimodal.process_file(event, room)
        await bot.send_message(room.room_id, f"📎 {result}")

    bot.image_callback = image_handler
    bot.file_callback = file_handler

    # 7. Start bot
    print("🚀 Starting HoloLoom ChatOps with all features...")
    print(f"📊 Performance optimizer: {optimizer.get_statistics()}")
    print(f"⚙️  Custom commands: {len(commands.list_commands())}")

    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Matrix credentials
MATRIX_HOMESERVER=https://matrix.org
MATRIX_USER_ID=@yourbot:matrix.org
MATRIX_ACCESS_TOKEN=your_access_token_here

# Admin users (comma-separated)
MATRIX_ADMIN_USERS=@admin:matrix.org,@ops:matrix.org

# Performance tuning
CACHE_SIZE=1000
CACHE_TTL=3600
ENABLE_PROFILING=true

# Feature flags
ENABLE_MULTIMODAL=true
ENABLE_THREADS=true
ENABLE_PROACTIVE=true
ENABLE_WORKFLOWS=true
ENABLE_INCIDENTS=true

# Pattern detection thresholds
PATTERN_DECISION_THRESHOLD=0.7
PATTERN_ACTION_THRESHOLD=0.6
PATTERN_QUESTION_THRESHOLD=0.5
PATTERN_URGENT_THRESHOLD=0.8
```

### YAML Configuration

Create `chatops_config.yaml`:

```yaml
matrix:
  homeserver: "https://matrix.org"
  user_id: "@yourbot:matrix.org"
  access_token: "YOUR_ACCESS_TOKEN"
  command_prefix: "!"
  admin_users:
    - "@admin:matrix.org"

hololoom:
  config_mode: "fast"  # bare, fast, or fused
  enable_kg_storage: true

features:
  multimodal:
    enabled: true
    storage_path: "./chatops_data/media"
    max_file_size_mb: 10

  threads:
    enabled: true
    max_depth: 5
    context_window: 10

  proactive:
    enabled: true
    suggestion_cooldown: 300
    patterns:
      decisions:
        threshold: 0.7
      action_items:
        threshold: 0.6
      questions:
        threshold: 0.5

  performance:
    cache_size: 1000
    cache_ttl: 3600
    enable_profiling: true

  workflows:
    enabled: true
    max_concurrent: 5
    approval_timeout: 3600

  incidents:
    enabled: true
    auto_remediation: true
    escalation_delay: 1800

  dashboards:
    enabled: true
    refresh_rate: 60
```

---

## Feature Integration

### 1. Workflow Automation

```python
from HoloLoom.chatops.innovative_features import WorkflowEngine, WorkflowStep

# Create workflow engine
engine = WorkflowEngine()

# Define deployment workflow
deploy_workflow = [
    WorkflowStep(
        step_id="build",
        type="action",
        name="Build application",
        config={"command": "npm run build"}
    ),
    WorkflowStep(
        step_id="test",
        type="action",
        name="Run tests",
        config={"command": "npm test"}
    ),
    WorkflowStep(
        step_id="approval",
        type="approval",
        name="Production approval",
        config={"approvers": ["@admin:matrix.org"], "timeout": 3600}
    ),
    WorkflowStep(
        step_id="deploy",
        type="action",
        name="Deploy to production",
        config={"command": "./deploy.sh production"}
    )
]

workflow = engine.create_workflow("deploy_to_prod", deploy_workflow)

# Execute in chat
# User: !workflow deploy_to_prod
# Bot: 🔄 Workflow 'deploy_to_prod' started (ID: exec_123)
#      Step 1/4: Build application ✅
#      Step 2/4: Run tests ✅
#      Step 3/4: Production approval ⏳ Waiting for approval...
```

### 2. Incident Management

```python
from HoloLoom.chatops.innovative_features import IncidentManager

manager = IncidentManager()

# Auto-detect incidents from monitoring alerts
async def handle_alert(alert_data):
    incident = await manager.detect_incident(alert_data)

    if incident:
        # Auto-remediation
        remediation = await manager.auto_remediate(incident)

        # Notify team in Matrix room
        notification = f"""
🚨 **Incident Detected**
ID: {incident.incident_id}
Severity: {incident.severity}
Title: {incident.title}

🔧 **Auto-Remediation**
Actions taken: {', '.join(remediation.actions_taken)}
Status: {remediation.status}
"""
        await bot.send_message(incident_room, notification)

# Generate postmortem
# User: !incident postmortem INC-20231201-001
postmortem = manager.generate_postmortem(incident_id)
```

### 3. Real-Time Dashboards

```python
from HoloLoom.chatops.innovative_features import ChatDashboard, DashboardWidget

dashboard = ChatDashboard()

# Create system health dashboard
dashboard.create_dashboard("system_health", [
    DashboardWidget(
        widget_id="cpu",
        type="gauge",
        title="CPU Usage",
        config={"min": 0, "max": 100, "unit": "%", "warning": 70, "critical": 90}
    ),
    DashboardWidget(
        widget_id="memory",
        type="gauge",
        title="Memory Usage",
        config={"min": 0, "max": 100, "unit": "%", "warning": 80, "critical": 95}
    ),
    DashboardWidget(
        widget_id="requests",
        type="chart",
        title="Request Rate",
        config={"height": 5, "width": 40}
    )
])

# Update and render
# User: !dashboard system_health
metrics = {
    "cpu": 45.2,
    "memory": 67.8,
    "requests": [120, 135, 142, 138, 150, 165, 158]
}
rendered = await dashboard.render("system_health", metrics)
```

### 4. Code Review Assistant

```python
from HoloLoom.chatops.advanced_chatops import CodeReviewAssistant

reviewer = CodeReviewAssistant(github_token="YOUR_GITHUB_TOKEN")

# Review PR in chat
# User: !review 123
# Bot: 📝 **PR#123: Add authentication middleware**
#      Author: @alice
#      Files: 3 changed (+142/-28)
#
#      🔍 **AI Analysis:**
#        ✅ Tests included
#        ⚠️  Missing error handling in auth.js:42
#        💡 Consider adding rate limiting

# Add comments
# User: !comment 123 auth.js:42 Add error handling for null tokens
await reviewer.add_comment(123, "auth.js:42", "Add error handling for null tokens", "@bob")

# Approve and merge
# User: !approve 123
# User: !merge 123
```

### 5. Context-Aware Agent

```python
from HoloLoom.chatops.advanced_chatops import ContextAwareAgent

agent = ContextAwareAgent()

# Automatic context detection
# User: "Let's discuss the authentication feature"
# Bot: 📂 **Switched to: Project Auth**
#      Type: project
#
#      **Active Topics:**
#      • JWT implementation
#      • OAuth integration
#
#      What would you like to know?

# User: "What about the API performance?"
# Bot: 🔄 **Switching to: Project API**
#      Would you like me to:
#      - Show recent metrics
#      - Check for incidents
#      - Review open PRs

# Manual context switch
# User: !context project_auth
context_msg = await agent.switch_context("project_auth", room.room_id)
```

### 6. Knowledge Mining

```python
from HoloLoom.chatops.advanced_chatops import KnowledgeMiner

miner = KnowledgeMiner()

# Mine conversations automatically
messages = [
    {"text": "We decided to use JWT for authentication", "sender": "@alice"},
    {"text": "Best practice: always validate input on the server side", "sender": "@bob"},
    {"text": "The API performance looks good after optimization", "sender": "@alice"}
]

insights = miner.mine(messages)

# Extract decisions
for decision in insights["decisions"]:
    print(f"Decision by {decision['author']}: {decision['text']}")

# Find topic experts
api_experts = miner.get_experts("api", limit=5)
# ["@alice", "@bob", ...]

# User: !experts authentication
# Bot: 🏆 **Top Experts - Authentication**
#      1. @alice (23 mentions)
#      2. @bob (18 mentions)
#      3. @charlie (12 mentions)
```

---

## Usage Examples

### Example 1: Automated Deployment with Approval

```
# In Matrix chat room

User: !workflow deploy_to_prod version=v2.1.0

Bot: 🔄 Workflow 'deploy_to_prod' started (ID: exec_456)

     ✅ Step 1/4: Build application (completed in 45s)
     ✅ Step 2/4: Run tests (completed in 120s)
     ⏳ Step 3/4: Production approval
        Waiting for approval from: @admin:matrix.org

        [Approve] [Reject]

Admin: !approve exec_456

Bot: ✅ Approval granted by @admin:matrix.org

     🚀 Step 4/4: Deploy to production (in progress...)
     ✅ Step 4/4: Deploy to production (completed in 30s)

     🎉 Workflow 'deploy_to_prod' completed successfully!
     Duration: 3m 35s
```

### Example 2: Incident Response

```
# Auto-detected from monitoring alert

Bot: 🚨 **Incident Detected**
     ID: INC-20231201-003
     Severity: HIGH
     Title: API response time spike (>2s)
     Affected service: api-gateway

     🔧 **Auto-Remediation in Progress**
     1. Scaling up api-gateway instances... ✅
     2. Clearing cache... ✅
     3. Checking database connections... ✅

     ✅ Remediation successful
     Response time: 2.1s → 0.3s

     👥 Incident assigned to: @ops-team

     [View Details] [Escalate] [Resolve]

User: !incident resolve INC-20231201-003

Bot: ✅ Incident INC-20231201-003 resolved
     Resolution time: 3m 12s

     📄 Postmortem generated: /incidents/INC-20231201-003/postmortem.md
```

### Example 3: Multi-Modal Analysis

```
# User uploads screenshot of error

User: [uploads error_screenshot.png]

Bot: 📷 **Image Analysis**
     Extracted text:
     "Error: Connection refused
      at port 5432
      PostgreSQL database unavailable"

     🔍 **Related Issues:**
     • Similar error 2 days ago (INC-20231129-001)
     • Database connection pool exhausted

     💡 **Suggestions:**
     • Check database connection pool settings
     • Verify database service status
     • Review recent configuration changes

     Would you like me to:
     [Create Incident] [Check DB Status] [View Related Issues]
```

### Example 4: Thread-Aware Discussion

```
User: We need to improve the authentication flow

Bot: 📂 **Switched to: Project Auth**

     Based on recent discussions:
     • Open PR#123: JWT middleware
     • 2 open issues related to auth
     • Last deployment: 3 days ago

User: → What are the current issues?     # Reply in thread

Bot: → **Authentication Issues:**        # Thread-aware response
      1. #456: Session timeout too aggressive
      2. #478: OAuth callback redirect broken

      Context: This thread started with improving auth flow
      Previous messages in thread: 1

User: → Let's prioritize #478

Bot: → ✅ Priority updated: #478 → HIGH

      📋 **Action Items Detected:**
      • Fix OAuth callback redirect (#478)
      • Assigned to: (needs assignment)

      Would you like me to assign this to someone?
```

---

## Troubleshooting

### Common Issues

#### 1. Bot Not Responding

```bash
# Check bot connection
python -c "from HoloLoom.chatops.verify_deployment import verify_matrix_connection; verify_matrix_connection()"

# Check logs
tail -f logs/chatops.log

# Verify access token
curl -H "Authorization: Bearer YOUR_TOKEN" https://matrix.org/_matrix/client/r0/account/whoami
```

#### 2. Slow Response Times

```python
# Check performance statistics
stats = optimizer.get_statistics()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.1%}")
print(f"Avg response time: {stats['profiling']['operations']['message_processing']['avg_ms']:.2f}ms")

# Adjust cache settings
optimizer = PerformanceOptimizer(
    cache_size=2000,      # Increase cache size
    cache_ttl=7200,       # Increase TTL
    enable_profiling=True
)
```

#### 3. Pattern Detection False Positives

```python
# Tune thresholds
from HoloLoom.chatops.pattern_tuning import PatternTuner

tuner = PatternTuner()

# Increase threshold to reduce false positives
tuner.set_threshold("decisions", 0.8)     # Was 0.7
tuner.set_threshold("action_items", 0.7)  # Was 0.6

# Test against sample messages
test_messages = ["Let's do this", "We decided to use React", "Maybe we should try Vue"]
for msg in test_messages:
    matches, confidence = tuner.detect(msg, "decisions", return_confidence=True)
    print(f"{msg}: {confidence:.2f} ({'MATCH' if matches else 'NO MATCH'})")
```

#### 4. Workflow Execution Failures

```python
# Check workflow status
execution = workflow_engine.get_execution(execution_id)
print(f"Status: {execution.status}")
print(f"Current step: {execution.current_step}")
print(f"Failed step: {execution.failed_step}")
print(f"Error: {execution.error}")

# Retry from failed step
await workflow_engine.retry_execution(execution_id, from_step=execution.failed_step)
```

---

## Performance Tuning

### Cache Optimization

```python
# Monitor cache effectiveness
cache_stats = optimizer.cache.get_stats()

print(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
print(f"Hit rate: {cache_stats['hit_rate']:.1%}")
print(f"Total requests: {cache_stats['total_requests']}")

# Optimal settings based on workload
if cache_stats['hit_rate'] < 0.5:
    # Low hit rate - increase TTL
    optimizer = PerformanceOptimizer(cache_ttl=7200)
elif cache_stats['size'] == cache_stats['max_size']:
    # Cache full - increase size
    optimizer = PerformanceOptimizer(cache_size=2000)
```

### Query Deduplication

```python
# Use deduplication for expensive operations
async def expensive_search(query):
    return await orchestrator.search(query)

# Deduplicate concurrent requests
result = await optimizer.deduplicate(
    key=f"search:{query}",
    func=expensive_search,
    query=query
)
```

### Profiling

```python
# Profile specific operations
with optimizer.profile("knowledge_graph_query"):
    results = await kg.query(conversation_id)

# Get profiling statistics
stats = optimizer.profiler.get_stats("knowledge_graph_query")
print(f"Average time: {stats['avg_ms']:.2f}ms")
print(f"Min/Max: {stats['min_ms']:.2f}ms / {stats['max_ms']:.2f}ms")
print(f"Total calls: {stats['count']}")
```

### Resource Monitoring

```python
# Monitor resource usage
resources = optimizer.get_resource_usage()

print(f"Memory: {resources['memory_mb']:.1f}MB")
print(f"CPU: {resources['cpu_percent']:.1f}%")
print(f"Threads: {resources['threads']}")
print(f"Open files: {resources['open_files']}")

# Alert if resources high
if resources['memory_mb'] > 500:
    print("⚠️  High memory usage - consider restarting")
```

---

## Next Steps

1. **Deploy to Test Environment**
   ```bash
   ./deploy_test.sh
   python verify_deployment.py
   ```

2. **Gather User Feedback**
   - Invite beta testers to Matrix room
   - Monitor usage patterns
   - Collect feature requests

3. **Tune for Production**
   - Adjust cache sizes based on load
   - Fine-tune pattern detection thresholds
   - Optimize workflow definitions

4. **Expand Integrations**
   - Connect to CI/CD pipelines
   - Integrate monitoring systems
   - Add custom workflows for your team

5. **Advanced Features** (Future)
   - ML-based code review
   - Predictive incident detection
   - Advanced analytics dashboards

---

## Support

- **Documentation**: See [INNOVATIVE_CHATOPS.md](INNOVATIVE_CHATOPS.md)
- **Examples**: Check `HoloLoom/chatops/examples/`
- **Issues**: Report bugs in your issue tracker
- **Community**: Join the Matrix room for support

---

**Status**: ✅ All features integrated and tested
**Version**: 1.0.0
**Last Updated**: 2024
