# Promptly Matrix Bot - Quick Start Guide

**Get started with Promptly in 5 minutes!**

---

## 🚀 Installation

### Option 1: Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/promptly/matrix-bot
cd matrix-bot

# 2. Configure environment
cp .env.example .env
nano .env  # Add your API keys

# 3. Start services
docker-compose up -d

# 4. Check logs
docker-compose logs -f promptly-bot
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Redis (optional - uses in-memory if unavailable)
docker run -d -p 6379:6379 redis:7-alpine

# 4. Configure environment
export MATRIX_HOMESERVER=https://matrix.org
export MATRIX_USER_ID=@promptly:matrix.org
export MATRIX_BOT_PASSWORD=your_password
export OPENAI_API_KEY=sk-your-key
export REDIS_URL=redis://localhost:6379

# 5. Run bot
python -m bot.promptly_bot
```

---

## 💬 Using the Bot

### Step 1: Invite Bot to Room

In your Matrix client (Element, FluffyChat, etc.):

```
/invite @promptly:matrix.org
```

Bot will auto-join and send welcome message.

### Step 2: Try Basic Commands

```
@promptly help
```

You'll see all available commands.

---

## 📝 Common Use Cases

### Use Case 1: Optimize a Prompt

**Goal**: Improve a prompt using examples

```
@promptly optimize
Task: Answer customer support questions
Examples: [
  {"input": "How do I reset my password?", "output": "Click Settings > Security > Reset Password. Follow the email instructions."},
  {"input": "Where is my order?", "output": "Check the Orders page in your account. Your order #12345 is in transit."}
]
```

**Bot Response**:
```
✅ Optimization complete!

**Optimized Prompt:**
You are a helpful customer support agent. Answer questions
clearly and provide step-by-step instructions when needed.

**Metrics:**
• Overall Score: 0.92
• Accuracy: 0.95
• Clarity: 0.90

**Next Steps:**
• Test: @promptly run <workflow> "test input"
• Save: @promptly save customer_support_v1
```

### Use Case 2: Save and Reuse Prompts

```
# Save the optimized prompt
@promptly save customer_support_v1

# List saved prompts
@promptly list

# Run a saved prompt
@promptly run customer_support_v1 "How do I change my email?"
```

### Use Case 3: Code Security Review

**Goal**: Check code for vulnerabilities before committing

```
@promptly code-review
```python
def login(username, password):
    query = f"SELECT * FROM users WHERE user='{username}' AND pass='{password}'"
    return db.execute(query).fetchone()
```
```

**Bot Response**:
```
🔴 Code Review Complete

**Language:** python
**Risk Score:** 6.0/10 (CRITICAL)

**Issues Found:**
• Critical: 1
• High: 0

**Top Issues:**

1. 🔴 **Potential SQL injection**
   Line 2: String concatenation in SQL query
   → Use parameterized queries
   CWE-89

⚠️ **Action Required:** Fix critical issues before deployment.
```

**Fix and Re-Check**:
```
@promptly code-review
```python
def login(username, password):
    query = "SELECT * FROM users WHERE user=? AND pass=?"
    return db.execute(query, (username, password)).fetchone()
```
```

**Bot Response**:
```
🟢 Code Review Complete

**Risk Score:** 0.5/10 (LOW)

✅ **Good to go!** No critical issues found.
```

---

## 👥 Team Features (Phase 2)

### Feature 1: Approval Workflows

**When to use**: High-risk actions that need team approval

**Example**: Deploy prompt to production

```python
# In your code
from bot.approval_workflow import get_approval_manager, ActionRisk

manager = get_approval_manager(state, client)

request = await manager.request_approval(
    room_id="!room:matrix.org",
    initiator="@alice:matrix.org",
    action="deploy_prompt",
    context={"prompt_name": "customer_support_v2"},
    risk_level=ActionRisk.HIGH  # Requires 2 approvals
)
```

**In Matrix**:
```
Bot: 🔔 **Approval Required** 🟠

Action: Deploy prompt 'customer_support_v2'
Requested by: @alice
Risk Level: HIGH
Approvals needed: 2

React with ✅/❌
```

Team members react with ✅ or ❌. After 2 approvals:

```
Bot: ✅ **Approved!**
Approvers: @bob, @charlie
Executing now...
```

### Feature 2: Multi-Step Workflows

**When to use**: Complex pipelines with multiple stages

**Example**: Complete deployment pipeline

```python
from bot.workflow_templates import create_deploy_prompt_workflow
from bot.workflow_engine import get_workflow_engine

# Create workflow
workflow = create_deploy_prompt_workflow(
    prompt_name="customer_support_v2",
    task="Answer customer support questions",
    examples=[
        {"input": "reset password?", "output": "Click Settings..."},
        {"input": "track order?", "output": "Check Orders page..."}
    ],
    room_id="!room:matrix.org",
    initiator="@alice:matrix.org"
)

# Execute
engine = get_workflow_engine(client, state, promptly_core)
result = await engine.execute(workflow)
```

**In Matrix** (real-time updates):
```
Bot: 🚀 Workflow Started
     Name: Deploy Prompt: customer_support_v2
     Steps: 4

Bot: ⏳ Step 'optimize' starting...
Bot: ✅ Step 'optimize' completed (3.2s)

Bot: ⏳ Step 'test' starting...
Bot: ✅ Step 'test' completed (0.5s)

Bot: 🔔 Approval Required 🟠
     [Team approves via reactions]

Bot: ✅ Step 'approval' completed (120.0s)

Bot: ⏳ Step 'deploy' starting...
Bot: ✅ Step 'deploy' completed (1.0s)

Bot: ✅ Workflow Complete
     Status: success
     Steps: 4/4
     Duration: 124.7s
```

---

## 🎯 Pre-Built Workflow Templates

### Template 1: Deploy Prompt

**Use case**: Optimize → Test → Approve → Deploy

```python
from bot.workflow_templates import create_deploy_prompt_workflow

workflow = create_deploy_prompt_workflow(
    prompt_name="my_prompt",
    task="Your task",
    examples=[...],
    room_id="!room:matrix.org",
    initiator="@user:matrix.org"
)
```

### Template 2: Code Review with Approval

**Use case**: Security scan → Conditional approval if risky

```python
from bot.workflow_templates import create_code_review_workflow

workflow = create_code_review_workflow(
    code=your_code,
    language="python",
    require_approval=True,  # If risk > 5.0
    room_id="!room:matrix.org",
    initiator="@user:matrix.org"
)
```

### Template 3: Testing Pipeline

**Use case**: Unit → Integration → E2E → Report

```python
from bot.workflow_templates import create_testing_workflow

workflow = create_testing_workflow(
    test_suites=["unit", "integration", "e2e"],
    room_id="!room:matrix.org",
    initiator="@user:matrix.org"
)
```

### Template 4: Emergency Rollback

**Use case**: Fast rollback with critical approval

```python
from bot.workflow_templates import create_rollback_workflow

workflow = create_rollback_workflow(
    deployment_id="deploy_123",
    room_id="!room:matrix.org",
    initiator="@ops:matrix.org"
)
```

### Template 5: Multi-Step Optimization

**Use case**: Optimize → Validate → Refine if needed → Approve

```python
from bot.workflow_templates import create_multi_step_optimization_workflow

workflow = create_multi_step_optimization_workflow(
    task="Classify support tickets",
    initial_examples=training_data,
    validation_examples=validation_data,
    room_id="!room:matrix.org",
    initiator="@ml_engineer:matrix.org"
)
```

---

## 🔧 Configuration

### Environment Variables

**Required**:
- `MATRIX_HOMESERVER` - Matrix server URL (default: https://matrix.org)
- `MATRIX_USER_ID` - Bot user ID (e.g., @promptly:matrix.org)
- `MATRIX_BOT_PASSWORD` - Bot password
- `OPENAI_API_KEY` - OpenAI API key for DSPy

**Optional**:
- `REDIS_URL` - Redis connection URL (default: in-memory)
- `LOG_LEVEL` - Logging level (default: INFO)
- `PROMPTLY_CONFIG` - Execution mode (default: fused)

### Risk Levels

Approval workflow risk levels:

| Risk Level | Approvals | Timeout | Use Case |
|-----------|-----------|---------|----------|
| **LOW** | 0 | None | Safe operations |
| **MEDIUM** | 1 | 24h | Standard changes |
| **HIGH** | 2 | 48h | Production deploys |
| **CRITICAL** | 3+ | 72h | Emergency changes |

Configure in code:
```python
from bot.approval_workflow import ActionRisk

risk_level=ActionRisk.HIGH  # Requires 2 approvals
```

---

## 📚 Advanced Usage

### Custom Workflow Steps

Define custom step handlers:

```python
from bot.workflow_engine import get_workflow_engine, WorkflowStep

engine = get_workflow_engine(client, state, promptly_core)

# Register custom handler
async def my_custom_step(step: WorkflowStep, workflow):
    # Your custom logic
    result = await my_function(step.params)
    return result

engine.register_handler('my_custom_type', my_custom_step)

# Use in workflow
workflow = Workflow(
    steps=[
        WorkflowStep(
            name="custom_step",
            type="my_custom_type",
            params={"foo": "bar"}
        )
    ]
)
```

### Error Handling Strategies

Per-step error handling:

```python
WorkflowStep(
    name="risky_step",
    type="run",
    on_error="retry",     # fail | skip | retry | rollback
    retry_count=3,        # Number of retries
    retry_delay=5         # Delay in seconds (exponential backoff)
)
```

### Conditional Steps

Execute steps only if condition met:

```python
WorkflowStep(
    name="conditional_step",
    type="custom",
    condition="previous_step['accuracy'] > 0.8",  # Python expression
    depends_on=["previous_step"]
)
```

### Parallel Execution

Steps without dependencies run in parallel:

```python
steps = [
    WorkflowStep(name="step1", type="run"),
    WorkflowStep(name="step2", type="run"),  # Runs parallel with step1
    WorkflowStep(name="step3", type="run", depends_on=["step1", "step2"])  # Waits for both
]
```

---

## 🐛 Troubleshooting

### Bot doesn't respond

**Check**:
1. Bot has joined room: `/invite @promptly:matrix.org`
2. Mention bot correctly: `@promptly` (lowercase)
3. Check logs: `docker-compose logs -f promptly-bot`

**Fix**:
```bash
# Restart bot
docker-compose restart promptly-bot
```

### Code review not working

**Check**:
1. Code block formatting:
   ```
   @promptly code-review
   ```python
   your code here
   ```
   ```
2. Language specified or detectable

**Test**:
```bash
python bot/code_reviewer.py
```

### Approval workflow not triggering

**Check**:
1. Bot client initialized: `approval_manager = get_approval_manager(state, client)`
2. Room ID and initiator provided
3. Reaction handling enabled

**Test**:
```bash
python bot/approval_workflow.py
```

### Workflow engine errors

**Check**:
1. Step dependencies form valid DAG (no cycles)
2. Required handlers registered
3. Promptly Core available for AI steps

**Test**:
```bash
python bot/workflow_engine.py
```

---

## 📖 Additional Resources

- **Full Documentation**: [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)
- **Phase 2 Features**: [PHASE_2_COMPLETE.md](PHASE_2_COMPLETE.md)
- **Architecture**: [MATRIX_INTEGRATION_ARCHITECTURE.md](MATRIX_INTEGRATION_ARCHITECTURE.md)
- **GitHub Issues**: https://github.com/promptly/matrix-bot/issues
- **Matrix Room**: #promptly:matrix.org

---

## 🎓 Learning Path

### Beginner (Day 1)
1. ✅ Install bot (Docker or local)
2. ✅ Try `@promptly help`
3. ✅ Optimize a simple prompt
4. ✅ Save and run prompts

### Intermediate (Week 1)
1. ✅ Code review integration
2. ✅ Basic approval workflows
3. ✅ Use workflow templates
4. ✅ Team collaboration

### Advanced (Month 1)
1. ✅ Custom workflow steps
2. ✅ Complex pipelines
3. ✅ Error handling strategies
4. ✅ Production deployment

---

## ✨ Tips & Best Practices

### Prompt Optimization
- **Start small**: 2-3 examples initially
- **Iterate**: Add more examples if accuracy < 0.8
- **Test**: Always validate on held-out data
- **Save versions**: Track prompt evolution

### Code Review
- **Run before commit**: Catch issues early
- **Fix critical first**: Address 🔴 issues immediately
- **Track CWE IDs**: For security compliance
- **Re-check after fixes**: Verify vulnerabilities resolved

### Approval Workflows
- **Match risk to approvals**: Critical = 3+ approvals
- **Set reasonable timeouts**: 24-72 hours based on urgency
- **Document context**: Help approvers make decisions
- **Track rejections**: Learn from what didn't pass

### Workflow Design
- **Keep steps focused**: One task per step
- **Use dependencies**: Ensure correct execution order
- **Add error handling**: Retry transient failures
- **Monitor progress**: Watch Matrix notifications

---

## 🚀 Next Steps

You're ready to use Promptly Matrix Bot!

**Start simple**:
1. Optimize one prompt
2. Review one piece of code
3. Request one approval
4. Run one workflow

**Scale up**:
1. Add more team members
2. Create custom workflows
3. Integrate with CI/CD
4. Deploy to production

---

**Welcome to chat-native AI reliability!** 💬🤖
