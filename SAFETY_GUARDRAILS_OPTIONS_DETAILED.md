# Safety Guardrails: Detailed Options Analysis

## 🔍 Current Situation

### What's Happening

Your agentic system has **multi-layer safety guardrails** built into the policy engine. When a query comes in, this is the flow:

```
User Query
    ↓
Agentic Orchestrator
    ↓
Full Learning Engine
    ↓
Weaving Orchestrator
    ↓
Policy Engine (unified.py:decide())
    ↓
Safety Guardrails Evaluation
    ↓
SafetyPolicy.get_risk_level()
    ↓
Decision:
  - ActionCategory.EXECUTION
  - RiskLevel.HIGH
  - requires_approval: True (because EXECUTION category is HIGH risk)
    ↓
PermissionError raised! ❌
```

### Why EXECUTION Requires Approval

From `safety_guardrails.py:188-198`:

```python
class SafetyPolicy:
    def __init__(self):
        self.default_risk_levels = {
            ActionCategory.EXECUTION: RiskLevel.HIGH,  # ← Tool execution = HIGH risk
            ActionCategory.DELETION: RiskLevel.HIGH,   # ← Also requires approval
            ActionCategory.SYSTEM: RiskLevel.CRITICAL, # ← Always blocked
            # ... lower risk categories ...
        }
```

And from lines 200-204:
```python
        # Actions that always require approval
        self.approval_required: Set[ActionCategory] = {
            ActionCategory.DELETION,
            ActionCategory.SYSTEM,
        }
```

**However**, the current code also requires approval for **HIGH** risk level, not just DELETION/SYSTEM.

From `unified.py:724-731`:
```python
if guardrail_decision.requires_approval:
    logger.warning(
        "Policy guardrails require approval for tool '%s': %s",
        tool_name,
        guardrail_decision.reason,
    )
    raise PermissionError(
        f"Tool selection requires approval: {guardrail_decision.reason}"
    )
```

---

## 📋 Option A: Testing Mode (Recommended for Development)

### Philosophy
**"Bypass approval gates while keeping all safety logging and monitoring active."**

This lets you:
- ✅ See all safety decisions in logs
- ✅ Track what would have required approval
- ✅ Test functionality without manual approval workflow
- ❌ But still get auditing and visibility

### Implementation (3 Steps)

#### Step 1: Add Config Flag

Add to `HoloLoom/config.py` (~line 200, after phase 5 settings):

```python
    # Layer 6: Safety Guardrails (optional)
    enable_safety_guardrails: bool = True  # Enable safety system
    safety_testing_mode: bool = False      # Bypass approval requirements for testing
    safety_log_decisions: bool = True      # Log all safety decisions
```

#### Step 2: Update SafetyPolicy

Modify `HoloLoom/alignment/safety_guardrails.py:178-208`:

```python
class SafetyPolicy:
    """
    Defines safety policies for different action categories.

    Configurable risk thresholds and approval requirements.
    """

    def __init__(self, testing_mode: bool = False):  # ← Add parameter
        """
        Initialize with default policies.

        Args:
            testing_mode: If True, bypass approval requirements (for testing only)
        """
        self.testing_mode = testing_mode  # ← Store flag

        # Default risk levels by action category
        self.default_risk_levels = {
            ActionCategory.QUERY: RiskLevel.SAFE,
            ActionCategory.RETRIEVAL: RiskLevel.SAFE,
            ActionCategory.ANALYSIS: RiskLevel.LOW,
            ActionCategory.STORAGE: RiskLevel.LOW,
            ActionCategory.MODIFICATION: RiskLevel.MEDIUM,
            ActionCategory.DELETION: RiskLevel.HIGH,
            ActionCategory.EXECUTION: RiskLevel.HIGH,
            ActionCategory.SYSTEM: RiskLevel.CRITICAL,
            ActionCategory.EXTERNAL: RiskLevel.HIGH,
        }

        # Actions that always require approval (unless testing_mode)
        if testing_mode:
            self.approval_required: Set[ActionCategory] = set()  # ← Empty in testing mode!
        else:
            self.approval_required: Set[ActionCategory] = {
                ActionCategory.DELETION,
                ActionCategory.SYSTEM,
            }

        # Custom risk evaluators (can be registered)
        self.custom_evaluators: List[Callable[[ActionRequest], Optional[RiskLevel]]] = []

    def requires_approval(self, category: ActionCategory, risk_level: RiskLevel) -> bool:
        """
        Check if action requires approval.

        Args:
            category: Action category
            risk_level: Evaluated risk level

        Returns:
            True if approval required
        """
        if self.testing_mode:
            return False  # ← Never require approval in testing mode

        # Check if category always requires approval
        if category in self.approval_required:
            return True

        # High risk also requires approval (in production)
        if risk_level == RiskLevel.HIGH:
            return True

        return False
```

#### Step 3: Update SafetyGuardrails Constructor

Modify `HoloLoom/alignment/safety_guardrails.py:287-302`:

```python
    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        enable_adversarial_detection: bool = True,
        testing_mode: bool = False,  # ← Add parameter
    ):
        """
        Initialize safety guardrails.

        Args:
            policy: Safety policy (uses default if None)
            enable_adversarial_detection: Whether to detect adversarial inputs
            testing_mode: If True, bypass approval requirements (testing only)
        """
        self.testing_mode = testing_mode  # ← Store
        self.policy = policy or SafetyPolicy(testing_mode=testing_mode)  # ← Pass to policy
        self.adversarial_detector = AdversarialDetector() if enable_adversarial_detection else None
        self.action_history: List[ActionRequest] = []
        self._setup_logging()
```

#### Step 4: Update Policy Creation

Modify `HoloLoom/policy/unified.py` where SafetyGuardrails is created (around line 600-650):

```python
    def __init__(self, cfg, emb, ...):
        # ... existing code ...

        # Initialize safety guardrails
        self.safety = SafetyGuardrails(
            testing_mode=cfg.safety_testing_mode  # ← Read from config
        )
```

#### Step 5: Enable in Server Config

Modify `HoloLoom/server/agentic_api_integrated.py:118-122`:

```python
async def _init_config():
    """Initialize configuration."""
    config = Config.fast()
    config.safety_testing_mode = True  # ← ENABLE TESTING MODE
    return config
```

### Result

```bash
$ python start_agentic_server.py
# Server starts...

$ curl -X POST http://localhost:8001/query -d '{"text": "What is Thompson Sampling?"}'

# Logs show:
INFO: Safety decision: action=policy_select_tool, category=execution,
      risk=high, allowed=True, requires_approval=False  # ← False!
INFO: Tool selected: answer
INFO: Generating LLM response...
✅ Success! {"response": "Thompson Sampling is...", "confidence": 0.87}
```

**Pros:**
- ✅ Quick to implement (5 small changes)
- ✅ All safety logic still runs (logging, risk assessment)
- ✅ Easy to toggle on/off
- ✅ Clear intent (testing_mode flag)

**Cons:**
- ⚠️ Must remember to disable in production
- ⚠️ No approval workflow practiced in development

---

## 📋 Option B: Implement Approval Workflow

### Philosophy
**"Build a complete approval system with human-in-the-loop for high-risk actions."**

This is the **production-ready** approach where you:
- ✅ Practice the real production workflow
- ✅ Build approval UI
- ✅ Test with actual safety gates

### Implementation (More Complex)

#### Step 1: Add Approval State Manager

Create `HoloLoom/alignment/approval_manager.py`:

```python
"""
Approval Manager
================
Manages pending approvals for high-risk actions.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from datetime import datetime
import uuid

@dataclass
class PendingApproval:
    """A pending approval request."""
    approval_id: str
    action: str
    category: str
    risk_level: str
    context: Dict
    requested_at: datetime = field(default_factory=datetime.now)
    approved: Optional[bool] = None
    approved_at: Optional[datetime] = None
    approver: Optional[str] = None

class ApprovalManager:
    """
    Manages approval workflow for high-risk actions.

    Supports:
    - Auto-approval for whitelisted users/contexts
    - Async approval with timeout
    - Approval history and auditing
    """

    def __init__(
        self,
        auto_approve_categories: set = None,
        approval_timeout_seconds: float = 300.0  # 5 minutes
    ):
        self.auto_approve_categories = auto_approve_categories or set()
        self.approval_timeout = approval_timeout_seconds
        self.pending: Dict[str, PendingApproval] = {}
        self.history: list[PendingApproval] = []

    async def request_approval(
        self,
        action: str,
        category: str,
        risk_level: str,
        context: Dict
    ) -> bool:
        """
        Request approval for an action.

        Args:
            action: Action name
            category: Action category
            risk_level: Risk level
            context: Action context

        Returns:
            True if approved, False if denied or timeout
        """
        # Check auto-approve
        if category in self.auto_approve_categories:
            return True

        # Create pending approval
        approval_id = str(uuid.uuid4())
        pending = PendingApproval(
            approval_id=approval_id,
            action=action,
            category=category,
            risk_level=risk_level,
            context=context
        )
        self.pending[approval_id] = pending

        # Wait for approval (with timeout)
        try:
            result = await asyncio.wait_for(
                self._wait_for_decision(approval_id),
                timeout=self.approval_timeout
            )
            return result
        except asyncio.TimeoutError:
            # Timeout = denial
            pending.approved = False
            self.history.append(pending)
            del self.pending[approval_id]
            return False

    async def _wait_for_decision(self, approval_id: str) -> bool:
        """Poll for approval decision."""
        while approval_id in self.pending:
            pending = self.pending[approval_id]
            if pending.approved is not None:
                # Decision made
                self.history.append(pending)
                del self.pending[approval_id]
                return pending.approved
            await asyncio.sleep(0.5)  # Poll every 500ms
        return False

    def approve(self, approval_id: str, approver: str = "system"):
        """Approve a pending request."""
        if approval_id in self.pending:
            self.pending[approval_id].approved = True
            self.pending[approval_id].approved_at = datetime.now()
            self.pending[approval_id].approver = approver

    def deny(self, approval_id: str, approver: str = "system"):
        """Deny a pending request."""
        if approval_id in self.pending:
            self.pending[approval_id].approved = False
            self.pending[approval_id].approved_at = datetime.now()
            self.pending[approval_id].approver = approver

    def get_pending(self) -> list[PendingApproval]:
        """Get all pending approvals."""
        return list(self.pending.values())
```

#### Step 2: Add API Endpoints

Add to `HoloLoom/server/agentic_api_integrated.py`:

```python
# Add to server state
class ServerState:
    # ... existing fields ...
    approval_manager: Optional[ApprovalManager] = None

# Initialize in startup
@app.on_event("startup")
async def startup():
    # ... existing init ...

    # Initialize approval manager
    state.approval_manager = ApprovalManager(
        auto_approve_categories={"analysis"}  # Auto-approve analysis actions
    )

# Add approval endpoints
@app.get("/approvals/pending")
async def get_pending_approvals():
    """Get all pending approvals."""
    if not state.approval_manager:
        raise HTTPException(500, "Approval manager not initialized")

    pending = state.approval_manager.get_pending()
    return {
        "pending": [
            {
                "approval_id": p.approval_id,
                "action": p.action,
                "category": p.category,
                "risk_level": p.risk_level,
                "requested_at": p.requested_at.isoformat(),
                "context": p.context
            }
            for p in pending
        ]
    }

@app.post("/approvals/{approval_id}/approve")
async def approve_action(approval_id: str, approver: str = "user"):
    """Approve a pending action."""
    if not state.approval_manager:
        raise HTTPException(500, "Approval manager not initialized")

    state.approval_manager.approve(approval_id, approver)
    return {"status": "approved", "approval_id": approval_id}

@app.post("/approvals/{approval_id}/deny")
async def deny_action(approval_id: str, approver: str = "user"):
    """Deny a pending action."""
    if not state.approval_manager:
        raise HTTPException(500, "Approval manager not initialized")

    state.approval_manager.deny(approval_id, approver)
    return {"status": "denied", "approval_id": approval_id}
```

#### Step 3: Update Policy to Use Approval Manager

Modify `HoloLoom/policy/unified.py:720-732`:

```python
            if guardrail_decision.requires_approval:
                logger.warning(
                    "Policy guardrails require approval for tool '%s': %s",
                    tool_name,
                    guardrail_decision.reason,
                )

                # Request approval through manager (if available)
                if hasattr(self, 'approval_manager') and self.approval_manager:
                    approved = await self.approval_manager.request_approval(
                        action=tool_name,
                        category=guardrail_decision.metadata.get("category", "unknown"),
                        risk_level=guardrail_decision.risk_level.value,
                        context={"query": query.text, "tool": tool_name}
                    )

                    if not approved:
                        raise PermissionError(
                            f"Tool selection approval denied or timed out: {guardrail_decision.reason}"
                        )
                    # If approved, continue...
                else:
                    # No approval manager = raise error (production safety)
                    raise PermissionError(
                        f"Tool selection requires approval: {guardrail_decision.reason}"
                    )
```

#### Step 4: Add UI for Approvals

Update `ui/agentic_learner_ui.py` to add approval panel:

```python
def get_pending_approvals():
    """Fetch pending approvals from server."""
    try:
        response = requests.get(f"{SERVER_URL}/approvals/pending", timeout=2)
        if response.status_code == 200:
            return response.json()["pending"]
    except:
        pass
    return []

def approve_action(approval_id: str):
    """Approve an action."""
    try:
        response = requests.post(
            f"{SERVER_URL}/approvals/{approval_id}/approve",
            json={"approver": "ui_user"},
            timeout=2
        )
        return response.status_code == 200
    except:
        return False

# Add to Gradio interface
with gr.Blocks() as demo:
    # ... existing interface ...

    with gr.Tab("Approvals"):
        gr.Markdown("### Pending Approvals")
        pending_list = gr.DataFrame(
            headers=["ID", "Action", "Category", "Risk", "Requested"],
            label="Pending Actions"
        )

        with gr.Row():
            approval_id_input = gr.Textbox(label="Approval ID")
            approve_btn = gr.Button("✅ Approve", variant="primary")
            deny_btn = gr.Button("❌ Deny", variant="stop")

        approval_result = gr.Textbox(label="Result", interactive=False)

        def handle_approve(approval_id):
            if approve_action(approval_id):
                return f"✅ Approved: {approval_id}"
            return f"❌ Failed to approve: {approval_id}"

        approve_btn.click(
            handle_approve,
            inputs=[approval_id_input],
            outputs=[approval_result]
        )
```

### Result

```bash
# Terminal 1: Start server
$ python start_agentic_server.py
INFO: Approval manager initialized

# Terminal 2: Send query
$ curl -X POST http://localhost:8001/query -d '{"text": "What is Thompson Sampling?"}'
INFO: Tool 'answer' requires approval
INFO: Approval request created: abc-123-def

# Query waits... (timeout: 5 minutes)

# Terminal 3: Check pending
$ curl http://localhost:8001/approvals/pending
{
  "pending": [
    {
      "approval_id": "abc-123-def",
      "action": "answer",
      "category": "execution",
      "risk_level": "high",
      "requested_at": "2025-11-02T00:30:00"
    }
  ]
}

# Terminal 3: Approve
$ curl -X POST http://localhost:8001/approvals/abc-123-def/approve
{"status": "approved"}

# Terminal 2: Query completes!
✅ {"response": "Thompson Sampling is...", "confidence": 0.87}
```

**Pros:**
- ✅ Full production workflow
- ✅ Human oversight for high-risk actions
- ✅ Audit trail of all approvals
- ✅ Can be automated (auto-approve by category/user/context)

**Cons:**
- ⚠️ Complex to implement (~200 lines)
- ⚠️ Slower development iteration (manual approvals)
- ⚠️ Requires UI or API client for approvals

---

## 📋 Option C: Hybrid Approach (Best of Both)

### Philosophy
**"Smart defaults with environment-based overrides."**

Combine both approaches:
- **Development**: Auto-approve everything
- **Staging**: Auto-approve some, require approval for risky
- **Production**: Require approval for all high-risk

### Implementation

```python
# In config.py
class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Config:
    # ... existing fields ...

    environment: Environment = Environment.DEVELOPMENT

    # Safety settings derived from environment
    @property
    def safety_testing_mode(self) -> bool:
        """Auto-approve in development."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def safety_auto_approve_categories(self) -> set:
        """Categories to auto-approve by environment."""
        if self.environment == Environment.DEVELOPMENT:
            return {"analysis", "execution", "query", "retrieval"}  # All safe categories
        elif self.environment == Environment.STAGING:
            return {"analysis", "query", "retrieval"}  # Most, but not execution
        else:  # PRODUCTION
            return set()  # None

# Set via environment variable
import os
env = os.getenv("HOLOLOOM_ENV", "development")
config.environment = Environment(env)
```

Then run:

```bash
# Development (auto-approve everything)
$ HOLOLOOM_ENV=development python start_agentic_server.py

# Staging (require approval for execution)
$ HOLOLOOM_ENV=staging python start_agentic_server.py

# Production (require approval for all high-risk)
$ HOLOLOOM_ENV=production python start_agentic_server.py
```

---

## 🎯 Recommendation

For **your current situation** (getting the demo working):

1. **Immediate**: Use **Option A** (Testing Mode)
   - Quickest path to working queries
   - Keeps all safety logging active
   - 5 small changes, ~30 lines of code

2. **Short-term**: Upgrade to **Option C** (Hybrid)
   - Set environment=DEVELOPMENT for now
   - Plan for staging/production later
   - Best long-term architecture

3. **Long-term**: Implement **Option B** (Full Approval Workflow)
   - Build approval UI in Gradio
   - Practice production workflow
   - Complete safety system

---

## 🚀 Next Steps

**To get your system working TODAY:**

1. I can implement **Option A** for you right now (5 file changes)
2. Test query: `"What is Thompson Sampling?"`
3. See it work! ✅
4. Then iterate on Option C/B for production

**Would you like me to implement Option A now?** It will take ~2 minutes.
