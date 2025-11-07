# Using Workflow Builder for CRM Pipelines & Lead Scoring

**Visual drag-and-drop interface for building CRM automation workflows**

---

## 🎯 Why Use Workflow Builder for CRM?

The HoloLoom Workflow Builder is **perfect** for CRM because:

✅ **Visual pipeline design** - See your entire sales process at a glance
✅ **Automated lead scoring** - Multi-step evaluation workflows
✅ **Conditional routing** - Route hot/warm/cold leads differently
✅ **Safety checks** - Ensure data quality before actions
✅ **Real-time execution** - See workflows run in real-time
✅ **Reusable workflows** - Save and share pipeline templates
✅ **No code required** - Drag, drop, connect, done!

---

## 🚀 Quick Start

### Step 1: Start the Backend

```bash
cd HoloLoom/web_dashboard
python workflow_executor.py
```

Server starts at http://localhost:8001

### Step 2: Open the Builder

Open `workflow_builder.html` in your browser

### Step 3: Build Your First CRM Workflow

See examples below!

---

## 📊 CRM Workflow Examples

### Example 1: Simple Lead Scoring Pipeline

**What it does**: Evaluate a contact and assign hot/warm/cold score

**Workflow:**
```
[Memory Search: Get Contact]
    ↓
[Context Retriever: Get Activities]
    ↓
[Synthesizer: Extract Engagement Signals]
    ↓
[Convergence Engine: Calculate Score]
    ↓
[Conditional Branch: Route by Score]
    ├─► [Hot Lead] → [Response: Send Proposal]
    ├─► [Warm Lead] → [Response: Schedule Follow-up]
    └─► [Cold Lead] → [Response: Add to Nurture Campaign]
```

**How to Build:**

1. **Drag** "Memory Search" onto canvas
   - Set query: "contact:{contact_name}"
   - Set max_results: 1

2. **Drag** "Context Retriever" below it
   - Connect Memory Search → Context Retriever
   - Set query: "activities with {contact_name}"

3. **Drag** "Synthesizer" below
   - Connect Context Retriever → Synthesizer
   - Enable extract_entities, extract_motifs

4. **Drag** "Convergence Engine"
   - Connect Synthesizer → Convergence
   - Set strategy: "bayesian_blend"

5. **Drag** "Conditional Branch"
   - Connect Convergence → Conditional
   - Set condition: "score >= 0.75" (hot)
   - Add branches for warm and cold

6. **Add Response Generators** for each branch
   - Hot: "Send proposal immediately"
   - Warm: "Schedule follow-up call"
   - Cold: "Add to nurture email sequence"

**Result**: Automatic lead routing based on engagement!

---

### Example 2: Deal Pipeline Automation

**What it does**: Monitor deals and take action based on stage

**Workflow:**
```
[Memory Search: Get All Deals]
    ↓
[Loop Iterator: For Each Deal]
    ↓
    [Context Retriever: Get Deal Details]
        ↓
    [Synthesizer: Extract Stage + Value]
        ↓
    [Conditional Branch: Check Stage]
        ├─► [Stage: Proposal] → [Safety Check] → [Send Contract]
        ├─► [Stage: Negotiation] → [Check Duration] → [Escalate if Stale]
        └─► [Stage: Closing] → [Notify Sales Manager]
```

**How to Build:**

1. **Memory Search** - Get all deals
   - Query: "type:deal AND stage:active"

2. **Loop Iterator** - Process each deal
   - Connect Memory Search → Loop Iterator
   - Set max_iterations: 100

3. **Inside Loop**:
   - **Context Retriever** - Get deal details
   - **Synthesizer** - Extract stage, value, duration
   - **Conditional Branch** - Route by stage

4. **Stage-Specific Actions**:
   - Proposal → Safety check → Auto-send contract
   - Negotiation → Check if stale (>30 days)
   - Closing → Notify manager

**Result**: Automated pipeline management!

---

### Example 3: Multi-Factor Lead Scoring

**What it does**: Score leads using multiple signals (recency, activity, sentiment, value)

**Workflow:**
```
[Memory Search: Get Contact]
    ↓
[Parallel Executor]
    ├─► [Context: Recent Activities] → [Synthesizer: Activity Score]
    ├─► [Context: Deal Value] → [Synthesizer: Value Score]
    ├─► [Context: Email Opens] → [Synthesizer: Engagement Score]
    └─► [Context: Last Contact] → [Synthesizer: Recency Score]
    ↓
[Knowledge Fusion: Combine Signals]
    ↓
[Thompson Sampler: Weighted Score]
    ↓
[Convergence Engine: Final Classification]
    ↓
[Memory Store: Save Lead Score]
```

**How to Build:**

1. **Memory Search** - Get contact
   - Query: "contact:{email}"

2. **Parallel Executor** - Run 4 parallel scorers
   - Connect Memory Search → Parallel Executor
   - Add 4 parallel branches

3. **Each Branch** scores different factor:
   - Branch 1: Activity frequency
   - Branch 2: Deal pipeline value
   - Branch 3: Email engagement
   - Branch 4: Last contact recency

4. **Knowledge Fusion** - Combine all signals
   - Connect all branches → Knowledge Fusion
   - Set max_depth: 2

5. **Thompson Sampler** - Weighted scoring
   - Exploration rate: 0.1 (mostly exploit learned weights)

6. **Convergence Engine** - Final score
   - Strategy: bayesian_blend

7. **Memory Store** - Save result
   - Backend: hybrid
   - Update contact record

**Result**: Sophisticated multi-factor scoring!

---

### Example 4: Contact Enrichment Pipeline

**What it does**: Enrich contact data from multiple sources

**Workflow:**
```
[Memory Search: Get Contact]
    ↓
[Parallel Executor]
    ├─► [Memory Search: Company Info]
    ├─► [Memory Search: Industry Data]
    ├─► [Memory Search: Social Profiles]
    └─► [Memory Search: Past Interactions]
    ↓
[Knowledge Fusion: Merge All Data]
    ↓
[Synthesizer: Extract Key Insights]
    ↓
[Matryoshka Embedder: Generate Rich Embedding]
    ↓
[Memory Store: Update Contact Record]
```

**How to Build:**

1. **Memory Search** - Base contact
2. **Parallel Executor** - 4 enrichment sources
3. **Knowledge Fusion** - Merge with max_depth: 3
4. **Synthesizer** - Extract entities, motifs
5. **Matryoshka Embedder** - Create rich semantic representation
6. **Memory Store** - Update contact

**Result**: Comprehensive contact profiles!

---

### Example 5: Daily Action Prioritization

**What it does**: Generate prioritized daily action list

**Workflow:**
```
[Memory Search: Get All Active Contacts]
    ↓
[Loop Iterator: For Each Contact]
    ↓
    [Multi-Query: Break Down Analysis]
        ├─► "What's this contact's engagement level?"
        ├─► "What stage are their deals in?"
        └─► "When was last contact?"
        ↓
    [HoloLoom Query: Full Context Analysis]
        ↓
    [Convergence Engine: Calculate Priority]
        ↓
    [Conditional Branch: Priority Level]
        ├─► [High] → Add to "Today's Actions"
        ├─► [Medium] → Add to "This Week"
        └─► [Low] → Add to "This Month"
    ↓
[Knowledge Fusion: Sort by Priority]
    ↓
[Response Generator: Format Daily Action List]
```

**How to Build:**

1. **Memory Search** - Active contacts (last 30 days)
2. **Loop Iterator** - Process each
3. **Multi-Query** - Break down into 3 sub-questions
4. **HoloLoom Query** - Full analysis with pattern: "fused"
5. **Convergence Engine** - Priority score
6. **Conditional Branch** - Route by priority
7. **Knowledge Fusion** - Sort results
8. **Response Generator** - Format as email/dashboard

**Result**: Automated daily action planning!

---

### Example 6: Deal Forecasting Pipeline

**What it does**: Predict deal close probability and revenue

**Workflow:**
```
[Memory Search: Get Deal]
    ↓
[Context Retriever: Get Historical Data]
    ↓
[Parallel Executor]
    ├─► [Synthesizer: Activity Volume]
    ├─► [Synthesizer: Sentiment Analysis]
    ├─► [Synthesizer: Stage Duration]
    └─► [Synthesizer: Contact Quality]
    ↓
[Thompson Sampler: Probabilistic Forecast]
    ↓
[Convergence Engine: Close Probability]
    ↓
[Conditional Branch: Confidence Level]
    ├─► [High Confidence] → [Add to Forecast]
    ├─► [Medium Confidence] → [Flag for Review]
    └─► [Low Confidence] → [Request More Data]
```

**How to Build:**

1. **Memory Search** - Deal details
2. **Context Retriever** - Historical similar deals
3. **Parallel Executor** - 4 signal extractors
4. **Thompson Sampler** - Probabilistic model
5. **Convergence Engine** - Final probability
6. **Conditional** - Route by confidence

**Result**: Data-driven forecasting!

---

## 🎨 Agent Reference for CRM

### Essential CRM Agents

| Agent | CRM Use Case | Configuration |
|-------|-------------|---------------|
| **Memory Search** | Find contacts, deals, activities | query, max_results |
| **Context Retriever** | Get related data (activities for contact) | k, use_fusion |
| **Knowledge Fusion** | Expand context (contact → company → industry) | max_depth, min_importance |
| **Synthesizer** | Extract signals (engagement, sentiment) | extract_entities, extract_motifs |
| **Thompson Sampler** | Adaptive scoring (learn what signals matter) | exploration_rate |
| **Convergence Engine** | Final classification (hot/warm/cold) | strategy |
| **Conditional Branch** | Route by score/stage | condition |
| **Loop Iterator** | Process all contacts/deals | max_iterations |
| **Parallel Executor** | Multi-factor scoring | - |
| **Memory Store** | Update records | backend |

---

## 💡 CRM Workflow Patterns

### Pattern 1: Search → Score → Route

```
[Search for Entity] → [Calculate Score] → [Route by Score]
```

**Use Cases:**
- Lead scoring → Hot/warm/cold routing
- Deal prioritization → Urgent/normal/low
- Contact segmentation → Industry/size/stage

### Pattern 2: Parallel Analysis → Fusion → Decision

```
[Parallel Signals] → [Fusion] → [Decision]
```

**Use Cases:**
- Multi-factor lead scoring
- Comprehensive contact enrichment
- Deal health assessment

### Pattern 3: Loop → Process → Aggregate

```
[Get All] → [Loop Each] → [Aggregate Results]
```

**Use Cases:**
- Daily action list generation
- Pipeline health dashboard
- Weekly forecasting report

### Pattern 4: Conditional Pipeline

```
[Check Status] → [If/Then Branches] → [Stage-Specific Actions]
```

**Use Cases:**
- Deal stage automation
- Lead nurturing workflows
- Escalation triggers

---

## 🔧 Configuration Examples

### Memory Search for Contacts

```json
{
  "agent_type": "memory_search",
  "config": {
    "query": "type:contact AND tags:hot_lead",
    "max_results": 50,
    "similarity_threshold": 0.7
  }
}
```

### Convergence Engine for Scoring

```json
{
  "agent_type": "convergence_engine",
  "config": {
    "strategy": "bayesian_blend",
    "neural_weight": 0.7,
    "bandit_weight": 0.3
  }
}
```

### Conditional Branch for Routing

```json
{
  "agent_type": "conditional_branch",
  "config": {
    "condition": "score >= 0.75",
    "true_branch": "send_proposal",
    "false_branch": "nurture_sequence"
  }
}
```

### Thompson Sampler for Learning

```json
{
  "agent_type": "thompson_sampler",
  "config": {
    "exploration_rate": 0.1,
    "alpha_prior": 1.0,
    "beta_prior": 1.0
  }
}
```

---

## 📦 Pre-Built CRM Workflow Templates

### Template 1: Lead Scoring (Simple)

**File**: `example_workflows/crm_lead_scoring_simple.json`

```json
{
  "version": "1.0",
  "name": "Simple Lead Scoring",
  "description": "Score lead and route based on engagement",
  "nodes": [
    {
      "id": "search_contact",
      "type": "memory_search",
      "config": {"query": "contact:{email}"}
    },
    {
      "id": "get_activities",
      "type": "context_retriever",
      "config": {"k": 10}
    },
    {
      "id": "calculate_score",
      "type": "convergence_engine",
      "config": {"strategy": "bayesian_blend"}
    },
    {
      "id": "route",
      "type": "conditional_branch",
      "config": {"condition": "score >= 0.75"}
    },
    {
      "id": "hot_action",
      "type": "response_generator",
      "config": {"template": "send_proposal"}
    },
    {
      "id": "cold_action",
      "type": "response_generator",
      "config": {"template": "nurture_email"}
    }
  ],
  "connections": [
    {"from": "search_contact", "to": "get_activities"},
    {"from": "get_activities", "to": "calculate_score"},
    {"from": "calculate_score", "to": "route"},
    {"from": "route", "to": "hot_action", "branch": "true"},
    {"from": "route", "to": "cold_action", "branch": "false"}
  ]
}
```

### Template 2: Daily Action List

**File**: `example_workflows/crm_daily_actions.json`

```json
{
  "version": "1.0",
  "name": "Daily Action Prioritization",
  "description": "Generate prioritized action list for today",
  "nodes": [
    {
      "id": "get_active_contacts",
      "type": "memory_search",
      "config": {"query": "type:contact AND last_activity:<30days"}
    },
    {
      "id": "process_each",
      "type": "loop_iterator",
      "config": {"max_iterations": 100}
    },
    {
      "id": "analyze_contact",
      "type": "hololoom_query",
      "config": {"pattern": "fast"}
    },
    {
      "id": "prioritize",
      "type": "convergence_engine",
      "config": {"strategy": "argmax"}
    },
    {
      "id": "sort_by_priority",
      "type": "knowledge_fusion",
      "config": {"max_depth": 1}
    },
    {
      "id": "format_list",
      "type": "response_generator",
      "config": {"format": "markdown"}
    }
  ],
  "connections": [
    {"from": "get_active_contacts", "to": "process_each"},
    {"from": "process_each", "to": "analyze_contact"},
    {"from": "analyze_contact", "to": "prioritize"},
    {"from": "prioritize", "to": "sort_by_priority"},
    {"from": "sort_by_priority", "to": "format_list"}
  ]
}
```

---

## 🚀 Advanced CRM Features

### 1. Adaptive Lead Scoring (Thompson Sampling)

**Problem**: Static scoring doesn't learn which factors actually predict conversions.

**Solution**: Use Thompson Sampler to adaptively weight factors.

**Workflow:**
```
[Get Contact]
    ↓
[Parallel: Extract All Factors]
    ↓
[Thompson Sampler: Learn Factor Weights]
    ↓
[Convergence: Weighted Score]
```

**Result**: System learns that "email opens" matter more than "company size" for your specific business.

### 2. Multi-Stage Pipeline Automation

**Problem**: Different deal stages need different actions.

**Solution**: Stage-specific conditional branches.

**Workflow:**
```
[Get Deal]
    ↓
[Conditional: Check Stage]
    ├─► [Lead] → Auto-qualify criteria
    ├─► [Qualified] → Send demo invite
    ├─► [Proposal] → Check if stale (>7 days)
    ├─► [Negotiation] → Notify manager
    └─► [Closing] → Generate contract
```

### 3. Predictive Deal Forecasting

**Problem**: Hard to predict which deals will close.

**Solution**: Historical pattern matching with Thompson Sampling.

**Workflow:**
```
[Current Deal]
    ↓
[Context: Similar Historical Deals]
    ↓
[Parallel: Extract Signals]
    ↓
[Thompson: Probabilistic Forecast]
    ↓
[Output: Close Probability + Expected Value]
```

**Result**: "Deal D001 has 78% close probability, expected value $39,000"

---

## 📊 Dashboard Integration

### Real-Time CRM Dashboard

You can create a dashboard that runs workflows automatically:

**Setup:**

1. **Backend** runs workflows on schedule (cron)
2. **WebSocket** broadcasts updates
3. **Dashboard** displays real-time metrics

**Example Dashboard Widgets:**

```javascript
// Widget 1: Lead Score Distribution
const leaderboardWorkflow = {
  name: "Lead Leaderboard",
  schedule: "every 1 hour",
  workflow: leadScoringWorkflow
};

// Widget 2: Deal Pipeline Health
const pipelineWorkflow = {
  name: "Pipeline Health",
  schedule: "every 30 minutes",
  workflow: dealPipelineWorkflow
};

// Widget 3: Action Items
const actionsWorkflow = {
  name: "Today's Actions",
  schedule: "every morning at 8am",
  workflow: dailyActionsWorkflow
};
```

---

## 🎯 Best Practices

### 1. Start Simple

Begin with:
```
[Search] → [Score] → [Route]
```

Then add complexity:
```
[Search] → [Parallel Signals] → [Fusion] → [Thompson] → [Route]
```

### 2. Use Parallel Execution

Don't do this:
```
[Signal 1] → [Signal 2] → [Signal 3] → [Signal 4]
```

Do this:
```
[Parallel: Signal 1, 2, 3, 4] → [Fusion]
```

**Why**: 4x faster!

### 3. Add Safety Checks

Before any critical action:
```
[Action Plan] → [Safety Guardrails] → [Execute]
```

### 4. Save and Version Workflows

Export successful workflows:
```
Ctrl+S → Save to example_workflows/my_pipeline_v2.json
```

### 5. Monitor Execution

Watch WebSocket output to debug:
```javascript
ws.onmessage = (e) => {
  console.log(JSON.parse(e.data));
};
```

---

## 🔮 Future CRM Workflow Ideas

### Coming Soon

1. **Natural Language Workflow Generation**
   - "Create a workflow that scores leads and sends proposals to hot leads"
   - AI generates workflow automatically

2. **Workflow Analytics**
   - Track success rates
   - A/B test different scoring algorithms
   - Optimize factor weights

3. **Integration Nodes**
   - Gmail (send emails)
   - Calendar (schedule meetings)
   - Slack (notify team)
   - Salesforce (sync data)

4. **Collaborative Workflows**
   - Share templates with team
   - Real-time collaborative editing
   - Version control with git

---

## 📚 Next Steps

### Immediate (Today)

1. **Start the server**: `python workflow_executor.py`
2. **Open builder**: `workflow_builder.html`
3. **Try Example 1**: Simple lead scoring
4. **Save your workflow**: Ctrl+S

### This Week

1. Build 3 CRM workflows:
   - Lead scoring
   - Daily actions
   - Deal pipeline automation
2. Test with real data
3. Refine and iterate

### This Month

1. Create comprehensive CRM workflow library
2. Integrate with dashboard
3. Train team on builder
4. Deploy to production

---

## 💡 Summary

The Workflow Builder is **ideal** for CRM because:

✅ **Visual** - See entire sales process
✅ **Flexible** - Customize for your business
✅ **Powerful** - 18 agent types, unlimited combinations
✅ **Real-time** - See workflows execute live
✅ **Learnable** - Thompson Sampling adapts over time
✅ **Shareable** - Export/import workflow templates

**Start building your CRM workflows now:**

```bash
cd HoloLoom/web_dashboard
python workflow_executor.py
# Open workflow_builder.html in browser
```

Then drag, drop, connect, and automate your entire sales pipeline!
