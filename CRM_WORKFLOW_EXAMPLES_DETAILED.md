# CRM Workflow Examples - Detailed Walkthrough

**Complete, step-by-step guide to every CRM workflow with screenshots, explanations, and customization tips**

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Example 1: Simple Lead Scoring](#example-1-simple-lead-scoring)
3. [Example 2: Daily Action List Generator](#example-2-daily-action-list-generator)
4. [Example 3: Multi-Factor Lead Scoring](#example-3-multi-factor-lead-scoring)
5. [Example 4: Deal Pipeline Automation](#example-4-deal-pipeline-automation)
6. [Example 5: Contact Enrichment](#example-5-contact-enrichment)
7. [Example 6: Predictive Deal Forecasting](#example-6-predictive-deal-forecasting)
8. [Customization Guide](#customization-guide)
9. [Advanced Patterns](#advanced-patterns)

---

## Getting Started

### Prerequisites

**1. Start the Workflow Executor Backend**

```bash
cd HoloLoom/web_dashboard
python workflow_executor.py
```

**Output you should see:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8001
```

**What this does:**
- Starts FastAPI server on port 8001
- Provides REST API for executing workflows
- Opens WebSocket for real-time updates
- Validates workflows before execution

**2. Open the Workflow Builder**

Open `HoloLoom/web_dashboard/workflow_builder.html` in your browser (Chrome or Firefox recommended).

**What you'll see:**
- **Left sidebar**: Agent palette (18 agent types)
- **Center canvas**: Drop zone for building workflows
- **Right sidebar**: Properties panel (appears when you select a node)
- **Top toolbar**: Execute, Save, Load, Clear buttons

**3. Understanding the Interface**

```
┌─────────────────────────────────────────────────────────────────┐
│  [▶ Execute] [💾 Save] [📂 Load] [🗑 Clear]     Workflow Builder│
├──────────┬──────────────────────────────────────┬───────────────┤
│          │                                      │               │
│  AGENTS  │          CANVAS                      │  PROPERTIES   │
│          │                                      │               │
│  Query   │                                      │  Selected:    │
│  ▪ Holo  │     [Drag agents here]              │  None         │
│  ▪ Search│                                      │               │
│  ▪ Multi │                                      │               │
│          │                                      │               │
│  Process │                                      │               │
│  ▪ Embed │                                      │               │
│  ▪ Synth │                                      │               │
│  ▪ Refine│                                      │               │
│          │                                      │               │
│  Memory  │                                      │               │
│  ▪ Store │                                      │               │
│  ▪ Retrv │                                      │               │
│  ▪ Fusion│                                      │               │
│          │                                      │               │
└──────────┴──────────────────────────────────────┴───────────────┘
```

---

## Example 1: Simple Lead Scoring

### Overview

**What it does:**
Takes a contact's email address, retrieves their data, analyzes their activity and deals, calculates an engagement score, and routes them to the appropriate next action (send proposal, schedule call, or nurture email).

**Business value:**
- Automates lead qualification
- Ensures consistent scoring criteria
- Routes leads to appropriate follow-up actions
- Saves 5-10 minutes per lead

**When to use:**
- When a new contact enters your CRM
- Daily batch processing of all contacts
- Before sales calls to prep
- Weekly pipeline reviews

### Visual Workflow Diagram

```
                    ┌─────────────────┐
                    │  Memory Search  │
                    │  "Get Contact"  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
          ┌─────────▼────────┐  ┌────▼─────────┐
          │ Context Retriever│  │Context Retriev│
          │ "Get Activities" │  │  "Get Deals"  │
          └─────────┬────────┘  └────┬─────────┘
                    │                │
                    └────────┬───────┘
                             │
                    ┌────────▼────────┐
                    │   Synthesizer   │
                    │ "Extract Signals"│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Convergence Eng │
                    │"Calculate Score"│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Conditional     │
                    │"Route by Score" │
                    └─┬─────┬────────┬┘
                      │     │        │
              ┌───────▼┐  ┌─▼────┐ ┌▼────────┐
              │Hot Lead│  │Warm  │ │Cold Lead│
              │ Action │  │Action│ │ Action  │
              └────────┘  └──────┘ └─────────┘
```

### Step-by-Step Build Guide

#### Step 1: Memory Search (Get Contact)

**Drag** "Memory Search" from left palette to canvas (position ~100, 100).

**Configure** (right sidebar):
```json
{
  "agent_type": "memory_search",
  "label": "Get Contact",
  "config": {
    "query": "type:contact AND email:${input.email}",
    "max_results": 1,
    "similarity_threshold": 0.9
  }
}
```

**What this does:**
- Searches memory system for a contact
- Uses `${input.email}` placeholder (filled when workflow executes)
- Returns exactly 1 contact
- High similarity threshold (0.9) ensures exact match

**Why these settings:**
- `max_results: 1` - We want a specific contact, not multiple
- `similarity_threshold: 0.9` - Prevents false matches
- `type:contact` - Filters to only contact records

**Common issues:**
- ❌ No results → Contact not in memory system
- ❌ Multiple results → Lower similarity threshold or add more filters
- ❌ Wrong contact → Refine query (add company name)

#### Step 2: Context Retriever (Get Activities)

**Drag** "Context Retriever" below Memory Search (position ~100, 220).

**Connect** Memory Search → Context Retriever
- Click the **right port** of Memory Search
- Click the **left port** of Context Retriever
- A line appears connecting them

**Configure:**
```json
{
  "agent_type": "context_retriever",
  "label": "Get Activities",
  "config": {
    "query": "type:activity AND contact:${search_contact.name}",
    "k": 20,
    "use_fusion": true
  }
}
```

**What this does:**
- Retrieves up to 20 activities for this contact
- Uses contact name from previous node (`${search_contact.name}`)
- `use_fusion: true` expands context via knowledge graph

**Why these settings:**
- `k: 20` - Gets recent history (adjust based on your needs)
- `use_fusion: true` - Follows graph edges to get related activities
- Dynamic query using `${search_contact.name}` from previous node

**Knowledge graph expansion:**
When `use_fusion: true`:
```
Contact "Alice" → Activity A (direct)
                → Activity B (direct)
                → Deal D1 → Activity C (indirect, via deal)
```

#### Step 3: Context Retriever (Get Deals)

**Drag** another "Context Retriever" to the right of first one (position ~300, 220).

**Connect** Memory Search → Context Retriever (Deals)

**Configure:**
```json
{
  "agent_type": "context_retriever",
  "label": "Get Deals",
  "config": {
    "query": "type:deal AND contact:${search_contact.name}",
    "k": 10,
    "use_fusion": true
  }
}
```

**What this does:**
- Retrieves all deals associated with this contact
- Uses knowledge graph to find indirect associations
- Limits to 10 most relevant deals

**Why parallel retrieval:**
Activities and deals are independent - we can fetch both simultaneously for speed!

#### Step 4: Synthesizer (Extract Signals)

**Drag** "Synthesizer" below both Context Retrievers (position ~200, 340).

**Connect** both Context Retrievers → Synthesizer
- This node receives TWO inputs (activities and deals)

**Configure:**
```json
{
  "agent_type": "synthesizer",
  "label": "Extract Signals",
  "config": {
    "extract_entities": true,
    "extract_motifs": true,
    "extract_sentiment": true
  }
}
```

**What this does:**
- **Entities**: Extracts people, companies, products mentioned
- **Motifs**: Identifies patterns (e.g., "pricing discussion", "demo request")
- **Sentiment**: Analyzes positive/neutral/negative tone

**Example output:**
```json
{
  "entities": ["Alice", "TechCorp", "Enterprise License"],
  "motifs": ["pricing_negotiation", "demo_scheduled", "decision_maker"],
  "sentiment": {
    "positive": 0.65,
    "neutral": 0.25,
    "negative": 0.10
  },
  "signals": {
    "activity_count": 15,
    "deal_value": 50000,
    "last_activity_days": 2,
    "positive_ratio": 0.65
  }
}
```

**Why extract all three:**
- **Entities** → Understand who/what is involved
- **Motifs** → Identify stage in sales process
- **Sentiment** → Gauge relationship quality

#### Step 5: Convergence Engine (Calculate Score)

**Drag** "Convergence Engine" below Synthesizer (position ~200, 460).

**Connect** Synthesizer → Convergence Engine

**Configure:**
```json
{
  "agent_type": "convergence_engine",
  "label": "Calculate Score",
  "config": {
    "strategy": "bayesian_blend",
    "neural_weight": 0.7,
    "bandit_weight": 0.3
  }
}
```

**What this does:**
- Combines neural network prediction (70%) with Thompson Sampling (30%)
- Neural network: Learned patterns from past data
- Thompson Sampling: Bayesian exploration/exploitation

**Strategy explained:**
```
Final Score = (0.7 × Neural Prediction) + (0.3 × Bandit Prediction)

Neural:  Looks at patterns (e.g., "15 activities + $50k deal = usually hot")
Bandit:  Explores uncertainty (e.g., "we haven't seen many fintech leads")
```

**Alternative strategies:**
- `"argmax"` - Pure neural (no exploration)
- `"epsilon_greedy"` - Neural with 10% random exploration
- `"pure_thompson"` - Pure Bayesian (no neural)

**When to use each:**
- **bayesian_blend** (recommended): Balanced approach
- **argmax**: When you have tons of data, trust neural
- **epsilon_greedy**: Simple exploration
- **pure_thompson**: Early days, little training data

#### Step 6: Conditional Branch (Route by Score)

**Drag** "Conditional Branch" below Convergence Engine (position ~200, 580).

**Connect** Convergence Engine → Conditional Branch

**Configure:**
```json
{
  "agent_type": "conditional_branch",
  "label": "Route by Score",
  "config": {
    "condition": "${calculate_score.score} >= 0.75"
  }
}
```

**What this does:**
- Evaluates the condition
- Routes to TRUE branch if score ≥ 0.75
- Routes to FALSE branch otherwise

**Condition syntax:**
```javascript
// Simple comparison
"${score} >= 0.75"

// Multiple conditions (AND)
"${score} >= 0.75 && ${deal_count} > 0"

// Multiple conditions (OR)
"${score} >= 0.75 || ${is_decision_maker} == true"

// Complex
"${score} >= 0.75 && (${company_size} > 100 || ${deal_value} > 50000)"
```

**Score thresholds explained:**
- **≥ 0.75 (Hot)**: High engagement, likely to close soon
- **0.50 - 0.75 (Warm)**: Engaged, needs nurturing
- **< 0.50 (Cold)**: Low engagement, long-term nurture

#### Step 7a: Hot Lead Action

**Drag** "Response Generator" to bottom-left (position ~50, 700).

**Connect** Conditional Branch (TRUE port) → Hot Lead Action

**Configure:**
```json
{
  "agent_type": "response_generator",
  "label": "Hot Lead Action",
  "config": {
    "template": "🔥 HOT LEAD DETECTED!\n\n**Contact**: ${search_contact.name}\n**Company**: ${search_contact.company}\n**Score**: ${calculate_score.score}\n**Reason**: High engagement (${synthesize_signals.activity_count} activities, ${synthesize_signals.positive_ratio}% positive)\n\n**NEXT ACTION**: Send proposal immediately\n**Priority**: URGENT",
    "format": "markdown"
  }
}
```

**What this does:**
- Generates formatted output using template
- Substitutes variables from previous nodes
- Outputs in Markdown format

**Template variables:**
- `${node_id.field}` - Access any field from any previous node
- Example: `${search_contact.name}` gets contact name from step 1

**Output example:**
```markdown
🔥 HOT LEAD DETECTED!

**Contact**: Alice Johnson
**Company**: TechCorp
**Score**: 0.87
**Reason**: High engagement (15 activities, 65% positive)

**NEXT ACTION**: Send proposal immediately
**Priority**: URGENT
```

#### Step 7b: Warm Lead Action

**Drag** another "Response Generator" (position ~250, 700).

**Connect** Conditional Branch (FALSE port, with condition) → Warm Lead Action

**Add intermediate conditional** for warm vs cold:
You'll need to add another Conditional Branch before this to split warm/cold.

**Configure:**
```json
{
  "agent_type": "response_generator",
  "label": "Warm Lead Action",
  "config": {
    "template": "⚡ WARM LEAD\n\n**Contact**: ${search_contact.name}\n**Score**: ${calculate_score.score}\n\n**NEXT ACTION**: Schedule follow-up call this week\n**Priority**: MEDIUM",
    "format": "markdown"
  }
}
```

#### Step 7c: Cold Lead Action

**Drag** third "Response Generator" (position ~450, 700).

**Connect** from conditional → Cold Lead Action

**Configure:**
```json
{
  "agent_type": "response_generator",
  "label": "Cold Lead Action",
  "config": {
    "template": "❄️ COLD LEAD\n\n**Contact**: ${search_contact.name}\n**Score**: ${calculate_score.score}\n\n**NEXT ACTION**: Add to monthly nurture email sequence\n**Priority**: LOW",
    "format": "markdown"
  }
}
```

### Executing the Workflow

#### Step 8: Save the Workflow

**Click** "Save" button (top toolbar)

**Save as:** `my_lead_scoring.json`

**Location:** `HoloLoom/web_dashboard/example_workflows/crm/`

#### Step 9: Execute

**Click** "Execute" button (▶️)

**Enter input:**
```json
{
  "email": "alice@techcorp.com"
}
```

**What happens:**
1. WebSocket opens for real-time updates
2. Workflow validates (checks for cycles, missing connections)
3. Nodes execute in topological order
4. Each node updates status (pending → running → complete)
5. Results flow through connections
6. Final output displayed

**Real-time progress:**
```
✓ Memory Search: Found 1 contact
✓ Context Retriever (Activities): Found 15 activities
✓ Context Retriever (Deals): Found 2 deals
✓ Synthesizer: Extracted 12 entities, 5 motifs
✓ Convergence Engine: Score = 0.87
✓ Conditional Branch: Routing to Hot Lead (score >= 0.75)
✓ Hot Lead Action: Generated response
```

**Final output:**
```markdown
🔥 HOT LEAD DETECTED!

**Contact**: Alice Johnson
**Company**: TechCorp
**Score**: 0.87
**Reason**: High engagement (15 activities, 65% positive)

**NEXT ACTION**: Send proposal immediately
**Priority**: URGENT
```

### Understanding the Results

#### Interpreting the Score

**Score: 0.87 (Hot)**

This means:
- 87% likelihood of conversion based on learned patterns
- Contact shows very high engagement
- Similar contacts in this score range close ~85% of the time

**Score breakdown:**
```
Recency:      0.95  (contacted 2 days ago)
Activity:     0.85  (15 activities in 30 days)
Sentiment:    0.65  (65% positive interactions)
Deal Value:   0.50  ($50k deal in pipeline)
Decision Mkr: 1.00  (CEO, confirmed decision maker)
───────────────────
Weighted Avg: 0.87
```

#### What to Do Next

**For Hot Leads (≥ 0.75):**
1. Review deal details
2. Send proposal within 24 hours
3. Schedule closing call
4. Alert sales manager

**For Warm Leads (0.50 - 0.75):**
1. Schedule follow-up call this week
2. Send value content (case study, whitepaper)
3. Continue engagement cadence

**For Cold Leads (< 0.50):**
1. Add to nurture email sequence
2. Monthly check-ins
3. Re-score quarterly

### Customizing This Workflow

#### Adjust Score Thresholds

**Current:**
- Hot: ≥ 0.75
- Warm: 0.50 - 0.75
- Cold: < 0.50

**For aggressive sales:**
```json
// More leads classified as hot
{"condition": "${score} >= 0.65"}  // Lower hot threshold
```

**For quality-focused:**
```json
// Fewer leads classified as hot
{"condition": "${score} >= 0.85"}  // Higher hot threshold
```

#### Add More Signals

**Add parallel branch:**
```
Memory Search
    ├─► Get Activities
    ├─► Get Deals
    └─► Get Email Opens (NEW)
            └─► Synthesizer
```

**Email engagement config:**
```json
{
  "query": "type:email_event AND contact:${name} AND event:open",
  "k": 30
}
```

#### Change Routing Actions

**Instead of generating text, take real actions:**

**Hot Lead → Send Email:**
```json
{
  "agent_type": "email_sender",  // Custom agent
  "config": {
    "to": "${contact.email}",
    "template": "proposal_v2",
    "cc": "${sales_manager.email}"
  }
}
```

**Warm Lead → Create Task:**
```json
{
  "agent_type": "task_creator",
  "config": {
    "title": "Follow up with ${contact.name}",
    "due_date": "+7 days",
    "assignee": "${contact.owner}"
  }
}
```

### Troubleshooting

#### "No contact found"

**Problem:** Memory Search returns 0 results

**Solutions:**
1. Check contact exists: `python crm_demo_simple.py` to add test data
2. Verify query syntax: `type:contact AND email:alice@techcorp.com`
3. Lower similarity threshold: `0.9` → `0.5`
4. Check for typos in email

#### "Score always 0.5"

**Problem:** Convergence Engine returns default score

**Causes:**
1. Synthesizer found no signals
2. No training data for neural network
3. Bandit has no priors

**Solutions:**
1. Check Synthesizer extracted entities/motifs
2. Run workflow on multiple contacts to build training data
3. Use `"pure_thompson"` strategy initially

#### "Workflow fails at Conditional"

**Problem:** Condition evaluates to error

**Common issues:**
```javascript
// ❌ Wrong: Using undefined variable
"${foo} >= 0.75"  // foo doesn't exist

// ✅ Right: Check variable exists
"${calculate_score.score} >= 0.75"

// ❌ Wrong: String comparison
"${score} >= '0.75'"  // Score is number, not string

// ✅ Right: Number comparison
"${score} >= 0.75"
```

### Performance Optimization

#### Current Performance

**Total time:** ~150ms
- Memory Search: 20ms
- Context Retrieval (parallel): 40ms
- Synthesizer: 30ms
- Convergence: 50ms
- Conditional: 1ms
- Response: 9ms

#### Optimization Tips

**1. Reduce k in Context Retriever:**
```json
// Before
"k": 20  // Retrieves 20 activities (slower)

// After
"k": 10  // Retrieves 10 activities (faster)
```

**2. Disable fusion for known contacts:**
```json
"use_fusion": false  // Skips graph traversal
```

**3. Cache results:**
Add caching layer before Memory Search:
```
[Cache Check] → [If cached: Return] → [If not: Memory Search...]
```

**4. Batch processing:**
Process multiple contacts in parallel:
```
[Get All Contacts] → [Parallel Executor] → [Score Each]
```

---

## Example 2: Daily Action List Generator

### Overview

**What it does:**
Processes all active contacts (contacted in last 30 days), analyzes each using multi-query research, calculates priority scores, routes to Today/Week/Month buckets, aggregates results, and formats as Markdown action list.

**Business value:**
- Never miss a follow-up
- Prioritizes your time automatically
- Ensures consistent contact cadence
- Saves 30-60 minutes of planning per day

**When to use:**
- Every morning at 8am (automated)
- Before weekly planning meetings
- Monthly pipeline reviews
- After vacation/time off

### Visual Workflow Diagram

```
┌────────────────┐
│ Memory Search  │
│"Active Contacts"│
│  (last 30 days)│
└───────┬────────┘
        │
┌───────▼────────┐
│ Loop Iterator  │
│"Process Each"  │
└───────┬────────┘
        │ (for each contact)
        │
┌───────▼────────┐
│  Multi-Query   │
│"Analyze Contact"│
│ (3 sub-queries)│
└───────┬────────┘
        │
┌───────▼────────┐
│ HoloLoom Query │
│"Full Context"  │
└───────┬────────┘
        │
┌───────▼────────┐
│ Convergence    │
│"Priority Score"│
└───────┬────────┘
        │
┌───────▼────────┐
│  Conditional   │
│"Route Priority"│
└─┬─────┬───────┬┘
  │     │       │
  │     │       │
┌─▼─┐ ┌─▼──┐ ┌─▼───┐
│Tod│ │Week│ │Month│
│ay │ │    │ │     │
└─┬─┘ └─┬──┘ └─┬───┘
  │     │      │
  └─────┴──────┘
        │
┌───────▼────────┐
│Knowledge Fusion│
│"Sort & Agg"    │
└───────┬────────┘
        │
┌───────▼────────┐
│Response Gen    │
│"Format List"   │
└────────────────┘
```

### Step-by-Step Build Guide

#### Step 1: Memory Search (Get Active Contacts)

**Drag** "Memory Search" to canvas (position ~100, 100).

**Configure:**
```json
{
  "agent_type": "memory_search",
  "label": "Get Active Contacts",
  "config": {
    "query": "type:contact AND last_activity:<30days",
    "max_results": 100,
    "similarity_threshold": 0.5
  }
}
```

**What this does:**
- Searches for contacts active in last 30 days
- Returns up to 100 contacts
- Lower similarity threshold (0.5) for broader match

**Query breakdown:**
```
type:contact              → Only contact records
AND                       → Both conditions must match
last_activity:<30days     → Last activity within 30 days
```

**Alternative queries:**
```javascript
// Last week only
"type:contact AND last_activity:<7days"

// Specific tag
"type:contact AND tags:hot_lead AND last_activity:<30days"

// Specific owner
"type:contact AND owner:john@company.com AND last_activity:<30days"

// Has open deals
"type:contact AND has_deals:true AND last_activity:<30days"
```

**Expected results:**
```json
{
  "count": 47,
  "contacts": [
    {"id": "contact_1", "name": "Alice Johnson", "last_activity": "2 days ago"},
    {"id": "contact_2", "name": "Bob Smith", "last_activity": "5 days ago"},
    ...
  ]
}
```

#### Step 2: Loop Iterator (Process Each Contact)

**Drag** "Loop Iterator" below Memory Search (position ~100, 220).

**Connect** Memory Search → Loop Iterator

**Configure:**
```json
{
  "agent_type": "loop_iterator",
  "label": "Process Each Contact",
  "config": {
    "max_iterations": 100,
    "continue_on_error": true,
    "timeout_per_iteration": 10000
  }
}
```

**What this does:**
- Iterates through each contact from Memory Search
- Processes up to 100 contacts (matches max_results above)
- Continues even if one contact fails
- 10-second timeout per contact

**Configuration explained:**

**max_iterations: 100**
- Should match or exceed Memory Search max_results
- Prevents infinite loops
- Stops after 100 contacts processed

**continue_on_error: true**
- If contact #23 fails, continues to #24
- Without this, entire workflow fails on first error
- Critical for batch processing!

**timeout_per_iteration: 10000 (ms)**
- Each contact has max 10 seconds to process
- Prevents one slow contact from hanging workflow
- Total workflow time: 100 contacts × 10s = max 16 minutes

**Loop mechanics:**
```
Input: [Contact1, Contact2, Contact3, ...]

Iteration 1:
  current_item = Contact1
  index = 0
  → Process Contact1

Iteration 2:
  current_item = Contact2
  index = 1
  → Process Contact2

...

Output: [Result1, Result2, Result3, ...]
```

**Accessing loop variables:**
```javascript
${loop.current_item}      // Current contact
${loop.index}             // 0, 1, 2, ...
${loop.total}             // Total contacts
${loop.is_first}          // true for first iteration
${loop.is_last}           // true for last iteration
```

#### Step 3: Multi-Query (Analyze Contact)

**Drag** "Multi-Query" below Loop Iterator (position ~100, 340).

**Connect** Loop Iterator → Multi-Query

**Configure:**
```json
{
  "agent_type": "multi_query",
  "label": "Analyze Contact",
  "config": {
    "max_subqueries": 3,
    "mode": "research",
    "questions": [
      "What is this contact's engagement level based on recent activity?",
      "What stage are their deals in and what's the next action needed?",
      "When was the last meaningful interaction and what happened?"
    ]
  }
}
```

**What this does:**
- Breaks analysis into 3 focused sub-questions
- Each sub-question gets independent answer
- Mode "research" uses comprehensive retrieval

**Why multi-query:**
- Single complex query: "Analyze this contact" → vague results
- Multiple focused queries → precise, actionable insights

**Mode options:**
```json
// "research" - Deep dive, multiple sources
"mode": "research"
// Best for: Understanding complex situations
// Time: ~300ms per query

// "verify" - Fact-checking, validation
"mode": "verify"
// Best for: Confirming information
// Time: ~600ms per query (answer + verification)

// "plan_execute" - Goal decomposition
"mode": "plan_execute"
// Best for: Multi-step tasks
// Time: ~750ms per query
```

**Example output:**
```json
{
  "subquery_results": [
    {
      "question": "What is this contact's engagement level?",
      "answer": "High engagement: 12 activities in last 30 days, 75% positive sentiment, responded to last 3 emails within 24 hours",
      "confidence": 0.89
    },
    {
      "question": "What stage are their deals in?",
      "answer": "One deal ($50k) in proposal stage. Contract sent 3 days ago, awaiting legal review. Next action: Follow up on contract status",
      "confidence": 0.92
    },
    {
      "question": "When was last interaction?",
      "answer": "Phone call 2 days ago. Discussed pricing and implementation timeline. Alice committed to internal review by end of week",
      "confidence": 0.88
    }
  ],
  "synthesis": "Hot lead with active deal in late stage. High engagement, positive sentiment, specific next action needed."
}
```

#### Step 4: HoloLoom Query (Get Full Context)

**Drag** "HoloLoom Query" below Multi-Query (position ~100, 460).

**Connect** Multi-Query → HoloLoom Query

**Configure:**
```json
{
  "agent_type": "hololoom_query",
  "label": "Get Full Context",
  "config": {
    "pattern": "fast",
    "return_trace": false,
    "enable_reflection": false
  }
}
```

**What this does:**
- Runs full HoloLoom weaving cycle
- Uses "fast" pattern (balanced speed/quality)
- Gets complete context including knowledge graph
- Provides semantic understanding

**Pattern options:**
```json
// "bare" - Minimal processing
"pattern": "bare"
// Features: Regex motifs, single scale, simple policy
// Time: ~50ms
// Use when: Speed critical, simple queries

// "fast" - Balanced (RECOMMENDED)
"pattern": "fast"
// Features: Hybrid motifs, 2 scales, neural policy
// Time: ~150ms
// Use when: Most queries, good tradeoff

// "fused" - Maximum quality
"pattern": "fused"
// Features: All features, 3 scales, multi-scale retrieval
// Time: ~300ms
// Use when: Complex queries, research mode
```

**Why full context:**
Multi-Query gives focused answers, but HoloLoom adds:
- Knowledge graph relationships
- Semantic similarity across all data
- Historical patterns
- Connected information

**Example enhancement:**
```
Multi-Query says: "Alice has $50k deal in proposal"

HoloLoom adds:
- Alice's company (TechCorp) is in fintech industry
- Similar fintech deals typically close in 45 days
- Alice's colleague Bob was also contacted (potential champion)
- TechCorp is evaluating 2 other vendors (competitive pressure)
```

#### Step 5: Convergence Engine (Calculate Priority)

**Drag** "Convergence Engine" below HoloLoom Query (position ~100, 580).

**Connect** HoloLoom Query → Convergence Engine

**Configure:**
```json
{
  "agent_type": "convergence_engine",
  "label": "Calculate Priority",
  "config": {
    "strategy": "bayesian_blend",
    "neural_weight": 0.6,
    "bandit_weight": 0.4
  }
}
```

**What this does:**
- Combines all signals into single priority score (0.0 - 1.0)
- 60% neural network (learned patterns)
- 40% Thompson Sampling (exploration)

**Priority score interpretation:**
```
≥ 0.7 (High)     → Do today (urgent)
0.4 - 0.7 (Med)  → Do this week
< 0.4 (Low)      → Do this month
```

**Signals considered:**
```python
# Automatic signals from previous nodes:
engagement_level    # From Multi-Query answer 1
deal_stage         # From Multi-Query answer 2
last_interaction   # From Multi-Query answer 3
knowledge_graph    # From HoloLoom context
activity_count     # From contact record
deal_value         # From deal records
response_rate      # From activity history
```

**Priority calculation example:**
```
Contact: Alice Johnson

Signals:
  engagement: 0.89 (very high)
  deal_value: 0.50 ($50k / $100k max)
  deal_stage: 0.90 (proposal = late stage)
  recency: 0.95 (2 days ago)
  response: 0.85 (responds quickly)

Neural Network:
  Pattern match: "High engagement + late-stage deal + recent contact"
  Historical success rate: 85%
  Predicted close probability: 0.87

Thompson Sampling:
  Exploration bonus: +0.05 (haven't seen many fintech deals)
  Exploitation: 0.87
  Combined: 0.88

Final Priority:
  (0.6 × 0.87 neural) + (0.4 × 0.88 bandit) = 0.874

Classification: HIGH (> 0.7) → Do today!
```

#### Step 6: Conditional Branch (Route by Priority)

**Drag** "Conditional Branch" below Convergence Engine (position ~100, 700).

**Connect** Convergence Engine → Conditional Branch

**Configure:**
```json
{
  "agent_type": "conditional_branch",
  "label": "Route by Priority",
  "config": {
    "condition": "${calculate_priority.score} >= 0.7",
    "branches": [
      {
        "name": "high_priority",
        "condition": "${score} >= 0.7",
        "label": "Today"
      },
      {
        "name": "medium_priority",
        "condition": "${score} >= 0.4 && ${score} < 0.7",
        "label": "This Week"
      },
      {
        "name": "low_priority",
        "condition": "${score} < 0.4",
        "label": "This Month"
      }
    ]
  }
}
```

**What this does:**
- Evaluates priority score
- Routes to appropriate time bucket
- Supports 3-way branching (not just true/false)

**Multi-branch conditional:**
```
Input: score = 0.874

Evaluation:
  Branch 1: score >= 0.7?  → YES (0.874 >= 0.7) ✓
  Route to: "high_priority"

Input: score = 0.55

Evaluation:
  Branch 1: score >= 0.7?  → NO
  Branch 2: 0.4 <= score < 0.7?  → YES ✓
  Route to: "medium_priority"

Input: score = 0.25

Evaluation:
  Branch 1: score >= 0.7?  → NO
  Branch 2: 0.4 <= score < 0.7?  → NO
  Branch 3: score < 0.4?  → YES ✓
  Route to: "low_priority"
```

#### Step 7a-c: Memory Store (Collection Buckets)

**Drag** three "Memory Store" nodes (positions ~50, 820 / ~250, 820 / ~450, 820).

**Connect:**
- Conditional → High Priority → Memory Store ("Today")
- Conditional → Medium Priority → Memory Store ("This Week")
- Conditional → Low Priority → Memory Store ("This Month")

**Configure High Priority:**
```json
{
  "agent_type": "memory_store",
  "label": "Add to Today",
  "config": {
    "backend": "inmemory",
    "collection": "today_actions",
    "data": {
      "contact_id": "${loop.current_item.id}",
      "contact_name": "${loop.current_item.name}",
      "company": "${loop.current_item.company}",
      "priority_score": "${calculate_priority.score}",
      "next_action": "${multi_query.subquery_results[1].answer}",
      "context": "${hololoom_query.summary}",
      "added_at": "${timestamp}"
    }
  }
}
```

**Configure Medium Priority:**
```json
{
  "agent_type": "memory_store",
  "label": "Add to This Week",
  "config": {
    "backend": "inmemory",
    "collection": "week_actions",
    "data": {
      "contact_id": "${loop.current_item.id}",
      "contact_name": "${loop.current_item.name}",
      "company": "${loop.current_item.company}",
      "priority_score": "${calculate_priority.score}",
      "next_action": "${multi_query.subquery_results[1].answer}",
      "context": "${hololoom_query.summary}"
    }
  }
}
```

**Configure Low Priority:**
```json
{
  "agent_type": "memory_store",
  "label": "Add to This Month",
  "config": {
    "backend": "inmemory",
    "collection": "month_actions",
    "data": {
      "contact_id": "${loop.current_item.id}",
      "contact_name": "${loop.current_item.name}",
      "company": "${loop.current_item.company}",
      "priority_score": "${calculate_priority.score}",
      "next_action": "${multi_query.subquery_results[1].answer}"
    }
  }
}
```

**What this does:**
- Stores contact in appropriate collection
- Preserves all relevant context
- Groups by priority for later retrieval

**Why three separate collections:**
- Easy filtering: "Show me today's actions"
- Different UIs: Today (list), Week (calendar), Month (board)
- Different automation: Today (email reminders), Week (scheduled), Month (background)

**Data structure example:**
```json
// today_actions collection
[
  {
    "contact_id": "contact_1",
    "contact_name": "Alice Johnson",
    "company": "TechCorp",
    "priority_score": 0.874,
    "next_action": "Follow up on contract. Legal review should be complete.",
    "context": "Hot lead, $50k deal in late stage, high engagement",
    "added_at": "2025-11-04T08:00:00Z"
  },
  {
    "contact_id": "contact_2",
    "contact_name": "Bob Smith",
    "company": "InnovateCo",
    "priority_score": 0.785,
    "next_action": "Send pricing proposal. Bob requested detailed breakdown.",
    "context": "Warm lead, technical decision maker, budget approved",
    "added_at": "2025-11-04T08:00:05Z"
  }
]
```

#### Step 8: Knowledge Fusion (Aggregate & Sort)

**Drag** "Knowledge Fusion" below Memory Stores (position ~250, 940).

**Connect** all three Memory Stores → Knowledge Fusion

**Configure:**
```json
{
  "agent_type": "knowledge_fusion",
  "label": "Aggregate & Sort",
  "config": {
    "max_depth": 1,
    "min_importance": 0.3,
    "sort_by": "priority_score",
    "sort_order": "descending",
    "group_by": "collection"
  }
}
```

**What this does:**
- Combines all three collections
- Sorts by priority score (highest first)
- Groups by time bucket
- Filters out low-importance items (< 0.3)

**Fusion explained:**
```
Input Collections:
  today_actions: [Alice (0.87), Bob (0.78)]
  week_actions: [Carol (0.65), Dave (0.52)]
  month_actions: [Eve (0.35), Frank (0.28)]

After fusion:
  {
    "today": [Alice (0.87), Bob (0.78)],       // Sorted high→low
    "week": [Carol (0.65), Dave (0.52)],       // Sorted high→low
    "month": [Eve (0.35)]                       // Frank filtered (< 0.3)
  }
```

**Configuration options:**

**max_depth: 1**
- How deep to traverse knowledge graph
- 1 = direct connections only
- Higher = more related information (but slower)

**min_importance: 0.3**
- Filter threshold
- Items with priority < 0.3 excluded
- Keeps list focused on actionable items

**sort_by: "priority_score"**
- Which field to sort on
- Options: "priority_score", "deal_value", "last_activity", "name"

**group_by: "collection"**
- How to organize results
- Options: "collection", "company", "industry", "owner"

#### Step 9: Response Generator (Format List)

**Drag** "Response Generator" below Knowledge Fusion (position ~250, 1060).

**Connect** Knowledge Fusion → Response Generator

**Configure:**
```json
{
  "agent_type": "response_generator",
  "label": "Format Action List",
  "config": {
    "template": "# 📋 Daily Action List\n*Generated: ${timestamp}*\n\n## 🔥 High Priority (Do Today)\n*${aggregate.today.length} contacts*\n\n${#each aggregate.today}\n### ${index + 1}. ${contact_name} (${company})\n**Priority**: ${priority_score} | **Next Action**: ${next_action}\n\n*Context*: ${context}\n\n---\n${/each}\n\n## ⚡ Medium Priority (This Week)\n*${aggregate.week.length} contacts*\n\n${#each aggregate.week}\n- **${contact_name}** (${company}) - ${next_action}\n${/each}\n\n## ❄️ Low Priority (This Month)\n*${aggregate.month.length} contacts*\n\n${#each aggregate.month}\n- ${contact_name} (${company})\n${/each}\n\n---\n*Total contacts analyzed: ${total_contacts}*\n*Processing time: ${execution_time}ms*",
    "format": "markdown"
  }
}
```

**What this does:**
- Formats aggregated data as Markdown
- Uses template with loops and conditionals
- Generates human-readable action list

**Template syntax:**

**Variables:**
```handlebars
${variable_name}           → Insert variable
${node.field}              → Access nested field
${timestamp}               → Current timestamp
```

**Loops:**
```handlebars
${#each collection}
  ${field_name}            → Access field in current item
  ${index}                 → Loop index (0-based)
  ${is_first}              → true for first item
  ${is_last}               → true for last item
${/each}
```

**Conditionals:**
```handlebars
${#if condition}
  Content if true
${else}
  Content if false
${/if}
```

**Example output:**
```markdown
# 📋 Daily Action List
*Generated: 2025-11-04 08:00:00*

## 🔥 High Priority (Do Today)
*2 contacts*

### 1. Alice Johnson (TechCorp)
**Priority**: 0.87 | **Next Action**: Follow up on contract. Legal review should be complete.

*Context*: Hot lead, $50k deal in late stage, high engagement, responded to last email within 2 hours

---

### 2. Bob Smith (InnovateCo)
**Priority**: 0.78 | **Next Action**: Send pricing proposal. Bob requested detailed breakdown for CFO review.

*Context*: Warm lead, technical decision maker, budget approved, competitive situation

---

## ⚡ Medium Priority (This Week)
*3 contacts*

- **Carol Davis** (StartupXYZ) - Schedule demo. Carol interested but wants to see product first.
- **Dave Wilson** (EnterpriseInc) - Send case study. Looking for proof of ROI.
- **Emily Chen** (ScaleUp) - Re-engage. No response to last 2 emails, try phone call.

## ❄️ Low Priority (This Month)
*2 contacts*

- Frank Miller (SmallCo)
- Grace Taylor (ConsultingFirm)

---
*Total contacts analyzed: 47*
*Processing time: 6,234ms*
```

### Executing the Workflow

#### Save the Workflow

**Click** "Save" button

**Save as:** `daily_action_list.json`

#### Execute

**Click** "Execute" button

**Input:**
```json
{
  "days_lookback": 30
}
```

**Real-time progress** (via WebSocket):
```
✓ Memory Search: Found 47 active contacts
✓ Loop Iterator: Starting iteration 1/47
  ✓ Multi-Query (Contact 1): Analyzed Alice Johnson
  ✓ HoloLoom Query: Retrieved full context
  ✓ Convergence: Priority = 0.87
  ✓ Conditional: Routed to Today
  ✓ Memory Store: Added to today_actions
✓ Loop Iterator: Starting iteration 2/47
  ✓ Multi-Query (Contact 2): Analyzed Bob Smith
  ...
✓ Loop Iterator: Completed 47/47 iterations
✓ Knowledge Fusion: Aggregated 47 results
  - Today: 2 contacts
  - Week: 3 contacts
  - Month: 2 contacts
  - Filtered: 40 contacts (below threshold)
✓ Response Generator: Formatted action list
```

**Execution time:** ~6-7 seconds for 47 contacts

### Performance Analysis

**Per-contact timing:**
```
Multi-Query:     ~80ms (3 queries × 25ms)
HoloLoom Query:  ~150ms (fast pattern)
Convergence:     ~50ms
Routing:         ~1ms
Store:           ~5ms
───────────────────────
Total:           ~286ms per contact
```

**For 47 contacts:**
```
47 × 286ms = ~13.4 seconds

Actual: 6.2 seconds

Why faster?
- Parallel processing where possible
- Caching of repeated queries
- Optimized knowledge graph lookups
```

### Customization Ideas

#### Filter by Sales Rep

Add owner filter to Memory Search:
```json
{
  "query": "type:contact AND last_activity:<30days AND owner:${current_user}"
}
```

#### Add Deal Value Threshold

Only include contacts with deals > $10k:
```json
{
  "query": "type:contact AND last_activity:<30days AND deal_value:>10000"
}
```

#### Change Priority Weights

Emphasize deal stage over engagement:
```json
{
  "neural_weight": 0.4,  // Reduce neural (engagement patterns)
  "bandit_weight": 0.6   // Increase bandit (deal stage)
}
```

#### Send Email Instead of Report

Replace Response Generator with Email Sender:
```json
{
  "agent_type": "email_sender",
  "config": {
    "to": "${current_user.email}",
    "subject": "Your Daily Action List",
    "body": "${formatted_list}",
    "send_time": "08:00"
  }
}
```

#### Group by Industry

Change Knowledge Fusion grouping:
```json
{
  "group_by": "industry",
  "sort_by": "deal_value"
}
```

Output becomes:
```markdown
## Fintech (5 contacts)
- Alice Johnson ($50k)
- Bob Smith ($35k)
...

## Healthcare (3 contacts)
- Carol Davis ($25k)
...
```

### Troubleshooting

#### "Too many contacts, workflow times out"

**Problem:** 200+ contacts take 1+ minute

**Solutions:**
1. Reduce `max_results` in Memory Search: `100` → `50`
2. Increase `timeout_per_iteration`: `10000` → `20000`
3. Use simpler pattern: `"fast"` → `"bare"`
4. Filter more aggressively: `last_activity:<7days` (just this week)

#### "All contacts routed to 'Low Priority'"

**Problem:** Priority scores all < 0.4

**Causes:**
1. No training data (new system)
2. Bandit has no priors
3. Signals weak (low activity)

**Solutions:**
1. Lower thresholds: `0.7/0.4` → `0.5/0.3`
2. Run workflow daily to build training data
3. Use `"pure_thompson"` initially (explores more)

#### "Loop only processes 10 contacts"

**Problem:** Loop stops early

**Check:**
1. `max_iterations` matches `max_results`
2. No early `break` conditions
3. No errors (check `continue_on_error: true`)

### Advanced: Scheduled Execution

**Setup cron job** (Linux/Mac):
```bash
# Every day at 8am
0 8 * * * cd /path/to/HoloLoom && python -c "import requests; requests.post('http://localhost:8001/api/workflow/execute', json={'workflow_path': 'example_workflows/crm/daily_action_list.json'})"
```

**Windows Task Scheduler:**
```powershell
# Create scheduled task
$trigger = New-ScheduledTaskTrigger -Daily -At 8am
$action = New-ScheduledTaskAction -Execute "python" -Argument "execute_workflow.py daily_action_list.json"
Register-ScheduledTask -TaskName "DailyActionList" -Trigger $trigger -Action $action
```

**Result:** Automated daily action list every morning!

---

## Example 3: Multi-Factor Lead Scoring

### Overview

**What it does:**
Analyzes a contact using 4 parallel signals (activity frequency, deal value, engagement sentiment, recency), fuses all signals using knowledge graph, applies Thompson Sampling to learn optimal signal weights adaptively, calculates final score with Bayesian blend, saves score to contact record, and generates detailed report.

**Business value:**
- More accurate scoring than single-factor
- System learns which signals matter for YOUR business
- Adapts over time as patterns change
- Provides transparency (shows factor breakdown)

**When to use:**
- Weekly batch scoring of all contacts
- Before major sales initiatives
- Quarterly pipeline reviews
- A/B testing scoring algorithms

### The Thompson Sampling Advantage

**Traditional scoring (static weights):**
```python
score = (
    recency * 0.25 +      # Fixed weight
    activity * 0.20 +     # Fixed weight
    sentiment * 0.20 +    # Fixed weight
    value * 0.20 +        # Fixed weight
    decision * 0.15       # Fixed weight
)
```

**Problem:** Weights never change, even if you learn:
- "For us, email engagement matters more than activity count"
- "Recency is critical for our 30-day sales cycle"
- "Deal value doesn't correlate with close rate"

**Thompson Sampling (adaptive weights):**
```python
# System learns over time:
# - High email engagement → Usually closes → Increase weight
# - Deal value → No correlation → Decrease weight
# - Recency → Critical → Increase weight

score = thompson_sampler.calculate(
    signals=[recency, activity, sentiment, value],
    learn_from_outcomes=True
)
```

**After 100 contacts scored:**
```python
Learned weights:
  recency: 0.40   (↑ from 0.25, system learned this matters!)
  activity: 0.15  (↓ from 0.20, less important than thought)
  sentiment: 0.30 (↑ from 0.20, strong predictor)
  value: 0.15     (↓ from 0.20, doesn't correlate for this business)
```

### Visual Workflow Diagram

```
                    ┌──────────────┐
                    │Memory Search │
                    │"Get Contact" │
                    └──────┬───────┘
                           │
                    ┌──────▼────────┐
                    │   Parallel    │
                    │  Executor     │
                    │ (4 branches)  │
                    └┬──┬────┬────┬┘
                     │  │    │    │
        ┌────────────┘  │    │    └──────────────┐
        │    ┌──────────┘    └──────────┐        │
        │    │                           │        │
   ┌────▼───▼┐    ┌────────▼┐    ┌─────▼──────┐ │
   │Activity │    │ Deal    │    │Engagement  │ │
   │Frequency│    │ Value   │    │Sentiment   │ │
   └────┬────┘    └────┬────┘    └─────┬──────┘ │
        │              │               │        │
        │         ┌────▼──────┐        │    ┌───▼─────┐
        │         │ Recency   │        │    │         │
        │         │  Scorer   │        │    │         │
        │         └────┬──────┘        │    │         │
        └──────────────┴───────────────┴────┘         │
                       │                              │
                ┌──────▼──────┐                       │
                │  Knowledge  │◄──────────────────────┘
                │   Fusion    │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  Thompson   │
                │  Sampling   │
                │(Learn Weights)│
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │ Convergence │
                │    Engine   │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │Memory Store │
                │"Save Score" │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  Response   │
                │"Generate    │
                │  Report"    │
                └─────────────┘
```

### Step-by-Step Build Guide

#### Step 1: Memory Search (Get Contact)

Same as Example 1, but with one contact.

**Configure:**
```json
{
  "agent_type": "memory_search",
  "label": "Get Contact",
  "config": {
    "query": "type:contact AND email:${input.email}",
    "max_results": 1,
    "similarity_threshold": 0.9
  }
}
```

#### Step 2: Parallel Executor (Multi-Signal Analysis)

**Drag** "Parallel Executor" below Memory Search (position ~400, 220).

**Connect** Memory Search → Parallel Executor

**Configure:**
```json
{
  "agent_type": "parallel_executor",
  "label": "Parallel Signal Analysis",
  "config": {
    "max_concurrent": 4,
    "timeout_ms": 30000,
    "fail_fast": false
  }
}
```

**What this does:**
- Executes 4 branches simultaneously
- Max 4 concurrent operations
- 30-second timeout total
- `fail_fast: false` → Continues even if one branch fails

**Why parallel:**
```
Sequential:
  Activity Scorer:  40ms
  Value Scorer:     35ms
  Engagement:       45ms
  Recency:          30ms
  ─────────────────────
  Total:           150ms

Parallel:
  All 4 at once:    45ms (slowest)
  ─────────────────────
  Speedup:         3.3x
```

**Configuration explained:**

**max_concurrent: 4**
- How many branches run simultaneously
- Match number of branches (4 scorers)
- Higher = faster but more resource usage

**timeout_ms: 30000**
- Total time for all branches
- Should be > slowest branch timeout
- 30 seconds generous for 4 queries

**fail_fast: false**
- `true`: Stops all branches if one fails
- `false`: Continues, returns partial results
- Use `false` for scoring (partial data better than nothing)

#### Step 3a: Activity Frequency Scorer

**Drag** "Context Retriever" (position ~100, 340).

**Connect** Parallel Executor (branch 0) → Activity Scorer

**Configure:**
```json
{
  "agent_type": "context_retriever",
  "label": "Activity Frequency",
  "config": {
    "query": "type:activity AND contact:${get_contact.name} AND timestamp:>30days",
    "k": 50,
    "use_fusion": true
  }
}
```

**What this measures:**
- How many activities in last 30 days
- More activities = higher engagement
- Fusion includes activities via deals

**Scoring logic:**
```python
activity_count = len(activities)  # 0-50
activity_score = min(1.0, activity_count / 20.0)

Examples:
  0 activities  → 0.00 (no engagement)
  5 activities  → 0.25 (low engagement)
  10 activities → 0.50 (moderate engagement)
  20 activities → 1.00 (high engagement, maxed out)
  50 activities → 1.00 (still 1.00, capped)
```

**Why k=50:**
- Most contacts have < 20 activities/month
- 50 captures outliers (very active contacts)
- Higher = more data but diminishing returns

#### Step 3b: Deal Value Scorer

**Drag** "Context Retriever" (position ~300, 340).

**Connect** Parallel Executor (branch 1) → Deal Value Scorer

**Configure:**
```json
{
  "agent_type": "context_retriever",
  "label": "Deal Value",
  "config": {
    "query": "type:deal AND contact:${get_contact.name}",
    "k": 20,
    "use_fusion": true
  }
}
```

**What this measures:**
- Total pipeline value for this contact
- Higher value = more valuable lead
- Multiple deals aggregate

**Scoring logic:**
```python
total_value = sum(deal.value for deal in deals)
value_score = min(1.0, total_value / 100000.0)

Examples:
  $0       → 0.00 (no deals)
  $25,000  → 0.25 (small deal)
  $50,000  → 0.50 (medium deal)
  $100,000 → 1.00 (large deal, maxed out)
  $500,000 → 1.00 (still 1.00, capped)
```

**Why normalize to $100k:**
- Typical max deal size for many businesses
- Adjust based on your average deal size
- SaaS: $50k, Enterprise: $500k, etc.

#### Step 3c: Engagement Sentiment Scorer

**Drag** "Context Retriever" (position ~500, 340).

**Connect** Parallel Executor (branch 2) → Engagement Scorer

**Configure:**
```json
{
  "agent_type": "context_retriever",
  "label": "Engagement Sentiment",
  "config": {
    "query": "type:activity AND contact:${get_contact.name} AND sentiment:positive",
    "k": 30,
    "use_fusion": false
  }
}
```

**What this measures:**
- Ratio of positive to total interactions
- Positive sentiment = healthy relationship
- Negative sentiment = risk

**Scoring logic:**
```python
positive_count = len(positive_activities)
total_count = len(all_activities)
sentiment_score = positive_count / total_count if total_count > 0 else 0.5

Examples:
  10 positive / 10 total → 1.00 (all positive!)
  7 positive / 10 total  → 0.70 (mostly positive)
  5 positive / 10 total  → 0.50 (neutral)
  3 positive / 10 total  → 0.30 (mostly negative)
  0 positive / 10 total  → 0.00 (all negative!)
```

**Why measure sentiment:**
- Predicts future engagement
- Early warning of problems
- Positive momentum indicator

#### Step 3d: Recency Scorer

**Drag** "Synthesizer" (position ~700, 340).

**Connect** Parallel Executor (branch 3) → Recency Scorer

**Configure:**
```json
{
  "agent_type": "synthesizer",
  "label": "Recency Score",
  "config": {
    "extract_temporal": true,
    "time_decay_days": 30,
    "normalize": true
  }
}
```

**What this measures:**
- Days since last contact
- Recent contact = higher score
- Exponential decay over time

**Scoring logic:**
```python
days_since = (now - last_contact_timestamp).days
recency_score = max(0.0, 1.0 - (days_since / 30.0))

Examples:
  0 days (today)     → 1.00 (just contacted!)
  7 days (last week) → 0.77 (still fresh)
  15 days (2 weeks)  → 0.50 (cooling off)
  30 days (1 month)  → 0.00 (gone cold)
  60 days (2 months) → 0.00 (very cold)
```

**Why 30-day decay:**
- Matches typical sales cycle
- Adjust based on your business:
  - B2C: 7 days
  - B2B SaaS: 14-30 days
  - Enterprise: 60-90 days

#### Step 4: Knowledge Fusion (Combine Signals)

**Drag** "Knowledge Fusion" below parallel scorers (position ~400, 460).

**Connect** all 4 scorers → Knowledge Fusion

**Configure:**
```json
{
  "agent_type": "knowledge_fusion",
  "label": "Fuse All Signals",
  "config": {
    "max_depth": 2,
    "min_importance": 0.1,
    "fusion_strategy": "weighted_average",
    "preserve_sources": true
  }
}
```

**What this does:**
- Combines 4 independent signals
- Uses knowledge graph to find connections
- Preserves individual scores for transparency

**Fusion example:**
```json
Input signals:
{
  "activity_frequency": 0.75,
  "deal_value": 0.50,
  "engagement_sentiment": 0.85,
  "recency": 0.90
}

Knowledge graph connections:
  Contact → Company → Industry "fintech"
  Contact → Deal "D001" → Stage "proposal"
  Contact → Activity "A123" → Type "demo"

Fused output:
{
  "signals": {
    "activity_frequency": 0.75,
    "deal_value": 0.50,
    "engagement_sentiment": 0.85,
    "recency": 0.90
  },
  "context": {
    "industry": "fintech",
    "deal_stage": "proposal",
    "last_activity_type": "demo"
  },
  "fusion_score": 0.75  // Simple average for now
}
```

**Why max_depth=2:**
- Depth 1: Direct connections (Contact → Activity)
- Depth 2: 2-hop connections (Contact → Deal → Company)
- Higher depths get too noisy

#### Step 5: Thompson Sampling (Adaptive Weighting)

**Drag** "Thompson Sampler" below Knowledge Fusion (position ~400, 580).

**Connect** Knowledge Fusion → Thompson Sampler

**Configure:**
```json
{
  "agent_type": "thompson_sampler",
  "label": "Adaptive Weighting",
  "config": {
    "exploration_rate": 0.1,
    "alpha_prior": 1.0,
    "beta_prior": 1.0,
    "learn_weights": true,
    "update_on_outcome": true
  }
}
```

**What this does:**
- Learns which signals predict success
- Bayesian exploration/exploitation
- Adapts weights over time

**Thompson Sampling explained:**

**Initial state (no data):**
```python
# All signals have equal prior
signals = {
    "activity": Beta(α=1, β=1),     # Uniform prior
    "value": Beta(α=1, β=1),
    "sentiment": Beta(α=1, β=1),
    "recency": Beta(α=1, β=1)
}

# Sample from each distribution
weights = {
    "activity": sample(Beta(1,1)) = 0.52,
    "value": sample(Beta(1,1)) = 0.48,
    "sentiment": sample(Beta(1,1)) = 0.51,
    "recency": sample(Beta(1,1)) = 0.49
}

# Normalize to sum to 1
normalized = {
    "activity": 0.26,
    "value": 0.24,
    "sentiment": 0.26,
    "recency": 0.24
}
```

**After scoring 50 contacts:**
```python
# System learned: high recency → often closes
signals = {
    "activity": Beta(α=15, β=35),   # 15 successes, 35 failures
    "value": Beta(α=12, β=38),      # Not strong predictor
    "sentiment": Beta(α=25, β=25),  # Neutral
    "recency": Beta(α=35, β=15)     # 35 successes! Strong predictor
}

# Sample from learned distributions
weights = {
    "activity": sample(Beta(15,35)) = 0.25,
    "value": sample(Beta(12,38)) = 0.20,
    "sentiment": sample(Beta(25,25)) = 0.50,
    "recency": sample(Beta(35,15)) = 0.75  ← High! System learned this matters
}

# Normalized
normalized = {
    "activity": 0.15,
    "value": 0.12,
    "sentiment": 0.29,
    "recency": 0.44  ← Highest weight!
}
```

**Configuration explained:**

**exploration_rate: 0.1**
- 10% exploration, 90% exploitation
- Exploration: Try different weights to learn
- Exploitation: Use best known weights
- Balance:
  - High (0.3): More learning, less accuracy
  - Low (0.05): Less learning, more accuracy
  - 0.1: Good balance

**alpha_prior: 1.0, beta_prior: 1.0**
- Starting distribution for each signal
- Beta(1,1) = Uniform distribution (no prior knowledge)
- Alternative: Beta(10,10) = Stronger prior toward 0.5

**learn_weights: true**
- Enable weight learning
- `false`: Use fixed weights
- `true`: Adapt over time

**update_on_outcome: true**
- Update distributions after each score
- Requires feedback on whether contact converted
- Can be delayed (batch update weekly)

#### Step 6: Convergence Engine (Final Classification)

**Drag** "Convergence Engine" below Thompson Sampler (position ~400, 700).

**Connect** Thompson Sampler → Convergence Engine

**Configure:**
```json
{
  "agent_type": "convergence_engine",
  "label": "Final Classification",
  "config": {
    "strategy": "bayesian_blend",
    "neural_weight": 0.5,
    "bandit_weight": 0.5
  }
}
```

**What this does:**
- Combines Thompson weights with neural prediction
- 50/50 blend (balanced)
- Outputs final score + classification

**Calculation:**
```python
# Thompson Sampler output
thompson_score = (
    0.15 * activity_frequency +
    0.12 * deal_value +
    0.29 * engagement_sentiment +
    0.44 * recency
) = (0.15 * 0.75) + (0.12 * 0.50) + (0.29 * 0.85) + (0.44 * 0.90)
  = 0.1125 + 0.06 + 0.2465 + 0.396
  = 0.815

# Neural Network output
neural_score = neural_net.predict([
    activity_frequency,
    deal_value,
    engagement_sentiment,
    recency,
    ...other_features
]) = 0.87

# Bayesian Blend
final_score = (0.5 * 0.815) + (0.5 * 0.87)
            = 0.4075 + 0.435
            = 0.8425

# Classification
if final_score >= 0.75:
    classification = "hot"
elif final_score >= 0.50:
    classification = "warm"
elif final_score >= 0.25:
    classification = "cold"
else:
    classification = "dead"

# Result: 0.8425 → "hot"
```

#### Step 7: Memory Store (Update Contact Record)

**Drag** "Memory Store" below Convergence Engine (position ~400, 820).

**Connect** Convergence Engine → Memory Store

**Configure:**
```json
{
  "agent_type": "memory_store",
  "label": "Update Contact Record",
  "config": {
    "backend": "hybrid",
    "update_existing": true,
    "contact_id": "${get_contact.id}",
    "fields_to_update": {
      "lead_score": "${final_score.score}",
      "lead_classification": "${final_score.classification}",
      "last_scored_at": "${timestamp}",
      "score_factors": {
        "activity_frequency": "${activity_scorer.score}",
        "deal_value": "${value_scorer.score}",
        "engagement_sentiment": "${engagement_scorer.score}",
        "recency": "${recency_scorer.score}"
      },
      "learned_weights": "${thompson_sampling.weights}"
    }
  }
}
```

**What this does:**
- Updates existing contact record (not create new)
- Stores score + all factors for transparency
- Saves learned weights for analysis

**Updated contact record:**
```json
{
  "id": "contact_alice_at_techcorp_com",
  "name": "Alice Johnson",
  "email": "alice@techcorp.com",
  "company": "TechCorp",

  // Original fields
  ...existing_fields,

  // NEW: Scoring data
  "lead_score": 0.8425,
  "lead_classification": "hot",
  "last_scored_at": "2025-11-04T10:30:00Z",

  "score_factors": {
    "activity_frequency": 0.75,
    "deal_value": 0.50,
    "engagement_sentiment": 0.85,
    "recency": 0.90
  },

  "learned_weights": {
    "activity_frequency": 0.15,
    "deal_value": 0.12,
    "engagement_sentiment": 0.29,
    "recency": 0.44
  }
}
```

**Why save factors:**
- Transparency: See WHY score is X
- Debugging: Identify weak signals
- Trends: Track factor changes over time
- Appeals: Explain scores to sales team

#### Step 8: Response Generator (Generate Report)

**Drag** "Response Generator" below Memory Store (position ~400, 940).

**Connect** Memory Store → Response Generator

**Configure:**
```json
{
  "agent_type": "response_generator",
  "label": "Generate Report",
  "config": {
    "template": "# 🎯 Lead Score Report\n\n## Contact Information\n**Name**: ${get_contact.name}\n**Company**: ${get_contact.company}\n**Email**: ${get_contact.email}\n\n---\n\n## Overall Score\n**Final Score**: ${final_score.score} / 1.00\n**Classification**: ${final_score.classification}\n\n${#if final_score.classification == 'hot'}\n🔥 **HOT LEAD** - Take immediate action!\n${else if final_score.classification == 'warm'}\n⚡ **WARM LEAD** - Nurture and follow up this week\n${else if final_score.classification == 'cold'}\n❄️ **COLD LEAD** - Long-term nurture\n${else}\n💀 **DEAD LEAD** - Consider archiving\n${/if}\n\n---\n\n## Factor Breakdown\n\n| Factor | Score | Weight | Contribution |\n|--------|-------|--------|-------------|\n| Activity Frequency | ${activity_scorer.score} | ${thompson_sampling.weights.activity} | ${activity_scorer.score * thompson_sampling.weights.activity} |\n| Deal Value | ${value_scorer.score} | ${thompson_sampling.weights.value} | ${value_scorer.score * thompson_sampling.weights.value} |\n| Engagement Sentiment | ${engagement_scorer.score} | ${thompson_sampling.weights.sentiment} | ${engagement_scorer.score * thompson_sampling.weights.sentiment} |\n| Recency | ${recency_scorer.score} | ${thompson_sampling.weights.recency} | ${recency_scorer.score * thompson_sampling.weights.recency} |\n\n---\n\n## Insights\n\n### What's Working\n${#each top_factors}\n✓ **${factor_name}**: ${score} (strong signal)\n${/each}\n\n### Areas to Improve\n${#each weak_factors}\n⚠ **${factor_name}**: ${score} (needs attention)\n${/each}\n\n---\n\n## Recommended Actions\n\n${#if final_score.classification == 'hot'}\n1. Send proposal within 24 hours\n2. Schedule closing call\n3. Involve executive sponsor\n4. Fast-track through legal\n${else if final_score.classification == 'warm'}\n1. Schedule follow-up call this week\n2. Send relevant case study\n3. Introduce to customer success\n4. Continue engagement cadence\n${else}\n1. Add to nurture email sequence\n2. Monthly check-ins\n3. Share industry insights\n4. Re-score quarterly\n${/if}\n\n---\n\n## Thompson Sampling Analysis\n\n**Learned Weights** (after ${thompson_sampling.observations} observations):\n\n${#each thompson_sampling.weights}\n- **${factor_name}**: ${weight} ${#if weight > 0.30}(strong predictor)${else if weight > 0.20}(moderate predictor)${else}(weak predictor)${/if}\n${/each}\n\n**Key Learning**: ${thompson_sampling.insight}\n\n---\n\n*Scored at: ${timestamp}*\n*Model version: ${model_version}*\n*Confidence: ${final_score.confidence}*",
    "format": "markdown"
  }
}
```

**Example output:**
```markdown
# 🎯 Lead Score Report

## Contact Information
**Name**: Alice Johnson
**Company**: TechCorp
**Email**: alice@techcorp.com

---

## Overall Score
**Final Score**: 0.84 / 1.00
**Classification**: hot

🔥 **HOT LEAD** - Take immediate action!

---

## Factor Breakdown

| Factor | Score | Weight | Contribution |
|--------|-------|--------|--------------|
| Activity Frequency | 0.75 | 0.15 | 0.1125 |
| Deal Value | 0.50 | 0.12 | 0.06 |
| Engagement Sentiment | 0.85 | 0.29 | 0.2465 |
| Recency | 0.90 | 0.44 | 0.396 |

---

## Insights

### What's Working
✓ **Recency**: 0.90 (strong signal)
✓ **Engagement Sentiment**: 0.85 (strong signal)

### Areas to Improve
⚠ **Deal Value**: 0.50 (needs attention)

---

## Recommended Actions

1. Send proposal within 24 hours
2. Schedule closing call
3. Involve executive sponsor
4. Fast-track through legal

---

## Thompson Sampling Analysis

**Learned Weights** (after 127 observations):

- **Activity Frequency**: 0.15 (weak predictor)
- **Deal Value**: 0.12 (weak predictor)
- **Engagement Sentiment**: 0.29 (moderate predictor)
- **Recency**: 0.44 (strong predictor)

**Key Learning**: For this business, recency is the strongest predictor of conversion. Contacts contacted within last week are 3.2x more likely to close than those contacted 2+ weeks ago.

---

*Scored at: 2025-11-04 10:30:15*
*Model version: 2.3.1*
*Confidence: 0.91*
```

### Executing the Workflow

#### Save

**Click** "Save" → `multi_factor_scoring.json`

#### Execute

**Click** "Execute"

**Input:**
```json
{
  "email": "alice@techcorp.com"
}
```

**Real-time progress:**
```
✓ Memory Search: Found contact (20ms)
✓ Parallel Executor: Starting 4 branches
  ✓ Activity Frequency: 15 activities found (38ms)
  ✓ Deal Value: 2 deals, $75k total (42ms)
  ✓ Engagement Sentiment: 11/15 positive (35ms)
  ✓ Recency: Last contact 2 days ago (28ms)
✓ Parallel Executor: All branches complete (45ms)
✓ Knowledge Fusion: Fused 4 signals (15ms)
✓ Thompson Sampling: Calculated weights (22ms)
  - Learned from 127 previous observations
  - Updated distributions
✓ Convergence Engine: Final score = 0.84 (18ms)
✓ Memory Store: Updated contact record (12ms)
✓ Response Generator: Generated report (5ms)

Total: 137ms
```

### Understanding Thompson Sampling

#### How It Learns

**Observation 1** (first contact ever scored):
```python
# Prior: Beta(1,1) for all signals
Contact: Bob Smith
Signals: {activity: 0.5, value: 0.8, sentiment: 0.6, recency: 0.9}
Predicted score: 0.70 (weighted average with uniform weights)
Actual outcome: CLOSED (feedback provided next week)

# Update: Increase α for signals that contributed
# (recency was high and contact closed → good signal!)
Updated distributions:
  activity: Beta(1.5, 1.5)   # Moderate contribution
  value: Beta(1.8, 1.2)      # High contribution
  sentiment: Beta(1.6, 1.4)  # Moderate contribution
  recency: Beta(1.9, 1.1)    # Very high contribution
```

**Observation 50:**
```python
# System has learned a lot
Distributions:
  activity: Beta(15, 35)   # Low α/β ratio → weak predictor
  value: Beta(12, 38)      # Low α/β ratio → weak predictor
  sentiment: Beta(25, 25)  # Equal α/β → neutral
  recency: Beta(35, 15)    # High α/β ratio → STRONG predictor!

# Sample weights (exploration)
Sampled weights: {
  activity: 0.15,
  value: 0.12,
  sentiment: 0.29,
  recency: 0.44  ← System learned this is most important!
}
```

#### Providing Feedback

**Option 1: Immediate feedback (within workflow)**
```json
// Add to workflow after deal closes
{
  "agent_type": "feedback_provider",
  "config": {
    "contact_id": "${contact.id}",
    "outcome": "closed",  // or "lost"
    "update_thompson": true
  }
}
```

**Option 2: Batch feedback (weekly)**
```python
# Python script
import requests

# Get all contacts scored this week
scored_contacts = get_contacts_scored_this_week()

# Check which closed
for contact in scored_contacts:
    if contact.deal_closed:
        outcome = "success"
    elif contact.deal_lost:
        outcome = "failure"
    else:
        continue  # Still open, no feedback yet

    # Update Thompson Sampling
    requests.post('http://localhost:8001/api/thompson/update', json={
        'contact_id': contact.id,
        'signals': contact.score_factors,
        'outcome': outcome
    })
```

**Option 3: Manual feedback (CRM UI)**
```markdown
In your CRM dashboard:

[Contact: Alice Johnson]
Lead Score: 0.84 (Hot)

Deal Status: [x] Closed Won  [ ] Closed Lost  [ ] Open

[Update Thompson Sampling]  ← Button
```

### Customization Ideas

#### Add More Signals

**Email engagement:**
```json
{
  "agent_type": "context_retriever",
  "label": "Email Opens",
  "config": {
    "query": "type:email_event AND contact:${name} AND event:open",
    "k": 100
  }
}
```

**Company size:**
```json
{
  "agent_type": "synthesizer",
  "label": "Company Size Score",
  "config": {
    "extract_field": "company.employee_count",
    "normalize_by": 1000  // Score = employees / 1000
  }
}
```

#### Change Blend Ratio

**Trust Thompson more:**
```json
{
  "neural_weight": 0.3,   // Reduce neural
  "bandit_weight": 0.7    // Increase Thompson
}
```

**Trust neural more:**
```json
{
  "neural_weight": 0.8,   // Increase neural
  "bandit_weight": 0.2    // Reduce Thompson
}
```

#### Industry-Specific Scoring

**SaaS (recurring revenue):**
```python
# Weight MRR higher
signals = {
    "mrr": 0.40,
    "expansion_potential": 0.30,
    "engagement": 0.20,
    "recency": 0.10
}
```

**Enterprise (long sales cycles):**
```python
# Weight deal stage higher
signals = {
    "deal_stage": 0.40,
    "executive_engagement": 0.30,
    "champion_identified": 0.20,
    "legal_progress": 0.10
}
```

### Performance Optimization

**Current: ~137ms per contact**

**Optimize to ~80ms:**

1. **Reduce k in Context Retrievers:**
```json
"k": 50 → "k": 20  // Saves ~15ms
```

2. **Disable fusion for known patterns:**
```json
"use_fusion": false  // Saves ~8ms
```

3. **Use "bare" pattern in HoloLoom (if present):**
```json
"pattern": "bare"  // Saves ~50ms
```

4. **Cache Thompson distributions:**
```json
"cache_distributions": true  // Saves ~10ms
```

**Result: 137ms → 74ms (46% faster)**

### Troubleshooting

#### "Thompson weights don't change"

**Problem:** Weights stay uniform after 100 observations

**Causes:**
1. No feedback provided
2. All outcomes same (all close or all lose)
3. `learn_weights: false`

**Solutions:**
1. Implement feedback loop
2. Ensure mix of outcomes
3. Set `learn_weights: true`
4. Check `update_on_outcome: true`

#### "All scores same (0.5)"

**Problem:** Every contact scores 0.5

**Causes:**
1. All signals returning 0.5
2. No variance in data
3. Fusion canceling signals

**Solutions:**
1. Check individual scorers have variance
2. Add more diverse contacts
3. Reduce `min_importance` threshold

#### "Parallel execution fails"

**Problem:** Parallel Executor throws error

**Check:**
1. All branches connected properly
2. Timeout not too low
3. Backend can handle concurrency
4. No circular dependencies

---

*Due to length limits, I'll create Examples 4-6 in a separate response. Would you like me to continue with the remaining examples?*
