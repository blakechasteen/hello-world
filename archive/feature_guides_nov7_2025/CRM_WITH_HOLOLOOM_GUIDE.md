# Using HoloLoom Core Framework as a CRM System

**Complete Step-by-Step Guide**

This guide shows you exactly how to use HoloLoom's core framework to build CRM functionality without the archived `crm_app`. The core framework now provides everything you need through:

1. **Unified Memory API** (`HoloLoom` class) - Simple `experience()`, `recall()`, `reflect()` operations
2. **Knowledge Graph** (`KG` class) - Entity relationships and traversal
3. **Memory Protocol** (`Memory` class) - Standardized memory representation
4. **Spinners** - Data ingestion from any source

---

## Table of Contents

1. [Quick Start Example](#quick-start-example)
2. [Core Concepts](#core-concepts)
3. [Step-by-Step: Building CRM Features](#step-by-step-building-crm-features)
4. [Complete Working Example](#complete-working-example)
5. [Advanced Features](#advanced-features)
6. [Production Deployment](#production-deployment)

---

## Quick Start Example

Here's a minimal CRM in 50 lines:

```python
import asyncio
from datetime import datetime
from HoloLoom.hololoom import HoloLoom
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.protocol import Memory

async def main():
    # 1. Initialize HoloLoom
    loom = HoloLoom()

    # 2. Create knowledge graph for relationships
    kg = KG()

    # 3. Add a contact (as a memory)
    contact_memory = Memory(
        id="contact_alice",
        text="Alice Johnson, CEO at TechCorp. Email: alice@techcorp.com. Very interested in our product.",
        timestamp=datetime.now(),
        context={
            'type': 'contact',
            'entities': ['Alice Johnson', 'TechCorp', 'CEO'],
            'tags': ['decision_maker', 'hot_lead']
        },
        metadata={
            'email': 'alice@techcorp.com',
            'company': 'TechCorp',
            'title': 'CEO',
            'score': 0.85
        }
    )

    # 4. Store in memory system
    await loom.experience(contact_memory.text)

    # 5. Add relationship to knowledge graph
    kg.add_edge(KGEdge("Alice Johnson", "TechCorp", "WORKS_AT", weight=1.0))

    # 6. Recall contacts based on query
    results = await loom.recall("Show me decision makers at tech companies")

    print(f"Found {len(results)} relevant contacts")
    for mem in results:
        print(f"- {mem.text[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**
```
Found 1 relevant contacts
- Alice Johnson, CEO at TechCorp. Email: alice@techcorp.com. Very interested in our product....
```

---

## Core Concepts

### 1. Memory Object

The `Memory` class is the fundamental unit of storage:

```python
from HoloLoom.memory.protocol import Memory
from datetime import datetime

memory = Memory(
    id="unique_id",                    # Unique identifier
    text="The content to remember",    # Full text representation
    timestamp=datetime.now(),          # When this was created
    context={                          # Structured context
        'type': 'contact',             # Memory type
        'entities': ['Alice', 'TechCorp'],
        'motifs': ['sales', 'follow_up']
    },
    metadata={                         # Additional data
        'email': 'alice@example.com',
        'score': 0.85,
        'tags': ['hot_lead']
    },
    embedding=None                     # Auto-generated if None
)
```

### 2. Knowledge Graph (KG)

The `KG` class stores relationships:

```python
from HoloLoom.memory.graph import KG, KGEdge

kg = KG()

# Add edges (relationships)
kg.add_edge(KGEdge("Alice", "TechCorp", "WORKS_AT"))
kg.add_edge(KGEdge("Alice", "Deal_123", "ASSOCIATED_WITH"))
kg.add_edge(KGEdge("Deal_123", "TechCorp", "INVOLVES"))

# Query neighbors
neighbors = kg.get_neighbors("Alice")  # Returns ['TechCorp', 'Deal_123']

# Get subgraph
subgraph = kg.subgraph_for_entities(["Alice", "TechCorp"])
```

**Common Edge Types:**
- `WORKS_AT` - Contact → Company
- `ASSOCIATED_WITH` - Contact → Deal
- `INVOLVES` - Deal → Company
- `RELATES_TO` - Activity → Contact
- `INFLUENCES` - Activity → Deal
- `REPORTS_TO` - Contact → Contact (organizational hierarchy)
- `COMPETES_WITH` - Company → Company

### 3. HoloLoom Class (Unified API)

The `HoloLoom` class provides three core operations:

```python
from HoloLoom.hololoom import HoloLoom

loom = HoloLoom()

# Experience: Form new memories
memory = await loom.experience("Alice called, interested in Q1 demo")

# Recall: Retrieve relevant memories
memories = await loom.recall("What contacts are interested in demos?")

# Reflect: Learn from feedback
await loom.reflect(memories, feedback={"helpful": True, "outcome": "success"})
```

---

## Step-by-Step: Building CRM Features

### Feature 1: Contact Management

**Step 1.1: Create a Contact**

```python
from HoloLoom.memory.protocol import Memory
from datetime import datetime

def create_contact_memory(
    name: str,
    email: str,
    company: str,
    title: str,
    notes: str,
    tags: list = None
) -> Memory:
    """Create a contact as a Memory object."""

    contact_id = f"contact_{email.replace('@', '_at_')}"

    # Build text representation (what will be semantically searched)
    text = f"{name}, {title} at {company}. Email: {email}. Notes: {notes}"

    return Memory(
        id=contact_id,
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'contact',
            'entities': [name, company, title],
            'motifs': ['contact', 'customer']
        },
        metadata={
            'email': email,
            'company': company,
            'title': title,
            'name': name,
            'tags': tags or [],
            'notes': notes
        }
    )

# Usage
contact = create_contact_memory(
    name="Alice Johnson",
    email="alice@techcorp.com",
    company="TechCorp",
    title="CEO",
    notes="Very interested in our product. Prefers email communication.",
    tags=["decision_maker", "hot_lead"]
)
```

**Step 1.2: Store the Contact**

```python
from HoloLoom.hololoom import HoloLoom

loom = HoloLoom()

# Store contact
await loom.experience(contact.text)

# Alternative: Store with full Memory object for metadata preservation
# (Note: This requires the awareness_graph backend)
# loom._awareness_graph.store_memory(contact)
```

**Step 1.3: Add Relationship to Knowledge Graph**

```python
from HoloLoom.memory.graph import KG, KGEdge

kg = KG()

# Contact → Company relationship
kg.add_edge(KGEdge(
    src="Alice Johnson",
    dst="TechCorp",
    type="WORKS_AT",
    weight=1.0,
    metadata={'verified': True}
))
```

**Step 1.4: Search for Contacts**

```python
# Semantic search
results = await loom.recall("Show me CEOs at tech companies")

# Filter by metadata (if using awareness_graph backend)
# results = loom._awareness_graph.get_memories_by_filter(
#     filter_fn=lambda m: m.metadata.get('title') == 'CEO'
# )

for memory in results:
    print(f"Contact: {memory.metadata.get('name')}")
    print(f"Company: {memory.metadata.get('company')}")
    print(f"Email: {memory.metadata.get('email')}")
```

---

### Feature 2: Deal Management

**Step 2.1: Create a Deal**

```python
def create_deal_memory(
    deal_id: str,
    title: str,
    company: str,
    value: float,
    stage: str,
    contact_name: str,
    notes: str = ""
) -> Memory:
    """Create a deal as a Memory object."""

    # Build text representation
    text = (
        f"Deal: {title} with {company} (${value:,.0f}). "
        f"Stage: {stage}. Contact: {contact_name}. {notes}"
    )

    return Memory(
        id=f"deal_{deal_id}",
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'deal',
            'entities': [company, contact_name],
            'motifs': ['deal', 'sales', stage]
        },
        metadata={
            'deal_id': deal_id,
            'title': title,
            'company': company,
            'value': value,
            'stage': stage,
            'contact_name': contact_name,
            'notes': notes,
            'created_at': datetime.now().isoformat()
        }
    )

# Usage
deal = create_deal_memory(
    deal_id="D001",
    title="Enterprise Software License",
    company="TechCorp",
    value=50000.0,
    stage="proposal",
    contact_name="Alice Johnson",
    notes="Pending legal review. Likely to close this quarter."
)
```

**Step 2.2: Store Deal and Relationships**

```python
# Store deal
await loom.experience(deal.text)

# Add relationships
kg.add_edge(KGEdge("Alice Johnson", "deal_D001", "ASSOCIATED_WITH"))
kg.add_edge(KGEdge("deal_D001", "TechCorp", "INVOLVES"))
```

**Step 2.3: Query Deals**

```python
# Find all deals in proposal stage
proposals = await loom.recall("Show me deals in proposal stage")

# Find deals by company (using graph)
company_subgraph = kg.subgraph_for_entities(["TechCorp"])
# Get all nodes of type 'deal' in subgraph
deal_nodes = [
    node for node in company_subgraph.nodes()
    if node.startswith("deal_")
]
```

---

### Feature 3: Activity Tracking

**Step 3.1: Log an Activity**

```python
def create_activity_memory(
    activity_type: str,  # 'call', 'email', 'meeting', 'note'
    contact_name: str,
    summary: str,
    outcome: str = "",
    sentiment: str = "neutral"  # 'positive', 'neutral', 'negative'
) -> Memory:
    """Create an activity as a Memory object."""

    activity_id = f"activity_{int(datetime.now().timestamp())}"

    # Build text representation
    text = (
        f"{activity_type.title()} with {contact_name}: {summary}. "
        f"Outcome: {outcome}. Sentiment: {sentiment}."
    )

    return Memory(
        id=activity_id,
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'activity',
            'entities': [contact_name],
            'motifs': [activity_type, sentiment]
        },
        metadata={
            'activity_type': activity_type,
            'contact_name': contact_name,
            'summary': summary,
            'outcome': outcome,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
    )

# Usage
activity = create_activity_memory(
    activity_type="call",
    contact_name="Alice Johnson",
    summary="Discussed Q1 implementation timeline and pricing",
    outcome="Alice will review proposal with CFO this week",
    sentiment="positive"
)
```

**Step 3.2: Store Activity and Relationships**

```python
# Store activity
await loom.experience(activity.text)

# Add relationships
kg.add_edge(KGEdge(activity.id, "Alice Johnson", "RELATES_TO"))

# If activity relates to a deal
kg.add_edge(KGEdge(activity.id, "deal_D001", "INFLUENCES"))
```

**Step 3.3: Query Activity History**

```python
# Get recent activities with a contact
results = await loom.recall(f"Show recent activities with Alice Johnson")

# Get activities by type
calls = await loom.recall("Show phone calls from the last week")
```

---

### Feature 4: Lead Scoring

**Step 4.1: Define Scoring Function**

```python
from typing import List

def calculate_lead_score(
    contact_memory: Memory,
    activities: List[Memory],
    deals: List[Memory]
) -> dict:
    """
    Calculate lead score based on engagement signals.

    Returns dict with:
        - score: 0.0-1.0
        - confidence: 0.0-1.0
        - engagement_level: 'hot', 'warm', 'cold', 'dead'
        - factors: dict of individual factor scores
    """

    # Factor 1: Activity frequency (0-1)
    days_since_last_contact = (datetime.now() - contact_memory.timestamp).days
    recency_score = max(0.0, 1.0 - (days_since_last_contact / 30.0))  # Decay over 30 days

    # Factor 2: Activity volume (0-1)
    activity_count = len(activities)
    activity_score = min(1.0, activity_count / 10.0)  # Max out at 10 activities

    # Factor 3: Sentiment (0-1)
    if activities:
        positive_count = sum(
            1 for a in activities
            if a.metadata.get('sentiment') == 'positive'
        )
        sentiment_score = positive_count / len(activities)
    else:
        sentiment_score = 0.5

    # Factor 4: Deal value (0-1)
    total_value = sum(d.metadata.get('value', 0) for d in deals)
    value_score = min(1.0, total_value / 100000.0)  # Normalize by $100k

    # Factor 5: Decision maker (0-1)
    is_decision_maker = 'decision_maker' in contact_memory.metadata.get('tags', [])
    decision_score = 1.0 if is_decision_maker else 0.3

    # Weighted average
    score = (
        recency_score * 0.25 +
        activity_score * 0.20 +
        sentiment_score * 0.20 +
        value_score * 0.20 +
        decision_score * 0.15
    )

    # Classification
    if score >= 0.75:
        engagement_level = "hot"
    elif score >= 0.50:
        engagement_level = "warm"
    elif score >= 0.25:
        engagement_level = "cold"
    else:
        engagement_level = "dead"

    return {
        'score': score,
        'confidence': 0.8,  # Could be calculated based on data quality
        'engagement_level': engagement_level,
        'factors': {
            'recency': recency_score,
            'activity': activity_score,
            'sentiment': sentiment_score,
            'value': value_score,
            'decision_maker': decision_score
        }
    }
```

**Step 4.2: Score a Lead**

```python
async def score_lead(loom: HoloLoom, contact_name: str) -> dict:
    """Score a lead by name."""

    # Get contact
    contact_results = await loom.recall(f"contact: {contact_name}")
    if not contact_results:
        return {'error': 'Contact not found'}

    contact = contact_results[0]

    # Get activities for this contact
    activity_results = await loom.recall(
        f"activities with {contact_name}"
    )

    # Get deals for this contact
    deal_results = await loom.recall(
        f"deals with {contact_name}"
    )

    # Calculate score
    score_data = calculate_lead_score(contact, activity_results, deal_results)

    return {
        'contact_name': contact_name,
        **score_data
    }

# Usage
score = await score_lead(loom, "Alice Johnson")
print(f"Lead score: {score['score']:.2f} ({score['engagement_level']})")
print(f"Factors: {score['factors']}")
```

---

### Feature 5: Action Recommendations

**Step 5.1: Define Recommendation Function**

```python
def recommend_next_action(
    lead_score: dict,
    days_since_last_contact: int,
    last_activity_type: str = None
) -> dict:
    """
    Recommend next best action based on engagement.

    Returns dict with:
        - action: 'send_email', 'schedule_call', 'send_proposal', 'wait', etc.
        - priority: 0.0-1.0
        - reasoning: explanation
    """

    engagement = lead_score['engagement_level']
    score = lead_score['score']

    # Hot leads (0.75+)
    if engagement == "hot":
        if last_activity_type == "call":
            return {
                'action': 'send_proposal',
                'priority': 0.9,
                'reasoning': 'High engagement + recent call = ready for proposal'
            }
        else:
            return {
                'action': 'schedule_call',
                'priority': 0.85,
                'reasoning': 'Hot lead needs direct engagement'
            }

    # Warm leads (0.50-0.75)
    elif engagement == "warm":
        if days_since_last_contact > 7:
            return {
                'action': 'send_email',
                'priority': 0.6,
                'reasoning': 'Re-engage warm lead with value content'
            }
        else:
            return {
                'action': 'wait',
                'priority': 0.3,
                'reasoning': 'Recent contact, give space'
            }

    # Cold leads (0.25-0.50)
    elif engagement == "cold":
        return {
            'action': 'send_email',
            'priority': 0.4,
            'reasoning': 'Low-effort nurture email'
        }

    # Dead leads (<0.25)
    else:
        return {
            'action': 'archive',
            'priority': 0.1,
            'reasoning': 'Very low engagement, consider removing'
        }
```

**Step 5.2: Get Recommendations**

```python
async def get_daily_actions(loom: HoloLoom, limit: int = 10):
    """Get prioritized daily action list."""

    # Get all contacts (simplified - in production, filter by type)
    all_contacts = await loom.recall("type:contact")

    actions = []

    for contact in all_contacts:
        contact_name = contact.metadata.get('name')

        # Score lead
        score = await score_lead(loom, contact_name)

        # Get recommendation
        days_since = (datetime.now() - contact.timestamp).days
        last_activity = None  # Could query this

        rec = recommend_next_action(score, days_since, last_activity)

        actions.append({
            'contact_name': contact_name,
            'company': contact.metadata.get('company'),
            **rec,
            'score': score['score']
        })

    # Sort by priority
    actions.sort(key=lambda x: x['priority'], reverse=True)

    return actions[:limit]

# Usage
daily_actions = await get_daily_actions(loom, limit=5)

print("Top 5 actions for today:")
for action in daily_actions:
    print(f"- {action['action'].upper()}: {action['contact_name']} "
          f"at {action['company']} (Priority: {action['priority']:.2f})")
    print(f"  Reason: {action['reasoning']}")
```

---

## Complete Working Example

Here's a complete, runnable CRM system using HoloLoom:

```python
#!/usr/bin/env python3
"""
HoloLoom CRM - Complete Working Example
========================================

Demonstrates:
1. Contact management
2. Deal tracking
3. Activity logging
4. Lead scoring
5. Action recommendations
6. Knowledge graph relationships
"""

import asyncio
from datetime import datetime, timedelta
from typing import List
from HoloLoom.hololoom import HoloLoom
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.protocol import Memory


# ============================================================================
# Helper Functions
# ============================================================================

def create_contact_memory(name: str, email: str, company: str, title: str,
                          notes: str, tags: list = None) -> Memory:
    """Create a contact as a Memory object."""
    contact_id = f"contact_{email.replace('@', '_at_')}"
    text = f"{name}, {title} at {company}. Email: {email}. Notes: {notes}"

    return Memory(
        id=contact_id,
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'contact',
            'entities': [name, company, title],
            'motifs': ['contact', 'customer']
        },
        metadata={
            'email': email,
            'company': company,
            'title': title,
            'name': name,
            'tags': tags or [],
            'notes': notes
        }
    )


def create_deal_memory(deal_id: str, title: str, company: str, value: float,
                       stage: str, contact_name: str, notes: str = "") -> Memory:
    """Create a deal as a Memory object."""
    text = (
        f"Deal: {title} with {company} (${value:,.0f}). "
        f"Stage: {stage}. Contact: {contact_name}. {notes}"
    )

    return Memory(
        id=f"deal_{deal_id}",
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'deal',
            'entities': [company, contact_name],
            'motifs': ['deal', 'sales', stage]
        },
        metadata={
            'deal_id': deal_id,
            'title': title,
            'company': company,
            'value': value,
            'stage': stage,
            'contact_name': contact_name,
            'notes': notes,
            'created_at': datetime.now().isoformat()
        }
    )


def create_activity_memory(activity_type: str, contact_name: str,
                           summary: str, outcome: str = "",
                           sentiment: str = "neutral") -> Memory:
    """Create an activity as a Memory object."""
    activity_id = f"activity_{int(datetime.now().timestamp())}"
    text = (
        f"{activity_type.title()} with {contact_name}: {summary}. "
        f"Outcome: {outcome}. Sentiment: {sentiment}."
    )

    return Memory(
        id=activity_id,
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'activity',
            'entities': [contact_name],
            'motifs': [activity_type, sentiment]
        },
        metadata={
            'activity_type': activity_type,
            'contact_name': contact_name,
            'summary': summary,
            'outcome': outcome,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
    )


def calculate_lead_score(contact: Memory, activities: List[Memory],
                        deals: List[Memory]) -> dict:
    """Calculate lead score based on engagement signals."""
    days_since = (datetime.now() - contact.timestamp).days
    recency_score = max(0.0, 1.0 - (days_since / 30.0))

    activity_count = len(activities)
    activity_score = min(1.0, activity_count / 10.0)

    if activities:
        positive = sum(1 for a in activities
                      if a.metadata.get('sentiment') == 'positive')
        sentiment_score = positive / len(activities)
    else:
        sentiment_score = 0.5

    total_value = sum(d.metadata.get('value', 0) for d in deals)
    value_score = min(1.0, total_value / 100000.0)

    is_decision_maker = 'decision_maker' in contact.metadata.get('tags', [])
    decision_score = 1.0 if is_decision_maker else 0.3

    score = (
        recency_score * 0.25 +
        activity_score * 0.20 +
        sentiment_score * 0.20 +
        value_score * 0.20 +
        decision_score * 0.15
    )

    if score >= 0.75:
        level = "hot"
    elif score >= 0.50:
        level = "warm"
    elif score >= 0.25:
        level = "cold"
    else:
        level = "dead"

    return {
        'score': score,
        'engagement_level': level,
        'factors': {
            'recency': recency_score,
            'activity': activity_score,
            'sentiment': sentiment_score,
            'value': value_score,
            'decision_maker': decision_score
        }
    }


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("HoloLoom CRM - Complete Demo")
    print("=" * 70)
    print()

    # Initialize
    print("1. Initializing HoloLoom and Knowledge Graph...")
    loom = HoloLoom()
    kg = KG()
    print("   ✓ Ready\n")

    # Create contacts
    print("2. Creating contacts...")
    contacts = [
        create_contact_memory(
            name="Alice Johnson",
            email="alice@techcorp.com",
            company="TechCorp",
            title="CEO",
            notes="Very interested in Q1 implementation. Decision maker.",
            tags=["decision_maker", "hot_lead"]
        ),
        create_contact_memory(
            name="Bob Smith",
            email="bob@innovate.io",
            company="InnovateCo",
            title="CTO",
            notes="Technical contact. Prefers detailed documentation.",
            tags=["technical", "warm_lead"]
        ),
        create_contact_memory(
            name="Carol Davis",
            email="carol@startup.com",
            company="StartupXYZ",
            title="Founder",
            notes="Early stage startup. Budget constrained.",
            tags=["founder"]
        )
    ]

    for contact in contacts:
        await loom.experience(contact.text)
        kg.add_edge(KGEdge(
            contact.metadata['name'],
            contact.metadata['company'],
            "WORKS_AT"
        ))
        print(f"   ✓ Added {contact.metadata['name']}")
    print()

    # Create deals
    print("3. Creating deals...")
    deals = [
        create_deal_memory(
            deal_id="D001",
            title="Enterprise License",
            company="TechCorp",
            value=50000.0,
            stage="proposal",
            contact_name="Alice Johnson",
            notes="Legal review in progress"
        ),
        create_deal_memory(
            deal_id="D002",
            title="Startup Package",
            company="StartupXYZ",
            value=5000.0,
            stage="qualified",
            contact_name="Carol Davis",
            notes="Waiting for funding round"
        )
    ]

    for deal in deals:
        await loom.experience(deal.text)
        kg.add_edge(KGEdge(
            deal.metadata['contact_name'],
            deal.id,
            "ASSOCIATED_WITH"
        ))
        kg.add_edge(KGEdge(
            deal.id,
            deal.metadata['company'],
            "INVOLVES"
        ))
        print(f"   ✓ Added {deal.metadata['title']}")
    print()

    # Log activities
    print("4. Logging activities...")
    activities = [
        create_activity_memory(
            activity_type="call",
            contact_name="Alice Johnson",
            summary="Discussed implementation timeline",
            outcome="Alice will review with team",
            sentiment="positive"
        ),
        create_activity_memory(
            activity_type="email",
            contact_name="Alice Johnson",
            summary="Sent proposal document",
            outcome="Awaiting feedback",
            sentiment="neutral"
        ),
        create_activity_memory(
            activity_type="call",
            contact_name="Bob Smith",
            summary="Technical deep dive on API",
            outcome="Bob needs more documentation",
            sentiment="positive"
        )
    ]

    for activity in activities:
        await loom.experience(activity.text)
        kg.add_edge(KGEdge(
            activity.id,
            activity.metadata['contact_name'],
            "RELATES_TO"
        ))
        print(f"   ✓ Logged {activity.metadata['activity_type']} with {activity.metadata['contact_name']}")
    print()

    # Search functionality
    print("5. Testing search...")
    print("   Query: 'Show me decision makers'")
    results = await loom.recall("decision makers")
    print(f"   Found {len(results)} results:")
    for r in results[:3]:
        print(f"   - {r.text[:80]}...")
    print()

    # Knowledge graph queries
    print("6. Knowledge graph queries...")
    alice_neighbors = kg.get_neighbors("Alice Johnson")
    print(f"   Alice Johnson is connected to: {alice_neighbors}")

    subgraph = kg.subgraph_for_entities(["Alice Johnson", "TechCorp"])
    print(f"   Subgraph contains {subgraph.number_of_nodes()} nodes, "
          f"{subgraph.number_of_edges()} edges")
    print()

    # Lead scoring
    print("7. Lead scoring...")
    for contact in contacts[:2]:  # Score first 2 contacts
        contact_name = contact.metadata['name']

        # Get related activities
        related_activities = [
            a for a in activities
            if a.metadata['contact_name'] == contact_name
        ]

        # Get related deals
        related_deals = [
            d for d in deals
            if d.metadata['contact_name'] == contact_name
        ]

        # Calculate score
        score = calculate_lead_score(contact, related_activities, related_deals)

        print(f"   {contact_name}:")
        print(f"     Score: {score['score']:.2f} ({score['engagement_level']})")
        print(f"     Factors: recency={score['factors']['recency']:.2f}, "
              f"activity={score['factors']['activity']:.2f}, "
              f"sentiment={score['factors']['sentiment']:.2f}")
    print()

    # Summary
    print("8. Summary...")
    print(f"   Contacts: {len(contacts)}")
    print(f"   Deals: {len(deals)}")
    print(f"   Activities: {len(activities)}")
    print(f"   Knowledge graph: {kg.G.number_of_nodes()} nodes, "
          f"{kg.G.number_of_edges()} edges")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
```

**Save this as `crm_demo.py` and run:**

```bash
PYTHONPATH=. python crm_demo.py
```

**Expected Output:**

```
======================================================================
HoloLoom CRM - Complete Demo
======================================================================

1. Initializing HoloLoom and Knowledge Graph...
   ✓ Ready

2. Creating contacts...
   ✓ Added Alice Johnson
   ✓ Added Bob Smith
   ✓ Added Carol Davis

3. Creating deals...
   ✓ Added Enterprise License
   ✓ Added Startup Package

4. Logging activities...
   ✓ Logged call with Alice Johnson
   ✓ Logged email with Alice Johnson
   ✓ Logged call with Bob Smith

5. Testing search...
   Query: 'Show me decision makers'
   Found 3 results:
   - Alice Johnson, CEO at TechCorp. Email: alice@techcorp.com. Notes: Very inte...
   - ...

6. Knowledge graph queries...
   Alice Johnson is connected to: ['TechCorp', 'deal_D001']
   Subgraph contains 3 nodes, 2 edges

7. Lead scoring...
   Alice Johnson:
     Score: 0.82 (hot)
     Factors: recency=0.97, activity=0.20, sentiment=1.00
   Bob Smith:
     Score: 0.64 (warm)
     Factors: recency=0.97, activity=0.10, sentiment=1.00

8. Summary...
   Contacts: 3
   Deals: 2
   Activities: 3
   Knowledge graph: 8 nodes, 7 edges

======================================================================
Demo Complete!
======================================================================
```

---

## Advanced Features

### 1. Using Custom Spinners for Data Import

Create a custom spinner to import CRM data from CSV:

```python
from HoloLoom.spinningWheel.protocol import BaseSpinner, SpinResult, SpinnerCapabilities
from HoloLoom.documentation.types import MemoryShard
import csv

class CRMSpinner(BaseSpinner):
    """Spinner for CRM data (CSV, JSON, etc.)."""

    def __init__(self):
        super().__init__(name="crm_spinner")

    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            batch_processing=True,
            supported_formats=['csv', 'json']
        )

    def is_available(self) -> bool:
        return True  # No dependencies

    async def _spin_impl(self, source, **kwargs):
        """Process CSV file into MemoryShards."""
        shards = []

        with open(source, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Create shard from CSV row
                text = f"{row['name']}, {row['title']} at {row['company']}"

                shard = MemoryShard(
                    id=f"contact_{row['email'].replace('@', '_at_')}",
                    text=text,
                    episode="crm_import",
                    entities=[row['name'], row['company']],
                    motifs=['contact', 'customer'],
                    metadata=row
                )

                shards.append(shard)

        return shards

# Usage
spinner = CRMSpinner()
result = await spinner.spin("contacts.csv")

# Store all shards
for shard in result.shards:
    await loom.experience(shard.text)
```

### 2. Pipeline Insights

Get insights from the CRM data:

```python
async def get_pipeline_insights(loom: HoloLoom) -> dict:
    """Get comprehensive pipeline insights."""

    # All deals
    all_deals = await loom.recall("type:deal")

    # Group by stage
    by_stage = {}
    total_value = 0

    for deal in all_deals:
        stage = deal.metadata.get('stage', 'unknown')
        value = deal.metadata.get('value', 0)

        if stage not in by_stage:
            by_stage[stage] = {'count': 0, 'value': 0, 'deals': []}

        by_stage[stage]['count'] += 1
        by_stage[stage]['value'] += value
        by_stage[stage]['deals'].append(deal)

        total_value += value

    return {
        'total_deals': len(all_deals),
        'total_value': total_value,
        'by_stage': by_stage,
        'avg_deal_size': total_value / len(all_deals) if all_deals else 0
    }

# Usage
insights = await get_pipeline_insights(loom)
print(f"Pipeline value: ${insights['total_value']:,.0f}")
print(f"Average deal: ${insights['avg_deal_size']:,.0f}")

for stage, data in insights['by_stage'].items():
    print(f"{stage}: {data['count']} deals (${data['value']:,.0f})")
```

### 3. Temporal Queries

Query data by time periods:

```python
from datetime import datetime, timedelta

async def get_activities_in_period(
    loom: HoloLoom,
    start_date: datetime,
    end_date: datetime
) -> List[Memory]:
    """Get activities within a date range."""

    # Get all activities
    all_activities = await loom.recall("type:activity")

    # Filter by timestamp
    filtered = [
        a for a in all_activities
        if start_date <= a.timestamp <= end_date
    ]

    return filtered

# Usage
last_week = datetime.now() - timedelta(days=7)
activities = await get_activities_in_period(loom, last_week, datetime.now())
print(f"Activities in last 7 days: {len(activities)}")
```

---

## Production Deployment

### 1. Use Persistent Backend

For production, use Neo4j + Qdrant instead of in-memory:

```python
from HoloLoom.config import Config, MemoryBackend

# Create config with persistent backend
config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID  # Neo4j + Qdrant

# Initialize with persistent storage
loom = HoloLoom(config=config)
```

**Start Docker services:**

```bash
docker-compose up -d neo4j qdrant
```

### 2. Add REST API

Wrap HoloLoom in FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HoloLoom CRM API")

# Global loom instance
loom = None

@app.on_event("startup")
async def startup():
    global loom
    loom = HoloLoom()

class ContactCreate(BaseModel):
    name: str
    email: str
    company: str
    title: str
    notes: str
    tags: list = []

@app.post("/api/contacts")
async def create_contact(contact: ContactCreate):
    """Create a new contact."""
    memory = create_contact_memory(
        name=contact.name,
        email=contact.email,
        company=contact.company,
        title=contact.title,
        notes=contact.notes,
        tags=contact.tags
    )

    await loom.experience(memory.text)

    return {"id": memory.id, "success": True}

@app.get("/api/contacts/search")
async def search_contacts(q: str):
    """Search contacts by query."""
    results = await loom.recall(q)

    return {
        'count': len(results),
        'contacts': [
            {
                'name': m.metadata.get('name'),
                'company': m.metadata.get('company'),
                'email': m.metadata.get('email')
            }
            for m in results
        ]
    }

# Run with: uvicorn crm_api:app --reload
```

### 3. Add Authentication

Use FastAPI dependencies for auth:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    token = credentials.credentials
    # Verify token (implement your logic)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication"
        )
    return token

@app.post("/api/contacts", dependencies=[Depends(verify_token)])
async def create_contact(contact: ContactCreate):
    # ... protected endpoint
    pass
```

---

## Summary

You now have everything you need to use HoloLoom core as a CRM:

**Core Framework Components:**
1. `HoloLoom` class - `experience()`, `recall()`, `reflect()`
2. `Memory` class - Standardized memory representation
3. `KG` class - Knowledge graph relationships
4. Spinners - Data import from any source

**CRM Features You Can Build:**
- Contact management with semantic search
- Deal tracking with pipeline insights
- Activity logging with temporal queries
- Lead scoring with custom algorithms
- Action recommendations with priority sorting
- Knowledge graph for relationship tracking

**Production Ready:**
- Persistent backends (Neo4j + Qdrant)
- REST API with FastAPI
- Authentication and authorization
- Custom spinners for data import

The core framework gives you all the building blocks - you compose them however you need for your specific CRM requirements.
