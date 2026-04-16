# HoloLoom CRM - Quick Reference Card

**1-page cheat sheet for using HoloLoom as a CRM**

---

## Initialization

```python
from HoloLoom.hololoom import HoloLoom
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.protocol import Memory

loom = HoloLoom()  # Memory system
kg = KG()          # Knowledge graph
```

---

## Creating CRM Entities

### Contact

```python
contact = Memory(
    id="contact_alice_at_techcorp_com",
    text="Alice Johnson, CEO at TechCorp. Email: alice@techcorp.com. Notes: Very interested.",
    timestamp=datetime.now(),
    context={
        'type': 'contact',
        'entities': ['Alice Johnson', 'TechCorp', 'CEO'],
        'motifs': ['contact', 'customer']
    },
    metadata={
        'name': 'Alice Johnson',
        'email': 'alice@techcorp.com',
        'company': 'TechCorp',
        'title': 'CEO',
        'tags': ['decision_maker', 'hot_lead']
    }
)

# Store
await loom.experience(contact.text)

# Add relationship
kg.add_edge(KGEdge("Alice Johnson", "TechCorp", "WORKS_AT"))
```

### Deal

```python
deal = Memory(
    id="deal_D001",
    text="Deal: Enterprise License with TechCorp ($50,000). Stage: proposal. Contact: Alice Johnson.",
    timestamp=datetime.now(),
    context={
        'type': 'deal',
        'entities': ['TechCorp', 'Alice Johnson'],
        'motifs': ['deal', 'sales', 'proposal']
    },
    metadata={
        'deal_id': 'D001',
        'title': 'Enterprise License',
        'company': 'TechCorp',
        'value': 50000.0,
        'stage': 'proposal',
        'contact_name': 'Alice Johnson'
    }
)

# Store
await loom.experience(deal.text)

# Add relationships
kg.add_edge(KGEdge("Alice Johnson", "deal_D001", "ASSOCIATED_WITH"))
kg.add_edge(KGEdge("deal_D001", "TechCorp", "INVOLVES"))
```

### Activity

```python
activity = Memory(
    id=f"activity_{int(datetime.now().timestamp())}",
    text="Call with Alice Johnson: Discussed implementation timeline. Outcome: Alice will review with team. Sentiment: positive.",
    timestamp=datetime.now(),
    context={
        'type': 'activity',
        'entities': ['Alice Johnson'],
        'motifs': ['call', 'positive']
    },
    metadata={
        'activity_type': 'call',
        'contact_name': 'Alice Johnson',
        'summary': 'Discussed implementation timeline',
        'outcome': 'Alice will review with team',
        'sentiment': 'positive'
    }
)

# Store
await loom.experience(activity.text)

# Add relationships
kg.add_edge(KGEdge(activity.id, "Alice Johnson", "RELATES_TO"))
kg.add_edge(KGEdge(activity.id, "deal_D001", "INFLUENCES"))  # If related to deal
```

---

## Querying

### Semantic Search

```python
# Find contacts
results = await loom.recall("Show me decision makers at tech companies")

# Find deals
results = await loom.recall("deals in proposal stage")

# Find activities
results = await loom.recall("recent calls with Alice")
```

### Knowledge Graph

```python
# Get neighbors
neighbors = kg.get_neighbors("Alice Johnson")
# Returns: ['TechCorp', 'deal_D001', ...]

# Get subgraph
subgraph = kg.subgraph_for_entities(["Alice Johnson", "TechCorp"])

# Count nodes/edges
num_nodes = subgraph.number_of_nodes()
num_edges = subgraph.number_of_edges()
```

---

## Lead Scoring (Example)

```python
def calculate_lead_score(contact: Memory, activities: List[Memory], deals: List[Memory]) -> dict:
    """Simple lead scoring algorithm."""

    # Recency (0-1)
    days_since = (datetime.now() - contact.timestamp).days
    recency = max(0.0, 1.0 - (days_since / 30.0))

    # Activity (0-1)
    activity_score = min(1.0, len(activities) / 10.0)

    # Sentiment (0-1)
    positive = sum(1 for a in activities if a.metadata.get('sentiment') == 'positive')
    sentiment = positive / len(activities) if activities else 0.5

    # Value (0-1)
    total_value = sum(d.metadata.get('value', 0) for d in deals)
    value_score = min(1.0, total_value / 100000.0)

    # Decision maker (0-1)
    is_dm = 'decision_maker' in contact.metadata.get('tags', [])
    decision_score = 1.0 if is_dm else 0.3

    # Weighted average
    score = (
        recency * 0.25 +
        activity_score * 0.20 +
        sentiment * 0.20 +
        value_score * 0.20 +
        decision_score * 0.15
    )

    # Classify
    if score >= 0.75:
        level = "hot"
    elif score >= 0.50:
        level = "warm"
    elif score >= 0.25:
        level = "cold"
    else:
        level = "dead"

    return {'score': score, 'level': level}
```

---

## Common Edge Types

| Edge Type | Example | Description |
|-----------|---------|-------------|
| `WORKS_AT` | Contact → Company | Employment |
| `ASSOCIATED_WITH` | Contact → Deal | Deal ownership |
| `INVOLVES` | Deal → Company | Deal participation |
| `RELATES_TO` | Activity → Contact | Activity subject |
| `INFLUENCES` | Activity → Deal | Deal impact |
| `REPORTS_TO` | Contact → Contact | Org hierarchy |
| `COMPETES_WITH` | Company → Company | Competition |

---

## Helper Template

```python
def create_contact_memory(name, email, company, title, notes, tags=None):
    """Quick contact creation."""
    return Memory(
        id=f"contact_{email.replace('@', '_at_')}",
        text=f"{name}, {title} at {company}. Email: {email}. Notes: {notes}",
        timestamp=datetime.now(),
        context={'type': 'contact', 'entities': [name, company, title]},
        metadata={'name': name, 'email': email, 'company': company,
                  'title': title, 'tags': tags or []}
    )
```

---

## Running the Demo

```bash
# From repository root
PYTHONPATH=. python crm_demo.py
```

---

## Full Documentation

See [CRM_WITH_HOLOLOOM_GUIDE.md](CRM_WITH_HOLOLOOM_GUIDE.md) for complete examples.
