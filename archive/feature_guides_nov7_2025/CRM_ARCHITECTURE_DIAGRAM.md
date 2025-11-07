# HoloLoom CRM - Architecture Diagram

Visual representation of how to use HoloLoom core as a CRM system.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your CRM Application                         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Contacts   │  │    Deals     │  │  Activities  │         │
│  │              │  │              │  │              │         │
│  │ - Add        │  │ - Create     │  │ - Log call   │         │
│  │ - Search     │  │ - Track      │  │ - Send email │         │
│  │ - Score      │  │ - Forecast   │  │ - Meeting    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  HoloLoom Core Components                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Memory (protocol.Memory)                                │  │
│  │  ─────────────────────────                               │  │
│  │  Standardized data representation:                       │  │
│  │  - id: Unique identifier                                 │  │
│  │  - text: Full text representation                        │  │
│  │  - timestamp: When created                               │  │
│  │  - context: Structured data (type, entities, motifs)     │  │
│  │  - metadata: Additional properties                       │  │
│  │  - embedding: Vector representation (auto-generated)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Knowledge Graph (graph.KG)                              │  │
│  │  ──────────────────────────                              │  │
│  │  NetworkX MultiDiGraph for relationships:                │  │
│  │  - add_edge(KGEdge): Add relationships                   │  │
│  │  - get_neighbors(entity): Find connected entities        │  │
│  │  - subgraph_for_entities([...]): Extract subgraphs       │  │
│  │                                                           │  │
│  │  Common edge types:                                       │  │
│  │  - WORKS_AT: Contact → Company                           │  │
│  │  - ASSOCIATED_WITH: Contact → Deal                       │  │
│  │  - INVOLVES: Deal → Company                              │  │
│  │  - RELATES_TO: Activity → Contact                        │  │
│  │  - INFLUENCES: Activity → Deal                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  HoloLoom Class (hololoom.HoloLoom)                      │  │
│  │  ──────────────────────────────────                      │  │
│  │  Unified memory system:                                   │  │
│  │  - experience(text): Form new memories                   │  │
│  │  - recall(query): Retrieve relevant memories             │  │
│  │  - reflect(feedback): Learn from outcomes                │  │
│  │                                                           │  │
│  │  Features:                                                │  │
│  │  - Semantic search across all data                       │  │
│  │  - Natural language queries                              │  │
│  │  - Automatic embedding generation                        │  │
│  │  - Context-aware retrieval                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Storage Backends                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  In-Memory   │  │    Neo4j     │  │   Qdrant     │         │
│  │  (Default)   │  │  (Graph DB)  │  │  (Vector DB) │         │
│  │              │  │              │  │              │         │
│  │  Fast        │  │  Persistent  │  │  Semantic    │         │
│  │  Simple      │  │  Scalable    │  │  Search      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Creating a Contact

```
1. Application Layer
   │
   │  contact = Memory(
   │    id="contact_alice_at_techcorp_com",
   │    text="Alice Johnson, CEO at TechCorp...",
   │    metadata={'name': 'Alice', 'email': 'alice@techcorp.com'}
   │  )
   │
   ▼
2. HoloLoom Core
   │
   │  await loom.experience(contact.text)
   │  ├─ Generates embedding
   │  ├─ Stores in memory backend
   │  └─ Makes searchable
   │
   ▼
3. Knowledge Graph
   │
   │  kg.add_edge(KGEdge("Alice", "TechCorp", "WORKS_AT"))
   │  └─ Stores relationship
   │
   ▼
4. Storage Backend
   │
   │  In-Memory: Stored in Python dict/list
   │  Neo4j: Stored in graph database
   │  Qdrant: Stored in vector database
   │
   └─ Data persisted
```

---

## Data Flow: Searching Contacts

```
1. Application Layer
   │
   │  results = await loom.recall("Show me CEOs at tech companies")
   │
   ▼
2. HoloLoom Core
   │
   │  ├─ Embeds query
   │  ├─ Semantic similarity search
   │  ├─ Ranks by relevance
   │  └─ Returns top matches
   │
   ▼
3. Knowledge Graph (Optional)
   │
   │  neighbors = kg.get_neighbors("Alice")
   │  └─ Expand context with relationships
   │
   ▼
4. Application Layer
   │
   │  for result in results:
   │    print(result.metadata['name'])
   │
   └─ Display results
```

---

## Data Flow: Lead Scoring

```
1. Get Contact Data
   │
   │  contact = await loom.recall("Alice Johnson")
   │  activities = await loom.recall("activities with Alice")
   │  deals = await loom.recall("deals with Alice")
   │
   ▼
2. Calculate Signals
   │
   │  recency_score = f(days_since_last_contact)
   │  activity_score = f(activity_count)
   │  sentiment_score = f(positive_activities)
   │  value_score = f(total_deal_value)
   │  decision_score = f(is_decision_maker)
   │
   ▼
3. Weighted Average
   │
   │  score = (
   │    recency * 0.25 +
   │    activity * 0.20 +
   │    sentiment * 0.20 +
   │    value * 0.20 +
   │    decision * 0.15
   │  )
   │
   ▼
4. Classification
   │
   │  if score >= 0.75: level = "hot"
   │  elif score >= 0.50: level = "warm"
   │  elif score >= 0.25: level = "cold"
   │  else: level = "dead"
   │
   └─ Return {'score': score, 'level': level}
```

---

## Entity Relationship Diagram (ERD)

```
┌─────────────┐
│   Contact   │
│─────────────│
│ id          │
│ name        │◄─────┐
│ email       │      │
│ company     │      │ WORKS_AT
│ title       │      │
│ tags[]      │      │
└──────┬──────┘      │
       │             │
       │ ASSOCIATED_ │  ┌─────────────┐
       │    WITH     │  │   Company   │
       │             │  │─────────────│
       ▼             │  │ name        │
┌─────────────┐      │  │ industry    │
│    Deal     │      │  │ size        │
│─────────────│      │  │ domain      │
│ id          │      │  └─────────────┘
│ title       │      │
│ value       │◄─────┘
│ stage       │
│ company     │◄─────┐
└──────┬──────┘      │
       │             │ INVOLVES
       │             │
       │ INFLUENCES  │
       │             │
       ▼             │
┌─────────────┐      │
│  Activity   │      │
│─────────────│      │
│ id          │──────┘
│ type        │
│ contact     │◄─────── RELATES_TO
│ summary     │
│ outcome     │
│ sentiment   │
└─────────────┘
```

---

## Memory Structure Example

### Contact Memory

```python
{
  "id": "contact_alice_at_techcorp_com",
  "text": "Alice Johnson, CEO at TechCorp. Email: alice@techcorp.com. Notes: Very interested.",
  "timestamp": "2025-11-04T12:00:00",
  "context": {
    "type": "contact",
    "entities": ["Alice Johnson", "TechCorp", "CEO"],
    "motifs": ["contact", "customer"]
  },
  "metadata": {
    "name": "Alice Johnson",
    "email": "alice@techcorp.com",
    "company": "TechCorp",
    "title": "CEO",
    "tags": ["decision_maker", "hot_lead"],
    "notes": "Very interested."
  },
  "embedding": [0.123, -0.456, 0.789, ...]  # 384-dim vector
}
```

### Deal Memory

```python
{
  "id": "deal_D001",
  "text": "Deal: Enterprise License with TechCorp ($50,000). Stage: proposal. Contact: Alice Johnson.",
  "timestamp": "2025-11-04T12:00:00",
  "context": {
    "type": "deal",
    "entities": ["TechCorp", "Alice Johnson"],
    "motifs": ["deal", "sales", "proposal"]
  },
  "metadata": {
    "deal_id": "D001",
    "title": "Enterprise License",
    "company": "TechCorp",
    "value": 50000.0,
    "stage": "proposal",
    "contact_name": "Alice Johnson",
    "notes": "Legal review in progress"
  },
  "embedding": [0.234, -0.567, 0.890, ...]
}
```

### Activity Memory

```python
{
  "id": "activity_1762234041",
  "text": "Call with Alice Johnson: Discussed implementation timeline. Outcome: Alice will review with team. Sentiment: positive.",
  "timestamp": "2025-11-04T12:00:00",
  "context": {
    "type": "activity",
    "entities": ["Alice Johnson"],
    "motifs": ["call", "positive"]
  },
  "metadata": {
    "activity_type": "call",
    "contact_name": "Alice Johnson",
    "summary": "Discussed implementation timeline",
    "outcome": "Alice will review with team",
    "sentiment": "positive"
  },
  "embedding": [0.345, -0.678, 0.901, ...]
}
```

---

## Knowledge Graph Structure Example

```
Nodes:
  - Alice Johnson (contact)
  - Bob Smith (contact)
  - TechCorp (company)
  - InnovateCo (company)
  - deal_D001 (deal)
  - activity_1762234041 (activity)

Edges:
  Alice Johnson ─[WORKS_AT]────────────► TechCorp
  Alice Johnson ─[ASSOCIATED_WITH]─────► deal_D001
  deal_D001 ─────[INVOLVES]────────────► TechCorp
  activity_1762234041 ─[RELATES_TO]────► Alice Johnson
  activity_1762234041 ─[INFLUENCES]────► deal_D001

  Bob Smith ─────[WORKS_AT]────────────► InnovateCo
```

---

## Component Interaction Sequence

```
┌───────────┐  1. Create    ┌──────────┐
│   User    │──────────────►│   App    │
└───────────┘               └─────┬────┘
                                  │ 2. Build Memory
                                  ▼
                            ┌──────────┐
                            │  Memory  │
                            │  Object  │
                            └─────┬────┘
                                  │ 3. Store
                                  ▼
                            ┌──────────┐
                            │ HoloLoom │◄──── Embeddings
                            │   Core   │
                            └─────┬────┘
                                  │ 4. Add edges
                                  ▼
                            ┌──────────┐
                            │    KG    │◄──── Relationships
                            └─────┬────┘
                                  │ 5. Persist
                                  ▼
                            ┌──────────┐
                            │ Backend  │
                            │ (Memory/ │
                            │  Neo4j/  │
                            │ Qdrant)  │
                            └──────────┘
```

---

## Deployment Options

### Development (In-Memory)

```
┌─────────────┐
│   Python    │
│   Script    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  HoloLoom   │
│   (Memory)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ In-Memory   │
│   Storage   │
└─────────────┘

Pros: Fast, simple, no dependencies
Cons: Data lost on restart
```

### Production (Docker)

```
┌─────────────┐
│  FastAPI    │
│   Server    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  HoloLoom   │
│   (HYBRID)  │
└──────┬──────┘
       │
       ├─────────────┐
       ▼             ▼
┌─────────────┐ ┌─────────────┐
│   Neo4j     │ │   Qdrant    │
│  (Docker)   │ │  (Docker)   │
│             │ │             │
│ - Graph DB  │ │ - Vector DB │
│ - Relations │ │ - Embeddings│
└─────────────┘ └─────────────┘

Pros: Persistent, scalable, production-ready
Cons: More complex setup
```

---

## File Organization

```
mythRL/
├── crm_demo_simple.py          # ← START HERE (working demo)
├── CRM_WITH_HOLOLOOM_GUIDE.md  # ← Complete tutorial
├── CRM_QUICK_REFERENCE.md      # ← Cheat sheet
├── CRM_IMPLEMENTATION_SUMMARY.md # ← Overview
├── CRM_ARCHITECTURE_DIAGRAM.md # ← This file
│
├── HoloLoom/
│   ├── memory/
│   │   ├── protocol.py         # Memory class
│   │   └── graph.py            # KG class
│   ├── hololoom.py             # HoloLoom class
│   ├── config.py               # Configuration
│   └── ...
│
└── archive/
    └── old_projects/
        └── crm_app/            # Old CRM (superseded)
```

---

## Summary

This diagram shows:

1. **How components connect**: App → Core → Storage
2. **Data structures**: Memory, KGEdge, relationships
3. **Data flow**: Create, search, score
4. **Entity relationships**: Contact ↔ Company ↔ Deal ↔ Activity
5. **Deployment options**: Dev (in-memory) vs Prod (Docker)

**Next Steps:**
1. Run `python crm_demo_simple.py`
2. Read [CRM_WITH_HOLOLOOM_GUIDE.md](CRM_WITH_HOLOLOOM_GUIDE.md)
3. Customize for your needs
