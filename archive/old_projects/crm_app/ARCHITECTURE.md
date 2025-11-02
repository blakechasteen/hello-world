# HoloLoom CRM Architecture

## Overview

An intelligent CRM application built on HoloLoom's neural decision-making framework, providing:
- **Smart Memory**: Contacts, companies, and interactions stored in HoloLoom's knowledge graph
- **Intelligent Routing**: Lead scoring and next-action recommendations via policy engine
- **Natural Language**: Query CRM data using semantic search
- **Learning System**: Reflection buffer learns from successful/failed deals

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│         CRM API Layer (FastAPI/Flask)           │
│  /contacts  /companies  /deals  /activities     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│        CRM Domain Layer (Business Logic)        │
│  ContactManager  DealPipeline  LeadScorer      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│      HoloLoom Integration Layer (Adapter)       │
│  CRMSpinner  CRMOrchestrator  CRMMemory        │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│        HoloLoom Core (Backend Framework)        │
│  WeavingOrchestrator  Memory  Policy  Graph    │
└─────────────────────────────────────────────────┘
```

## Core CRM Entities

### 1. Contact
```python
@dataclass
class Contact:
    id: str
    name: str
    email: str
    phone: Optional[str]
    company_id: Optional[str]
    tags: List[str]
    created_at: datetime
    last_contact: Optional[datetime]
    notes: str
    custom_fields: Dict[str, Any]
```

### 2. Company
```python
@dataclass
class Company:
    id: str
    name: str
    industry: str
    size: str  # "1-10", "11-50", "51-200", etc.
    website: Optional[str]
    tags: List[str]
    created_at: datetime
```

### 3. Deal
```python
@dataclass
class Deal:
    id: str
    title: str
    contact_id: str
    company_id: Optional[str]
    value: float
    currency: str
    stage: str  # "lead", "qualified", "proposal", "negotiation", "closed_won", "closed_lost"
    probability: float  # 0.0-1.0
    expected_close: datetime
    created_at: datetime
    closed_at: Optional[datetime]
```

### 4. Activity
```python
@dataclass
class Activity:
    id: str
    type: str  # "call", "email", "meeting", "note", "task"
    contact_id: str
    deal_id: Optional[str]
    subject: str
    content: str
    timestamp: datetime
    outcome: Optional[str]  # "positive", "neutral", "negative"
```

## HoloLoom Integration

### Memory Storage

**Yarn Graph (Knowledge Graph)**:
- Nodes: Contacts, Companies, Deals
- Edges:
  - Contact -[WORKS_AT]-> Company
  - Contact -[ASSOCIATED_WITH]-> Deal
  - Activity -[RELATES_TO]-> Contact/Deal
  - Deal -[INFLUENCED_BY]-> Activity

**Memory Shards**:
Each CRM entity becomes a MemoryShard with:
- `text`: Searchable description (name, notes, activity content)
- `embedding`: Semantic vector for similarity search
- `metadata`: Entity type, ID, timestamps
- `importance`: Priority score (frequently accessed = higher importance)

### Intelligent Features

#### 1. Lead Scoring (Policy Engine)
```python
# Uses HoloLoom's neural policy + Thompson Sampling
features = extract_contact_features(contact)
# Features: engagement frequency, deal value, time since last contact,
#           industry relevance, interaction sentiment

score = policy_engine.predict(features)
# Returns: probability distribution over outcomes (hot/warm/cold/dead)
```

#### 2. Next Action Recommendation (Decision Engine)
```python
# Thompson Sampling explores different strategies
action = convergence_engine.collapse({
    "send_email": 0.4,
    "schedule_call": 0.35,
    "send_proposal": 0.15,
    "wait": 0.1
})
# Balances exploitation (proven strategies) vs exploration (new approaches)
```

#### 3. Smart Search (Semantic Query)
```python
# Natural language queries
results = await orchestrator.weave(
    Query(text="Show me all enterprise leads in fintech who haven't been contacted in 2 weeks")
)
# Returns: Ranked results using multi-scale embeddings + graph structure
```

#### 4. Deal Success Prediction (Reflection Learning)
```python
# Learn from historical outcomes
reflection_buffer.store(
    spacetime=deal_outcome,
    feedback={"successful": deal.stage == "closed_won", "value": deal.value}
)
# System learns: which features correlate with won deals
```

### CRM Spinners

**ContactSpinner**: Converts contact data into MemoryShards
```python
class ContactSpinner(BaseSpinner):
    async def spin(self, contact: Contact) -> List[MemoryShard]:
        text = f"{contact.name} at {contact.company}. {contact.notes}"
        return [MemoryShard(
            text=text,
            source="contact",
            metadata={"id": contact.id, "type": "contact", "tags": contact.tags},
            timestamp=contact.created_at
        )]
```

**ActivitySpinner**: Processes interaction history
```python
class ActivitySpinner(BaseSpinner):
    async def spin(self, activity: Activity) -> List[MemoryShard]:
        text = f"{activity.type}: {activity.subject}. {activity.content}"
        return [MemoryShard(
            text=text,
            source="activity",
            metadata={
                "id": activity.id,
                "contact_id": activity.contact_id,
                "outcome": activity.outcome
            },
            timestamp=activity.timestamp
        )]
```

## API Endpoints

### Contacts
- `POST /api/contacts` - Create contact
- `GET /api/contacts/{id}` - Get contact details
- `PUT /api/contacts/{id}` - Update contact
- `DELETE /api/contacts/{id}` - Archive contact
- `GET /api/contacts` - List contacts (with filters)
- `GET /api/contacts/{id}/score` - Get lead score
- `GET /api/contacts/{id}/next-action` - Recommend next action

### Companies
- `POST /api/companies` - Create company
- `GET /api/companies/{id}` - Get company details
- `GET /api/companies/{id}/contacts` - List company contacts

### Deals
- `POST /api/deals` - Create deal
- `GET /api/deals/{id}` - Get deal details
- `PUT /api/deals/{id}/stage` - Update deal stage
- `GET /api/deals` - List deals (pipeline view)
- `GET /api/deals/{id}/probability` - Predict success probability

### Activities
- `POST /api/activities` - Log activity
- `GET /api/activities` - List activities (timeline view)
- `GET /api/contacts/{id}/activities` - Contact activity history

### Intelligence
- `POST /api/query` - Natural language query
- `GET /api/insights/top-leads` - Get highest-scoring leads
- `GET /api/insights/at-risk` - Get deals at risk
- `GET /api/insights/recommendations` - Get daily action recommendations

## Data Flow Examples

### Creating a Contact
```
1. API receives POST /api/contacts
2. ContactManager validates and creates Contact
3. ContactSpinner converts to MemoryShard
4. WeavingOrchestrator stores in Yarn Graph
5. Relationships created (Contact -[WORKS_AT]-> Company)
6. Return Contact ID
```

### Scoring a Lead
```
1. API receives GET /api/contacts/{id}/score
2. LeadScorer fetches contact + activity history
3. Extract features (engagement, recency, deal value)
4. Policy engine predicts score distribution
5. Return {score: 0.78, confidence: 0.85, reasoning: "High engagement, recent activity"}
```

### Natural Language Query
```
1. API receives POST /api/query {"text": "enterprise leads in fintech"}
2. CRMOrchestrator wraps in Query object
3. WeavingOrchestrator processes:
   - Motif detection: "enterprise", "fintech"
   - Semantic search: multi-scale embeddings
   - Graph traversal: find Contact -[WORKS_AT]-> Company[industry=fintech]
4. Convergence engine ranks results
5. Return ranked contacts with relevance scores
```

## Technology Stack

**Backend Framework**: HoloLoom (memory, policy, knowledge graph)
**API Layer**: FastAPI (async, OpenAPI docs, Pydantic validation)
**Storage**:
- Development: INMEMORY backend
- Production: HYBRID (Neo4j + Qdrant) with auto-fallback
**Frontend**: Optional (React/Vue, or use API directly)

## Smart Features Roadmap

### Phase 1: Core CRM (Week 1)
- [x] Basic CRUD for Contacts, Companies, Deals
- [x] Activity logging
- [x] HoloLoom memory integration

### Phase 2: Intelligence (Week 2)
- [ ] Lead scoring with policy engine
- [ ] Next action recommendations
- [ ] Natural language search

### Phase 3: Learning (Week 3)
- [ ] Deal success prediction
- [ ] Reflection learning from outcomes
- [ ] Automated insights dashboard

### Phase 4: Advanced (Week 4+)
- [ ] Email integration (Gmail/Outlook)
- [ ] Calendar sync
- [ ] Automated follow-up sequences
- [ ] Multi-user support with permissions

## Configuration

```python
# crm_app/config.py
from HoloLoom.config import Config, MemoryBackend

class CRMConfig:
    # HoloLoom backend
    hololoom_mode = "FAST"  # BARE/FAST/FUSED
    memory_backend = MemoryBackend.HYBRID

    # CRM settings
    lead_score_threshold = 0.7  # Hot leads
    activity_retention_days = 365
    auto_archive_inactive_days = 180

    # Intelligence
    enable_lead_scoring = True
    enable_recommendations = True
    enable_learning = True
```

## Development Workflow

1. **Setup environment**:
   ```bash
   cd crm_app
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Start backend services** (production mode):
   ```bash
   docker-compose up -d  # Neo4j + Qdrant
   ```

3. **Run API server**:
   ```bash
   PYTHONPATH=.. python crm_app/api/server.py
   ```

4. **Test endpoints**:
   ```bash
   curl http://localhost:8000/docs  # OpenAPI documentation
   ```

## Security Considerations

- Authentication: JWT tokens with role-based access
- Authorization: User can only access their own contacts/deals
- Data encryption: TLS for API, encryption at rest for Neo4j
- Audit logging: Track all CRM data changes
- GDPR compliance: Contact data export/deletion

## Next Steps

1. Implement core domain models
2. Create CRM spinners for data ingestion
3. Build FastAPI server with CRUD endpoints
4. Integrate HoloLoom orchestrator for intelligence
5. Add lead scoring and recommendations
6. Create demo scripts and documentation