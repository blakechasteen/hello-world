# HoloLoom CRM

An intelligent Customer Relationship Management system built on HoloLoom's neural decision-making framework.

## Features

- **Smart Lead Scoring**: AI-powered lead scoring based on engagement, recency, sentiment, and more
- **Action Recommendations**: Thompson Sampling-based recommendations for next best actions
- **Deal Predictions**: Predict deal success probability using activity patterns
- **Knowledge Graph**: Store contacts, companies, and deals in a semantic knowledge graph
- **Natural Language Queries**: Search CRM data using natural language
- **REST API**: Full-featured API with OpenAPI documentation

## Architecture

```
CRM API Layer (FastAPI)
    ↓
CRM Domain (Business Logic)
    ↓
HoloLoom Integration (Spinners, Orchestrator)
    ↓
HoloLoom Core (Memory, Policy, Graph)
```

## Quick Start

### 1. Install Dependencies

```bash
cd crm_app
pip install -r requirements.txt
```

### 2. Run Demo

```bash
PYTHONPATH=.. python demo.py
```

This creates sample data and demonstrates:
- Lead scoring (hot/warm/cold/dead classification)
- Action recommendations (send_email, schedule_call, etc.)
- Deal success predictions
- Pipeline summary

### 3. Start API Server

```bash
PYTHONPATH=.. python api.py
```

Server starts at http://localhost:8000

Visit http://localhost:8000/docs for interactive API documentation.

## API Endpoints

### Contacts
- `POST /api/contacts` - Create contact
- `GET /api/contacts/{id}` - Get contact
- `PUT /api/contacts/{id}` - Update contact
- `GET /api/contacts` - List contacts (with filters)
- `GET /api/contacts/{id}/score` - Get lead score
- `GET /api/contacts/{id}/next-action` - Get recommended action

### Companies
- `POST /api/companies` - Create company
- `GET /api/companies/{id}` - Get company
- `GET /api/companies/{id}/contacts` - List company contacts

### Deals
- `POST /api/deals` - Create deal
- `GET /api/deals/{id}` - Get deal
- `PUT /api/deals/{id}` - Update deal
- `GET /api/deals` - List deals (pipeline view)
- `GET /api/deals/{id}/probability` - Predict success
- `GET /api/pipeline` - Pipeline summary

### Activities
- `POST /api/activities` - Log activity
- `GET /api/activities` - List activities

### Intelligence
- `GET /api/insights` - Comprehensive insights
- `GET /api/insights/top-leads` - Top scoring leads
- `GET /api/insights/daily-actions` - Daily recommendations
- `POST /api/query` - Natural language query

## Example Usage

### Create a Contact

```bash
curl -X POST http://localhost:8000/api/contacts \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1-555-1234",
    "title": "CEO",
    "tags": ["decision_maker"],
    "notes": "Met at conference. Very interested."
  }'
```

### Score a Lead

```bash
curl http://localhost:8000/api/contacts/{contact_id}/score
```

Response:
```json
{
  "contact_id": "...",
  "score": 0.78,
  "confidence": 0.85,
  "engagement_level": "hot",
  "factors": {
    "engagement_frequency": 0.82,
    "recency": 0.95,
    "activity_sentiment": 0.88
  },
  "reasoning": "High engagement, recent activity, positive sentiment"
}
```

### Get Daily Actions

```bash
curl http://localhost:8000/api/insights/daily-actions
```

### Natural Language Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me enterprise leads in fintech"}'
```

## Intelligence Features

### Lead Scoring

Scores contacts 0-1 based on:
- **Engagement frequency**: How often they interact
- **Recency**: Time since last contact
- **Activity sentiment**: Positive/neutral/negative outcomes
- **Deal value**: Total pipeline value
- **Response rate**: How often they respond
- **Company fit**: Company size and industry alignment

Engagement levels:
- **Hot** (0.75+): High priority, ready to close
- **Warm** (0.50-0.75): Engaged, needs nurturing
- **Cold** (0.25-0.50): Low engagement, low-effort outreach
- **Dead** (<0.25): Consider archiving

### Action Recommendations

Recommends next best action:
- **send_email**: Reach out via email
- **schedule_call**: Set up phone call
- **send_proposal**: Send formal proposal
- **schedule_meeting**: In-person or video meeting
- **wait**: No action needed yet

Uses engagement level, recent activity, and deal stage to determine priority.

### Deal Predictions

Predicts deal success based on:
- Activity volume
- Positive sentiment ratio
- Stage progress (lead → qualified → proposal → negotiation)
- Time in pipeline
- Contact quality score

## HoloLoom Integration

### Memory Spinners

Converts CRM entities to MemoryShards:
- **ContactSpinner**: Transforms contacts with company context
- **CompanySpinner**: Transforms companies with contact counts
- **DealSpinner**: Transforms deals with contact/company names
- **ActivitySpinner**: Transforms activities with sentiment

### Knowledge Graph

Stores relationships:
- Contact -[WORKS_AT]-> Company
- Contact -[ASSOCIATED_WITH]-> Deal
- Activity -[RELATES_TO]-> Contact
- Activity -[INFLUENCES]-> Deal

### Weaving Orchestrator

Natural language queries use HoloLoom's full weaving cycle:
1. Query parsing and motif detection
2. Multi-scale semantic search
3. Knowledge graph traversal
4. Policy engine ranking
5. Result synthesis

## Development

### Project Structure

```
crm_app/
├── __init__.py          # Package init
├── models.py            # Domain models (Contact, Company, Deal, Activity)
├── spinners.py          # HoloLoom data ingestion
├── storage.py           # CRUD operations + knowledge graph
├── intelligence.py      # Lead scoring, recommendations, predictions
├── api.py               # FastAPI server
├── demo.py              # Demo script
├── requirements.txt     # Dependencies
├── README.md            # This file
└── ARCHITECTURE.md      # Detailed architecture docs
```

### Running Tests

```bash
# Run demo to verify core functionality
PYTHONPATH=.. python demo.py

# Start API and test endpoints
PYTHONPATH=.. python api.py
curl http://localhost:8000/health
```

## Production Deployment

### Database Backends

For production, use HoloLoom's HYBRID backend (Neo4j + Qdrant):

```bash
# Start services
docker-compose up -d neo4j qdrant

# Configure CRM to use HYBRID backend
# (See HoloLoom/config.py for configuration)
```

### Scaling Considerations

- Add database connection pooling
- Implement caching for lead scores
- Use background workers for batch scoring
- Add authentication/authorization
- Implement rate limiting

## Future Enhancements

- Email integration (Gmail, Outlook)
- Calendar sync
- Automated follow-up sequences
- Multi-user support with permissions
- Web frontend (React/Vue)
- Mobile app
- Reporting dashboard
- Advanced analytics

## License

Same as HoloLoom parent project.

## Support

See parent HoloLoom documentation for more details on the underlying framework.