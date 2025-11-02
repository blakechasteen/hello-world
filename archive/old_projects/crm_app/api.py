"""
CRM REST API Server

FastAPI server exposing CRM operations and intelligence features.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import uvicorn

from crm_app.models import (
    Contact, Company, Deal, Activity,
    DealStage, ActivityType, ActivityOutcome, CompanySize
)
from crm_app.storage import CRMStorage
from crm_app.intelligence import CRMIntelligence
from HoloLoom.documentation.types import Query
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config

# Phase 2: Semantic Intelligence
from crm_app.embedding_service import CRMEmbeddingService
from crm_app.similarity_service import SimilarityService
from crm_app.nl_query_service import NaturalLanguageQueryService


# ========== Request/Response Models ==========

class ContactCreate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    company_id: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = []
    notes: str = ""


class ContactUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    company_id: Optional[str] = None
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class CompanyCreate(BaseModel):
    name: str
    industry: str
    size: CompanySize
    website: Optional[str] = None
    tags: List[str] = []
    notes: str = ""


class CompanyUpdate(BaseModel):
    name: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[CompanySize] = None
    website: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class DealCreate(BaseModel):
    title: str
    contact_id: str
    company_id: Optional[str] = None
    value: float = 0.0
    currency: str = "USD"
    stage: DealStage = DealStage.LEAD
    probability: float = 0.1
    expected_close: Optional[datetime] = None
    notes: str = ""


class DealUpdate(BaseModel):
    title: Optional[str] = None
    value: Optional[float] = None
    currency: Optional[str] = None
    stage: Optional[DealStage] = None
    probability: Optional[float] = None
    expected_close: Optional[datetime] = None
    notes: Optional[str] = None


class ActivityCreate(BaseModel):
    type: ActivityType
    contact_id: str
    deal_id: Optional[str] = None
    subject: str = ""
    content: str = ""
    outcome: Optional[ActivityOutcome] = None
    metadata: Dict[str, Any] = {}


class NLQueryRequest(BaseModel):
    query: str


# ========== FastAPI App ==========

app = FastAPI(
    title="HoloLoom CRM",
    description="Intelligent CRM powered by HoloLoom neural decision-making",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
storage = CRMStorage()
intelligence = CRMIntelligence(storage)
hololoom_config = Config.fast()
hololoom_orchestrator = None

# Phase 2: Semantic Intelligence services
embedding_service = None
similarity_service = None
nl_query_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize HoloLoom orchestrator and Phase 2 services on startup"""
    global hololoom_orchestrator, embedding_service, similarity_service, nl_query_service

    # Get memory shards from CRM storage
    shards = storage.get_memory_shards()
    hololoom_orchestrator = WeavingOrchestrator(cfg=hololoom_config, shards=shards)

    # Phase 2: Initialize semantic intelligence services
    embedding_service = CRMEmbeddingService()
    similarity_service = SimilarityService(storage, embedding_service)
    nl_query_service = NaturalLanguageQueryService(
        storage,
        embedding_service,
        similarity_service,
        config=hololoom_config
    )

    print("HoloLoom CRM API started with Phase 2 Semantic Intelligence")


# ========== Contact Endpoints ==========

@app.post("/api/contacts", response_model=Dict[str, Any])
def create_contact(contact_data: ContactCreate):
    """Create a new contact"""
    contact = Contact.create(
        name=contact_data.name,
        email=contact_data.email,
        phone=contact_data.phone,
        company_id=contact_data.company_id,
        title=contact_data.title,
        tags=contact_data.tags,
        notes=contact_data.notes
    )
    storage.create_contact(contact)
    return contact.to_dict()


@app.get("/api/contacts/{contact_id}", response_model=Dict[str, Any])
def get_contact(contact_id: str):
    """Get contact by ID"""
    contact = storage.get_contact(contact_id)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact.to_dict()


@app.put("/api/contacts/{contact_id}", response_model=Dict[str, Any])
def update_contact(contact_id: str, updates: ContactUpdate):
    """Update contact"""
    update_dict = updates.dict(exclude_unset=True)
    contact = storage.update_contact(contact_id, update_dict)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact.to_dict()


@app.delete("/api/contacts/{contact_id}")
def delete_contact(contact_id: str):
    """Archive contact"""
    success = storage.delete_contact(contact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"message": "Contact archived"}


@app.get("/api/contacts", response_model=List[Dict[str, Any]])
def list_contacts(
    company_id: Optional[str] = None,
    tag: Optional[str] = None,
    min_score: Optional[float] = None,
    not_contacted_days: Optional[int] = None
):
    """List contacts with filters"""
    filters = {}
    if company_id:
        filters["company_id"] = company_id
    if tag:
        filters["tag"] = tag
    if min_score is not None:
        filters["min_score"] = min_score
    if not_contacted_days is not None:
        filters["not_contacted_days"] = not_contacted_days

    contacts = storage.list_contacts(filters if filters else None)
    return [c.to_dict() for c in contacts]


@app.get("/api/contacts/{contact_id}/score", response_model=Dict[str, Any])
def score_contact(contact_id: str):
    """Get lead score for contact"""
    contact = storage.get_contact(contact_id)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    lead_score = intelligence.lead_scorer.score_lead(contact)
    return lead_score.to_dict()


@app.get("/api/contacts/{contact_id}/next-action", response_model=Dict[str, Any])
def get_next_action(contact_id: str):
    """Get recommended next action for contact"""
    contact = storage.get_contact(contact_id)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    recommendation = intelligence.action_recommender.recommend_action(contact)
    return recommendation.to_dict()


@app.get("/api/contacts/{contact_id}/activities", response_model=List[Dict[str, Any]])
def get_contact_activities(contact_id: str, limit: int = 50):
    """Get activity history for contact"""
    activities = storage.get_contact_activities(contact_id, limit)
    return [a.to_dict() for a in activities]


@app.get("/api/contacts/{contact_id}/similar", response_model=List[Dict[str, Any]])
def find_similar_contacts(
    contact_id: str,
    limit: int = 10,
    min_similarity: float = 0.3
):
    """
    Find contacts similar to the specified contact (Phase 2: Semantic Intelligence)

    Uses multi-scale Matryoshka embeddings to find semantically similar contacts
    based on name, title, company, and engagement patterns.

    Args:
        contact_id: ID of the target contact
        limit: Maximum number of similar contacts to return (default: 10)
        min_similarity: Minimum similarity threshold 0-1 (default: 0.3)

    Returns:
        List of similar contacts with similarity scores
    """
    contact = storage.get_contact(contact_id)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    # Find similar contacts using similarity service
    results = similarity_service.find_similar_contacts(
        contact_id,
        limit=limit,
        min_similarity=min_similarity
    )

    return [
        {
            "contact": r.entity.to_dict(),
            "similarity": r.similarity,
            "reasoning": f"Semantic similarity: {r.similarity:.2%}"
        }
        for r in results
    ]


# ========== Company Endpoints ==========

@app.post("/api/companies", response_model=Dict[str, Any])
def create_company(company_data: CompanyCreate):
    """Create a new company"""
    company = Company.create(
        name=company_data.name,
        industry=company_data.industry,
        size=company_data.size,
        website=company_data.website,
        tags=company_data.tags,
        notes=company_data.notes
    )
    storage.create_company(company)
    return company.to_dict()


@app.get("/api/companies/{company_id}", response_model=Dict[str, Any])
def get_company(company_id: str):
    """Get company by ID"""
    company = storage.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company.to_dict()


@app.put("/api/companies/{company_id}", response_model=Dict[str, Any])
def update_company(company_id: str, updates: CompanyUpdate):
    """Update company"""
    update_dict = updates.dict(exclude_unset=True)
    company = storage.update_company(company_id, update_dict)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company.to_dict()


@app.get("/api/companies", response_model=List[Dict[str, Any]])
def list_companies(industry: Optional[str] = None, size: Optional[str] = None):
    """List companies with filters"""
    filters = {}
    if industry:
        filters["industry"] = industry
    if size:
        filters["size"] = size

    companies = storage.list_companies(filters if filters else None)
    return [c.to_dict() for c in companies]


@app.get("/api/companies/{company_id}/contacts", response_model=List[Dict[str, Any]])
def get_company_contacts(company_id: str):
    """Get all contacts for a company"""
    contacts = storage.get_company_contacts(company_id)
    return [c.to_dict() for c in contacts]


# ========== Deal Endpoints ==========

@app.post("/api/deals", response_model=Dict[str, Any])
def create_deal(deal_data: DealCreate):
    """Create a new deal"""
    deal = Deal.create(
        title=deal_data.title,
        contact_id=deal_data.contact_id,
        company_id=deal_data.company_id,
        value=deal_data.value,
        currency=deal_data.currency,
        stage=deal_data.stage,
        probability=deal_data.probability,
        expected_close=deal_data.expected_close,
        notes=deal_data.notes
    )
    storage.create_deal(deal)
    return deal.to_dict()


@app.get("/api/deals/{deal_id}", response_model=Dict[str, Any])
def get_deal(deal_id: str):
    """Get deal by ID"""
    deal = storage.get_deal(deal_id)
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    return deal.to_dict()


@app.put("/api/deals/{deal_id}", response_model=Dict[str, Any])
def update_deal(deal_id: str, updates: DealUpdate):
    """Update deal"""
    update_dict = updates.dict(exclude_unset=True)
    deal = storage.update_deal(deal_id, update_dict)
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    return deal.to_dict()


@app.get("/api/deals", response_model=List[Dict[str, Any]])
def list_deals(
    stage: Optional[DealStage] = None,
    contact_id: Optional[str] = None,
    company_id: Optional[str] = None,
    min_value: Optional[float] = None,
    open_only: bool = False
):
    """List deals with filters"""
    filters = {}
    if stage:
        filters["stage"] = stage
    if contact_id:
        filters["contact_id"] = contact_id
    if company_id:
        filters["company_id"] = company_id
    if min_value is not None:
        filters["min_value"] = min_value
    if open_only:
        filters["open_only"] = True

    deals = storage.list_deals(filters if filters else None)
    return [d.to_dict() for d in deals]


@app.get("/api/deals/{deal_id}/probability", response_model=Dict[str, Any])
def predict_deal_success(deal_id: str):
    """Predict deal success probability"""
    deal = storage.get_deal(deal_id)
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")

    prediction = intelligence.deal_predictor.predict_success(deal)
    return prediction


@app.get("/api/pipeline", response_model=Dict[str, Any])
def get_pipeline():
    """Get deal pipeline summary"""
    return storage.get_pipeline_summary()


# ========== Activity Endpoints ==========

@app.post("/api/activities", response_model=Dict[str, Any])
def create_activity(activity_data: ActivityCreate):
    """Log a new activity"""
    activity = Activity.create(
        type=activity_data.type,
        contact_id=activity_data.contact_id,
        deal_id=activity_data.deal_id,
        subject=activity_data.subject,
        content=activity_data.content,
        outcome=activity_data.outcome,
        metadata=activity_data.metadata
    )
    storage.create_activity(activity)
    return activity.to_dict()


@app.get("/api/activities/{activity_id}", response_model=Dict[str, Any])
def get_activity(activity_id: str):
    """Get activity by ID"""
    activity = storage.get_activity(activity_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    return activity.to_dict()


@app.get("/api/activities", response_model=List[Dict[str, Any]])
def list_activities(
    contact_id: Optional[str] = None,
    deal_id: Optional[str] = None,
    type: Optional[ActivityType] = None
):
    """List activities with filters"""
    filters = {}
    if contact_id:
        filters["contact_id"] = contact_id
    if deal_id:
        filters["deal_id"] = deal_id
    if type:
        filters["type"] = type

    activities = storage.list_activities(filters if filters else None)
    return [a.to_dict() for a in activities]


# ========== Intelligence Endpoints ==========

@app.get("/api/insights", response_model=Dict[str, Any])
def get_insights():
    """Get comprehensive CRM insights"""
    return intelligence.get_insights()


@app.get("/api/insights/top-leads", response_model=List[Dict[str, Any]])
def get_top_leads(limit: int = 20):
    """Get highest-scoring leads"""
    results = intelligence.lead_scorer.score_all_leads()[:limit]
    return [
        {"contact": c.to_dict(), "score": s.to_dict()}
        for c, s in results
    ]


@app.get("/api/insights/daily-actions", response_model=List[Dict[str, Any]])
def get_daily_actions(limit: int = 20):
    """Get daily action recommendations"""
    results = intelligence.action_recommender.get_daily_recommendations(limit)
    return [
        {"contact": c.to_dict(), "recommendation": r.to_dict()}
        for c, r in results
    ]


@app.post("/api/query", response_model=Dict[str, Any])
async def natural_language_query(
    request: NLQueryRequest,
    max_results: int = 10,
    use_orchestrator: bool = True
):
    """
    Natural language query interface using Phase 2 Semantic Intelligence

    This endpoint uses the NaturalLanguageQueryService which integrates:
    - HoloLoom WeavingOrchestrator for pattern detection
    - Multi-scale semantic embeddings for understanding
    - Intent classification and entity extraction
    - Fallback to simple processing if orchestrator unavailable

    Examples:
    - "Show me hot leads in fintech"
    - "Find contacts similar to Alice Johnson"
    - "Which enterprise deals are over $100k?"
    - "Who should I contact today?"

    Args:
        request: Query request with natural language text
        max_results: Maximum number of results to return (default: 10)
        use_orchestrator: Use full HoloLoom orchestrator vs simple fallback (default: True)

    Returns:
        Query results with entities, relevance scores, and metadata
    """
    # Process query using Phase 2 NL query service
    async with nl_query_service:
        result = await nl_query_service.query(
            request.query,
            max_results=max_results,
            use_orchestrator=use_orchestrator
        )

    # Format response
    return {
        "query": result.query,
        "intent": result.intent,
        "entities": [
            {
                "type": type(e).__name__,
                "data": e.to_dict(),
                "relevance": result.relevance_scores[i]
            }
            for i, e in enumerate(result.entities)
        ],
        "metadata": result.metadata,
        "trace": result.trace if result.trace else None
    }


# ========== Health Check ==========

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "contacts": len(storage.contacts),
        "companies": len(storage.companies),
        "deals": len(storage.deals),
        "activities": len(storage.activities)
    }


# ========== Main ==========

if __name__ == "__main__":
    uvicorn.run(
        "crm_app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )