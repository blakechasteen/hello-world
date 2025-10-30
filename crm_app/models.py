"""
CRM Domain Models

Core data structures for contacts, companies, deals, and activities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import numpy as np


class DealStage(str, Enum):
    """Deal pipeline stages"""
    LEAD = "lead"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class ActivityType(str, Enum):
    """Types of customer interactions"""
    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    NOTE = "note"
    TASK = "task"


class ActivityOutcome(str, Enum):
    """Sentiment/outcome of activity"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class CompanySize(str, Enum):
    """Company size brackets"""
    MICRO = "1-10"
    SMALL = "11-50"
    MEDIUM = "51-200"
    LARGE = "201-1000"
    ENTERPRISE = "1000+"


@dataclass
class Contact:
    """Contact/Lead entity"""
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    company_id: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_contact: Optional[datetime] = None
    notes: str = ""
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    lead_score: Optional[float] = None
    engagement_level: Optional[str] = None
    # Phase 2: Semantic Intelligence
    embedding: Optional[np.ndarray] = None  # 384-dim Matryoshka embedding

    @classmethod
    def create(cls, name: str, email: str, **kwargs) -> "Contact":
        """Factory method to create new contact with generated ID"""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            email=email,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "company_id": self.company_id,
            "title": self.title,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_contact": self.last_contact.isoformat() if self.last_contact else None,
            "notes": self.notes,
            "custom_fields": self.custom_fields,
            "lead_score": self.lead_score,
            "engagement_level": self.engagement_level
        }


@dataclass
class Company:
    """Company/Organization entity"""
    id: str
    name: str
    industry: str
    size: CompanySize
    website: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""
    # Phase 2: Semantic Intelligence
    embedding: Optional[np.ndarray] = None  # 384-dim Matryoshka embedding

    @classmethod
    def create(cls, name: str, industry: str, size: CompanySize, **kwargs) -> "Company":
        """Factory method to create new company with generated ID"""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            industry=industry,
            size=size,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "industry": self.industry,
            "size": self.size.value,
            "website": self.website,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "notes": self.notes
        }


@dataclass
class Deal:
    """Deal/Opportunity entity"""
    id: str
    title: str
    contact_id: str
    company_id: Optional[str] = None
    value: float = 0.0
    currency: str = "USD"
    stage: DealStage = DealStage.LEAD
    probability: float = 0.1
    expected_close: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    notes: str = ""
    # Phase 2: Semantic Intelligence
    embedding: Optional[np.ndarray] = None  # 384-dim Matryoshka embedding

    @classmethod
    def create(cls, title: str, contact_id: str, **kwargs) -> "Deal":
        """Factory method to create new deal with generated ID"""
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            contact_id=contact_id,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "contact_id": self.contact_id,
            "company_id": self.company_id,
            "value": self.value,
            "currency": self.currency,
            "stage": self.stage.value,
            "probability": self.probability,
            "expected_close": self.expected_close.isoformat() if self.expected_close else None,
            "created_at": self.created_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "notes": self.notes
        }

    @property
    def is_closed(self) -> bool:
        """Check if deal is closed (won or lost)"""
        return self.stage in (DealStage.CLOSED_WON, DealStage.CLOSED_LOST)

    @property
    def is_won(self) -> bool:
        """Check if deal was won"""
        return self.stage == DealStage.CLOSED_WON


@dataclass
class Activity:
    """Activity/Interaction entity"""
    id: str
    type: ActivityType
    contact_id: str
    deal_id: Optional[str] = None
    subject: str = ""
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    outcome: Optional[ActivityOutcome] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Phase 2: Semantic Intelligence
    embedding: Optional[np.ndarray] = None  # 384-dim Matryoshka embedding

    @classmethod
    def create(cls, type: ActivityType, contact_id: str, **kwargs) -> "Activity":
        """Factory method to create new activity with generated ID"""
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            contact_id=contact_id,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "contact_id": self.contact_id,
            "deal_id": self.deal_id,
            "subject": self.subject,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "outcome": self.outcome.value if self.outcome else None,
            "metadata": self.metadata
        }


@dataclass
class LeadScore:
    """Lead scoring result with explanation"""
    contact_id: str
    score: float
    confidence: float
    engagement_level: str
    factors: Dict[str, float]
    reasoning: str
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "contact_id": self.contact_id,
            "score": self.score,
            "confidence": self.confidence,
            "engagement_level": self.engagement_level,
            "factors": self.factors,
            "reasoning": self.reasoning,
            "computed_at": self.computed_at.isoformat()
        }


@dataclass
class ActionRecommendation:
    """Recommended next action for a contact/deal"""
    contact_id: str
    action: str
    priority: float
    reasoning: str
    expected_outcome: str
    suggested_timing: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "contact_id": self.contact_id,
            "action": self.action,
            "priority": self.priority,
            "reasoning": self.reasoning,
            "expected_outcome": self.expected_outcome,
            "suggested_timing": self.suggested_timing.isoformat() if self.suggested_timing else None
        }
