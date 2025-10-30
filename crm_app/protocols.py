"""
CRM Protocol Definitions

Protocol-based architecture for clean separation of concerns and easy testing.
Inspired by HoloLoom's protocol-driven design.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from abc import abstractmethod

from crm_app.models import (
    Contact, Company, Deal, Activity,
    LeadScore, ActionRecommendation
)
from HoloLoom.documentation.types import MemoryShard


# ============================================================================
# Storage Protocols
# ============================================================================

@runtime_checkable
class ContactRepository(Protocol):
    """Protocol for contact storage operations"""

    def create(self, contact: Contact) -> Contact:
        """Create a new contact"""
        ...

    def get(self, contact_id: str) -> Optional[Contact]:
        """Get contact by ID"""
        ...

    def update(self, contact_id: str, updates: Dict[str, Any]) -> Optional[Contact]:
        """Update contact fields"""
        ...

    def delete(self, contact_id: str) -> bool:
        """Archive contact (soft delete)"""
        ...

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Contact]:
        """List contacts with optional filters"""
        ...


@runtime_checkable
class CompanyRepository(Protocol):
    """Protocol for company storage operations"""

    def create(self, company: Company) -> Company:
        ...

    def get(self, company_id: str) -> Optional[Company]:
        ...

    def update(self, company_id: str, updates: Dict[str, Any]) -> Optional[Company]:
        ...

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Company]:
        ...


@runtime_checkable
class DealRepository(Protocol):
    """Protocol for deal storage operations"""

    def create(self, deal: Deal) -> Deal:
        ...

    def get(self, deal_id: str) -> Optional[Deal]:
        ...

    def update(self, deal_id: str, updates: Dict[str, Any]) -> Optional[Deal]:
        ...

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Deal]:
        ...

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        ...


@runtime_checkable
class ActivityRepository(Protocol):
    """Protocol for activity storage operations"""

    def create(self, activity: Activity) -> Activity:
        ...

    def get(self, activity_id: str) -> Optional[Activity]:
        ...

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Activity]:
        ...

    def get_for_contact(self, contact_id: str, limit: int = 50) -> List[Activity]:
        """Get recent activities for a contact"""
        ...


# ============================================================================
# Intelligence Protocols
# ============================================================================

@runtime_checkable
class ScoringStrategy(Protocol):
    """Protocol for lead scoring strategies"""

    def score(self, contact: Contact, activities: List[Activity]) -> LeadScore:
        """Score a lead and return detailed result"""
        ...


@runtime_checkable
class RecommendationStrategy(Protocol):
    """Protocol for action recommendation strategies"""

    def recommend(
        self,
        contact: Contact,
        activities: List[Activity],
        lead_score: Optional[LeadScore] = None
    ) -> ActionRecommendation:
        """Recommend next action for a contact"""
        ...


@runtime_checkable
class PredictionStrategy(Protocol):
    """Protocol for deal prediction strategies"""

    def predict(self, deal: Deal, activities: List[Activity]) -> Dict[str, Any]:
        """Predict deal success probability"""
        ...


# ============================================================================
# Data Transformation Protocols
# ============================================================================

@runtime_checkable
class EntitySpinner(Protocol):
    """Protocol for converting entities to MemoryShards"""

    def spin(self, entity: Any, **context) -> MemoryShard:
        """Convert entity to MemoryShard with optional context"""
        ...


# ============================================================================
# Service Protocols
# ============================================================================

@runtime_checkable
class CRMService(Protocol):
    """High-level CRM service protocol"""

    @property
    def contacts(self) -> ContactRepository:
        """Access contact repository"""
        ...

    @property
    def companies(self) -> CompanyRepository:
        """Access company repository"""
        ...

    @property
    def deals(self) -> DealRepository:
        """Access deal repository"""
        ...

    @property
    def activities(self) -> ActivityRepository:
        """Access activity repository"""
        ...

    def get_memory_shards(self) -> List[MemoryShard]:
        """Get all CRM data as HoloLoom memory shards"""
        ...


@runtime_checkable
class IntelligenceService(Protocol):
    """High-level intelligence service protocol"""

    def score_lead(self, contact_id: str) -> LeadScore:
        """Score a lead by ID"""
        ...

    def recommend_action(self, contact_id: str) -> ActionRecommendation:
        """Recommend next action for a contact"""
        ...

    def predict_deal_success(self, deal_id: str) -> Dict[str, Any]:
        """Predict deal success probability"""
        ...

    def get_top_leads(self, limit: int = 20) -> List[tuple[Contact, LeadScore]]:
        """Get highest-scoring leads"""
        ...

    def get_daily_actions(self, limit: int = 20) -> List[tuple[Contact, ActionRecommendation]]:
        """Get daily action recommendations"""
        ...

    def get_insights(self) -> Dict[str, Any]:
        """Get comprehensive CRM insights"""
        ...
