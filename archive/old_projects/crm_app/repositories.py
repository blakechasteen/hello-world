"""
Repository Implementations

Clean data access layer implementing repository protocols.
Separates data access from business logic for better testability.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from crm_app.models import Contact, Company, Deal, Activity, DealStage, ActivityType
from crm_app.protocols import (
    ContactRepository, CompanyRepository, DealRepository, ActivityRepository
)


# ============================================================================
# In-Memory Repositories
# ============================================================================

class InMemoryContactRepository:
    """In-memory contact repository for development/testing"""

    def __init__(self):
        self._contacts: Dict[str, Contact] = {}

    def create(self, contact: Contact) -> Contact:
        """Create a new contact"""
        self._contacts[contact.id] = contact
        return contact

    def get(self, contact_id: str) -> Optional[Contact]:
        """Get contact by ID"""
        return self._contacts.get(contact_id)

    def update(self, contact_id: str, updates: Dict[str, Any]) -> Optional[Contact]:
        """Update contact fields"""
        contact = self._contacts.get(contact_id)
        if not contact:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(contact, key):
                setattr(contact, key, value)

        return contact

    def delete(self, contact_id: str) -> bool:
        """Archive contact (soft delete)"""
        contact = self._contacts.get(contact_id)
        if not contact:
            return False

        # Soft delete - add archived tag
        if "archived" not in contact.tags:
            contact.tags.append("archived")
        return True

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Contact]:
        """List contacts with optional filters"""
        contacts = list(self._contacts.values())

        if not filters:
            return contacts

        # Apply filters
        if "company_id" in filters:
            contacts = [c for c in contacts if c.company_id == filters["company_id"]]

        if "tag" in filters:
            contacts = [c for c in contacts if filters["tag"] in c.tags]

        if "min_score" in filters:
            contacts = [
                c for c in contacts
                if c.lead_score and c.lead_score >= filters["min_score"]
            ]

        if "not_contacted_days" in filters:
            cutoff = datetime.utcnow() - timedelta(days=filters["not_contacted_days"])
            contacts = [
                c for c in contacts
                if not c.last_contact or c.last_contact < cutoff
            ]

        return contacts


class InMemoryCompanyRepository:
    """In-memory company repository for development/testing"""

    def __init__(self):
        self._companies: Dict[str, Company] = {}

    def create(self, company: Company) -> Company:
        """Create a new company"""
        self._companies[company.id] = company
        return company

    def get(self, company_id: str) -> Optional[Company]:
        """Get company by ID"""
        return self._companies.get(company_id)

    def update(self, company_id: str, updates: Dict[str, Any]) -> Optional[Company]:
        """Update company fields"""
        company = self._companies.get(company_id)
        if not company:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(company, key):
                setattr(company, key, value)

        return company

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Company]:
        """List companies with optional filters"""
        companies = list(self._companies.values())

        if not filters:
            return companies

        # Apply filters
        if "industry" in filters:
            companies = [c for c in companies if c.industry == filters["industry"]]

        if "size" in filters:
            companies = [c for c in companies if c.size.value == filters["size"]]

        return companies


class InMemoryDealRepository:
    """In-memory deal repository for development/testing"""

    def __init__(self):
        self._deals: Dict[str, Deal] = {}

    def create(self, deal: Deal) -> Deal:
        """Create a new deal"""
        self._deals[deal.id] = deal
        return deal

    def get(self, deal_id: str) -> Optional[Deal]:
        """Get deal by ID"""
        return self._deals.get(deal_id)

    def update(self, deal_id: str, updates: Dict[str, Any]) -> Optional[Deal]:
        """Update deal fields"""
        deal = self._deals.get(deal_id)
        if not deal:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(deal, key):
                setattr(deal, key, value)

        # Handle stage transitions
        if "stage" in updates:
            new_stage = updates["stage"]
            if isinstance(new_stage, str):
                new_stage = DealStage(new_stage)

            if new_stage in (DealStage.CLOSED_WON, DealStage.CLOSED_LOST):
                deal.closed_at = datetime.utcnow()

        return deal

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Deal]:
        """List deals with optional filters"""
        deals = list(self._deals.values())

        if not filters:
            return deals

        # Apply filters
        if "stage" in filters:
            stage = filters["stage"]
            if isinstance(stage, str):
                stage = DealStage(stage)
            deals = [d for d in deals if d.stage == stage]

        if "contact_id" in filters:
            deals = [d for d in deals if d.contact_id == filters["contact_id"]]

        if "company_id" in filters:
            deals = [d for d in deals if d.company_id == filters["company_id"]]

        if "min_value" in filters:
            deals = [d for d in deals if d.value >= filters["min_value"]]

        if "open_only" in filters and filters["open_only"]:
            deals = [d for d in deals if not d.is_closed]

        return deals

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        deals = list(self._deals.values())
        open_deals = [d for d in deals if not d.is_closed]

        summary = {
            "total_deals": len(deals),
            "open_deals": len(open_deals),
            "total_value": sum(d.value for d in open_deals),
            "weighted_value": sum(d.value * d.probability for d in open_deals),
            "by_stage": {}
        }

        for stage in DealStage:
            stage_deals = [d for d in open_deals if d.stage == stage]
            summary["by_stage"][stage.value] = {
                "count": len(stage_deals),
                "value": sum(d.value for d in stage_deals),
                "weighted_value": sum(d.value * d.probability for d in stage_deals)
            }

        return summary


class InMemoryActivityRepository:
    """In-memory activity repository for development/testing"""

    def __init__(self):
        self._activities: Dict[str, Activity] = {}

    def create(self, activity: Activity) -> Activity:
        """Create a new activity"""
        self._activities[activity.id] = activity
        return activity

    def get(self, activity_id: str) -> Optional[Activity]:
        """Get activity by ID"""
        return self._activities.get(activity_id)

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[Activity]:
        """List activities with optional filters"""
        activities = list(self._activities.values())

        if not filters:
            return sorted(activities, key=lambda a: a.timestamp, reverse=True)

        # Apply filters
        if "contact_id" in filters:
            activities = [a for a in activities if a.contact_id == filters["contact_id"]]

        if "deal_id" in filters:
            activities = [a for a in activities if a.deal_id == filters["deal_id"]]

        if "type" in filters:
            activity_type = filters["type"]
            if isinstance(activity_type, str):
                activity_type = ActivityType(activity_type)
            activities = [a for a in activities if a.type == activity_type]

        if "since" in filters:
            activities = [a for a in activities if a.timestamp >= filters["since"]]

        return sorted(activities, key=lambda a: a.timestamp, reverse=True)

    def get_for_contact(self, contact_id: str, limit: int = 50) -> List[Activity]:
        """Get recent activities for a contact"""
        activities = [a for a in self._activities.values() if a.contact_id == contact_id]
        activities.sort(key=lambda a: a.timestamp, reverse=True)
        return activities[:limit]


# ============================================================================
# Repository Factory
# ============================================================================

class RepositoryFactory:
    """Factory for creating repository instances"""

    @staticmethod
    def create_contact_repository(backend: str = "memory") -> ContactRepository:
        """Create a contact repository"""
        if backend == "memory":
            return InMemoryContactRepository()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def create_company_repository(backend: str = "memory") -> CompanyRepository:
        """Create a company repository"""
        if backend == "memory":
            return InMemoryCompanyRepository()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def create_deal_repository(backend: str = "memory") -> DealRepository:
        """Create a deal repository"""
        if backend == "memory":
            return InMemoryDealRepository()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def create_activity_repository(backend: str = "memory") -> ActivityRepository:
        """Create an activity repository"""
        if backend == "memory":
            return InMemoryActivityRepository()
        else:
            raise ValueError(f"Unknown backend: {backend}")
