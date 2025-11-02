"""
CRM Spinners - Convert CRM entities into HoloLoom MemoryShards

These spinners transform Contact, Company, Deal, and Activity entities
into MemoryShards that can be stored in HoloLoom's knowledge graph.
"""

from typing import List, Dict, Any
from datetime import datetime

from HoloLoom.documentation.types import MemoryShard
from crm_app.models import Contact, Company, Deal, Activity


class ContactSpinner:
    """Converts Contact entities into MemoryShards"""

    @staticmethod
    def spin(contact: Contact, company_name: str = None) -> MemoryShard:
        """
        Convert a contact to a memory shard

        Args:
            contact: Contact entity
            company_name: Optional company name for richer context

        Returns:
            MemoryShard with contact information
        """
        # Build searchable text representation
        parts = [f"Contact: {contact.name}"]

        if contact.title:
            parts.append(f"Title: {contact.title}")

        if company_name:
            parts.append(f"Company: {company_name}")
        elif contact.company_id:
            parts.append(f"Company ID: {contact.company_id}")

        parts.append(f"Email: {contact.email}")

        if contact.phone:
            parts.append(f"Phone: {contact.phone}")

        if contact.tags:
            parts.append(f"Tags: {', '.join(contact.tags)}")

        if contact.notes:
            parts.append(f"Notes: {contact.notes}")

        text = ". ".join(parts)

        # Calculate importance based on engagement
        importance = 0.5  # Default
        if contact.lead_score:
            importance = contact.lead_score
        elif contact.last_contact:
            # More recent = more important
            days_since = (datetime.utcnow() - contact.last_contact).days
            importance = max(0.1, 1.0 - (days_since / 365.0))

        return MemoryShard(
            id=contact.id,
            text=text,
            episode="crm_contact",
            entities=[contact.name, contact.email],
            motifs=contact.tags,
            metadata={
                "entity_type": "contact",
                "name": contact.name,
                "email": contact.email,
                "company_id": contact.company_id,
                "lead_score": contact.lead_score,
                "engagement_level": contact.engagement_level,
                "importance": importance,
                "created_at": contact.created_at.isoformat(),
                "last_contact": contact.last_contact.isoformat() if contact.last_contact else None
            }
        )


class CompanySpinner:
    """Converts Company entities into MemoryShards"""

    @staticmethod
    def spin(company: Company, contact_count: int = 0) -> MemoryShard:
        """
        Convert a company to a memory shard

        Args:
            company: Company entity
            contact_count: Number of contacts at this company

        Returns:
            MemoryShard with company information
        """
        # Build searchable text representation
        parts = [f"Company: {company.name}"]
        parts.append(f"Industry: {company.industry}")
        parts.append(f"Size: {company.size.value} employees")

        if company.website:
            parts.append(f"Website: {company.website}")

        if contact_count > 0:
            parts.append(f"Contacts: {contact_count}")

        if company.tags:
            parts.append(f"Tags: {', '.join(company.tags)}")

        if company.notes:
            parts.append(f"Notes: {company.notes}")

        text = ". ".join(parts)

        # Larger companies and more contacts = higher importance
        size_importance = {
            "1-10": 0.3,
            "11-50": 0.5,
            "51-200": 0.7,
            "201-1000": 0.85,
            "1000+": 0.95
        }
        importance = size_importance.get(company.size.value, 0.5)
        if contact_count > 0:
            importance = min(1.0, importance + (contact_count * 0.05))

        return MemoryShard(
            id=company.id,
            text=text,
            episode="crm_company",
            entities=[company.name, company.industry],
            motifs=company.tags,
            metadata={
                "entity_type": "company",
                "name": company.name,
                "industry": company.industry,
                "size": company.size.value,
                "website": company.website,
                "contact_count": contact_count,
                "importance": importance,
                "created_at": company.created_at.isoformat()
            }
        )


class DealSpinner:
    """Converts Deal entities into MemoryShards"""

    @staticmethod
    def spin(deal: Deal, contact_name: str = None, company_name: str = None) -> MemoryShard:
        """
        Convert a deal to a memory shard

        Args:
            deal: Deal entity
            contact_name: Optional contact name for richer context
            company_name: Optional company name for richer context

        Returns:
            MemoryShard with deal information
        """
        # Build searchable text representation
        parts = [f"Deal: {deal.title}"]
        parts.append(f"Stage: {deal.stage.value}")
        parts.append(f"Value: {deal.currency} {deal.value:,.2f}")
        parts.append(f"Probability: {deal.probability * 100:.0f}%")

        if contact_name:
            parts.append(f"Contact: {contact_name}")

        if company_name:
            parts.append(f"Company: {company_name}")

        if deal.expected_close:
            parts.append(f"Expected close: {deal.expected_close.strftime('%Y-%m-%d')}")

        if deal.notes:
            parts.append(f"Notes: {deal.notes}")

        text = ". ".join(parts)

        # Higher value and probability = higher importance
        value_importance = min(0.5, deal.value / 100000.0)  # Up to 50% from value
        prob_importance = deal.probability * 0.5  # Up to 50% from probability
        importance = value_importance + prob_importance

        return MemoryShard(
            id=deal.id,
            text=text,
            episode="crm_deal",
            entities=[deal.title, contact_name, company_name] if contact_name or company_name else [deal.title],
            motifs=[deal.stage.value],
            metadata={
                "entity_type": "deal",
                "title": deal.title,
                "contact_id": deal.contact_id,
                "company_id": deal.company_id,
                "value": deal.value,
                "currency": deal.currency,
                "stage": deal.stage.value,
                "probability": deal.probability,
                "importance": importance,
                "expected_close": deal.expected_close.isoformat() if deal.expected_close else None,
                "created_at": deal.created_at.isoformat(),
                "closed_at": deal.closed_at.isoformat() if deal.closed_at else None,
                "is_closed": deal.is_closed,
                "is_won": deal.is_won
            }
        )


class ActivitySpinner:
    """Converts Activity entities into MemoryShards"""

    @staticmethod
    def spin(activity: Activity, contact_name: str = None) -> MemoryShard:
        """
        Convert an activity to a memory shard

        Args:
            activity: Activity entity
            contact_name: Optional contact name for richer context

        Returns:
            MemoryShard with activity information
        """
        # Build searchable text representation
        parts = [f"Activity: {activity.type.value}"]

        if contact_name:
            parts.append(f"Contact: {contact_name}")

        if activity.subject:
            parts.append(f"Subject: {activity.subject}")

        if activity.content:
            parts.append(f"Content: {activity.content}")

        if activity.outcome:
            parts.append(f"Outcome: {activity.outcome.value}")

        text = ". ".join(parts)

        # Recent activities and positive outcomes = higher importance
        days_ago = (datetime.utcnow() - activity.timestamp).days
        recency_importance = max(0.1, 1.0 - (days_ago / 90.0))  # Decay over 90 days

        outcome_boost = 0.0
        if activity.outcome:
            outcome_boost = {
                "positive": 0.3,
                "neutral": 0.0,
                "negative": -0.2
            }.get(activity.outcome.value, 0.0)

        importance = max(0.1, min(1.0, recency_importance + outcome_boost))

        return MemoryShard(
            id=activity.id,
            text=text,
            episode="crm_activity",
            entities=[contact_name, activity.type.value] if contact_name else [activity.type.value],
            motifs=[activity.outcome.value] if activity.outcome else [],
            metadata={
                "entity_type": "activity",
                "type": activity.type.value,
                "contact_id": activity.contact_id,
                "deal_id": activity.deal_id,
                "subject": activity.subject,
                "outcome": activity.outcome.value if activity.outcome else None,
                "importance": importance,
                "timestamp": activity.timestamp.isoformat(),
                **activity.metadata
            }
        )


class CRMSpinnerCollection:
    """Convenience class for all CRM spinners"""

    def __init__(self):
        self.contact = ContactSpinner()
        self.company = CompanySpinner()
        self.deal = DealSpinner()
        self.activity = ActivitySpinner()

    def spin_all(self, entities: Dict[str, List[Any]]) -> List[MemoryShard]:
        """
        Spin multiple entities at once

        Args:
            entities: Dict with keys 'contacts', 'companies', 'deals', 'activities'

        Returns:
            List of all MemoryShards
        """
        shards = []

        # Spin contacts
        for contact in entities.get("contacts", []):
            shards.append(self.contact.spin(contact))

        # Spin companies
        for company in entities.get("companies", []):
            shards.append(self.company.spin(company))

        # Spin deals
        for deal in entities.get("deals", []):
            shards.append(self.deal.spin(deal))

        # Spin activities
        for activity in entities.get("activities", []):
            shards.append(self.activity.spin(activity))

        return shards