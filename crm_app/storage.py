"""
CRM Storage Manager

Provides CRUD operations for CRM entities with HoloLoom integration.
Stores entities in memory and creates knowledge graph relationships.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from crm_app.models import Contact, Company, Deal, Activity, DealStage, ActivityType
from crm_app.spinners import CRMSpinnerCollection
from HoloLoom.documentation.types import MemoryShard
from HoloLoom.memory.graph import KG, KGEdge


class CRMStorage:
    """Storage manager for CRM entities with HoloLoom knowledge graph"""

    def __init__(self):
        # In-memory storage (can be replaced with database)
        self.contacts: Dict[str, Contact] = {}
        self.companies: Dict[str, Company] = {}
        self.deals: Dict[str, Deal] = {}
        self.activities: Dict[str, Activity] = {}

        # HoloLoom integration
        self.spinners = CRMSpinnerCollection()
        self.knowledge_graph = KG()

    # ========== Contact Operations ==========

    def create_contact(self, contact: Contact) -> Contact:
        """Create a new contact"""
        self.contacts[contact.id] = contact

        # Create memory shard and add to knowledge graph
        company_name = None
        if contact.company_id and contact.company_id in self.companies:
            company_name = self.companies[contact.company_id].name

        shard = self.spinners.contact.spin(contact, company_name)
        self._add_to_graph(shard)

        # Create relationship to company if exists
        if contact.company_id and contact.company_id in self.companies:
            self.knowledge_graph.add_edge(KGEdge(
                src=contact.id,
                dst=contact.company_id,
                type="WORKS_AT"
            ))

        return contact

    def get_contact(self, contact_id: str) -> Optional[Contact]:
        """Get contact by ID"""
        return self.contacts.get(contact_id)

    def update_contact(self, contact_id: str, updates: Dict[str, Any]) -> Optional[Contact]:
        """Update contact fields"""
        contact = self.contacts.get(contact_id)
        if not contact:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(contact, key):
                setattr(contact, key, value)

        # Re-spin and update graph
        company_name = None
        if contact.company_id and contact.company_id in self.companies:
            company_name = self.companies[contact.company_id].name

        shard = self.spinners.contact.spin(contact, company_name)
        self._add_to_graph(shard)

        return contact

    def delete_contact(self, contact_id: str) -> bool:
        """Delete contact (soft delete - archive)"""
        if contact_id not in self.contacts:
            return False

        # Archive instead of delete
        contact = self.contacts[contact_id]
        contact.tags.append("archived")
        return True

    def list_contacts(self, filters: Optional[Dict[str, Any]] = None) -> List[Contact]:
        """List contacts with optional filters"""
        contacts = list(self.contacts.values())

        if not filters:
            return contacts

        # Apply filters
        if "company_id" in filters:
            contacts = [c for c in contacts if c.company_id == filters["company_id"]]

        if "tag" in filters:
            contacts = [c for c in contacts if filters["tag"] in c.tags]

        if "min_score" in filters:
            contacts = [c for c in contacts if c.lead_score and c.lead_score >= filters["min_score"]]

        if "not_contacted_days" in filters:
            cutoff = datetime.utcnow() - timedelta(days=filters["not_contacted_days"])
            contacts = [c for c in contacts if not c.last_contact or c.last_contact < cutoff]

        return contacts

    # ========== Company Operations ==========

    def create_company(self, company: Company) -> Company:
        """Create a new company"""
        self.companies[company.id] = company

        # Create memory shard and add to knowledge graph
        contact_count = len([c for c in self.contacts.values() if c.company_id == company.id])
        shard = self.spinners.company.spin(company, contact_count)
        self._add_to_graph(shard)

        return company

    def get_company(self, company_id: str) -> Optional[Company]:
        """Get company by ID"""
        return self.companies.get(company_id)

    def update_company(self, company_id: str, updates: Dict[str, Any]) -> Optional[Company]:
        """Update company fields"""
        company = self.companies.get(company_id)
        if not company:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(company, key):
                setattr(company, key, value)

        # Re-spin and update graph
        contact_count = len([c for c in self.contacts.values() if c.company_id == company.id])
        shard = self.spinners.company.spin(company, contact_count)
        self._add_to_graph(shard)

        return company

    def list_companies(self, filters: Optional[Dict[str, Any]] = None) -> List[Company]:
        """List companies with optional filters"""
        companies = list(self.companies.values())

        if not filters:
            return companies

        # Apply filters
        if "industry" in filters:
            companies = [c for c in companies if c.industry == filters["industry"]]

        if "size" in filters:
            companies = [c for c in companies if c.size.value == filters["size"]]

        return companies

    def get_company_contacts(self, company_id: str) -> List[Contact]:
        """Get all contacts for a company"""
        return [c for c in self.contacts.values() if c.company_id == company_id]

    # ========== Deal Operations ==========

    def create_deal(self, deal: Deal) -> Deal:
        """Create a new deal"""
        self.deals[deal.id] = deal

        # Create memory shard and add to knowledge graph
        contact = self.contacts.get(deal.contact_id)
        company = self.companies.get(deal.company_id) if deal.company_id else None

        shard = self.spinners.deal.spin(
            deal,
            contact.name if contact else None,
            company.name if company else None
        )
        self._add_to_graph(shard)

        # Create relationships
        if deal.contact_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=deal.id,
                dst=deal.contact_id,
                type="ASSOCIATED_WITH"
            ))

        if deal.company_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=deal.id,
                dst=deal.company_id,
                type="INVOLVES"
            ))

        return deal

    def get_deal(self, deal_id: str) -> Optional[Deal]:
        """Get deal by ID"""
        return self.deals.get(deal_id)

    def update_deal(self, deal_id: str, updates: Dict[str, Any]) -> Optional[Deal]:
        """Update deal fields"""
        deal = self.deals.get(deal_id)
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

        # Re-spin and update graph
        contact = self.contacts.get(deal.contact_id)
        company = self.companies.get(deal.company_id) if deal.company_id else None

        shard = self.spinners.deal.spin(
            deal,
            contact.name if contact else None,
            company.name if company else None
        )
        self._add_to_graph(shard)

        return deal

    def list_deals(self, filters: Optional[Dict[str, Any]] = None) -> List[Deal]:
        """List deals with optional filters"""
        deals = list(self.deals.values())

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
        """Get deal pipeline summary"""
        deals = list(self.deals.values())
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

    # ========== Activity Operations ==========

    def create_activity(self, activity: Activity) -> Activity:
        """Create a new activity"""
        self.activities[activity.id] = activity

        # Update contact's last_contact timestamp
        if activity.contact_id in self.contacts:
            self.contacts[activity.contact_id].last_contact = activity.timestamp

        # Create memory shard and add to knowledge graph
        contact = self.contacts.get(activity.contact_id)
        shard = self.spinners.activity.spin(
            activity,
            contact.name if contact else None
        )
        self._add_to_graph(shard)

        # Create relationships
        if activity.contact_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=activity.id,
                dst=activity.contact_id,
                type="RELATES_TO"
            ))

        if activity.deal_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=activity.id,
                dst=activity.deal_id,
                type="INFLUENCES"
            ))

        return activity

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        """Get activity by ID"""
        return self.activities.get(activity_id)

    def list_activities(self, filters: Optional[Dict[str, Any]] = None) -> List[Activity]:
        """List activities with optional filters"""
        activities = list(self.activities.values())

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

    def get_contact_activities(self, contact_id: str, limit: int = 50) -> List[Activity]:
        """Get recent activities for a contact"""
        activities = [a for a in self.activities.values() if a.contact_id == contact_id]
        activities.sort(key=lambda a: a.timestamp, reverse=True)
        return activities[:limit]

    # ========== Knowledge Graph Operations ==========

    def _add_to_graph(self, shard: MemoryShard) -> None:
        """Add memory shard to knowledge graph"""
        entity_id = shard.metadata.get("id")
        entity_type = shard.metadata.get("entity_type")

        if not entity_id:
            return

        # Add node with attributes
        self.knowledge_graph.add_node(
            entity_id,
            type=entity_type,
            text=shard.text,
            importance=shard.importance,
            **shard.metadata
        )

    def get_memory_shards(self) -> List[MemoryShard]:
        """Get all CRM data as memory shards for HoloLoom"""
        shards = []

        # Spin all entities
        for contact in self.contacts.values():
            company_name = None
            if contact.company_id and contact.company_id in self.companies:
                company_name = self.companies[contact.company_id].name
            shards.append(self.spinners.contact.spin(contact, company_name))

        for company in self.companies.values():
            contact_count = len([c for c in self.contacts.values() if c.company_id == company.id])
            shards.append(self.spinners.company.spin(company, contact_count))

        for deal in self.deals.values():
            contact = self.contacts.get(deal.contact_id)
            company = self.companies.get(deal.company_id) if deal.company_id else None
            shards.append(self.spinners.deal.spin(
                deal,
                contact.name if contact else None,
                company.name if company else None
            ))

        for activity in self.activities.values():
            contact = self.contacts.get(activity.contact_id)
            shards.append(self.spinners.activity.spin(
                activity,
                contact.name if contact else None
            ))

        return shards

    def get_knowledge_graph(self) -> KG:
        """Get the knowledge graph"""
        return self.knowledge_graph