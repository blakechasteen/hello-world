"""
CRM Service

Main service layer that coordinates repositories, knowledge graph, and HoloLoom integration.
Clean architecture with proper separation of concerns.
"""

from typing import List, Optional

from crm_app.models import Contact, Company, Deal, Activity
from crm_app.protocols import CRMService, ContactRepository, CompanyRepository, DealRepository, ActivityRepository
from crm_app.repositories import RepositoryFactory
from crm_app.spinners import CRMSpinnerCollection
from HoloLoom.documentation.types import MemoryShard
from HoloLoom.memory.graph import KG, KGEdge


class CompleteCRMService:
    """
    Complete CRM service with repositories, knowledge graph, and HoloLoom integration

    This is the main entry point for CRM operations. It coordinates:
    - Data access through repositories
    - Relationship tracking via knowledge graph
    - HoloLoom integration through spinners
    """

    def __init__(
        self,
        contact_repo: Optional[ContactRepository] = None,
        company_repo: Optional[CompanyRepository] = None,
        deal_repo: Optional[DealRepository] = None,
        activity_repo: Optional[ActivityRepository] = None,
        backend: str = "memory"
    ):
        """
        Initialize CRM service with repositories

        Args:
            contact_repo: Contact repository (created if not provided)
            company_repo: Company repository (created if not provided)
            deal_repo: Deal repository (created if not provided)
            activity_repo: Activity repository (created if not provided)
            backend: Storage backend ("memory", "database", etc.)
        """
        # Repositories
        factory = RepositoryFactory()
        self._contact_repo = contact_repo or factory.create_contact_repository(backend)
        self._company_repo = company_repo or factory.create_company_repository(backend)
        self._deal_repo = deal_repo or factory.create_deal_repository(backend)
        self._activity_repo = activity_repo or factory.create_activity_repository(backend)

        # HoloLoom integration
        self.spinners = CRMSpinnerCollection()
        self.knowledge_graph = KG()

    # ========================================================================
    # Repository Access (Protocol Implementation)
    # ========================================================================

    @property
    def contacts(self) -> ContactRepository:
        """Access contact repository"""
        return self._contact_repo

    @property
    def companies(self) -> CompanyRepository:
        """Access company repository"""
        return self._company_repo

    @property
    def deals(self) -> DealRepository:
        """Access deal repository"""
        return self._deal_repo

    @property
    def activities(self) -> ActivityRepository:
        """Access activity repository"""
        return self._activity_repo

    # ========================================================================
    # Enhanced CRUD with Knowledge Graph Integration
    # ========================================================================

    def create_contact(self, contact: Contact) -> Contact:
        """
        Create contact with knowledge graph integration

        Creates contact, generates memory shard, and establishes relationships.
        """
        # Store in repository
        created = self.contacts.create(contact)

        # Get company name for richer context
        company_name = None
        if created.company_id:
            company = self.companies.get(created.company_id)
            company_name = company.name if company else None

        # Create memory shard and add to knowledge graph
        shard = self.spinners.contact.spin(created, company_name)
        self._add_to_knowledge_graph(shard)

        # Create relationship to company
        if created.company_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=created.id,
                dst=created.company_id,
                type="WORKS_AT"
            ))

        return created

    def create_company(self, company: Company) -> Company:
        """Create company with knowledge graph integration"""
        created = self.companies.create(company)

        # Count contacts for context
        contact_count = len(self.contacts.list({"company_id": created.id}))

        # Create memory shard
        shard = self.spinners.company.spin(created, contact_count)
        self._add_to_knowledge_graph(shard)

        return created

    def create_deal(self, deal: Deal) -> Deal:
        """Create deal with knowledge graph integration"""
        created = self.deals.create(deal)

        # Get context
        contact = self.contacts.get(created.contact_id)
        company = self.companies.get(created.company_id) if created.company_id else None

        # Create memory shard
        shard = self.spinners.deal.spin(
            created,
            contact.name if contact else None,
            company.name if company else None
        )
        self._add_to_knowledge_graph(shard)

        # Create relationships
        if created.contact_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=created.id,
                dst=created.contact_id,
                type="ASSOCIATED_WITH"
            ))

        if created.company_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=created.id,
                dst=created.company_id,
                type="INVOLVES"
            ))

        return created

    def create_activity(self, activity: Activity) -> Activity:
        """Create activity with knowledge graph integration"""
        created = self.activities.create(activity)

        # Update contact's last_contact timestamp
        contact = self.contacts.get(created.contact_id)
        if contact:
            self.contacts.update(created.contact_id, {"last_contact": created.timestamp})

        # Create memory shard
        contact_name = contact.name if contact else None
        shard = self.spinners.activity.spin(created, contact_name)
        self._add_to_knowledge_graph(shard)

        # Create relationships
        if created.contact_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=created.id,
                dst=created.contact_id,
                type="RELATES_TO"
            ))

        if created.deal_id:
            self.knowledge_graph.add_edge(KGEdge(
                src=created.id,
                dst=created.deal_id,
                type="INFLUENCES"
            ))

        return created

    # ========================================================================
    # HoloLoom Integration
    # ========================================================================

    def get_memory_shards(self) -> List[MemoryShard]:
        """
        Get all CRM data as HoloLoom memory shards

        Returns:
            List of MemoryShards for use with WeavingOrchestrator
        """
        shards = []

        # Spin contacts
        for contact in self.contacts.list():
            company = None
            if contact.company_id:
                company = self.companies.get(contact.company_id)

            shard = self.spinners.contact.spin(
                contact,
                company.name if company else None
            )
            shards.append(shard)

        # Spin companies
        for company in self.companies.list():
            contact_count = len(self.contacts.list({"company_id": company.id}))
            shard = self.spinners.company.spin(company, contact_count)
            shards.append(shard)

        # Spin deals
        for deal in self.deals.list():
            contact = self.contacts.get(deal.contact_id)
            company = self.companies.get(deal.company_id) if deal.company_id else None

            shard = self.spinners.deal.spin(
                deal,
                contact.name if contact else None,
                company.name if company else None
            )
            shards.append(shard)

        # Spin activities
        for activity in self.activities.list():
            contact = self.contacts.get(activity.contact_id)
            shard = self.spinners.activity.spin(
                activity,
                contact.name if contact else None
            )
            shards.append(shard)

        return shards

    def get_knowledge_graph(self) -> KG:
        """Get the knowledge graph"""
        return self.knowledge_graph

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _add_to_knowledge_graph(self, shard: MemoryShard) -> None:
        """Add memory shard as node in knowledge graph"""
        entity_id = shard.id
        entity_type = shard.metadata.get("entity_type")

        if not entity_id:
            return

        # Add node with attributes
        # Note: KG.add_node doesn't exist in current implementation
        # Instead, nodes are created automatically when edges are added
        # We'll rely on edge creation to populate the graph
        pass

    def stats(self) -> dict:
        """Get CRM statistics"""
        return {
            "contacts": len(self.contacts.list()),
            "companies": len(self.companies.list()),
            "deals": len(self.deals.list()),
            "activities": len(self.activities.list()),
            "knowledge_graph": self.knowledge_graph.stats()
        }


# ============================================================================
# Async Context Manager Support (for future async operations)
# ============================================================================

class AsyncCRMService(CompleteCRMService):
    """
    Async-ready CRM service with lifecycle management

    For future async database operations and proper resource cleanup.
    """

    async def __aenter__(self):
        """Async context manager entry"""
        # Future: Initialize async database connections
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Future: Close async database connections
        pass
