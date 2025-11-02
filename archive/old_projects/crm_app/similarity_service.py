"""
CRM Similarity Service

Semantic similarity search for CRM entities using embeddings.
Enables "find contacts like X" and semantic search functionality.

Features:
- Cosine similarity-based ranking
- Multi-scale search (coarse-to-fine)
- Batch processing for efficiency
- Configurable thresholds and limits
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from crm_app.models import Contact, Company, Deal, Activity
from crm_app.protocols import CRMService
from crm_app.embedding_service import CRMEmbeddingService


@dataclass
class SimilarityResult:
    """Result from similarity search"""
    entity: Any  # Contact, Company, Deal, or Activity
    similarity: float
    metadata: Dict[str, Any]


class SimilarityService:
    """
    Semantic similarity search for CRM entities

    This service provides similarity-based search and ranking using
    semantic embeddings. It enables queries like:
    - "Find contacts similar to Alice"
    - "Which deals are like this one?"
    - "Similar companies to Acme Corp"

    Architecture:
    - Uses CRMEmbeddingService for vector generation
    - Cosine similarity for ranking
    - Multi-scale search for efficiency (optional)
    - Caches embeddings for performance
    """

    def __init__(
        self,
        crm_service: CRMService,
        embedding_service: CRMEmbeddingService
    ):
        """
        Initialize similarity service

        Args:
            crm_service: CRM data access service
            embedding_service: Embedding generation service
        """
        self.crm = crm_service
        self.embeddings = embedding_service

    # ========================================================================
    # Contact Similarity
    # ========================================================================

    def find_similar_contacts(
        self,
        contact_id: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SimilarityResult]:
        """
        Find contacts similar to given contact

        Args:
            contact_id: Target contact ID
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            filters: Optional filters (tags, company_id, etc.)

        Returns:
            List of SimilarityResult sorted by similarity

        Example:
            ```python
            similar = service.find_similar_contacts(
                "alice_id",
                limit=5,
                min_similarity=0.5,
                filters={"tags": ["decision_maker"]}
            )

            for result in similar:
                print(f"{result.entity.name}: {result.similarity:.3f}")
            ```
        """
        # Get target contact
        target = self.crm.contacts.get(contact_id)
        if not target:
            return []

        # Get target embedding
        target_company = None
        if target.company_id:
            target_company = self.crm.companies.get(target.company_id)

        target_emb = self.embeddings.embed_contact(target, target_company)

        # Compare with all contacts
        results = []
        for contact in self.crm.contacts.list(filters):
            if contact.id == contact_id:
                continue  # Skip self

            # Get contact embedding
            company = None
            if contact.company_id:
                company = self.crm.companies.get(contact.company_id)

            contact_emb = self.embeddings.embed_contact(contact, company)

            # Compute similarity
            similarity = self.embeddings.cosine_similarity(target_emb, contact_emb)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    entity=contact,
                    similarity=similarity,
                    metadata={
                        "target_id": contact_id,
                        "company_id": contact.company_id,
                        "tags": contact.tags
                    }
                ))

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    # ========================================================================
    # Company Similarity
    # ========================================================================

    def find_similar_companies(
        self,
        company_id: str,
        limit: int = 10,
        min_similarity: float = 0.3
    ) -> List[SimilarityResult]:
        """Find companies similar to given company"""

        target = self.crm.companies.get(company_id)
        if not target:
            return []

        target_emb = self.embeddings.embed_company(target)

        results = []
        for company in self.crm.companies.list():
            if company.id == company_id:
                continue

            company_emb = self.embeddings.embed_company(company)
            similarity = self.embeddings.cosine_similarity(target_emb, company_emb)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    entity=company,
                    similarity=similarity,
                    metadata={"industry": company.industry, "size": company.size.value}
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    # ========================================================================
    # Deal Similarity
    # ========================================================================

    def find_similar_deals(
        self,
        deal_id: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        same_stage_only: bool = False
    ) -> List[SimilarityResult]:
        """Find deals similar to given deal"""

        target = self.crm.deals.get(deal_id)
        if not target:
            return []

        # Get context
        target_contact = self.crm.contacts.get(target.contact_id)
        target_company = None
        if target.company_id:
            target_company = self.crm.companies.get(target.company_id)

        target_emb = self.embeddings.embed_deal(target, target_contact, target_company)

        results = []
        for deal in self.crm.deals.list():
            if deal.id == deal_id:
                continue

            # Filter by stage if requested
            if same_stage_only and deal.stage != target.stage:
                continue

            # Get context
            contact = self.crm.contacts.get(deal.contact_id)
            company = None
            if deal.company_id:
                company = self.crm.companies.get(deal.company_id)

            deal_emb = self.embeddings.embed_deal(deal, contact, company)
            similarity = self.embeddings.cosine_similarity(target_emb, deal_emb)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    entity=deal,
                    similarity=similarity,
                    metadata={
                        "stage": deal.stage.value,
                        "value": deal.value,
                        "probability": deal.probability
                    }
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    # ========================================================================
    # Semantic Search (Query by Text)
    # ========================================================================

    def search_by_text(
        self,
        query_text: str,
        entity_type: str = "contact",
        limit: int = 10,
        min_similarity: float = 0.2
    ) -> List[SimilarityResult]:
        """
        Semantic search by text query

        Args:
            query_text: Natural language query
            entity_type: Type to search ('contact', 'company', 'deal', 'activity')
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            Ranked results by semantic similarity

        Example:
            ```python
            results = service.search_by_text(
                "technical decision makers interested in AI",
                entity_type="contact",
                limit=10
            )
            ```
        """
        # Generate query embedding
        query_emb = self.embeddings.embed_text(query_text)

        if entity_type == "contact":
            return self._search_contacts(query_emb, limit, min_similarity)
        elif entity_type == "company":
            return self._search_companies(query_emb, limit, min_similarity)
        elif entity_type == "deal":
            return self._search_deals(query_emb, limit, min_similarity)
        elif entity_type == "activity":
            return self._search_activities(query_emb, limit, min_similarity)
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    def _search_contacts(
        self,
        query_emb: np.ndarray,
        limit: int,
        min_similarity: float
    ) -> List[SimilarityResult]:
        """Search contacts by query embedding"""
        results = []

        for contact in self.crm.contacts.list():
            company = None
            if contact.company_id:
                company = self.crm.companies.get(contact.company_id)

            contact_emb = self.embeddings.embed_contact(contact, company)
            similarity = self.embeddings.cosine_similarity(query_emb, contact_emb)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    entity=contact,
                    similarity=similarity,
                    metadata={
                        "lead_score": contact.lead_score,
                        "engagement": contact.engagement_level
                    }
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def _search_companies(
        self,
        query_emb: np.ndarray,
        limit: int,
        min_similarity: float
    ) -> List[SimilarityResult]:
        """Search companies by query embedding"""
        results = []

        for company in self.crm.companies.list():
            company_emb = self.embeddings.embed_company(company)
            similarity = self.embeddings.cosine_similarity(query_emb, company_emb)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    entity=company,
                    similarity=similarity,
                    metadata={"industry": company.industry}
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def _search_deals(
        self,
        query_emb: np.ndarray,
        limit: int,
        min_similarity: float
    ) -> List[SimilarityResult]:
        """Search deals by query embedding"""
        results = []

        for deal in self.crm.deals.list():
            contact = self.crm.contacts.get(deal.contact_id)
            company = None
            if deal.company_id:
                company = self.crm.companies.get(deal.company_id)

            deal_emb = self.embeddings.embed_deal(deal, contact, company)
            similarity = self.embeddings.cosine_similarity(query_emb, deal_emb)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    entity=deal,
                    similarity=similarity,
                    metadata={"stage": deal.stage.value, "value": deal.value}
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def _search_activities(
        self,
        query_emb: np.ndarray,
        limit: int,
        min_similarity: float
    ) -> List[SimilarityResult]:
        """Search activities by query embedding"""
        results = []

        for activity in self.crm.activities.list():
            contact = self.crm.contacts.get(activity.contact_id)
            activity_emb = self.embeddings.embed_activity(activity, contact)
            similarity = self.embeddings.cosine_similarity(query_emb, activity_emb)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    entity=activity,
                    similarity=similarity,
                    metadata={"type": activity.type.value}
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    # ========================================================================
    # Batch Similarity (Efficient)
    # ========================================================================

    def batch_find_similar(
        self,
        contact_ids: List[str],
        limit_per_contact: int = 5
    ) -> Dict[str, List[SimilarityResult]]:
        """
        Find similar contacts for multiple targets (batch processing)

        More efficient than calling find_similar_contacts multiple times.

        Args:
            contact_ids: List of target contact IDs
            limit_per_contact: Results per contact

        Returns:
            Dict mapping contact_id -> List[SimilarityResult]
        """
        # Pre-compute all embeddings (batch)
        all_contacts = self.crm.contacts.list()
        contact_map = {c.id: c for c in all_contacts}

        # Generate all embeddings at once
        embeddings_map = {}
        for contact in all_contacts:
            company = None
            if contact.company_id:
                company = self.crm.companies.get(contact.company_id)

            embeddings_map[contact.id] = self.embeddings.embed_contact(contact, company)

        # Find similar for each target
        results = {}
        for target_id in contact_ids:
            if target_id not in embeddings_map:
                results[target_id] = []
                continue

            target_emb = embeddings_map[target_id]

            # Compare with all others
            similarities = []
            for contact_id, contact_emb in embeddings_map.items():
                if contact_id == target_id:
                    continue

                similarity = self.embeddings.cosine_similarity(target_emb, contact_emb)
                similarities.append((contact_id, similarity))

            # Sort and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = similarities[:limit_per_contact]

            # Create results
            results[target_id] = [
                SimilarityResult(
                    entity=contact_map[cid],
                    similarity=sim,
                    metadata={"target_id": target_id}
                )
                for cid, sim in top_k
            ]

        return results

    # ========================================================================
    # Clustering (Group Similar Entities)
    # ========================================================================

    def cluster_contacts(
        self,
        contact_ids: Optional[List[str]] = None,
        n_clusters: int = 5,
        min_cluster_size: int = 2
    ) -> List[List[Contact]]:
        """
        Cluster contacts by similarity

        Args:
            contact_ids: Specific contacts to cluster (default: all)
            n_clusters: Target number of clusters
            min_cluster_size: Minimum contacts per cluster

        Returns:
            List of contact clusters

        Note:
            This is a simple greedy clustering. For production,
            consider k-means or hierarchical clustering.
        """
        # Get contacts
        if contact_ids:
            contacts = [self.crm.contacts.get(cid) for cid in contact_ids]
            contacts = [c for c in contacts if c]
        else:
            contacts = self.crm.contacts.list()

        if len(contacts) < min_cluster_size:
            return [contacts] if contacts else []

        # Generate embeddings
        embeddings = []
        for contact in contacts:
            company = None
            if contact.company_id:
                company = self.crm.companies.get(contact.company_id)

            emb = self.embeddings.embed_contact(contact, company)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        # Simple greedy clustering (for demo - use sklearn in production)
        clusters = []
        used = set()

        for i, contact in enumerate(contacts):
            if i in used:
                continue

            # Start new cluster
            cluster = [contact]
            cluster_emb = embeddings[i]
            used.add(i)

            # Find similar contacts
            for j in range(i + 1, len(contacts)):
                if j in used:
                    continue

                similarity = self.embeddings.cosine_similarity(cluster_emb, embeddings[j])

                if similarity > 0.6:  # Threshold for same cluster
                    cluster.append(contacts[j])
                    used.add(j)

                    # Update cluster centroid
                    cluster_emb = np.mean([embeddings[k] for k in used if k <= j], axis=0)

                if len(cluster) >= len(contacts) // n_clusters:
                    break  # Cluster is big enough

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters


# ============================================================================
# Convenience Functions
# ============================================================================

def create_similarity_service(
    crm_service: CRMService,
    embedding_service: Optional[CRMEmbeddingService] = None
) -> SimilarityService:
    """
    Factory function to create similarity service

    Args:
        crm_service: CRM data access service
        embedding_service: Optional embedding service (created if not provided)

    Returns:
        Configured SimilarityService
    """
    if embedding_service is None:
        from crm_app.embedding_service import create_embedding_service
        embedding_service = create_embedding_service()

    return SimilarityService(crm_service, embedding_service)
