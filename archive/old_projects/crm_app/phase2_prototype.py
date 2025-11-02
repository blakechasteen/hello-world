"""
Phase 2 Prototype: Semantic Intelligence

Demonstrates embedding generation, similarity search, and NL queries.
This is a working prototype to validate the Phase 2 roadmap.

Run with: PYTHONPATH=.. python -m crm_app.phase2_prototype
"""

import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

from crm_app.models import Contact, Company, Deal, Activity, CompanySize, DealStage, ActivityType
from crm_app.service import CompleteCRMService


# ============================================================================
# 1. Embedding Service (Simplified Prototype)
# ============================================================================

class SimpleEmbeddingService:
    """
    Simplified embedding service for prototype

    In production, this would use HoloLoom's SpectralEmbedder with
    multi-scale Matryoshka embeddings. For the prototype, we use
    simple TF-IDF-style embeddings for demonstration.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocabulary = set()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate simple embedding from text"""
        # Tokenize
        tokens = text.lower().split()

        # Update vocabulary
        self.vocabulary.update(tokens)

        # Create simple bag-of-words embedding
        embedding = np.zeros(self.dimension)

        for i, token in enumerate(tokens[:self.dimension]):
            # Simple hash-based embedding
            idx = hash(token) % self.dimension
            embedding[idx] += 1.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_contact(self, contact: Contact, company: Optional[Company] = None) -> np.ndarray:
        """Generate embedding for contact"""
        parts = [
            contact.name,
            contact.title or "",
            company.name if company else "",
            company.industry if company else "",
            contact.notes,
            " ".join(contact.tags)
        ]

        text = " ".join(p for p in parts if p)
        return self.embed_text(text)

    def embed_deal(self, deal: Deal) -> np.ndarray:
        """Generate embedding for deal"""
        text = f"{deal.title} {deal.stage.value} {deal.notes}"
        return self.embed_text(text)

    def embed_activity(self, activity: Activity) -> np.ndarray:
        """Generate embedding for activity"""
        text = f"{activity.type.value} {activity.subject} {activity.content}"
        return self.embed_text(text)


# ============================================================================
# 2. Similarity Service
# ============================================================================

class SimpleSimilarityService:
    """
    Similarity search using embeddings

    Demonstrates semantic search capabilities that will be enhanced
    with HoloLoom's multi-scale retrieval in production.
    """

    def __init__(self, crm_service: CompleteCRMService, embedding_service: SimpleEmbeddingService):
        self.crm = crm_service
        self.embeddings = embedding_service

    def find_similar_contacts(
        self,
        contact_id: str,
        limit: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[Contact, float]]:
        """Find contacts similar to given contact"""

        target = self.crm.contacts.get(contact_id)
        if not target:
            return []

        # Get target embedding
        target_company = self.crm.companies.get(target.company_id) if target.company_id else None
        target_emb = self.embeddings.embed_contact(target, target_company)

        # Compare with all contacts
        results = []
        for contact in self.crm.contacts.list():
            if contact.id == contact_id:
                continue

            company = self.crm.companies.get(contact.company_id) if contact.company_id else None
            contact_emb = self.embeddings.embed_contact(contact, company)

            similarity = self._cosine_similarity(target_emb, contact_emb)

            if similarity >= min_similarity:
                results.append((contact, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search_by_text(
        self,
        query_text: str,
        entity_type: str = "contact",
        limit: int = 10
    ) -> List[Tuple[Any, float]]:
        """Semantic search by text query"""

        query_emb = self.embeddings.embed_text(query_text)

        results = []

        if entity_type == "contact":
            for contact in self.crm.contacts.list():
                company = self.crm.companies.get(contact.company_id) if contact.company_id else None
                entity_emb = self.embeddings.embed_contact(contact, company)
                similarity = self._cosine_similarity(query_emb, entity_emb)

                if similarity > 0.1:  # Threshold
                    results.append((contact, similarity))

        elif entity_type == "deal":
            for deal in self.crm.deals.list():
                entity_emb = self.embeddings.embed_deal(deal)
                similarity = self._cosine_similarity(query_emb, entity_emb)

                if similarity > 0.1:
                    results.append((deal, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# ============================================================================
# 3. Natural Language Query Service (Simplified)
# ============================================================================

class SimpleNLQueryService:
    """
    Simplified natural language query processing

    In production, this would use HoloLoom's WeavingOrchestrator with
    full pattern detection and semantic routing. This prototype uses
    simple keyword matching and semantic search.
    """

    def __init__(self, crm_service: CompleteCRMService, similarity_service: SimpleSimilarityService):
        self.crm = crm_service
        self.similarity = similarity_service

    def query(self, natural_language: str) -> Dict[str, Any]:
        """Process natural language query"""

        nl_lower = natural_language.lower()

        # Simple intent detection
        if "similar to" in nl_lower or "like" in nl_lower:
            return self._handle_similarity_query(natural_language)
        elif "hot lead" in nl_lower or "warm lead" in nl_lower:
            return self._handle_lead_query(natural_language)
        elif "enterprise" in nl_lower or "fintech" in nl_lower or "tech" in nl_lower:
            return self._handle_industry_query(natural_language)
        elif "deal" in nl_lower and "contact" in nl_lower:
            return self._handle_activity_query(natural_language)
        else:
            # Fallback to semantic search
            return self._handle_semantic_search(natural_language)

    def _handle_similarity_query(self, query: str) -> Dict[str, Any]:
        """Handle 'find contacts like X' queries"""

        # Extract name (simple pattern matching)
        words = query.split()
        name_idx = words.index("like") if "like" in words else -1

        if name_idx > 0 and name_idx < len(words) - 1:
            name = words[name_idx + 1]

            # Find contact by name
            contacts = self.crm.contacts.list()
            target = next((c for c in contacts if name.lower() in c.name.lower()), None)

            if target:
                similar = self.similarity.find_similar_contacts(target.id, limit=5)

                return {
                    "query": query,
                    "intent": "similarity",
                    "target": target.to_dict(),
                    "results": [
                        {"contact": c.to_dict(), "similarity": sim}
                        for c, sim in similar
                    ]
                }

        return {"query": query, "intent": "similarity", "results": [], "error": "Contact not found"}

    def _handle_lead_query(self, query: str) -> Dict[str, Any]:
        """Handle lead filtering queries"""

        # Filter by engagement level
        if "hot" in query.lower():
            threshold = 0.75
        elif "warm" in query.lower():
            threshold = 0.50
        else:
            threshold = 0.0

        contacts = [
            c for c in self.crm.contacts.list()
            if c.lead_score and c.lead_score >= threshold
        ]

        # Sort by score
        contacts.sort(key=lambda c: c.lead_score or 0, reverse=True)

        return {
            "query": query,
            "intent": "lead_filter",
            "results": [c.to_dict() for c in contacts[:10]]
        }

    def _handle_industry_query(self, query: str) -> Dict[str, Any]:
        """Handle industry-based queries"""

        # Simple keyword extraction
        industries = ["technology", "fintech", "finance", "saas", "enterprise"]
        query_lower = query.lower()

        matched_industry = next((ind for ind in industries if ind in query_lower), None)

        if matched_industry:
            # Find companies in that industry
            companies = [
                c for c in self.crm.companies.list()
                if matched_industry in c.industry.lower()
            ]

            # Get contacts at those companies
            contacts = []
            for company in companies:
                company_contacts = self.crm.contacts.list({"company_id": company.id})
                contacts.extend(company_contacts)

            return {
                "query": query,
                "intent": "industry_filter",
                "industry": matched_industry,
                "results": [c.to_dict() for c in contacts[:10]]
            }

        return self._handle_semantic_search(query)

    def _handle_activity_query(self, query: str) -> Dict[str, Any]:
        """Handle activity-based queries"""

        # Parse time expressions
        if "2 weeks" in query or "two weeks" in query:
            cutoff = datetime.utcnow() - timedelta(weeks=2)
        elif "month" in query:
            cutoff = datetime.utcnow() - timedelta(days=30)
        else:
            cutoff = datetime.utcnow() - timedelta(days=7)

        # Find contacts with no recent activity
        contacts = []
        for contact in self.crm.contacts.list():
            if not contact.last_contact or contact.last_contact < cutoff:
                contacts.append(contact)

        return {
            "query": query,
            "intent": "activity_filter",
            "cutoff": cutoff.isoformat(),
            "results": [c.to_dict() for c in contacts[:10]]
        }

    def _handle_semantic_search(self, query: str) -> Dict[str, Any]:
        """Fallback: semantic search"""

        results = self.similarity.search_by_text(query, entity_type="contact", limit=10)

        return {
            "query": query,
            "intent": "semantic_search",
            "results": [
                {"contact": c.to_dict(), "relevance": sim}
                for c, sim in results
            ]
        }


# ============================================================================
# Demo
# ============================================================================

def create_sample_data(service: CompleteCRMService):
    """Create sample data for demonstration"""

    # Companies
    acme = Company.create(
        name="Acme Corp",
        industry="Technology",
        size=CompanySize.ENTERPRISE
    )
    service.create_company(acme)

    techstart = Company.create(
        name="TechStart Inc",
        industry="SaaS",
        size=CompanySize.SMALL
    )
    service.create_company(techstart)

    bigbank = Company.create(
        name="BigBank Financial",
        industry="Finance",
        size=CompanySize.LARGE
    )
    service.create_company(bigbank)

    # Contacts
    alice = Contact.create(
        name="Alice Johnson",
        email="alice@acme.com",
        company_id=acme.id,
        title="VP of Engineering",
        tags=["decision_maker", "technical", "enterprise"],
        notes="Very interested in our platform. Budget owner. Loves AI/ML features.",
        lead_score=0.85
    )
    alice.engagement_level = "hot"
    service.create_contact(alice)

    bob = Contact.create(
        name="Bob Smith",
        email="bob@techstart.io",
        company_id=techstart.id,
        title="CTO",
        tags=["decision_maker", "technical", "startup"],
        notes="Looking for scalable solutions. Price sensitive. Strong technical background.",
        lead_score=0.80
    )
    bob.engagement_level = "hot"
    service.create_contact(bob)

    carol = Contact.create(
        name="Carol Williams",
        email="carol@bigbank.com",
        company_id=bigbank.id,
        title="Head of Innovation",
        tags=["decision_maker", "enterprise", "finance"],
        notes="Long sales cycle. Multiple stakeholders. Interested in compliance features.",
        lead_score=0.55
    )
    carol.engagement_level = "warm"
    service.create_contact(carol)

    # Similar contact to Alice
    dave = Contact.create(
        name="Dave Brown",
        email="dave@techcorp.com",
        title="VP of Product",
        tags=["decision_maker", "technical"],
        notes="Interested in AI/ML. Budget owner. Fast decision maker.",
        lead_score=0.75
    )
    dave.engagement_level = "hot"
    service.create_contact(dave)

    # Activities
    service.create_activity(Activity.create(
        type=ActivityType.CALL,
        contact_id=alice.id,
        subject="Discovery Call",
        content="Discussed AI capabilities and integration options"
    ))

    service.create_activity(Activity.create(
        type=ActivityType.EMAIL,
        contact_id=bob.id,
        subject="Pricing Discussion",
        content="Sent pricing for startup package"
    ))

    return {
        "companies": [acme, techstart, bigbank],
        "contacts": [alice, bob, carol, dave]
    }


def demo_phase2():
    """Demonstrate Phase 2 features"""

    print("\n" + "=" * 60)
    print("  Phase 2 Prototype: Semantic Intelligence")
    print("=" * 60 + "\n")

    # Setup
    service = CompleteCRMService()
    embedding_service = SimpleEmbeddingService()
    similarity_service = SimpleSimilarityService(service, embedding_service)
    nl_query = SimpleNLQueryService(service, similarity_service)

    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(service)
    print(f"  Created {len(data['companies'])} companies, {len(data['contacts'])} contacts\n")

    # Demo 1: Similarity Search
    print("=" * 60)
    print("Demo 1: Similarity Search")
    print("=" * 60 + "\n")

    alice = data["contacts"][0]
    print(f"Finding contacts similar to: {alice.name}")
    print(f"  Title: {alice.title}")
    print(f"  Tags: {', '.join(alice.tags)}")
    print(f"  Notes: {alice.notes[:60]}...\n")

    similar = similarity_service.find_similar_contacts(alice.id, limit=3)

    print(f"Found {len(similar)} similar contacts:\n")
    for contact, similarity in similar:
        print(f"  {contact.name} (similarity: {similarity:.3f})")
        print(f"    Title: {contact.title}")
        print(f"    Tags: {', '.join(contact.tags)}")
        print()

    # Demo 2: Semantic Search
    print("=" * 60)
    print("Demo 2: Semantic Search")
    print("=" * 60 + "\n")

    query = "technical decision makers interested in AI"
    print(f"Query: \"{query}\"\n")

    results = similarity_service.search_by_text(query, entity_type="contact", limit=3)

    print(f"Found {len(results)} relevant contacts:\n")
    for contact, relevance in results:
        print(f"  {contact.name} (relevance: {relevance:.3f})")
        print(f"    {contact.title} at {contact.company_id}")
        print()

    # Demo 3: Natural Language Queries
    print("=" * 60)
    print("Demo 3: Natural Language Queries")
    print("=" * 60 + "\n")

    queries = [
        "Find contacts like Alice",
        "Show me hot leads",
        "Which contacts are in technology or fintech?",
        "Contacts with no activity in 2 weeks"
    ]

    for query_text in queries:
        print(f"Query: \"{query_text}\"")
        result = nl_query.query(query_text)

        print(f"  Intent: {result['intent']}")
        print(f"  Results: {len(result['results'])} found")

        if result['results']:
            for item in result['results'][:2]:  # Show top 2
                if "contact" in item:
                    c = item["contact"]
                    score = item.get("similarity") or item.get("relevance") or 0
                    print(f"    - {c['name']} (score: {score:.3f})")

        print()

    print("=" * 60)
    print("  Phase 2 Prototype Complete!")
    print("=" * 60 + "\n")

    print("Key Takeaways:")
    print("  [OK] Semantic similarity finds relevant contacts")
    print("  [OK] Embeddings enable \"find contacts like X\" queries")
    print("  [OK] Natural language queries work with simple intent detection")
    print("  [OK] Production version will use HoloLoom's full capabilities")
    print()

    print("Next Steps:")
    print("  1. Replace SimpleEmbeddingService with HoloLoom's SpectralEmbedder")
    print("  2. Use WeavingOrchestrator for full NL query processing")
    print("  3. Add multi-scale retrieval for better ranking")
    print("  4. Integrate reflection buffer for learning")


if __name__ == "__main__":
    demo_phase2()
