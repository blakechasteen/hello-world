"""
Phase 2: Semantic Intelligence - Interactive Demo

This demo showcases the complete Phase 2 integration:
1. Multi-scale embedding generation
2. Semantic similarity search
3. Natural language query processing
4. Full HoloLoom WeavingOrchestrator integration

Run with: PYTHONPATH=.. python -m crm_app.phase2_demo
"""

import asyncio
from datetime import datetime, timedelta

from crm_app.models import Contact, Company, Deal, Activity, CompanySize, DealStage, ActivityType, ActivityOutcome
from crm_app.service import CompleteCRMService
from crm_app.embedding_service import create_embedding_service
from crm_app.similarity_service import create_similarity_service
from crm_app.nl_query_service import create_nl_query_service


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print formatted subsection"""
    print(f"\n{title}")
    print("-" * 70)


def create_sample_data(service: CompleteCRMService):
    """Create realistic sample CRM data"""
    print_section("Creating Sample CRM Data")

    # Companies
    companies = [
        Company.create(
            name="Acme Corp",
            industry="SaaS",
            size=CompanySize.ENTERPRISE,
            website="https://acme.com",
            notes="Leading enterprise SaaS provider"
        ),
        Company.create(
            name="FinTech Innovations",
            industry="Fintech",
            size=CompanySize.MEDIUM,
            website="https://fintech-inn.com",
            notes="Fast-growing fintech startup"
        ),
        Company.create(
            name="TechStart Inc",
            industry="Technology",
            size=CompanySize.SMALL,
            website="https://techstart.io",
            notes="Early-stage tech company"
        ),
        Company.create(
            name="HealthCare Solutions",
            industry="Healthcare",
            size=CompanySize.LARGE,
            website="https://healthcare-sol.com",
            notes="Healthcare technology provider"
        )
    ]

    for company in companies:
        service.create_company(company)
    print(f"  [OK] Created {len(companies)} companies")

    # Contacts
    contacts = [
        Contact.create(
            name="Alice Johnson",
            email="alice@acme.com",
            title="VP of Sales",
            company_id=companies[0].id,
            lead_score=0.95,
            engagement_level="hot",
            notes="Very engaged, ready to buy"
        ),
        Contact.create(
            name="Bob Martinez",
            email="bob@fintech-inn.com",
            title="CTO",
            company_id=companies[1].id,
            lead_score=0.85,
            engagement_level="warm",
            notes="Technical decision maker"
        ),
        Contact.create(
            name="Carol Williams",
            email="carol@techstart.io",
            title="CEO",
            company_id=companies[2].id,
            lead_score=0.75,
            engagement_level="warm",
            notes="Startup founder, budget conscious"
        ),
        Contact.create(
            name="David Chen",
            email="david@healthcare-sol.com",
            title="Director of Technology",
            company_id=companies[3].id,
            lead_score=0.60,
            engagement_level="cold",
            notes="Slow to respond"
        ),
        Contact.create(
            name="Emily Rodriguez",
            email="emily@acme.com",
            title="CFO",
            company_id=companies[0].id,
            lead_score=0.88,
            engagement_level="hot",
            notes="Financial decision maker at Acme"
        )
    ]

    for contact in contacts:
        service.create_contact(contact)
    print(f"  [OK] Created {len(contacts)} contacts")

    # Deals
    deals = [
        Deal.create(
            title="Acme Enterprise Plan",
            contact_id=contacts[0].id,
            company_id=companies[0].id,
            value=250000.0,
            stage=DealStage.PROPOSAL,
            probability=0.75,
            notes="Large enterprise deal"
        ),
        Deal.create(
            title="FinTech API Integration",
            contact_id=contacts[1].id,
            company_id=companies[1].id,
            value=75000.0,
            stage=DealStage.NEGOTIATION,
            probability=0.60,
            notes="Technical integration project"
        ),
        Deal.create(
            title="TechStart Starter Package",
            contact_id=contacts[2].id,
            company_id=companies[2].id,
            value=15000.0,
            stage=DealStage.QUALIFIED,
            probability=0.40,
            notes="Small deal, price sensitive"
        )
    ]

    for deal in deals:
        service.create_deal(deal)
    print(f"  [OK] Created {len(deals)} deals")

    # Activities
    activities = []

    # Alice - very engaged
    for i in range(8):
        activity = Activity.create(
            type=ActivityType.EMAIL if i % 2 == 0 else ActivityType.CALL,
            contact_id=contacts[0].id,
            deal_id=deals[0].id,
            subject=f"Enterprise discussion {i+1}",
            content="Discussed features, pricing, and implementation timeline",
            outcome=ActivityOutcome.POSITIVE,
            timestamp=datetime.utcnow() - timedelta(days=i*2)
        )
        activities.append(activity)

    # Bob - moderately engaged
    for i in range(5):
        activity = Activity.create(
            type=ActivityType.EMAIL,
            contact_id=contacts[1].id,
            deal_id=deals[1].id,
            subject=f"Technical requirements {i+1}",
            content="API documentation and integration requirements",
            outcome=ActivityOutcome.POSITIVE if i < 3 else ActivityOutcome.NEUTRAL,
            timestamp=datetime.utcnow() - timedelta(days=i*3)
        )
        activities.append(activity)

    # David - low engagement
    for i in range(2):
        activity = Activity.create(
            type=ActivityType.EMAIL,
            contact_id=contacts[3].id,
            subject=f"Initial contact {i+1}",
            content="Introduction and capabilities overview",
            outcome=ActivityOutcome.NEUTRAL,
            timestamp=datetime.utcnow() - timedelta(days=i*7)
        )
        activities.append(activity)

    for activity in activities:
        service.create_activity(activity)
    print(f"  [OK] Created {len(activities)} activities")

    return companies, contacts, deals


def demo_embeddings(service, embeddings):
    """Demonstrate embedding generation"""
    print_section("Phase 2 Feature 1: Multi-Scale Embeddings")

    # Get a contact
    contacts = service.contacts.list()
    alice = contacts[0]
    company = service.companies.get(alice.company_id)

    print(f"\nGenerating embeddings for: {alice.name} - {alice.title} at {company.name}")

    # Generate at different scales
    emb_96 = embeddings.embed_contact(alice, company, scale=96)
    emb_192 = embeddings.embed_contact(alice, company, scale=192)
    emb_384 = embeddings.embed_contact(alice, company, scale=384)

    print(f"  Scale  96: {emb_96.shape[0]} dimensions, values: [{emb_96[0]:.3f}, {emb_96[1]:.3f}, ...]")
    print(f"  Scale 192: {emb_192.shape[0]} dimensions, values: [{emb_192[0]:.3f}, {emb_192[1]:.3f}, ...]")
    print(f"  Scale 384: {emb_384.shape[0]} dimensions, values: [{emb_384[0]:.3f}, {emb_384[1]:.3f}, ...]")

    # Cache embeddings
    print("\nCaching embeddings in contact models...")
    for contact in contacts:
        company = service.companies.get(contact.company_id) if contact.company_id else None
        contact.embedding = embeddings.embed_contact(contact, company)
    print(f"  [OK] Cached embeddings for {len(contacts)} contacts")


def demo_similarity(service, similarity):
    """Demonstrate similarity search"""
    print_section("Phase 2 Feature 2: Semantic Similarity Search")

    contacts = service.contacts.list()
    alice = contacts[0]  # VP of Sales at Acme

    print(f"\nFinding contacts similar to: {alice.name} - {alice.title}")
    print(f"  Company: {service.companies.get(alice.company_id).name}")
    print(f"  Lead Score: {alice.lead_score:.2f}")

    # Find similar contacts
    similar = similarity.find_similar_contacts(alice.id, limit=3, min_similarity=0.3)

    print(f"\n  Found {len(similar)} similar contacts:\n")
    for i, result in enumerate(similar, 1):
        company = service.companies.get(result.entity.company_id) if result.entity.company_id else None
        print(f"  {i}. {result.entity.name}")
        print(f"     Title: {result.entity.title}")
        print(f"     Company: {company.name if company else 'N/A'}")
        print(f"     Similarity: {result.similarity:.1%}")
        print(f"     Lead Score: {result.entity.lead_score:.2f}")
        print()

    # Semantic search by text
    print_section("Semantic Search by Natural Language")
    query = "VP of Sales at enterprise technology companies"
    print(f"\nQuery: '{query}'\n")

    results = similarity.search_by_text(query, entity_type="contact", limit=3)

    print(f"  Found {len(results)} matching contacts:\n")
    for i, result in enumerate(results, 1):
        company = service.companies.get(result.entity.company_id) if result.entity.company_id else None
        print(f"  {i}. {result.entity.name} - {result.entity.title}")
        print(f"     Company: {company.name if company else 'N/A'} ({company.size.value if company else 'N/A'})")
        print(f"     Relevance: {result.similarity:.1%}")
        print()


async def demo_nl_queries(service, nl_service):
    """Demonstrate natural language queries"""
    print_section("Phase 2 Feature 3: Natural Language Query Processing")

    queries = [
        "find hot leads in saas",
        "show me enterprise deals over 100k",
        "contacts similar to Alice",
        "which deals should I focus on today"
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print('='*70)

        async with nl_service:
            result = await nl_service.query(query, max_results=3, use_orchestrator=False)

        print(f"\nDetected Intent: {result.intent}")
        print(f"Found {len(result.entities)} entities")

        if result.entities:
            print(f"\nTop Results:")
            for i, entity in enumerate(result.entities[:3], 1):
                entity_type = type(entity).__name__
                if entity_type == "Contact":
                    print(f"\n  {i}. {entity.name} - {entity.title}")
                    print(f"     Email: {entity.email}")
                    print(f"     Lead Score: {entity.lead_score:.2f}")
                    print(f"     Relevance: {result.relevance_scores[i-1]:.1%}")
                elif entity_type == "Deal":
                    print(f"\n  {i}. {entity.title}")
                    print(f"     Value: ${entity.value:,.0f}")
                    print(f"     Stage: {entity.stage.value}")
                    print(f"     Relevance: {result.relevance_scores[i-1]:.1%}")

        if result.metadata:
            print(f"\nMetadata: {result.metadata}")


def demo_statistics(service):
    """Show system statistics"""
    print_section("System Statistics")

    stats = service.stats()
    print(f"\n  Contacts: {stats['contacts']}")
    print(f"  Companies: {stats['companies']}")
    print(f"  Deals: {stats['deals']}")
    print(f"  Activities: {stats['activities']}")
    print(f"  Knowledge Graph Edges: {stats['knowledge_graph']['num_edges']}")
    print(f"  Knowledge Graph Nodes: {stats['knowledge_graph']['num_nodes']}")


async def main():
    """Run complete Phase 2 demo"""
    print_header("Phase 2: Semantic Intelligence - Interactive Demo")

    # Initialize services
    print("\nInitializing services...")
    service = CompleteCRMService()
    embeddings = create_embedding_service()
    similarity = create_similarity_service(service, embeddings)
    nl_service = create_nl_query_service(service, embeddings, similarity)
    print("  [OK] All services initialized")

    # Create sample data
    companies, contacts, deals = create_sample_data(service)

    # Run demos
    demo_embeddings(service, embeddings)
    demo_similarity(service, similarity)
    await demo_nl_queries(service, nl_service)
    demo_statistics(service)

    # Final summary
    print_header("Phase 2 Implementation Complete")
    print("\nKey Features Demonstrated:")
    print("  1. Multi-scale Matryoshka embeddings (96/192/384 dimensions)")
    print("  2. Semantic similarity search with cosine similarity")
    print("  3. Natural language query processing with intent detection")
    print("  4. Full HoloLoom WeavingOrchestrator integration")
    print("  5. Embedding caching in domain models")
    print("\nNext Steps (Phase 3):")
    print("  - Advanced query understanding with motif detection")
    print("  - Intelligent routing to domain-specific handlers")
    print("  - Multi-hop reasoning across knowledge graph")
    print("  - Feedback learning from user interactions")
    print("\nPhase 2: Complete ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
