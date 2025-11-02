"""
CRM Demo Script

Demonstrates the HoloLoom CRM system with sample data and intelligence features.
"""

from datetime import datetime, timedelta
from crm_app.models import (
    Contact, Company, Deal, Activity,
    DealStage, ActivityType, ActivityOutcome, CompanySize
)
from crm_app.storage import CRMStorage
from crm_app.intelligence import CRMIntelligence


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def create_sample_data(storage: CRMStorage):
    """Create sample CRM data"""
    print_section("Creating Sample Data")

    # Create companies
    companies = [
        Company.create(
            name="Acme Corp",
            industry="Technology",
            size=CompanySize.ENTERPRISE,
            website="https://acme.com",
            tags=["enterprise", "tech"],
            notes="Leading enterprise software company"
        ),
        Company.create(
            name="TechStart Inc",
            industry="SaaS",
            size=CompanySize.SMALL,
            website="https://techstart.io",
            tags=["startup", "saas"],
            notes="Fast-growing SaaS startup"
        ),
        Company.create(
            name="BigBank Financial",
            industry="Finance",
            size=CompanySize.LARGE,
            website="https://bigbank.com",
            tags=["enterprise", "fintech"],
            notes="Major financial institution"
        )
    ]

    for company in companies:
        storage.create_company(company)
        print(f"Created company: {company.name} ({company.industry}, {company.size.value})")

    # Create contacts
    contacts = [
        Contact.create(
            name="Alice Johnson",
            email="alice@acme.com",
            phone="+1-555-0101",
            company_id=companies[0].id,
            title="VP of Engineering",
            tags=["decision_maker", "technical"],
            notes="Very interested in our platform. Budget owner."
        ),
        Contact.create(
            name="Bob Smith",
            email="bob@techstart.io",
            phone="+1-555-0102",
            company_id=companies[1].id,
            title="CTO",
            tags=["decision_maker", "technical", "startup"],
            notes="Looking for scalable solutions. Price sensitive."
        ),
        Contact.create(
            name="Carol Williams",
            email="carol@bigbank.com",
            phone="+1-555-0103",
            company_id=companies[2].id,
            title="Head of Innovation",
            tags=["decision_maker", "enterprise"],
            notes="Long sales cycle. Multiple stakeholders."
        )
    ]

    for contact in contacts:
        storage.create_contact(contact)
        print(f"Created contact: {contact.name} ({contact.title})")

    # Create deals
    deals = [
        Deal.create(
            title="Acme Platform License",
            contact_id=contacts[0].id,
            company_id=companies[0].id,
            value=150000.0,
            stage=DealStage.PROPOSAL,
            probability=0.7,
            expected_close=datetime.utcnow() + timedelta(days=30),
            notes="Sent proposal last week. Waiting for feedback."
        ),
        Deal.create(
            title="TechStart Starter Package",
            contact_id=contacts[1].id,
            company_id=companies[1].id,
            value=25000.0,
            stage=DealStage.NEGOTIATION,
            probability=0.8,
            expected_close=datetime.utcnow() + timedelta(days=15),
            notes="Price negotiation in progress. Very close to closing."
        ),
        Deal.create(
            title="BigBank Enterprise Deal",
            contact_id=contacts[2].id,
            company_id=companies[2].id,
            value=500000.0,
            stage=DealStage.QUALIFIED,
            probability=0.3,
            expected_close=datetime.utcnow() + timedelta(days=90),
            notes="Early stage. Need to engage more stakeholders."
        )
    ]

    for deal in deals:
        storage.create_deal(deal)
        print(f"Created deal: {deal.title} (${deal.value:,.0f}, {deal.stage.value})")

    # Create activities
    activities = [
        # Alice (Acme) - Hot lead
        Activity.create(
            type=ActivityType.CALL,
            contact_id=contacts[0].id,
            deal_id=deals[0].id,
            subject="Discovery Call",
            content="Discussed requirements. Alice is very interested.",
            outcome=ActivityOutcome.POSITIVE,
            timestamp=datetime.utcnow() - timedelta(days=14)
        ),
        Activity.create(
            type=ActivityType.MEETING,
            contact_id=contacts[0].id,
            deal_id=deals[0].id,
            subject="Demo Session",
            content="Live demo. Alice loved the features.",
            outcome=ActivityOutcome.POSITIVE,
            timestamp=datetime.utcnow() - timedelta(days=7)
        ),
        # Bob (TechStart) - Very hot
        Activity.create(
            type=ActivityType.CALL,
            contact_id=contacts[1].id,
            deal_id=deals[1].id,
            subject="Initial Outreach",
            content="Bob reached out to us. Very motivated.",
            outcome=ActivityOutcome.POSITIVE,
            timestamp=datetime.utcnow() - timedelta(days=21)
        ),
        Activity.create(
            type=ActivityType.CALL,
            contact_id=contacts[1].id,
            deal_id=deals[1].id,
            subject="Price Negotiation",
            content="Agreed on discount. Bob checking with finance.",
            outcome=ActivityOutcome.POSITIVE,
            timestamp=datetime.utcnow() - timedelta(days=2)
        ),
        # Carol (BigBank) - Cold
        Activity.create(
            type=ActivityType.EMAIL,
            contact_id=contacts[2].id,
            deal_id=deals[2].id,
            subject="Introduction Email",
            content="Initial outreach. Carol responded with interest.",
            outcome=ActivityOutcome.NEUTRAL,
            timestamp=datetime.utcnow() - timedelta(days=45)
        )
    ]

    for activity in activities:
        storage.create_activity(activity)

    print(f"\nCreated {len(companies)} companies, {len(contacts)} contacts, {len(deals)} deals, {len(activities)} activities")
    return companies, contacts, deals, activities


def demo_lead_scoring(intelligence: CRMIntelligence):
    """Demonstrate lead scoring"""
    print_section("Lead Scoring")

    results = intelligence.lead_scorer.score_all_leads()

    for contact, score in results:
        print(f"{contact.name}: {score.score:.2f} ({score.engagement_level})")
        print(f"  Confidence: {score.confidence:.2f}")
        print(f"  {score.reasoning}\n")


def demo_action_recommendations(intelligence: CRMIntelligence):
    """Demonstrate action recommendations"""
    print_section("Action Recommendations")

    results = intelligence.action_recommender.get_daily_recommendations(limit=5)

    for contact, rec in results:
        print(f"{contact.name}: {rec.action} (priority: {rec.priority:.2f})")
        print(f"  {rec.reasoning}\n")


def demo_pipeline(storage: CRMStorage):
    """Demonstrate pipeline summary"""
    print_section("Pipeline Summary")

    summary = storage.get_pipeline_summary()
    print(f"Total Deals: {summary['total_deals']}")
    print(f"Open Deals: {summary['open_deals']}")
    print(f"Pipeline Value: ${summary['total_value']:,.0f}")
    print(f"Weighted Value: ${summary['weighted_value']:,.0f}")


def main():
    """Run the demo"""
    print("\n" + "=" * 60)
    print("  HoloLoom CRM Demo")
    print("=" * 60)

    storage = CRMStorage()
    intelligence = CRMIntelligence(storage)

    create_sample_data(storage)
    demo_lead_scoring(intelligence)
    demo_action_recommendations(intelligence)
    demo_pipeline(storage)

    print_section("Demo Complete!")
    print("To start API server:")
    print("  PYTHONPATH=.. python crm_app/api.py")
    print("\nVisit http://localhost:8000/docs for API documentation")


if __name__ == "__main__":
    main()