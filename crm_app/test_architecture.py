"""
Architecture Validation Tests

Tests for the refactored elegant architecture with protocols, strategies, and repositories.

Run with: PYTHONPATH=.. python -m crm_app.test_architecture
"""

from crm_app.models import Contact, Company, Deal, Activity, CompanySize, DealStage, ActivityType, ActivityOutcome
from crm_app.service import CompleteCRMService
from crm_app.intelligence_service import CRMIntelligenceService
from crm_app.strategies import (
    WeightedFeatureScoringStrategy,
    EngagementBasedRecommendationStrategy,
    ActivityBasedPredictionStrategy,
    StrategyFactory
)


def test_protocol_based_architecture():
    """Test that protocols are properly implemented"""
    print("Testing protocol-based architecture...")

    service = CompleteCRMService()

    # Test that repositories follow protocols
    assert hasattr(service.contacts, 'create')
    assert hasattr(service.contacts, 'get')
    assert hasattr(service.contacts, 'update')
    assert hasattr(service.contacts, 'delete')
    assert hasattr(service.contacts, 'list')

    assert hasattr(service.companies, 'create')
    assert hasattr(service.deals, 'create')
    assert hasattr(service.activities, 'create')

    print("  [OK] All repositories implement protocols correctly")


def test_strategy_pattern():
    """Test composable strategy pattern"""
    print("\nTesting strategy pattern...")

    # Create strategies using factory
    factory = StrategyFactory()
    scoring = factory.create_scoring_strategy("weighted")
    recommendations = factory.create_recommendation_strategy("engagement")
    predictions = factory.create_prediction_strategy("activity")

    assert isinstance(scoring, WeightedFeatureScoringStrategy)
    assert isinstance(recommendations, EngagementBasedRecommendationStrategy)
    assert isinstance(predictions, ActivityBasedPredictionStrategy)

    print("  [OK] Strategy factory creates correct strategy types")

    # Test that strategies are swappable
    service = CompleteCRMService()
    intelligence = CRMIntelligenceService(
        service,
        scoring_strategy=scoring,
        recommendation_strategy=recommendations,
        prediction_strategy=predictions
    )

    assert intelligence.scoring is scoring
    assert intelligence.recommendations is recommendations
    assert intelligence.predictions is predictions

    print("  [OK] Strategies are properly injected and swappable")


def test_repository_pattern():
    """Test repository pattern with clean data access"""
    print("\nTesting repository pattern...")

    service = CompleteCRMService()

    # Create entities through repositories
    company = Company.create(
        name="Test Corp",
        industry="Technology",
        size=CompanySize.SMALL
    )
    service.companies.create(company)

    contact = Contact.create(
        name="Test User",
        email="test@example.com",
        company_id=company.id
    )
    service.contacts.create(contact)

    # Test retrieval
    retrieved_contact = service.contacts.get(contact.id)
    assert retrieved_contact is not None
    assert retrieved_contact.name == "Test User"

    # Test filtering
    company_contacts = service.contacts.list({"company_id": company.id})
    assert len(company_contacts) == 1

    print("  [OK] Repository pattern works correctly")


def test_service_layer_integration():
    """Test that service layer properly coordinates everything"""
    print("\nTesting service layer integration...")

    service = CompleteCRMService()

    # Create complete workflow
    company = Company.create(
        name="Acme Corp",
        industry="Technology",
        size=CompanySize.ENTERPRISE
    )
    service.create_company(company)

    contact = Contact.create(
        name="Alice",
        email="alice@acme.com",
        company_id=company.id
    )
    service.create_contact(contact)

    deal = Deal.create(
        title="Big Deal",
        contact_id=contact.id,
        company_id=company.id,
        value=100000.0
    )
    service.create_deal(deal)

    activity = Activity.create(
        type=ActivityType.CALL,
        contact_id=contact.id,
        deal_id=deal.id,
        subject="Discovery call",
        outcome=ActivityOutcome.POSITIVE
    )
    service.create_activity(activity)

    # Verify knowledge graph was updated
    kg_stats = service.knowledge_graph.stats()
    assert kg_stats["num_edges"] > 0

    # Verify memory shards are generated
    shards = service.get_memory_shards()
    assert len(shards) == 4  # company, contact, deal, activity

    print("  [OK] Service layer coordinates all components")


def test_intelligence_with_strategies():
    """Test intelligence service using strategies"""
    print("\nTesting intelligence with strategies...")

    service = CompleteCRMService()
    intelligence = CRMIntelligenceService(service)

    # Create test data
    company = Company.create(
        name="Tech Inc",
        industry="SaaS",
        size=CompanySize.MEDIUM
    )
    service.create_company(company)

    contact = Contact.create(
        name="Bob",
        email="bob@tech.com",
        company_id=company.id
    )
    service.create_contact(contact)

    # Create multiple activities to establish engagement
    for i in range(5):
        activity = Activity.create(
            type=ActivityType.EMAIL,
            contact_id=contact.id,
            subject=f"Email {i}",
            outcome=ActivityOutcome.POSITIVE
        )
        service.create_activity(activity)

    # Test scoring
    lead_score = intelligence.score_lead(contact.id)
    assert 0.0 <= lead_score.score <= 1.0
    assert lead_score.engagement_level in ["hot", "warm", "cold", "dead"]
    print(f"  [OK] Lead scoring works: {lead_score.score:.2f} ({lead_score.engagement_level})")

    # Test recommendations
    recommendation = intelligence.recommend_action(contact.id)
    assert recommendation.action in ["send_email", "schedule_call", "send_proposal", "schedule_meeting", "wait"]
    print(f"  [OK] Recommendations work: {recommendation.action}")

    # Test deal prediction
    deal = Deal.create(
        title="Test Deal",
        contact_id=contact.id,
        value=50000.0,
        stage=DealStage.PROPOSAL
    )
    service.create_deal(deal)

    prediction = intelligence.predict_deal_success(deal.id)
    assert 0.0 <= prediction["probability"] <= 1.0
    print(f"  [OK] Deal prediction works: {prediction['probability']:.2%}")


def test_feature_extraction():
    """Test feature extraction utilities"""
    print("\nTesting feature extraction...")

    from crm_app.strategies import FeatureExtractor
    from datetime import datetime, timedelta

    extractor = FeatureExtractor()

    # Test engagement frequency
    activities = [
        Activity.create(type=ActivityType.CALL, contact_id="c1", timestamp=datetime.utcnow() - timedelta(days=i))
        for i in range(5)
    ]
    freq = extractor.engagement_frequency(activities)
    assert 0.0 <= freq <= 1.0
    print(f"  [OK] Engagement frequency: {freq:.2f}")

    # Test recency score
    recent = extractor.recency_score(datetime.utcnow() - timedelta(days=7))
    old = extractor.recency_score(datetime.utcnow() - timedelta(days=90))
    assert recent > old
    print(f"  [OK] Recency scoring: recent={recent:.2f}, old={old:.2f}")

    # Test sentiment score
    positive_activities = [
        Activity.create(type=ActivityType.CALL, contact_id="c1", outcome=ActivityOutcome.POSITIVE)
        for _ in range(3)
    ]
    sentiment = extractor.sentiment_score(positive_activities)
    assert sentiment > 0.8
    print(f"  [OK] Sentiment scoring: {sentiment:.2f}")


def test_statistics():
    """Test statistics gathering"""
    print("\nTesting statistics...")

    service = CompleteCRMService()

    # Create some data
    company = Company.create(name="Stats Corp", industry="Tech", size=CompanySize.SMALL)
    service.create_company(company)

    contact = Contact.create(name="Stats User", email="stats@example.com")
    service.create_contact(contact)

    stats = service.stats()
    assert stats["contacts"] == 1
    assert stats["companies"] == 1
    assert "knowledge_graph" in stats

    print(f"  [OK] Statistics: {stats['contacts']} contacts, {stats['companies']} companies")


def test_memory_shard_generation():
    """Test memory shard generation for HoloLoom"""
    print("\nTesting memory shard generation...")

    service = CompleteCRMService()

    # Create entities
    company = Company.create(name="Shard Corp", industry="Tech", size=CompanySize.SMALL)
    service.create_company(company)

    contact = Contact.create(name="Shard User", email="shard@example.com", company_id=company.id)
    service.create_contact(contact)

    # Get shards
    shards = service.get_memory_shards()
    assert len(shards) == 2

    # Validate shard structure
    for shard in shards:
        assert hasattr(shard, 'id')
        assert hasattr(shard, 'text')
        assert hasattr(shard, 'entities')
        assert hasattr(shard, 'metadata')
        assert shard.metadata.get('entity_type') in ['contact', 'company', 'deal', 'activity']

    print(f"  [OK] Generated {len(shards)} memory shards with correct structure")


def main():
    """Run all architecture tests"""
    print("\n" + "=" * 60)
    print("  CRM Architecture Validation Tests")
    print("=" * 60 + "\n")

    try:
        test_protocol_based_architecture()
        test_strategy_pattern()
        test_repository_pattern()
        test_service_layer_integration()
        test_intelligence_with_strategies()
        test_feature_extraction()
        test_statistics()
        test_memory_shard_generation()

        print("\n" + "=" * 60)
        print("  [SUCCESS] All architecture tests passed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
