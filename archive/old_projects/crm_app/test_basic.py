"""
Basic validation tests for CRM functionality

Run with: PYTHONPATH=.. python test_basic.py
"""

from crm_app.models import Contact, Company, Deal, Activity, CompanySize, DealStage, ActivityType
from crm_app.storage import CRMStorage
from crm_app.intelligence import CRMIntelligence


def test_create_entities():
    """Test basic entity creation"""
    print("Testing entity creation...")

    storage = CRMStorage()

    # Create company
    company = Company.create(
        name="Test Corp",
        industry="Technology",
        size=CompanySize.SMALL
    )
    storage.create_company(company)
    assert len(storage.companies) == 1
    print("  [OK] Company created")

    # Create contact
    contact = Contact.create(
        name="Test User",
        email="test@example.com",
        company_id=company.id
    )
    storage.create_contact(contact)
    assert len(storage.contacts) == 1
    print("  [OK] Contact created")

    # Create deal
    deal = Deal.create(
        title="Test Deal",
        contact_id=contact.id,
        value=10000.0
    )
    storage.create_deal(deal)
    assert len(storage.deals) == 1
    print("  [OK] Deal created")

    # Create activity
    activity = Activity.create(
        type=ActivityType.CALL,
        contact_id=contact.id,
        subject="Test Call"
    )
    storage.create_activity(activity)
    assert len(storage.activities) == 1
    print("  [OK] Activity created")

    return storage


def test_knowledge_graph(storage):
    """Test knowledge graph relationships"""
    print("\nTesting knowledge graph...")

    kg = storage.get_knowledge_graph()
    stats = kg.stats()
    assert stats["num_nodes"] > 0
    print(f"  [OK] Knowledge graph has {stats['num_nodes']} nodes")
    print(f"  [OK] Knowledge graph has {stats['num_edges']} edges")


def test_memory_shards(storage):
    """Test memory shard generation"""
    print("\nTesting memory shards...")

    shards = storage.get_memory_shards()
    assert len(shards) > 0
    print(f"  [OK] Generated {len(shards)} memory shards")

    # Check shard content
    for shard in shards[:3]:
        assert shard.text
        assert shard.metadata.get("entity_type")
        print(f"    - {shard.metadata['entity_type']}: {shard.text[:50]}...")


def test_lead_scoring(storage):
    """Test lead scoring"""
    print("\nTesting lead scoring...")

    intelligence = CRMIntelligence(storage)

    # Get a contact
    contact = list(storage.contacts.values())[0]

    # Score it
    score = intelligence.lead_scorer.score_lead(contact)
    assert 0.0 <= score.score <= 1.0
    assert score.engagement_level in ["hot", "warm", "cold", "dead"]
    print(f"  [OK] Lead score: {score.score:.2f} ({score.engagement_level})")
    print(f"    Confidence: {score.confidence:.2f}")


def test_recommendations(storage):
    """Test action recommendations"""
    print("\nTesting action recommendations...")

    intelligence = CRMIntelligence(storage)
    contact = list(storage.contacts.values())[0]

    recommendation = intelligence.action_recommender.recommend_action(contact)
    assert recommendation.action in ["send_email", "schedule_call", "send_proposal", "schedule_meeting", "wait"]
    assert 0.0 <= recommendation.priority <= 1.0
    print(f"  [OK] Recommendation: {recommendation.action} (priority: {recommendation.priority:.2f})")
    print(f"    Reasoning: {recommendation.reasoning}")


def test_deal_prediction(storage):
    """Test deal success prediction"""
    print("\nTesting deal predictions...")

    intelligence = CRMIntelligence(storage)
    deal = list(storage.deals.values())[0]

    prediction = intelligence.deal_predictor.predict_success(deal)
    assert 0.0 <= prediction["probability"] <= 1.0
    print(f"  [OK] Deal success probability: {prediction['probability']:.2%}")
    print(f"    Confidence: {prediction['confidence']:.2f}")


def test_pipeline_summary(storage):
    """Test pipeline summary"""
    print("\nTesting pipeline summary...")

    summary = storage.get_pipeline_summary()
    assert summary["total_deals"] > 0
    assert "by_stage" in summary
    print(f"  [OK] Pipeline: {summary['open_deals']} open deals")
    print(f"    Total value: ${summary['total_value']:,.0f}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  CRM Basic Validation Tests")
    print("=" * 60 + "\n")

    try:
        storage = test_create_entities()
        test_knowledge_graph(storage)
        test_memory_shards(storage)
        test_lead_scoring(storage)
        test_recommendations(storage)
        test_deal_prediction(storage)
        test_pipeline_summary(storage)

        print("\n" + "=" * 60)
        print("  [SUCCESS] All tests passed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
