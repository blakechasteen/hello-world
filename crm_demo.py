#!/usr/bin/env python3
"""
HoloLoom CRM - Complete Working Example
========================================

Demonstrates:
1. Contact management
2. Deal tracking
3. Activity logging
4. Lead scoring
5. Action recommendations
6. Knowledge graph relationships
"""

import asyncio
from datetime import datetime, timedelta
from typing import List
from hololoom.hololoom import HoloLoom
from hololoom.memory.graph import KG, KGEdge
from hololoom.memory.protocol import Memory


# ============================================================================
# Helper Functions
# ============================================================================

def create_contact_memory(name: str, email: str, company: str, title: str,
                          notes: str, tags: list = None) -> Memory:
    """Create a contact as a Memory object."""
    contact_id = f"contact_{email.replace('@', '_at_')}"
    text = f"{name}, {title} at {company}. Email: {email}. Notes: {notes}"

    return Memory(
        id=contact_id,
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'contact',
            'entities': [name, company, title],
            'motifs': ['contact', 'customer']
        },
        metadata={
            'email': email,
            'company': company,
            'title': title,
            'name': name,
            'tags': tags or [],
            'notes': notes
        }
    )


def create_deal_memory(deal_id: str, title: str, company: str, value: float,
                       stage: str, contact_name: str, notes: str = "") -> Memory:
    """Create a deal as a Memory object."""
    text = (
        f"Deal: {title} with {company} (${value:,.0f}). "
        f"Stage: {stage}. Contact: {contact_name}. {notes}"
    )

    return Memory(
        id=f"deal_{deal_id}",
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'deal',
            'entities': [company, contact_name],
            'motifs': ['deal', 'sales', stage]
        },
        metadata={
            'deal_id': deal_id,
            'title': title,
            'company': company,
            'value': value,
            'stage': stage,
            'contact_name': contact_name,
            'notes': notes,
            'created_at': datetime.now().isoformat()
        }
    )


def create_activity_memory(activity_type: str, contact_name: str,
                           summary: str, outcome: str = "",
                           sentiment: str = "neutral") -> Memory:
    """Create an activity as a Memory object."""
    activity_id = f"activity_{int(datetime.now().timestamp())}"
    text = (
        f"{activity_type.title()} with {contact_name}: {summary}. "
        f"Outcome: {outcome}. Sentiment: {sentiment}."
    )

    return Memory(
        id=activity_id,
        text=text,
        timestamp=datetime.now(),
        context={
            'type': 'activity',
            'entities': [contact_name],
            'motifs': [activity_type, sentiment]
        },
        metadata={
            'activity_type': activity_type,
            'contact_name': contact_name,
            'summary': summary,
            'outcome': outcome,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
    )


def calculate_lead_score(contact: Memory, activities: List[Memory],
                        deals: List[Memory]) -> dict:
    """Calculate lead score based on engagement signals."""
    days_since = (datetime.now() - contact.timestamp).days
    recency_score = max(0.0, 1.0 - (days_since / 30.0))

    activity_count = len(activities)
    activity_score = min(1.0, activity_count / 10.0)

    if activities:
        positive = sum(1 for a in activities
                      if a.metadata.get('sentiment') == 'positive')
        sentiment_score = positive / len(activities)
    else:
        sentiment_score = 0.5

    total_value = sum(d.metadata.get('value', 0) for d in deals)
    value_score = min(1.0, total_value / 100000.0)

    is_decision_maker = 'decision_maker' in contact.metadata.get('tags', [])
    decision_score = 1.0 if is_decision_maker else 0.3

    score = (
        recency_score * 0.25 +
        activity_score * 0.20 +
        sentiment_score * 0.20 +
        value_score * 0.20 +
        decision_score * 0.15
    )

    if score >= 0.75:
        level = "hot"
    elif score >= 0.50:
        level = "warm"
    elif score >= 0.25:
        level = "cold"
    else:
        level = "dead"

    return {
        'score': score,
        'engagement_level': level,
        'factors': {
            'recency': recency_score,
            'activity': activity_score,
            'sentiment': sentiment_score,
            'value': value_score,
            'decision_maker': decision_score
        }
    }


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("HoloLoom CRM - Complete Demo")
    print("=" * 70)
    print()

    # Initialize
    print("1. Initializing HoloLoom and Knowledge Graph...")
    loom = HoloLoom()
    kg = KG()
    print("   ✓ Ready\n")

    # Create contacts
    print("2. Creating contacts...")
    contacts = [
        create_contact_memory(
            name="Alice Johnson",
            email="alice@techcorp.com",
            company="TechCorp",
            title="CEO",
            notes="Very interested in Q1 implementation. Decision maker.",
            tags=["decision_maker", "hot_lead"]
        ),
        create_contact_memory(
            name="Bob Smith",
            email="bob@innovate.io",
            company="InnovateCo",
            title="CTO",
            notes="Technical contact. Prefers detailed documentation.",
            tags=["technical", "warm_lead"]
        ),
        create_contact_memory(
            name="Carol Davis",
            email="carol@startup.com",
            company="StartupXYZ",
            title="Founder",
            notes="Early stage startup. Budget constrained.",
            tags=["founder"]
        )
    ]

    for contact in contacts:
        await loom.experience(contact.text)
        kg.add_edge(KGEdge(
            contact.metadata['name'],
            contact.metadata['company'],
            "WORKS_AT"
        ))
        print(f"   ✓ Added {contact.metadata['name']}")
    print()

    # Create deals
    print("3. Creating deals...")
    deals = [
        create_deal_memory(
            deal_id="D001",
            title="Enterprise License",
            company="TechCorp",
            value=50000.0,
            stage="proposal",
            contact_name="Alice Johnson",
            notes="Legal review in progress"
        ),
        create_deal_memory(
            deal_id="D002",
            title="Startup Package",
            company="StartupXYZ",
            value=5000.0,
            stage="qualified",
            contact_name="Carol Davis",
            notes="Waiting for funding round"
        )
    ]

    for deal in deals:
        await loom.experience(deal.text)
        kg.add_edge(KGEdge(
            deal.metadata['contact_name'],
            deal.id,
            "ASSOCIATED_WITH"
        ))
        kg.add_edge(KGEdge(
            deal.id,
            deal.metadata['company'],
            "INVOLVES"
        ))
        print(f"   ✓ Added {deal.metadata['title']}")
    print()

    # Log activities
    print("4. Logging activities...")
    activities = [
        create_activity_memory(
            activity_type="call",
            contact_name="Alice Johnson",
            summary="Discussed implementation timeline",
            outcome="Alice will review with team",
            sentiment="positive"
        ),
        create_activity_memory(
            activity_type="email",
            contact_name="Alice Johnson",
            summary="Sent proposal document",
            outcome="Awaiting feedback",
            sentiment="neutral"
        ),
        create_activity_memory(
            activity_type="call",
            contact_name="Bob Smith",
            summary="Technical deep dive on API",
            outcome="Bob needs more documentation",
            sentiment="positive"
        )
    ]

    for activity in activities:
        await loom.experience(activity.text)
        kg.add_edge(KGEdge(
            activity.id,
            activity.metadata['contact_name'],
            "RELATES_TO"
        ))
        print(f"   ✓ Logged {activity.metadata['activity_type']} with {activity.metadata['contact_name']}")
    print()

    # Search functionality
    print("5. Testing search...")
    print("   Query: 'Show me decision makers'")
    results = await loom.recall("decision makers")
    print(f"   Found {len(results)} results:")
    for r in results[:3]:
        print(f"   - {r.text[:80]}...")
    print()

    # Knowledge graph queries
    print("6. Knowledge graph queries...")
    alice_neighbors = kg.get_neighbors("Alice Johnson")
    print(f"   Alice Johnson is connected to: {alice_neighbors}")

    subgraph = kg.subgraph_for_entities(["Alice Johnson", "TechCorp"])
    print(f"   Subgraph contains {subgraph.number_of_nodes()} nodes, "
          f"{subgraph.number_of_edges()} edges")
    print()

    # Lead scoring
    print("7. Lead scoring...")
    for contact in contacts[:2]:  # Score first 2 contacts
        contact_name = contact.metadata['name']

        # Get related activities
        related_activities = [
            a for a in activities
            if a.metadata['contact_name'] == contact_name
        ]

        # Get related deals
        related_deals = [
            d for d in deals
            if d.metadata['contact_name'] == contact_name
        ]

        # Calculate score
        score = calculate_lead_score(contact, related_activities, related_deals)

        print(f"   {contact_name}:")
        print(f"     Score: {score['score']:.2f} ({score['engagement_level']})")
        print(f"     Factors: recency={score['factors']['recency']:.2f}, "
              f"activity={score['factors']['activity']:.2f}, "
              f"sentiment={score['factors']['sentiment']:.2f}")
    print()

    # Summary
    print("8. Summary...")
    print(f"   Contacts: {len(contacts)}")
    print(f"   Deals: {len(deals)}")
    print(f"   Activities: {len(activities)}")
    print(f"   Knowledge graph: {kg.G.number_of_nodes()} nodes, "
          f"{kg.G.number_of_edges()} edges")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
