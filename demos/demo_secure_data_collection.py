"""
Secure Data Collection Demo

Demonstrates privacy-preserving data collection integrated with HoloLoom.

Shows:
1. Collecting interactions with PII anonymization
2. Differential privacy on aggregates
3. GDPR compliance (right to be forgotten, data portability)
4. Integration with HoloLoom's memory system

Usage:
    PYTHONPATH=. python demos/demo_secure_data_collection.py
"""

import asyncio
import numpy as np
from pathlib import Path

from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query
from HoloLoom.privacy import SecureDataCollectionLoop


async def demo_basic_collection():
    """Demonstrate basic privacy-preserving collection."""
    print("=" * 70)
    print("DEMO 1: Basic Privacy-Preserving Data Collection")
    print("=" * 70)

    collector = SecureDataCollectionLoop(
        retention_days=30,
        privacy_epsilon=1.0
    )

    # Simulate user interactions
    interactions = [
        ("alice@example.com", "What is Thompson Sampling?", 0.92),
        ("bob@example.com", "How does reinforcement learning work?", 0.87),
        ("alice@example.com", "Explain Matryoshka embeddings", 0.78),
        ("charlie@example.com", "What is differential privacy?", 0.95),
        ("bob@example.com", "Show me an example of PPO", 0.88),
    ]

    print("\n📊 Collecting interactions...\n")

    for user_id, query, confidence in interactions:
        user_hash = await collector.collect_interaction(
            user_id=user_id,
            query=query,
            confidence=confidence
        )
        print(f"✅ {user_id[:15]:.<20} → {user_hash}")
        print(f"   Query: {query}")
        print(f"   Type: {collector.classify_query_type(query).value}")
        print(f"   Confidence: {confidence:.2f}\n")

    # Get privatized statistics
    print("\n" + "=" * 70)
    print("📈 Privacy-Preserving Statistics (Differential Privacy Applied)")
    print("=" * 70)

    stats = await collector.get_privacy_statistics()

    print(f"\nTotal interactions: {stats['total_interactions']:.1f} (noisy)")
    print(f"Privacy budget (ε): {stats['privacy_budget']}")
    print(f"Retention: {stats['retention_days']} days\n")

    print("Query type distribution:")
    for query_type, count in stats['query_type_distribution'].items():
        bar = "█" * int(count / 2)
        print(f"  {query_type:.<20} {count:>5.1f} {bar}")

    print(f"\nAverage confidence: {stats['avg_confidence']:.2f}")

    print("\n💡 Note: All counts are noisy (differential privacy)!")
    print("   Individual queries cannot be recovered from aggregates.")


async def demo_hololoom_integration():
    """Demonstrate integration with HoloLoom."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: HoloLoom Integration")
    print("=" * 70)

    config = Config.fast()
    collector = SecureDataCollectionLoop(retention_days=30)

    async with HoloLoom(config=config) as loom:
        # Simulate user queries
        queries = [
            "What is Thompson Sampling?",
            "Explain Matryoshka embeddings",
            "How does differential privacy work?",
        ]

        print("\n🔄 Processing queries through HoloLoom...\n")

        for i, query_text in enumerate(queries):
            user_id = f"user_{i % 2}"  # Alternate users

            # 1. Process through HoloLoom
            memories = await loom.recall(Query(text=query_text))

            # Calculate mock confidence
            confidence = 0.85 + np.random.rand() * 0.1

            # 2. Collect interaction (privacy-preserving)
            user_hash = await collector.collect_interaction(
                user_id=user_id,
                query=query_text,
                confidence=confidence,
                # Optional: Include embedding (will be encrypted)
                query_embedding=np.random.rand(384).astype(np.float32)
            )

            print(f"Query {i+1}: {query_text}")
            print(f"  User: {user_id} → {user_hash}")
            print(f"  Memories recalled: {len(memories)}")
            print(f"  Confidence: {confidence:.2f}\n")

        # Get privatized statistics
        stats = await collector.get_privacy_statistics()
        print(f"📊 Collected {stats['total_interactions']:.0f} interactions (noisy)")


async def demo_gdpr_compliance():
    """Demonstrate GDPR compliance features."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: GDPR Compliance")
    print("=" * 70)

    collector = SecureDataCollectionLoop(retention_days=30)

    # Collect some data
    print("\n📊 Collecting data for Alice and Bob...\n")

    interactions = [
        ("alice@example.com", "What is GDPR?", 0.92),
        ("alice@example.com", "How to implement privacy by design?", 0.87),
        ("bob@example.com", "What is data minimization?", 0.95),
    ]

    for user_id, query, confidence in interactions:
        await collector.collect_interaction(user_id, query, confidence)
        print(f"✅ Collected: {user_id} - {query}")

    # Right to data portability
    print("\n\n📦 GDPR Article 20: Right to Data Portability")
    print("-" * 70)

    alice_data = await collector.export_user_data("alice@example.com")
    print(f"\n✅ Exported {len(alice_data)} records for Alice:")
    for i, record in enumerate(alice_data, 1):
        print(f"\nRecord {i}:")
        print(f"  Query type: {record['query_type']}")
        print(f"  Confidence: {record['confidence']}")
        print(f"  Created: {record['created_at'][:19]}")
        print(f"  Expires: {record['expires_at'][:10]}")

    # Right to be forgotten
    print("\n\n🗑️  GDPR Article 17: Right to Be Forgotten")
    print("-" * 70)

    deleted = await collector.delete_user_data("alice@example.com")
    print(f"\n✅ Deleted {deleted} records for Alice")

    # Verify deletion
    alice_data_after = await collector.export_user_data("alice@example.com")
    print(f"✅ Verification: {len(alice_data_after)} records remain (should be 0)")

    # Bob's data should remain
    bob_data = await collector.export_user_data("bob@example.com")
    print(f"✅ Bob's data intact: {len(bob_data)} records")


async def demo_differential_privacy():
    """Demonstrate differential privacy properties."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Differential Privacy Properties")
    print("=" * 70)

    from HoloLoom.privacy import DifferentialPrivacy

    # True counts
    true_counts = {
        "factual": 150,
        "procedural": 80,
        "analytical": 45,
        "conversational": 25
    }

    print("\n📊 True counts (hidden):")
    for query_type, count in true_counts.items():
        print(f"  {query_type}: {count}")

    # Apply differential privacy with different budgets
    print("\n\n🔒 Differential Privacy with Various Budgets:")
    print("-" * 70)

    for epsilon in [0.1, 1.0, 10.0]:
        dp = DifferentialPrivacy(epsilon=epsilon)
        noisy_counts = dp.privatize_histogram(true_counts)

        print(f"\nε = {epsilon} {'(very private)' if epsilon < 1 else '(balanced)' if epsilon == 1 else '(less private)'}")
        for query_type, true_count in true_counts.items():
            noisy = noisy_counts[query_type]
            error = abs(noisy - true_count)
            print(f"  {query_type:.<20} {true_count:>3} → {noisy:>6.1f} (error: {error:.1f})")

    print("\n💡 Key insight: Lower ε = more privacy but less accuracy")
    print("   Recommended: ε=1.0 for production use")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("🔒 SECURE PRIVATE DATA COLLECTION LOOP")
    print("Privacy-Preserving Learning for HoloLoom")
    print("=" * 70)

    # Demo 1: Basic collection
    await demo_basic_collection()

    # Demo 2: HoloLoom integration
    await demo_hololoom_integration()

    # Demo 3: GDPR compliance
    await demo_gdpr_compliance()

    # Demo 4: Differential privacy
    await demo_differential_privacy()

    print("\n\n" + "=" * 70)
    print("✅ All demos complete!")
    print("=" * 70)
    print("\n📚 Learn more:")
    print("  - SECURE_PRIVATE_DATA_LOOP.md (architecture)")
    print("  - HoloLoom/privacy/secure_collection.py (implementation)")
    print("\n🔑 Security checklist:")
    print("  ✅ Set USER_HASH_SALT environment variable")
    print("     export USER_HASH_SALT=$(openssl rand -hex 32)")
    print("  ✅ Secure .keys/ directory permissions (chmod 700)")
    print("  ✅ Enable encryption (pip install cryptography)")
    print("  ✅ Configure retention policy (default: 30 days)")
    print("  ✅ Review privacy policy and terms of service")
    print("\n💡 Remember: The best way to protect data is not to collect it!")
    print("   Minimize, encrypt, delete. In that order.\n")


if __name__ == "__main__":
    asyncio.run(main())
