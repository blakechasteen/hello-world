"""
Demonstration of Sleep-Based Memory Consolidation System

This demo shows the key algorithms without requiring full HoloLoom dependencies.
Run with: PYTHONPATH=. python demos/demo_sleep_consolidation.py

Created: 2025-11-17
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SimpleMemoryStats:
    """Simplified memory access stats for demo."""
    memory_id: str
    access_count: int = 0
    last_access: Optional[datetime] = None
    base_importance: float = 1.0
    current_importance: float = 1.0


class SimpleSleepConsolidation:
    """Simplified consolidation demo (no dependencies)."""

    def __init__(self, decay_rate: float = 0.95):
        self.decay_rate = decay_rate
        self.access_stats: Dict[str, SimpleMemoryStats] = {}

    def record_access(self, memory_id: str):
        """Record memory access."""
        now = datetime.now()

        if memory_id not in self.access_stats:
            self.access_stats[memory_id] = SimpleMemoryStats(
                memory_id=memory_id,
                last_access=now
            )

        stats = self.access_stats[memory_id]
        stats.access_count += 1
        stats.last_access = now

    def compute_decay(self, memory_id: str) -> float:
        """
        Compute exponential decay.

        Formula: importance = base_importance × (decay_rate ^ days_since_access)
        """
        if memory_id not in self.access_stats:
            return 1.0

        stats = self.access_stats[memory_id]

        if stats.last_access is None:
            return stats.base_importance

        days_since = (datetime.now() - stats.last_access).total_seconds() / 86400.0
        decayed = stats.base_importance * (self.decay_rate ** days_since)
        stats.current_importance = decayed

        return decayed


def main():
    """Run demonstration."""
    print("="*70)
    print("Sleep-Based Memory Consolidation Demo")
    print("Human-Like Memory Processing for HoloLoom")
    print("="*70)

    consolidator = SimpleSleepConsolidation(decay_rate=0.95)

    # Demo 1: Access Pattern Tracking
    print("\n1. ACCESS PATTERN TRACKING")
    print("-" * 70)

    memories = ["thompson_sampling", "ppo_algorithm", "curiosity_modules"]

    for memory in memories:
        consolidator.record_access(memory)

    for memory in memories:
        stats = consolidator.access_stats[memory]
        print(f"   {memory:20s}: {stats.access_count} access(es)")

    # Demo 2: Frequent Access (Promotion Candidate)
    print("\n2. FREQUENT ACCESS → PROMOTION CANDIDATE")
    print("-" * 70)

    frequent_memory = "thompson_sampling"
    for _ in range(10):
        consolidator.record_access(frequent_memory)

    stats = consolidator.access_stats[frequent_memory]
    print(f"   Memory: {frequent_memory}")
    print(f"   Accesses: {stats.access_count}")
    print(f"   Status: {'✅ Promote to long-term' if stats.access_count >= 5 else '⏳ Keep in short-term'}")

    # Demo 3: Decay Algorithm
    print("\n3. EXPONENTIAL DECAY ALGORITHM")
    print("-" * 70)
    print(f"   Formula: importance = base_importance × (decay_rate ^ days)")
    print(f"   Decay rate: {consolidator.decay_rate} (5% daily decay)\n")

    test_memory = "old_memory"
    consolidator.record_access(test_memory)
    stats = consolidator.access_stats[test_memory]

    # Simulate different time periods
    time_periods = [0, 1, 7, 14, 30, 60]

    print(f"   {'Days':<10} {'Importance':<15} {'Status':<20}")
    print(f"   {'-'*10} {'-'*15} {'-'*20}")

    for days in time_periods:
        stats.last_access = datetime.now() - timedelta(days=days)
        importance = consolidator.compute_decay(test_memory)

        status = "Fresh" if importance > 0.8 else \
                 "Decaying" if importance > 0.3 else \
                 "Archive candidate" if importance > 0.1 else \
                 "Archive now"

        print(f"   {days:<10} {importance:<15.4f} {status:<20}")

    # Demo 4: Archive Threshold
    print("\n4. ARCHIVE THRESHOLD")
    print("-" * 70)

    archive_threshold = 0.1
    print(f"   Archive threshold: {archive_threshold}")

    # Very old memory
    very_old = "ancient_memory"
    consolidator.record_access(very_old)
    stats = consolidator.access_stats[very_old]
    stats.last_access = datetime.now() - timedelta(days=50)

    importance = consolidator.compute_decay(very_old)

    print(f"\n   Memory: {very_old}")
    print(f"   Days since access: 50")
    print(f"   Current importance: {importance:.4f}")

    if importance < archive_threshold:
        print(f"   Action: 🗄️  ARCHIVE (importance {importance:.4f} < {archive_threshold})")
    else:
        print(f"   Action: ✅ KEEP (importance {importance:.4f} ≥ {archive_threshold})")

    # Demo 5: Consolidation Decision Matrix
    print("\n5. CONSOLIDATION DECISION MATRIX")
    print("-" * 70)

    print(f"\n   {'Memory':<20} {'Accesses':<10} {'Days Old':<10} {'Importance':<12} {'Decision':<20}")
    print(f"   {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*20}")

    test_cases = [
        ("frequently_used", 15, 0, "Promote to LT"),
        ("occasionally_used", 3, 5, "Keep in ST"),
        ("old_memory", 1, 30, "Decay"),
        ("very_old_memory", 1, 60, "Archive"),
    ]

    for memory_name, accesses, days_old, expected in test_cases:
        consolidator.record_access(memory_name)
        stats = consolidator.access_stats[memory_name]

        # Simulate access count
        stats.access_count = accesses

        # Simulate age
        stats.last_access = datetime.now() - timedelta(days=days_old)
        importance = consolidator.compute_decay(memory_name)

        print(f"   {memory_name:<20} {accesses:<10} {days_old:<10} {importance:<12.4f} {expected:<20}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"   Total memories tracked: {len(consolidator.access_stats)}")

    frequent = [m for m, s in consolidator.access_stats.items() if s.access_count >= 5]
    decayed = [m for m, s in consolidator.access_stats.items() if s.current_importance < 0.3]
    archive = [m for m, s in consolidator.access_stats.items() if s.current_importance < 0.1]

    print(f"   Promotion candidates (≥5 accesses): {len(frequent)}")
    print(f"   Decayed memories (<0.3 importance): {len(decayed)}")
    print(f"   Archive candidates (<0.1 importance): {len(archive)}")

    print("\n✅ Demo complete! See HoloLoom/memory/consolidation.py for full implementation.")
    print("="*70)


if __name__ == "__main__":
    main()
