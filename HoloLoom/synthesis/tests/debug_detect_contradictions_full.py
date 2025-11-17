"""Debug full contradiction detection flow."""

from datetime import datetime
from HoloLoom.synthesis.contradiction_detection import ContradictionDetector

memories = [
    {
        'id': 'm1',
        'content': 'I love Python for data science',
        'timestamp': datetime(2025, 1, 1)
    },
    {
        'id': 'm2',
        'content': 'Python is too slow for production',
        'timestamp': datetime(2025, 3, 1)
    }
]

detector = ContradictionDetector()

print("=== Adding Memories ===")
for memory in memories:
    detector.add_memory(memory)
    print(f"Added: {memory['id']}")

print(f"\nMemory history: {len(detector.memory_history)} memories")

print("\n=== Detecting Contradictions ===")
contradictions = detector.detect_contradictions()

print(f"Contradictions found: {len(contradictions)}")

if contradictions:
    for i, c in enumerate(contradictions):
        print(f"\nContradiction {i+1}:")
        print(f"  Memory 1: {c.memory1_id}")
        print(f"  Memory 2: {c.memory2_id}")
        print(f"  Type: {c.type.value}")
        print(f"  Score: {c.score:.2f}")
        print(f"  Explanation: {c.explanation}")
else:
    print("No contradictions found!")
    print("\n=== Debugging ===")
    print(f"Config enabled: {detector.config.enabled}")
    print(f"Similarity threshold: {detector.config.similarity_threshold}")
    print(f"Contradiction threshold: {detector.config.contradiction_threshold}")

    # Manual pair check
    print("\nManual pair check:")
    mem1 = detector.memory_history[0]
    mem2 = detector.memory_history[1]

    similarity = detector._calculate_similarity(mem1, mem2)
    print(f"Similarity: {similarity:.2f}")

    contradiction = detector._check_contradiction(mem1, mem2)
    if contradiction:
        print(f"Contradiction found: {contradiction.type.value}, score={contradiction.score:.2f}")
    else:
        print("No contradiction found in manual check")
