"""Debug contradiction detection."""

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

for memory in memories:
    detector.add_memory(memory)

# Calculate similarity manually
words1 = set(memories[0]['content'].lower().split())
words2 = set(memories[1]['content'].lower().split())

stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and', 'in', 'on', 'at'}
words1 -= stopwords
words2 -= stopwords

intersection = len(words1 & words2)
union = len(words1 | words2)
similarity = intersection / union if union > 0 else 0.0

print(f"Memory 1: {memories[0]['content']}")
print(f"Memory 2: {memories[1]['content']}")
print(f"\nWords 1: {words1}")
print(f"Words 2: {words2}")
print(f"Intersection: {words1 & words2}")
print(f"Similarity: {similarity:.2f}")
print(f"Threshold: {detector.config.similarity_threshold}")
print(f"Similar enough: {similarity >= detector.config.similarity_threshold}")

contradictions = detector.detect_contradictions()
print(f"\nContradictions found: {len(contradictions)}")

if contradictions:
    for c in contradictions:
        print(f"  Type: {c.type.value}, Score: {c.score:.2f}")
