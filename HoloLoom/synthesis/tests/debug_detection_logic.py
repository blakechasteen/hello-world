"""Debug contradiction detection logic in detail."""

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

# Debug sentiment detection
text1 = memories[0]['content'].lower()
text2 = memories[1]['content'].lower()

words1 = set(text1.split())
words2 = set(text2.split())

print("=== Sentiment Analysis ===")
print(f"Text 1: {text1}")
print(f"Text 2: {text2}")

positive1 = len(words1 & detector.positive_words)
negative1 = len(words1 & detector.negative_words)
positive2 = len(words2 & detector.positive_words)
negative2 = len(words2 & detector.negative_words)

print(f"\nText 1:")
print(f"  Positive words: {words1 & detector.positive_words} (count: {positive1})")
print(f"  Negative words: {words1 & detector.negative_words} (count: {negative1})")

print(f"\nText 2:")
print(f"  Positive words: {words2 & detector.positive_words} (count: {positive2})")
print(f"  Negative words: {words2 & detector.negative_words} (count: {negative2})")

sentiment1 = 'positive' if positive1 > negative1 else 'negative' if negative1 > positive1 else 'neutral'
sentiment2 = 'positive' if positive2 > negative2 else 'negative' if negative2 > positive2 else 'neutral'

print(f"\nSentiment 1: {sentiment1}")
print(f"Sentiment 2: {sentiment2}")

reversal = (sentiment1 == 'positive' and sentiment2 == 'negative') or \
           (sentiment1 == 'negative' and sentiment2 == 'positive')

print(f"Sentiment reversal detected: {reversal}")

# Check actual contradiction detection
result = detector._check_contradiction(memories[0], memories[1])
if result:
    print(f"\n=== Contradiction Found ===")
    print(f"Type: {result.type.value}")
    print(f"Score: {result.score:.2f}")
    print(f"Explanation: {result.explanation}")
else:
    print(f"\n=== No Contradiction Found ===")
    print("Checking why...")

    # Manual similarity check
    similarity = detector._calculate_similarity(memories[0], memories[1])
    print(f"Similarity: {similarity:.2f} (threshold: {detector.config.similarity_threshold})")

    if similarity < detector.config.similarity_threshold:
        print("FAILED: Similarity too low")
    else:
        print("PASSED: Similarity check")
