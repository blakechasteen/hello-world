"""Debug similarity calculation."""

from datetime import datetime
from HoloLoom.synthesis.pattern_synthesis import PatternSynthesizer

def get_sample_queries():
    return [
        {
            'text': 'What is Thompson Sampling?',
            'confidence': 0.92,
            'timestamp': datetime(2025, 11, 1, 10, 0),
            'entities': ['Thompson Sampling'],
            'motifs': ['exploration', 'exploitation']
        },
        {
            'text': 'How does Thompson Sampling work?',
            'confidence': 0.89,
            'timestamp': datetime(2025, 11, 1, 10, 15),
            'entities': ['Thompson Sampling'],
            'motifs': ['algorithm', 'bayesian']
        },
        {
            'text': 'Thompson vs epsilon-greedy',
            'confidence': 0.91,
            'timestamp': datetime(2025, 11, 1, 10, 30),
            'entities': ['Thompson Sampling', 'epsilon-greedy'],
            'motifs': ['comparison', 'exploration']
        }
    ]

synthesizer = PatternSynthesizer()
queries = get_sample_queries()

# Test similarity between queries
for i, q1 in enumerate(queries):
    for j, q2 in enumerate(queries[i+1:], i+1):
        # Check if q2 would be similar to cluster containing q1
        similarity = synthesizer._is_similar_to_cluster(q2, [q1])
        words1 = set(q1['text'].lower().split())
        words2 = set(q2['text'].lower().split())
        overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        print(f"\nQ{i+1} vs Q{j+1}:")
        print(f"  '{q1['text']}'")
        print(f"  '{q2['text']}'")
        print(f"  Overlap: {overlap:.2f}")
        print(f"  Similar: {similarity}")
        print(f"  Threshold: {synthesizer.config.similarity_threshold}")
