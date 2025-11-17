"""Debug pattern synthesis."""

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

for query in queries:
    synthesizer.add_query(query)

print(f"Query history: {len(synthesizer.query_history)}")
print(f"High confidence queries: {len([q for q in synthesizer.query_history if q.get('confidence', 0) >= 0.85])}")
print(f"Confidence threshold: {synthesizer.config.confidence_threshold}")
print(f"Min cluster size: {synthesizer.config.min_cluster_size}")

patterns = synthesizer.synthesize_patterns()
print(f"Patterns found: {len(patterns)}")

if patterns:
    for i, p in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"  Content: {p.content[:100]}")
        print(f"  Sources: {len(p.source_memories)}")
        print(f"  Confidence: {p.confidence:.2f}")
