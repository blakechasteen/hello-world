"""Debug clustering logic in detail."""

from datetime import datetime
from HoloLoom.synthesis.pattern_synthesis import PatternSynthesizer, PatternSynthesisConfig

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

config = PatternSynthesisConfig(similarity_threshold=0.15)
synthesizer = PatternSynthesizer(config=config)
queries = get_sample_queries()

# Add queries
for query in queries:
    synthesizer.add_query(query)

# Get high-confidence queries
high_conf = [
    q for q in synthesizer.query_history
    if q.get('confidence', 0) >= synthesizer.config.confidence_threshold
]

print(f"High-confidence queries: {len(high_conf)}")
print(f"Min cluster size: {synthesizer.config.min_cluster_size}")

# Manual clustering to debug
clusters = synthesizer._cluster_queries(high_conf)
print(f"\nClusters formed: {len(clusters)}")

for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1}: {len(cluster)} queries")
    for q in cluster:
        print(f"  - {q['text']}")

    # Try to generate summary
    summary = synthesizer._generate_pattern_summary(cluster)
    if summary:
        print(f"  Summary: {summary.content}")
    else:
        print(f"  No summary generated (missing common entities/motifs)")
        # Debug why
        all_entities = []
        all_motifs = []
        for q in cluster:
            all_entities.extend(q.get('entities', []))
            all_motifs.extend(q.get('motifs', []))
        print(f"  All entities: {all_entities}")
        print(f"  All motifs: {all_motifs}")

        common_entities = synthesizer._find_common_terms(all_entities, min_count=2)
        common_motifs = synthesizer._find_common_terms(all_motifs, min_count=2)
        print(f"  Common entities (min_count=2): {common_entities}")
        print(f"  Common motifs (min_count=2): {common_motifs}")
