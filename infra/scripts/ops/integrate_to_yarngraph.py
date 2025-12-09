#!/usr/bin/env python3
"""
Integrate YouTube Transcripts into HoloLoom YarnGraph

Shows how saved transcripts flow into the knowledge graph system.
This is the foundation for semantic search and knowledge retrieval.

Architecture:
    YouTube → YouTubeSpinner → MemoryShards → YarnGraph → Query/Retrieval

Usage:
    python integrate_to_yarngraph.py path/to/transcript.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_transcript(filepath: Path) -> Dict[str, Any]:
    """Load a saved transcript JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_memory_shards(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert transcript into MemoryShards format.

    This is the format expected by HoloLoom's orchestrator and YarnGraph.
    """
    video_id = transcript['video_id']

    # Option 1: Single shard for entire video
    single_shard = {
        'id': f"youtube_{video_id}",
        'text': transcript['full_text'],
        'episode': f"youtube_{video_id}",
        'entities': extract_entities_from_text(transcript['full_text']),
        'motifs': [],  # Will be filled by orchestrator's motif detector
        'metadata': {
            'source': 'youtube',
            'video_id': video_id,
            'url': transcript['url'],
            'language': transcript['language'],
            'duration': transcript['duration'],
            'is_generated': transcript['is_generated'],
            'ingested_at': transcript['generated_at']
        }
    }

    # Option 2: Multiple shards (one per minute, for better granularity)
    chunked_shards = []
    chunk_duration = 60.0  # 60 seconds per chunk
    current_chunk = []
    chunk_index = 0
    chunk_start = 0

    for seg in transcript['segments']:
        seg_start = seg['start']

        # Start new chunk if needed
        if seg_start >= chunk_start + chunk_duration and current_chunk:
            chunk_text = ' '.join([s['text'] for s in current_chunk])

            chunked_shards.append({
                'id': f"youtube_{video_id}_chunk_{chunk_index:03d}",
                'text': chunk_text,
                'episode': f"youtube_{video_id}",
                'entities': extract_entities_from_text(chunk_text),
                'motifs': [],
                'metadata': {
                    'source': 'youtube',
                    'video_id': video_id,
                    'url': f"{transcript['url']}&t={int(chunk_start)}",
                    'chunk_index': chunk_index,
                    'chunk_start': chunk_start,
                    'chunk_end': seg_start,
                    'language': transcript['language']
                }
            })

            current_chunk = []
            chunk_start = seg_start
            chunk_index += 1

        current_chunk.append(seg)

    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join([s['text'] for s in current_chunk])
        chunked_shards.append({
            'id': f"youtube_{video_id}_chunk_{chunk_index:03d}",
            'text': chunk_text,
            'episode': f"youtube_{video_id}",
            'entities': extract_entities_from_text(chunk_text),
            'motifs': [],
            'metadata': {
                'source': 'youtube',
                'video_id': video_id,
                'url': f"{transcript['url']}&t={int(chunk_start)}",
                'chunk_index': chunk_index,
                'chunk_start': chunk_start,
                'chunk_end': transcript['duration'],
                'language': transcript['language']
            }
        })

    return {
        'single_shard': single_shard,
        'chunked_shards': chunked_shards
    }


def extract_entities_from_text(text: str) -> List[str]:
    """
    Simple entity extraction.
    In production, this would use:
    - spaCy NER
    - Ollama local LLM
    - Or the HoloLoom motif detector
    """
    import re

    # Extract capitalized words (simple heuristic)
    entities = []
    common_words = {'The', 'This', 'That', 'These', 'Those', 'I', 'You', 'We', 'They', 'A', 'An'}

    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

    for word in words:
        if word not in common_words and len(word) > 2:
            entities.append(word)

    return list(set(entities))[:20]  # Limit to top 20


def create_neo4j_nodes(shards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create Neo4j node definitions for the YarnGraph.

    Graph structure:
        (Video) -[HAS_SEGMENT]-> (Segment)
        (Segment) -[MENTIONS]-> (Entity)
        (Entity) -[RELATED_TO]-> (Entity)
    """
    nodes = []
    relationships = []

    # Video node
    video_id = shards[0]['metadata']['video_id']
    nodes.append({
        'type': 'Video',
        'id': video_id,
        'properties': {
            'url': shards[0]['metadata']['url'],
            'language': shards[0]['metadata']['language'],
            'duration': shards[0]['metadata'].get('duration', 0)
        }
    })

    # Segment nodes and relationships
    for shard in shards:
        # Segment node
        nodes.append({
            'type': 'Segment',
            'id': shard['id'],
            'properties': {
                'text': shard['text'][:500],  # Truncate for graph storage
                'chunk_start': shard['metadata'].get('chunk_start', 0),
                'chunk_end': shard['metadata'].get('chunk_end', 0)
            }
        })

        # Video -> Segment relationship
        relationships.append({
            'from': video_id,
            'to': shard['id'],
            'type': 'HAS_SEGMENT',
            'properties': {
                'order': shard['metadata'].get('chunk_index', 0)
            }
        })

        # Entity nodes and relationships
        for entity in shard['entities']:
            nodes.append({
                'type': 'Entity',
                'id': entity,
                'properties': {
                    'name': entity
                }
            })

            # Segment -> Entity relationship
            relationships.append({
                'from': shard['id'],
                'to': entity,
                'type': 'MENTIONS',
                'properties': {
                    'timestamp': shard['metadata'].get('chunk_start', 0)
                }
            })

    return {
        'nodes': nodes,
        'relationships': relationships
    }


def create_qdrant_points(shards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create Qdrant point definitions for vector storage.

    Each shard becomes a vector point for semantic search.
    Note: Actual embedding generation would happen in the orchestrator.
    """
    points = []

    for shard in shards:
        points.append({
            'id': shard['id'],
            'vector': None,  # Would be generated by embedding model
            'payload': {
                'text': shard['text'],
                'video_id': shard['metadata']['video_id'],
                'url': shard['metadata']['url'],
                'chunk_start': shard['metadata'].get('chunk_start', 0),
                'chunk_end': shard['metadata'].get('chunk_end', 0),
                'entities': shard['entities']
            }
        })

    return points


def main():
    """Demonstrate the integration pipeline."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python integrate_to_yarngraph.py transcripts/VIDEO_ID_20251023_143022.json")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])

    if not transcript_path.exists():
        print(f"Error: File not found: {transcript_path}")
        sys.exit(1)

    print("="*80)
    print("YouTube → YarnGraph Integration Pipeline")
    print("="*80)
    print()

    # Step 1: Load transcript
    print("Step 1: Loading transcript...")
    transcript = load_transcript(transcript_path)
    print(f"  ✓ Loaded: {transcript['video_id']}")
    print(f"    Duration: {transcript['duration']:.1f}s")
    print(f"    Segments: {len(transcript['segments'])}")
    print()

    # Step 2: Create memory shards
    print("Step 2: Creating MemoryShards...")
    shards = create_memory_shards(transcript)
    print(f"  ✓ Single shard: 1")
    print(f"  ✓ Chunked shards: {len(shards['chunked_shards'])}")
    print()

    # Step 3: Prepare Neo4j graph structure
    print("Step 3: Preparing Neo4j graph structure...")
    graph_data = create_neo4j_nodes(shards['chunked_shards'])
    print(f"  ✓ Nodes: {len(graph_data['nodes'])}")
    print(f"  ✓ Relationships: {len(graph_data['relationships'])}")
    print()

    # Step 4: Prepare Qdrant vectors
    print("Step 4: Preparing Qdrant vector points...")
    vector_points = create_qdrant_points(shards['chunked_shards'])
    print(f"  ✓ Vector points: {len(vector_points)}")
    print()

    # Summary
    print("="*80)
    print("INTEGRATION SUMMARY")
    print("="*80)
    print()
    print("Memory Shards:")
    print(f"  - {len(shards['chunked_shards'])} chunks ready for ingestion")
    print(f"  - Entities extracted: {len(shards['chunked_shards'][0]['entities'])} (sample)")
    print()
    print("Neo4j (YarnGraph - Symbolic):")
    print(f"  - Video nodes: 1")
    print(f"  - Segment nodes: {len(shards['chunked_shards'])}")
    print(f"  - Entity nodes: {len(set([e for s in shards['chunked_shards'] for e in s['entities']]))}")
    print(f"  - Total relationships: {len(graph_data['relationships'])}")
    print()
    print("Qdrant (Vector Store - Semantic):")
    print(f"  - Vector points: {len(vector_points)}")
    print(f"  - Ready for embedding generation")
    print()
    print("Next Steps:")
    print("  1. Generate embeddings using sentence-transformers or OpenAI")
    print("  2. Insert graph data into Neo4j")
    print("  3. Insert vectors into Qdrant")
    print("  4. Query via HoloLoom Orchestrator")
    print()
    print("Example query: 'What topics are discussed in this video?'")
    print("  → Orchestrator retrieves relevant shards via semantic search")
    print("  → Expands context using graph relationships")
    print("  → Generates response using policy engine")
    print()
    print("="*80)


if __name__ == '__main__':
    main()
