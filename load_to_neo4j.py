#!/usr/bin/env python3
"""
Load YouTube Transcripts into Neo4j

Takes your saved JSON transcripts and creates the knowledge graph.

Usage:
    pip install neo4j
    python load_to_neo4j.py transcripts/VIDEO_ID_*.json

Prerequisites:
    - Neo4j running on localhost:7687
    - Password set to 'password' (or update NEO4J_PASSWORD below)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this if you used a different password


def load_transcript(filepath: Path) -> Dict[str, Any]:
    """Load transcript JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_graph_from_transcript(tx, transcript: Dict[str, Any], chunk_size: int = 60):
    """
    Create Neo4j graph from transcript.

    Graph structure:
        (Video) -[HAS_SEGMENT]-> (Segment) -[MENTIONS]-> (Entity)
    """
    video_id = transcript['video_id']

    # 1. Create Video node
    tx.run("""
        MERGE (v:Video {id: $video_id})
        SET v.url = $url,
            v.language = $language,
            v.duration = $duration,
            v.is_generated = $is_generated,
            v.ingested_at = $ingested_at
        RETURN v
    """,
        video_id=video_id,
        url=transcript['url'],
        language=transcript['language'],
        duration=transcript['duration'],
        is_generated=transcript['is_generated'],
        ingested_at=transcript['generated_at']
    )

    print(f"  ✓ Created Video node: {video_id}")

    # 2. Create Segments (chunk by time)
    segments = transcript['segments']
    current_chunk = []
    chunk_index = 0
    chunk_start = 0.0

    for seg in segments:
        seg_start = seg['start']

        # Start new chunk if needed
        if seg_start >= chunk_start + chunk_size and current_chunk:
            chunk_text = ' '.join([s['text'] for s in current_chunk])
            segment_id = f"{video_id}_seg_{chunk_index:03d}"

            # Create Segment node
            tx.run("""
                MERGE (s:Segment {id: $segment_id})
                SET s.text = $text,
                    s.chunk_index = $chunk_index,
                    s.start_time = $start_time,
                    s.end_time = $end_time,
                    s.url = $url
                RETURN s
            """,
                segment_id=segment_id,
                text=chunk_text[:1000],  # Truncate for graph storage
                chunk_index=chunk_index,
                start_time=chunk_start,
                end_time=seg_start,
                url=f"{transcript['url']}&t={int(chunk_start)}"
            )

            # Create relationship: Video -> Segment
            tx.run("""
                MATCH (v:Video {id: $video_id})
                MATCH (s:Segment {id: $segment_id})
                MERGE (v)-[r:HAS_SEGMENT {order: $order}]->(s)
                RETURN r
            """,
                video_id=video_id,
                segment_id=segment_id,
                order=chunk_index
            )

            # Extract and create Entity nodes
            entities = extract_entities(chunk_text)
            for entity in entities:
                # Create Entity node
                tx.run("""
                    MERGE (e:Entity {name: $entity})
                    RETURN e
                """, entity=entity)

                # Create relationship: Segment -> Entity
                tx.run("""
                    MATCH (s:Segment {id: $segment_id})
                    MATCH (e:Entity {name: $entity})
                    MERGE (s)-[r:MENTIONS {timestamp: $timestamp}]->(e)
                    RETURN r
                """,
                    segment_id=segment_id,
                    entity=entity,
                    timestamp=chunk_start
                )

            # Reset for next chunk
            current_chunk = []
            chunk_start = seg_start
            chunk_index += 1

        current_chunk.append(seg)

    # Handle last chunk
    if current_chunk:
        chunk_text = ' '.join([s['text'] for s in current_chunk])
        segment_id = f"{video_id}_seg_{chunk_index:03d}"

        tx.run("""
            MERGE (s:Segment {id: $segment_id})
            SET s.text = $text,
                s.chunk_index = $chunk_index,
                s.start_time = $start_time,
                s.end_time = $end_time,
                s.url = $url
            RETURN s
        """,
            segment_id=segment_id,
            text=chunk_text[:1000],
            chunk_index=chunk_index,
            start_time=chunk_start,
            end_time=transcript['duration'],
            url=f"{transcript['url']}&t={int(chunk_start)}"
        )

        tx.run("""
            MATCH (v:Video {id: $video_id})
            MATCH (s:Segment {id: $segment_id})
            MERGE (v)-[r:HAS_SEGMENT {order: $order}]->(s)
            RETURN r
        """,
            video_id=video_id,
            segment_id=segment_id,
            order=chunk_index
        )

        entities = extract_entities(chunk_text)
        for entity in entities:
            tx.run("""
                MERGE (e:Entity {name: $entity})
                RETURN e
            """, entity=entity)

            tx.run("""
                MATCH (s:Segment {id: $segment_id})
                MATCH (e:Entity {name: $entity})
                MERGE (s)-[r:MENTIONS {timestamp: $timestamp}]->(e)
                RETURN r
            """,
                segment_id=segment_id,
                entity=entity,
                timestamp=chunk_start
            )

    print(f"  ✓ Created {chunk_index + 1} Segment nodes")

    return chunk_index + 1


def extract_entities(text: str) -> List[str]:
    """Simple entity extraction - capitalize words."""
    import re

    common_words = {'The', 'This', 'That', 'These', 'Those', 'I', 'You',
                   'We', 'They', 'A', 'An', 'And', 'But', 'Or', 'So',
                   'It', 'In', 'On', 'At', 'To', 'For', 'Of', 'With'}

    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

    entities = []
    for word in words:
        if word not in common_words and len(word) > 2:
            entities.append(word)

    return list(set(entities))[:15]  # Limit to top 15


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python load_to_neo4j.py transcripts/4g251atrdX8_*.json")
        sys.exit(1)

    # Import neo4j driver
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("Error: neo4j driver not installed")
        print("Install with: pip install neo4j")
        sys.exit(1)

    # Connect to Neo4j
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✓ Connected to Neo4j\n")
    except Exception as e:
        print(f"✗ Error connecting to Neo4j: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker ps")
        print("  http://localhost:7474")
        sys.exit(1)

    # Process each transcript file
    transcript_files = []
    for pattern in sys.argv[1:]:
        transcript_files.extend(Path('.').glob(pattern))

    if not transcript_files:
        print(f"No transcript files found matching: {sys.argv[1:]}")
        sys.exit(1)

    print(f"Found {len(transcript_files)} transcript file(s)\n")
    print("="*80)

    total_segments = 0

    for filepath in transcript_files:
        print(f"\nProcessing: {filepath.name}")
        print("-"*80)

        # Load transcript
        transcript = load_transcript(filepath)

        # Create graph
        with driver.session() as session:
            num_segments = session.execute_write(
                create_graph_from_transcript,
                transcript
            )
            total_segments += num_segments

        print(f"✓ Loaded into Neo4j")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Videos loaded: {len(transcript_files)}")
    print(f"Total segments: {total_segments}")
    print()
    print("View in Neo4j Browser: http://localhost:7474")
    print()
    print("Try these queries:")
    print()
    print("1. View all videos:")
    print("   MATCH (v:Video) RETURN v")
    print()
    print("2. View video with segments:")
    print("   MATCH (v:Video)-[r:HAS_SEGMENT]->(s:Segment)")
    print("   RETURN v, r, s LIMIT 50")
    print()
    print("3. Find entities:")
    print("   MATCH (e:Entity)<-[:MENTIONS]-(s:Segment)")
    print("   RETURN e.name, count(s) as mentions")
    print("   ORDER BY mentions DESC")
    print()
    print("="*80)

    driver.close()


if __name__ == '__main__':
    main()
