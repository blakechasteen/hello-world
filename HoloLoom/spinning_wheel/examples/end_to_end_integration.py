#!/usr/bin/env python3
"""
End-to-End SpinningWheel Integration Example
=============================================

Demonstrates complete workflow:
1. Ingest data from multiple modalities (text, code, audio)
2. Apply enrichment pipeline
3. Feed shards to HoloLoom orchestrator
4. Process queries and generate responses

This showcases the full spinningWheel → HoloLoom pipeline.

Usage:
    python end_to_end_integration.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinning_wheel import (
    AudioSpinner, TextSpinner, CodeSpinner,
    SpinnerConfig, TextSpinnerConfig, CodeSpinnerConfig
)


class SpinningWheelDemo:
    """Comprehensive demo of SpinningWheel capabilities."""

    def __init__(self):
        self.all_shards = []

    async def demo_01_multi_modal_ingestion(self):
        """Demo 1: Ingest data from multiple modalities."""
        print("\n" + "="*60)
        print("DEMO 1: Multi-Modal Data Ingestion")
        print("="*60 + "\n")

        # 1. Audio data (beekeeping log)
        print("1. Ingesting audio transcript...")
        audio_spinner = AudioSpinner()
        audio_data = {
            'transcript': "Today I inspected all hives. Hive Jodi looks strong with "
                         "good brood pattern. Found some varroa mites in hive Melody, "
                         "will need treatment next week.",
            'summary': "Hive inspection: Jodi healthy, Melody needs varroa treatment",
            'tasks': ['Order varroa treatment supplies', 'Schedule Melody treatment'],
            'episode': 'inspection_2025_10_26'
        }

        audio_shards = await audio_spinner.spin(audio_data)
        self.all_shards.extend(audio_shards)
        print(f"   Created {len(audio_shards)} audio shards")

        # 2. Text data (notes)
        print("\n2. Ingesting text notes...")
        text_spinner = TextSpinner(TextSpinnerConfig(
            chunk_by='paragraph',
            chunk_size=200,
            extract_entities=True
        ))

        text_data = {
            'text': '''# Beekeeping Notes - Fall 2025

## Hive Status
All hives are being prepared for winter. Jodi and Melody are the strongest
colonies. Sarah's hive is smaller but should survive with feeding.

## Treatment Schedule
Varroa mite counts are elevated in Melody. Treatment protocol:
- Day 1: Apply oxalic acid treatment
- Day 7: Second application
- Day 14: Recount mites

## Winter Prep
Need to ensure all hives have:
- 40+ pounds of honey stores
- Upper entrance for ventilation
- Mouse guards installed
''',
            'source': 'fall_notes_2025.md',
            'metadata': {'author': 'beekeeper', 'season': 'fall'}
        }

        text_shards = await text_spinner.spin(text_data)
        self.all_shards.extend(text_shards)
        print(f"   Created {len(text_shards)} text shards")

        # 3. Code data (beekeeping tracking system)
        print("\n3. Ingesting code from tracking system...")
        code_spinner = CodeSpinner(CodeSpinnerConfig(
            chunk_by=None,
            extract_imports=True,
            extract_entities=True
        ))

        code_data = {
            'type': 'file',
            'path': 'hive_tracker.py',
            'content': '''
import datetime
from dataclasses import dataclass
from typing import List

@dataclass
class HiveInspection:
    """Record of hive inspection."""
    hive_name: str
    date: datetime.date
    mite_count: int
    brood_pattern: str
    honey_stores: float  # pounds
    notes: str

class HiveTracker:
    """Track multiple hives and their health."""

    def __init__(self):
        self.hives = {}
        self.inspections = []

    def add_inspection(self, inspection: HiveInspection):
        """Add inspection record."""
        self.inspections.append(inspection)

    def get_hive_history(self, hive_name: str) -> List[HiveInspection]:
        """Get all inspections for a hive."""
        return [i for i in self.inspections if i.hive_name == hive_name]

    def needs_treatment(self, hive_name: str, threshold: int = 5) -> bool:
        """Check if hive needs varroa treatment."""
        history = self.get_hive_history(hive_name)
        if not history:
            return False
        latest = history[-1]
        return latest.mite_count > threshold
''',
            'language': 'python'
        }

        code_shards = await code_spinner.spin(code_data)
        self.all_shards.extend(code_shards)
        print(f"   Created {len(code_shards)} code shards")

        print(f"\nTotal shards collected: {len(self.all_shards)}")

    async def demo_02_shard_analysis(self):
        """Demo 2: Analyze collected shards."""
        print("\n" + "="*60)
        print("DEMO 2: Shard Analysis")
        print("="*60 + "\n")

        # Count by type
        shard_types = {}
        for shard in self.all_shards:
            shard_type = shard.metadata.get('type', 'unknown')
            shard_types[shard_type] = shard_types.get(shard_type, 0) + 1

        print("Shards by type:")
        for shard_type, count in shard_types.items():
            print(f"  {shard_type}: {count}")

        # Entity extraction
        all_entities = []
        for shard in self.all_shards:
            all_entities.extend(shard.entities)

        unique_entities = set(all_entities)
        print(f"\nTotal unique entities extracted: {len(unique_entities)}")
        print(f"Entities: {', '.join(list(unique_entities)[:10])}...")

        # Find hive-related shards
        hive_shards = [s for s in self.all_shards if 'hive' in s.text.lower()]
        print(f"\nShards mentioning 'hive': {len(hive_shards)}")

    async def demo_03_semantic_search(self):
        """Demo 3: Simple semantic search across shards."""
        print("\n" + "="*60)
        print("DEMO 3: Simple Semantic Search")
        print("="*60 + "\n")

        queries = [
            "varroa mite treatment",
            "hive Jodi status",
            "winter preparation"
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")

            # Simple keyword-based search
            query_terms = set(query.lower().split())
            matches = []

            for shard in self.all_shards:
                shard_terms = set(shard.text.lower().split())
                overlap = len(query_terms & shard_terms)
                if overlap > 0:
                    matches.append((shard, overlap))

            # Sort by relevance
            matches.sort(key=lambda x: x[1], reverse=True)

            print(f"Found {len(matches)} matching shards:")
            for shard, score in matches[:3]:
                preview = shard.text[:80].replace('\n', ' ')
                print(f"  - Score {score}: {preview}...")

    async def demo_04_integration_ready(self):
        """Demo 4: Prepare shards for orchestrator integration."""
        print("\n" + "="*60)
        print("DEMO 4: Orchestrator Integration Readiness")
        print("="*60 + "\n")

        print("Shards are now ready for HoloLoom orchestrator!")
        print("\nIntegration points:")
        print("  1. Pass shards to orchestrator via:")
        print("     orchestrator = HoloLoomOrchestrator(shards=self.all_shards)")
        print("\n  2. Query processing:")
        print("     response = await orchestrator.process(Query(text='...'))")
        print("\n  3. Memory integration:")
        print("     - Shards stored in vector cache")
        print("     - Entities added to knowledge graph")
        print("     - Spectral features computed")

        print("\nShard statistics for orchestrator:")
        total_chars = sum(len(s.text) for s in self.all_shards)
        total_entities = sum(len(s.entities) for s in self.all_shards)

        print(f"  - Total shards: {len(self.all_shards)}")
        print(f"  - Total characters: {total_chars:,}")
        print(f"  - Total entities: {total_entities}")
        print(f"  - Episodes: {len(set(s.episode for s in self.all_shards))}")

    async def demo_05_enrichment_pipeline(self):
        """Demo 5: Optional enrichment pipeline."""
        print("\n" + "="*60)
        print("DEMO 5: Enrichment Pipeline (Optional)")
        print("="*60 + "\n")

        print("Enrichment options for enhanced context:")

        print("\n1. Metadata Enrichment:")
        print("   - Extract hashtags, priority, categories")
        print("   - Auto-tag based on content analysis")
        print("   - Example: #urgent, #health_check, #treatment")

        print("\n2. Semantic Enrichment (Ollama):")
        print("   - Entity extraction with LLM")
        print("   - Motif/theme detection")
        print("   - Sentiment analysis")
        print("   - Requires: pip install ollama + ollama pull llama3.2:3b")

        print("\n3. Temporal Enrichment:")
        print("   - Date extraction and normalization")
        print("   - Relative time interpretation ('yesterday' -> '2025-10-25')")
        print("   - Seasonal context detection")

        print("\n4. Graph Enrichment (Neo4j):")
        print("   - Lookup entities in knowledge graph")
        print("   - Find related entities")
        print("   - Retrieve historical relationships")
        print("   - Requires: Neo4j database connection")

        print("\n5. Memory Enrichment (Mem0):")
        print("   - Search similar past interactions")
        print("   - Pattern detection across episodes")
        print("   - Temporal continuity")
        print("   - Requires: pip install mem0ai + API key")

        print("\nTo enable enrichment:")
        print("  config = SpinnerConfig(enable_enrichment=True)")
        print("  spinner = AudioSpinner(config)")

    async def run_all_demos(self):
        """Run complete demo suite."""
        print("\n" + "="*70)
        print(" "*10 + "SpinningWheel End-to-End Integration Demo")
        print("="*70)

        await self.demo_01_multi_modal_ingestion()
        await self.demo_02_shard_analysis()
        await self.demo_03_semantic_search()
        await self.demo_04_integration_ready()
        await self.demo_05_enrichment_pipeline()

        print("\n" + "="*70)
        print(" "*20 + "Demo Complete!")
        print("="*70 + "\n")


async def main():
    """Run the demo."""
    demo = SpinningWheelDemo()
    await demo.run_all_demos()


if __name__ == '__main__':
    asyncio.run(main())
