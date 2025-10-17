#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AudioSpinner Example
====================
Demonstrates how to use AudioSpinner with HoloLoom Orchestrator.

Usage:
    python examples/audio_spinner_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add holoLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from holoLoom.spinningWheel import AudioSpinner, SpinnerConfig
from holoLoom.orchestrator import HoloLoomOrchestrator
from holoLoom.config import Config
from holoLoom.documentation.types import Query


async def main():
    """Example: Audio transcript -> Spinner -> Orchestrator -> Response"""
    
    print("="*80)
    print("AudioSpinner + HoloLoom Orchestrator Example")
    print("="*80 + "\n")
    
    # Sample audio transcript (bee inspection)
    raw_audio_data = {
        'transcript': """
        Today is September 15th. I'm inspecting the hives.
        
        Hive Jodi looks strong with 8 frames of brood. 
        Temperature is 72 degrees. The bees are bringing in 
        goldenrod pollen heavily.
        
        I applied thymol treatment to prevent mites. The colony 
        activity is excellent. Planning to add a honey super next week.
        """,
        'tasks': [
            {
                'title': 'Order thymol treatment',
                'priority': 1,
                'tag': 'supplies',
                'notes': 'Need for next treatment cycle'
            },
            {
                'title': 'Add honey super to Jodi',
                'priority': 2,
                'tag': 'hive_management',
                'notes': 'Colony is strong enough'
            }
        ]
    }
    
    # Configure spinner (no enrichment for this example)
    spinner_config = SpinnerConfig(
        enable_enrichment=False  # Set to True to enable Ollama
    )
    
    # Create AudioSpinner
    print("Step 1: Creating AudioSpinner...")
    spinner = AudioSpinner(spinner_config)
    
    # Spin raw data -> MemoryShards
    print("Step 2: Spinning audio data into MemoryShards...")
    shards = await spinner.spin(raw_audio_data)
    print(f"  Created {len(shards)} memory shards\n")
    
    # Show generated shards
    print("Generated Shards:")
    print("-" * 80)
    for i, shard in enumerate(shards[:3]):  # Show first 3
        print(f"Shard {i+1}:")
        print(f"  ID: {shard.id}")
        print(f"  Text: {shard.text[:60]}...")
        print(f"  Entities: {shard.entities}")
        print(f"  Motifs: {shard.motifs}")
        print()
    
    if len(shards) > 3:
        print(f"... and {len(shards) - 3} more shards\n")
    
    # Create HoloLoom config
    print("Step 3: Creating HoloLoom Orchestrator...")
    config = Config.fused()
    
    # Create orchestrator with shards
    orchestrator = HoloLoomOrchestrator(cfg=config, shards=shards)
    print("  Orchestrator initialized\n")
    
    # Process a query
    print("Step 4: Processing query...")
    query = Query(text="What's the status of Hive Jodi?")
    print(f"  Query: {query.text}\n")
    
    response = await orchestrator.process(query)
    
    # Display response
    print("="*80)
    print("RESPONSE")
    print("="*80)
    print(f"Status: {response['status']}")
    print(f"Tool: {response['tool']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Context Shards: {response['context_shards']}")
    print(f"Motifs: {response['motifs']}")
    print(f"\nResponse Text:")
    print(f"  {response['response']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())