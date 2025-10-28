#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gated Multipass TextSpinner Demo with Ollama Enrichment
========================================================
Demonstrates gate-based multipass processing WITH Ollama enrichment:
1. Overview pass first (quick scan)
2. Gate decision (what concepts are present?)
3. Conditional deep passes (only if gate triggers)
4. Ollama enrichment for selected shards

Philosophy: Smart processing + AI-powered insights, but only where needed!
"""

import asyncio
import sys
import io
from pathlib import Path
from collections import Counter
import json

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct import to avoid package init issues
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load dependencies
repo_root = Path(__file__).parent
load_module("HoloLoom.spinning_wheel.base", repo_root / "HoloLoom" / "spinningWheel" / "base.py")
text_module = load_module("HoloLoom.spinning_wheel.text", repo_root / "HoloLoom" / "spinningWheel" / "text.py")

TextSpinner = text_module.TextSpinner
TextSpinnerConfig = text_module.TextSpinnerConfig
spin_text = text_module.spin_text


# The mem0ai PyPI page text (shortened for demo)
MEM0_TEXT = """Skip to main content
PyPI
mem0ai 0.1.20a0

Mem0 - The Memory Layer for Personalized AI

Introduction

Mem0 (pronounced as "mem-zero") enhances AI assistants and agents with an intelligent memory layer, enabling personalized AI interactions. Mem0 remembers user preferences, adapts to individual needs, and continuously improves over time, making it ideal for customer support chatbots, AI assistants, and autonomous systems.

New Feature: Introducing Graph Memory. Check out our documentation.

Core Features

    Multi-Level Memory: User, Session, and AI Agent memory retention
    Adaptive Personalization: Continuous improvement based on interactions
    Developer-Friendly API: Simple integration into various applications
    Cross-Platform Consistency: Uniform behavior across devices
    Managed Service: Hassle-free hosted solution

How Mem0 works?

Mem0 leverages a hybrid database approach to manage and retrieve long-term memories for AI agents and assistants. Each memory is associated with a unique identifier, such as a user ID or agent ID, allowing Mem0 to organize and access memories specific to an individual or context.

When a message is added to the Mem0 using add() method, the system extracts relevant facts and preferences and stores it across data stores: a vector database, a key-value database, and a graph database. This hybrid approach ensures that different types of information are stored in the most efficient manner, making subsequent searches quick and effective.

Basic Usage

Mem0 requires an LLM to function, with gpt-4o from OpenAI as the default. However, it supports a variety of LLMs; for details, refer to our Supported LLMs documentation.

First step is to instantiate the memory:

from mem0 import Memory

m = Memory()

You can perform the following task on the memory:

    Add: Store a memory from any unstructured text
    Update: Update memory of a given memory_id
    Search: Fetch memories based on a query
    Get: Return memories for a certain user/agent/session
    History: Describe how a memory has changed over time for a specific memory ID

# 1. Add: Store a memory from any unstructured text
result = m.add("I am working on improving my tennis skills.", user_id="alice")

# 2. Update: update the memory
result = m.update(memory_id=<memory_id>, data="Likes to play tennis on weekends")

# 3. Search: search related memories
related_memories = m.search(query="What are Alice's hobbies?", user_id="alice")

Graph Memory

To initialize Graph Memory you'll need to set up your configuration with graph store providers. Currently, we support Neo4j as a graph store provider.

from mem0 import Memory

config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://xxx",
            "username": "neo4j",
            "password": "xxx"
        }
    },
    "version": "v1.1"
}

m = Memory.from_config(config_dict=config)
"""


class ProcessingGate:
    """
    Determines processing strategy based on overview scan.
    """

    def __init__(self):
        self.triggers = {
            'technical_doc': ['API', 'Python', 'install', 'documentation', 'code'],
            'graph_relevant': ['Neo4j', 'Graph', 'graph', 'knowledge', 'relationship'],
            'memory_system': ['Memory', 'memory', 'remember', 'recall', 'store'],
            'ai_agent': ['AI', 'agent', 'LLM', 'assistant', 'chatbot'],
        }

    def evaluate(self, overview_shard):
        """Evaluate overview shard and determine processing gates."""
        text = overview_shard.text.lower()
        entities = [e.lower() for e in overview_shard.entities]

        # Check each gate
        gates = {}
        for gate_name, keywords in self.triggers.items():
            matches = sum(1 for kw in keywords if kw.lower() in text or kw.lower() in ' '.join(entities))
            gates[gate_name] = matches > 0
            gates[f'{gate_name}_score'] = matches

        # Determine processing strategy
        strategy = self._determine_strategy(gates, overview_shard)

        return {
            'gates': gates,
            'strategy': strategy,
            'shard_length': len(overview_shard.text),
            'entity_count': len(overview_shard.entities),
        }

    def _determine_strategy(self, gates, shard):
        """Determine processing strategy based on gate results."""
        if shard.metadata.get('char_count', 0) < 100:
            return 'SKIP'

        if gates['graph_relevant'] and gates['technical_doc']:
            return 'FINE_TECHNICAL'
        elif gates['memory_system'] or gates['ai_agent']:
            return 'MEDIUM_SEMANTIC'
        elif gates['technical_doc']:
            return 'COARSE_OVERVIEW'
        else:
            return 'SINGLE_SHARD'


async def enrich_with_ollama(shard, model="llama3.2:3b"):
    """
    Enrich a shard with Ollama-powered insights.

    Extracts:
    - Key concepts
    - Technical terms
    - Relationships
    """
    import requests

    prompt = f"""Analyze this technical text and extract:
1. Key concepts (3-5 main ideas)
2. Technical terms (specific jargon)
3. Relationships (how concepts connect)

Text: {shard.text[:500]}

Respond in JSON format:
{{"concepts": ["concept1", "concept2"], "terms": ["term1", "term2"], "relationships": ["rel1", "rel2"]}}"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.1}
            },
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '{}')

            # Try to extract JSON from response
            try:
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    enrichment = json.loads(json_str)
                    return enrichment
            except:
                pass

        return {'concepts': [], 'terms': [], 'relationships': []}
    except Exception as e:
        print(f"  [Warning] Ollama enrichment failed: {e}")
        return {'concepts': [], 'terms': [], 'relationships': []}


async def pass0_overview(text, source='document'):
    """Pass 0: Quick overview scan to determine gates."""
    print("\n" + "=" * 80)
    print("PASS 0: OVERVIEW SCAN (Gate Determination)")
    print("=" * 80)

    shards = await spin_text(text=text, source=source, chunk_by=None)
    overview = shards[0]

    print(f"\nOverview Shard Generated:")
    print(f"  ID: {overview.id}")
    print(f"  Length: {len(overview.text)} chars")
    print(f"  Entities extracted: {len(overview.entities)}")
    print(f"  Top entities: {overview.entities[:10]}")

    # Evaluate gates
    gate = ProcessingGate()
    decision = gate.evaluate(overview)

    print(f"\nGATE EVALUATION:")
    print(f"  Technical Doc:    {'YES' if decision['gates']['technical_doc'] else 'NO'}  (score: {decision['gates']['technical_doc_score']})")
    print(f"  Graph Relevant:   {'YES' if decision['gates']['graph_relevant'] else 'NO'}  (score: {decision['gates']['graph_relevant_score']})")
    print(f"  Memory System:    {'YES' if decision['gates']['memory_system'] else 'NO'}  (score: {decision['gates']['memory_system_score']})")
    print(f"  AI Agent Topic:   {'YES' if decision['gates']['ai_agent'] else 'NO'}  (score: {decision['gates']['ai_agent_score']})")

    print(f"\nGATE DECISION:")
    print(f"  Strategy: {decision['strategy']}")

    return overview, decision


async def pass1_conditional(text, source, strategy):
    """Pass 1: Conditional processing based on gate decision."""
    print("\n" + "=" * 80)
    print(f"PASS 1: CONDITIONAL PROCESSING ({strategy})")
    print("=" * 80)

    if strategy == 'SKIP':
        print("  X Skipping - document too small")
        return []

    elif strategy == 'SINGLE_SHARD':
        print("  -> Keeping as single shard")
        shards = await spin_text(text=text, source=source, chunk_by=None)

    elif strategy == 'COARSE_OVERVIEW':
        print("  -> Coarse chunking (500 chars)")
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=500, extract_entities=True)
        spinner = TextSpinner(config)
        shards = await spinner.spin({'text': text, 'source': source})

    elif strategy == 'MEDIUM_SEMANTIC':
        print("  -> Medium chunking (300 chars)")
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=300, extract_entities=True)
        spinner = TextSpinner(config)
        shards = await spinner.spin({'text': text, 'source': source})

    elif strategy == 'FINE_TECHNICAL':
        print("  -> Fine chunking (150 chars)")
        config = TextSpinnerConfig(chunk_by='sentence', chunk_size=150, min_chunk_size=50, extract_entities=True)
        spinner = TextSpinner(config)
        shards = await spinner.spin({'text': text, 'source': source})

    print(f"\n  Generated {len(shards)} shards")
    return shards


async def pass2_ollama_enrichment(shards, gates, max_enrich=3):
    """Pass 2: Ollama-powered enrichment for selected shards."""
    print("\n" + "=" * 80)
    print("PASS 2: OLLAMA ENRICHMENT (AI-Powered Analysis)")
    print("=" * 80)

    # Select shards for enrichment (most relevant ones)
    enrichment_candidates = []

    for shard in shards[:max_enrich]:  # Limit to first few shards for demo
        text_lower = shard.text.lower()

        # Prioritize graph-related or memory-related shards
        if 'neo4j' in text_lower or 'graph' in text_lower or 'memory' in text_lower:
            enrichment_candidates.append(shard)

    print(f"\n  Selected {len(enrichment_candidates)} shards for Ollama enrichment...")

    enriched_shards = []
    for i, shard in enumerate(enrichment_candidates, 1):
        print(f"\n  [{i}/{len(enrichment_candidates)}] Enriching shard: {shard.id[:40]}...")
        print(f"      Text preview: {shard.text[:80]}...")

        enrichment = await enrich_with_ollama(shard)

        if enrichment.get('concepts'):
            concepts = [str(c) for c in enrichment['concepts'][:3]]
            print(f"      Concepts: {', '.join(concepts)}")
        if enrichment.get('terms'):
            terms = [str(t) for t in enrichment['terms'][:3]]
            print(f"      Terms: {', '.join(terms)}")
        if enrichment.get('relationships'):
            rels = [str(r) for r in enrichment['relationships'][:2]]
            print(f"      Relationships: {', '.join(rels)}")

        # Add enrichment to shard metadata
        shard.metadata['ollama_enrichment'] = enrichment
        enriched_shards.append(shard)

    return enriched_shards


async def main():
    """Run gated multipass processing with Ollama enrichment."""
    print("\n" + "=" * 80)
    print("GATED MULTIPASS TEXT SPINNER + OLLAMA DEMO")
    print("   Philosophy: Overview -> Gate -> Conditional -> AI Enrichment")
    print("=" * 80)

    # Pass 0: Overview scan
    overview_shard, decision = await pass0_overview(MEM0_TEXT, 'mem0ai_docs.txt')

    # Pass 1: Conditional processing
    strategy = decision['strategy']
    shards = await pass1_conditional(MEM0_TEXT, 'mem0ai_docs.txt', strategy)

    # Pass 2: Ollama enrichment (only for relevant shards)
    enriched_shards = []
    if shards and strategy != 'SKIP':
        enriched_shards = await pass2_ollama_enrichment(shards, decision['gates'], max_enrich=3)

    # Summary
    print("\n" + "=" * 80)
    print("GATED PROCESSING + OLLAMA COMPLETE")
    print("=" * 80)
    print(f"\n  Overview: 1 shard analyzed")
    print(f"  Strategy: {strategy}")
    print(f"  Processing: {len(shards)} shards generated")
    print(f"  Ollama Enrichment: {len(enriched_shards)} shards enhanced with AI insights")

    print("\nKey Benefits:")
    print("  1. Overview pass decides processing strategy")
    print("  2. Gate-based conditional processing (efficient)")
    print("  3. Ollama enrichment adds AI-powered insights")
    print("  4. Fully local - no API costs!")
    print("  5. Enrichment only applied where valuable")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
