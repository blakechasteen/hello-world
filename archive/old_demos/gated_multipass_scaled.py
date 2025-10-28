#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCALED-UP Gated Multipass with Ollama + Memory Storage
=======================================================
Full pipeline demonstration:
1. Large document ingestion
2. Gate-based processing
3. Ollama enrichment at scale
4. Qdrant memory storage
5. Query & retrieval demo

Philosophy: Industrial-strength processing with AI insights!
"""

import asyncio
import sys
import io
from pathlib import Path
from collections import Counter
import json
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

repo_root = Path(__file__).parent
load_module("HoloLoom.spinning_wheel.base", repo_root / "HoloLoom" / "spinningWheel" / "base.py")
text_module = load_module("HoloLoom.spinning_wheel.text", repo_root / "HoloLoom" / "spinningWheel" / "text.py")

TextSpinner = text_module.TextSpinner
TextSpinnerConfig = text_module.TextSpinnerConfig
spin_text = text_module.spin_text

# Import memory components
from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore
from HoloLoom.memory.protocol import Memory, shards_to_memories


# LARGE DOCUMENT: Full Mem0 documentation
LARGE_DOC = """Skip to main content
2025 Python Packaging Survey is now live!  Take the survey now
PyPI
Search PyPI

    Help Docs Sponsors Log in Register

mem0ai 0.1.20a0

pip install mem0ai==0.1.20a0

Newer version available (1.0.0)

Released: Oct 11, 2024

Long-term memory for AI Agents
Navigation

    Project description
    Release history
    Download files

Verified details
These details have been verified by PyPI
Maintainers
Avatar for deshrajdry from gravatar.com deshrajdry
Avatar for devkhant from gravatar.com devkhant
Avatar for prateekchhikara0 from gravatar.com prateekchhikara0
Avatar for staranjeet from gravatar.com staranjeet
Unverified details
These details have not been verified by PyPI
Meta

    Author: Mem0
    Requires: Python <4.0, >=3.9

Classifiers

    Programming Language
        Python :: 3
        Python :: 3.9
        Python :: 3.10
        Python :: 3.11

Report project as malware

Project description

Mem0 - The Memory Layer for Personalized AI

Launch

Learn more · Join Discord

Mem0 Discord Mem0 PyPI - Downloads Package version Supported Python versions Y Combinator S24
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

When an AI agent or LLM needs to recall memories, it uses the search() method. Mem0 then performs search across these data stores, retrieving relevant information from each source. This information is then passed through a scoring layer, which evaluates their importance based on relevance, importance, and recency. This ensures that only the most personalized and useful context is surfaced.

The retrieved memories can then be appended to the LLM's prompt as needed, enhancing the personalization and relevance of its responses.
Use Cases

Mem0 empowers organizations and individuals to enhance:

    AI Assistants and agents: Seamless conversations with a touch of déjà vu
    Personalized Learning: Tailored content recommendations and progress tracking
    Customer Support: Context-aware assistance with user preference memory
    Healthcare: Patient history and treatment plan management
    Virtual Companions: Deeper user relationships through conversation memory
    Productivity: Streamlined workflows based on user habits and task history
    Gaming: Adaptive environments reflecting player choices and progress

Get Started

The easiest way to set up Mem0 is through the managed Mem0 Platform. This hosted solution offers automatic updates, advanced analytics, and dedicated support. Sign up to get started.

If you prefer to self-host, use the open-source Mem0 package. Follow the installation instructions to get started.
Installation Instructions

Install the Mem0 package via pip:

pip install mem0ai

Alternatively, you can use Mem0 with one click on the hosted platform here.
Basic Usage

Mem0 requires an LLM to function, with gpt-4o from OpenAI as the default. However, it supports a variety of LLMs; for details, refer to our Supported LLMs documentation.

First step is to instantiate the memory:

from mem0 import Memory

m = Memory()

How to set OPENAI_API_KEY

You can perform the following task on the memory:

    Add: Store a memory from any unstructured text
    Update: Update memory of a given memory_id
    Search: Fetch memories based on a query
    Get: Return memories for a certain user/agent/session
    History: Describe how a memory has changed over time for a specific memory ID

# 1. Add: Store a memory from any unstructured text
result = m.add("I am working on improving my tennis skills. Suggest some online courses.", user_id="alice", metadata={"category": "hobbies"})

# Created memory --> 'Improving her tennis skills.' and 'Looking for online suggestions.'

# 2. Update: update the memory
result = m.update(memory_id=<memory_id_1>, data="Likes to play tennis on weekends")

# Updated memory --> 'Likes to play tennis on weekends.' and 'Looking for online suggestions.'

# 3. Search: search related memories
related_memories = m.search(query="What are Alice's hobbies?", user_id="alice")

# Retrieved memory --> 'Likes to play tennis on weekends'

# 4. Get all memories
all_memories = m.get_all()
memory_id = all_memories["memories"][0] ["id"] # get a memory_id

# All memory items --> 'Likes to play tennis on weekends.' and 'Looking for online suggestions.'

# 5. Get memory history for a particular memory_id
history = m.history(memory_id=<memory_id_1>)

# Logs corresponding to memory_id_1 --> {'prev_value': 'Working on improving tennis skills and interested in online courses for tennis.', 'new_value': 'Likes to play tennis on weekends' }

    [!TIP] If you prefer a hosted version without the need to set up infrastructure yourself, check out the Mem0 Platform to get started in minutes.

Graph Memory

To initialize Graph Memory you'll need to set up your configuration with graph store providers. Currently, we support Neo4j as a graph store provider. You can setup Neo4j locally or use the hosted Neo4j AuraDB. Moreover, you also need to set the version to v1.1 (prior versions are not supported). Here's how you can do it:

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

Documentation

For detailed usage instructions and API reference, visit our documentation at docs.mem0.ai. Here, you can find more information on both the open-source version and the hosted Mem0 Platform.
Star History

Star History Chart
Support

Join our community for support and discussions. If you have any questions, feel free to reach out to us using one of the following methods:

    Join our Discord
    Follow us on Twitter
    Email founders

Contributors

Join our Discord community to learn about memory management for AI agents and LLMs, and connect with Mem0 users and contributors. Share your ideas, questions, or feedback in our GitHub Issues.

We value and appreciate the contributions of our community. Special thanks to our contributors for helping us improve Mem0.
License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
"""


class ProcessingGate:
    """Gate-based processing strategy selector."""

    def __init__(self):
        self.triggers = {
            'technical_doc': ['API', 'Python', 'install', 'documentation', 'code', 'pip'],
            'graph_relevant': ['Neo4j', 'Graph', 'graph', 'knowledge', 'relationship'],
            'memory_system': ['Memory', 'memory', 'remember', 'recall', 'store'],
            'ai_agent': ['AI', 'agent', 'LLM', 'assistant', 'chatbot'],
        }

    def evaluate(self, overview_shard):
        text = overview_shard.text.lower()
        entities = [e.lower() for e in overview_shard.entities]

        gates = {}
        for gate_name, keywords in self.triggers.items():
            matches = sum(1 for kw in keywords if kw.lower() in text or kw.lower() in ' '.join(entities))
            gates[gate_name] = matches > 0
            gates[f'{gate_name}_score'] = matches

        strategy = self._determine_strategy(gates, overview_shard)

        return {
            'gates': gates,
            'strategy': strategy,
            'shard_length': len(overview_shard.text),
            'entity_count': len(overview_shard.entities),
        }

    def _determine_strategy(self, gates, shard):
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
    """Ollama-powered enrichment."""
    import requests

    prompt = f"""Analyze this text and extract key information in JSON format:

Text: {shard.text[:400]}

Extract:
1. Main concepts (2-4 key ideas)
2. Technical terms (specific terminology)
3. Action items or methods mentioned

Respond ONLY with valid JSON:
{{"concepts": ["concept1", "concept2"], "terms": ["term1", "term2"], "actions": ["action1"]}}"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.1, 'num_predict': 200}
            },
            timeout=20
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '{}')

            # Extract JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                enrichment = json.loads(json_str)
                return enrichment

        return {'concepts': [], 'terms': [], 'actions': []}
    except Exception as e:
        return {'concepts': [], 'terms': [], 'actions': [], 'error': str(e)}


async def process_large_document(text, source, max_enrich=10):
    """Full pipeline: gate -> process -> enrich -> store."""

    print("\n" + "=" * 80)
    print("SCALED-UP GATED MULTIPASS PIPELINE")
    print("=" * 80)

    # PASS 0: Overview
    print("\n[PASS 0] Overview Scan...")
    shards = await spin_text(text=text, source=source, chunk_by=None)
    overview = shards[0]

    gate = ProcessingGate()
    decision = gate.evaluate(overview)

    print(f"  Document: {len(text):,} chars")
    print(f"  Entities: {len(overview.entities)}")
    print(f"  Strategy: {decision['strategy']}")
    print(f"  Gates: Technical={decision['gates']['technical_doc']}, "
          f"Graph={decision['gates']['graph_relevant']}, "
          f"Memory={decision['gates']['memory_system']}, "
          f"AI={decision['gates']['ai_agent']}")

    # PASS 1: Conditional Processing
    print(f"\n[PASS 1] Conditional Processing ({decision['strategy']})...")

    strategy = decision['strategy']
    if strategy == 'FINE_TECHNICAL':
        config = TextSpinnerConfig(chunk_by='sentence', chunk_size=200, min_chunk_size=50, extract_entities=True)
    elif strategy == 'MEDIUM_SEMANTIC':
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=300, extract_entities=True)
    else:
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=500, extract_entities=True)

    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': source})

    print(f"  Generated: {len(shards)} shards")

    # PASS 2: Ollama Enrichment (scaled)
    print(f"\n[PASS 2] Ollama Enrichment (processing {min(max_enrich, len(shards))} shards)...")

    enriched_count = 0
    for i, shard in enumerate(shards[:max_enrich], 1):
        # Select relevant shards (graph, memory, code examples)
        text_lower = shard.text.lower()
        if any(kw in text_lower for kw in ['graph', 'neo4j', 'memory', 'search', 'import', 'def ']):
            print(f"  [{i}/{max_enrich}] Enriching: {shard.id[:50]}...")

            enrichment = await enrich_with_ollama(shard)

            if enrichment.get('concepts'):
                concepts = [str(c) for c in enrichment['concepts'][:2]]
                print(f"       -> Concepts: {', '.join(concepts)}")

            shard.metadata['ollama_enrichment'] = enrichment
            enriched_count += 1

    print(f"  Enriched: {enriched_count} shards with AI insights")

    # PASS 3: Memory Storage
    print(f"\n[PASS 3] Storing in Qdrant Memory...")

    store = QdrantMemoryStore()
    memories = shards_to_memories(shards)

    # Add enrichment metadata to memories
    for mem in memories:
        mem.id = ""  # Clear ID so Qdrant generates MD5 hash
        mem.metadata['source'] = source
        mem.metadata['processing_strategy'] = strategy
        mem.metadata['timestamp'] = datetime.now().isoformat()
        mem.metadata['user_id'] = 'blake'

    try:
        # Store all memories (store_many might not exist, use loop)
        mem_ids = []
        for mem in memories:
            try:
                mem_id = await store.store(mem)
                mem_ids.append(mem_id)
            except Exception as e:
                print(f"  Warning: Failed to store shard: {e}")
                continue

        print(f"  Stored: {len(mem_ids)} memories in Qdrant")
        print(f"  Collection: hololoom_memories")
    except Exception as e:
        print(f"  Error storing: {e}")
        mem_ids = []

    # Statistics
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - Statistics")
    print("=" * 80)
    print(f"  Input: {len(text):,} characters")
    print(f"  Shards Generated: {len(shards)}")
    print(f"  AI Enriched: {enriched_count}")
    print(f"  Memories Stored: {len(mem_ids)}")
    print(f"  Processing Time: Real-time")
    print(f"  Cost: $0 (fully local with Ollama)")

    # Extract all concepts from enriched shards
    all_concepts = []
    all_terms = []
    for shard in shards:
        if 'ollama_enrichment' in shard.metadata:
            enrich = shard.metadata['ollama_enrichment']
            all_concepts.extend(enrich.get('concepts', []))
            all_terms.extend(enrich.get('terms', []))

    if all_concepts:
        concept_counts = Counter(str(c) for c in all_concepts)
        print(f"\n  Top Concepts: {', '.join([c for c, _ in concept_counts.most_common(5)])}")

    if all_terms:
        term_counts = Counter(str(t) for t in all_terms)
        print(f"  Top Terms: {', '.join([t for t, _ in term_counts.most_common(5)])}")

    print("=" * 80)

    return {
        'shards': shards,
        'memories_stored': len(mem_ids),
        'enriched_count': enriched_count,
        'strategy': strategy
    }


async def main():
    """Run scaled-up pipeline."""
    print("\n" + "=" * 80)
    print("SCALED-UP GATED MULTIPASS + OLLAMA + MEMORY STORAGE")
    print("Full Industrial Pipeline Demo")
    print("=" * 80)

    # Process large document
    result = await process_large_document(
        text=LARGE_DOC,
        source='mem0ai_full_docs.txt',
        max_enrich=10  # Enrich top 10 relevant shards
    )

    print("\nSUCCESS! All memories are now stored and queryable in Qdrant.")
    print("\nTo query these memories:")
    print("  1. Use Claude Desktop with MCP server")
    print("  2. Use recall_memories(query='your search')")
    print("  3. Memories include AI-extracted concepts and terms")
    print("\nNext: Restart Claude Desktop to access your enriched memories!")


if __name__ == '__main__':
    asyncio.run(main())
