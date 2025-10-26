#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gated Multipass TextSpinner Demo
=================================
Demonstrates gate-based multipass processing:
1. Overview pass first (quick scan)
2. Gate decision (what concepts are present?)
3. Conditional deep passes (only if gate triggers)

Philosophy: Don't waste cycles on irrelevant docs.
"""

import asyncio
import sys
import io
from pathlib import Path
from collections import Counter

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
load_module("HoloLoom.spinningWheel.base", repo_root / "HoloLoom" / "spinningWheel" / "base.py")
text_module = load_module("HoloLoom.spinningWheel.text", repo_root / "HoloLoom" / "spinningWheel" / "text.py")

TextSpinner = text_module.TextSpinner
TextSpinnerConfig = text_module.TextSpinnerConfig
spin_text = text_module.spin_text


# The mem0ai PyPI page text
MEM0_TEXT = """Skip to main content
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
Help

    Installing packages
    Uploading packages
    User guide
    Project name retention
    FAQs

About PyPI

    PyPI Blog
    Infrastructure dashboard
    Statistics
    Logos & trademarks
    Our sponsors

Contributing to PyPI

    Bugs and feedback
    Contribute on GitHub
    Translate PyPI
    Sponsor PyPI
    Development credits

Using PyPI

    Terms of Service
    Report security issue
    Code of conduct
    Privacy Notice
    Acceptable Use Policy

Status: All Systems Operational

Developed and maintained by the Python community, for the Python community.
Donate today!

"PyPI", "Python Package Index", and the blocks logos are registered trademarks of the Python Software Foundation.

© 2025 Python Software Foundation
Site map
"""


class ProcessingGate:
    """
    Determines processing strategy based on overview scan.

    The gate decides:
    - Should we process this at all?
    - What chunking strategy to use?
    - Which downstream passes to trigger?
    """

    def __init__(self):
        self.triggers = {
            'technical_doc': ['API', 'Python', 'install', 'documentation', 'code'],
            'graph_relevant': ['Neo4j', 'Graph', 'graph', 'knowledge', 'relationship'],
            'memory_system': ['Memory', 'memory', 'remember', 'recall', 'store'],
            'ai_agent': ['AI', 'agent', 'LLM', 'assistant', 'chatbot'],
        }

    def evaluate(self, overview_shard):
        """
        Evaluate overview shard and determine processing gates.

        Returns:
            dict with gate decisions
        """
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

        # Check if document is worth processing
        if shard.metadata.get('char_count', 0) < 100:
            return 'SKIP'  # Too small

        # Determine chunking strategy based on gates
        if gates['graph_relevant'] and gates['technical_doc']:
            return 'FINE_TECHNICAL'  # Small chunks for graph extraction
        elif gates['memory_system'] or gates['ai_agent']:
            return 'MEDIUM_SEMANTIC'  # Medium chunks for semantic processing
        elif gates['technical_doc']:
            return 'COARSE_OVERVIEW'  # Large chunks for documentation
        else:
            return 'SINGLE_SHARD'  # Keep as-is


async def pass0_overview(text, source='document'):
    """Pass 0: Quick overview scan to determine gates."""
    print("\n" + "=" * 80)
    print("PASS 0: OVERVIEW SCAN (Gate Determination)")
    print("=" * 80)

    # Single shard overview
    shards = await spin_text(
        text=text,
        source=source,
        chunk_by=None  # No chunking for overview
    )

    overview = shards[0]

    print(f"\nOverview Shard Generated:")
    print(f"  ID: {overview.id}")
    print(f"  Length: {len(overview.text)} chars")
    print(f"  Entities extracted: {len(overview.entities)}")
    print(f"  Top entities: {overview.entities[:10]}")

    # Evaluate gates
    gate = ProcessingGate()
    decision = gate.evaluate(overview)

    print(f"\n🚪 GATE EVALUATION:")
    print(f"  Technical Doc:    {'✓' if decision['gates']['technical_doc'] else '✗'}  (score: {decision['gates']['technical_doc_score']})")
    print(f"  Graph Relevant:   {'✓' if decision['gates']['graph_relevant'] else '✗'}  (score: {decision['gates']['graph_relevant_score']})")
    print(f"  Memory System:    {'✓' if decision['gates']['memory_system'] else '✗'}  (score: {decision['gates']['memory_system_score']})")
    print(f"  AI Agent Topic:   {'✓' if decision['gates']['ai_agent'] else '✗'}  (score: {decision['gates']['ai_agent_score']})")

    print(f"\n📊 GATE DECISION:")
    print(f"  Strategy: {decision['strategy']}")

    return overview, decision


async def pass1_conditional(text, source, strategy):
    """Pass 1: Conditional processing based on gate decision."""
    print("\n" + "=" * 80)
    print(f"PASS 1: CONDITIONAL PROCESSING ({strategy})")
    print("=" * 80)

    if strategy == 'SKIP':
        print("  ⊘ Skipping - document too small")
        return []

    elif strategy == 'SINGLE_SHARD':
        print("  → Keeping as single shard")
        shards = await spin_text(text=text, source=source, chunk_by=None)

    elif strategy == 'COARSE_OVERVIEW':
        print("  → Coarse chunking (500 chars)")
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=500, extract_entities=True)
        spinner = TextSpinner(config)
        shards = await spinner.spin({'text': text, 'source': source})

    elif strategy == 'MEDIUM_SEMANTIC':
        print("  → Medium chunking (300 chars)")
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=300, extract_entities=True)
        spinner = TextSpinner(config)
        shards = await spinner.spin({'text': text, 'source': source})

    elif strategy == 'FINE_TECHNICAL':
        print("  → Fine chunking (150 chars)")
        config = TextSpinnerConfig(chunk_by='sentence', chunk_size=150, min_chunk_size=50, extract_entities=True)
        spinner = TextSpinner(config)
        shards = await spinner.spin({'text': text, 'source': source})

    print(f"\n  Generated {len(shards)} shards")

    return shards


async def pass2_targeted_extraction(shards, gates):
    """Pass 2: Targeted extraction based on gate triggers."""
    print("\n" + "=" * 80)
    print("PASS 2: TARGETED EXTRACTION")
    print("=" * 80)

    extractions = {
        'graph_shards': [],
        'api_shards': [],
        'memory_shards': [],
        'code_shards': []
    }

    for shard in shards:
        text_lower = shard.text.lower()

        # Extract graph-related shards
        if gates['graph_relevant'] and ('neo4j' in text_lower or 'graph' in text_lower):
            extractions['graph_shards'].append(shard)

        # Extract API documentation shards
        if 'api' in text_lower or 'import' in text_lower or 'def ' in text_lower:
            extractions['api_shards'].append(shard)

        # Extract memory-related shards
        if gates['memory_system'] and ('memory' in text_lower or 'recall' in text_lower or 'store' in text_lower):
            extractions['memory_shards'].append(shard)

        # Extract code examples
        if 'from ' in shard.text or 'import ' in shard.text or '```' in shard.text:
            extractions['code_shards'].append(shard)

    print(f"\n  Targeted Extractions:")
    print(f"    Graph-related shards:  {len(extractions['graph_shards'])}")
    print(f"    API documentation:     {len(extractions['api_shards'])}")
    print(f"    Memory system shards:  {len(extractions['memory_shards'])}")
    print(f"    Code examples:         {len(extractions['code_shards'])}")

    # Show samples
    if extractions['graph_shards']:
        print(f"\n  Example graph shard:")
        print(f"    {extractions['graph_shards'][0].text[:150]}...")

    if extractions['code_shards']:
        print(f"\n  Example code shard:")
        print(f"    {extractions['code_shards'][0].text[:150]}...")

    return extractions


async def main():
    """Run gated multipass processing."""
    print("\n" + "=" * 80)
    print("🚪 GATED MULTIPASS TEXT SPINNER DEMO")
    print("   Philosophy: Overview → Gate → Conditional Processing")
    print("=" * 80)

    # Pass 0: Overview scan (ALWAYS RUNS FIRST)
    overview_shard, decision = await pass0_overview(MEM0_TEXT, 'mem0ai_pypi.txt')

    # Pass 1: Conditional processing (based on gate)
    strategy = decision['strategy']
    shards = await pass1_conditional(MEM0_TEXT, 'mem0ai_pypi.txt', strategy)

    # Pass 2: Targeted extraction (only if gates triggered)
    if shards and strategy != 'SKIP':
        extractions = await pass2_targeted_extraction(shards, decision['gates'])
    else:
        extractions = {}

    # Summary
    print("\n" + "=" * 80)
    print("✅ GATED PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\n  Overview: 1 shard analyzed")
    print(f"  Strategy: {strategy}")
    print(f"  Processing: {len(shards)} shards generated")
    if extractions:
        total_targeted = sum(len(v) for v in extractions.values())
        print(f"  Targeted extraction: {total_targeted} specialized shards")

    print("\n🎯 Key Benefits:")
    print("  1. Overview pass is cheap - decides if deep processing is worth it")
    print("  2. Gate prevents wasting cycles on irrelevant docs")
    print("  3. Strategy adapts to document type")
    print("  4. Targeted extraction finds specific content types")
    print("  5. Efficient resource usage - only process what matters")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
