#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multipass TextSpinner Demo
==========================
Demonstrates multiple processing passes on the same text with different strategies.
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


async def pass1_coarse_chunking():
    """Pass 1: Coarse paragraph chunking (500 chars)"""
    print("\n" + "=" * 80)
    print("PASS 1: COARSE CHUNKING (500 char paragraphs)")
    print("=" * 80)

    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=500,
        extract_entities=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': MEM0_TEXT,
        'source': 'mem0ai_pypi.txt',
        'metadata': {'pass': 1, 'strategy': 'coarse_paragraph'}
    })

    print(f"\nGenerated {len(shards)} shards in Pass 1")

    # Extract all entities
    all_entities = []
    for idx, shard in enumerate(shards):
        all_entities.extend(shard.entities)
        print(f"\nShard {idx + 1}/{len(shards)}")
        print(f"  ID: {shard.id}")
        print(f"  Length: {len(shard.text)} chars")
        print(f"  Entities: {shard.entities[:5]}{'...' if len(shard.entities) > 5 else ''}")
        print(f"  Preview: {shard.text[:100]}...")

    entity_freq = Counter(all_entities)
    print(f"\n  Top entities in Pass 1: {entity_freq.most_common(10)}")

    return shards, entity_freq


async def pass2_fine_chunking():
    """Pass 2: Fine-grained sentence chunking (200 chars)"""
    print("\n" + "=" * 80)
    print("PASS 2: FINE CHUNKING (200 char sentences)")
    print("=" * 80)

    config = TextSpinnerConfig(
        chunk_by='sentence',
        chunk_size=200,
        min_chunk_size=50,
        extract_entities=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': MEM0_TEXT,
        'source': 'mem0ai_pypi.txt',
        'metadata': {'pass': 2, 'strategy': 'fine_sentence'}
    })

    print(f"\nGenerated {len(shards)} shards in Pass 2")

    # Extract all entities
    all_entities = []
    for idx, shard in enumerate(shards[:10]):  # Show first 10
        all_entities.extend(shard.entities)
        print(f"\nShard {idx + 1}/{len(shards)}")
        print(f"  Entities: {shard.entities}")
        print(f"  Text: {shard.text[:120]}...")

    if len(shards) > 10:
        print(f"\n  ... and {len(shards) - 10} more shards")

    # Count all entities
    for shard in shards[10:]:
        all_entities.extend(shard.entities)

    entity_freq = Counter(all_entities)
    print(f"\n  Top entities in Pass 2: {entity_freq.most_common(10)}")

    return shards, entity_freq


async def pass3_single_shard():
    """Pass 3: Process as single shard (baseline)"""
    print("\n" + "=" * 80)
    print("PASS 3: SINGLE SHARD (no chunking)")
    print("=" * 80)

    shards = await spin_text(
        text=MEM0_TEXT,
        source='mem0ai_pypi.txt',
        chunk_by=None  # No chunking
    )

    print(f"\nGenerated {len(shards)} shard in Pass 3")

    shard = shards[0]
    print(f"\nShard: {shard.id}")
    print(f"  Total length: {len(shard.text)} chars")
    print(f"  Total entities: {len(shard.entities)}")
    print(f"  Entities: {shard.entities[:15]}...")

    entity_freq = Counter(shard.entities)
    print(f"\n  Top entities in Pass 3: {entity_freq.most_common(10)}")

    return shards, entity_freq


async def pass4_analysis(pass1_shards, pass2_shards, pass3_shards,
                         ent1, ent2, ent3):
    """Pass 4: Cross-pass analysis"""
    print("\n" + "=" * 80)
    print("PASS 4: MULTI-PASS ANALYSIS")
    print("=" * 80)

    print("\n📊 SHARD COUNT COMPARISON:")
    print(f"  Pass 1 (coarse paragraphs): {len(pass1_shards)} shards")
    print(f"  Pass 2 (fine sentences):    {len(pass2_shards)} shards")
    print(f"  Pass 3 (single doc):        {len(pass3_shards)} shard(s)")

    print("\n📊 ENTITY EXTRACTION COMPARISON:")
    print(f"  Pass 1 unique entities: {len(ent1)}")
    print(f"  Pass 2 unique entities: {len(ent2)}")
    print(f"  Pass 3 unique entities: {len(ent3)}")

    print("\n📊 TOP CONCEPTS ACROSS ALL PASSES:")
    all_entities = ent1 + ent2 + ent3
    combined = Counter(all_entities)
    for entity, count in combined.most_common(15):
        print(f"  {entity:25s} : {count:3d} mentions")

    print("\n📊 CHUNKING STRATEGY INSIGHTS:")
    print(f"  - Coarse chunking: Better for high-level topics ({len(pass1_shards)} chunks)")
    print(f"  - Fine chunking: Better for detailed extraction ({len(pass2_shards)} chunks)")
    print(f"  - Single shard: Best for overview/summary")

    print("\n📊 MEMORY SHARD DISTRIBUTION:")
    avg_len_1 = sum(len(s.text) for s in pass1_shards) / len(pass1_shards)
    avg_len_2 = sum(len(s.text) for s in pass2_shards) / len(pass2_shards)
    print(f"  Pass 1 avg shard size: {avg_len_1:.0f} chars")
    print(f"  Pass 2 avg shard size: {avg_len_2:.0f} chars")

    # Find shards mentioning specific concepts
    print("\n📊 CONCEPT LOCALIZATION:")
    neo4j_shards = [s for s in pass2_shards if 'Neo4j' in s.text or 'neo4j' in s.text]
    graph_shards = [s for s in pass2_shards if 'Graph' in s.entities or 'graph' in s.text.lower()]
    memory_shards = [s for s in pass2_shards if 'Memory' in s.entities]

    print(f"  Shards mentioning Neo4j: {len(neo4j_shards)}")
    print(f"  Shards about graphs: {len(graph_shards)}")
    print(f"  Shards about memory: {len(memory_shards)}")

    if neo4j_shards:
        print(f"\n  Example Neo4j mention:")
        print(f"    '{neo4j_shards[0].text[:150]}...'")


async def main():
    """Run multipass analysis"""
    print("\n" + "=" * 80)
    print("🧵 HOLOLOOM MULTIPASS TEXT SPINNER DEMO")
    print("   Processing: mem0ai PyPI Documentation")
    print("=" * 80)

    # Run all passes
    pass1_shards, ent1 = await pass1_coarse_chunking()
    pass2_shards, ent2 = await pass2_fine_chunking()
    pass3_shards, ent3 = await pass3_single_shard()

    # Cross-pass analysis
    await pass4_analysis(pass1_shards, pass2_shards, pass3_shards,
                         ent1, ent2, ent3)

    print("\n" + "=" * 80)
    print("✅ MULTIPASS ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Different chunking strategies reveal different aspects")
    print("  2. Coarse chunks good for topic modeling")
    print("  3. Fine chunks good for detailed entity extraction")
    print("  4. Single shard good for document-level features")
    print("  5. Multi-pass enables hierarchical memory structures")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())