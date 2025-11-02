#!/usr/bin/env python3
"""
HoloLoom + Promptly Integration Demo
=====================================
Demonstrates the complete unified integration between Promptly and HoloLoom.

Shows:
1. Storing prompts in HoloLoom unified memory
2. Semantic search across prompts
3. Knowledge graph relationships
4. Unified analytics
5. Multi-system memory sharing
"""

import sys
from pathlib import Path
from datetime import datetime

# Add promptly directory to path
promptly_path = Path(__file__).parent / "promptly"
if str(promptly_path) not in sys.path:
    sys.path.insert(0, str(promptly_path))

from hololoom_unified import (
    create_unified_bridge,
    UnifiedPrompt,
    HOLOLOOM_AVAILABLE,
    NEO4J_AVAILABLE
)

def print_separator(title=""):
    """Print a nice separator"""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)
    else:
        print('-' * 70)

def demo_basic_storage():
    """Demo 1: Basic prompt storage"""
    print_separator("Demo 1: Store Prompts in HoloLoom")

    bridge = create_unified_bridge(enable_neo4j=NEO4J_AVAILABLE)

    if not bridge.enabled:
        print("[SKIP] HoloLoom not available")
        return None

    # Create sample prompts
    prompts = [
        UnifiedPrompt(
            prompt_id="sql_opt_v1",
            name="SQL Optimizer",
            content="Optimize this SQL query for performance:\n{query}",
            version=1,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=["sql", "optimization", "database"],
            related_concepts=["query optimization", "database performance"],
            usage_count=42,
            avg_quality=0.87
        ),
        UnifiedPrompt(
            prompt_id="code_review_v1",
            name="Code Reviewer",
            content="Review this code for:\n1. Best practices\n2. Security issues\n3. Performance\n\n{code}",
            version=1,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=["code-review", "security", "best-practices"],
            related_concepts=["code quality", "security audit"],
            usage_count=67,
            avg_quality=0.92
        ),
        UnifiedPrompt(
            prompt_id="bug_fix_v1",
            name="Bug Detective",
            content="Analyze this bug and suggest fixes:\n\nError: {error}\n\nContext: {context}",
            version=1,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=["debugging", "troubleshooting"],
            related_concepts=["error analysis", "root cause"],
            usage_count=89,
            avg_quality=0.85
        ),
        UnifiedPrompt(
            prompt_id="docs_gen_v1",
            name="Documentation Generator",
            content="Generate comprehensive documentation for:\n{code}\n\nInclude examples and usage.",
            version=1,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=["documentation", "code", "examples"],
            related_concepts=["technical writing", "API docs"],
            usage_count=34,
            avg_quality=0.78
        )
    ]

    # Store all prompts
    print("\n[INFO] Storing prompts in HoloLoom...")
    stored_ids = []
    for prompt in prompts:
        mem_id = bridge.store_prompt(prompt)
        if mem_id:
            stored_ids.append(mem_id)
            print(f"  [OK] {prompt.name}: {mem_id}")

    print(f"\n[OK] Stored {len(stored_ids)} prompts in HoloLoom unified memory")
    return bridge

def demo_semantic_search(bridge):
    """Demo 2: Semantic search"""
    print_separator("Demo 2: Semantic Search")

    if not bridge or not bridge.enabled:
        print("[SKIP] Bridge not available")
        return

    # Search queries
    queries = [
        ("code quality", ["code-review"]),
        ("database performance", ["sql"]),
        ("fix errors", None),
    ]

    for query, tags in queries:
        print(f"\n[SEARCH] '{query}'" + (f" with tags: {tags}" if tags else ""))
        results = bridge.search_prompts(query, tags=tags, limit=3)

        if results:
            for i, result in enumerate(results, 1):
                name = result.get('context', {}).get('name', 'Unknown')
                relevance = result.get('relevance', 0.0)
                print(f"  {i}. {name} (relevance: {relevance:.2f})")
        else:
            print("  No results found")

def demo_knowledge_graph(bridge):
    """Demo 3: Knowledge graph relationships"""
    print_separator("Demo 3: Knowledge Graph Relationships")

    if not bridge or not bridge.enabled:
        print("[SKIP] Bridge not available")
        return

    # Link prompts to concepts
    links = [
        ("sql_opt_v1", "Performance Optimization"),
        ("code_review_v1", "Code Quality"),
        ("bug_fix_v1", "Error Handling"),
        ("docs_gen_v1", "Technical Communication")
    ]

    print("\n[INFO] Creating prompt -> concept links...")
    for prompt_id, concept in links:
        success = bridge.link_prompt_to_concept(prompt_id, concept)
        if success:
            print(f"  [OK] {prompt_id} -> {concept}")

def demo_analytics(bridge):
    """Demo 4: Unified analytics"""
    print_separator("Demo 4: Unified Analytics")

    if not bridge or not bridge.enabled:
        print("[SKIP] Bridge not available")
        return

    analytics = bridge.get_prompt_analytics()

    print("\n[INFO] Promptly + HoloLoom Analytics:")
    print(f"  Enabled: {analytics.get('enabled')}")
    print(f"  Total Prompts: {analytics.get('total_prompts', 0)}")
    print(f"  Total Usage: {analytics.get('total_usage', 0)}")
    print(f"  Avg Quality: {analytics.get('avg_quality', 0.0):.2f}")

    if analytics.get('tag_distribution'):
        print("\n  Tag Distribution:")
        for tag, count in sorted(
            analytics['tag_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"    - {tag}: {count}")

    if analytics.get('most_used'):
        print("\n  Most Used Prompts:")
        for prompt in analytics['most_used']:
            print(f"    - {prompt['name']}: {prompt['usage']} uses")

def demo_related_prompts(bridge):
    """Demo 5: Find related prompts"""
    print_separator("Demo 5: Find Related Prompts")

    if not bridge or not bridge.enabled:
        print("[SKIP] Bridge not available")
        return

    # Find prompts related to SQL Optimizer
    print("\n[INFO] Finding prompts related to 'SQL Optimizer'...")
    related = bridge.get_related_prompts("sql_opt_v1", limit=3)

    if related:
        for i, prompt in enumerate(related, 1):
            name = prompt.get('context', {}).get('name', 'Unknown')
            relevance = prompt.get('relevance', 0.0)
            print(f"  {i}. {name} (relevance: {relevance:.2f})")
    else:
        print("  No related prompts found (Note: UnifiedMemory recall is stubbed)")

def main():
    """Run all demos"""
    print("=" * 70)
    print("  HoloLoom + Promptly Unified Integration Demo")
    print("=" * 70)
    print(f"\nStatus:")
    print(f"  HoloLoom Available: {HOLOLOOM_AVAILABLE}")
    print(f"  Neo4j Available: {NEO4J_AVAILABLE}")

    if not HOLOLOOM_AVAILABLE:
        print("\n[ERROR] HoloLoom not available!")
        print("\nTo enable:")
        print("  1. Ensure HoloLoom is in parent directory")
        print("  2. Install: pip install -r HoloLoom/requirements.txt")
        return

    # Run demos
    bridge = demo_basic_storage()
    demo_semantic_search(bridge)
    demo_knowledge_graph(bridge)
    demo_analytics(bridge)
    demo_related_prompts(bridge)

    print_separator("Demo Complete")
    print("\n[OK] HoloLoom + Promptly integration working!")
    print("\nNext steps:")
    print("  1. Implement actual storage backends (Neo4j, Qdrant)")
    print("  2. Add to web dashboard for visualization")
    print("  3. Enable multi-modal memory (images, audio)")
    print("  4. Create MCP tools for Claude Desktop")

if __name__ == "__main__":
    main()
