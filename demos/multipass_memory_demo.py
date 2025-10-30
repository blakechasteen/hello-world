#!/usr/bin/env python3
"""
Recursive Gated Multipass Memory Crawling Demo
==============================================
Demonstrates the advanced memory crawling system integrated into the Shuttle.

Features demonstrated:
1. Gated Retrieval - Initial broad exploration → focused expansion
2. Matryoshka Importance Gating - Increasing thresholds by depth
3. Graph Traversal - Following entity relationships
4. Multipass Fusion - Intelligent result combination and ranking

This shows how the Shuttle's internal intelligence handles complex memory exploration.
"""

import asyncio
import sys
import os

# Add the dev directory to path to import our enhanced protocol module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dev'))

from protocol_modules_mythrl import (
    MythRLShuttle, DemoMemoryBackend, DemoPatternSelection, 
    DemoDecisionEngine, DemoToolExecution, ComplexityLevel
)


async def demo_multipass_memory_crawling():
    """Demonstrate the recursive gated multipass memory crawling system."""
    
    print("🕷️ RECURSIVE GATED MULTIPASS MEMORY CRAWLING DEMO")
    print("=" * 70)
    print("Shuttle Internal Intelligence Features:")
    print("• Gated Retrieval: Broad exploration → Focused expansion")
    print("• Matryoshka Importance Gating: 0.6 → 0.75 → 0.85 thresholds")
    print("• Graph Traversal: Follow entity relationships")
    print("• Multipass Fusion: Intelligent result combination")
    print()
    
    # Create enhanced memory backend with rich knowledge graph
    memory_backend = DemoMemoryBackend()
    
    # Add some additional interconnected knowledge for better demonstration
    additional_knowledge = {
        'pollination_1': {
            'id': 'pollination_1',
            'content': 'Bee pollination efficiency in agricultural systems',
            'relevance': 0.88,
            'related': ['bee_1', 'crop_1', 'ecosystem_1']
        },
        'crop_1': {
            'id': 'crop_1',
            'content': 'Crop yield dependency on bee pollination',
            'relevance': 0.82,
            'related': ['pollination_1', 'agriculture_1']
        },
        'ecosystem_1': {
            'id': 'ecosystem_1',
            'content': 'Ecosystem services provided by bee populations',
            'relevance': 0.79,
            'related': ['pollination_1', 'biodiversity_1']
        },
        'agriculture_1': {
            'id': 'agriculture_1',
            'content': 'Sustainable agriculture practices for bee conservation',
            'relevance': 0.76,
            'related': ['crop_1', 'pesticide_1']
        },
        'biodiversity_1': {
            'id': 'biodiversity_1',
            'content': 'Biodiversity impact on bee colony resilience',
            'relevance': 0.73,
            'related': ['ecosystem_1']
        },
        'pesticide_1': {
            'id': 'pesticide_1',
            'content': 'Pesticide effects on bee colony health',
            'relevance': 0.70,
            'related': ['agriculture_1', 'disease_1']
        }
    }
    
    # Add to memory backend
    for item_id, item_data in additional_knowledge.items():
        memory_backend.knowledge_graph[item_id] = item_data
    
    print(f"📚 Enhanced Knowledge Graph: {len(memory_backend.knowledge_graph)} interconnected items")
    print()
    
    # Create Shuttle and register protocols
    shuttle = MythRLShuttle()
    shuttle.register_protocol('memory_backend', memory_backend)
    shuttle.register_protocol('pattern_selection', DemoPatternSelection())
    shuttle.register_protocol('decision_engine', DemoDecisionEngine())
    shuttle.register_protocol('tool_execution', DemoToolExecution())
    
    # Test queries with increasing complexity to show multipass crawling scaling
    test_queries = [
        {
            'query': 'bee health monitoring',
            'expected_complexity': ComplexityLevel.FAST,
            'description': 'Simple search - should use 2-pass crawling'
        },
        {
            'query': 'analyze complex relationship between bee colony health and agricultural sustainability',
            'expected_complexity': ComplexityLevel.RESEARCH,
            'description': 'Complex analysis - should use 4-pass deep crawling with graph traversal'
        },
        {
            'query': 'research innovative multi-hop pollination ecosystem optimization strategies',
            'expected_complexity': ComplexityLevel.RESEARCH,
            'description': 'Research query - maximum crawling depth with multipass fusion'
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"MULTIPASS CRAWLING TEST {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print('='*70)
        
        result = await shuttle.weave(test_case['query'])
        
        print(f"\n🧠 SHUTTLE INTELLIGENCE ANALYSIS:")
        print(f"  Assessed Complexity: {result.complexity_level.name} (expected: {test_case['expected_complexity'].name})")
        print(f"  Final Confidence: {result.confidence:.3f}")
        
        print(f"\n🕷️ MULTIPASS CRAWLING RESULTS:")
        if 'crawl_stats' in result.provenance.shuttle_events[-1].get('data', {}):
            crawl_stats = result.provenance.shuttle_events[-1]['data']
            print(f"  Crawl Passes: {crawl_stats.get('passes', 0)}")
            print(f"  Total Items Retrieved: {crawl_stats.get('total_items', 0)}")
            print(f"  Depth Distribution: {crawl_stats.get('depth_stats', {})}")
            print(f"  Fusion Events: {crawl_stats.get('fusion_events', 0)}")
        
        # Show memory results if available
        if hasattr(result, 'memory_results') or 'memory_results' in str(result.output):
            memory_count = str(result.output).count('item') if 'item' in str(result.output) else 0
            print(f"  Memory Items Found: {memory_count}")
        
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        perf = result.get_performance_summary()
        print(f"  Total Duration: {perf['total_duration_ms']:.1f}ms")
        print(f"  Protocol Calls: {perf['protocol_calls']}")
        print(f"  Modules Activated: {result.provenance.modules_invoked}")
        
        print(f"\n🌐 GRAPH TRAVERSAL TRACE:")
        memory_calls = [call for call in result.provenance.protocol_calls if call['protocol'] == 'memory_backend']
        for call in memory_calls:
            print(f"  {call['method']}: {call['result_summary']} ({call['duration_ms']:.1f}ms)")
        
        print(f"\n🔄 SYNTHESIS & TEMPORAL CONTEXT:")
        print(f"  Synthesis Events: {len(result.provenance.synthesis_chain)}")
        print(f"  Temporal Contexts: {len(result.provenance.temporal_contexts)}")
        if result.provenance.temporal_contexts:
            for ctx in result.provenance.temporal_contexts:
                print(f"    • {ctx['type']}: {ctx['duration']} ({ctx['bias']})")
    
    # Demonstrate the crawling configuration differences
    print(f"\n{'='*70}")
    print("🎯 MULTIPASS CRAWLING CONFIGURATION BY COMPLEXITY")
    print('='*70)
    
    shuttle_instance = MythRLShuttle()
    for complexity in ComplexityLevel:
        config = shuttle_instance._get_crawl_config(complexity)
        print(f"\n{complexity.name} ({complexity.value} steps):")
        print(f"  Max Depth: {config['max_depth']}")
        print(f"  Thresholds: {config['thresholds']}")
        print(f"  Initial Limit: {config['initial_limit']}")
        print(f"  Max Total Items: {config['max_total_items']}")
        print(f"  Importance Threshold: {config['importance_threshold']}")


if __name__ == "__main__":
    asyncio.run(demo_multipass_memory_crawling())