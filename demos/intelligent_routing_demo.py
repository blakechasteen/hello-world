"""
Demo: Intelligent Routing
Demonstrates auto mode selection based on query complexity.

USAGE:
    cd c:\\Users\\blake\\Documents\\mythRL
    $env:PYTHONPATH = "."; python demos/intelligent_routing_demo.py
"""

import asyncio
from mythRL import Weaver


async def main():
    print("=" * 60)
    print("Intelligent Routing Demo")
    print("=" * 60)
    print()
    
    # Create Weaver with intelligent routing enabled
    weaver = await Weaver.create(
        pattern='auto',
        intelligent_routing=True
    )
    
    # Test queries at different complexity levels
    test_queries = [
        # LITE: Simple factual queries
        ("What is HoloLoom?", "lite"),
        ("Who created mythRL?", "lite"),
        
        # FAST: Standard questions
        ("How does Thompson Sampling work?", "fast"),
        ("Explain the Weaving architecture", "fast"),
        
        # FULL: Complex analysis
        ("Analyze the differences between NETWORKX and NEO4J_QDRANT memory backends", "full"),
        ("Compare and contrast PolicyEngine implementations", "full"),
        
        # RESEARCH: Deep dives
        ("Provide a comprehensive analysis of the recursive gated multipass memory crawling system", "research"),
        ("Deep dive into the 3-5-7-9 progressive complexity system with detailed examples", "research"),
    ]
    
    print(f"Testing {len(test_queries)} queries across complexity levels:\n")
    
    for query, expected_mode in test_queries:
        print(f"Query: {query[:60]}...")
        print(f"Expected mode: {expected_mode}")
        
        # Execute query with intelligent routing
        result = await weaver.query(query)
        
        print(f"Actual pattern: {result.pattern}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Duration: {result.duration_ms:.1f}ms")
        print(f"Context shards: {result.context_count}")
        print(f"Response preview: {result.response[:100]}...")
        print()
        print("-" * 60)
        print()
    
    print("✓ Intelligent routing demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
