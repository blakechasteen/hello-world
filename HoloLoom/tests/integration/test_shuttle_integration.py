"""Test Shuttle-HoloLoom Integration (Task 1.2)"""
import asyncio
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.protocols import ComplexityLevel

async def main():
    print("=" * 60)
    print("Testing Enhanced WeavingShuttle with mythRL Integration")
    print("=" * 60)
    
    # Test data
    shards = [
        MemoryShard(id='1', text='HoloLoom uses Thompson Sampling for tool selection.'),
        MemoryShard(id='2', text='The system has 9-step weaving cycle.')
    ]
    
    cfg = Config.fast()
    shuttle = WeavingShuttle(cfg=cfg, shards=shards, enable_reflection=False)
    
    print("\n[TEST 1] Complexity Assessment")
    print("-" * 60)
    queries = [
        # LITE: Greetings and simple commands (1-3 words, no questions)
        ("Hi", ComplexityLevel.LITE, "greeting"),
        ("Hello there", ComplexityLevel.LITE, "multi-word greeting"),
        ("Show me", ComplexityLevel.LITE, "simple command"),
        
        # FAST: Questions and knowledge queries (question words or 4-20 words)
        ("What is Thompson Sampling?", ComplexityLevel.FAST, "question with 'what'"),
        ("How does it work?", ComplexityLevel.FAST, "question with 'how'"),
        ("Explain neural networks", ComplexityLevel.FAST, "knowledge verb"),
        ("Tell me about Bayesian inference", ComplexityLevel.FAST, "5 words with knowledge verb"),
        
        # FULL: Detailed queries (21-50 words)
        ("Can you provide a detailed explanation of how Thompson Sampling works in the context of multi-armed bandit problems and reinforcement learning?", ComplexityLevel.FULL, "long query"),
        
        # RESEARCH: Analysis verbs or research keywords
        ("Analyze and compare Thompson Sampling versus UCB", ComplexityLevel.RESEARCH, "analysis verb 'analyze'"),
        ("Compare different approaches", ComplexityLevel.RESEARCH, "analysis verb 'compare'"),
        ("Evaluate the performance", ComplexityLevel.RESEARCH, "analysis verb 'evaluate'"),
        ("Provide comprehensive research", ComplexityLevel.RESEARCH, "research keyword 'comprehensive'")
    ]
    
    passed = 0
    failed = 0
    for query_text, expected, description in queries:
        q = Query(text=query_text)
        complexity = shuttle._assess_complexity_level(q)
        status = "PASS" if complexity == expected else "FAIL"
        if complexity == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status}: '{description}' -> {complexity.name} (expected {expected.name})")
    
    print(f"\nResults: {passed}/{len(queries)} passed, {failed} failed")
    
    print("\n[TEST 2] Full Weaving with Complexity Parameter")
    print("-" * 60)
    result = await shuttle.weave(Query(text='What is Thompson Sampling?'), complexity=ComplexityLevel.FAST)
    
    # Result is Spacetime object with 'response' field
    print(f"Result length: {len(result.response)}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Duration: {result.trace.duration_ms:.1f}ms")
    print(f"Tool used: {result.tool_used}")
    
    # Check mythRL enhancements on result
    if hasattr(result, 'complexity'):
        print(f"PASS: Complexity tracked: {result.complexity.name}")
    else:
        print("FAIL: Complexity not tracked in result")
    
    if hasattr(result, 'provenance'):
        print(f"PASS: Provenance protocol calls: {len(result.provenance.protocol_calls)}")
        print(f"PASS: Provenance shuttle events: {len(result.provenance.shuttle_events)}")
    else:
        print("FAIL: Provenance not tracked in result")
    
    print("\n[TEST 3] Protocol Registration")
    print("-" * 60)
    
    class TestProtocol:
        async def test_method(self):
            return "test"
    
    shuttle.register_protocol('test_protocol', TestProtocol())
    registered = 'test_protocol' in shuttle._protocols
    print(f"{'PASS' if registered else 'FAIL'}: Protocol registration working")
    
    print("\n" + "=" * 60)
    print("Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
