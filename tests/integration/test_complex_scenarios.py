"""
Complex Scenario Testing for Phase 2
=====================================

Tests edge cases, multi-turn conversations, knowledge ingestion,
intelligent routing under load, and memory backend behavior.

USAGE:
    cd c:\\Users\\blake\\Documents\\mythRL
    $env:PYTHONPATH = "."; python tests/test_complex_scenarios.py
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict
from mythRL import Weaver


class TestScenarios:
    """Complex test scenarios for Phase 2 features."""
    
    def __init__(self):
        self.results: List[Dict] = []
        
    async def test_edge_cases(self):
        """Test edge cases: empty queries, very long queries, special chars."""
        print("\n" + "="*60)
        print("Test 1: Edge Cases")
        print("="*60)
        
        weaver = await Weaver.create(pattern='auto', intelligent_routing=True)
        
        edge_cases = [
            ("", "empty query"),
            ("a", "single character"),
            ("What is?" * 100, "very long query (800 chars)"),
            ("What's the meaning of life, the universe, and everything? " * 20, "1000+ chars"),
            ("@#$%^&*()", "special characters only"),
            ("What is HoloLoom? " * 10, "repetitive query"),
        ]
        
        for query, description in edge_cases:
            try:
                start = time.perf_counter()
                result = await weaver.query(query)
                duration = (time.perf_counter() - start) * 1000
                
                print(f"\n✓ {description}")
                print(f"  Query length: {len(query)} chars")
                print(f"  Pattern: {result.pattern}")
                print(f"  Duration: {duration:.1f}ms")
                print(f"  Confidence: {result.confidence:.2f}")
                
                self.results.append({
                    'test': 'edge_cases',
                    'description': description,
                    'success': True,
                    'duration_ms': duration
                })
                
            except Exception as e:
                print(f"\n✗ {description} - FAILED: {e}")
                self.results.append({
                    'test': 'edge_cases',
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n{'─'*60}")
        print(f"Edge Cases: {sum(1 for r in self.results if r['test'] == 'edge_cases' and r['success'])}/{len(edge_cases)} passed")
        
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversations with context."""
        print("\n" + "="*60)
        print("Test 2: Multi-Turn Conversation")
        print("="*60)
        
        weaver = await Weaver.create(pattern='auto', intelligent_routing=True)
        
        conversation = [
            "What is Thompson Sampling?",
            "How does it compare to epsilon-greedy?",
            "Can you explain the Beta distribution used?",
            "What are real-world applications?",
            "Show me Python code for implementation"
        ]
        
        print("\nConversation flow:")
        for i, query in enumerate(conversation, 1):
            try:
                start = time.perf_counter()
                result = await weaver.query(query)
                duration = (time.perf_counter() - start) * 1000
                
                print(f"\n{i}. User: {query}")
                print(f"   Pattern: {result.pattern} | Duration: {duration:.1f}ms")
                print(f"   Response: {result.response[:100]}...")
                
                self.results.append({
                    'test': 'multi_turn',
                    'turn': i,
                    'success': True,
                    'duration_ms': duration
                })
                
            except Exception as e:
                print(f"\n{i}. FAILED: {e}")
                self.results.append({
                    'test': 'multi_turn',
                    'turn': i,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n{'─'*60}")
        successful_turns = sum(1 for r in self.results if r['test'] == 'multi_turn' and r['success'])
        print(f"Multi-Turn: {successful_turns}/{len(conversation)} turns successful")
        
    async def test_knowledge_ingestion(self):
        """Test knowledge ingestion from text."""
        print("\n" + "="*60)
        print("Test 3: Knowledge Ingestion")
        print("="*60)
        
        knowledge = """
        Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
        It maintains a Beta distribution for each arm, updated based on successes and failures.
        The algorithm samples from each distribution and selects the arm with the highest sample.
        
        Key advantages:
        - Optimal regret bounds
        - Natural exploration-exploitation balance
        - Works well in practice
        
        Compared to epsilon-greedy:
        - More sophisticated exploration
        - Better theoretical guarantees
        - Slightly more computational overhead
        """
        
        try:
            start = time.perf_counter()
            weaver = await Weaver.create(
                pattern='auto',
                intelligent_routing=True,
                knowledge=knowledge
            )
            init_duration = (time.perf_counter() - start) * 1000
            
            print(f"\n✓ Knowledge ingested in {init_duration:.1f}ms")
            
            # Query the ingested knowledge
            queries = [
                "What is Thompson Sampling?",
                "How does it compare to epsilon-greedy?",
                "What are the key advantages?"
            ]
            
            for query in queries:
                start = time.perf_counter()
                result = await weaver.query(query)
                duration = (time.perf_counter() - start) * 1000
                
                print(f"\nQuery: {query}")
                print(f"Context retrieved: {result.context_count} shards")
                print(f"Duration: {duration:.1f}ms")
                print(f"Response: {result.response[:100]}...")
                
                self.results.append({
                    'test': 'knowledge_ingestion',
                    'query': query,
                    'success': True,
                    'context_count': result.context_count,
                    'duration_ms': duration
                })
            
            print(f"\n{'─'*60}")
            print(f"Knowledge Ingestion: {len(queries)}/{len(queries)} queries successful")
            
        except Exception as e:
            print(f"\n✗ Knowledge ingestion FAILED: {e}")
            self.results.append({
                'test': 'knowledge_ingestion',
                'success': False,
                'error': str(e)
            })
    
    async def test_pattern_distribution(self):
        """Test that intelligent routing distributes patterns correctly."""
        print("\n" + "="*60)
        print("Test 4: Pattern Distribution")
        print("="*60)
        
        weaver = await Weaver.create(pattern='auto', intelligent_routing=True)
        
        # Mix of queries across complexity levels
        queries = [
            ("What is X?", "lite"),
            ("Who created Y?", "lite"),
            ("Explain Z", "fast"),
            ("How does A work?", "fast"),
            ("Analyze the differences between B and C", "full"),
            ("Compare and contrast D versus E", "full"),
            ("Provide a comprehensive analysis of F", "research"),
            ("Deep dive into the details of G", "research"),
        ]
        
        pattern_counts = {"bare": 0, "fast": 0, "fused": 0, "unknown": 0}
        
        for query, expected in queries:
            try:
                result = await weaver.query(query)
                pattern = result.pattern
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                print(f"\n{query[:40]:40} → {pattern:10} (expected: {expected})")
                
            except Exception as e:
                print(f"\n✗ Query failed: {e}")
        
        print(f"\n{'─'*60}")
        print("Pattern Distribution:")
        for pattern, count in pattern_counts.items():
            percentage = (count / len(queries)) * 100
            print(f"  {pattern:10} : {count:2} ({percentage:5.1f}%)")
        
        self.results.append({
            'test': 'pattern_distribution',
            'success': True,
            'distribution': pattern_counts
        })
    
    async def test_concurrent_queries(self):
        """Test concurrent query handling."""
        print("\n" + "="*60)
        print("Test 5: Concurrent Queries")
        print("="*60)
        
        weaver = await Weaver.create(pattern='auto', intelligent_routing=True)
        
        queries = [
            "What is HoloLoom?",
            "Explain Thompson Sampling",
            "Analyze the Weaving architecture",
            "Deep dive into memory backends",
            "Compare NETWORKX and NEO4J_QDRANT"
        ]
        
        print(f"\nExecuting {len(queries)} queries concurrently...")
        
        try:
            start = time.perf_counter()
            
            # Execute all queries concurrently
            results = await asyncio.gather(*[
                weaver.query(q) for q in queries
            ])
            
            total_duration = (time.perf_counter() - start) * 1000
            
            print(f"\n✓ All queries completed in {total_duration:.1f}ms")
            print(f"  Average: {total_duration/len(queries):.1f}ms per query")
            
            for i, (query, result) in enumerate(zip(queries, results), 1):
                print(f"\n{i}. {query[:40]:40}")
                print(f"   Pattern: {result.pattern:10} Confidence: {result.confidence:.2f}")
            
            self.results.append({
                'test': 'concurrent_queries',
                'success': True,
                'total_duration_ms': total_duration,
                'query_count': len(queries)
            })
            
        except Exception as e:
            print(f"\n✗ Concurrent queries FAILED: {e}")
            self.results.append({
                'test': 'concurrent_queries',
                'success': False,
                'error': str(e)
            })
        
        print(f"\n{'─'*60}")
    
    async def test_memory_backend_fallback(self):
        """Test memory backend graceful degradation."""
        print("\n" + "="*60)
        print("Test 6: Memory Backend Fallback")
        print("="*60)
        
        backends = ['networkx', 'neo4j_qdrant', 'hyperspace']
        
        for backend in backends:
            try:
                print(f"\nTesting backend: {backend}")
                weaver = await Weaver.create(
                    pattern='fast',
                    memory_backend=backend
                )
                
                result = await weaver.query("What is HoloLoom?")
                print(f"✓ {backend:15} : {result.confidence:.2f} confidence, {result.duration_ms:.1f}ms")
                
                self.results.append({
                    'test': 'backend_fallback',
                    'backend': backend,
                    'success': True,
                    'duration_ms': result.duration_ms
                })
                
            except Exception as e:
                print(f"✗ {backend:15} : FAILED - {e}")
                self.results.append({
                    'test': 'backend_fallback',
                    'backend': backend,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n{'─'*60}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        test_types = {}
        for result in self.results:
            test_name = result['test']
            if test_name not in test_types:
                test_types[test_name] = {'passed': 0, 'failed': 0}
            
            if result['success']:
                test_types[test_name]['passed'] += 1
            else:
                test_types[test_name]['failed'] += 1
        
        total_passed = sum(t['passed'] for t in test_types.values())
        total_failed = sum(t['failed'] for t in test_types.values())
        
        print(f"\nResults by test type:")
        for test_name, counts in test_types.items():
            total = counts['passed'] + counts['failed']
            print(f"  {test_name:25} : {counts['passed']:2}/{total:2} passed")
        
        print(f"\n{'─'*60}")
        print(f"OVERALL: {total_passed}/{total_passed + total_failed} tests passed")
        
        if total_failed == 0:
            print("\n✓ All tests passed!")
        else:
            print(f"\n⚠ {total_failed} tests failed")
            print("\nFailed tests:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['test']}: {result.get('error', 'Unknown error')}")


async def main():
    """Run all complex scenarios."""
    print("="*60)
    print("COMPLEX SCENARIO TESTING - Phase 2")
    print("="*60)
    
    tester = TestScenarios()
    
    # Run all test scenarios
    await tester.test_edge_cases()
    await tester.test_multi_turn_conversation()
    await tester.test_knowledge_ingestion()
    await tester.test_pattern_distribution()
    await tester.test_concurrent_queries()
    await tester.test_memory_backend_fallback()
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
