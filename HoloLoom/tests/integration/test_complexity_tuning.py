"""Comprehensive Complexity Threshold Tuning Test (Task 1.2)"""
import asyncio
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.protocols import ComplexityLevel

async def main():
    print("=" * 80)
    print("COMPREHENSIVE COMPLEXITY THRESHOLD TUNING TEST")
    print("=" * 80)
    
    # Minimal setup
    shards = [MemoryShard(id='1', text='Test shard')]
    cfg = Config.fast()
    shuttle = WeavingShuttle(cfg=cfg, shards=shards, enable_reflection=False)
    
    # Test cases organized by expected complexity
    test_cases = [
        # ===================================================================
        # LITE: Greetings, simple commands (1-3 words, no questions)
        # ===================================================================
        ("Hi", ComplexityLevel.LITE, "single-word greeting"),
        ("Hello", ComplexityLevel.LITE, "single-word greeting"),
        ("Thanks", ComplexityLevel.LITE, "single-word thanks"),
        ("Show me", ComplexityLevel.LITE, "simple 2-word command"),
        ("Get data", ComplexityLevel.LITE, "simple 2-word command"),
        ("Hello there", ComplexityLevel.LITE, "2-word greeting"),
        ("Thank you", ComplexityLevel.LITE, "2-word thanks"),
        ("Hey friend", ComplexityLevel.LITE, "2-word greeting"),
        
        # ===================================================================
        # FAST: Questions, knowledge queries (question words OR 4-20 words)
        # ===================================================================
        ("What is X?", ComplexityLevel.FAST, "what question (3 words)"),
        ("How does it work?", ComplexityLevel.FAST, "how question (4 words)"),
        ("Why is this important?", ComplexityLevel.FAST, "why question (4 words)"),
        ("When was it created?", ComplexityLevel.FAST, "when question (4 words)"),
        ("Where can I find it?", ComplexityLevel.FAST, "where question (5 words)"),
        ("Who invented Thompson Sampling?", ComplexityLevel.FAST, "who question (4 words)"),
        ("Which algorithm is better?", ComplexityLevel.FAST, "which question (4 words)"),
        
        ("What is Thompson Sampling?", ComplexityLevel.FAST, "knowledge question (4 words)"),
        ("Explain neural networks", ComplexityLevel.FAST, "explain verb (3 words)"),
        ("Describe the process", ComplexityLevel.FAST, "describe verb (3 words)"),
        ("Tell me about AI", ComplexityLevel.FAST, "tell verb (4 words)"),
        ("Define machine learning", ComplexityLevel.FAST, "define verb (3 words)"),
        
        ("This is a longer question without analysis verbs or keywords", ComplexityLevel.FAST, "9 words, no special keywords"),
        ("Can you help me understand the basics?", ComplexityLevel.FAST, "7-word question"),
        ("I need to know how this works", ComplexityLevel.FAST, "7-word statement"),
        
        # ===================================================================
        # FULL: Detailed queries (21-50 words, complex questions)
        # ===================================================================
        ("What are the key differences between supervised and unsupervised learning, and when would you use each approach in a real-world scenario?", 
         ComplexityLevel.FULL, "complex question (23 words)"),
        
        ("Can you explain the underlying mathematics of gradient descent optimization and how it relates to backpropagation in neural network training?",
         ComplexityLevel.FULL, "detailed technical question (21 words)"),
        
        ("I'm trying to understand the trade-offs between different database architectures and need guidance on choosing between SQL and NoSQL solutions for my use case",
         ComplexityLevel.FULL, "long explanatory request (28 words)"),
        
        # ===================================================================
        # RESEARCH: Analysis verbs, research keywords, or 50+ words
        # ===================================================================
        ("Analyze the performance", ComplexityLevel.RESEARCH, "analyze verb (3 words)"),
        ("Compare these algorithms", ComplexityLevel.RESEARCH, "compare verb (3 words)"),
        ("Evaluate the approach", ComplexityLevel.RESEARCH, "evaluate verb (3 words)"),
        ("Investigate the issue", ComplexityLevel.RESEARCH, "investigate verb (3 words)"),
        ("Examine the data thoroughly", ComplexityLevel.RESEARCH, "examine verb (4 words)"),
        
        ("Provide a comprehensive analysis", ComplexityLevel.RESEARCH, "comprehensive keyword (4 words)"),
        ("Give me detailed information", ComplexityLevel.RESEARCH, "detailed keyword (4 words)"),
        ("Need thorough research on this", ComplexityLevel.RESEARCH, "thorough + research keywords (5 words)"),
        ("An in-depth exploration is required", ComplexityLevel.RESEARCH, "in-depth keyword (5 words)"),
        
        ("Analyze and compare Thompson Sampling versus UCB algorithms", ComplexityLevel.RESEARCH, "analyze + compare verbs (8 words)"),
        ("Conduct comprehensive research on AGI capabilities", ComplexityLevel.RESEARCH, "comprehensive + research keywords (6 words)"),
        
        # Edge case: Very long query (50+ words)
        ("This is an extremely long query that contains more than fifty words and should automatically be classified as research-level complexity regardless of whether it contains any special keywords or analysis verbs because the length alone indicates that this is a complex information request that requires substantial processing and detailed analysis to provide an adequate response",
         ComplexityLevel.RESEARCH, "50+ words (very long)"),
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    print("\n" + "=" * 80)
    print("TEST RESULTS BY COMPLEXITY LEVEL")
    print("=" * 80)
    
    # Group by expected complexity
    for expected_level in [ComplexityLevel.LITE, ComplexityLevel.FAST, ComplexityLevel.FULL, ComplexityLevel.RESEARCH]:
        level_cases = [(q, exp, desc) for q, exp, desc in test_cases if exp == expected_level]
        
        if not level_cases:
            continue
            
        print(f"\n{expected_level.name} ({expected_level.value} steps) - {len(level_cases)} tests")
        print("-" * 80)
        
        for query_text, expected, description in level_cases:
            query = Query(text=query_text)
            result = shuttle._assess_complexity_level(query)
            
            if result == expected:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1
            
            # Format output
            query_display = query_text[:50] + "..." if len(query_text) > 50 else query_text
            print(f"  {status}: '{query_display}'")
            print(f"        Expected: {expected.name}, Got: {result.name} ({description})")
            
            if status == "FAIL":
                print(f"        >>> MISMATCH <<<")
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 95:
        print("\n*** EXCELLENT: Threshold tuning is highly accurate! ***")
    elif accuracy >= 85:
        print("\n*** GOOD: Minor tuning may improve accuracy ***")
    elif accuracy >= 75:
        print("\n*** FAIR: Threshold adjustments recommended ***")
    else:
        print("\n*** NEEDS WORK: Significant threshold tuning required ***")
    
    print("=" * 80)
    
    return accuracy >= 90  # Success if 90%+ accuracy

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
