"""
Promptly API Integration Example
=================================
Demonstrates how to use Promptly via its dashboard API.

This example:
1. Initializes the Promptly API
2. Executes prompts with quality evaluation
3. Runs A/B tests
4. Gets system status and analytics
"""

import asyncio
from promptly.api import PromptlyAPI, create_api


async def main():
    """Run Promptly API example."""
    print("=" * 70)
    print("Promptly - API Integration Example")
    print("=" * 70)
    print()

    # 1. Initialize API
    print("1. Initializing Promptly API...")
    api = create_api(enable_judge=True, enable_analytics=True)

    async with api:
        print("   ✓ API initialized")
        print(f"   ✓ Judge enabled: {api._enable_judge}")
        print(f"   ✓ Analytics enabled: {api._enable_analytics}")
        print()

        # 2. Execute simple prompt
        print("2. Executing simple prompt...")
        result = await api.execute_prompt(
            prompt="Explain quantum computing in one sentence",
            evaluate_quality=True
        )

        print()
        print(f"   Prompt:   {result.prompt}")
        print(f"   Response: {result.response[:100]}...")
        print(f"   Duration: {result.duration_ms:.1f}ms")

        if result.quality_score:
            print(f"   Quality:  {result.quality_score:.2f}/1.0")

        if result.quality_breakdown:
            print("   Breakdown:")
            for criterion, score in result.quality_breakdown.items():
                print(f"      • {criterion:<15} {score:.2f}")

        print()

        # 3. Run A/B test
        print("3. Running A/B test...")
        test_cases = [
            {"topic": "machine learning"},
            {"topic": "neural networks"},
            {"topic": "deep learning"},
        ]

        ab_result = await api.run_ab_test(
            prompt_a="Explain {topic} clearly",
            prompt_b="Describe {topic} with examples",
            test_cases=test_cases
        )

        print()
        print(f"   Winner:        Prompt {ab_result.winner}")
        print(f"   Confidence:    {ab_result.confidence:.2%}")
        print(f"   Prompt A avg:  {ab_result.prompt_a_avg_score:.3f}")
        print(f"   Prompt B avg:  {ab_result.prompt_b_avg_score:.3f}")
        print(f"   P-value:       {ab_result.p_value if ab_result.p_value else 'N/A'}")
        print()

        # 4. Execute loop composition
        print("4. Executing loop composition...")
        loop_result = await api.execute_loop(
            loop_definition="""
            LOOP chain:
              - analyze_input
              - generate_response
              - refine_output
            """,
            variables={"input": "How do black holes work?"}
        )

        print()
        print(f"   Success:       {loop_result['success']}")
        print(f"   Steps executed: {loop_result['steps_executed']}")
        print()

        # 5. Get system status
        print("5. Checking system status...")
        status = await api.get_status()

        print()
        print(f"   Status:          {status.status}")
        print(f"   Uptime:          {status.uptime_seconds:.1f}s")
        print(f"   Total executions: {status.total_executions}")
        print(f"   Success rate:    {status.success_rate:.1%}")
        print(f"   Avg time:        {status.avg_execution_time_ms:.1f}ms")
        print()

        print("   Components:")
        print(f"      • Promptly:   {'✓' if status.promptly_ready else '✗'}")
        print(f"      • Judge:      {'✓' if status.judge_ready else '✗'}")
        print(f"      • Analytics:  {'✓' if status.analytics_ready else '✗'}")
        print()

        # 6. Get analytics
        print("6. Retrieving analytics...")
        analytics = await api.get_analytics()

        print()
        if "error" not in analytics:
            print("   Analytics available:")
            print(f"      • Recent prompts: {len(analytics.get('recent_prompts', []))}")
            print(f"      • Quality trends: {len(analytics.get('quality_trends', {}))}")
            print(f"      • Cost breakdown: {len(analytics.get('cost_breakdown', {}))}")
        else:
            print(f"   {analytics['error']}")

        print()

        # 7. Multiple executions for metrics
        print("7. Running multiple executions for metrics...")
        prompts = [
            "What is Python?",
            "Explain machine learning",
            "How does the internet work?",
        ]

        print()
        for i, prompt in enumerate(prompts, 1):
            result = await api.execute_prompt(prompt, evaluate_quality=False)
            print(f"   {i}. {prompt[:30]:<32} {result.duration_ms:>6.1f}ms")

        print()

        # 8. Final status
        print("8. Final system status...")
        final_status = await api.get_status()

        print()
        print(f"   Total executions: {final_status.total_executions}")
        print(f"   Avg time:         {final_status.avg_execution_time_ms:.1f}ms")
        print(f"   Success rate:     {final_status.success_rate:.1%}")

        if final_status.total_cost_usd > 0:
            print(f"   Total cost:       ${final_status.total_cost_usd:.4f}")
            print(f"   Total tokens:     {final_status.total_tokens:,}")

        print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Next steps:")
    print("  • Integrate with your LLM backend")
    print("  • Set up advanced A/B testing")
    print("  • Create custom loop compositions")
    print("  • Track costs and quality over time")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
