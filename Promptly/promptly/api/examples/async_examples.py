"""
Async SDK Examples
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sdk import AsyncPromptlyClient


async def example_basic_async():
    """Basic async usage"""
    print("=== Basic Async Usage ===\n")

    async with AsyncPromptlyClient(
        base_url="http://localhost:8000",
        api_key="pk_dev_key_12345"
    ) as client:

        # Check health
        health = await client.health()
        print(f"API Status: {health['status']}\n")

        # Create prompt
        await client.create_prompt(
            name="async_test",
            content="Async prompt: {input}"
        )
        print("Created prompt\n")

        # Get prompt
        prompt = await client.get_prompt("async_test")
        print(f"Prompt: {prompt['name']}")
        print(f"Content: {prompt['content']}\n")


async def example_concurrent_operations():
    """Concurrent operations example"""
    print("=== Concurrent Operations ===\n")

    async with AsyncPromptlyClient(api_key="pk_dev_key_12345") as client:

        # Create multiple prompts concurrently
        tasks = [
            client.create_prompt(f"prompt_{i}", f"Content {i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        print(f"Created {len(results)} prompts concurrently\n")

        # Fetch them all concurrently
        fetch_tasks = [
            client.get_prompt(f"prompt_{i}")
            for i in range(5)
        ]

        prompts = await asyncio.gather(*fetch_tasks)
        print("Fetched prompts:")
        for p in prompts:
            print(f"  - {p['name']} (v{p['version']})")


async def example_async_evaluation():
    """Async evaluation example"""
    print("\n=== Async Evaluation ===\n")

    async with AsyncPromptlyClient(api_key="pk_dev_key_12345") as client:

        # Create prompt
        await client.create_prompt(
            name="async_eval",
            content="Process: {input}"
        )

        # Run evaluation
        result = await client.run_evaluation(
            prompt_name="async_eval",
            test_cases=[
                {"inputs": {"input": f"test_{i}"}}
                for i in range(3)
            ]
        )

        print(f"Evaluation: {result['evaluation_id']}")
        print(f"Tests: {result['total_tests']}")
        print(f"Results: {len(result['results'])}")


async def example_async_chains():
    """Async chain example"""
    print("\n=== Async Chains ===\n")

    async with AsyncPromptlyClient(api_key="pk_dev_key_12345") as client:

        # Create chain steps
        await client.create_prompt("step_a", "Step A: {input}")
        await client.create_prompt("step_b", "Step B: {output}")

        # Create chain
        await client.create_chain(
            name="async_chain",
            steps=["step_a", "step_b"]
        )

        # Execute chain
        result = await client.execute_chain(
            chain_name="async_chain",
            initial_input={"input": "test"}
        )

        print(f"Execution: {result['execution_id']}")
        print(f"Status: {result['status']}")
        print(f"Steps: {len(result['steps'])}")


async def example_batch_operations():
    """Batch operations example"""
    print("\n=== Batch Operations ===\n")

    async with AsyncPromptlyClient(api_key="pk_dev_key_12345") as client:

        # Batch create
        prompts_to_create = [
            ("batch_1", "Content 1", {"priority": "high"}),
            ("batch_2", "Content 2", {"priority": "medium"}),
            ("batch_3", "Content 3", {"priority": "low"}),
        ]

        create_tasks = [
            client.create_prompt(name, content, metadata)
            for name, content, metadata in prompts_to_create
        ]

        await asyncio.gather(*create_tasks)
        print("Created 3 prompts in batch\n")

        # Batch retrieve
        retrieve_tasks = [
            client.get_prompt(name)
            for name, _, _ in prompts_to_create
        ]

        prompts = await asyncio.gather(*retrieve_tasks)

        print("Retrieved prompts:")
        for p in prompts:
            print(f"  - {p['name']}: {p['metadata'].get('priority')}")


async def main():
    """Run all async examples"""
    print("Promptly Async SDK Examples\n")
    print("=" * 50 + "\n")

    await example_basic_async()
    await example_concurrent_operations()
    await example_async_evaluation()
    await example_async_chains()
    await example_batch_operations()

    print("\n" + "=" * 50)
    print("Async examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
