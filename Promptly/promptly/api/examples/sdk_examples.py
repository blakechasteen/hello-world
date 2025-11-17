"""
Promptly SDK Examples
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sdk import PromptlyClient


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage ===\n")

    # Initialize client
    client = PromptlyClient(
        base_url="http://localhost:8000",
        api_key="pk_dev_key_12345"
    )

    try:
        # Check API health
        health = client.health()
        print(f"API Status: {health['status']}")
        print(f"Version: {health['version']}\n")

        # Create a prompt
        result = client.create_prompt(
            name="greeting",
            content="Hello {name}, welcome to Promptly!",
            metadata={"category": "greetings"}
        )
        print(f"Created: {result['message']}\n")

        # Get the prompt
        prompt = client.get_prompt("greeting")
        print(f"Prompt: {prompt['name']}")
        print(f"Content: {prompt['content']}")
        print(f"Version: {prompt['version']}\n")

        # List all prompts
        prompts = client.list_prompts()
        print(f"Total prompts: {len(prompts)}")
        for p in prompts:
            print(f"  - {p['name']} (v{p['version']})")

    finally:
        client.close()


def example_versioning():
    """Versioning example"""
    print("\n=== Versioning ===\n")

    with PromptlyClient(api_key="pk_dev_key_12345") as client:
        # Create first version
        client.create_prompt(
            name="summarizer",
            content="Summarize: {text}"
        )
        print("Created v1")

        # Create second version
        client.create_prompt(
            name="summarizer",
            content="Provide a brief summary of: {text}"
        )
        print("Created v2")

        # Create third version
        client.create_prompt(
            name="summarizer",
            content="Please summarize the following text:\n\n{text}"
        )
        print("Created v3\n")

        # Get different versions
        v1 = client.get_prompt("summarizer", version=1)
        v3 = client.get_prompt("summarizer", version=3)

        print(f"V1: {v1['content']}")
        print(f"V3: {v3['content']}\n")

        # Get diff
        diff = client.get_prompt_diff("summarizer", version_old=1, version_new=3)
        print(f"Changes: {diff['changes_summary']}")
        print(f"\nDiff:\n{diff['diff']}")


def example_branching():
    """Branching example"""
    print("\n=== Branching ===\n")

    with PromptlyClient(api_key="pk_dev_key_12345") as client:
        # List current branches
        branches = client.list_branches()
        print(f"Current branch: {branches['current_branch']}")
        print(f"All branches: {[b['name'] for b in branches['branches']]}\n")

        # Create a feature branch
        result = client.create_branch("feature/experiments", from_branch="main")
        print(f"{result['message']}\n")

        # Checkout the new branch
        result = client.checkout_branch("feature/experiments")
        print(f"{result['message']}\n")

        # Create prompts on feature branch
        client.create_prompt(
            name="experimental",
            content="Experimental prompt: {input}"
        )
        print("Created prompt on feature branch\n")

        # Switch back to main
        client.checkout_branch("main")
        print("Switched back to main")


def example_evaluation():
    """Evaluation example"""
    print("\n=== Evaluation ===\n")

    with PromptlyClient(api_key="pk_dev_key_12345") as client:
        # Create a prompt
        client.create_prompt(
            name="qa_prompt",
            content="Answer the question: {question}"
        )

        # Run evaluation
        result = client.run_evaluation(
            prompt_name="qa_prompt",
            test_cases=[
                {
                    "inputs": {"question": "What is AI?"},
                    "expected": "artificial intelligence"
                },
                {
                    "inputs": {"question": "What is ML?"},
                    "expected": "machine learning"
                }
            ],
            evaluator="keyword"
        )

        print(f"Evaluation ID: {result['evaluation_id']}")
        print(f"Total tests: {result['total_tests']}")
        print(f"Passed: {result['passed_tests']}")
        print(f"Failed: {result['failed_tests']}")
        print(f"Average score: {result['average_score']:.2f}\n")

        # List evaluations
        evals = client.list_evaluations(prompt_name="qa_prompt")
        print(f"Total evaluations: {len(evals)}")


def example_chains():
    """Chain execution example"""
    print("\n=== Chains ===\n")

    with PromptlyClient(api_key="pk_dev_key_12345") as client:
        # Create prompts for chain
        client.create_prompt(
            name="extract_keywords",
            content="Extract keywords from: {text}"
        )

        client.create_prompt(
            name="expand_query",
            content="Expand query: {keywords}"
        )

        client.create_prompt(
            name="generate_summary",
            content="Summarize results: {output}"
        )

        # Create chain
        result = client.create_chain(
            name="research_pipeline",
            steps=["extract_keywords", "expand_query", "generate_summary"],
            description="Research pipeline for document analysis"
        )
        print(f"{result['message']}\n")

        # Execute chain
        execution = client.execute_chain(
            chain_name="research_pipeline",
            initial_input={"text": "Artificial intelligence and machine learning"}
        )

        print(f"Execution ID: {execution['execution_id']}")
        print(f"Status: {execution['status']}")
        print(f"Steps completed: {len(execution['steps'])}")
        print(f"Total time: {execution['total_execution_time_ms']:.2f}ms\n")

        for i, step in enumerate(execution['steps'], 1):
            print(f"Step {i}: {step['step_name']}")
            print(f"  Status: {step['status']}")


def example_search():
    """Search example"""
    print("\n=== Search ===\n")

    with PromptlyClient(api_key="pk_dev_key_12345") as client:
        # Create prompts with tags
        client.create_prompt(
            name="nlp_summarizer",
            content="Summarize: {text}",
            metadata={"tags": ["nlp", "summarization"]}
        )

        client.create_prompt(
            name="nlp_translator",
            content="Translate to {language}: {text}",
            metadata={"tags": ["nlp", "translation"]}
        )

        client.create_prompt(
            name="cv_detector",
            content="Detect objects in image",
            metadata={"tags": ["cv", "detection"]}
        )

        # Search by tag
        nlp_prompts = client.search_prompts(tags=["nlp"])
        print(f"NLP prompts: {len(nlp_prompts)}")
        for p in nlp_prompts:
            print(f"  - {p['name']}")

        print()

        # Search by query
        summary_prompts = client.search_prompts(query="summarize")
        print(f"Summary prompts: {len(summary_prompts)}")
        for p in summary_prompts:
            print(f"  - {p['name']}")


def example_context_manager():
    """Context manager example"""
    print("\n=== Context Manager ===\n")

    # Using context manager (recommended)
    with PromptlyClient(api_key="pk_dev_key_12345") as client:
        prompts = client.list_prompts()
        print(f"Found {len(prompts)} prompts")
        # Session automatically closed when exiting context


def example_error_handling():
    """Error handling example"""
    print("\n=== Error Handling ===\n")

    from sdk.exceptions import PromptNotFoundError, AuthenticationError

    client = PromptlyClient(api_key="pk_dev_key_12345")

    try:
        # Try to get non-existent prompt
        prompt = client.get_prompt("nonexistent_prompt")
    except PromptNotFoundError as e:
        print(f"Error: {e}")

    try:
        # Try with invalid API key
        bad_client = PromptlyClient(api_key="invalid_key")
        bad_client.list_prompts()
    except AuthenticationError as e:
        print(f"Auth Error: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    print("Promptly SDK Examples\n")
    print("=" * 50)

    # Run examples
    example_basic_usage()
    example_versioning()
    example_branching()
    example_evaluation()
    example_chains()
    example_search()
    example_context_manager()
    example_error_handling()

    print("\n" + "=" * 50)
    print("Examples completed!")
