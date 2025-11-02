#!/usr/bin/env python3
"""
Advanced Promptly Integration Example
Demonstrates how to use Promptly with Claude API for evaluation and chaining
"""

from promptly import Promptly
import json

# Mock model function (replace with actual API calls)
def mock_model(prompt):
    """
    Replace this with actual LLM API calls
    Example with Anthropic Claude:
    
    import anthropic
    client = anthropic.Anthropic(api_key="your-api-key")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
    """
    return f"[Mock response to: {prompt[:50]}...]"


def example_1_basic_usage():
    """Example 1: Basic prompt usage"""
    print("\n=== Example 1: Basic Usage ===\n")
    
    promptly = Promptly()
    
    # Get a prompt and format it
    prompt_data = promptly.get('summarizer')
    if prompt_data:
        formatted = prompt_data['content'].format(
            text="Promptly is a command-line tool for managing prompts with versioning."
        )
        
        print(f"Prompt: {formatted}\n")
        
        # Call your LLM
        response = mock_model(formatted)
        print(f"Response: {response}\n")


def example_2_evaluation():
    """Example 2: Automated evaluation with scoring"""
    print("\n=== Example 2: Evaluation with Scoring ===\n")
    
    promptly = Promptly()
    
    # Define test cases with custom evaluator
    def score_similarity(actual, expected):
        """Simple scoring function - replace with better metrics"""
        if not actual or not expected:
            return 0.0
        
        # Simple word overlap scoring (replace with semantic similarity)
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = actual_words & expected_words
        return len(overlap) / len(expected_words)
    
    test_cases = [
        {
            'id': 'test-1',
            'inputs': {'text': 'Machine learning enables computers to learn from data.'},
            'expected': 'machine learning computers data',
            'evaluator': score_similarity
        },
        {
            'id': 'test-2',
            'inputs': {'text': 'Climate change affects global temperatures.'},
            'expected': 'climate change temperature global',
            'evaluator': score_similarity
        }
    ]
    
    # Run evaluation
    results = promptly.eval_prompt('summarizer', test_cases, model_func=mock_model)
    
    print("Evaluation Results:")
    for i, result in enumerate(results, 1):
        print(f"\nTest {i}:")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Actual: {result['actual']}")
    
    # Calculate average score
    avg_score = sum(r['score'] for r in results if r['score']) / len(results)
    print(f"\nAverage Score: {avg_score:.2f}")


def example_3_chaining():
    """Example 3: Chain execution with data flow"""
    print("\n=== Example 3: Prompt Chaining ===\n")
    
    promptly = Promptly()
    
    # Define initial input
    initial_input = {
        'topic': 'Artificial Intelligence',
        'style': 'educational',
        'audience': 'beginners'
    }
    
    # Execute chain
    results = promptly.execute_chain(
        'content-pipeline',
        initial_input,
        model_func=mock_model
    )
    
    print("Chain Execution Results:")
    for i, step in enumerate(results, 1):
        print(f"\nStep {i}: {step['step']}")
        print(f"  Prompt: {step['prompt'][:100]}...")
        print(f"  Output: {step['output'][:100]}...")


def example_4_ab_testing():
    """Example 4: A/B testing different prompt versions"""
    print("\n=== Example 4: A/B Testing ===\n")
    
    promptly = Promptly()
    
    # Test data
    test_inputs = [
        {'text': 'AI is transforming industries.'},
        {'text': 'Machine learning models need data.'},
        {'text': 'Neural networks mimic the brain.'}
    ]
    
    # Test version 1
    print("Testing Version 1:")
    v1_results = []
    for test_input in test_inputs:
        prompt_data = promptly.get('summarizer', version=1)
        if prompt_data:
            formatted = prompt_data['content'].format(**test_input)
            response = mock_model(formatted)
            v1_results.append(response)
    
    # Test version 2
    print("\nTesting Version 2:")
    v2_results = []
    for test_input in test_inputs:
        prompt_data = promptly.get('summarizer', version=2)
        if prompt_data:
            formatted = prompt_data['content'].format(**test_input)
            response = mock_model(formatted)
            v2_results.append(response)
    
    print("\nComparison complete - analyze results to choose winner!")


def example_5_metadata_tracking():
    """Example 5: Track prompt performance with metadata"""
    print("\n=== Example 5: Metadata Tracking ===\n")
    
    promptly = Promptly()
    
    # Add prompt with detailed metadata
    metadata = {
        'model': 'claude-sonnet-4',
        'temperature': 0.7,
        'max_tokens': 500,
        'tags': ['summarization', 'production'],
        'performance': {
            'avg_latency_ms': 850,
            'success_rate': 0.95
        }
    }
    
    promptly.add(
        'production-summarizer',
        'Provide a concise summary: {text}',
        metadata=metadata
    )
    
    # Retrieve and show metadata
    prompt_data = promptly.get('production-summarizer')
    if prompt_data:
        print("Prompt Metadata:")
        print(json.dumps(prompt_data['metadata'], indent=2))


def example_6_batch_evaluation():
    """Example 6: Batch evaluation across multiple prompts"""
    print("\n=== Example 6: Batch Evaluation ===\n")
    
    promptly = Promptly()
    
    # Get all prompts
    prompts = promptly.list_prompts()
    
    # Define common test cases
    test_cases = [
        {
            'id': 'consistency-test',
            'inputs': {'text': 'Test input for consistency check.'},
            'expected': 'consistent output'
        }
    ]
    
    results_summary = {}
    
    for prompt in prompts:
        try:
            results = promptly.eval_prompt(
                prompt['name'],
                test_cases,
                model_func=mock_model
            )
            
            avg_score = sum(r['score'] for r in results if r['score']) / len(results) if results else 0
            results_summary[prompt['name']] = avg_score
            
        except Exception as e:
            print(f"Error evaluating {prompt['name']}: {e}")
    
    print("\nBatch Evaluation Summary:")
    for prompt_name, score in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prompt_name}: {score:.2f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Promptly Advanced Integration Examples")
    print("=" * 60)
    
    # Note: These examples assume you've already initialized prompty
    # and added some prompts. Run the demo script first!
    
    try:
        example_1_basic_usage()
        example_2_evaluation()
        # example_3_chaining()  # Uncomment if you've created chains
        # example_4_ab_testing()  # Uncomment if you have multiple versions
        example_5_metadata_tracking()
        # example_6_batch_evaluation()  # Uncomment for batch testing
        
        print("\n" + "=" * 60)
        print("Examples complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you've run 'prompty init' and added some prompts first!")
        print("Run the demo.sh script to set up examples.")
