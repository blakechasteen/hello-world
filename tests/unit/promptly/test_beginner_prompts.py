#!/usr/bin/env python3
"""
Test script for beginner prompts functionality
Tests all 4 template types programmatically
"""

from HoloLoom.promptly.beginner_prompts import (
    generate_basic_optimization_prompt,
    generate_hololoom_qa_prompt,
    generate_workflow_optimization_prompt,
    generate_code_review_prompt
)

def test_basic_optimization():
    """Test 1: Basic task optimization prompt"""
    print("\n" + "="*70)
    print("TEST 1: Basic Task Optimization")
    print("="*70)

    task = "Summarize technical articles in simple language"
    examples = [
        {
            "input": "Transformers use attention mechanisms...",
            "output": "Transformers are neural networks that focus on important parts of text"
        },
        {
            "input": "Gradient descent optimizes loss functions...",
            "output": "Gradient descent is how neural networks learn from mistakes"
        },
        {
            "input": "Embeddings represent text as vectors...",
            "output": "Embeddings turn words into numbers that capture meaning"
        }
    ]

    try:
        prompt = generate_basic_optimization_prompt(task, examples)

        # Verify prompt structure
        assert "My Task" in prompt
        assert task in prompt
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "Example 3" in prompt
        assert "Scoring System" in prompt
        assert "Optimization Process" in prompt

        print(">> SUCCESS: Basic optimization prompt generated")
        print(f">> Prompt length: {len(prompt)} characters")
        print(f">> Contains all required sections: YES")

        # Show first 500 chars
        print("\n>> Preview (first 500 chars):")
        print(prompt[:500] + "...")

        return True
    except Exception as e:
        print(f">> ERROR: {e}")
        return False


def test_hololoom_qa():
    """Test 2: HoloLoom Q&A pre-configured prompt"""
    print("\n" + "="*70)
    print("TEST 2: HoloLoom Q&A (Pre-configured)")
    print("="*70)

    try:
        prompt = generate_hololoom_qa_prompt()

        # Verify HoloLoom-specific content
        assert "Thompson Sampling" in prompt
        assert "Matryoshka embedding" in prompt
        assert "BARE, FAST, and FUSED" in prompt
        assert "Retrieved Context" in prompt
        assert "Accuracy (0-10)" in prompt
        assert "Clarity (0-10)" in prompt
        assert "Completeness (0-10)" in prompt
        assert "Conciseness (0-10)" in prompt

        print(">> SUCCESS: HoloLoom Q&A prompt generated")
        print(f">> Prompt length: {len(prompt)} characters")
        print(f">> Contains HoloLoom examples: YES")
        print(f">> Contains 4-criteria scoring: YES")

        # Count examples
        example_count = prompt.count("Example")
        print(f">> Number of examples: {example_count}")

        return True
    except Exception as e:
        print(f">> ERROR: {e}")
        return False


def test_workflow_optimization():
    """Test 3: Workflow optimization prompt"""
    print("\n" + "="*70)
    print("TEST 3: Workflow Optimization")
    print("="*70)

    workflow_desc = "Multi-step research pipeline with verification"
    workflow_steps = [
        "Break question into sub-questions",
        "Retrieve context for each sub-question",
        "Answer each sub-question independently",
        "Synthesize answers into final response"
    ]

    examples = [
        {
            "input": "What are the tradeoffs of Thompson Sampling?",
            "step1_output": "1. What is Thompson Sampling? 2. What are its advantages? 3. What are its disadvantages?",
            "step2_output": "Context retrieved for all 3 sub-questions",
            "final_output": "Thompson Sampling balances exploration and exploitation..."
        },
        {
            "input": "How does Matryoshka embedding work?",
            "step1_output": "1. What is Matryoshka? 2. How are embeddings structured? 3. What are the benefits?",
            "step2_output": "Context retrieved for all 3 sub-questions",
            "final_output": "Matryoshka embeddings use nested representations..."
        },
        {
            "input": "Compare BARE vs FUSED mode",
            "step1_output": "1. What is BARE mode? 2. What is FUSED mode? 3. When to use each?",
            "step2_output": "Context retrieved for all 3 sub-questions",
            "final_output": "BARE mode prioritizes speed while FUSED mode prioritizes quality..."
        }
    ]

    try:
        prompt = generate_workflow_optimization_prompt(
            workflow_desc,
            workflow_steps,
            examples
        )

        # Verify workflow structure
        assert "My Task" in prompt
        assert workflow_desc in prompt
        assert "Workflow Steps" in prompt
        assert "Step 1" in prompt and workflow_steps[0] in prompt
        assert "Step 2" in prompt and workflow_steps[1] in prompt
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "Example 3" in prompt
        assert "Step Coherence" in prompt
        assert "Information Preservation" in prompt

        print(">> SUCCESS: Workflow optimization prompt generated")
        print(f">> Prompt length: {len(prompt)} characters")
        print(f">> Workflow steps: {len(workflow_steps)}")
        print(f">> Examples: {len(examples)}")
        print(f">> Contains all workflow sections: YES")

        return True
    except Exception as e:
        print(f">> ERROR: {e}")
        return False


def test_code_review():
    """Test 4: Code review pre-configured prompt"""
    print("\n" + "="*70)
    print("TEST 4: Code Review (Pre-configured)")
    print("="*70)

    try:
        prompt = generate_code_review_prompt()

        # Verify code review specific content
        assert "SQL injection" in prompt
        assert "calculate_total" in prompt
        assert "process_file" in prompt
        assert "Security Coverage" in prompt
        assert "Actionability" in prompt
        assert "Prioritization" in prompt
        assert "Code Quality" in prompt
        assert "CRITICAL SECURITY ISSUE" in prompt

        print(">> SUCCESS: Code review prompt generated")
        print(f">> Prompt length: {len(prompt)} characters")
        print(f">> Contains security examples: YES")
        print(f">> Contains 4-criteria scoring: YES")

        # Count code examples
        code_example_count = prompt.count("```python")
        print(f">> Number of code examples: {code_example_count}")

        return True
    except Exception as e:
        print(f">> ERROR: {e}")
        return False


def test_file_saving():
    """Test 5: File saving functionality"""
    print("\n" + "="*70)
    print("TEST 5: File Saving")
    print("="*70)

    import os
    import tempfile

    try:
        prompt = generate_hololoom_qa_prompt()

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(prompt)
            temp_path = f.name

        # Verify file exists and has content
        assert os.path.exists(temp_path)
        file_size = os.path.getsize(temp_path)
        assert file_size > 0

        # Read back and verify
        with open(temp_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()

        assert saved_content == prompt

        # Cleanup
        os.unlink(temp_path)

        print(">> SUCCESS: File saving works correctly")
        print(f">> File size: {file_size} bytes")
        print(f">> Content preserved: YES")

        return True
    except Exception as e:
        print(f">> ERROR: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(">>> BEGINNER PROMPTS TEST SUITE")
    print("="*70)

    results = {
        "Basic Optimization": test_basic_optimization(),
        "HoloLoom Q&A": test_hololoom_qa(),
        "Workflow Optimization": test_workflow_optimization(),
        "Code Review": test_code_review(),
        "File Saving": test_file_saving()
    }

    # Summary
    print("\n" + "="*70)
    print(">>> TEST SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n>> Results: {passed}/{total} tests passed")

    if passed == total:
        print(">> SUCCESS: All tests passed!")
        return 0
    else:
        print(">> ERROR: Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
