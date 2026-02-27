#!/usr/bin/env python3
"""
Beginner-Friendly DSPy Prompts for HoloLoom (Promptly)

Provides chat-based DSPy optimization without touching Python or terminals.
Based on principles from Ethan Mollick's DSPy explanation.

These prompts can be used directly in ChatGPT or Claude to get DSPy-like
optimization without any technical setup.

Part of Promptly - The Universal AI Reliability Layer
Open source (MIT License): https://github.com/yourusername/promptly
"""

from typing import Dict, List

# ============================================================================
# BEGINNER PROMPT TEMPLATES
# ============================================================================

BASIC_OPTIMIZATION_PROMPT = """
I need to create a self-optimizing prompt system.

**My Task**: {task_description}

**My Examples** (at least 3 input-output pairs):

Example 1:
- Input: {input_1}
- Output: {output_1}

Example 2:
- Input: {input_2}
- Output: {output_2}

Example 3:
- Input: {input_3}
- Output: {output_3}

**Scoring System**: Please create a scoring system with these criteria:
1. Functionality (does it accomplish the task?)
2. Format (does it match the expected format?)
3. Completeness (does it include all necessary information?)

**Optimization Process**:
1. Please write 3 different prompts that could handle my task
2. Test each prompt on ALL the examples I gave you
3. Score each result using the rubric above
4. Take the best one and improve it by fixing whatever scored lowest
5. Give me the final improved prompt with the scoring breakdown

Please show your work for each step so I can see the optimization process.
"""


HOLOLOOM_QA_OPTIMIZATION_PROMPT = """
I need to create a self-optimizing prompt system for question-answering using HoloLoom's knowledge base.

**My Task**: Answer technical questions accurately using retrieved context from hololoom memory.

**My Examples**:

Example 1:
- Input Question: "What is Thompson Sampling?"
- Retrieved Context: "Thompson Sampling is a Bayesian bandit algorithm that balances exploration and exploitation by sampling from posterior distributions."
- Desired Output: "Thompson Sampling is a probabilistic approach to the multi-armed bandit problem. It works by maintaining a probability distribution for each action's reward, then sampling from these distributions to choose actions. This naturally balances exploration (trying uncertain actions) with exploitation (choosing seemingly good actions). It's particularly effective because it automatically adjusts exploration based on uncertainty."

Example 2:
- Input Question: "How does Matryoshka embedding work?"
- Retrieved Context: "Matryoshka embeddings use nested representation learning where different dimensions capture information at different scales (96, 192, 384 dimensions)."
- Desired Output: "Matryoshka embeddings create multi-scale representations of data. Like Russian nesting dolls, each dimensional level contains a complete but progressively more detailed representation. The 96-dimensional version captures coarse information, 192 adds more detail, and 384 provides the full resolution. This allows flexible trade-offs between speed and accuracy."

Example 3:
- Input Question: "What's the difference between BARE, FAST, and FUSED modes?"
- Retrieved Context: "HoloLoom has three execution modes: BARE (minimal processing, fastest), FAST (balanced, recommended), FUSED (full processing, highest quality)."
- Desired Output: "HoloLoom offers three execution modes with different speed-quality trade-offs: BARE mode uses minimal processing for maximum speed (~50ms queries), FAST mode balances performance and quality (~150ms), and FUSED mode enables all features for highest quality responses (~300ms). Choose based on your latency requirements and quality needs."

**Scoring System**:
1. Accuracy (0-10): Does the answer correctly represent the context?
2. Clarity (0-10): Is the explanation clear and well-structured?
3. Completeness (0-10): Does it cover all key aspects from the context?
4. Conciseness (0-10): Is it appropriately detailed without being verbose?

**Optimization Process**:
1. Write 3 different prompts that could handle question-answering with context
2. Test each prompt on ALL the examples above
3. Score each result using the 4-criteria rubric (max 40 points)
4. Identify which criteria scored lowest for the best prompt
5. Improve the prompt by specifically addressing that weakness
6. Show me the final optimized prompt and scoring breakdown

Please show your reasoning at each step so I can understand the optimization process.
"""


WORKFLOW_OPTIMIZATION_PROMPT = """
I need to create a self-optimizing prompt system for a multi-step workflow.

**My Task**: {workflow_description}

**Workflow Steps**:
{workflow_steps}

**My Examples** (full workflow inputs and outputs):

Example 1:
- Initial Input: {workflow_input_1}
- Step 1 Output: {step1_output_1}
- Step 2 Output: {step2_output_1}
- Final Output: {final_output_1}

Example 2:
- Initial Input: {workflow_input_2}
- Step 1 Output: {step1_output_2}
- Step 2 Output: {step2_output_2}
- Final Output: {final_output_2}

Example 3:
- Initial Input: {workflow_input_3}
- Step 1 Output: {step1_output_3}
- Step 2 Output: {step2_output_3}
- Final Output: {final_output_3}

**Scoring System**:
1. Step Coherence (0-10): Do steps flow logically from each other?
2. Information Preservation (0-10): Is key information maintained across steps?
3. Final Quality (0-10): Does the final output meet requirements?
4. Efficiency (0-10): Are steps concise without sacrificing quality?

**Optimization Process**:
1. For EACH step in the workflow, write 2 different prompt versions
2. Test all combinations on the examples (if 3 steps with 2 versions each = 8 combinations)
3. Score each complete workflow using the rubric above
4. Identify the best performing combination
5. For that combination, improve the lowest-scoring step
6. Give me the final optimized prompt for EACH step with scoring breakdown

Please show your work including which combinations you tested.
"""


CODE_REVIEW_OPTIMIZATION_PROMPT = """
I need to create a self-optimizing prompt system for code review.

**My Task**: Review code for security, style, and best practices.

**My Examples**:

Example 1:
- Input Code:
```python
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id={user_id}"
    return db.execute(query)
```
- Desired Review:
"CRITICAL SECURITY ISSUE: SQL injection vulnerability. The user_id parameter is directly interpolated into the query string without sanitization.

FIX: Use parameterized queries:
```python
def get_user_data(user_id):
    query = \"SELECT * FROM users WHERE id=?\"
    return db.execute(query, (user_id,))
```

STYLE: Function lacks type hints and docstring. Consider adding.
BEST PRACTICE: Consider limiting columns retrieved instead of SELECT *."

Example 2:
- Input Code:
```python
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item.price
    return total
```
- Desired Review:
"SECURITY: No issues detected (no external input or dangerous operations).
STYLE: Code is clear and readable. Consider adding type hints.
BEST PRACTICE: Could be more Pythonic using sum():
```python
def calculate_total(items: List[Item]) -> float:
    \"\"\"Calculate total price of items.\"\"\"
    return sum(item.price for item in items)
```"

Example 3:
- Input Code:
```python
def process_file(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return data
```
- Desired Review:
"BEST PRACTICE: File not closed if exception occurs. Use context manager:
```python
def process_file(filename: str) -> str:
    \"\"\"Read and return file contents.\"\"\"
    with open(filename, 'r') as f:
        return f.read()
```

STYLE: Add type hints and docstring (shown above).
SECURITY: Consider validating filename to prevent path traversal attacks."

**Scoring System**:
1. Security Coverage (0-10): Are vulnerabilities identified?
2. Actionability (0-10): Are fixes clear and implementable?
3. Prioritization (0-10): Are issues properly classified (CRITICAL/STYLE/BEST PRACTICE)?
4. Code Quality (0-10): Are suggested improvements actually better?

**Optimization Process**:
1. Write 3 different code review prompts
2. Test each on ALL examples above
3. Score using the 4-criteria rubric (max 40 points)
4. Take the best one and improve the lowest-scoring criterion
5. Give me the final optimized code review prompt with scoring

Show your work for transparency.
"""


# ============================================================================
# PROMPT GENERATOR FUNCTIONS
# ============================================================================

def generate_basic_optimization_prompt(
    task_description: str,
    examples: List[Dict[str, str]]
) -> str:
    """
    Generate a basic optimization prompt for any task.

    Args:
        task_description: What the prompt should accomplish
        examples: List of dicts with 'input' and 'output' keys

    Returns:
        Formatted prompt ready to paste into ChatGPT
    """
    if len(examples) < 3:
        raise ValueError("Need at least 3 examples for optimization")

    prompt = BASIC_OPTIMIZATION_PROMPT.format(
        task_description=task_description,
        input_1=examples[0]["input"],
        output_1=examples[0]["output"],
        input_2=examples[1]["input"],
        output_2=examples[1]["output"],
        input_3=examples[2]["input"],
        output_3=examples[2]["output"]
    )

    return prompt


def generate_hololoom_qa_prompt() -> str:
    """
    Generate the pre-configured HoloLoom Q&A optimization prompt.

    Returns:
        Formatted prompt ready to use
    """
    return HOLOLOOM_QA_OPTIMIZATION_PROMPT


def generate_workflow_optimization_prompt(
    workflow_description: str,
    workflow_steps: List[str],
    examples: List[Dict[str, any]]
) -> str:
    """
    Generate a workflow optimization prompt.

    Args:
        workflow_description: What the workflow accomplishes
        workflow_steps: List of step descriptions
        examples: List of complete workflow examples

    Returns:
        Formatted prompt ready to use
    """
    if len(examples) < 3:
        raise ValueError("Need at least 3 complete workflow examples")

    steps_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(workflow_steps))

    prompt = WORKFLOW_OPTIMIZATION_PROMPT.format(
        workflow_description=workflow_description,
        workflow_steps=steps_str,
        workflow_input_1=examples[0]["input"],
        step1_output_1=examples[0]["step1_output"],
        step2_output_1=examples[0]["step2_output"],
        final_output_1=examples[0]["final_output"],
        workflow_input_2=examples[1]["input"],
        step1_output_2=examples[1]["step1_output"],
        step2_output_2=examples[1]["step2_output"],
        final_output_2=examples[1]["final_output"],
        workflow_input_3=examples[2]["input"],
        step1_output_3=examples[2]["step1_output"],
        step2_output_3=examples[2]["step2_output"],
        final_output_3=examples[2]["final_output"]
    )

    return prompt


def generate_code_review_prompt() -> str:
    """
    Generate the pre-configured code review optimization prompt.

    Returns:
        Formatted prompt ready to use
    """
    return CODE_REVIEW_OPTIMIZATION_PROMPT


# ============================================================================
# CLI FOR GENERATING PROMPTS
# ============================================================================

def main():
    """Interactive CLI for generating beginner prompts"""
    print("\n" + "="*70)
    print(">>> Promptly: Beginner-Friendly DSPy Prompt Generator")
    print("="*70)
    print("Open Source (MIT) | https://github.com/yourusername/promptly")
    print("="*70 + "\n")

    print("Choose a template:")
    print("  1. Basic Task Optimization")
    print("  2. HoloLoom Q&A (pre-configured)")
    print("  3. Workflow Optimization")
    print("  4. Code Review (pre-configured)")
    print()

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        print("\n>> Basic Task Optimization\n")
        task = input("Describe your task: ")

        examples = []
        for i in range(3):
            print(f"\nExample {i+1}:")
            inp = input("  Input: ")
            out = input("  Output: ")
            examples.append({"input": inp, "output": out})

        prompt = generate_basic_optimization_prompt(task, examples)

    elif choice == "2":
        print("\n>> Generating HoloLoom Q&A prompt...\n")
        prompt = generate_hololoom_qa_prompt()

    elif choice == "3":
        print("\n>> Workflow Optimization\n")
        workflow_desc = input("Describe your workflow: ")

        num_steps = int(input("How many steps? "))
        steps = []
        for i in range(num_steps):
            steps.append(input(f"  Step {i+1} description: "))

        # For simplicity, just generate template
        print("\n>> WARNING: For full workflow prompts, use generate_workflow_optimization_prompt() programmatically")
        prompt = "See beginner_prompts.py for workflow template"

    elif choice == "4":
        print("\n>> Generating Code Review prompt...\n")
        prompt = generate_code_review_prompt()

    else:
        print(">> ERROR: Invalid choice")
        return

    print("\n" + "="*70)
    print(">> SUCCESS: Generated Prompt (copy and paste into ChatGPT or Claude):")
    print("="*70 + "\n")
    print(prompt)
    print("\n" + "="*70)

    # Optionally save to file
    save = input("\nSave to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Filename (e.g., my_prompt.txt): ").strip()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f">> SUCCESS: Saved to {filename}")


if __name__ == "__main__":
    main()
