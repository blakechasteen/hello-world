#!/usr/bin/env python3
"""
Demonstration: Complete Beginner Workflow

Shows step-by-step how a beginner (non-coder) would use the DSPy integration:
1. Generate optimization prompt using beginner_prompts.py
2. Copy-paste into ChatGPT/Claude
3. Get back optimized prompt
4. Use optimized prompt for their task

This simulates the real-world workflow for non-technical users.
"""

from beginner_prompts import generate_hololoom_qa_prompt
import textwrap

def print_section(title: str, content: str):
    """Print a formatted section"""
    print("\n" + "="*70)
    print(f">>> {title}")
    print("="*70)
    print(content)


def main():
    """Demonstrate complete beginner workflow"""

    print("\n" + "="*70)
    print(">>> BEGINNER WORKFLOW DEMONSTRATION")
    print(">>> No Code Required - Chat-Based DSPy Optimization")
    print("="*70)

    # STEP 1: Generate prompt
    print_section(
        "STEP 1: Generate Optimization Prompt",
        textwrap.dedent("""
        A beginner wants to optimize a question-answering system for HoloLoom.
        They don't know Python, so they use the beginner prompt generator:

        > python beginner_prompts.py
        > Choose option 2: HoloLoom Q&A (pre-configured)

        The system generates a complete prompt with:
        - Task description
        - 3 example input-output pairs
        - 4-criteria scoring system (Accuracy, Clarity, Completeness, Conciseness)
        - Step-by-step optimization instructions
        """)
    )

    # Generate the actual prompt
    optimization_prompt = generate_hololoom_qa_prompt()

    print("\n>> Generated prompt preview (first 800 characters):")
    print("-" * 70)
    print(optimization_prompt[:800])
    print("... [prompt continues] ...")
    print("-" * 70)
    print(f"\n>> Full prompt length: {len(optimization_prompt)} characters")

    # STEP 2: Copy to ChatGPT
    print_section(
        "STEP 2: Copy-Paste Into ChatGPT/Claude",
        textwrap.dedent("""
        The beginner copies the entire prompt and pastes it into ChatGPT or Claude.

        The LLM will:
        1. Write 3 different Q&A prompts
        2. Test each on all 3 examples
        3. Score using the 4-criteria rubric (max 40 points)
        4. Identify the lowest-scoring criterion
        5. Improve the best prompt to address weaknesses
        6. Return the final optimized prompt with scores

        This takes about 30-60 seconds and requires zero coding.
        """)
    )

    # STEP 3: Example output from ChatGPT
    print_section(
        "STEP 3: ChatGPT Returns Optimized Prompt",
        textwrap.dedent("""
        ChatGPT responds with something like:

        -----------------------------------------------------------
        I tested 3 different Q&A prompts on your examples:

        PROMPT 1 SCORES:
        - Accuracy: 8/10
        - Clarity: 7/10
        - Completeness: 8/10
        - Conciseness: 6/10
        Total: 29/40

        PROMPT 2 SCORES:
        - Accuracy: 9/10
        - Clarity: 9/10
        - Completeness: 7/10
        - Conciseness: 8/10
        Total: 33/40

        PROMPT 3 SCORES:
        - Accuracy: 7/10
        - Clarity: 8/10
        - Completeness: 9/10
        - Conciseness: 7/10
        Total: 31/40

        BEST: Prompt 2 (33/40)
        WEAKNESS: Completeness (7/10)

        IMPROVED PROMPT 2:
        "Given the following context, provide a complete and accurate
        answer that covers all key aspects. Structure your response with:
        1. Direct answer (1-2 sentences)
        2. Key details and mechanisms
        3. Practical implications or trade-offs

        Context: {context}
        Question: {question}

        Ensure you cover ALL information from the context without
        assuming prior knowledge."

        FINAL SCORES:
        - Accuracy: 9/10
        - Clarity: 9/10
        - Completeness: 9/10 (improved!)
        - Conciseness: 8/10
        Total: 35/40

        This prompt is now optimized and ready to use!
        -----------------------------------------------------------
        """)
    )

    # STEP 4: Use optimized prompt
    print_section(
        "STEP 4: Use The Optimized Prompt",
        textwrap.dedent("""
        The beginner can now use the optimized prompt in their application:

        OPTION A: Manual use (copy-paste into any LLM)
        OPTION B: Integrate into HoloLoom (requires developer help)
        OPTION C: Add to team's prompt library for others to use

        The key innovation: They went from vague idea to optimized prompt
        in under 2 minutes, with ZERO code written.
        """)
    )

    # Benefits summary
    print_section(
        ">>> KEY BENEFITS",
        textwrap.dedent("""
        1. NO CODE REQUIRED
           - Just copy-paste into ChatGPT
           - Get back optimized prompts
           - Start using immediately

        2. SYSTEMATIC OPTIMIZATION
           - Not guesswork - metrics-driven
           - Clear scoring (0-10 per criterion)
           - Iterative improvement

        3. BEGINNER ACCESSIBLE
           - No Python knowledge needed
           - No DSPy installation required
           - No terminal commands

        4. PROFESSIONAL RESULTS
           - Same quality as developer-written code
           - Reproducible optimization
           - Documented with scores

        5. TEAM SCALING
           - Share prompts as text files
           - Version control friendly
           - Easy collaboration

        This lowers the barrier to entry from:
        "You need to be a Python developer with ML experience"
        to:
        "If you can copy-paste, you can optimize prompts"
        """)
    )

    # Next steps
    print_section(
        ">>> NEXT STEPS FOR BEGINNERS",
        textwrap.dedent("""
        After mastering chat-based optimization, beginners can:

        1. Save prompts to files (built-in feature)
        2. Share with team via Git/Dropbox/email
        3. Request developer integration into HoloLoom
        4. Graduate to programmatic DSPy (optional)

        The system grows with the user:
        Beginner (chat) → Intermediate (YAML) → Advanced (Python)
        """)
    )

    print("\n" + "="*70)
    print(">>> END OF DEMONSTRATION")
    print("="*70)
    print("\n>> Want to try it yourself?")
    print(">> 1. Run: python beginner_prompts.py")
    print(">> 2. Choose option 2 (HoloLoom Q&A)")
    print(">> 3. Copy output into ChatGPT")
    print(">> 4. Get optimized prompt in 30 seconds!")
    print("\n")


if __name__ == "__main__":
    main()
