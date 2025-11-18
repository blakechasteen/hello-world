#!/usr/bin/env python3
"""
Demo: Contract-First Prompting

Interactive demonstration of contract-first prompting for clarity of intent.

Created: 2025-11-18
Usage:
    PYTHONPATH=. python demos/demo_contract_first_prompting.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.prompting import (
    ContractFirstPrompting,
    DiggingStrategy,
)


async def demo_basic_usage():
    """Demo 1: Basic usage with simple user input."""
    print("=" * 60)
    print("DEMO 1: Basic Contract-First Prompting")
    print("=" * 60)
    print()

    async with ContractFirstPrompting(
        confidence_threshold=0.95,
        max_questions=10,
        digging_strategy=DiggingStrategy.ADAPTIVE,
    ) as cfp:
        # Step 1: Start with rough idea
        rough_idea = "I need a function to validate email addresses"
        print(f"User: {rough_idea}")
        print()

        welcome = await cfp.start(rough_idea)
        print(f"Assistant: {welcome}")

        # Step 2: Iterative questioning
        simulated_answers = [
            "Python",  # Tech stack
            "Return a detailed error object with error type and message",  # Error handling
            "Support international domains with Unicode characters",  # Edge cases
            "Should handle plus addressing like user+tag@domain.com",  # More edge cases
            "Regex validation is fine, no DNS lookup needed",  # Performance
            "Used by web form validation",  # Purpose
            "Web developers",  # Audience
            "Returns clear error messages and validates correctly",  # Success criteria
        ]

        answer_index = 0
        while not cfp.is_confident() and answer_index < len(simulated_answers):
            question = await cfp.next_question()
            if not question:
                break

            print(f"Assistant: {question}")

            # Simulate user answer
            answer = simulated_answers[answer_index]
            print(f"User: {answer}")
            print()

            await cfp.answer(answer)
            answer_index += 1

        # Step 3: Echo check
        print("--- Echo Check ---")
        echo = await cfp.echo_check()
        print(f"Assistant:\n{echo}")
        print()

        # Step 4: User approval (simulated)
        print("User: yes")
        print()

        approved = await cfp.approve("yes")

        if approved:
            # Step 5: Execute
            print("--- Building Deliverable ---")
            deliverable = await cfp.execute()
            print(deliverable)
            print()

        # Step 6: Feedback
        await cfp.reflect({"successful": True, "clarity_achieved": True})


async def demo_with_blueprint():
    """Demo 2: Request blueprint before approval."""
    print()
    print("=" * 60)
    print("DEMO 2: Contract-First with Blueprint Request")
    print("=" * 60)
    print()

    async with ContractFirstPrompting(
        confidence_threshold=0.90,  # Lower threshold for demo
        max_questions=5,
        digging_strategy=DiggingStrategy.ESSENTIAL_ONLY,  # Fewer questions
    ) as cfp:
        rough_idea = "Write a technical document explaining Thompson Sampling"
        print(f"User: {rough_idea}")
        print()

        await cfp.start(rough_idea)

        # Simulated Q&A
        qa_pairs = [
            (
                "Who will use/read/see this?",
                "ML engineers and data scientists with basic probability knowledge",
            ),
            (
                "What problem does this solve? Why does it need to exist?",
                "Help engineers understand when to use Thompson Sampling for A/B testing",
            ),
            (
                "How will we know this is good enough?",
                "Readers can implement Thompson Sampling and understand the math",
            ),
            ("What tone?", "Technical but accessible, with concrete examples"),
            ("How long should it be?", "2000-3000 words with code examples"),
        ]

        for expected_q, answer in qa_pairs:
            question = await cfp.next_question()
            if question:
                print(f"Assistant: {question}")
                print(f"User: {answer}")
                print()
                await cfp.answer(answer)

        # Echo check
        echo = await cfp.echo_check()
        print("--- Echo Check ---")
        print(f"Assistant:\n{echo}")
        print()

        # Request blueprint
        print("User: blueprint")
        print()

        blueprint = await cfp.blueprint_view()
        print("--- Blueprint ---")
        print(blueprint.render())
        print()

        # Approve
        print("User: yes")
        approved = await cfp.approve("yes")

        if approved:
            deliverable = await cfp.execute()
            print("--- Deliverable ---")
            print(deliverable[:500] + "..." if len(deliverable) > 500 else deliverable)


async def demo_with_risks():
    """Demo 3: Request risk analysis."""
    print()
    print("=" * 60)
    print("DEMO 3: Contract-First with Risk Analysis")
    print("=" * 60)
    print()

    async with ContractFirstPrompting(
        confidence_threshold=0.85,
        max_questions=5,
        digging_strategy=DiggingStrategy.BREADTH_FIRST,
    ) as cfp:
        rough_idea = "Build a real-time chat system with WebSocket support"
        print(f"User: {rough_idea}")
        print()

        await cfp.start(rough_idea)

        # Quick Q&A
        qa_pairs = [
            ("What programming language and frameworks?", "Python with FastAPI and WebSockets"),
            ("What problem does this solve?", "Enable real-time communication for web app"),
            ("Who will use/read/see this?", "Web developers"),
            ("Any performance requirements?", "Support 1000 concurrent connections"),
            ("How will we know this is good enough?", "Handles 1000 users with <100ms latency"),
        ]

        for _, answer in qa_pairs:
            question = await cfp.next_question()
            if question:
                print(f"Assistant: {question}")
                print(f"User: {answer}")
                print()
                await cfp.answer(answer)

        # Echo check
        echo = await cfp.echo_check()
        print("--- Echo Check ---")
        print(f"Assistant:\n{echo}")
        print()

        # Request risks
        print("User: risks")
        print()

        risks = await cfp.analyze_risks()
        print("--- Risk Analysis ---")
        print(risks.render())
        print()

        # Approve
        print("User: yes")
        await cfp.approve("yes")

        deliverable = await cfp.execute()
        print("--- Deliverable ---")
        print(deliverable[:500] + "..." if len(deliverable) > 500 else deliverable)


async def demo_interactive():
    """Demo 4: Interactive mode with real user input."""
    print()
    print("=" * 60)
    print("DEMO 4: Interactive Contract-First Prompting")
    print("=" * 60)
    print()
    print("This demo allows you to interact with the system.")
    print("Enter your rough idea, then answer questions one at a time.")
    print()

    async with ContractFirstPrompting(
        confidence_threshold=0.95,
        max_questions=15,
        digging_strategy=DiggingStrategy.ADAPTIVE,
    ) as cfp:
        # Get rough idea
        rough_idea = input("What do you need? > ")
        print()

        welcome = await cfp.start(rough_idea)
        print(f"Assistant: {welcome}")

        # Iterative questioning
        while not cfp.is_confident():
            question = await cfp.next_question()
            if not question:
                break

            print(f"\nAssistant: {question}")
            answer = input("> ")

            if answer.lower() == "skip":
                print("(Skipping this question)")
                answer = "N/A"

            await cfp.answer(answer)

        # Echo check
        print("\n--- Echo Check ---")
        echo = await cfp.echo_check()
        print(f"Assistant:\n{echo}\n")

        # Approval loop
        approved = False
        while not approved:
            response = input("Response (yes/edit/blueprint/risks): ")
            print()

            if response.lower() == "yes":
                approved = await cfp.approve("yes")
                break
            elif response.lower() == "edit":
                edit_request = input("What would you like to change? ")
                await cfp.approve("edit")
                # Show updated echo check
                echo = await cfp.echo_check()
                print(f"\nUpdated:\n{echo}\n")
            elif response.lower() == "blueprint":
                blueprint = await cfp.blueprint_view()
                print(blueprint.render())
                print()
            elif response.lower() == "risks":
                risks = await cfp.analyze_risks()
                print(risks.render())
                print()
            else:
                print("Invalid response. Use: yes, edit, blueprint, or risks")

        # Execute
        if approved:
            print("\n--- Building Deliverable ---")
            deliverable = await cfp.execute()
            print(deliverable)
            print()

            # Feedback
            helpful = input("\nWas this helpful? (yes/no): ")
            await cfp.reflect({"helpful": helpful.lower() == "yes"})


async def main():
    """Run all demos."""
    print("\n🎯 Contract-First Prompting Demos\n")
    print("Demonstrating clarity of intent through structured questioning.\n")

    # Run demos
    await demo_basic_usage()
    await demo_with_blueprint()
    await demo_with_risks()

    # Ask if user wants interactive demo
    print()
    print("=" * 60)
    run_interactive = input("Run interactive demo? (yes/no): ")

    if run_interactive.lower() == "yes":
        await demo_interactive()

    print()
    print("=" * 60)
    print("✅ Demos complete!")
    print()
    print("Key Takeaways:")
    print("1. Start with rough ideas - don't over-prepare")
    print("2. Answer one question at a time - trust the process")
    print("3. Use blueprint for complex work - review structure first")
    print("4. Use risks for high-stakes work - understand tradeoffs")
    print("5. Iterate with 'edit' - refine until contract is right")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
