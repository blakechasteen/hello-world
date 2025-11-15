"""
EdWIN Simple Demo

Demonstrates basic EdWIN usage: creating a student and answering questions.

Usage:
    PYTHONPATH=. python demos/edwin_simple_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure proper path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EduVerse.edwin import EdWIN


async def main():
    """Simple EdWIN demo"""

    print("=" * 70)
    print(" EdWIN AI Tutor - Simple Demo")
    print("=" * 70)
    print()

    # Create EdWIN instance
    print("Creating EdWIN instance for Jane Doe (Grade 8)...")
    print()

    edwin = EdWIN(
        student_id="demo_student_001",
        student_name="Jane Doe",
        grade=8,
        llm_provider="anthropic",  # Change to "ollama" for local
        enable_safety=True
    )

    # Initialize (this loads curriculum and sets up all components)
    await edwin.initialize()

    # Sample questions
    questions = [
        "What is a linear equation?",
        "How do I solve 2x + 5 = 13?",
        "What's the Pythagorean theorem?",
    ]

    print("=" * 70)
    print(" Q&A Session")
    print("=" * 70)
    print()

    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 70)

        # Ask question
        answer = await edwin.teach(question)

        print(f"EdWIN: {answer}")
        print()

    # Get progress summary
    print("=" * 70)
    print(" Progress Summary")
    print("=" * 70)
    print()

    progress = edwin.get_progress_summary()

    print(f"Student: {progress['name']}")
    print(f"Grade: {progress['grade']}")
    print(f"Mastered: {progress['mastered_count']} / {progress['total_count']} objectives")
    print(f"Mastery: {progress['mastery_percentage']:.1f}%")
    print(f"In Progress: {progress['in_progress_count']}")
    print(f"Level: {progress['level']}")
    print(f"Total XP: {progress['total_xp']}")
    print()

    # Get next lesson suggestion
    print("=" * 70)
    print(" Next Lesson Suggestion")
    print("=" * 70)
    print()

    suggestion = await edwin.suggest_next_lesson()

    print(f"Recommended: {suggestion['title']}")
    print(f"Description: {suggestion['description']}")
    print(f"Grade Level: {suggestion['grade_level']}")
    print(f"Bloom Level: {suggestion['bloom_level']}")
    print(f"Estimated Time: {suggestion['estimated_time']} hours")
    print(f"Difficulty: {suggestion['difficulty']:.2f}")
    print(f"Expected Success: {suggestion['expected_success']:.1%}")
    print()

    # Save progress
    save_path = "./data/demo_student_001.json"
    edwin.save_progress(save_path)
    print(f"💾 Progress saved to: {save_path}")
    print()

    print("=" * 70)
    print(" Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
