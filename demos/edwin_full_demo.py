"""
EdWIN Full Feature Demo

Demonstrates all EdWIN features:
- Curriculum knowledge graph
- Student progress tracking
- Adaptive difficulty
- Safety guardrails
- Learning path generation

Usage:
    PYTHONPATH=. python demos/edwin_full_demo.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from EduVerse.edwin import (
    EdWIN,
    EdWINKnowledgeGraph,
    EdWINStudentModel,
    AdaptiveDifficultyEngine,
    EdWINSafetyLayer
)
from EduVerse.education.curriculum import Subject


async def demo_curriculum_graph():
    """Demo: Curriculum Knowledge Graph"""
    print("\n" + "=" * 70)
    print(" DEMO 1: Curriculum Knowledge Graph")
    print("=" * 70)
    print()

    kg = EdWINKnowledgeGraph()
    await kg.ingest_curriculum()

    # Show statistics
    stats = kg._get_stats()
    print(f"Total Objectives: {stats['objectives']}")
    print(f"Total Edges: {stats['edges']}")
    print()

    # Show math objectives
    print("Math Objectives (sample):")
    math_objs = kg.get_subject_objectives(Subject.MATH, grade_range=(8, 8))
    for obj in math_objs[:5]:
        print(f"  - {obj.title} (Grade {obj.grade_level}, {obj.bloom_level.name})")
    print()

    # Show learning path
    print("Learning Path Example:")
    path = kg.get_learning_path(
        "math.algebra.6.expressions",
        "math.algebra.10.systems"
    )
    if path:
        print("  From: Algebraic expressions")
        print("  To: Systems of equations")
        print("  Path:")
        for i, obj in enumerate(path, 1):
            print(f"    {i}. {obj.title} (Grade {obj.grade_level})")
    print()

    return kg


async def demo_student_model(kg):
    """Demo: Student Progress Tracking"""
    print("\n" + "=" * 70)
    print(" DEMO 2: Student Progress Tracking")
    print("=" * 70)
    print()

    # Create student
    student = EdWINStudentModel(
        student_id="demo_002",
        name="John Smith",
        grade=8
    )

    print(f"Student: {student.name}")
    print(f"Grade: {student.grade}")
    print()

    # Simulate learning sessions
    print("Simulating learning sessions:")
    sessions = [
        ("math.algebra.6.expressions", True, 0.85),
        ("math.algebra.7.equations", True, 0.90),
        ("math.algebra.8.linear_equations", False, 0.60),
        ("math.algebra.8.linear_equations", True, 0.95),
    ]

    for obj_id, success, confidence in sessions:
        obj = kg.get_objective(obj_id)
        if obj:
            mastered = student.update_mastery(obj_id, success, confidence)
            status = "✅ MASTERED!" if mastered else f"📈 {confidence:.0%}"
            print(f"  {obj.title}: {status}")
    print()

    # Show progress
    progress = student.get_progress_summary(kg)
    print(f"Mastered: {progress['mastered_count']} objectives")
    print(f"In Progress: {progress['in_progress_count']} objectives")
    print(f"Mastery: {progress['mastery_percentage']:.1f}%")
    print()

    # Show recommendations
    print("Recommended Next Objectives:")
    for i, obj in enumerate(progress['recommended'][:3], 1):
        print(f"  {i}. {obj.title} (Grade {obj.grade_level})")
    print()

    return student


async def demo_adaptive_difficulty(student, kg):
    """Demo: Adaptive Difficulty Engine"""
    print("\n" + "=" * 70)
    print(" DEMO 3: Adaptive Difficulty (Thompson Sampling)")
    print("=" * 70)
    print()

    engine = AdaptiveDifficultyEngine(student, kg)

    # Get available objectives
    available = kg.get_next_objectives(
        mastered_ids=list(student.mastered_objectives),
        grade_filter=student.grade
    )

    print(f"Available objectives: {len(available)}")
    print()

    # Select next objective
    print("Thompson Sampling Selection:")
    next_obj = engine.select_next_objective(available[:10])

    difficulty = engine.estimate_difficulty(next_obj.id)
    expected = engine.expected_reward(next_obj.id)

    print(f"  Selected: {next_obj.title}")
    print(f"  Difficulty: {difficulty:.2f}")
    print(f"  Expected Success: {expected:.1%}")
    print()

    # Simulate interaction
    print("Simulating student attempt...")
    success = True
    engagement = 0.85

    engine.update_after_interaction(next_obj.id, success, engagement)
    print(f"  Result: {'✅ Success' if success else '❌ Failed'}")
    print(f"  Engagement: {engagement:.0%}")
    print(f"  Thompson Sampling updated")
    print()

    return engine


async def demo_safety_layer():
    """Demo: K-12 Safety Guardrails"""
    print("\n" + "=" * 70)
    print(" DEMO 4: K-12 Safety Guardrails")
    print("=" * 70)
    print()

    safety = EdWINSafetyLayer()

    test_cases = [
        {
            "response": "Photosynthesis is how plants make food using sunlight.",
            "grade": 5,
            "expected": "✅ Safe"
        },
        {
            "response": "The mitochondrial electron transport chain couples oxidative phosphorylation via chemiosmotic gradient...",
            "grade": 5,
            "expected": "❌ Too complex"
        },
        {
            "response": "Simple explanation at 3rd grade level.",
            "grade": 8,
            "expected": "✅ Safe (below grade OK)"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['expected']}")
        print(f"  Response: {test['response'][:60]}...")
        print(f"  Grade: {test['grade']}")

        result = await safety.validate_response(
            query="Test question",
            response=test['response'],
            grade_level=test['grade']
        )

        print(f"  Reading Level: {result.reading_level:.1f}" if result.reading_level else "")
        print(f"  Result: {'✅ PASS' if result.allowed else '❌ FAIL'}")
        if not result.allowed:
            print(f"  Reason: {result.reason}")
        print()


async def demo_full_edwin():
    """Demo: Complete EdWIN System"""
    print("\n" + "=" * 70)
    print(" DEMO 5: Complete EdWIN System")
    print("=" * 70)
    print()

    # Create and initialize EdWIN
    edwin = EdWIN(
        student_id="demo_003",
        student_name="Alice Johnson",
        grade=8,
        llm_provider="anthropic"
    )

    await edwin.initialize()

    # Ask questions
    questions = [
        "What is photosynthesis?",
        "Explain the water cycle",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 70)

        answer = await edwin.teach(question)
        print(f"EdWIN: {answer[:200]}..." if len(answer) > 200 else f"EdWIN: {answer}")
        print()

    # Get suggestion
    suggestion = await edwin.suggest_next_lesson()
    print(f"Next Lesson: {suggestion['title']}")
    print(f"  Difficulty: {suggestion['difficulty']:.2f}")
    print(f"  Expected Success: {suggestion['expected_success']:.1%}")
    print()


async def main():
    """Run all demos"""

    print("\n")
    print("*" * 70)
    print(" EdWIN Full Feature Demo")
    print(" Demonstrating all components of the AI tutoring system")
    print("*" * 70)

    # Demo 1: Curriculum Graph
    kg = await demo_curriculum_graph()

    # Demo 2: Student Model
    student = await demo_student_model(kg)

    # Demo 3: Adaptive Difficulty
    engine = await demo_adaptive_difficulty(student, kg)

    # Demo 4: Safety Layer
    await demo_safety_layer()

    # Demo 5: Full EdWIN
    await demo_full_edwin()

    print("\n" + "=" * 70)
    print(" ✅ All Demos Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
