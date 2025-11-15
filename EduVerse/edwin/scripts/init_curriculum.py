"""
Initialize EdWIN Curriculum

One-time script to load and save curriculum knowledge graph.

Usage:
    python EduVerse/edwin/scripts/init_curriculum.py
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from EduVerse.edwin import EdWINKnowledgeGraph


async def main():
    """Initialize curriculum knowledge graph"""

    print("=" * 60)
    print("EdWIN Curriculum Initialization")
    print("=" * 60)
    print()

    # Create knowledge graph
    kg = EdWINKnowledgeGraph()

    # Ingest curriculum
    stats = await kg.ingest_curriculum()

    # Display statistics
    print()
    print("=" * 60)
    print("Curriculum Statistics")
    print("=" * 60)
    print()
    print(f"  Total Objectives: {stats['objectives']}")
    print(f"  Total Edges: {stats['edges']}")
    print(f"  Total Subjects: {stats['subjects']}")
    print()

    # Show breakdown by subject
    from EduVerse.education.curriculum import Subject

    print("Breakdown by Subject:")
    for subject in Subject:
        objectives = kg.get_subject_objectives(subject)
        print(f"  - {subject.value.upper()}: {len(objectives)} objectives")

    print()

    # Show breakdown by grade
    print("Breakdown by Grade:")
    for grade in range(4, 13):
        objectives = kg.filter_by_grade(grade, buffer=0)
        print(f"  - Grade {grade}: {len(objectives)} objectives")

    print()

    # Save to file
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "edwin_curriculum_graph.json"
    kg.save(str(output_file))

    print()
    print("=" * 60)
    print(f"✅ Curriculum saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
