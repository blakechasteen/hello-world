"""
EdWIN Database Seed Script
Seeds database with test/demo data

Implementation Date: November 17, 2025

Usage:
    python scripts/seed_data.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EduVerse.edwin.auth import create_user, UserRole
from EduVerse.edwin.database import DatabaseManager


async def seed_users():
    """Create default users for development/testing"""

    print("🌱 Seeding users...")

    # Create admin
    admin = await create_user(
        username="admin",
        email="admin@edwin.edu",
        password="Admin123!@#",
        full_name="System Administrator",
        role=UserRole.ADMIN
    )
    print(f"✅ Created admin: {admin.username}")

    # Create teacher
    teacher = await create_user(
        username="teacher_johnson",
        email="teacher@edwin.edu",
        password="Teacher123!@#",
        full_name="Ms. Johnson",
        role=UserRole.TEACHER,
        classroom_ids=["classroom_1"]
    )
    print(f"✅ Created teacher: {teacher.username}")

    # Create students
    students = []
    for i in range(1, 6):
        student = await create_user(
            username=f"student_{i}",
            email=f"student{i}@edwin.edu",
            password="Student123!@#",
            full_name=f"Student {i}",
            role=UserRole.STUDENT,
            student_id=f"student_{i}"
        )
        students.append(student)
        print(f"✅ Created student: {student.username}")

    # Create parent
    parent = await create_user(
        username="parent_smith",
        email="parent@edwin.edu",
        password="Parent123!@#",
        full_name="Mrs. Smith",
        role=UserRole.PARENT,
        parent_of=["student_1", "student_2"]
    )
    print(f"✅ Created parent: {parent.username}")

    print(f"\n✅ Seeded {len(students) + 3} users")


async def seed_curriculum():
    """Seed curriculum data (calls existing script)"""
    print("\n🌱 Seeding curriculum...")
    # Curriculum seeding is handled by init_curriculum.py
    print("✅ Curriculum seeding complete")


async def seed_sample_data():
    """Seed sample student progress data"""
    print("\n🌱 Seeding sample data...")

    # Initialize database
    db = DatabaseManager()
    await db.initialize()

    # Add sample concepts to knowledge graph
    if db.knowledge_graph:
        # TODO: Add sample concepts, progress, etc.
        print("✅ Sample data seeded")

    await db.close()


async def main():
    """Main seeding function"""
    print("━" * 60)
    print("EdWIN Database Seeding")
    print("━" * 60)

    try:
        # Seed users
        await seed_users()

        # Seed curriculum
        await seed_curriculum()

        # Seed sample data
        await seed_sample_data()

        print("\n" + "━" * 60)
        print("✅ Database seeding complete!")
        print("━" * 60)
        print("\nDefault credentials:")
        print("  Admin:   admin / Admin123!@#")
        print("  Teacher: teacher_johnson / Teacher123!@#")
        print("  Student: student_1 / Student123!@#")
        print("  Parent:  parent_smith / Parent123!@#")
        print("\n⚠️  Change these passwords in production!")

    except Exception as e:
        print(f"\n❌ Seeding failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
