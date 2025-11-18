"""Seed initial data for LMS Orchestration

Run this script after migrations to populate the database with:
- Sample institution
- Admin user
- Sample plugins
- Sample course and lessons
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from passlib.hash import bcrypt


# Database connection string
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lms_user:lms_dev_password@localhost:5432/lms_dev"
)


async def seed_institutions(conn):
    """Create sample institution"""
    print("Seeding institutions...")

    institution_id = await conn.fetchval("""
        INSERT INTO institutions (name, domain, settings)
        VALUES ($1, $2, $3)
        ON CONFLICT (domain) DO UPDATE SET name = EXCLUDED.name
        RETURNING institution_id
    """, "Demo University", "demo.edu", {"timezone": "America/New_York"})

    print(f"  ✓ Created institution: {institution_id}")
    return institution_id


async def seed_users(conn, institution_id):
    """Create sample users"""
    print("Seeding users...")

    # Admin user
    admin_password = bcrypt.hash("admin123")
    admin_id = await conn.fetchval("""
        INSERT INTO users (institution_id, email, password_hash, first_name, last_name, role)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash
        RETURNING user_id
    """, institution_id, "admin@demo.edu", admin_password, "Admin", "User", "admin")
    print(f"  ✓ Created admin: {admin_id}")

    # Instructor
    instructor_password = bcrypt.hash("instructor123")
    instructor_id = await conn.fetchval("""
        INSERT INTO users (institution_id, email, password_hash, first_name, last_name, role)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash
        RETURNING user_id
    """, institution_id, "instructor@demo.edu", instructor_password, "Jane", "Smith", "instructor")
    print(f"  ✓ Created instructor: {instructor_id}")

    # Sample students
    student_ids = []
    for i in range(1, 6):
        student_password = bcrypt.hash(f"student{i}123")
        student_id = await conn.fetchval("""
            INSERT INTO users (institution_id, email, password_hash, first_name, last_name, role)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash
            RETURNING user_id
        """, institution_id, f"student{i}@demo.edu", student_password, f"Student", f"{i}", "student")
        student_ids.append(student_id)
    print(f"  ✓ Created {len(student_ids)} students")

    return admin_id, instructor_id, student_ids


async def seed_plugins(conn):
    """Register sample plugins"""
    print("Seeding plugins...")

    plugins = [
        {
            "plugin_id": "simple-quiz",
            "name": "Simple Quiz",
            "version": "1.0.0",
            "category": "assessment",
            "description": "Basic quiz plugin with multiple choice, true/false, and short answer",
            "author": "LMS Team",
            "permissions": ["database:read", "database:write", "knowledge_graph:write"],
            "hooks": ["after_assessment_submit", "before_assessment_render"],
            "status": "active"
        },
        {
            "plugin_id": "peer-review-system",
            "name": "Peer Review System",
            "version": "1.0.0",
            "category": "assessment",
            "description": "Comprehensive peer review with calibration, quality scoring, and dispute resolution",
            "author": "LMS Team",
            "permissions": ["database:read", "database:write", "knowledge_graph:write", "email:send"],
            "hooks": ["after_assessment_submit", "on_review_complete", "on_review_disputed"],
            "status": "active"
        },
        {
            "plugin_id": "video-player",
            "name": "Interactive Video Player",
            "version": "1.0.0",
            "category": "content",
            "description": "Video player with embedded quizzes and analytics",
            "author": "LMS Team",
            "permissions": ["database:read", "events:publish"],
            "hooks": ["before_lesson_render", "on_video_progress"],
            "status": "active"
        },
        {
            "plugin_id": "analytics-dashboard",
            "name": "Analytics Dashboard",
            "version": "1.0.0",
            "category": "analytics",
            "description": "Student and course analytics with visualizations",
            "author": "LMS Team",
            "permissions": ["database:read", "analytics:read"],
            "hooks": ["on_student_dashboard", "on_instructor_dashboard"],
            "status": "active"
        },
    ]

    for plugin in plugins:
        await conn.execute("""
            INSERT INTO plugins (
                plugin_id, name, version, category, description, author,
                permissions, hooks, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (plugin_id, version) DO UPDATE
            SET name = EXCLUDED.name, status = EXCLUDED.status
        """,
            plugin["plugin_id"],
            plugin["name"],
            plugin["version"],
            plugin["category"],
            plugin["description"],
            plugin["author"],
            plugin["permissions"],
            plugin["hooks"],
            plugin["status"]
        )

    print(f"  ✓ Registered {len(plugins)} plugins")


async def seed_course(conn, institution_id, instructor_id, student_ids):
    """Create sample course with lessons"""
    print("Seeding courses...")

    # Create course
    course_id = await conn.fetchval("""
        INSERT INTO courses (
            institution_id, code, name, description, semester, year, status, created_by
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING course_id
    """,
        institution_id,
        "CS101",
        "Introduction to Computer Science",
        "Fundamental concepts of programming and computational thinking",
        "Fall",
        2025,
        "active",
        instructor_id
    )
    print(f"  ✓ Created course: {course_id}")

    # Enroll instructor
    await conn.execute("""
        INSERT INTO course_enrollments (course_id, user_id, role)
        VALUES ($1, $2, $3)
        ON CONFLICT (course_id, user_id) DO NOTHING
    """, course_id, instructor_id, "instructor")

    # Enroll students
    for student_id in student_ids:
        await conn.execute("""
            INSERT INTO course_enrollments (course_id, user_id, role)
            VALUES ($1, $2, $3)
            ON CONFLICT (course_id, user_id) DO NOTHING
        """, course_id, student_id, "student")
    print(f"  ✓ Enrolled {len(student_ids)} students")

    # Create lessons
    lessons = [
        {
            "title": "Welcome to Programming",
            "description": "Introduction to the course and basic concepts",
            "theme": "lecture",
            "concept": "programming_basics",
            "order_index": 1
        },
        {
            "title": "Variables and Data Types",
            "description": "Learn about variables, integers, floats, and strings",
            "theme": "flipped",
            "concept": "variables",
            "prerequisites": ["programming_basics"],
            "order_index": 2
        },
        {
            "title": "Control Flow: Conditionals",
            "description": "If statements, else clauses, and boolean logic",
            "theme": "socratic",
            "concept": "conditionals",
            "prerequisites": ["variables"],
            "order_index": 3
        },
        {
            "title": "Build a Calculator",
            "description": "Project: Create a working calculator program",
            "theme": "project",
            "concept": "basic_calculator",
            "prerequisites": ["conditionals"],
            "order_index": 4
        },
    ]

    for lesson in lessons:
        await conn.execute("""
            INSERT INTO lessons (
                course_id, title, description, theme, concept, prerequisites,
                order_index, status, created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
            course_id,
            lesson["title"],
            lesson["description"],
            lesson["theme"],
            lesson["concept"],
            lesson.get("prerequisites", []),
            lesson["order_index"],
            "published",
            instructor_id
        )

    print(f"  ✓ Created {len(lessons)} lessons")

    return course_id


async def seed_assessments(conn, course_id, instructor_id):
    """Create sample assessments"""
    print("Seeding assessments...")

    # Quiz assessment
    quiz_id = await conn.fetchval("""
        INSERT INTO assessments (
            course_id, title, description, assessment_type, plugin_id,
            concept, max_score, passing_score, attempts_allowed,
            due_date, status, created_by
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING assessment_id
    """,
        course_id,
        "Variables Quiz",
        "Test your understanding of variables and data types",
        "quiz",
        "simple-quiz",
        "variables",
        100.0,
        70.0,
        3,
        datetime.now() + timedelta(days=7),
        "published",
        instructor_id
    )
    print(f"  ✓ Created quiz: {quiz_id}")

    # Peer review assignment
    peer_review_id = await conn.fetchval("""
        INSERT INTO assessments (
            course_id, title, description, instructions, assessment_type, plugin_id,
            concept, max_score, due_date, status, created_by
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING assessment_id
    """,
        course_id,
        "Essay: The Future of Programming",
        "Write a 500-word essay on programming paradigms",
        "Write clearly, cite sources, and provide concrete examples",
        "peer_review",
        "peer-review-system",
        "programming_paradigms",
        100.0,
        datetime.now() + timedelta(days=14),
        "published",
        instructor_id
    )

    # Create peer review configuration
    rubric = {
        "criteria": [
            {
                "criterion_id": "c1",
                "name": "Clarity",
                "description": "Writing is clear and well-organized",
                "max_score": 30.0
            },
            {
                "criterion_id": "c2",
                "name": "Technical Accuracy",
                "description": "Content is technically correct",
                "max_score": 40.0
            },
            {
                "criterion_id": "c3",
                "name": "Examples",
                "description": "Provides concrete examples",
                "max_score": 30.0
            }
        ]
    }

    await conn.execute("""
        INSERT INTO peer_review_assignments (
            assignment_id, review_type, rubric, reviews_per_submission,
            enable_calibration, review_deadline, dispute_deadline
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    """,
        peer_review_id,
        "double_blind",
        rubric,
        3,
        True,
        datetime.now() + timedelta(days=21),
        datetime.now() + timedelta(days=24)
    )
    print(f"  ✓ Created peer review assignment: {peer_review_id}")

    return quiz_id, peer_review_id


async def main():
    """Run all seed functions"""
    print("=" * 60)
    print("LMS Orchestration - Database Seeding")
    print("=" * 60)
    print()

    # Connect to database
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        # Seed data
        institution_id = await seed_institutions(conn)
        admin_id, instructor_id, student_ids = await seed_users(conn, institution_id)
        await seed_plugins(conn)
        course_id = await seed_course(conn, institution_id, instructor_id, student_ids)
        quiz_id, peer_review_id = await seed_assessments(conn, course_id, instructor_id)

        print()
        print("=" * 60)
        print("Seeding complete!")
        print("=" * 60)
        print()
        print("Login credentials:")
        print("  Admin:      admin@demo.edu / admin123")
        print("  Instructor: instructor@demo.edu / instructor123")
        print("  Students:   student1@demo.edu / student1123 (through student5)")
        print()
        print(f"Sample course ID: {course_id}")
        print(f"Quiz ID: {quiz_id}")
        print(f"Peer Review ID: {peer_review_id}")
        print()

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
