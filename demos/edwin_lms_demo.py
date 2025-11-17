"""
EdWIN LMS Integration Demo

Demonstrates complete LMS integration workflow:
1. Connect to Canvas/Google Classroom
2. Sync roster
3. Create assignments
4. Map to EdWIN objectives
5. Sync grades

Author: Agent D
Date: 2025-11-15
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EduVerse.edwin.integrations.oauth_manager import OAuth2Manager, OAuth2Provider
from EduVerse.edwin.integrations.canvas_integration import CanvasIntegration
from EduVerse.edwin.integrations.google_classroom import GoogleClassroomIntegration
from EduVerse.edwin.integrations.assignment_mapper import AssignmentMapper, GradingScheme
from EduVerse.edwin.integrations.gradebook_sync import GradebookSync, SyncStrategy, SyncDirection
from EduVerse.edwin.integrations.roster_manager import RosterManager


async def demo_canvas_integration():
    """Demo Canvas LMS integration"""
    print("=" * 60)
    print("EdWIN + Canvas LMS Integration Demo")
    print("=" * 60)
    print()

    # Initialize OAuth manager
    oauth_manager = OAuth2Manager()

    # Create Canvas integration
    canvas = CanvasIntegration(
        base_url="https://canvas.test.instructure.com",  # Demo Canvas
        client_id="demo_client_id",
        client_secret="demo_client_secret",
        oauth_manager=oauth_manager,
    )

    print("✓ Canvas integration initialized")
    print()

    # In production, you would:
    # 1. Generate OAuth URL
    # 2. User visits URL and authorizes
    # 3. Callback receives code
    # 4. Exchange code for token

    print("📚 Step 1: Connection Test")
    print("-" * 60)

    try:
        status = await canvas.test_connection()
        print(f"Status: {status['status']}")
        if status['status'] == 'connected':
            print(f"User: {status['user']['name']}")
            print(f"Email: {status['user']['email']}")
    except Exception as e:
        print(f"Note: Connection test requires valid OAuth token")
        print(f"In production, complete OAuth flow first")
    print()

    # Demo data
    print("📋 Step 2: Roster Sync")
    print("-" * 60)

    # Mock EdWIN students
    edwin_students = [
        {"id": "1", "name": "Alice Johnson", "email": "alice@school.edu"},
        {"id": "2", "name": "Bob Smith", "email": "bob@school.edu"},
        {"id": "3", "name": "Carol Williams", "email": "carol@school.edu"},
    ]

    roster_manager = RosterManager(canvas)

    print(f"EdWIN students: {len(edwin_students)}")
    print("Syncing with Canvas...")
    print("(In production, would fetch from Canvas API)")
    print()

    # Mock assignments
    print("📝 Step 3: Assignment Mapping")
    print("-" * 60)

    assignment_mapper = AssignmentMapper()

    # Create mapping
    mapping = assignment_mapper.create_mapping(
        lms_assignment_id="canvas_assign_123",
        lms_assignment_name="Fraction Addition Quiz",
        edwin_objectives=["fractions.add", "fractions.simplify"],
        grading_scheme=GradingScheme.MASTERY_PERCENTAGE,
        grading_weights={
            "fractions.add": 0.6,
            "fractions.simplify": 0.4,
        },
        max_points=100.0,
    )

    print(f"Assignment: {mapping.lms_assignment_name}")
    print(f"Objectives: {', '.join(mapping.edwin_objectives)}")
    print(f"Grading: {mapping.grading_scheme.value}")
    print(f"Weights: {mapping.grading_weights}")
    print()

    # Calculate grade
    student_scores = {
        "fractions.add": 0.85,  # 85% mastery
        "fractions.simplify": 0.92,  # 92% mastery
    }

    grade = assignment_mapper.calculate_grade(
        "canvas_assign_123",
        student_scores,
    )

    print(f"Student mastery scores:")
    for obj, score in student_scores.items():
        print(f"  - {obj}: {score * 100:.0f}%")
    print(f"Final grade: {grade:.1f}%")
    print()

    print("💾 Step 4: Grade Sync")
    print("-" * 60)

    gradebook = GradebookSync(canvas, assignment_mapper)

    print("Syncing grades to Canvas...")
    print("(In production, would POST to Canvas API)")

    # Simulate sync
    all_student_scores = {
        "1": {"fractions.add": 0.85, "fractions.simplify": 0.92},
        "2": {"fractions.add": 0.78, "fractions.simplify": 0.88},
        "3": {"fractions.add": 0.95, "fractions.simplify": 0.97},
    }

    print(f"Students to sync: {len(all_student_scores)}")
    print()

    for student_id, scores in all_student_scores.items():
        grade = assignment_mapper.calculate_grade("canvas_assign_123", scores)
        print(f"Student {student_id}: {grade:.1f}%")

    print()
    print("✓ Demo complete!")
    print()


async def demo_google_classroom_integration():
    """Demo Google Classroom integration"""
    print("=" * 60)
    print("EdWIN + Google Classroom Integration Demo")
    print("=" * 60)
    print()

    # Initialize OAuth manager
    oauth_manager = OAuth2Manager()

    # Create Google Classroom integration
    google = GoogleClassroomIntegration(
        client_id="demo_client_id.apps.googleusercontent.com",
        client_secret="demo_client_secret",
        oauth_manager=oauth_manager,
    )

    print("✓ Google Classroom integration initialized")
    print()

    print("📚 Step 1: Connection Test")
    print("-" * 60)

    try:
        status = await google.test_connection()
        print(f"Status: {status['status']}")
        if status['status'] == 'connected':
            print(f"User: {status['user']['name']}")
            print(f"Email: {status['user']['email']}")
    except Exception as e:
        print(f"Note: Connection test requires valid OAuth token")
        print(f"In production, complete OAuth flow first")
    print()

    print("📋 Step 2: Course and Roster")
    print("-" * 60)
    print("Courses:")
    print("  - Math 101 (25 students)")
    print("  - Science 202 (30 students)")
    print()

    print("📝 Step 3: Assignment Creation")
    print("-" * 60)

    from EduVerse.edwin.integrations.lms_base import Assignment
    from datetime import datetime, timedelta

    assignment = Assignment(
        id="",  # Will be assigned by Google
        name="Decimals Practice",
        description="Complete decimal addition and subtraction problems",
        due_date=datetime.now() + timedelta(days=7),
        points_possible=50,
        course_id="course_123",
    )

    print(f"Assignment: {assignment.name}")
    print(f"Points: {assignment.points_possible}")
    print(f"Due: {assignment.due_date.strftime('%Y-%m-%d')}")
    print("(In production, would POST to Google Classroom API)")
    print()

    print("✓ Demo complete!")
    print()


async def demo_complete_workflow():
    """Demo complete workflow from setup to grade sync"""
    print("=" * 60)
    print("Complete EdWIN LMS Integration Workflow")
    print("=" * 60)
    print()

    print("Workflow Steps:")
    print("1. ✓ Admin connects LMS (Canvas or Google Classroom)")
    print("2. ✓ OAuth flow completes")
    print("3. ✓ Teacher selects course")
    print("4. ✓ Roster syncs automatically")
    print("5. ✓ Teacher maps assignments to EdWIN objectives")
    print("6. ✓ Students complete EdWIN objectives")
    print("7. ✓ Grades sync back to LMS gradebook")
    print("8. ✓ Real-time updates via webhooks")
    print()

    print("Benefits:")
    print("✓ Single sign-on (no separate login for students)")
    print("✓ Automatic roster sync (students added/dropped)")
    print("✓ Seamless grade passback (mastery → percentage)")
    print("✓ Teacher workflow integration (no extra tools)")
    print("✓ School adoption made easy")
    print()


async def main():
    """Run all demos"""
    print("\n")
    await demo_canvas_integration()

    print("\n")
    await demo_google_classroom_integration()

    print("\n")
    await demo_complete_workflow()


if __name__ == "__main__":
    asyncio.run(main())
