"""
EdWIN Parent Portal Demo

Comprehensive demo showing:
- Parent registration + child linking
- Viewing progress reports
- Receiving notifications
- Messaging with teacher
- Setting and tracking goals

**Implementation Date**: November 15, 2025

Usage:
    PYTHONPATH=. python demos/edwin_parent_portal_demo.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EduVerse.edwin.parent_portal import (
    ParentPortalAPI,
    NotificationType,
    NotificationChannel,
    MessageType,
    GoalType
)
from EduVerse.edwin.notifications import NotificationEngine
from EduVerse.edwin.report_generator import ReportGenerator
from EduVerse.edwin.analytics import LearningAnalytics
from EduVerse.edwin.student_model import EdWINStudentModel
from EduVerse.edwin.curriculum_graph import create_curriculum_graph


async def main():
    """Run parent portal demo"""

    print("=" * 80)
    print("🎓 EdWIN Parent Portal - Comprehensive Demo")
    print("=" * 80)
    print()

    # ========================================================================
    # Setup
    # ========================================================================

    print("📚 Setting up EdWIN components...")
    print()

    # Create curriculum
    curriculum_kg = await create_curriculum_graph()

    # Create analytics
    analytics = LearningAnalytics(curriculum_kg)

    # Create parent portal
    portal = ParentPortalAPI(
        analytics=analytics,
        curriculum_kg=curriculum_kg
    )

    # Create notification engine
    notifications = NotificationEngine(
        smtp_server=None,  # Demo mode - no actual emails
        smtp_username=None,
        smtp_password=None
    )

    # Create report generator
    reports = ReportGenerator(
        analytics=analytics,
        curriculum_kg=curriculum_kg
    )

    print("✅ Components initialized\n")

    # ========================================================================
    # 1. Parent Registration
    # ========================================================================

    print("=" * 80)
    print("1️⃣  PARENT REGISTRATION")
    print("=" * 80)
    print()

    # Register parent
    parent = await portal.register_parent(
        email="jane.doe@example.com",
        name="Jane Doe",
        phone="+1234567890"
    )

    print(f"✅ Registered parent: {parent.name}")
    print(f"   Email: {parent.email}")
    print(f"   ID: {parent.id}")
    print(f"   Phone: {parent.phone}")
    print()

    # ========================================================================
    # 2. Child Linking
    # ========================================================================

    print("=" * 80)
    print("2️⃣  CHILD LINKING")
    print("=" * 80)
    print()

    # Create student model
    student = EdWINStudentModel(
        student_id="student_001",
        name="Emma Johnson",
        grade=8
    )

    # Simulate some progress
    for i in range(15):
        mastered = student.update_mastery(
            objective_id=f"math.algebra.8.obj{i}",
            success=True,
            confidence=0.85
        )

        # Track interaction
        analytics.track_interaction(
            student_id=student.student_id,
            interaction_type="practice",
            objective_id=f"math.algebra.8.obj{i}",
            success=True,
            confidence=0.85,
            mastered=mastered,
            xp_gained=25
        )

    # Update player stats
    student.player_model.stats.total_xp = 375
    student.player_model.stats.level = 5
    student.player_model.stats.streak_days = 7

    # Link child to parent
    link = await portal.link_child(
        parent_id=parent.id,
        student_id=student.student_id,
        relationship="parent",
        verification_code="verified"  # Simplified for demo
    )

    print(f"✅ Linked child: Emma Johnson")
    print(f"   Student ID: {link.student_id}")
    print(f"   Relationship: {link.relationship}")
    print(f"   Verified: {link.verified}")
    print()

    # ========================================================================
    # 3. Progress Reports
    # ========================================================================

    print("=" * 80)
    print("3️⃣  PROGRESS REPORTS")
    print("=" * 80)
    print()

    # Daily report
    daily = await portal.get_daily_report(student.student_id)
    print("📊 Daily Report:")
    print(f"   Objectives Attempted: {daily['objectives_attempted']}")
    print(f"   Objectives Mastered: {daily['objectives_mastered']}")
    print(f"   Questions Asked: {daily['questions_asked']}")
    print(f"   Time Spent: {daily['time_spent_minutes']} minutes")
    print(f"   XP Earned: {daily['xp_earned']}")
    print()

    # Weekly report
    weekly = await portal.get_weekly_report(student.student_id, student)
    print("📈 Weekly Report:")
    print(f"   Mastered: {weekly['overall_progress']['mastered_count']} objectives")
    print(f"   Level: {weekly['overall_progress']['level']}")
    print(f"   Total XP: {weekly['overall_progress']['total_xp']}")
    print(f"   Streak: {weekly['overall_progress']['streak_days']} days")
    print(f"   Velocity: {weekly['learning_velocity']['objectives_per_week']:.1f} obj/week")
    print(f"   Trend: {weekly['learning_velocity']['trend']}")
    print()

    # Generate HTML report
    html_report = await reports.generate_weekly_report_html(student)
    report_path = Path("demos/output/weekly_report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_report)
    print(f"💾 Saved HTML report to: {report_path}")
    print()

    # ========================================================================
    # 4. Notifications
    # ========================================================================

    print("=" * 80)
    print("4️⃣  NOTIFICATIONS")
    print("=" * 80)
    print()

    # Send achievement notification
    notif1 = await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.ACHIEVEMENT,
        title="🎉 New Achievement Unlocked!",
        message="Emma earned the 'Algebra Master' badge!",
        data={"achievement_id": "algebra_master", "xp_earned": 100}
    )
    print(f"✅ Sent achievement notification: {notif1.title}")

    # Send mastery update
    notif2 = await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.MASTERY_UPDATE,
        title="📚 Objective Mastered!",
        message="Emma mastered: Linear Equations",
        data={"objective_id": "math.algebra.8.linear_equations"}
    )
    print(f"✅ Sent mastery notification: {notif2.title}")

    # Send daily summary
    notif3 = await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.DAILY_SUMMARY,
        title="📊 Daily Summary Ready",
        message="Emma's learning summary for today is ready",
        data=daily
    )
    print(f"✅ Sent daily summary notification: {notif3.title}")
    print()

    # Get notifications
    notifications_list = await portal.get_notifications(parent.id, unread_only=True)
    print(f"📬 Unread Notifications: {len(notifications_list)}")
    for notif in notifications_list[:3]:
        print(f"   - {notif.title} ({notif.type.value})")
    print()

    # ========================================================================
    # 5. Parent-Teacher Messaging
    # ========================================================================

    print("=" * 80)
    print("5️⃣  PARENT-TEACHER MESSAGING")
    print("=" * 80)
    print()

    # Send message to teacher
    message = await portal.send_message(
        sender_id=parent.id,
        sender_type="parent",
        recipient_id="teacher_001",
        recipient_type="teacher",
        student_id=student.student_id,
        subject="Question about algebra homework",
        body="Hi Ms. Smith, I noticed Emma is working on linear equations. Are there any additional resources you'd recommend for practice?",
        message_type=MessageType.GENERAL_INQUIRY
    )

    print(f"✅ Sent message to teacher:")
    print(f"   Thread ID: {message.thread_id}")
    print(f"   Subject: {message.subject}")
    print(f"   Type: {message.message_type.value}")
    print()

    # Simulate teacher reply
    reply = await portal.send_message(
        sender_id="teacher_001",
        sender_type="teacher",
        recipient_id=parent.id,
        recipient_type="parent",
        student_id=student.student_id,
        subject="Re: Question about algebra homework",
        body="Hi Jane! Great question. I recommend the practice problems in Chapter 4, and Khan Academy has excellent video tutorials on linear equations. Emma is doing wonderfully!",
        message_type=MessageType.GENERAL_INQUIRY
    )

    print(f"✅ Received reply from teacher:")
    print(f"   From: teacher_001")
    print(f"   Message: {reply.body[:80]}...")
    print()

    # Get conversation thread
    thread = await portal.get_thread(message.thread_id)
    print(f"💬 Conversation Thread ({len(thread)} messages):")
    for msg in thread:
        sender = "Parent" if msg.sender_type == "parent" else "Teacher"
        print(f"   {sender}: {msg.body[:60]}...")
    print()

    # ========================================================================
    # 6. Goal Setting
    # ========================================================================

    print("=" * 80)
    print("6️⃣  GOAL SETTING & TRACKING")
    print("=" * 80)
    print()

    # Create mastery goal
    goal1 = await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.MASTERY,
        title="Master 10 Algebra Objectives",
        description="Complete and master 10 algebra objectives by end of month",
        target_value=10,
        unit="objectives",
        deadline=datetime.utcnow() + timedelta(days=30),
        created_by=parent.id,
        created_by_role="parent"
    )

    print(f"✅ Created goal: {goal1.title}")
    print(f"   Type: {goal1.type.value}")
    print(f"   Target: {goal1.target_value} {goal1.unit}")
    print(f"   Progress: {goal1.progress_percentage:.0f}%")
    print()

    # Update goal progress
    await portal.update_goal_progress(
        goal_id=goal1.id,
        student_id=student.student_id,
        current_value=7  # 7 objectives mastered
    )

    print(f"📈 Updated goal progress:")
    print(f"   Current: {goal1.current_value}/{goal1.target_value}")
    print(f"   Progress: {goal1.progress_percentage:.0f}%")
    print()

    # Create streak goal
    goal2 = await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.STREAK,
        title="Maintain 30-Day Streak",
        description="Study every day for 30 consecutive days",
        target_value=30,
        unit="days",
        deadline=None,
        created_by=parent.id,
        created_by_role="parent"
    )

    await portal.update_goal_progress(
        goal_id=goal2.id,
        student_id=student.student_id,
        current_value=7  # Current streak
    )

    print(f"✅ Created streak goal: {goal2.title}")
    print(f"   Current: {goal2.current_value}/{goal2.target_value} days")
    print(f"   Progress: {goal2.progress_percentage:.0f}%")
    print()

    # Get all goals
    goals = await portal.get_goals(student.student_id, active_only=True)
    print(f"🎯 Active Goals ({len(goals)}):")
    for goal in goals:
        print(f"   - {goal.title}: {goal.progress_percentage:.0f}%")
    print()

    # ========================================================================
    # 7. Privacy Settings
    # ========================================================================

    print("=" * 80)
    print("7️⃣  PRIVACY CONTROLS (FERPA Compliance)")
    print("=" * 80)
    print()

    # Update privacy settings
    await portal.update_privacy_settings(
        parent_id=parent.id,
        student_id=student.student_id,
        settings={
            "consents": {
                "progress_tracking": True,
                "teacher_communication": True,
                "aggregate_analytics": True,
                "third_party_integrations": False,
                "data_export": True
            },
            "data_sharing_enabled": True,
            "profile_visibility": "teachers_only",
            "communication_enabled": True
        }
    )

    # Get privacy settings
    privacy = await portal.get_privacy_settings(parent.id, student.student_id)

    print(f"✅ Privacy Settings Updated:")
    print(f"   Progress Tracking: {'✓' if privacy.consents.get('progress_tracking') else '✗'}")
    print(f"   Teacher Communication: {'✓' if privacy.consents.get('teacher_communication') else '✗'}")
    print(f"   Aggregate Analytics: {'✓' if privacy.consents.get('aggregate_analytics') else '✗'}")
    print(f"   Third-Party Integrations: {'✓' if privacy.consents.get('third_party_integrations') else '✗'}")
    print(f"   Profile Visibility: {privacy.profile_visibility}")
    print(f"   Data Sharing: {'Enabled' if privacy.data_sharing_enabled else 'Disabled'}")
    print()

    # ========================================================================
    # 8. Email Notifications (Demo)
    # ========================================================================

    print("=" * 80)
    print("8️⃣  EMAIL NOTIFICATIONS (Demo)")
    print("=" * 80)
    print()

    # Send achievement email
    await notifications.send_achievement_notification(
        parent_email=parent.email,
        student_name=student.name,
        achievement_title="Algebra Master",
        achievement_description="has mastered all basic algebra objectives!",
        xp_earned=100
    )
    print(f"📧 Sent achievement email to {parent.email}")

    # Send daily summary email
    await notifications.send_daily_summary(
        parent_email=parent.email,
        student_name=student.name,
        objectives_attempted=daily['objectives_attempted'],
        objectives_mastered=daily['objectives_mastered'],
        questions_asked=daily['questions_asked'],
        time_spent_minutes=daily['time_spent_minutes'],
        xp_earned=daily['xp_earned'],
        streak_days=student.player_model.stats.streak_days
    )
    print(f"📧 Sent daily summary email to {parent.email}")

    # Send goal completed email
    await notifications.send_goal_completed(
        parent_email=parent.email,
        student_name=student.name,
        goal_title=goal1.title,
        completion_percentage=100.0,
        days_to_complete=23
    )
    print(f"📧 Sent goal completion email to {parent.email}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("✅ DEMO COMPLETE - Summary")
    print("=" * 80)
    print()

    print(f"Parent Account:")
    print(f"  - Name: {parent.name}")
    print(f"  - Email: {parent.email}")
    print(f"  - Children: {len(parent.children)}")
    print()

    print(f"Student Progress:")
    print(f"  - Name: {student.name}")
    print(f"  - Objectives Mastered: {len(student.mastered_objectives)}")
    print(f"  - Total XP: {student.player_model.stats.total_xp}")
    print(f"  - Level: {student.player_model.stats.level}")
    print(f"  - Streak: {student.player_model.stats.streak_days} days")
    print()

    print(f"Portal Activity:")
    print(f"  - Notifications Sent: {len(notifications_list)}")
    print(f"  - Messages Exchanged: {len(thread)}")
    print(f"  - Active Goals: {len(goals)}")
    print(f"  - Reports Generated: 1 (weekly)")
    print()

    print(f"📁 Generated Files:")
    print(f"  - {report_path}")
    print()

    print("🎉 Parent Portal Demo Complete!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
