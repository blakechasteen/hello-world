"""
EdWIN Parent Portal Tests

Comprehensive test coverage for parent portal functionality.

**Implementation Date**: November 15, 2025

Usage:
    pytest EduVerse/edwin/tests/test_parent_portal.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from EduVerse.edwin.parent_portal import (
    ParentPortalAPI,
    ParentAccount,
    ChildLink,
    Notification,
    Message,
    Goal,
    NotificationType,
    NotificationChannel,
    MessageType,
    GoalType
)
from EduVerse.edwin.analytics import LearningAnalytics
from EduVerse.edwin.student_model import EdWINStudentModel
from EduVerse.edwin.curriculum_graph import EdWINKnowledgeGraph, create_curriculum_graph


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def curriculum_kg():
    """Create curriculum knowledge graph"""
    return await create_curriculum_graph()


@pytest.fixture
def analytics(curriculum_kg):
    """Create analytics engine"""
    return LearningAnalytics(curriculum_kg)


@pytest.fixture
def portal(analytics, curriculum_kg):
    """Create parent portal"""
    return ParentPortalAPI(
        analytics=analytics,
        curriculum_kg=curriculum_kg
    )


@pytest.fixture
async def parent(portal):
    """Create test parent account"""
    return await portal.register_parent(
        email="test@example.com",
        name="Test Parent",
        phone="+1234567890"
    )


@pytest.fixture
def student(curriculum_kg):
    """Create test student"""
    student = EdWINStudentModel(
        student_id="test_student",
        name="Test Student",
        grade=8
    )

    # Add some progress
    for i in range(5):
        student.update_mastery(
            objective_id=f"test_obj_{i}",
            success=True,
            confidence=0.8
        )

    return student


# ============================================================================
# Parent Management Tests
# ============================================================================

@pytest.mark.asyncio
async def test_register_parent(portal):
    """Test parent registration"""
    parent = await portal.register_parent(
        email="parent@example.com",
        name="Jane Doe",
        phone="+1234567890"
    )

    assert parent.email == "parent@example.com"
    assert parent.name == "Jane Doe"
    assert parent.phone == "+1234567890"
    assert len(parent.children) == 0
    assert parent.id.startswith("parent_")


@pytest.mark.asyncio
async def test_register_duplicate_email(portal):
    """Test duplicate email registration fails"""
    await portal.register_parent(
        email="duplicate@example.com",
        name="Parent 1"
    )

    with pytest.raises(Exception):  # Should raise HTTPException
        await portal.register_parent(
            email="duplicate@example.com",
            name="Parent 2"
        )


@pytest.mark.asyncio
async def test_link_child(portal, parent, student):
    """Test child linking"""
    link = await portal.link_child(
        parent_id=parent.id,
        student_id=student.student_id,
        relationship="parent",
        verification_code="verified"
    )

    assert link.parent_id == parent.id
    assert link.student_id == student.student_id
    assert link.relationship == "parent"
    assert link.verified == True
    assert student.student_id in parent.children


@pytest.mark.asyncio
async def test_unlink_child(portal, parent, student):
    """Test child unlinking"""
    # Link first
    await portal.link_child(
        parent_id=parent.id,
        student_id=student.student_id
    )

    # Unlink
    await portal.unlink_child(
        parent_id=parent.id,
        student_id=student.student_id
    )

    assert student.student_id not in parent.children


@pytest.mark.asyncio
async def test_get_parent_children(portal, parent, student):
    """Test getting parent's children"""
    await portal.link_child(
        parent_id=parent.id,
        student_id=student.student_id
    )

    children = await portal.get_parent_children(parent.id)

    assert len(children) == 1
    assert children[0].student_id == student.student_id


# ============================================================================
# Progress Report Tests
# ============================================================================

@pytest.mark.asyncio
async def test_daily_report(portal, student, analytics):
    """Test daily report generation"""
    # Track some interactions
    analytics.track_interaction(
        student_id=student.student_id,
        interaction_type="practice",
        objective_id="test_obj_1",
        success=True,
        confidence=0.8,
        xp_gained=25
    )

    report = await portal.get_daily_report(student.student_id)

    assert "student_id" in report
    assert "date" in report
    assert "objectives_attempted" in report
    assert "xp_earned" in report


@pytest.mark.asyncio
async def test_weekly_report(portal, student, analytics):
    """Test weekly report generation"""
    # Add some interactions
    for i in range(10):
        analytics.track_interaction(
            student_id=student.student_id,
            interaction_type="practice",
            objective_id=f"test_obj_{i}",
            success=True,
            confidence=0.8,
            xp_gained=25
        )

    report = await portal.get_weekly_report(student.student_id, student)

    assert "overall_progress" in report
    assert "subject_progress" in report
    assert "learning_velocity" in report


@pytest.mark.asyncio
async def test_monthly_report(portal, student):
    """Test monthly report generation"""
    report = await portal.get_monthly_report(student.student_id, student)

    assert "overall_progress" in report
    assert "subject_breakdown" in report
    assert "engagement_metrics" in report


# ============================================================================
# Notification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_send_notification(portal, parent, student):
    """Test sending notification"""
    notif = await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.ACHIEVEMENT,
        title="Test Achievement",
        message="Test message"
    )

    assert notif is not None
    assert notif.parent_id == parent.id
    assert notif.student_id == student.student_id
    assert notif.type == NotificationType.ACHIEVEMENT
    assert notif.title == "Test Achievement"


@pytest.mark.asyncio
async def test_get_notifications(portal, parent, student):
    """Test getting notifications"""
    # Send some notifications
    await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.ACHIEVEMENT,
        title="Notification 1",
        message="Message 1"
    )

    await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.MASTERY_UPDATE,
        title="Notification 2",
        message="Message 2"
    )

    notifications = await portal.get_notifications(parent.id)

    assert len(notifications) == 2


@pytest.mark.asyncio
async def test_mark_notification_read(portal, parent, student):
    """Test marking notification as read"""
    notif = await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.ACHIEVEMENT,
        title="Test",
        message="Test"
    )

    await portal.mark_notification_read(parent.id, notif.id)

    assert notif.read == True


@pytest.mark.asyncio
async def test_notification_preferences(portal, parent):
    """Test updating notification preferences"""
    preferences = {
        NotificationType.ACHIEVEMENT: [NotificationChannel.EMAIL],
        NotificationType.INTERVENTION_ALERT: [NotificationChannel.EMAIL, NotificationChannel.SMS]
    }

    await portal.update_notification_preferences(parent.id, preferences)

    assert NotificationChannel.EMAIL in parent.notification_preferences[NotificationType.ACHIEVEMENT]


# ============================================================================
# Messaging Tests
# ============================================================================

@pytest.mark.asyncio
async def test_send_message(portal, parent, student):
    """Test sending message"""
    message = await portal.send_message(
        sender_id=parent.id,
        sender_type="parent",
        recipient_id="teacher_001",
        recipient_type="teacher",
        student_id=student.student_id,
        subject="Test Subject",
        body="Test message body",
        message_type=MessageType.GENERAL_INQUIRY
    )

    assert message.sender_id == parent.id
    assert message.recipient_id == "teacher_001"
    assert message.subject == "Test Subject"


@pytest.mark.asyncio
async def test_get_messages(portal, parent, student):
    """Test getting messages"""
    # Send a message
    await portal.send_message(
        sender_id=parent.id,
        sender_type="parent",
        recipient_id="teacher_001",
        recipient_type="teacher",
        student_id=student.student_id,
        subject="Test",
        body="Test body"
    )

    messages = await portal.get_messages(parent.id)

    assert len(messages) == 1


@pytest.mark.asyncio
async def test_get_thread(portal, parent, student):
    """Test getting conversation thread"""
    # Send message
    msg1 = await portal.send_message(
        sender_id=parent.id,
        sender_type="parent",
        recipient_id="teacher_001",
        recipient_type="teacher",
        student_id=student.student_id,
        subject="Test",
        body="Message 1"
    )

    # Reply
    msg2 = await portal.send_message(
        sender_id="teacher_001",
        sender_type="teacher",
        recipient_id=parent.id,
        recipient_type="parent",
        student_id=student.student_id,
        subject="Re: Test",
        body="Message 2"
    )

    thread = await portal.get_thread(msg1.thread_id)

    assert len(thread) >= 1


# ============================================================================
# Goal Tracking Tests
# ============================================================================

@pytest.mark.asyncio
async def test_create_goal(portal, student, parent):
    """Test creating goal"""
    goal = await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.MASTERY,
        title="Master 10 Objectives",
        description="Complete 10 objectives",
        target_value=10,
        unit="objectives",
        deadline=datetime.utcnow() + timedelta(days=30),
        created_by=parent.id,
        created_by_role="parent"
    )

    assert goal.student_id == student.student_id
    assert goal.type == GoalType.MASTERY
    assert goal.title == "Master 10 Objectives"
    assert goal.target_value == 10


@pytest.mark.asyncio
async def test_update_goal_progress(portal, student, parent):
    """Test updating goal progress"""
    goal = await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.MASTERY,
        title="Test Goal",
        description="Test",
        target_value=10,
        unit="objectives",
        deadline=None,
        created_by=parent.id
    )

    await portal.update_goal_progress(
        goal_id=goal.id,
        student_id=student.student_id,
        current_value=5
    )

    assert goal.current_value == 5
    assert goal.progress_percentage == 50.0


@pytest.mark.asyncio
async def test_goal_completion(portal, student, parent):
    """Test goal completion"""
    goal = await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.MASTERY,
        title="Test Goal",
        description="Test",
        target_value=10,
        unit="objectives",
        deadline=None,
        created_by=parent.id
    )

    await portal.update_goal_progress(
        goal_id=goal.id,
        student_id=student.student_id,
        current_value=10
    )

    assert goal.completed == True
    assert goal.completed_at is not None


@pytest.mark.asyncio
async def test_get_goals(portal, student, parent):
    """Test getting goals"""
    # Create goals
    await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.MASTERY,
        title="Goal 1",
        description="Test",
        target_value=10,
        unit="objectives",
        deadline=None,
        created_by=parent.id
    )

    await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.STREAK,
        title="Goal 2",
        description="Test",
        target_value=30,
        unit="days",
        deadline=None,
        created_by=parent.id
    )

    goals = await portal.get_goals(student.student_id, active_only=True)

    assert len(goals) == 2


# ============================================================================
# Privacy Controls Tests
# ============================================================================

@pytest.mark.asyncio
async def test_update_privacy_settings(portal, parent, student):
    """Test updating privacy settings"""
    await portal.update_privacy_settings(
        parent_id=parent.id,
        student_id=student.student_id,
        settings={
            "consents": {
                "progress_tracking": True,
                "teacher_communication": False
            },
            "data_sharing_enabled": False,
            "profile_visibility": "private"
        }
    )

    privacy = await portal.get_privacy_settings(parent.id, student.student_id)

    assert privacy.data_sharing_enabled == False
    assert privacy.profile_visibility == "private"


@pytest.mark.asyncio
async def test_get_privacy_settings_default(portal, parent, student):
    """Test getting default privacy settings"""
    privacy = await portal.get_privacy_settings(parent.id, student.student_id)

    # Should have default consents
    assert len(privacy.consents) > 0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_parent_workflow(portal, analytics, curriculum_kg):
    """Test complete parent workflow"""
    # 1. Register parent
    parent = await portal.register_parent(
        email="workflow@example.com",
        name="Workflow Parent"
    )

    # 2. Create student
    student = EdWINStudentModel(
        student_id="workflow_student",
        name="Workflow Student",
        grade=8
    )

    # 3. Link child
    await portal.link_child(
        parent_id=parent.id,
        student_id=student.student_id
    )

    # 4. Add progress
    for i in range(3):
        student.update_mastery(
            objective_id=f"workflow_obj_{i}",
            success=True,
            confidence=0.8
        )

        analytics.track_interaction(
            student_id=student.student_id,
            interaction_type="practice",
            objective_id=f"workflow_obj_{i}",
            success=True,
            confidence=0.8,
            xp_gained=25
        )

    # 5. Get reports
    daily = await portal.get_daily_report(student.student_id)
    weekly = await portal.get_weekly_report(student.student_id, student)

    # 6. Send notification
    await portal.send_notification(
        parent_id=parent.id,
        student_id=student.student_id,
        type=NotificationType.ACHIEVEMENT,
        title="Test Achievement",
        message="Test"
    )

    # 7. Create goal
    goal = await portal.create_goal(
        student_id=student.student_id,
        type=GoalType.MASTERY,
        title="Workflow Goal",
        description="Test",
        target_value=5,
        unit="objectives",
        deadline=None,
        created_by=parent.id
    )

    # 8. Send message
    message = await portal.send_message(
        sender_id=parent.id,
        sender_type="parent",
        recipient_id="teacher_001",
        recipient_type="teacher",
        student_id=student.student_id,
        subject="Workflow Test",
        body="Test message"
    )

    # Verify everything
    assert len(parent.children) == 1
    assert len(student.mastered_objectives) == 3
    assert daily["objectives_mastered"] >= 0
    assert weekly["overall_progress"]["mastered_count"] >= 0
    assert goal.student_id == student.student_id
    assert message.sender_id == parent.id


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
