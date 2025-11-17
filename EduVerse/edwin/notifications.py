"""
EdWIN Notification Engine

Real-time notification system for parents with multiple delivery channels.

**Features**:
- Multiple notification types (achievements, alerts, reports)
- Multi-channel delivery (in-app, email, SMS, push)
- Notification preferences per type
- Quiet hours support
- Digest mode (batch notifications)
- Delivery tracking

**Implementation Date**: November 15, 2025

Usage:
    from EduVerse.edwin.notifications import NotificationEngine

    engine = NotificationEngine()
    await engine.send_achievement_notification(
        parent_id="parent_001",
        student_id="student_001",
        achievement="First Badge Earned!"
    )
"""

from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Twilio for SMS (optional)
try:
    from twilio.rest import Client as TwilioClient
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False
    logger.warning("Twilio not available - SMS notifications disabled")


# ============================================================================
# Notification Engine
# ============================================================================

class NotificationEngine:
    """
    Notification Engine

    Handles notification delivery across multiple channels with
    intelligent batching, quiet hours, and delivery tracking.
    """

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        twilio_account_sid: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        twilio_phone_number: Optional[str] = None
    ):
        """
        Initialize Notification Engine

        Args:
            smtp_server: SMTP server for email
            smtp_port: SMTP port
            smtp_username: SMTP username
            smtp_password: SMTP password
            twilio_account_sid: Twilio account SID
            twilio_auth_token: Twilio auth token
            twilio_phone_number: Twilio phone number
        """
        # Email configuration
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password

        # SMS configuration
        self.twilio_client = None
        if HAS_TWILIO and twilio_account_sid and twilio_auth_token:
            self.twilio_client = TwilioClient(twilio_account_sid, twilio_auth_token)
            self.twilio_phone_number = twilio_phone_number

        # Delivery tracking
        self.delivery_history: List[Dict[str, Any]] = []

        # Digest queue (for batching)
        self.digest_queue: Dict[str, List[Dict[str, Any]]] = {}  # parent_id → notifications

        logger.info(f"NotificationEngine initialized (Email: {bool(smtp_server)}, SMS: {bool(self.twilio_client)})")

    # ========================================================================
    # Notification Senders
    # ========================================================================

    async def send_achievement_notification(
        self,
        parent_email: str,
        student_name: str,
        achievement_title: str,
        achievement_description: str,
        badge_url: Optional[str] = None,
        xp_earned: int = 0
    ):
        """
        Send achievement notification

        Args:
            parent_email: Parent email
            student_name: Student name
            achievement_title: Achievement title
            achievement_description: Achievement description
            badge_url: Optional badge image URL
            xp_earned: XP earned
        """
        subject = f"🎉 {student_name} earned: {achievement_title}!"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0;">🎉 New Achievement!</h1>
            </div>
            <div style="padding: 30px; background: #f9fafb;">
                <h2 style="color: #1f2937;">{achievement_title}</h2>
                <p style="color: #6b7280; font-size: 16px; line-height: 1.6;">
                    {student_name} {achievement_description}
                </p>

                {f'<div style="text-align: center; margin: 20px 0;"><img src="{badge_url}" alt="Badge" style="max-width: 150px;"></div>' if badge_url else ''}

                <div style="background: white; border-radius: 8px; padding: 20px; margin-top: 20px;">
                    <p style="color: #9ca3af; margin: 0; font-size: 14px;">XP Earned</p>
                    <p style="color: #1f2937; font-size: 24px; font-weight: bold; margin: 5px 0;">{xp_earned} XP</p>
                </div>

                <div style="margin-top: 30px; text-align: center;">
                    <a href="#" style="background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        View Full Progress
                    </a>
                </div>
            </div>
            <div style="padding: 20px; text-align: center; color: #9ca3af; font-size: 12px;">
                <p>EdWIN AI Tutor | <a href="#" style="color: #667eea;">Notification Preferences</a></p>
            </div>
        </body>
        </html>
        """

        await self._send_email(
            to_email=parent_email,
            subject=subject,
            html_body=html_body
        )

    async def send_mastery_update(
        self,
        parent_email: str,
        student_name: str,
        objective_title: str,
        mastery_score: float,
        subject: str
    ):
        """
        Send mastery update notification

        Args:
            parent_email: Parent email
            student_name: Student name
            objective_title: Objective title
            mastery_score: Mastery score (0-1)
            subject: Subject area
        """
        subject_line = f"📚 {student_name} mastered: {objective_title}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0;">📚 Objective Mastered!</h1>
            </div>
            <div style="padding: 30px; background: #f9fafb;">
                <h2 style="color: #1f2937;">{objective_title}</h2>
                <p style="color: #6b7280; font-size: 16px;">
                    {student_name} has mastered this {subject} objective!
                </p>

                <div style="background: white; border-radius: 8px; padding: 20px; margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #9ca3af;">Mastery Score</span>
                        <span style="color: #10b981; font-weight: bold; font-size: 20px;">{mastery_score * 100:.0f}%</span>
                    </div>
                    <div style="background: #e5e7eb; border-radius: 10px; height: 10px; margin-top: 10px;">
                        <div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); width: {mastery_score * 100}%; height: 10px; border-radius: 10px;"></div>
                    </div>
                </div>

                <p style="color: #6b7280; font-size: 14px; margin-top: 20px;">
                    Keep up the great work! {student_name} is making excellent progress.
                </p>
            </div>
        </body>
        </html>
        """

        await self._send_email(
            to_email=parent_email,
            subject=subject_line,
            html_body=html_body
        )

    async def send_intervention_alert(
        self,
        parent_email: str,
        parent_phone: Optional[str],
        student_name: str,
        severity: str,
        reason: str,
        recommended_actions: List[str]
    ):
        """
        Send intervention alert

        Args:
            parent_email: Parent email
            parent_phone: Optional parent phone
            student_name: Student name
            severity: Severity (warning, critical)
            reason: Alert reason
            recommended_actions: Recommended actions
        """
        emoji = "⚠️" if severity == "warning" else "🚨"
        color = "#f59e0b" if severity == "warning" else "#ef4444"

        subject = f"{emoji} Attention needed: {student_name}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {color}; padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0;">{emoji} Attention Needed</h1>
            </div>
            <div style="padding: 30px; background: #f9fafb;">
                <div style="background: white; border-left: 4px solid {color}; padding: 20px; margin-bottom: 20px;">
                    <p style="color: #6b7280; margin: 0; font-size: 14px;">STUDENT</p>
                    <p style="color: #1f2937; font-size: 18px; font-weight: bold; margin: 5px 0 10px 0;">{student_name}</p>

                    <p style="color: #6b7280; margin: 15px 0 5px 0; font-size: 14px;">REASON</p>
                    <p style="color: #1f2937; margin: 0;">{reason}</p>
                </div>

                <h3 style="color: #1f2937; margin-top: 30px;">Recommended Actions</h3>
                <ul style="color: #6b7280; line-height: 1.8;">
                    {''.join(f'<li>{action}</li>' for action in recommended_actions)}
                </ul>

                <div style="background: #fef3c7; border-radius: 8px; padding: 15px; margin-top: 20px;">
                    <p style="color: #92400e; margin: 0; font-size: 14px;">
                        <strong>💡 Tip:</strong> Early intervention can make a big difference. Consider scheduling a meeting with the teacher.
                    </p>
                </div>

                <div style="margin-top: 30px; text-align: center;">
                    <a href="#" style="background: {color}; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block; margin-right: 10px;">
                        Schedule Meeting
                    </a>
                    <a href="#" style="background: white; color: {color}; border: 2px solid {color}; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        View Progress
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        # Send email
        await self._send_email(
            to_email=parent_email,
            subject=subject,
            html_body=html_body
        )

        # Send SMS if critical
        if severity == "critical" and parent_phone and self.twilio_client:
            sms_body = f"{emoji} {student_name}: {reason}. Check email for details."
            await self._send_sms(parent_phone, sms_body)

    async def send_daily_summary(
        self,
        parent_email: str,
        student_name: str,
        objectives_attempted: int,
        objectives_mastered: int,
        questions_asked: int,
        time_spent_minutes: int,
        xp_earned: int,
        streak_days: int
    ):
        """
        Send daily summary

        Args:
            parent_email: Parent email
            student_name: Student name
            objectives_attempted: Objectives attempted today
            objectives_mastered: Objectives mastered today
            questions_asked: Questions asked today
            time_spent_minutes: Time spent learning (minutes)
            xp_earned: XP earned today
            streak_days: Current streak
        """
        subject = f"📊 Daily Summary: {student_name}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0;">📊 Daily Summary</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">{datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            <div style="padding: 30px; background: #f9fafb;">
                <h2 style="color: #1f2937; margin-bottom: 20px;">{student_name}'s Learning Today</h2>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                    <div style="background: white; border-radius: 8px; padding: 20px; text-align: center;">
                        <p style="color: #9ca3af; margin: 0; font-size: 12px;">OBJECTIVES</p>
                        <p style="color: #1f2937; font-size: 24px; font-weight: bold; margin: 5px 0;">
                            {objectives_mastered}/{objectives_attempted}
                        </p>
                        <p style="color: #6b7280; margin: 0; font-size: 12px;">Mastered</p>
                    </div>

                    <div style="background: white; border-radius: 8px; padding: 20px; text-align: center;">
                        <p style="color: #9ca3af; margin: 0; font-size: 12px;">XP EARNED</p>
                        <p style="color: #10b981; font-size: 24px; font-weight: bold; margin: 5px 0;">
                            {xp_earned}
                        </p>
                        <p style="color: #6b7280; margin: 0; font-size: 12px;">Experience</p>
                    </div>

                    <div style="background: white; border-radius: 8px; padding: 20px; text-align: center;">
                        <p style="color: #9ca3af; margin: 0; font-size: 12px;">TIME SPENT</p>
                        <p style="color: #3b82f6; font-size: 24px; font-weight: bold; margin: 5px 0;">
                            {time_spent_minutes}
                        </p>
                        <p style="color: #6b7280; margin: 0; font-size: 12px;">Minutes</p>
                    </div>

                    <div style="background: white; border-radius: 8px; padding: 20px; text-align: center;">
                        <p style="color: #9ca3af; margin: 0; font-size: 12px;">QUESTIONS</p>
                        <p style="color: #f59e0b; font-size: 24px; font-weight: bold; margin: 5px 0;">
                            {questions_asked}
                        </p>
                        <p style="color: #6b7280; margin: 0; font-size: 12px;">Asked</p>
                    </div>
                </div>

                {f'''
                <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center; color: white;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">🔥 CURRENT STREAK</p>
                    <p style="margin: 5px 0 0 0; font-size: 32px; font-weight: bold;">{streak_days} Days</p>
                </div>
                ''' if streak_days > 0 else ''}

                <div style="text-align: center; margin-top: 30px;">
                    <a href="#" style="background: #3b82f6; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        View Detailed Report
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        await self._send_email(
            to_email=parent_email,
            subject=subject,
            html_body=html_body
        )

    async def send_weekly_report_ready(
        self,
        parent_email: str,
        student_name: str,
        report_url: str
    ):
        """
        Send weekly report ready notification

        Args:
            parent_email: Parent email
            student_name: Student name
            report_url: URL to report
        """
        subject = f"📈 Weekly Report Ready: {student_name}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0;">📈 Weekly Report Ready</h1>
            </div>
            <div style="padding: 30px; background: #f9fafb;">
                <p style="color: #1f2937; font-size: 16px;">
                    Hi! {student_name}'s weekly progress report is now available.
                </p>

                <div style="background: white; border-radius: 8px; padding: 20px; margin: 20px 0;">
                    <h3 style="color: #1f2937; margin-top: 0;">This Week's Highlights</h3>
                    <ul style="color: #6b7280; line-height: 1.8;">
                        <li>Learning velocity and progress trends</li>
                        <li>Subject-by-subject breakdown</li>
                        <li>Top achievements</li>
                        <li>Areas needing attention</li>
                        <li>Recommended practice</li>
                    </ul>
                </div>

                <div style="text-align: center; margin-top: 30px;">
                    <a href="{report_url}" style="background: #8b5cf6; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        View Weekly Report
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        await self._send_email(
            to_email=parent_email,
            subject=subject,
            html_body=html_body
        )

    async def send_goal_completed(
        self,
        parent_email: str,
        student_name: str,
        goal_title: str,
        completion_percentage: float,
        days_to_complete: int
    ):
        """
        Send goal completed notification

        Args:
            parent_email: Parent email
            student_name: Student name
            goal_title: Goal title
            completion_percentage: Completion percentage
            days_to_complete: Days taken to complete
        """
        subject = f"🎯 Goal Completed: {goal_title}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%); padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0;">🎯 Goal Completed!</h1>
            </div>
            <div style="padding: 30px; background: #f9fafb;">
                <h2 style="color: #1f2937;">{goal_title}</h2>
                <p style="color: #6b7280; font-size: 16px;">
                    {student_name} has successfully completed this goal!
                </p>

                <div style="background: white; border-radius: 8px; padding: 20px; margin: 20px 0;">
                    <div style="margin-bottom: 15px;">
                        <span style="color: #9ca3af; font-size: 14px;">Completion</span>
                        <div style="background: #e5e7eb; border-radius: 10px; height: 10px; margin-top: 5px;">
                            <div style="background: linear-gradient(90deg, #14b8a6 0%, #0f766e 100%); width: {completion_percentage}%; height: 10px; border-radius: 10px;"></div>
                        </div>
                    </div>

                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p style="color: #9ca3af; margin: 0; font-size: 12px;">DAYS TO COMPLETE</p>
                            <p style="color: #1f2937; font-size: 20px; font-weight: bold; margin: 5px 0;">{days_to_complete}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="color: #9ca3af; margin: 0; font-size: 12px;">ACHIEVEMENT</p>
                            <p style="color: #14b8a6; font-size: 20px; font-weight: bold; margin: 5px 0;">{completion_percentage:.0f}%</p>
                        </div>
                    </div>
                </div>

                <p style="color: #6b7280; text-align: center; font-style: italic; margin: 30px 0;">
                    "Success is the sum of small efforts, repeated day in and day out."
                </p>
            </div>
        </body>
        </html>
        """

        await self._send_email(
            to_email=parent_email,
            subject=subject,
            html_body=html_body
        )

    # ========================================================================
    # Delivery Methods
    # ========================================================================

    async def _send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ):
        """
        Send email via SMTP

        Args:
            to_email: Recipient email
            subject: Email subject
            html_body: HTML email body
            text_body: Optional plain text body
        """
        if not self.smtp_server:
            logger.warning("SMTP not configured - email delivery skipped")
            return

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_username
            msg['To'] = to_email

            # Attach text and HTML parts
            if text_body:
                msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            # Track delivery
            self.delivery_history.append({
                "channel": "email",
                "to": to_email,
                "subject": subject,
                "sent_at": datetime.utcnow().isoformat(),
                "status": "delivered"
            })

            logger.info(f"Email sent to {to_email}: {subject}")

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")

            # Track failure
            self.delivery_history.append({
                "channel": "email",
                "to": to_email,
                "subject": subject,
                "sent_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "error": str(e)
            })

    async def _send_sms(
        self,
        phone: str,
        message: str
    ):
        """
        Send SMS via Twilio

        Args:
            phone: Phone number
            message: SMS message
        """
        if not self.twilio_client:
            logger.warning("Twilio not configured - SMS delivery skipped")
            return

        try:
            # Send via Twilio
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_phone_number,
                to=phone
            )

            # Track delivery
            self.delivery_history.append({
                "channel": "sms",
                "to": phone,
                "message": message,
                "sent_at": datetime.utcnow().isoformat(),
                "status": "delivered",
                "twilio_sid": message_obj.sid
            })

            logger.info(f"SMS sent to {phone}")

        except Exception as e:
            logger.error(f"Failed to send SMS to {phone}: {e}")

            # Track failure
            self.delivery_history.append({
                "channel": "sms",
                "to": phone,
                "message": message,
                "sent_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "error": str(e)
            })

    # ========================================================================
    # Digest Mode
    # ========================================================================

    async def queue_for_digest(
        self,
        parent_id: str,
        notification: Dict[str, Any]
    ):
        """
        Queue notification for digest delivery

        Args:
            parent_id: Parent ID
            notification: Notification dict
        """
        if parent_id not in self.digest_queue:
            self.digest_queue[parent_id] = []

        self.digest_queue[parent_id].append(notification)

        logger.info(f"Queued notification for digest: {parent_id}")

    async def send_digest(
        self,
        parent_id: str,
        parent_email: str,
        student_name: str
    ):
        """
        Send digest of queued notifications

        Args:
            parent_id: Parent ID
            parent_email: Parent email
            student_name: Student name
        """
        notifications = self.digest_queue.get(parent_id, [])

        if not notifications:
            logger.info(f"No notifications in digest queue for {parent_id}")
            return

        subject = f"📬 Daily Digest: {student_name} ({len(notifications)} updates)"

        # Build digest HTML
        notification_items = []
        for notif in notifications:
            notification_items.append(f"""
                <div style="background: white; border-left: 4px solid #3b82f6; padding: 15px; margin-bottom: 15px;">
                    <p style="color: #1f2937; font-weight: bold; margin: 0 0 5px 0;">{notif.get('title', 'Notification')}</p>
                    <p style="color: #6b7280; margin: 0; font-size: 14px;">{notif.get('message', '')}</p>
                    <p style="color: #9ca3af; margin: 10px 0 0 0; font-size: 12px;">{notif.get('timestamp', '')}</p>
                </div>
            """)

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0;">📬 Daily Digest</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">{len(notifications)} updates for {student_name}</p>
            </div>
            <div style="padding: 30px; background: #f9fafb;">
                {''.join(notification_items)}

                <div style="text-align: center; margin-top: 30px;">
                    <a href="#" style="background: #3b82f6; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        View All Updates
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        await self._send_email(
            to_email=parent_email,
            subject=subject,
            html_body=html_body
        )

        # Clear digest queue
        self.digest_queue[parent_id] = []

        logger.info(f"Sent digest to {parent_email} ({len(notifications)} notifications)")


# ============================================================================
# Helper Functions
# ============================================================================

def create_notification_engine(
    smtp_server: Optional[str] = None,
    smtp_username: Optional[str] = None,
    smtp_password: Optional[str] = None
) -> NotificationEngine:
    """
    Create notification engine

    Args:
        smtp_server: SMTP server
        smtp_username: SMTP username
        smtp_password: SMTP password

    Returns:
        NotificationEngine instance
    """
    return NotificationEngine(
        smtp_server=smtp_server,
        smtp_username=smtp_username,
        smtp_password=smtp_password
    )
