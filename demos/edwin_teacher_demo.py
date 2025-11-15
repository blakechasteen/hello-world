"""
EdWIN Teacher Dashboard Demo - Real-time classroom monitoring

Demonstrates Phase 3 teacher dashboard features:
- Real-time classroom monitoring
- Student progress tracking
- Intervention detection
- Live activity feed
- Analytics dashboard

Requires dashboard server running:
    PYTHONPATH=. python -m EduVerse.edwin.teacher_dashboard

Author: EdWIN Team
Date: 2025-11-15
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, List


class DashboardClient:
    """Client for teacher dashboard API"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def get_snapshot(self) -> Dict[str, Any]:
        """Get classroom snapshot"""
        async with self.session.get(f"{self.base_url}/api/classroom/snapshot") as resp:
            return await resp.json()

    async def get_students(self) -> List[Dict[str, Any]]:
        """Get all students"""
        async with self.session.get(f"{self.base_url}/api/classroom/students") as resp:
            return await resp.json()

    async def get_student_detail(self, student_id: str) -> Dict[str, Any]:
        """Get student detail"""
        async with self.session.get(f"{self.base_url}/api/student/{student_id}") as resp:
            return await resp.json()

    async def get_interventions(self) -> List[Dict[str, Any]]:
        """Get intervention alerts"""
        async with self.session.get(f"{self.base_url}/api/interventions") as resp:
            return await resp.json()

    async def get_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activity"""
        async with self.session.get(f"{self.base_url}/api/activity?limit={limit}") as resp:
            return await resp.json()


def format_timestamp(ts: float) -> str:
    """Format timestamp for display"""
    now = datetime.now().timestamp()
    diff = now - ts

    if diff < 60:
        return "Just now"
    elif diff < 3600:
        return f"{int(diff / 60)}m ago"
    elif diff < 86400:
        return f"{int(diff / 3600)}h ago"
    else:
        return f"{int(diff / 86400)}d ago"


async def demo_classroom_overview():
    """Demonstrate classroom overview"""

    print("=" * 80)
    print("Section 1: Classroom Overview")
    print("=" * 80)
    print()

    async with DashboardClient() as client:
        snapshot = await client.get_snapshot()

        print("📊 Classroom Snapshot")
        print("-" * 80)
        print(f"Total Students:     {snapshot['total_students']}")
        print(f"Active (24h):       {snapshot['active_students']}")
        print(f"Avg Engagement:     {snapshot['avg_engagement_score']:.0%}")
        print(f"Interventions:      {snapshot['pending_interventions']}")
        print(f"Total Mastered:     {snapshot['total_objectives_mastered']}")
        print(f"Avg Mastery:        {snapshot['avg_mastery_percentage']:.1f}%")
        print()

        # Subject breakdown
        if snapshot['subject_breakdown']:
            print("📚 Subject Breakdown:")
            subjects = sorted(
                snapshot['subject_breakdown'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            subject_names = {
                'MATH': 'Mathematics',
                'SCIENCE': 'Science',
                'READING': 'Reading',
                'WRITING': 'Writing',
                'SOCIAL': 'Social Studies'
            }

            for subject, count in subjects:
                name = subject_names.get(subject, subject)
                bar = "█" * (count // 5) + "░" * (20 - count // 5)
                print(f"  {name:15} {bar} {count}")

        print()


async def demo_student_monitoring():
    """Demonstrate student monitoring"""

    print("=" * 80)
    print("Section 2: Student Monitoring")
    print("=" * 80)
    print()

    async with DashboardClient() as client:
        students = await client.get_students()

        print(f"👥 Students ({len(students)} total)")
        print("-" * 80)
        print()

        # Group by risk level
        by_risk = {'high': [], 'medium': [], 'low': []}
        for student in students:
            by_risk[student['risk_level']].append(student)

        # Show high risk first
        for risk_level in ['high', 'medium', 'low']:
            risk_students = by_risk[risk_level]

            if not risk_students:
                continue

            # Risk indicators
            indicators = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }

            print(f"{indicators[risk_level]} {risk_level.upper()} RISK ({len(risk_students)} students)")
            print()

            for student in risk_students[:5]:  # Show top 5 per category
                last_active = format_timestamp(student['last_active']) if student['last_active'] else 'Never'

                print(f"  {student['name']}")
                print(f"    Mastered: {student['mastered_count']} | "
                      f"Engagement: {student['engagement_score']:.0%} | "
                      f"Streak: {student['current_streak']} days")
                print(f"    Last active: {last_active}")
                print()

            if len(risk_students) > 5:
                print(f"    ... and {len(risk_students) - 5} more")
                print()


async def demo_interventions():
    """Demonstrate intervention system"""

    print("=" * 80)
    print("Section 3: Intervention Alerts")
    print("=" * 80)
    print()

    async with DashboardClient() as client:
        interventions = await client.get_interventions()

        if not interventions:
            print("🎉 No interventions needed - all students on track!")
            print()
            return

        print(f"🚨 Active Interventions ({len(interventions)} alerts)")
        print("-" * 80)
        print()

        # Group by severity
        by_severity = {'critical': [], 'warning': []}
        for alert in interventions:
            by_severity[alert['severity']].append(alert)

        # Critical first
        for severity in ['critical', 'warning']:
            alerts = by_severity[severity]

            if not alerts:
                continue

            severity_icons = {
                'critical': '🔴',
                'warning': '🟡'
            }

            print(f"{severity_icons[severity]} {severity.upper()} ({len(alerts)} alerts)")
            print()

            for alert in alerts:
                print(f"  Student: {alert['student_id']}")
                print(f"  Reason: {alert['reason']}")
                print()
                print(f"  Recommended Actions:")
                for action in alert['recommended_actions']:
                    print(f"    • {action}")
                print()


async def demo_student_detail():
    """Demonstrate detailed student view"""

    print("=" * 80)
    print("Section 4: Student Detail View")
    print("=" * 80)
    print()

    async with DashboardClient() as client:
        # Get first student
        students = await client.get_students()

        if not students:
            print("No students available")
            return

        student = students[0]
        student_id = student['student_id']

        print(f"🔍 Detailed View: {student['name']}")
        print("-" * 80)
        print()

        detail = await client.get_student_detail(student_id)

        # Learning velocity
        velocity = detail['velocity']
        print("📈 Learning Velocity")
        print(f"  Objectives/week: {velocity['objectives_per_week']:.1f}")
        print(f"  XP/week:         {velocity['xp_per_week']:.0f}")
        print(f"  Avg confidence:  {velocity['avg_confidence']:.0%}")
        print(f"  Trend:           {velocity['trend']}")
        print(f"  Current streak:  {velocity['current_streak_days']} days")
        print(f"  Best streak:     {velocity['best_streak_days']} days")
        print()

        # Knowledge gaps
        gaps = detail['gaps']
        print("🎯 Knowledge Gaps")
        print(f"  Total gaps:      {gaps['total_gaps']}")
        print(f"  Critical gaps:   {len(gaps['critical_gaps'])}")
        print(f"  Catchup time:    {gaps['estimated_catchup_hours']:.1f} hours")
        print()

        if gaps['critical_gaps']:
            print("  Critical prerequisites:")
            for gap in gaps['critical_gaps'][:3]:
                print(f"    • {gap}")
            if len(gaps['critical_gaps']) > 3:
                print(f"    ... and {len(gaps['critical_gaps']) - 3} more")
            print()

        # Engagement
        engagement = detail['engagement']
        print("⚡ Engagement Metrics")
        print(f"  Engagement score:  {engagement['engagement_score']:.0%}")
        print(f"  Success rate:      {engagement['success_rate']:.0%}")
        print(f"  Risk level:        {engagement['risk_level']}")
        print(f"  Total interactions: {engagement['total_interactions']}")
        print(f"  Questions asked:   {engagement['questions_asked']}")
        print()


async def demo_activity_feed():
    """Demonstrate live activity feed"""

    print("=" * 80)
    print("Section 5: Live Activity Feed")
    print("=" * 80)
    print()

    async with DashboardClient() as client:
        activities = await client.get_activity(limit=15)

        if not activities:
            print("No recent activity")
            return

        print(f"📝 Recent Activity (last {len(activities)} events)")
        print("-" * 80)
        print()

        for activity in activities:
            timestamp = format_timestamp(activity['timestamp'])
            student = activity['student_name']
            event_type = activity['event_type']
            details = activity['details']

            # Format event
            if event_type == 'question':
                event_str = f"asked: \"{details.get('question', 'N/A')[:50]}...\""
            elif event_type == 'mastery_update':
                success = "✓ mastered" if details.get('success') else "○ attempted"
                event_str = f"{success} {details.get('objective_id', 'N/A')}"
            elif event_type == 'achievement':
                event_str = f"🎉 earned: {details.get('name', 'achievement')}"
            else:
                event_str = event_type

            print(f"[{timestamp:8}] {student}: {event_str}")

        print()


async def demo_realtime_simulation():
    """Simulate real-time dashboard updates"""

    print("=" * 80)
    print("Section 6: Real-time Updates Simulation")
    print("=" * 80)
    print()

    print("📡 WebSocket Connection Status")
    print("-" * 80)
    print()

    print("In production, the dashboard receives real-time updates via WebSocket:")
    print()
    print("  ✓ Student activity events (questions, mastery updates)")
    print("  ✓ Classroom snapshots (every 30 seconds)")
    print("  ✓ Intervention alerts (as they occur)")
    print("  ✓ Engagement metric updates")
    print()

    print("Dashboard auto-refreshes without page reload!")
    print()


async def main():
    """Run teacher dashboard demo"""

    print("\n" + "=" * 80)
    print("EdWIN Teacher Dashboard Demo - Phase 3 Classroom Orchestration")
    print("=" * 80)
    print()

    try:
        # Run demos
        await demo_classroom_overview()
        await demo_student_monitoring()
        await demo_interventions()
        await demo_student_detail()
        await demo_activity_feed()
        await demo_realtime_simulation()

        print("=" * 80)
        print("✓ Demo Complete!")
        print("=" * 80)
        print()

        print("Key Features Demonstrated:")
        print("  ✓ Real-time classroom snapshot")
        print("  ✓ Student monitoring with risk levels")
        print("  ✓ Automatic intervention detection")
        print("  ✓ Detailed student analytics")
        print("  ✓ Live activity feed")
        print("  ✓ WebSocket real-time updates")
        print()

        print("To view the interactive dashboard:")
        print("  1. Start the dashboard server:")
        print("     PYTHONPATH=. python -m EduVerse.edwin.teacher_dashboard")
        print("  2. Open browser to: http://localhost:8001")
        print()

    except aiohttp.ClientConnectorError:
        print("\n❌ ERROR: Could not connect to dashboard server.")
        print("   Please start the server first:")
        print("   PYTHONPATH=. python -m EduVerse.edwin.teacher_dashboard\n")


if __name__ == "__main__":
    print("\n⚠️  Make sure dashboard server is running:")
    print("    PYTHONPATH=. python -m EduVerse.edwin.teacher_dashboard\n")

    asyncio.run(main())
