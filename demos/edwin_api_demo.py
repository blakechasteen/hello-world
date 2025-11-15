"""
EdWIN API Demo - Comprehensive FastAPI integration example

Demonstrates all API endpoints from Phase 2:
- Student management
- Tutoring (text + multimodal)
- Progress tracking
- Learning paths
- Recommendations
- Analytics

Requires API server running:
    PYTHONPATH=. uvicorn EduVerse.edwin.api:app --reload --port 8000

Author: EdWIN Team
Date: 2025-11-15
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any
from pathlib import Path


class EdWINAPIClient:
    """Client for EdWIN API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def create_student(self, first_name: str, last_name: str, grade_level: int) -> Dict[str, Any]:
        """Create a new student"""
        async with self.session.post(
            f"{self.base_url}/students",
            json={
                "first_name": first_name,
                "last_name": last_name,
                "grade_level": grade_level
            }
        ) as resp:
            return await resp.json()

    async def ask_question(
        self,
        student_id: str,
        question: str,
        mode: str = "verify"
    ) -> Dict[str, Any]:
        """Ask a question"""
        async with self.session.post(
            f"{self.base_url}/tutor/ask",
            json={
                "student_id": student_id,
                "question": question,
                "mode": mode
            }
        ) as resp:
            return await resp.json()

    async def update_progress(
        self,
        student_id: str,
        objective_id: str,
        success: bool,
        confidence: float
    ) -> Dict[str, Any]:
        """Update mastery progress"""
        async with self.session.post(
            f"{self.base_url}/progress",
            json={
                "student_id": student_id,
                "objective_id": objective_id,
                "success": success,
                "confidence": confidence
            }
        ) as resp:
            return await resp.json()

    async def get_progress(self, student_id: str) -> Dict[str, Any]:
        """Get student progress"""
        async with self.session.get(
            f"{self.base_url}/progress/{student_id}"
        ) as resp:
            return await resp.json()

    async def get_learning_path(
        self,
        student_id: str,
        target_objective: str
    ) -> Dict[str, Any]:
        """Get learning path to target"""
        async with self.session.post(
            f"{self.base_url}/learning-path",
            json={
                "student_id": student_id,
                "target_objective": target_objective
            }
        ) as resp:
            return await resp.json()

    async def get_recommendations(
        self,
        student_id: str,
        count: int = 5
    ) -> Dict[str, Any]:
        """Get personalized recommendations"""
        async with self.session.post(
            f"{self.base_url}/recommend",
            json={
                "student_id": student_id,
                "count": count
            }
        ) as resp:
            return await resp.json()

    async def get_analytics(self, student_id: str) -> Dict[str, Any]:
        """Get student analytics"""
        async with self.session.get(
            f"{self.base_url}/analytics/student/{student_id}"
        ) as resp:
            return await resp.json()

    async def get_platform_analytics(self) -> Dict[str, Any]:
        """Get platform-wide analytics"""
        async with self.session.get(
            f"{self.base_url}/analytics/summary"
        ) as resp:
            return await resp.json()


async def main():
    """Run comprehensive API demo"""

    print("=" * 80)
    print("EdWIN API Demo - Phase 2 Features")
    print("=" * 80)
    print()

    async with EdWINAPIClient() as client:

        # ===== Section 1: Student Management =====
        print("📋 Section 1: Student Management")
        print("-" * 80)

        # Create student
        print("Creating student: Jane Doe, Grade 8")
        student = await client.create_student("Jane", "Doe", 8)
        student_id = student["student_id"]
        print(f"✓ Student created: {student_id}")
        print(f"  Name: {student['name']}")
        print(f"  Grade: {student['grade_level']}")
        print()

        # ===== Section 2: Tutoring Q&A =====
        print("📚 Section 2: Tutoring Q&A")
        print("-" * 80)

        questions = [
            "What is the Pythagorean theorem?",
            "How do I solve quadratic equations?",
            "What are prime numbers?"
        ]

        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            response = await client.ask_question(student_id, question, mode="verify")

            print(f"Answer (confidence: {response['confidence']:.0%}):")
            print(f"  {response['answer'][:200]}...")
            print(f"  Sources: {len(response['sources'])} retrieved")

        print()

        # ===== Section 3: Progress Tracking =====
        print("📊 Section 3: Progress Tracking")
        print("-" * 80)

        # Simulate mastery updates
        objectives_to_master = [
            ("MATH.8.1", True, 0.95),
            ("MATH.8.2", True, 0.88),
            ("MATH.8.3", False, 0.65),
            ("MATH.8.4", True, 0.92),
        ]

        for obj_id, success, confidence in objectives_to_master:
            result = await client.update_progress(student_id, obj_id, success, confidence)
            status = "✓ Mastered" if result["newly_mastered"] else "○ Attempted"
            print(f"{status} {obj_id} (confidence: {confidence:.0%})")

        # Get progress summary
        print("\nProgress Summary:")
        progress = await client.get_progress(student_id)
        print(f"  Total Mastered: {progress['mastered_count']}")
        print(f"  Total XP: {progress['total_xp']}")
        print(f"  Current Streak: {progress['current_streak']} days")
        print()

        # ===== Section 4: Learning Paths =====
        print("🗺️  Section 4: Learning Paths")
        print("-" * 80)

        target = "MATH.10.5"  # Advanced objective
        print(f"Finding path to: {target}")

        path = await client.get_learning_path(student_id, target)

        if path["path_exists"]:
            print(f"✓ Path found ({path['steps']} steps):")
            for step in path["objectives"][:5]:  # Show first 5
                print(f"  → {step['objective_id']}: {step['content'][:60]}...")
            if path['steps'] > 5:
                print(f"  ... and {path['steps'] - 5} more steps")
            print(f"  Estimated time: {path['estimated_hours']:.1f} hours")
        else:
            print(f"✗ Path requires {len(path['missing_prerequisites'])} missing prerequisites")

        print()

        # ===== Section 5: Personalized Recommendations =====
        print("🎯 Section 5: Personalized Recommendations (Thompson Sampling)")
        print("-" * 80)

        recommendations = await client.get_recommendations(student_id, count=5)

        print(f"Top {len(recommendations['objectives'])} recommendations:")
        for i, rec in enumerate(recommendations['objectives'], 1):
            print(f"\n{i}. {rec['objective_id']} (difficulty: {rec['difficulty']:.2f})")
            print(f"   {rec['content'][:80]}...")
            print(f"   Expected reward: {rec['expected_reward']:.2f}")

        print(f"\nRecommendations based on: {recommendations['strategy']}")
        print()

        # ===== Section 6: Learning Analytics =====
        print("📈 Section 6: Learning Analytics")
        print("-" * 80)

        analytics = await client.get_analytics(student_id)

        # Velocity
        velocity = analytics['velocity']
        print("Learning Velocity:")
        print(f"  Objectives/week: {velocity['objectives_per_week']:.1f}")
        print(f"  XP/week: {velocity['xp_per_week']:.0f}")
        print(f"  Trend: {velocity['trend']}")
        print(f"  Current streak: {velocity['current_streak_days']} days")

        # Gaps
        gaps = analytics['gaps']
        print(f"\nKnowledge Gaps:")
        print(f"  Total gaps: {gaps['total_gaps']}")
        print(f"  Critical gaps: {len(gaps['critical_gaps'])}")
        if gaps['critical_gaps']:
            print(f"    → {gaps['critical_gaps'][0]}")
        print(f"  Estimated catchup: {gaps['estimated_catchup_hours']:.1f} hours")

        # Engagement
        engagement = analytics['engagement']
        print(f"\nEngagement:")
        print(f"  Score: {engagement['engagement_score']:.0%}")
        print(f"  Risk level: {engagement['risk_level']}")
        print(f"  Success rate: {engagement['success_rate']:.0%}")
        print(f"  Total interactions: {engagement['total_interactions']}")

        print()

        # ===== Section 7: Platform Analytics =====
        print("🌐 Section 7: Platform Analytics")
        print("-" * 80)

        platform = await client.get_platform_analytics()

        print(f"Platform Statistics:")
        print(f"  Total students: {platform['total_students']}")
        print(f"  Active students: {platform['active_students']}")
        print(f"  Total objectives mastered: {platform['total_objectives_mastered']}")
        print(f"  Average engagement: {platform['avg_engagement_score']:.0%}")

        print()

    print("=" * 80)
    print("✓ Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n⚠️  Make sure API server is running:")
    print("    PYTHONPATH=. uvicorn EduVerse.edwin.api:app --reload --port 8000\n")

    try:
        asyncio.run(main())
    except aiohttp.ClientConnectorError:
        print("\n❌ ERROR: Could not connect to API server.")
        print("   Please start the server first:")
        print("   PYTHONPATH=. uvicorn EduVerse.edwin.api:app --reload --port 8000\n")
