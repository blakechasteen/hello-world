"""
EduVerse - Text-Based Proof of Concept Demo

**Purpose**: Demonstrate complete learning pipeline
**Date**: November 13, 2025
**Pipeline**: Player Model → Curriculum → Quest Engine → Minigames → Rewards

This demo shows:
- Player creation and progression tracking
- Adaptive quest generation (Thompson Sampling)
- Quest selection and completion
- Minigame execution (Quiz, Puzzle, Code)
- Reward application and skill growth
- Learning analytics

Run: PYTHONPATH=. python EduVerse/demo_poc.py
"""

import asyncio
from typing import Optional

# EduVerse imports
from EduVerse.education.player_model import PlayerModel, Skill, Concept, SkillLevel
from EduVerse.education.curriculum import CurriculumFramework, Subject
from EduVerse.game.quest_engine import QuestEngine, Quest
from EduVerse.game.minigame_framework import MinigameFactory, MinigameType, QuizMinigame


class EduVerseDemo:
    """Complete EduVerse demonstration"""

    def __init__(self):
        self.player: Optional[PlayerModel] = None
        self.curriculum: Optional[CurriculumFramework] = None
        self.quest_engine: Optional[QuestEngine] = None

    def setup(self):
        """Initialize the system"""
        print("="*70)
        print(" "*20 + "EDUVERSE POC DEMO")
        print(" "*15 + "AI-Powered Learning Platform")
        print("="*70)
        print()

        # Create player
        print("[1/3] Creating player profile...")
        self.player = PlayerModel(
            student_id="demo_student_001",
            name="Alex Chen",
            grade=8
        )

        # Add some skills
        algebra_skill = Skill(
            id="math_algebra_basics",
            name="Algebra Basics",
            subject="math",
            description="Solve equations and work with variables",
            level=SkillLevel.UNDERSTAND,
            xp=25.0
        )
        self.player.add_skill(algebra_skill)

        print(f"   Player created: {self.player.name} (Grade {self.player.grade})")
        print(f"   Starting XP: {self.player.stats.total_xp}")
        print()

        # Create curriculum
        print("[2/3] Loading curriculum...")
        self.curriculum = CurriculumFramework()
        stats = self.curriculum.get_stats()
        print(f"   Loaded {stats['total_objectives']} learning objectives")
        print(f"   Subjects: {', '.join(stats['subjects'])}")
        print()

        # Create quest engine
        print("[3/3] Initializing quest engine...")
        self.quest_engine = QuestEngine(self.curriculum, self.player)
        print("   Quest engine ready with adaptive difficulty (Thompson Sampling)")
        print()

        print("Setup complete! Ready to learn.")
        print("="*70)
        print()

    def show_player_stats(self):
        """Display player statistics"""
        print("\n" + "-"*70)
        print(" "*25 + "PLAYER STATS")
        print("-"*70)
        print(f"Name: {self.player.name}")
        print(f"Grade: {self.player.grade}")
        print(f"Total XP: {self.player.stats.total_xp:.0f}")
        print(f"Quests Completed: {self.player.stats.quests_completed}/{self.player.stats.quests_attempted}")
        print(f"Minigames Completed: {self.player.stats.minigames_completed}")

        print(f"\nSkills ({len(self.player.skills)}):")
        for skill in self.player.skills.values():
            mastery = self.player.get_skill_mastery(skill.id)
            print(f"  - {skill.name}: {skill.level.name} ({mastery:.1%} mastery)")

        print(f"\nConcepts Mastered: {len(self.player.get_mastered_concepts())}/{len(self.player.concepts)}")

        print("-"*70 + "\n")

    def generate_quests(self, count: int = 3):
        """Generate recommended quests"""
        print("\n" + "="*70)
        print(" "*20 + "RECOMMENDED QUESTS")
        print("="*70)

        quests = self.quest_engine.generate_recommended_quests(count=count)

        if not quests:
            print("No quests available. Complete prerequisites to unlock more!")
            return quests

        for i, quest in enumerate(quests, 1):
            print(f"\n[{i}] {quest.title}")
            print(f"    {quest.description}")
            print(f"    Type: {quest.quest_type.value.upper()}")
            print(f"    Difficulty: {self._get_difficulty_stars(quest.difficulty)}")
            print(f"    Subject: {quest.subject.value if quest.subject else 'N/A'}")
            print(f"    Grade Level: {quest.grade_level}")
            print(f"    Time: ~{quest.estimated_minutes} min")
            print(f"    XP Reward: {quest.rewards.xp:.0f}")

        print("\n" + "="*70)

        return quests

    def _get_difficulty_stars(self, difficulty: float) -> str:
        """Convert difficulty to star rating"""
        stars = int(difficulty * 5)
        return "*" * max(1, stars) + "." * (5 - stars)

    def select_quest(self, quests: list) -> Optional[Quest]:
        """Let player select a quest"""
        if not quests:
            return None

        print("\n" + "-"*70)
        while True:
            try:
                choice = input(f"Select quest (1-{len(quests)}, or 'skip'): ").strip().lower()
                if choice == 'skip':
                    return None
                choice_int = int(choice)
                if 1 <= choice_int <= len(quests):
                    return quests[choice_int - 1]
                else:
                    print(f"Please enter a number between 1 and {len(quests)}")
            except ValueError:
                print("Invalid input. Please enter a number or 'skip'.")

    def start_quest(self, quest: Quest):
        """Start and execute a quest"""
        print("\n" + "="*70)
        print(f" "*15 + f"QUEST: {quest.title.upper()}")
        print("="*70)
        print()

        # Start quest
        self.quest_engine.start_quest(quest.id)
        print(f"Quest started: {quest.title}")
        print(f"Objective: {quest.description}")
        print()

        # Create minigame
        print(f"Loading minigame: {quest.minigame_id}...")
        print()

        # For demo, create a quiz minigame
        minigame_config = {
            "questions": self._generate_quiz_questions(quest),
            "demo_mode": True,  # Auto-complete for demo
            "time_limit_seconds": 300,
            "passing_score": 0.7
        }

        minigame = QuizMinigame(minigame_config)

        # Run minigame
        result = minigame.run()

        print("\n" + "-"*70)
        print(" "*25 + "RESULTS")
        print("-"*70)
        print(f"Score: {result.score:.1%} ({result.correct_answers}/{result.total_questions} correct)")
        print(f"Time: {result.time_seconds} seconds")
        print(f"Status: {'PASSED' if result.passed() else 'FAILED'}")
        print(f"Feedback: {result.feedback}")
        print("-"*70)

        # Complete quest
        rewards = self.quest_engine.complete_quest(
            quest.id,
            result.score,
            result.time_seconds
        )

        if rewards:
            print("\n" + " "*25 + "REWARDS EARNED")
            print(f"XP: +{rewards.xp:.0f}")
            if rewards.skill_xp:
                print(f"Skill XP: {', '.join(f'{k}: +{v:.0f}' for k, v in rewards.skill_xp.items())}")
            if rewards.achievements:
                print(f"Achievements: {', '.join(rewards.achievements)}")

        print("\n" + "="*70)

        return result

    def _generate_quiz_questions(self, quest: Quest) -> list:
        """Generate quiz questions based on quest"""
        # In real game, questions come from teacher-created content
        # For demo, generate sample questions

        subject = quest.subject.value if quest.subject else "general"

        if subject == "math":
            return [
                {
                    "question": "What is 12 x 9?",
                    "options": ["98", "108", "118", "128"],
                    "correct_index": 1,
                    "explanation": "12 x 9 = 108"
                },
                {
                    "question": "Solve: 3x + 7 = 22",
                    "options": ["x = 3", "x = 5", "x = 7", "x = 9"],
                    "correct_index": 1,
                    "explanation": "Subtract 7: 3x = 15, then divide by 3: x = 5"
                }
            ]
        elif subject == "science":
            return [
                {
                    "question": "What is the powerhouse of the cell?",
                    "options": ["Nucleus", "Mitochondria", "Ribosome", "Chloroplast"],
                    "correct_index": 1,
                    "explanation": "Mitochondria produce ATP energy for the cell"
                }
            ]
        elif subject == "ai_readiness":
            return [
                {
                    "question": "What does 'training data' mean in machine learning?",
                    "options": [
                        "Data used to test the model",
                        "Data used to teach the model",
                        "Data used to deploy the model",
                        "Data used to debug the model"
                    ],
                    "correct_index": 1,
                    "explanation": "Training data is used to teach/train the ML model"
                }
            ]
        else:
            return [
                {
                    "question": f"Sample question for {subject}",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "This is a placeholder question"
                }
            ]

    def show_progress_summary(self):
        """Show session summary"""
        print("\n\n" + "="*70)
        print(" "*20 + "SESSION SUMMARY")
        print("="*70)

        engine_stats = self.quest_engine.get_stats()

        print(f"\nQuests Attempted: {self.player.stats.quests_attempted}")
        print(f"Quests Completed: {self.player.stats.quests_completed}")
        print(f"Completion Rate: {engine_stats['completion_rate']:.1%}")
        print(f"Average Score: {engine_stats['average_score']:.1%}")

        print(f"\nTotal XP Earned: {self.player.stats.total_xp:.0f}")
        print(f"Play Time: {self.player.stats.total_play_time_minutes} minutes")

        print(f"\nSkills Progressed:")
        for skill in self.player.skills.values():
            print(f"  - {skill.name}: {skill.level.name} ({skill.xp:.0f} XP)")

        print("\n" + "="*70)
        print("\nGreat work today! Come back tomorrow to continue your learning journey.")
        print("="*70 + "\n")

    def run(self):
        """Run complete demo"""
        # Setup
        self.setup()
        input("Press Enter to continue...")

        # Show player stats
        self.show_player_stats()
        input("Press Enter to view recommended quests...")

        # Game loop (3 quests)
        for round_num in range(1, 4):
            print(f"\n\n{'='*70}")
            print(f" "*28 + f"ROUND {round_num}/3")
            print("="*70)

            # Generate quests
            quests = self.generate_quests(count=3)
            if not quests:
                print("\nNo more quests available!")
                break

            # Select quest
            selected_quest = self.select_quest(quests)
            if not selected_quest:
                print("\nSkipping quest selection...")
                continue

            # Run quest
            result = self.start_quest(selected_quest)

            # Update player stats
            self.player.record_minigame_completion()

            # Show progress
            print(f"\n[Round {round_num} complete]")
            input("Press Enter to continue...")

        # Final summary
        self.show_progress_summary()


def main():
    """Main entry point"""
    demo = EduVerseDemo()
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Thanks for trying EduVerse!")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
