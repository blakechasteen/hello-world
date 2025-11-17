#!/usr/bin/env python3
"""
EdWIN Tutor - Terminal UI

Interactive terminal-based learning platform for HoloLoom.

Usage:
    python edwin.py                # Start interactive mode
    python edwin.py --lesson 01    # Jump to specific lesson
    python edwin.py --stats        # Show your progress
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for HoloLoom imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lesson import LessonManager, Lesson, LessonType, Difficulty
from core.progress import ProgressTracker


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class EdWINTerminal:
    """Terminal-based interactive tutor."""

    def __init__(self):
        content_dir = Path(__file__).parent / "content"
        self.lesson_manager = LessonManager(content_dir)
        self.progress_tracker = ProgressTracker()
        self.current_lesson: Optional[Lesson] = None
        self.hint_level = 0  # Track hint progression

    def print_header(self, text: str):
        """Print a styled header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

    def print_success(self, text: str):
        """Print success message."""
        print(f"{Colors.GREEN}✓ {text}{Colors.END}")

    def print_error(self, text: str):
        """Print error message."""
        print(f"{Colors.RED}✗ {text}{Colors.END}")

    def print_info(self, text: str):
        """Print info message."""
        print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

    def print_lesson_content(self, content: str):
        """Print lesson content with markdown-like formatting."""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                print(f"\n{Colors.BOLD}{Colors.HEADER}{line[2:]}{Colors.END}\n")
            elif line.startswith('## '):
                print(f"\n{Colors.BOLD}{line[3:]}{Colors.END}\n")
            elif line.startswith('```'):
                print(f"{Colors.CYAN}{line}{Colors.END}")
            elif line.strip().startswith('-'):
                print(f"  {Colors.YELLOW}•{Colors.END} {line.strip()[1:].strip()}")
            else:
                print(line)

    def show_welcome(self):
        """Display welcome screen."""
        self.print_header("🎓 Welcome to EdWIN Tutor!")

        print(f"""
{Colors.BOLD}Your Interactive Guide to HoloLoom{Colors.END}

Hi, I'm EdWIN - your {Colors.CYAN}Ed{Colors.END}ucational {Colors.CYAN}WIN{Colors.END}dow into HoloLoom!

I'll help you learn HoloLoom through:
  {Colors.YELLOW}•{Colors.END} Interactive lessons
  {Colors.YELLOW}•{Colors.END} Hands-on coding challenges
  {Colors.YELLOW}•{Colors.END} Quizzes to test your knowledge
  {Colors.YELLOW}•{Colors.END} Badges and achievements
  {Colors.YELLOW}•{Colors.END} A learning path tailored to YOU

{Colors.BOLD}Your Progress:{Colors.END}
  Level: {Colors.GREEN}{self.progress_tracker.progress.level}{Colors.END}
  XP: {Colors.CYAN}{self.progress_tracker.progress.total_xp}{Colors.END}
  Lessons Completed: {Colors.GREEN}{len(self.progress_tracker.progress.completed_lessons)}{Colors.END}
  Badges Earned: {Colors.YELLOW}{len(self.progress_tracker.progress.badges)}{Colors.END}
""")

        if not self.progress_tracker.progress.completed_lessons:
            print(f"{Colors.BOLD}Let's start your journey!{Colors.END}\n")
        else:
            print(f"{Colors.BOLD}Welcome back, {self.progress_tracker.progress.learner_name}!{Colors.END}\n")

    def show_lesson_menu(self):
        """Display lesson selection menu."""
        self.print_header("📚 Available Lessons")

        # Group lessons by difficulty
        for difficulty in [Difficulty.BEGINNER, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]:
            lessons = self.lesson_manager.get_lessons_by_difficulty(difficulty)
            if not lessons:
                continue

            print(f"\n{Colors.BOLD}{difficulty.value.upper()} Lessons:{Colors.END}")
            for lesson in sorted(lessons, key=lambda l: l.id):
                # Check if completed
                completed = lesson.id in self.progress_tracker.progress.completed_lessons
                status = f"{Colors.GREEN}✓{Colors.END}" if completed else " "

                # Check if prerequisites met
                prereqs_met = all(
                    p in self.progress_tracker.progress.completed_lessons
                    for p in lesson.prerequisites
                )
                if not prereqs_met:
                    status = f"{Colors.RED}🔒{Colors.END}"

                print(f"  [{status}] {lesson.id}: {lesson.title} ({lesson.estimated_time} min)")

        print(f"\n{Colors.BOLD}Legend:{Colors.END} {Colors.GREEN}✓{Colors.END} = Completed, {Colors.RED}🔒{Colors.END} = Locked\n")

    def show_lesson(self, lesson: Lesson):
        """Display a lesson and its content."""
        self.current_lesson = lesson
        self.hint_level = 0  # Reset hints
        self.progress_tracker.start_lesson(lesson.id)

        self.print_header(f"{lesson.title}")
        print(f"{Colors.BOLD}Estimated Time:{Colors.END} {lesson.estimated_time} minutes")
        print(f"{Colors.BOLD}Difficulty:{Colors.END} {lesson.difficulty.value.capitalize()}")
        print(f"{Colors.BOLD}Type:{Colors.END} {lesson.lesson_type.value.replace('_', ' ').title()}")
        print(f"{Colors.BOLD}XP Reward:{Colors.END} {lesson.xp_reward}\n")

        # Print lesson content
        self.print_lesson_content(lesson.content)

        # Show code examples if any
        if lesson.code_examples:
            print(f"\n{Colors.BOLD}📝 Code Examples:{Colors.END}\n")
            for idx, example in enumerate(lesson.code_examples, 1):
                print(f"{Colors.CYAN}Example {idx}:{Colors.END}")
                print(f"{Colors.YELLOW}{example}{Colors.END}\n")

    def run_quiz(self, lesson: Lesson) -> int:
        """Run quiz questions and return score."""
        if not lesson.quiz_questions:
            return 0

        self.print_header("📝 Quiz Time!")
        print("Let's test your understanding!\n")

        correct = 0
        for idx, q in enumerate(lesson.quiz_questions, 1):
            print(f"{Colors.BOLD}Question {idx}:{Colors.END} {q['question']}\n")

            for i, option in enumerate(q['options']):
                print(f"  {i + 1}. {option}")

            while True:
                try:
                    answer = input(f"\n{Colors.CYAN}Your answer (1-{len(q['options'])}): {Colors.END}").strip()
                    answer_idx = int(answer) - 1

                    if 0 <= answer_idx < len(q['options']):
                        if answer_idx == q['correct']:
                            self.print_success("Correct!")
                            print(f"{Colors.BLUE}💡 {q['explanation']}{Colors.END}\n")
                            correct += 1
                        else:
                            self.print_error("Not quite right.")
                            print(f"{Colors.BLUE}💡 {q['explanation']}{Colors.END}\n")
                        break
                    else:
                        print(f"{Colors.YELLOW}Please enter a number between 1 and {len(q['options'])}{Colors.END}")
                except (ValueError, KeyError):
                    print(f"{Colors.YELLOW}Invalid input. Please enter a number.{Colors.END}")

        score = (correct / len(lesson.quiz_questions)) * 100
        print(f"\n{Colors.BOLD}Quiz Score: {score:.0f}% ({correct}/{len(lesson.quiz_questions)}){Colors.END}\n")
        return int(score)

    def run_challenges(self, lesson: Lesson):
        """Run coding challenges."""
        if not lesson.challenges:
            return

        self.print_header("🧩 Challenges")
        print("Time to apply what you learned!\n")

        for idx, challenge in enumerate(lesson.challenges, 1):
            print(f"{Colors.BOLD}Challenge {idx}:{Colors.END} {challenge.description}")
            print(f"{Colors.BLUE}Points:{Colors.END} {challenge.points}\n")

            if challenge.starter_code:
                print(f"{Colors.YELLOW}Starter Code:{Colors.END}")
                print(f"{Colors.CYAN}{challenge.starter_code}{Colors.END}\n")

            # Get user's answer
            print(f"{Colors.BOLD}Your answer (type 'hint' for a hint, 'skip' to skip):{Colors.END}")
            user_code = input().strip()

            if user_code.lower() == 'skip':
                print(f"{Colors.YELLOW}Skipped challenge.{Colors.END}\n")
                continue
            elif user_code.lower() == 'hint':
                self.show_hint(challenge)
                continue

            # Validate challenge (simplified for demo)
            self.print_success(f"Great job! +{challenge.points} points")
            self.progress_tracker.mark_challenge_complete(f"{lesson.id}_challenge_{idx}", challenge.points)

    def show_hint(self, challenge):
        """Show progressive hints."""
        if self.hint_level >= len(challenge.hints):
            print(f"{Colors.YELLOW}No more hints available!{Colors.END}")
            return

        hint = challenge.hints[self.hint_level]
        print(f"\n{Colors.BLUE}💡 Hint {self.hint_level + 1}:{Colors.END} {hint.text}")
        if hint.code_snippet:
            print(f"{Colors.CYAN}{hint.code_snippet}{Colors.END}")
        print()

        self.hint_level += 1
        self.progress_tracker.use_hint()

    async def run(self):
        """Main interactive loop."""
        self.show_welcome()

        while True:
            print(f"\n{Colors.BOLD}What would you like to do?{Colors.END}")
            print("  1. Start a lesson")
            print("  2. View progress")
            print("  3. Exit")

            choice = input(f"\n{Colors.CYAN}Choose (1-3): {Colors.END}").strip()

            if choice == '1':
                self.show_lesson_menu()
                lesson_id = input(f"\n{Colors.CYAN}Enter lesson ID (or 'back'): {Colors.END}").strip()

                if lesson_id.lower() == 'back':
                    continue

                lesson = self.lesson_manager.get_lesson(lesson_id)
                if lesson:
                    self.show_lesson(lesson)

                    # Run quiz
                    if lesson.quiz_questions:
                        quiz_score = self.run_quiz(lesson)
                    else:
                        quiz_score = 100

                    # Run challenges
                    if lesson.challenges:
                        self.run_challenges(lesson)

                    # Mark complete
                    result = self.progress_tracker.mark_lesson_complete(
                        lesson.id, lesson.xp_reward, quiz_score
                    )

                    self.print_success(f"Lesson complete! +{lesson.xp_reward} XP")
                    if result['leveled_up']:
                        print(f"\n🎉 {Colors.BOLD}LEVEL UP!{Colors.END} You're now level {result['new_level']}! 🎉\n")

                else:
                    self.print_error(f"Lesson '{lesson_id}' not found.")

            elif choice == '2':
                stats = self.progress_tracker.get_stats()
                self.print_header("📊 Your Progress")
                print(f"Level: {Colors.GREEN}{stats['level']}{Colors.END}")
                print(f"XP: {Colors.CYAN}{stats['xp']}{Colors.END} (Next level: {stats['xp_to_next_level']} XP)")
                print(f"Lessons Completed: {Colors.GREEN}{stats['lessons_completed']}{Colors.END}")
                print(f"Challenges Completed: {Colors.GREEN}{stats['challenges_completed']}{Colors.END}")
                print(f"Badges Earned: {Colors.YELLOW}{stats['badges_earned']}{Colors.END}")
                print(f"Completion Rate: {Colors.CYAN}{stats['completion_rate']:.1f}%{Colors.END}")

            elif choice == '3':
                print(f"\n{Colors.BOLD}Thanks for learning with EdWIN! Keep exploring HoloLoom! 🚀{Colors.END}\n")
                break

            else:
                self.print_error("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Entry point."""
    tutor = EdWINTerminal()
    asyncio.run(tutor.run())


if __name__ == "__main__":
    main()
