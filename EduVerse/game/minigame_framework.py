"""
Minigame Framework - Plugin Architecture for Learning Games

**Purpose**: Base classes and implementations for educational minigames
**Date**: November 13, 2025
**Plugin Architecture**: Teachers can create custom minigames

This module implements the minigame system:
- Base minigame class (abstract, defines lifecycle)
- 3 built-in types (Quiz, Puzzle, Code Challenge)
- JSON configuration format
- Scoring and feedback system
- Teacher SDK integration (future)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import random
import json


class MinigameType(Enum):
    """Types of minigames"""
    QUIZ = "quiz"
    PUZZLE = "puzzle"
    CODE_CHALLENGE = "code"
    SIMULATION = "simulation"
    EXPLORATION = "exploration"
    CREATIVE = "creative"


class MinigameState(Enum):
    """Minigame execution states"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MinigameResult:
    """Result of a minigame attempt"""
    score: float  # 0.0 - 1.0
    correct_answers: int
    total_questions: int
    time_seconds: int
    attempts: int
    feedback: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def passed(self, passing_threshold: float = 0.7) -> bool:
        """Check if player passed"""
        return self.score >= passing_threshold

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "score": self.score,
            "correct_answers": self.correct_answers,
            "total_questions": self.total_questions,
            "time_seconds": self.time_seconds,
            "attempts": self.attempts,
            "feedback": self.feedback,
            "passed": self.passed(),
            "details": self.details
        }


class Minigame(ABC):
    """
    Abstract base class for all minigames

    All minigames must implement:
    - setup(): Initialize minigame state
    - execute(): Run the minigame (main logic)
    - evaluate(): Score the player's performance
    - cleanup(): Clean up resources

    Lifecycle:
    1. __init__(config) - Create minigame with config
    2. setup() - Initialize state
    3. execute() - Run game loop
    4. evaluate() - Score performance
    5. cleanup() - Clean up
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = MinigameState.NOT_STARTED
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[MinigameResult] = None

    @abstractmethod
    def setup(self) -> None:
        """Initialize minigame state (load questions, set up UI, etc.)"""
        pass

    @abstractmethod
    def execute(self) -> None:
        """Run the minigame (main game loop)"""
        pass

    @abstractmethod
    def evaluate(self) -> MinigameResult:
        """Evaluate player performance and return result"""
        pass

    def cleanup(self) -> None:
        """Clean up resources (optional, override if needed)"""
        pass

    def run(self) -> MinigameResult:
        """
        Run complete minigame lifecycle

        Returns:
            MinigameResult with score and feedback
        """
        self.started_at = datetime.utcnow()
        self.state = MinigameState.IN_PROGRESS

        try:
            # Lifecycle
            self.setup()
            self.execute()
            self.result = self.evaluate()

            # Mark completed or failed
            if self.result.passed():
                self.state = MinigameState.COMPLETED
            else:
                self.state = MinigameState.FAILED

        except Exception as e:
            self.state = MinigameState.FAILED
            self.result = MinigameResult(
                score=0.0,
                correct_answers=0,
                total_questions=1,
                time_seconds=0,
                attempts=1,
                feedback=f"Error: {str(e)}"
            )

        finally:
            self.completed_at = datetime.utcnow()
            self.cleanup()

        return self.result


# ========== Quiz Minigame ==========

@dataclass
class QuizQuestion:
    """A single quiz question"""
    question: str
    options: List[str]
    correct_index: int
    explanation: str = ""
    difficulty: float = 0.5  # 0.0 - 1.0


class QuizMinigame(Minigame):
    """
    Multiple choice quiz minigame

    Config format:
    {
        "questions": [
            {
                "question": "What is 2 + 2?",
                "options": ["3", "4", "5", "6"],
                "correct_index": 1,
                "explanation": "2 + 2 = 4"
            }
        ],
        "time_limit_seconds": 300,
        "passing_score": 0.7,
        "allow_review": true
    }
    """

    def setup(self) -> None:
        """Load questions and initialize state"""
        questions_data = self.config.get("questions", [])
        self.questions = [
            QuizQuestion(
                question=q["question"],
                options=q["options"],
                correct_index=q["correct_index"],
                explanation=q.get("explanation", ""),
                difficulty=q.get("difficulty", 0.5)
            )
            for q in questions_data
        ]

        # Shuffle questions if configured
        if self.config.get("shuffle_questions", True):
            random.shuffle(self.questions)

        # Player answers
        self.answers: List[Optional[int]] = [None] * len(self.questions)
        self.current_question_index = 0

        # Timing
        self.time_limit = self.config.get("time_limit_seconds", 300)
        self.time_started = datetime.utcnow()

    def execute(self) -> None:
        """
        Run quiz (text-based for now, Unity UI later)

        In a real game, this would display UI and wait for input.
        For POC, we'll simulate player answers.
        """
        print(f"\n=== Quiz Start: {len(self.questions)} questions ===\n")

        for i, question in enumerate(self.questions):
            print(f"Q{i + 1}. {question.question}")
            for j, option in enumerate(question.options):
                print(f"  {j + 1}. {option}")

            # In Unity, this would be a button click
            # For demo, we'll simulate an answer
            if self.config.get("demo_mode", False):
                # Simulate correct answer 80% of the time
                if random.random() < 0.8:
                    answer = question.correct_index
                else:
                    # Random wrong answer
                    wrong_answers = [j for j in range(len(question.options)) if j != question.correct_index]
                    answer = random.choice(wrong_answers)

                self.answers[i] = answer
                print(f"  → Selected: {answer + 1}")
            else:
                # Get user input
                while True:
                    try:
                        answer = int(input(f"  Your answer (1-{len(question.options)}): ")) - 1
                        if 0 <= answer < len(question.options):
                            self.answers[i] = answer
                            break
                        else:
                            print(f"  Please enter a number between 1 and {len(question.options)}")
                    except ValueError:
                        print("  Invalid input. Please enter a number.")

            # Show feedback if enabled
            if self.config.get("immediate_feedback", False):
                if self.answers[i] == question.correct_index:
                    print(f"  ✓ Correct! {question.explanation}")
                else:
                    print(f"  ✗ Incorrect. {question.explanation}")

            print()

    def evaluate(self) -> MinigameResult:
        """Score the quiz"""
        correct_count = 0
        for i, question in enumerate(self.questions):
            if self.answers[i] == question.correct_index:
                correct_count += 1

        total = len(self.questions)
        score = correct_count / total if total > 0 else 0.0

        time_elapsed = (datetime.utcnow() - self.time_started).total_seconds()

        # Generate feedback
        if score >= 0.9:
            feedback = "Excellent work! You've mastered this topic."
        elif score >= 0.7:
            feedback = "Good job! Keep practicing to improve."
        elif score >= 0.5:
            feedback = "You're making progress. Review the material and try again."
        else:
            feedback = "Let's review the basics together. You'll get there!"

        return MinigameResult(
            score=score,
            correct_answers=correct_count,
            total_questions=total,
            time_seconds=int(time_elapsed),
            attempts=1,
            feedback=feedback,
            details={
                "answers": self.answers,
                "time_limit": self.time_limit,
                "time_elapsed": time_elapsed
            }
        )


# ========== Puzzle Minigame ==========

class PuzzleMinigame(Minigame):
    """
    Logic puzzle minigame (pattern matching, sequencing, etc.)

    Config format:
    {
        "puzzle_type": "pattern",
        "pattern": [1, 2, 4, 8, "?"],
        "answer": 16,
        "hint": "Each number doubles",
        "difficulty": 0.6
    }
    """

    def setup(self) -> None:
        """Load puzzle configuration"""
        self.puzzle_type = self.config.get("puzzle_type", "pattern")
        self.pattern = self.config.get("pattern", [])
        self.correct_answer = self.config.get("answer")
        self.hint = self.config.get("hint", "Look for the pattern!")
        self.player_answer = None

    def execute(self) -> None:
        """Run puzzle"""
        print(f"\n=== Puzzle Challenge ===")
        print(f"Type: {self.puzzle_type}")
        print(f"Pattern: {' → '.join(str(x) for x in self.pattern)}")
        print(f"Hint: {self.hint}\n")

        if self.config.get("demo_mode", False):
            # Simulate answer (correct 70% of the time)
            if random.random() < 0.7:
                self.player_answer = self.correct_answer
            else:
                self.player_answer = self.correct_answer + random.randint(-5, 5)
            print(f"Answer: {self.player_answer}\n")
        else:
            self.player_answer = input("Your answer: ")
            # Try to convert to number if possible
            try:
                self.player_answer = float(self.player_answer)
            except ValueError:
                pass  # Keep as string

    def evaluate(self) -> MinigameResult:
        """Score the puzzle"""
        correct = (self.player_answer == self.correct_answer)
        score = 1.0 if correct else 0.0

        feedback = "Correct! You found the pattern." if correct else f"Not quite. The answer was {self.correct_answer}. {self.hint}"

        return MinigameResult(
            score=score,
            correct_answers=1 if correct else 0,
            total_questions=1,
            time_seconds=0,  # Not timed for now
            attempts=1,
            feedback=feedback,
            details={"player_answer": self.player_answer, "correct_answer": self.correct_answer}
        )


# ========== Code Challenge Minigame ==========

class CodeChallengeMinigame(Minigame):
    """
    Programming challenge minigame

    Config format:
    {
        "language": "python",
        "challenge": "Write a function that returns the sum of two numbers",
        "test_cases": [
            {"input": [2, 3], "output": 5},
            {"input": [10, -5], "output": 5}
        ],
        "starter_code": "def add(a, b):\n    # Your code here\n    pass",
        "difficulty": 0.7
    }
    """

    def setup(self) -> None:
        """Load challenge configuration"""
        self.language = self.config.get("language", "python")
        self.challenge = self.config.get("challenge", "")
        self.test_cases = self.config.get("test_cases", [])
        self.starter_code = self.config.get("starter_code", "")
        self.player_code = ""

    def execute(self) -> None:
        """Run code challenge"""
        print(f"\n=== Code Challenge ({self.language.upper()}) ===")
        print(f"Challenge: {self.challenge}\n")
        print("Starter code:")
        print(self.starter_code)
        print()

        if self.config.get("demo_mode", False):
            # Simulate solution (correct 60% of the time)
            if random.random() < 0.6:
                # Simulate correct solution
                if self.language == "python":
                    self.player_code = "def add(a, b):\n    return a + b"
            else:
                # Simulate wrong solution
                self.player_code = "def add(a, b):\n    return a * b"  # Wrong!

            print(f"[Simulated solution submitted]\n")
        else:
            print("Enter your code (type 'DONE' on a new line when finished):")
            code_lines = []
            while True:
                line = input()
                if line.strip() == "DONE":
                    break
                code_lines.append(line)
            self.player_code = "\n".join(code_lines)

    def evaluate(self) -> MinigameResult:
        """Score the code challenge"""
        if not self.player_code:
            return MinigameResult(
                score=0.0,
                correct_answers=0,
                total_questions=len(self.test_cases),
                time_seconds=0,
                attempts=1,
                feedback="No code submitted."
            )

        # Run test cases (simplified - in real game, use safe sandboxed execution)
        passed_tests = 0
        for test in self.test_cases:
            try:
                # WARNING: This is unsafe! Only for demo. Use sandboxed execution in production.
                if self.language == "python":
                    # Create a namespace and execute code
                    namespace = {}
                    exec(self.player_code, namespace)

                    # Find the function (assume first function defined)
                    func = None
                    for name, obj in namespace.items():
                        if callable(obj) and not name.startswith("__"):
                            func = obj
                            break

                    if func:
                        result = func(*test["input"])
                        if result == test["output"]:
                            passed_tests += 1
            except Exception as e:
                # Execution error
                pass

        total_tests = len(self.test_cases)
        score = passed_tests / total_tests if total_tests > 0 else 0.0

        if score == 1.0:
            feedback = "Perfect! All test cases passed."
        elif score >= 0.5:
            feedback = f"Good attempt! {passed_tests}/{total_tests} test cases passed."
        else:
            feedback = f"Keep trying! {passed_tests}/{total_tests} test cases passed."

        return MinigameResult(
            score=score,
            correct_answers=passed_tests,
            total_questions=total_tests,
            time_seconds=0,
            attempts=1,
            feedback=feedback,
            details={
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "language": self.language
            }
        )


# ========== Minigame Factory ==========

class MinigameFactory:
    """Factory for creating minigames from configuration"""

    MINIGAME_TYPES = {
        MinigameType.QUIZ: QuizMinigame,
        MinigameType.PUZZLE: PuzzleMinigame,
        MinigameType.CODE_CHALLENGE: CodeChallengeMinigame
    }

    @classmethod
    def create(cls, minigame_type: MinigameType, config: Dict[str, Any]) -> Minigame:
        """
        Create a minigame from type and config

        Args:
            minigame_type: Type of minigame
            config: Configuration dictionary

        Returns:
            Minigame instance
        """
        minigame_class = cls.MINIGAME_TYPES.get(minigame_type)
        if not minigame_class:
            raise ValueError(f"Unknown minigame type: {minigame_type}")

        return minigame_class(config)

    @classmethod
    def create_from_json(cls, json_path: str) -> Minigame:
        """Create minigame from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        minigame_type = MinigameType(data.get("type", "quiz"))
        config = data.get("config", {})

        return cls.create(minigame_type, config)


# ========== Sample Minigames ==========

def create_sample_quiz() -> QuizMinigame:
    """Create sample math quiz"""
    config = {
        "questions": [
            {
                "question": "What is 7 × 8?",
                "options": ["54", "56", "58", "64"],
                "correct_index": 1,
                "explanation": "7 × 8 = 56"
            },
            {
                "question": "Solve for x: 2x + 5 = 13",
                "options": ["x = 2", "x = 4", "x = 6", "x = 8"],
                "correct_index": 1,
                "explanation": "Subtract 5, then divide by 2: x = 4"
            },
            {
                "question": "What is 15% of 200?",
                "options": ["25", "30", "35", "40"],
                "correct_index": 1,
                "explanation": "15% of 200 = 0.15 × 200 = 30"
            }
        ],
        "time_limit_seconds": 180,
        "passing_score": 0.7,
        "demo_mode": True
    }

    return QuizMinigame(config)


def create_sample_puzzle() -> PuzzleMinigame:
    """Create sample pattern puzzle"""
    config = {
        "puzzle_type": "pattern",
        "pattern": [2, 4, 8, 16, "?"],
        "answer": 32,
        "hint": "Each number doubles",
        "difficulty": 0.5,
        "demo_mode": True
    }

    return PuzzleMinigame(config)


def create_sample_code_challenge() -> CodeChallengeMinigame:
    """Create sample Python coding challenge"""
    config = {
        "language": "python",
        "challenge": "Write a function that adds two numbers",
        "test_cases": [
            {"input": [2, 3], "output": 5},
            {"input": [10, -5], "output": 5},
            {"input": [0, 0], "output": 0}
        ],
        "starter_code": "def add(a, b):\n    # Your code here\n    pass",
        "difficulty": 0.3,
        "demo_mode": True
    }

    return CodeChallengeMinigame(config)


if __name__ == "__main__":
    print("=== EduVerse Minigame Framework Demo ===\n")

    # Test Quiz
    print("1. Testing Quiz Minigame...")
    quiz = create_sample_quiz()
    result = quiz.run()
    print(f"Result: {result.score:.2%} ({result.correct_answers}/{result.total_questions})")
    print(f"Feedback: {result.feedback}\n")

    # Test Puzzle
    print("2. Testing Puzzle Minigame...")
    puzzle = create_sample_puzzle()
    result = puzzle.run()
    print(f"Result: {result.score:.2%}")
    print(f"Feedback: {result.feedback}\n")

    # Test Code Challenge
    print("3. Testing Code Challenge Minigame...")
    code = create_sample_code_challenge()
    result = code.run()
    print(f"Result: {result.score:.2%} ({result.correct_answers}/{result.total_questions} tests passed)")
    print(f"Feedback: {result.feedback}\n")

    print("Minigame framework ready! All 3 types working.")
