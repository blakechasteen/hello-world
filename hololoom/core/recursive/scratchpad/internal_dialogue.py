#!/usr/bin/env python3
"""
Internal Dialogue Engine
========================

Self-reflective reasoning where the system talks to itself.

Created: 2025-01-20

Philosophy:
"The system doesn't just answer questions—it asks itself questions about
its own answers, recursively exploring understanding."

Dialogue Modes:
- EXPLORATORY: Open-ended questioning
- VERIFICATION: Self-checking and validation
- SYNTHESIS: Integration and pattern finding
- HOFSTADTER: Strange loop detection and level-crossing
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from hololoom.recursive.scratchpad.recursive_scratchpad import (
    DialogueTree,
    Thought,
    ThoughtType,
)


class DialogueMode(Enum):
    """Mode of internal dialogue."""
    EXPLORATORY = "exploratory"     # Open-ended exploration
    VERIFICATION = "verification"   # Self-verification
    SYNTHESIS = "synthesis"         # Integration
    HOFSTADTER = "hofstadter"      # Strange loops


@dataclass
class DialogueStep:
    """Single step in dialogue."""
    thought: Thought
    question: str | None = None
    answer: Thought | None = None
    reflection: Thought | None = None


class InternalDialogue:
    """
    Internal dialogue engine.

    Drives recursive self-questioning and reflection.
    """

    def __init__(
        self,
        max_depth: int = 10,
        enable_verification: bool = True,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize dialogue engine.

        Args:
            max_depth: Maximum dialogue depth
            enable_verification: Enable DS-STAR verification
            confidence_threshold: Confidence threshold for convergence
        """
        self.max_depth = max_depth
        self.enable_verification = enable_verification
        self.confidence_threshold = confidence_threshold

        # Question generators for each mode
        self.question_generators: dict[DialogueMode, Callable] = {
            DialogueMode.EXPLORATORY: self._generate_exploratory_questions,
            DialogueMode.VERIFICATION: self._generate_verification_questions,
            DialogueMode.SYNTHESIS: self._generate_synthesis_questions,
            DialogueMode.HOFSTADTER: self._generate_hofstadter_questions,
        }

    async def start_dialogue(
        self,
        initial_thought: Thought,
        max_depth: int | None = None,
        mode: str = "exploratory"
    ) -> DialogueTree:
        """
        Start internal dialogue from initial thought.

        Args:
            initial_thought: Starting thought
            max_depth: Maximum depth (None = use default)
            mode: Dialogue mode

        Returns:
            Complete dialogue tree
        """
        max_depth = max_depth or self.max_depth
        dialogue_mode = DialogueMode(mode)

        # Create tree
        tree = DialogueTree(root=initial_thought)

        # Dialogue loop
        await self._dialogue_recursive(
            tree=tree,
            current_thought=initial_thought,
            mode=dialogue_mode,
            current_depth=0,
            max_depth=max_depth
        )

        return tree

    async def _dialogue_recursive(
        self,
        tree: DialogueTree,
        current_thought: Thought,
        mode: DialogueMode,
        current_depth: int,
        max_depth: int
    ) -> None:
        """
        Recursive dialogue exploration.

        Args:
            tree: Dialogue tree
            current_thought: Current thought
            mode: Dialogue mode
            current_depth: Current depth
            max_depth: Maximum depth
        """
        # Stop conditions
        if current_depth >= max_depth:
            return

        if current_thought.confidence >= self.confidence_threshold:
            # High confidence - maybe ask one synthesis question
            if mode != DialogueMode.SYNTHESIS:
                return

        # Generate questions about current thought
        questions = await self._generate_questions(current_thought, mode)

        # For each question, create answer and recurse
        for question_text in questions:
            # Create question thought
            question = Thought(
                id=f"q_{len(tree.thoughts)}",
                text=question_text,
                type=ThoughtType.QUESTION,
                level=current_depth + 1,
                timestamp=current_thought.timestamp,
                parent_id=current_thought.id
            )
            tree.add_thought(question)

            # Generate answer
            answer = await self._generate_answer(question, current_thought, tree)
            tree.add_thought(answer)

            # Maybe add reflection
            if answer.confidence < 0.6 or mode == DialogueMode.HOFSTADTER:
                reflection = await self._generate_reflection(answer, tree)
                if reflection:
                    tree.add_thought(reflection)
                    # Recurse on reflection
                    await self._dialogue_recursive(
                        tree=tree,
                        current_thought=reflection,
                        mode=mode,
                        current_depth=current_depth + 2,
                        max_depth=max_depth
                    )
            else:
                # Recurse on answer
                await self._dialogue_recursive(
                    tree=tree,
                    current_thought=answer,
                    mode=mode,
                    current_depth=current_depth + 1,
                    max_depth=max_depth
                )

    async def _generate_questions(
        self,
        thought: Thought,
        mode: DialogueMode
    ) -> list[str]:
        """
        Generate questions about a thought.

        Args:
            thought: Thought to question
            mode: Dialogue mode

        Returns:
            List of questions
        """
        generator = self.question_generators.get(mode)
        if not generator:
            return []

        return await generator(thought)

    async def _generate_exploratory_questions(
        self,
        thought: Thought
    ) -> list[str]:
        """Generate open-ended exploratory questions."""
        questions = []

        # What questions
        if "is" in thought.text.lower() or "are" in thought.text.lower():
            questions.append("What exactly does that mean?")

        # Why questions
        questions.append("Why is this the case?")

        # How questions
        if thought.confidence < 0.7:
            questions.append("How can I be more certain about this?")

        # Alternative perspectives
        questions.append("What if the opposite were true?")

        return questions[:2]  # Limit to 2 questions per thought

    async def _generate_verification_questions(
        self,
        thought: Thought
    ) -> list[str]:
        """Generate verification questions (DS-STAR)."""
        questions = []

        # Domain check
        questions.append("Is this relevant to the original question?")

        # Sensibility check
        questions.append("Does this make logical sense?")

        # Argument check
        if thought.verification:
            if thought.verification.get('argument', 1.0) < 0.7:
                questions.append("Do I have evidence to support this claim?")

        # Reference check
        if thought.verification:
            if thought.verification.get('reference', 1.0) < 0.7:
                questions.append("Can I trace this back to reliable sources?")

        return questions[:2]

    async def _generate_synthesis_questions(
        self,
        thought: Thought
    ) -> list[str]:
        """Generate synthesis questions."""
        questions = []

        # Integration
        questions.append("How does this connect to what I learned before?")

        # Pattern recognition
        questions.append("What pattern am I seeing here?")

        # Meta-insight
        if thought.level > 3:
            questions.append("What's the big picture insight?")

        return questions[:1]  # Synthesis is slower

    async def _generate_hofstadter_questions(
        self,
        thought: Thought
    ) -> list[str]:
        """Generate Hofstadter-style strange loop questions."""
        questions = []

        # Self-reference
        questions.append("What assumptions am I making in thinking this?")

        # Level-crossing
        if thought.level > 2:
            questions.append("How does thinking about this change my understanding of the original question?")

        # Meta-reasoning
        questions.append("Is my reasoning process itself flawed?")

        return questions[:1]  # Strange loops are deep

    async def _generate_answer(
        self,
        question: Thought,
        context_thought: Thought,
        tree: DialogueTree
    ) -> Thought:
        """
        Generate answer to a question.

        Args:
            question: Question thought
            context_thought: Context thought
            tree: Dialogue tree

        Returns:
            Answer thought
        """
        # In production, this would call RAG/HoloLoom
        # For now, generate synthetic answer

        answer_text = self._synthesize_answer(question.text, context_thought, tree)

        answer = Thought(
            id=f"a_{len(tree.thoughts)}",
            text=answer_text,
            type=ThoughtType.ANSWER,
            level=question.level + 1,
            timestamp=question.timestamp,
            parent_id=question.id,
            confidence=0.6 + (0.2 * (1 - question.level / self.max_depth))  # Confidence decreases with depth
        )

        return answer

    def _synthesize_answer(
        self,
        question: str,
        context: Thought,
        tree: DialogueTree
    ) -> str:
        """Synthesize answer from question and context."""
        # Simple heuristic-based answer generation
        # In production, this would use RAG/HoloLoom

        q_lower = question.lower()

        if "why" in q_lower:
            return f"Because {context.text.lower().split('.')[0]}."

        if "how can i be more certain" in q_lower:
            return "I could gather more evidence or cross-reference with other sources."

        if "what if the opposite" in q_lower:
            return "If the opposite were true, it would contradict current understanding and require re-evaluation."

        if "does this make logical sense" in q_lower:
            return "Yes, this follows logically from the premises."

        if "how does this connect" in q_lower:
            return "This builds on earlier thoughts by extending the reasoning."

        if "what pattern" in q_lower:
            return "The pattern shows increasing refinement through recursive questioning."

        if "what assumptions" in q_lower:
            return "I'm assuming the initial framing is correct and that reasoning is sufficient."

        if "is my reasoning process itself flawed" in q_lower:
            return "Possibly - all reasoning has limitations and blind spots."

        # Default
        return f"Exploring: {question}"

    async def _generate_reflection(
        self,
        answer: Thought,
        tree: DialogueTree
    ) -> Thought | None:
        """
        Generate reflection on an answer.

        Args:
            answer: Answer thought
            tree: Dialogue tree

        Returns:
            Reflection thought or None
        """
        # Reflect if confidence is low or depth is high
        if answer.confidence < 0.5 or answer.level > 5:
            reflection = Thought(
                id=f"r_{len(tree.thoughts)}",
                text=f"This reasoning might be incomplete. Confidence is {answer.confidence:.2f}.",
                type=ThoughtType.REFLECTION,
                level=answer.level + 1,
                timestamp=answer.timestamp,
                parent_id=answer.id,
                confidence=answer.confidence * 0.9  # Reflection inherits but slightly lower
            )
            return reflection

        return None

    def detect_convergence(self, tree: DialogueTree) -> bool:
        """
        Detect if dialogue has converged.

        Args:
            tree: Dialogue tree

        Returns:
            True if converged
        """
        # Check if recent thoughts have high confidence
        recent_thoughts = sorted(
            tree.thoughts.values(),
            key=lambda t: t.timestamp,
            reverse=True
        )[:5]

        avg_confidence = sum(t.confidence for t in recent_thoughts) / len(recent_thoughts)

        return avg_confidence >= self.confidence_threshold
