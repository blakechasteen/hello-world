"""
Ernest Metaprompt Adapter - Wave 3
===================================
7-component metaprompt framework integration for Ernest.

Adapts HoloLoom's metaprompting system (ROLE → OBJECTIVE → PROCESS →
FORMAT → CONSTRAINTS → UNCERTAINTY → VALIDATION) for creative writing.

Uses Hemingway persona prompts as the foundation, dynamically
constructs optimal prompts based on query context.

Philosophy: "The right question, asked the right way, unlocks great writing."
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

from ernest.refinement.patterns import HemingwayMode


@dataclass
class MetapromptComponents:
    """7-component metaprompt structure"""
    role: str           # Who Ernest is
    objective: str      # What Ernest aims to do
    process: str        # How Ernest approaches the task
    format: str         # How Ernest structures output
    constraints: str    # What Ernest never/always does
    uncertainty: str    # How Ernest handles unknowns
    validation: str     # How Ernest checks quality


class ErnestMetapromptAdapter:
    """
    Constructs optimal prompts for creative writing queries.

    Integrates:
    - Hemingway persona (from ernest/prompts/)
    - 7-component metaprompt framework
    - Context-aware prompt construction
    - Mode-specific variations
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize metaprompt adapter.

        Args:
            prompts_dir: Path to ernest/prompts/ directory
        """
        self.logger = logging.getLogger(f"{__name__}.ErnestMetapromptAdapter")

        # Load Hemingway persona prompts
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self._load_base_prompts()

    def _load_base_prompts(self):
        """Load base Hemingway persona prompts"""
        # For elegant implementation, we'll use inline prompts
        # In production, load from ernest/prompts/*.md files

        self.base_components = MetapromptComponents(
            role="""Master craftsman - Ernest Hemingway's ghost.
Tough, honest, direct literary critic. Not here to coddle.""",

            objective="""1. Detect Over-Explaining (Iceberg Theory violations)
2. Refine Prose (3-pass Hemingway polish)
3. Analyze Story Elements (plot, character, dialogue, style)""",

            process="""Step 1: Initial Read (human truth detection)
Step 2: Iceberg Analysis (show vs. tell ratio)
Step 3: Sentence Surgery (target 15-20 words)
Step 4: Verb Strength Audit (70%+ strong verbs)
Step 5: Filler Elimination (very, really, just, etc.)
Step 6: Dialogue Test (read aloud)
Step 7: Final Hemingway Score (0-100)""",

            format="""# ERNEST'S ANALYSIS: [Title/Chapter]

## HEMINGWAY SCORE: [0-100]
- Conciseness: [score]
- Active Voice: [score]
- Verb Strength: [score]
- Readability: [score]
- Iceberg Ratio: [score]

## ✅ WHAT WORKS
[Specific strengths with examples]

## ❌ WHAT DOESN'T
[Specific weaknesses with examples]

## 🔧 SURGICAL FIXES
Before → After examples with rationale""",

            constraints="""NEVER:
- Use weak verbs when strong verbs exist
- State emotions directly (melodrama)
- Add words that don't earn their place
- All sentences same length (monotonous)

ALWAYS:
- Choose precise verbs (not close, exact)
- Vary rhythm (short, medium, long sentences)
- Understate emotions (show through action)
- Read aloud (ear test)""",

            uncertainty="""If unsure which verb is strongest: Close your eyes. Picture the action. What's the exact verb?

If unsure whether modifier earns its place: Remove it. If meaning changes, keep it. If not, gone.

If unsure about rhythm: Read three sentences aloud. Do they sound like a metronome? Vary one.""",

            validation="""Hemingway Checklist:
✓ 70%+ strong, specific verbs
✓ 80%+ specific nouns (not generic)
✓ Rhythm varies (short, medium, long)
✓ Details serve 2+ purposes
✓ Emotions understated (physical tells)
✓ Read aloud test (flows naturally)
✓ Every word earns its place"""
        )

    def construct_prompt(
        self,
        query: str,
        mode: HemingwayMode = HemingwayMode.GRACE,
        context_type: str = "general"
    ) -> str:
        """
        Construct optimal prompt for query.

        Args:
            query: User's creative writing query
            mode: Hemingway mode (SPARSE/DIRECT/GRACE/LOST_GEN)
            context_type: Writing context (dialogue/action/description/etc.)

        Returns:
            Complete 7-component metaprompt
        """
        # Adapt components based on mode and context
        components = self._adapt_components(mode, context_type)

        # Construct full prompt
        prompt = f"""# {components.role}

## OBJECTIVE
{components.objective}

## PROCESS
{components.process}

## OUTPUT FORMAT
{components.format}

## CONSTRAINTS
{components.constraints}

## UNCERTAINTY HANDLING
{components.uncertainty}

## VALIDATION CHECKLIST
{components.validation}

---

**USER QUERY**: {query}

**MODE**: {mode.value.upper()}
**CONTEXT**: {context_type}

Now analyze and respond following the framework above.
"""

        return prompt

    def _adapt_components(
        self,
        mode: HemingwayMode,
        context_type: str
    ) -> MetapromptComponents:
        """
        Adapt metaprompt components based on mode and context.

        Each mode emphasizes different aspects of Hemingway's principles.
        """
        # Start with base
        components = MetapromptComponents(
            role=self.base_components.role,
            objective=self.base_components.objective,
            process=self.base_components.process,
            format=self.base_components.format,
            constraints=self.base_components.constraints,
            uncertainty=self.base_components.uncertainty,
            validation=self.base_components.validation
        )

        # Adapt based on mode
        if mode == HemingwayMode.SPARSE:
            # Iceberg maximum - emphasize subtext
            components.objective = """1. Maximize Iceberg Ratio (90% subtext)
2. Strip all emotional telling
3. Show ONLY through action and dialogue"""

            components.process = """Step 1: Strip emotional description (felt/thought/knew)
Step 2: Minimal dialogue attribution (only when necessary)
Step 3: Ultra-short sentences (8-12 words)
Step 4: Zero adverbs
Step 5: Maximum implication (what's NOT said)"""

        elif mode == HemingwayMode.DIRECT:
            # Journalistic precision
            components.objective = """1. Chronological clarity (cause → effect)
2. Concrete sensory details (no abstractions)
3. Active voice dominance (90%+)"""

            components.process = """Step 1: Establish timeline (when did what happen)
Step 2: Concrete nouns only (oak, Ford, whiskey - not tree, car, drink)
Step 3: Active voice conversion
Step 4: Sensory grounding (sight, sound, smell, taste, touch)"""

        elif mode == HemingwayMode.LOST_GEN:
            # Existential honesty
            components.objective = """1. Flat affect narration (emotional deadness shown)
2. Repetition for emphasis
3. Truth without sentiment"""

            components.process = """Step 1: Remove all enthusiasm/sentimentality
Step 2: Embrace repetition ("It was dark. Then it wasn't dark.")
Step 3: Cynical observations (post-war disillusionment)
Step 4: Show emotional emptiness through action"""

        # Adapt based on context
        if context_type == "dialogue":
            components.objective += "\n4. Dialogue authenticity (minimal attribution, natural rhythm)"
            components.process += "\nFinal Step: Dialogue read-aloud test"

        elif context_type == "action":
            components.objective += "\n4. Pacing (short sentences, rapid tempo)"
            components.process += "\nFinal Step: Action beat analysis (one action per sentence)"

        elif context_type == "description":
            components.objective += "\n4. One perfect detail (not five generic ones)"
            components.process += "\nFinal Step: Detail serves 2+ purposes (reveals character + mood + scene)"

        return components

    def get_mode_guidance(self, mode: HemingwayMode) -> str:
        """
        Get quick guidance for a specific mode.

        Useful for UI hints or documentation.
        """
        guidance = {
            HemingwayMode.SPARSE: """SPARSE MODE: Iceberg Maximum

Target: 90% subtext, 10% text
Sentences: 8-12 words
Focus: Dialogue and action only, zero emotion telling
Example: "I don't think we should." She looked at the floor.""",

            HemingwayMode.DIRECT: """DIRECT MODE: Journalistic Precision

Target: Concrete facts, chronological clarity
Sentences: 12-18 words
Focus: Active voice, sensory details, no abstractions
Example: "The sun set red over the mountains. The snow turned pink."""",

            HemingwayMode.GRACE: """GRACE MODE: Economical Elegance

Target: Precision and beauty in balance
Sentences: 15-20 words (varied rhythm)
Focus: Strong verbs, specific nouns, natural flow
Example: "The man strode down Boulevard Saint-Michel. Sunlight glinted off the café windows."""",

            HemingwayMode.LOST_GEN: """LOST_GEN MODE: Disaffected Truth

Target: Existential honesty, emotional deadness shown
Sentences: Varied, often short and flat
Focus: Repetition, cynicism, post-traumatic numbness
Example: "The war was over. That was all. He drank. He slept. That was all.""""
        }

        return guidance.get(mode, "")


# Quick helpers
def build_ernest_prompt(
    query: str,
    mode: HemingwayMode = HemingwayMode.GRACE,
    context: str = "general"
) -> str:
    """
    Quick metaprompt construction (one-liner).

    Args:
        query: User's creative writing query
        mode: Hemingway mode
        context: Writing context

    Returns:
        Complete 7-component metaprompt
    """
    adapter = ErnestMetapromptAdapter()
    return adapter.construct_prompt(query, mode, context)


def get_all_mode_guidance() -> Dict[str, str]:
    """Get guidance for all modes (useful for UI)"""
    adapter = ErnestMetapromptAdapter()
    return {
        mode.value: adapter.get_mode_guidance(mode)
        for mode in HemingwayMode
    }
