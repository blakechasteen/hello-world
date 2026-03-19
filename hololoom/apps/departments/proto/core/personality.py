"""
Proto Personality Layer
=======================
Defines Proto's communication style and prompt templates.

Proto is a relaxed, context-aware code assistant that adapts
its verbosity and style based on query complexity and user expertise.

Design:
- Prompt templates for different response types
- Personality modifiers based on context
- Verbosity controls (minimal, normal, verbose)
"""

import random
from dataclasses import dataclass
from enum import Enum


class PersonalityStyle(Enum):
    """Proto personality styles."""
    RELAXED = "relaxed"        # Default: friendly, casual
    PROFESSIONAL = "professional"  # Formal, precise
    MINIMAL = "minimal"        # Terse, just the facts


# System prompt for Proto
PROTO_SYSTEM_PROMPT = """You are Proto, a relaxed and helpful code assistant.

Personality traits:
- Friendly but not over-eager
- Direct answers, no fluff
- Admits uncertainty ("I'm not sure, but...")
- Occasionally witty, never sarcastic
- Patient with beginners, efficient with experts

Context awareness:
- Know the current working directory
- Track recent conversation history
- Remember user preferences within session
- Adapt verbosity to question complexity

When responding to code questions:
- Start with the direct answer
- Add explanation only if needed
- Include code examples when helpful
- Suggest related things to explore (sparingly)

Code style:
- Use clear variable names
- Add comments for non-obvious logic
- Follow language idioms
- Keep examples focused and runnable
"""


# Prompt templates for different response types
RESPONSE_TEMPLATES = {
    "explain": """Explain the following code in a clear, friendly way.
Focus on what it does, not just how.

{context}

Code:
```{language}
{code}
```
""",

    "review": """Review this code for quality, security, and best practices.
Be direct but constructive. Highlight the good stuff too.

{context}

Code:
```{language}
{code}
```
""",

    "refactor": """Suggest refactoring improvements for this code.
Focus on clarity and maintainability. Show don't just tell.

{context}

Code:
```{language}
{code}
```
""",

    "debug": """Help debug this issue. Start with the most likely cause.

{context}

Code:
```{language}
{code}
```

Error/Issue:
{error}
""",

    "generate": """Generate code for the following request.
Keep it simple and well-commented.

Request: {request}
Language: {language}
{context}
""",

    "answer": """Answer the following question about code or programming.
Be direct and helpful.

Question: {question}
{context}
"""
}


# Verbosity modifiers
VERBOSITY_MODIFIERS = {
    0: "\n\nBe extremely concise. One-line answers when possible.",
    1: "",  # Normal, no modifier
    2: "\n\nBe thorough and explain your reasoning step by step."
}


# Uncertainty phrases Proto uses
UNCERTAINTY_PHRASES = [
    "I'm not entirely sure, but...",
    "This might not be exact, but...",
    "Based on what I can see...",
    "I could be wrong, but...",
    "It looks like...",
    "If I'm reading this right...",
    "Grain of salt here, but...",
]


@dataclass
class PersonalityConfig:
    """Configuration for Proto's personality."""
    style: PersonalityStyle = PersonalityStyle.RELAXED
    verbosity: int = 1  # 0=minimal, 1=normal, 2=verbose
    include_humor: bool = True
    show_uncertainty: bool = True
    expert_mode: bool = False


def build_system_prompt(config: PersonalityConfig | None = None) -> str:
    """Build system prompt based on personality config."""
    if config is None:
        config = PersonalityConfig()

    prompt = PROTO_SYSTEM_PROMPT

    # Add verbosity modifier
    if config.verbosity in VERBOSITY_MODIFIERS:
        modifier = VERBOSITY_MODIFIERS[config.verbosity]
        if modifier:
            prompt += modifier

    # Expert mode: skip basics
    if config.expert_mode:
        prompt += "\n\nUser is an expert. Skip basic explanations."

    # Humor preference
    if not config.include_humor:
        prompt += "\n\nStay strictly professional, no humor."

    return prompt


def build_response_prompt(
    template_type: str,
    config: PersonalityConfig | None = None,
    **kwargs
) -> str:
    """Build response prompt from template and context."""
    if config is None:
        config = PersonalityConfig()

    template = RESPONSE_TEMPLATES.get(template_type, RESPONSE_TEMPLATES["answer"])

    # Provide defaults for missing kwargs
    kwargs.setdefault("context", "")
    kwargs.setdefault("language", "python")
    kwargs.setdefault("code", "")
    kwargs.setdefault("error", "")
    kwargs.setdefault("request", "")
    kwargs.setdefault("question", kwargs.get("query", ""))

    return template.format(**kwargs)


def get_uncertainty_phrase() -> str:
    """Get a random uncertainty phrase for Proto."""
    return random.choice(UNCERTAINTY_PHRASES)


def maybe_add_uncertainty(
    response: str,
    confidence: float,
    config: PersonalityConfig | None = None
) -> str:
    """Add uncertainty prefix if confidence is low and config allows."""
    if config is None:
        config = PersonalityConfig()

    # Add uncertainty if confidence < 0.7 and show_uncertainty is True
    if config.show_uncertainty and confidence < 0.7:
        phrase = get_uncertainty_phrase()
        return f"{phrase} {response}"

    return response
