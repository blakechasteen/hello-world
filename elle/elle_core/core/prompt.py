"""
Elle Core - Prompt Composition

Builds the full prompt sent to the LLM by combining:
- Core Elle prompt
- Scene description
- User state
- Memory context
"""
from pathlib import Path
from typing import Optional
from ..models import SceneSnapshot, UserState, UserIntent, MemorySnapshot


def load_core_prompt() -> str:
    """Load the Elle core prompt from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "elle_core_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Core prompt not found: {prompt_path}")
    return prompt_path.read_text()


def format_scene(scene: SceneSnapshot) -> str:
    """Format scene snapshot for the prompt."""
    parts = [
        f"**Location**: {scene.location}",
        f"**Description**: {scene.description}",
    ]

    if scene.objects:
        parts.append("\n**Objects present**:")
        for obj in scene.objects:
            accessibility = "✓" if obj.accessible else "✗"
            visibility = "visible" if obj.visible else "not visible"
            parts.append(f"  - {obj.name} (id: {obj.id}, {visibility}, accessible: {accessibility})")

    if scene.metadata:
        parts.append("\n**Context**:")
        for key, value in scene.metadata.items():
            parts.append(f"  - {key}: {value}")

    return "\n".join(parts)


def format_user_state(state: UserState) -> str:
    """Format user state for the prompt."""
    return (
        f"**Energy**: {state.energy.value}\n"
        f"**Focus**: {state.focus.value}\n"
        f"**Mood**: {state.mood.value}"
    )


def format_user_intent(intent: Optional[UserIntent]) -> str:
    """Format user intent for the prompt."""
    if not intent:
        return "**Intent**: (none explicitly stated)"

    intent_type = "explicit" if intent.explicit else f"inferred ({intent.confidence:.0%} confidence)"
    return f"**Intent** ({intent_type}): {intent.description}"


def format_memory(memory: Optional[MemorySnapshot]) -> str:
    """Format memory snapshot for the prompt."""
    if not memory or not memory.all_memories():
        return "**Relevant memories**: (none)"

    parts = ["**Relevant memories**:"]

    # Show recent memories
    if memory.recent:
        parts.append("  Recent:")
        for mem in memory.recent[:3]:  # Limit to 3 most recent
            parts.append(f"    - {mem.content}")

    # Show contextually relevant memories
    if memory.relevant:
        parts.append("  Contextually relevant:")
        for mem in memory.relevant[:3]:  # Limit to 3 most relevant
            parts.append(f"    - {mem.content}")

    # Show known preferences
    if memory.preferences:
        parts.append("  Known preferences:")
        for key, value in list(memory.preferences.items())[:3]:  # Limit to 3
            parts.append(f"    - {key}: {value}")

    return "\n".join(parts)


def compose_prompt(
    scene: SceneSnapshot,
    user_state: UserState,
    user_intent: Optional[UserIntent] = None,
    memory: Optional[MemorySnapshot] = None,
) -> tuple[str, str]:
    """
    Compose the full prompt for Elle's LLM.

    Args:
        scene: Current scene snapshot
        user_state: Current user state
        user_intent: Optional user intent
        memory: Optional memory snapshot

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # System prompt is the core Elle identity
    system_prompt = load_core_prompt()

    # User prompt contains the specific context
    user_prompt_parts = [
        "# Current Situation\n",
        "## Scene",
        format_scene(scene),
        "\n## User State",
        format_user_state(user_state),
        "\n## User Intent",
        format_user_intent(user_intent),
    ]

    # Add memory context if available
    if memory:
        user_prompt_parts.extend([
            "\n## Memory Context",
            format_memory(memory),
        ])

    user_prompt_parts.append(
        "\n# Your Response\n\n"
        "Based on the above situation, decide how to respond. "
        "Output valid JSON matching the format specified in your instructions."
    )

    user_prompt = "\n".join(user_prompt_parts)

    return system_prompt, user_prompt
