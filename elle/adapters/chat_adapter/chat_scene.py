"""Virtual scene for chat-only mode.

When chatting with Elle without AR, we create a virtual "thinking space"
that gives her context without requiring actual spatial awareness.
"""

from datetime import datetime
from typing import Optional, List

from elle.domain.scene import SceneSnapshot, ObjectInScene


def create_virtual_scene(
    location: str = "thinking space",
    context_hint: Optional[str] = None,
    projects: Optional[List[str]] = None,
) -> SceneSnapshot:
    """
    Create a virtual scene for chat-only interactions.

    Args:
        location: Virtual location name (e.g., "thinking space", "workshop", "garden")
        context_hint: Optional hint about what user is thinking about
        projects: Optional list of active projects to include as "objects"

    Returns:
        SceneSnapshot representing a virtual environment
    """
    objects = []
    tags = ["virtual", "chat_mode"]

    # Add project-based "objects" if provided
    if projects:
        for i, project in enumerate(projects):
            objects.append(ObjectInScene(
                id=f"project_{i}",
                name=project,
                category="project",
                location="mind",
            ))
        tags.append("has_projects")

    # Build summary based on context
    summary = "A calm, virtual space for thinking and conversation."
    if context_hint:
        summary = f"User is thinking about: {context_hint}"

    return SceneSnapshot(
        scene_id=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now(),
        location=location,
        objects=objects,
        relations=[],
        weather=None,  # Virtual space has no weather
        time_of_day=_get_time_of_day(),
        summary=summary,
        tags=tags,
    )


def _get_time_of_day() -> str:
    """Get approximate time of day."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


# Pre-defined scene templates for common contexts
WORKSHOP_SCENE = SceneSnapshot(
    scene_id="virtual_workshop",
    timestamp=datetime.now(),
    location="virtual workshop",
    objects=[
        ObjectInScene(id="workbench", name="workbench", category="furniture", location="center"),
        ObjectInScene(id="tools", name="assorted tools", category="tools", location="scattered"),
    ],
    relations=[],
    summary="A virtual workshop space for thinking about physical projects.",
    tags=["virtual", "workshop", "making"],
)

GARDEN_SCENE = SceneSnapshot(
    scene_id="virtual_garden",
    timestamp=datetime.now(),
    location="virtual garden",
    objects=[
        ObjectInScene(id="beds", name="garden beds", category="garden", location="around"),
        ObjectInScene(id="plants", name="growing plants", category="plants", location="beds"),
    ],
    relations=[],
    summary="A virtual garden space for thinking about growing things.",
    tags=["virtual", "garden", "nature"],
)

OFFICE_SCENE = SceneSnapshot(
    scene_id="virtual_office",
    timestamp=datetime.now(),
    location="virtual office",
    objects=[
        ObjectInScene(id="desk", name="desk", category="furniture", location="center"),
        ObjectInScene(id="computer", name="computer", category="tech", location="desk"),
    ],
    relations=[],
    summary="A virtual office space for thinking about work.",
    tags=["virtual", "office", "work"],
)


def detect_scene_from_text(text: str) -> SceneSnapshot:
    """
    Infer appropriate virtual scene from user's text.

    Simple keyword matching for now - could be enhanced with NLP.
    """
    text_lower = text.lower()

    # Workshop/making keywords
    workshop_keywords = ["workshop", "shed", "tools", "build", "make", "fix", "repair", "wood", "metal"]
    if any(kw in text_lower for kw in workshop_keywords):
        return WORKSHOP_SCENE

    # Garden keywords
    garden_keywords = ["garden", "plant", "grow", "weed", "soil", "seeds", "harvest", "bees", "hive"]
    if any(kw in text_lower for kw in garden_keywords):
        return GARDEN_SCENE

    # Office/work keywords
    office_keywords = ["work", "office", "computer", "email", "meeting", "project", "deadline"]
    if any(kw in text_lower for kw in office_keywords):
        return OFFICE_SCENE

    # Default: neutral thinking space
    return create_virtual_scene()
