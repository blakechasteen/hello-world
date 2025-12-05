"""Chat adapter for text-based Elle interaction."""

from .chat_adapter import ChatAdapter, create_chat_adapter
from .conversation import (
    Conversation,
    ConversationTurn,
    ConversationManager,
    TurnRole,
)
from .chat_scene import (
    create_virtual_scene,
    detect_scene_from_text,
    WORKSHOP_SCENE,
    GARDEN_SCENE,
    OFFICE_SCENE,
)

__all__ = [
    # Main adapter
    "ChatAdapter",
    "create_chat_adapter",
    # Conversation
    "Conversation",
    "ConversationTurn",
    "ConversationManager",
    "TurnRole",
    # Scenes
    "create_virtual_scene",
    "detect_scene_from_text",
    "WORKSHOP_SCENE",
    "GARDEN_SCENE",
    "OFFICE_SCENE",
]
