"""Adapter stubs for AR and Matrix."""

# These are placeholders for the actual adapters.
# They will be implemented when AR/Matrix integration happens.

# AR Adapter contract:
# - Listen to AR events (gaze, scan, voice)
# - Convert to UserIntent + SceneSnapshot
# - Send to ElleEngine
# - Render ElleAction in AR space

# Matrix Adapter contract:
# - Listen to Matrix room events
# - Convert to normalized event objects
# - Send to ElleEngine
# - Return responses as Matrix messages/JSON

from ..domain import ElleRequest, ElleAction


class AdapterInterface:
    """Base interface all adapters should follow."""
    
    def normalize_event(self, raw_event) -> ElleRequest:
        """Convert platform-specific event to ElleRequest."""
        raise NotImplementedError
    
    def render_action(self, action: ElleAction) -> None:
        """Render ElleAction in platform-specific way."""
        raise NotImplementedError
