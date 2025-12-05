"""Request normalization and metadata."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from ..domain import SceneSnapshot, UserIntent, UserState, ElleRequest


@dataclass(frozen=True)
class RequestMeta:
    """Metadata about the request itself."""
    
    request_id: str
    timestamp: datetime
    source_adapter: str  # "ar", "matrix", "cli"
    
    # Optional trace info
    trace_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_request(
    scene: SceneSnapshot,
    intent: UserIntent,
    user: UserState,
    source_adapter: str,
    **kwargs
) -> ElleRequest:
    """
    Factory for creating normalized requests.
    
    Ensures all requests have proper IDs and timestamps.
    """
    
    request_id = kwargs.pop('request_id', str(uuid.uuid4()))
    timestamp = kwargs.pop('timestamp', datetime.now())
    
    return ElleRequest(
        scene=scene,
        intent=intent,
        user=user,
        request_id=request_id,
        timestamp=timestamp,
        source_adapter=source_adapter,
        metadata=kwargs
    )
