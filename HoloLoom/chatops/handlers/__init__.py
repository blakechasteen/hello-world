"""
HoloLoom ChatOps - Advanced Handlers
=====================================

Optional advanced features:
- multimodal_handler: Image and file processing
- thread_handler: Thread-aware responses
- proactive_agent: Proactive suggestions
- hololoom_handlers: HoloLoom-specific handlers
- pattern_tuning: Pattern optimization
"""

# Graceful imports - these are optional
__all__ = []

try:
    from HoloLoom.chatops.handlers.multimodal_handler import MultimodalHandler
    __all__.append("MultimodalHandler")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.thread_handler import ThreadHandler
    __all__.append("ThreadHandler")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.proactive_agent import ProactiveAgent
    __all__.append("ProactiveAgent")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.hololoom_handlers import HoloLoomHandlers
    __all__.append("HoloLoomHandlers")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.pattern_tuning import PatternTuner
    __all__.append("PatternTuner")
except ImportError:
    pass
