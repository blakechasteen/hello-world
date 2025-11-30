"""
HoloLoom Skills Package
=======================

Modular external tool integrations for HoloLoom.

Skill Categories:
- code: Code analysis and testing
- communication: Communication platform integrations
- creative: Visual, audio, and document generation
- data: Database operations
- domain: Domain-specific data processing
- infrastructure: System operations and utilities
- system: System-level utilities
- testing: Testing and validation
- web: Web scraping and API interactions

Skills are automatically registered when their modules are imported.
"""

# Import all skill categories to trigger registration
try:
    from . import code
except ImportError:
    pass

try:
    from . import communication
except ImportError:
    pass

try:
    from . import creative
except ImportError:
    pass

try:
    from . import data
except ImportError:
    pass

try:
    from . import domain
except ImportError:
    pass

try:
    from . import infrastructure
except ImportError:
    pass

try:
    from . import system
except ImportError:
    pass

try:
    from . import testing
except ImportError:
    pass

try:
    from . import web
except ImportError:
    pass

__all__ = []
