"""
HoloLoom Distributed Module

This module provides distributed computing capabilities for HoloLoom:
- Celery-based worker pools
- RabbitMQ message queue abstraction
- Redis caching layer
- Distributed task coordination
"""

from .worker import app as celery_app
from .queue import QueueManager
from .cache import CacheManager
from .coordinator import TaskCoordinator

__all__ = [
    'celery_app',
    'QueueManager',
    'CacheManager',
    'TaskCoordinator',
]
