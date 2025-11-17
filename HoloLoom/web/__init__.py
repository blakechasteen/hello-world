"""
Web - Interactive Chat Interface
=================================
FastAPI-based chat interface with streaming responses.
"""

from .app import create_app, ChatConfig

__all__ = ['create_app', 'ChatConfig']
