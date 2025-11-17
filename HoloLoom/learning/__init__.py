"""
HoloLoom Learning Module
========================

Tracks user interactions to improve AI suggestions over time.
"""

from .tracker import LearningTracker, QueryFeedback, load_learning_data

__all__ = ['LearningTracker', 'QueryFeedback', 'load_learning_data']
