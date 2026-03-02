"""
System Skills Package
=====================
Skills for system-level operations and utilities.

Available Skills:
- file_search: Fast file content search using ripgrep
"""

from .file_search_skill import FileSearchSkill

__all__ = ['FileSearchSkill']
