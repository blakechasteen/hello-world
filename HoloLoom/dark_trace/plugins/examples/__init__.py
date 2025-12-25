"""
Dark Trace Plugin Examples
==========================

Example plugins demonstrating the Dark Trace plugin architecture.

Available Examples:
- HelloWorldPlugin: Minimal example showing plugin basics

Date: December 2025
Phase: 11 - Plugin Ecosystem
"""

from HoloLoom.dark_trace.plugins.examples.hello_plugin import (
    HelloWorldPlugin,
    HelloAnalysisResult,
    create_test_template,
)

__all__ = [
    "HelloWorldPlugin",
    "HelloAnalysisResult",
    "create_test_template",
]
