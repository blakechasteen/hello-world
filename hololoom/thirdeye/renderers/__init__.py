"""
ThirdEye Renderers
==================

Rendering implementations for different visualization modes.

Available Renderers:
- WebGLRenderer: Three.js-based 3D rendering
- (Future) TufteRenderer: 2D SVG/CSS Tufte-style
- (Future) ARRenderer: WebXR augmented reality

Created: 2025-11-30
Author: HoloLoom Team
"""

from HoloLoom.thirdeye.renderers.webgl import (
    WebGLRenderer,
    create_webgl_renderer,
)

__all__ = [
    'WebGLRenderer',
    'create_webgl_renderer',
]
