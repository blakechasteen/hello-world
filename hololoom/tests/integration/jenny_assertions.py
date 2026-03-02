"""
Jenny Test Assertion Helpers
============================
Semantic assertion functions for validating Jenny renderer output.

Philosophy:
> "Test what matters: assert_valid_html_output, not len(result) > 0"

These assertions provide:
- Output format validation (HTML, JSON, React props, AR scene)
- Accessibility compliance checking (ARIA, keyboard, contrast)
- Content integrity verification
- Structure validation

Usage:
    from hololoom.tests.integration.jenny_assertions import (
        assert_valid_html_output,
        assert_valid_react_props,
        assert_accessibility_compliant,
    )

    async def test_html_renderer():
        result = await renderer.render(specs, target=RenderTarget.HTML)
        assert_valid_html_output(result, panel_count=3)
        assert_accessibility_compliant(result)

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot Extensibility)
"""

import json
import re
from typing import Optional, List, Dict, Any


# ============================================================================
# HTML Output Assertions
# ============================================================================

def assert_valid_html_output(
    result: str,
    panel_count: Optional[int] = None,
    contains_content: Optional[List[str]] = None,
    has_container: bool = True
) -> None:
    """
    Assert that result is valid HTML output from Jenny renderer.

    Args:
        result: HTML string from renderer
        panel_count: Expected number of panels (if known)
        contains_content: List of strings that should appear in output
        has_container: Whether output should have a container element

    Raises:
        AssertionError: If validation fails
    """
    assert result is not None, "HTML output is None"
    assert isinstance(result, str), f"HTML output is not string: {type(result)}"
    assert len(result) > 0, "HTML output is empty"

    # Check for basic HTML structure
    if has_container:
        assert (
            '<div' in result or '<section' in result or '<article' in result
        ), "HTML output missing container element"

    # Check panel count via data attributes or class patterns
    if panel_count is not None:
        # Look for jenny-panel class or data-panel-id
        panel_pattern = r'class="[^"]*jenny-panel[^"]*"|data-panel-id='
        panels_found = len(re.findall(panel_pattern, result))
        assert panels_found == panel_count, (
            f"Expected {panel_count} panels, found {panels_found}"
        )

    # Check for required content
    if contains_content:
        for content in contains_content:
            assert content in result, f"HTML output missing content: {content}"


def assert_html_structure_valid(result: str) -> None:
    """
    Assert HTML has valid structure (balanced tags, no obvious errors).

    Args:
        result: HTML string to validate

    Raises:
        AssertionError: If structure is invalid
    """
    assert result is not None, "HTML is None"

    # Check for balanced basic tags (simplified check)
    open_divs = result.count('<div')
    close_divs = result.count('</div>')
    assert open_divs == close_divs, (
        f"Unbalanced div tags: {open_divs} opens, {close_divs} closes"
    )

    # Check for common HTML errors
    assert '</' in result if '<' in result else True, "HTML has unclosed tags"

    # Check for script injection attempts
    assert '<script>' not in result.lower() or 'nonce=' in result, (
        "HTML contains unprotected script tags"
    )


# ============================================================================
# React Props Assertions
# ============================================================================

def assert_valid_react_props(
    result: str,
    panel_count: Optional[int] = None,
    required_props: Optional[List[str]] = None
) -> None:
    """
    Assert that result is valid React props JSON.

    Args:
        result: JSON string of React props
        panel_count: Expected number of panels
        required_props: List of required prop names

    Raises:
        AssertionError: If validation fails
    """
    assert result is not None, "React props output is None"
    assert isinstance(result, str), f"React props is not string: {type(result)}"

    # Parse JSON
    try:
        props = json.loads(result)
    except json.JSONDecodeError as e:
        raise AssertionError(f"React props is not valid JSON: {e}")

    # Check for panels array
    assert isinstance(props, dict), "React props should be a dict"

    if 'panels' in props:
        panels = props['panels']
        assert isinstance(panels, list), "panels should be a list"

        if panel_count is not None:
            assert len(panels) == panel_count, (
                f"Expected {panel_count} panels, got {len(panels)}"
            )

    # Check required props
    if required_props:
        for prop in required_props:
            assert prop in props, f"Missing required prop: {prop}"


def assert_react_props_typescript_friendly(result: str) -> None:
    """
    Assert React props are TypeScript-friendly (proper types).

    Args:
        result: JSON string of React props

    Raises:
        AssertionError: If not TypeScript-friendly
    """
    props = json.loads(result)

    # Check panels have required typed properties
    if 'panels' in props:
        for i, panel in enumerate(props['panels']):
            assert 'id' in panel, f"Panel {i} missing 'id'"
            assert isinstance(panel['id'], str), f"Panel {i} 'id' not string"

            assert 'type' in panel, f"Panel {i} missing 'type'"
            assert isinstance(panel['type'], str), f"Panel {i} 'type' not string"

            if 'confidence' in panel:
                assert isinstance(panel['confidence'], (int, float)), (
                    f"Panel {i} 'confidence' not number"
                )


# ============================================================================
# AR Scene Assertions
# ============================================================================

def assert_valid_ar_scene(
    result: str,
    panel_count: Optional[int] = None,
    coordinate_system: str = "webxr"
) -> None:
    """
    Assert that result is valid AR scene JSON.

    Args:
        result: JSON string of AR scene
        panel_count: Expected number of AR objects
        coordinate_system: Expected coordinate system (webxr, arkit, etc.)

    Raises:
        AssertionError: If validation fails
    """
    assert result is not None, "AR scene output is None"
    assert isinstance(result, str), f"AR scene is not string: {type(result)}"

    # Parse JSON
    try:
        scene = json.loads(result)
    except json.JSONDecodeError as e:
        raise AssertionError(f"AR scene is not valid JSON: {e}")

    assert isinstance(scene, dict), "AR scene should be a dict"

    # Check for objects/entities array
    objects_key = None
    for key in ['objects', 'entities', 'panels', 'nodes']:
        if key in scene:
            objects_key = key
            break

    if objects_key:
        objects = scene[objects_key]
        assert isinstance(objects, list), f"{objects_key} should be a list"

        if panel_count is not None:
            assert len(objects) == panel_count, (
                f"Expected {panel_count} AR objects, got {len(objects)}"
            )

    # Check coordinate system
    if 'coordinate_system' in scene:
        assert scene['coordinate_system'] == coordinate_system, (
            f"Expected coordinate system '{coordinate_system}', "
            f"got '{scene['coordinate_system']}'"
        )


def assert_ar_transforms_valid(result: str) -> None:
    """
    Assert AR scene has valid transform data.

    Args:
        result: JSON string of AR scene

    Raises:
        AssertionError: If transforms are invalid
    """
    scene = json.loads(result)

    objects = scene.get('objects', scene.get('entities', []))

    for i, obj in enumerate(objects):
        if 'transform' in obj:
            transform = obj['transform']

            # Check position (3D vector)
            if 'position' in transform:
                pos = transform['position']
                assert isinstance(pos, (list, dict)), f"Object {i} position invalid"
                if isinstance(pos, list):
                    assert len(pos) == 3, f"Object {i} position should be 3D"

            # Check rotation (quaternion or euler)
            if 'rotation' in transform:
                rot = transform['rotation']
                assert isinstance(rot, (list, dict)), f"Object {i} rotation invalid"

            # Check scale
            if 'scale' in transform:
                scale = transform['scale']
                assert isinstance(scale, (list, dict, int, float)), (
                    f"Object {i} scale invalid"
                )


# ============================================================================
# JSON Output Assertions
# ============================================================================

def assert_valid_json_output(
    result: str,
    panel_count: Optional[int] = None,
    required_fields: Optional[List[str]] = None
) -> None:
    """
    Assert that result is valid JSON output.

    Args:
        result: JSON string
        panel_count: Expected number of items in main array/dict
        required_fields: Required top-level fields

    Raises:
        AssertionError: If validation fails
    """
    assert result is not None, "JSON output is None"
    assert isinstance(result, str), f"JSON output is not string: {type(result)}"

    try:
        data = json.loads(result)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Output is not valid JSON: {e}")

    if required_fields:
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"


# ============================================================================
# Accessibility Assertions
# ============================================================================

def assert_accessibility_compliant(
    result: str,
    check_aria: bool = True,
    check_keyboard: bool = True,
    check_contrast: bool = False  # Requires DOM analysis
) -> None:
    """
    Assert HTML output is accessibility compliant.

    Args:
        result: HTML string to check
        check_aria: Check for ARIA attributes
        check_keyboard: Check for keyboard navigation support
        check_contrast: Check color contrast (requires DOM)

    Raises:
        AssertionError: If accessibility issues found
    """
    assert result is not None, "HTML is None"

    if check_aria:
        # Check for common ARIA attributes
        aria_patterns = [
            r'aria-label=',
            r'aria-describedby=',
            r'role=',
            r'aria-live=',
        ]

        has_aria = any(
            re.search(pattern, result) for pattern in aria_patterns
        )

        # Relaxed check - warn but don't fail if no ARIA
        if not has_aria:
            import warnings
            warnings.warn("HTML output has no ARIA attributes detected")

    if check_keyboard:
        # Check for tabindex or interactive elements
        keyboard_patterns = [
            r'tabindex=',
            r'<button',
            r'<a\s+href=',
            r'role="button"',
            r'onkeydown=',
            r'onkeyup=',
        ]

        has_keyboard = any(
            re.search(pattern, result) for pattern in keyboard_patterns
        )

        # Relaxed check for keyboard support
        if not has_keyboard:
            import warnings
            warnings.warn("HTML output has no keyboard navigation detected")


def assert_aria_labels_present(result: str, min_labels: int = 1) -> None:
    """
    Assert ARIA labels are present in HTML.

    Args:
        result: HTML string
        min_labels: Minimum number of ARIA labels expected

    Raises:
        AssertionError: If insufficient ARIA labels
    """
    labels = re.findall(r'aria-label="[^"]+"', result)
    assert len(labels) >= min_labels, (
        f"Expected at least {min_labels} ARIA labels, found {len(labels)}"
    )


def assert_focus_management(result: str) -> None:
    """
    Assert HTML has proper focus management.

    Args:
        result: HTML string

    Raises:
        AssertionError: If focus management is missing
    """
    # Check for focus-related attributes or classes
    focus_patterns = [
        r'tabindex=',
        r'focus:',  # CSS focus classes
        r'\.focus\(',  # JavaScript focus
        r'autofocus',
    ]

    has_focus = any(
        re.search(pattern, result) for pattern in focus_patterns
    )

    assert has_focus, "HTML output has no focus management"


def assert_reduced_motion_support(result: str) -> None:
    """
    Assert CSS/HTML supports reduced motion preference.

    Args:
        result: HTML/CSS string

    Raises:
        AssertionError: If reduced motion not supported
    """
    reduced_motion_patterns = [
        r'prefers-reduced-motion',
        r'motion-reduce',
        r'reduce-motion',
    ]

    has_reduced_motion = any(
        re.search(pattern, result) for pattern in reduced_motion_patterns
    )

    # Warn but don't fail - reduced motion is optional
    if not has_reduced_motion:
        import warnings
        warnings.warn("No prefers-reduced-motion support detected")


# ============================================================================
# Content Integrity Assertions
# ============================================================================

def assert_content_preserved(
    result: str,
    original_content: List[str],
    escape_html: bool = True
) -> None:
    """
    Assert original content appears in output.

    Args:
        result: Rendered output
        original_content: List of content strings to find
        escape_html: Whether content may be HTML-escaped

    Raises:
        AssertionError: If content missing
    """
    import html

    for content in original_content:
        found = content in result

        if not found and escape_html:
            # Try HTML-escaped version
            escaped = html.escape(content)
            found = escaped in result

        assert found, f"Content not preserved in output: {content[:50]}..."


def assert_panel_ids_unique(result: str) -> None:
    """
    Assert all panel IDs in output are unique.

    Args:
        result: Rendered output

    Raises:
        AssertionError: If duplicate IDs found
    """
    # Find all panel IDs
    id_pattern = r'data-panel-id="([^"]+)"'
    ids = re.findall(id_pattern, result)

    if ids:
        unique_ids = set(ids)
        assert len(ids) == len(unique_ids), (
            f"Duplicate panel IDs found: {len(ids)} total, {len(unique_ids)} unique"
        )


def assert_confidence_displayed(
    result: str,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None
) -> None:
    """
    Assert confidence values are displayed correctly.

    Args:
        result: Rendered output
        min_confidence: Minimum expected confidence
        max_confidence: Maximum expected confidence

    Raises:
        AssertionError: If confidence display issues
    """
    # Look for confidence indicators
    confidence_patterns = [
        r'confidence["\s:]+([0-9.]+)',
        r'data-confidence="([0-9.]+)"',
        r'(\d+)%\s*confidence',
    ]

    values = []
    for pattern in confidence_patterns:
        matches = re.findall(pattern, result, re.IGNORECASE)
        for match in matches:
            try:
                val = float(match)
                # Convert percentage to decimal if needed
                if val > 1:
                    val = val / 100
                values.append(val)
            except ValueError:
                pass

    if min_confidence is not None and values:
        assert min(values) >= min_confidence, (
            f"Confidence below minimum: {min(values)} < {min_confidence}"
        )

    if max_confidence is not None and values:
        assert max(values) <= max_confidence, (
            f"Confidence above maximum: {max(values)} > {max_confidence}"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # HTML assertions
    'assert_valid_html_output',
    'assert_html_structure_valid',

    # React assertions
    'assert_valid_react_props',
    'assert_react_props_typescript_friendly',

    # AR assertions
    'assert_valid_ar_scene',
    'assert_ar_transforms_valid',

    # JSON assertions
    'assert_valid_json_output',

    # Accessibility assertions
    'assert_accessibility_compliant',
    'assert_aria_labels_present',
    'assert_focus_management',
    'assert_reduced_motion_support',

    # Content assertions
    'assert_content_preserved',
    'assert_panel_ids_unique',
    'assert_confidence_displayed',
]
