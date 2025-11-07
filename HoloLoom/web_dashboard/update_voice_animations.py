#!/usr/bin/env python3
"""
Update voice animations in agentic_dashboard.html

Replaces the voice CSS section with enhanced animations:
- Button press effect (smooth scale down)
- Pulsing red recording light with ripple effects
- Glowing recording indicator
"""

from pathlib import Path

def update_animations():
    """Update voice CSS with enhanced animations"""

    dashboard_path = Path(__file__).parent / "agentic_dashboard.html"
    enhanced_css_path = Path(__file__).parent / "voice_enhanced.css"

    if not dashboard_path.exists():
        print(f"Dashboard not found: {dashboard_path}")
        return False

    if not enhanced_css_path.exists():
        print(f"Enhanced CSS not found: {enhanced_css_path}")
        return False

    print("Reading files...")
    html_content = dashboard_path.read_text(encoding='utf-8')
    enhanced_css = enhanced_css_path.read_text(encoding='utf-8')

    # Find the voice integration styles section
    start_marker = "/* ========================================"
    start_marker2 = "           VOICE INTEGRATION STYLES"
    end_marker = "/* Audio Player (hidden) */"
    end_content = """#audioPlayer {
            display: none;
        }"""

    # Find start and end positions
    start_pos = html_content.find(start_marker)
    if start_pos == -1:
        print("Could not find voice CSS section!")
        return False

    # Find the end of the voice CSS section
    end_pos = html_content.find(end_content, start_pos)
    if end_pos == -1:
        print("Could not find end of voice CSS section!")
        return False

    # Include the closing of #audioPlayer
    end_pos = end_pos + len(end_content)

    print(f"Found voice CSS section at positions {start_pos}-{end_pos}")

    # Extract just the CSS content from enhanced file (without opening comment)
    css_lines = enhanced_css.split('\n')
    # Skip the first 3 comment lines
    enhanced_css_content = '\n'.join(css_lines[4:])

    # Add proper indentation (8 spaces to match existing)
    indented_css = '\n'.join('        ' + line if line.strip() else ''
                             for line in enhanced_css_content.split('\n'))

    # Replace the section
    new_html = (
        html_content[:start_pos] +
        "/* ========================================\n" +
        "           ENHANCED VOICE INTEGRATION STYLES\n" +
        "           ======================================== */\n\n" +
        indented_css +
        html_content[end_pos:]
    )

    # Write updated file
    print("Writing updated dashboard...")
    dashboard_path.write_text(new_html, encoding='utf-8')

    print("SUCCESS! Voice animations updated!")
    print()
    print("New features:")
    print("  - Button press animation (smooth scale down)")
    print("  - Pulsing red recording light with ripples")
    print("  - Glowing recording indicator")
    print("  - Enhanced hover effects")
    print()
    print("Hard refresh your browser (Ctrl+Shift+R) to see the changes!")

    return True


if __name__ == '__main__':
    update_animations()
