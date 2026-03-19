#!/usr/bin/env python3
"""
Add voice JavaScript inline to HTML (instead of external file)

This ensures the JavaScript loads properly without needing static file serving.
"""

from pathlib import Path


def add_inline_voice_js():
    """Add voice.js content inline to HTML"""

    dashboard_path = Path(__file__).parent / "agentic_dashboard.html"
    voice_js_path = Path(__file__).parent / "voice.js"

    if not dashboard_path.exists():
        print(f"Dashboard not found: {dashboard_path}")
        return False

    if not voice_js_path.exists():
        print(f"voice.js not found: {voice_js_path}")
        return False

    print("Reading files...")
    html_content = dashboard_path.read_text(encoding='utf-8')
    voice_js = voice_js_path.read_text(encoding='utf-8')

    # Check if already inline
    if 'Loading voice integration...' in html_content:
        print("Voice JS already inline!")
        return False

    # Find the voice.js script tag and replace it with inline script
    old_script_tag = '    <script src="voice.js"></script>'

    if old_script_tag not in html_content:
        print("Could not find voice.js script tag!")
        return False

    # Create inline script
    inline_script = f"""    <script>
{voice_js}
    </script>"""

    # Replace
    new_html = html_content.replace(old_script_tag, inline_script)

    # Write
    print("Writing updated dashboard with inline voice JS...")
    dashboard_path.write_text(new_html, encoding='utf-8')

    print("SUCCESS! Voice JavaScript is now inline.")
    print()
    print("Hard refresh your browser (Ctrl+Shift+R)")
    print("Then check console - you should see:")
    print("  Loading voice integration...")
    print("  Voice integration script loaded ✓")
    print("  DOM loaded, initializing voice...")
    print("  Voice integration initialized ✓")

    return True


if __name__ == '__main__':
    add_inline_voice_js()
