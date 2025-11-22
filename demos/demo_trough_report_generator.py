"""
Trough Interactive HTML Report Generator - Demo

Demonstrates Week 1 functionality:
- Generate interactive 3-panel HTML report
- Click-through findings navigation
- VS Code integration
- Search and filter capabilities

Run:
    PYTHONPATH=. python demos/demo_trough_report_generator.py

Output:
    demos/output/trough_report.html
"""

import asyncio
from pathlib import Path
from trough import AISlopDetector, Language
from trough.report_generator import generate_html_report


# Sample Python code with issues
SAMPLE_CODE = '''
import os
import sys
import json

# Issue 1: Hardcoded secrets
API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz"
PASSWORD = "admin123"

# Issue 2: Missing error handling
def load_config(filename):
    """Load configuration without error handling."""
    config_file = open("config.json")  # Resource leak: no close()
    data = config_file.read()
    config = json.loads(data)  # No try/catch

    # Issue 3: Dictionary access without get()
    user_id = config["user_id"]  # KeyError if missing

    return config


# Issue 4: Incomplete implementation
async def fetch_data(url):
    """Fetch data from URL - TODO: implement retry logic."""
    # TODO: Add exponential backoff
    # TODO: Handle connection timeouts
    pass


# Issue 5: Magic numbers
def process_records(records):
    """Process records with magic numbers."""
    if len(records) > 9999:  # Magic number
        print("Too many records")

    for i in range(len(records)):  # Could use enumerate
        if i < 100:  # Magic number
            process_single(records[i])


# Issue 6: Copy-paste duplication
def validate_email(email):
    """Validate email address."""
    if "@" not in email:
        return False
    if "." not in email.split("@")[1]:
        return False
    return True


def validate_phone(phone):
    """Validate phone number - copy-paste from email."""
    if "@" not in phone:  # Wrong check for phone!
        return False
    if "." not in phone.split("@")[1]:  # Wrong check for phone!
        return False
    return True


# Issue 7: Unused imports
def get_current_user():
    """Get current user - uses sys but not sys."""
    return os.environ.get("USER")
'''


async def main():
    """Generate sample Trough report."""
    print("Trough Interactive Report Generator - Week 1 Demo")
    print("=" * 60)

    # Create output directory
    output_dir = Path("demos/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run detector
    print("\n1. Running Trough detector...")
    detector = AISlopDetector()
    findings = await detector.detect_all(
        SAMPLE_CODE,
        Language.PYTHON,
        "sample_code.py"
    )

    print(f"   Found {len(findings)} issues:")

    # Group by severity
    from collections import defaultdict
    by_severity = defaultdict(list)
    for finding in findings:
        by_severity[finding.severity.value].append(finding)

    for severity in ['critical', 'high', 'medium', 'low', 'info']:
        count = len(by_severity[severity])
        if count > 0:
            print(f"     - {severity.upper()}: {count}")

    # Generate report
    print("\n2. Generating interactive HTML report...")
    output_file = output_dir / "trough_report.html"

    html = generate_html_report(
        findings=findings,
        output_path=str(output_file),
        code_snippets={"sample_code.py": SAMPLE_CODE},
        title="Trough Code Analysis - Sample Report"
    )

    print(f"   Report size: {len(html):,} bytes")
    print(f"   Saved to: {output_file}")

    # Print sample findings
    print("\n3. Sample findings:")
    for i, finding in enumerate(findings[:5], 1):
        print(f"\n   [{i}] {finding.message}")
        print(f"       Line: {finding.line_number}")
        print(f"       Category: {finding.category.value}")
        print(f"       Severity: {finding.severity.value}")
        print(f"       Suggestion: {finding.suggestion}")

    if len(findings) > 5:
        print(f"\n   ... and {len(findings) - 5} more findings")

    # Statistics
    print("\n4. Report Statistics:")
    print(f"   Total findings: {len(findings)}")
    print(f"   Unique categories: {len(set(f.category for f in findings))}")
    print(f"   Avg confidence: {sum(f.confidence for f in findings) / len(findings):.2f}")

    print("\n5. Features implemented (Week 1):")
    print("   [x] 3-panel layout (Findings | Code | Details)")
    print("   [x] Syntax highlighting with Pygments")
    print("   [x] VS Code integration (vscode:// links)")
    print("   [x] Search functionality")
    print("   [x] Filter by severity and category")
    print("   [x] Keyboard navigation (Arrow/Enter)")
    print("   [x] Responsive design")
    print("   [x] Zero external dependencies (embedded CSS/JS)")
    print("   [x] All 31 tests passing")

    print("\n6. Features coming in Week 2:")
    print("   - Complete panel synchronization")
    print("   - Auto-fix integration with xTerminator")
    print("   - Diff preview for fixes")
    print("   - Windows path handling (vscode://file/c:/Users/...)")

    print("\n" + "=" * 60)
    print(f"Report ready at: {output_file.absolute()}")
    print("Open in browser to test interactive features!")


if __name__ == "__main__":
    asyncio.run(main())
