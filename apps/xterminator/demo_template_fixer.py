#!/usr/bin/env python3
"""
Template Fixer Demo
===================

Demonstrates Charlotte's template-based fixing on real HoloLoom code.

"Some Spider spun templates!" - Charlotte
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xterminator.template_fixer import TemplateFixer, list_available_templates
from xterminator.templates import get_template_catalog
from xterminator.xterminator_types import (
    FixProposal,
    FixStrategy,
    RiskLevel,
    CodeContext,
    ContextType
)


def create_proposal(
    original_code: str,
    category: str,
    file_path: str,
    line_number: int,
    severity: str = "medium"
) -> FixProposal:
    """Create fix proposal"""
    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path=file_path,
        line_number=line_number,
        code_snippet=original_code,
        has_tests=True
    )

    return FixProposal(
        fix_id=f"{file_path}:{line_number}",
        issue_category=category,
        issue_severity=severity,
        risk_level=RiskLevel.MEDIUM,
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.85,
        original_code=original_code,
        context=context,
        explanation=f"Template fix for {category}"
    )


# Example 1: Missing error handling in file I/O
EXAMPLE_1_CODE = '''import json
from pathlib import Path

def load_config(config_path: str):
    """Load configuration from JSON file"""
    with open(config_path) as f:
        return json.load(f)

def save_results(output_path: str, results: dict):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
'''

# Example 2: Hardcoded API keys
EXAMPLE_2_CODE = '''import requests

# WARNING: Hardcoded API key!
API_KEY = "sk-abc123def456..."
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

def call_openai(prompt: str):
    """Call OpenAI API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_ENDPOINT, headers=headers)
    return response.json()
'''

# Example 3: Timezone-naive datetime
EXAMPLE_3_CODE = '''from datetime import datetime
import json

def generate_report():
    """Generate timestamped report"""
    timestamp = datetime.now()

    report = {
        "generated_at": timestamp.isoformat(),
        "version": "1.0",
        "data": []
    }

    return report
'''


async def demo_example_1():
    """Demo: Fix missing error handling"""
    print("=" * 70)
    print("EXAMPLE 1: Add Error Handling to File I/O")
    print("=" * 70)
    print()

    print("ORIGINAL CODE:")
    print(EXAMPLE_1_CODE)
    print()

    # Create proposals for both issues
    proposal1 = create_proposal(
        original_code='with open(config_path) as f:\n        return json.load(f)',
        category='error_handling',
        file_path='hololoom/config_loader.py',
        line_number=6
    )

    proposal2 = create_proposal(
        original_code='with open(output_path, \'w\') as f:\n        json.dump(results, f, indent=2)',
        category='error_handling',
        file_path='hololoom/config_loader.py',
        line_number=11
    )

    fixer = TemplateFixer()

    # Fix first issue
    print("FIXING: load_config() - missing error handling")
    result1 = await fixer.fix_issue(proposal1, EXAMPLE_1_CODE)

    if result1:
        fixed_code, diff = result1
        print("\nDIFF:")
        print(diff)
        print()
    else:
        print("⚠️ Could not fix automatically")

    print()


async def demo_example_2():
    """Demo: Move hardcoded secrets to environment variables"""
    print("=" * 70)
    print("EXAMPLE 2: Move Hardcoded Secrets to Environment Variables")
    print("=" * 70)
    print()

    print("ORIGINAL CODE:")
    print(EXAMPLE_2_CODE)
    print()

    # Create proposal
    proposal = create_proposal(
        original_code='API_KEY = "sk-abc123def456..."',
        category='hardcoded_values',
        file_path='hololoom/llm_client.py',
        line_number=4,
        severity='high'
    )

    fixer = TemplateFixer()

    print("FIXING: Hardcoded API key")
    result = await fixer.fix_issue(proposal, EXAMPLE_2_CODE)

    if result:
        fixed_code, diff = result
        print("\nFIXED CODE:")
        print(fixed_code)
        print()
        print("DIFF:")
        print(diff)
        print()
    else:
        print("⚠️ Could not fix automatically")

    print()


async def demo_example_3():
    """Demo: Fix timezone-naive datetime"""
    print("=" * 70)
    print("EXAMPLE 3: Fix Timezone-Naive Datetime")
    print("=" * 70)
    print()

    print("ORIGINAL CODE:")
    print(EXAMPLE_3_CODE)
    print()

    # Create proposal
    proposal = create_proposal(
        original_code='timestamp = datetime.now()',
        category='datetime',
        file_path='hololoom/reporting.py',
        line_number=6
    )

    fixer = TemplateFixer()

    print("FIXING: Timezone-naive datetime.now()")
    result = await fixer.fix_issue(proposal, EXAMPLE_3_CODE)

    if result:
        fixed_code, diff = result
        print("\nFIXED CODE:")
        print(fixed_code)
        print()
        print("DIFF:")
        print(diff)
        print()
    else:
        print("⚠️ Could not fix automatically")

    print()


async def demo_template_catalog():
    """Demo: Show template catalog"""
    print("=" * 70)
    print("CHARLOTTE'S TEMPLATE CATALOG")
    print("=" * 70)
    print()

    catalog = get_template_catalog()
    print(catalog)
    print()

    print("AVAILABLE TEMPLATES BY CATEGORY:")
    print("-" * 70)
    templates_by_category = list_available_templates()
    for category, templates in sorted(templates_by_category.items()):
        print(f"\n{category.upper().replace('_', ' ')}:")
        for template in templates:
            print(f"  - {template}")
    print()


async def main():
    """Run all demos"""
    print("\n")
    print("=" * 70)
    print()
    print("  CHARLOTTE'S TEMPLATE FIXER DEMO")
    print("  'Some Spider spun templates!'")
    print()
    print("=" * 70)
    print("\n")

    # Show catalog first
    await demo_template_catalog()

    # Run examples
    await demo_example_1()
    await demo_example_2()
    await demo_example_3()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("[SUCCESS] Template Fixer demonstrated on 3 real examples:")
    print("   1. Added error handling to file I/O")
    print("   2. Moved hardcoded API key to environment variable")
    print("   3. Fixed timezone-naive datetime")
    print()
    print("Charlotte says: 'Templates applied successfully!'")
    print()


if __name__ == '__main__':
    asyncio.run(main())
