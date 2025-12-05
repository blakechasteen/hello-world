#!/usr/bin/env python3
"""
xTerminator Classification Engine Demo
=======================================

Demonstrates the classification system on real code examples.
Shows how xTerminator decides what to fix, how, and with what confidence.
"""

import asyncio
from dataclasses import dataclass


@dataclass
class MockIssue:
    """Mock SlopIssue for demonstration"""
    category: str
    severity: str
    message: str
    file_path: str
    line_number: int
    code_snippet: str
    suggestion: str
    confidence: float = 0.8


# Example code files with various issues
EXAMPLE_1_COPYPASTE = '''
import requests

def process_user_data(user_id):
    """Process user data"""
    url = f"https://api.example.com/users/{user_id}"
    response = requests.get(url)
    data = response.json()
    return data['name'], data['email']

def process_order_data(order_id):
    """Process order data"""
    url = f"https://api.example.com/orders/{order_id}"
    response = requests.get(url)
    data = response.json()
    return data['item'], data['price']

def test_process_user():
    name, email = process_user_data(123)
    assert name is not None
'''

EXAMPLE_2_ERROR_HANDLING = '''
import os
import json

def load_config(config_path):
    """Load configuration from file"""
    # Missing error handling!
    with open(config_path) as f:
        return json.load(f)

def save_config(config_path, data):
    """Save configuration to file"""
    # Also missing error handling!
    with open(config_path, 'w') as f:
        json.dump(data, f)
'''

EXAMPLE_3_FALSE_POSITIVE = '''
"""
Database module

Example SQL queries:
- SELECT * FROM users WHERE active = true
- DELETE FROM sessions WHERE expired = true

Always use parameterized queries to prevent SQL injection!
"""

import sqlite3

def get_active_users(conn):
    """Get all active users - SAFE, uses parameterized query"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE active = ?", (True,))
    return cursor.fetchall()
'''

EXAMPLE_4_SECURITY = '''
import subprocess
import os

def run_command(user_input):
    """Execute user command - UNSAFE!"""
    # Command injection vulnerability
    os.system(f"echo {user_input}")

def backup_database(db_name):
    """Backup database - UNSAFE!"""
    # Shell injection possible
    subprocess.call(f"mysqldump {db_name} > backup.sql", shell=True)
'''


async def demo_classification_engine():
    """Demonstrate the classification engine"""
    from xterminator import ClassificationEngine

    engine = ClassificationEngine()

    print("=" * 70)
    print("xTerminator Classification Engine Demo")
    print("=" * 70)
    print()
    print("Demonstrating how xTerminator classifies issues and proposes fixes.")
    print()

    # Demo 1: Copy-Paste (Auto-fixable)
    print("-" * 70)
    print("Example 1: Copy-Paste Code (Auto-fixable)")
    print("-" * 70)
    print()

    issue1 = MockIssue(
        category='copy_paste',
        severity='medium',
        message='Duplicated code pattern detected',
        file_path='api_client.py',
        line_number=6,
        code_snippet='url = f"https://api.example.com/users/{user_id}"',
        suggestion='Extract common API call logic',
        confidence=0.85
    )

    print(f"[>>] Issue detected at {issue1.file_path}:{issue1.line_number}")
    print(f"   Category: {issue1.category}")
    print(f"   Severity: {issue1.severity}")
    print(f"   Message: {issue1.message}")
    print()

    proposal1 = await engine.classify_and_propose(issue1, full_code=EXAMPLE_1_COPYPASTE)

    print("[>>] Classification Results:")
    print(f"   Context: {proposal1.context.context_type.value}")
    print(f"   Risk Level: {proposal1.risk_level.value}")
    print(f"   Fix Strategy: {proposal1.fix_strategy.value}")
    print(f"   Confidence: {proposal1.confidence:.3f}")
    print(f"   Has Tests: {proposal1.context.has_tests}")
    print()

    print(f"[>>] Decision: {'[AUTO] AUTOFIX' if proposal1.safe_to_autofix else '[WARN] NEEDS REVIEW'}")
    print(f"   {proposal1.explanation}")
    print()

    if proposal1.metadata.get('implementation_hints'):
        print("[>>] Implementation Hints:")
        for hint in proposal1.metadata['implementation_hints'][:3]:
            print(f"   - {hint}")
        print()

    # Demo 2: Error Handling (Needs Review)
    print("-" * 70)
    print("Example 2: Missing Error Handling (Needs Review)")
    print("-" * 70)
    print()

    issue2 = MockIssue(
        category='error_handling',
        severity='high',
        message='File operation without error handling',
        file_path='config.py',
        line_number=7,
        code_snippet='with open(config_path) as f:',
        suggestion='Add try/except for FileNotFoundError and JSONDecodeError',
        confidence=0.80
    )

    print(f"[>>] Issue detected at {issue2.file_path}:{issue2.line_number}")
    print(f"   Category: {issue2.category}")
    print(f"   Severity: {issue2.severity}")
    print(f"   Message: {issue2.message}")
    print()

    proposal2 = await engine.classify_and_propose(issue2, full_code=EXAMPLE_2_ERROR_HANDLING)

    print("[>>] Classification Results:")
    print(f"   Context: {proposal2.context.context_type.value}")
    print(f"   Risk Level: {proposal2.risk_level.value}")
    print(f"   Fix Strategy: {proposal2.fix_strategy.value}")
    print(f"   Confidence: {proposal2.confidence:.3f}")
    print(f"   Has Tests: {proposal2.context.has_tests}")
    print()

    print(f"[>>] Decision: {'[AUTO] AUTOFIX' if proposal2.safe_to_autofix else '[WARN] NEEDS REVIEW'}")
    print(f"   {proposal2.explanation}")
    print()

    if proposal2.metadata.get('implementation_hints'):
        print("[>>] Implementation Hints:")
        for hint in proposal2.metadata['implementation_hints'][:4]:
            print(f"   - {hint}")
        print()

    # Demo 3: False Positive (SQL in Comment)
    print("-" * 70)
    print("Example 3: SQL Keywords in Documentation (False Positive)")
    print("-" * 70)
    print()

    issue3 = MockIssue(
        category='sql_injection',
        severity='critical',
        message='SQL keyword detected',
        file_path='database.py',
        line_number=5,
        code_snippet='- SELECT * FROM users WHERE active = true',
        suggestion='Use parameterized queries',
        confidence=0.75
    )

    print(f"[>>] Issue detected at {issue3.file_path}:{issue3.line_number}")
    print(f"   Category: {issue3.category}")
    print(f"   Severity: {issue3.severity}")
    print(f"   Message: {issue3.message}")
    print()

    classification3 = await engine.classify(issue3, full_code=EXAMPLE_3_FALSE_POSITIVE)

    print("[>>] Classification Results:")
    print(f"   Context: {classification3.context.context_type.value}")
    print(f"   Risk Level: {classification3.risk_level.value}")
    print(f"   Selected Strategy: {classification3.selected_strategy.value}")
    print(f"   Confidence: {classification3.confidence:.3f}")
    print(f"   Is False Positive: {classification3.is_false_positive}")
    print()

    if classification3.is_false_positive:
        print(f"[>>] Decision: SKIP (False Positive)")
        print(f"   Reason: {classification3.false_positive_reason}")
        print()

    # Demo 4: Security Issue (Manual Only)
    print("-" * 70)
    print("Example 4: Command Injection (Manual Fix Required)")
    print("-" * 70)
    print()

    issue4 = MockIssue(
        category='command_injection',
        severity='critical',
        message='Unsafe shell command execution with user input',
        file_path='admin_utils.py',
        line_number=7,
        code_snippet='os.system(f"echo {user_input}")',
        suggestion='Use subprocess with argument list, never shell=True',
        confidence=0.95
    )

    print(f"[>>] Issue detected at {issue4.file_path}:{issue4.line_number}")
    print(f"   Category: {issue4.category}")
    print(f"   Severity: {issue4.severity}")
    print(f"   Message: {issue4.message}")
    print()

    proposal4 = await engine.classify_and_propose(issue4, full_code=EXAMPLE_4_SECURITY)

    print("[>>] Classification Results:")
    print(f"   Context: {proposal4.context.context_type.value}")
    print(f"   Risk Level: {proposal4.risk_level.value}")
    print(f"   Fix Strategy: {proposal4.fix_strategy.value}")
    print(f"   Confidence: {proposal4.confidence:.3f}")
    print()

    print(f"[>>] Decision: MANUAL FIX REQUIRED")
    print(f"   {proposal4.explanation}")
    print()

    print("[WARN]  Security issues are NEVER auto-fixed!")
    print("   Human review and testing required.")
    print()

    # Summary Statistics
    print("-" * 70)
    print("Summary Statistics")
    print("-" * 70)
    print()

    all_results = await engine.classify_batch(
        [issue1, issue2, issue3, issue4],
        full_code=EXAMPLE_1_COPYPASTE  # Just for demonstration
    )

    stats = engine.get_statistics(all_results)

    print(f"Total Issues Analyzed: {stats['total_issues']}")
    print(f"False Positives: {stats['false_positives']} ({stats['false_positive_rate']:.1%})")
    print(f"Auto-Fixable: {stats['auto_fixable']} ({stats['auto_fixable_rate']:.1%})")
    print(f"Needs Review: {stats['needs_review']}")
    print(f"Average Confidence: {stats['avg_confidence']:.3f}")
    print()

    print("Strategy Distribution:")
    for strategy, count in stats['strategy_distribution'].items():
        print(f"  {strategy}: {count}")
    print()

    print("Risk Distribution:")
    for risk, count in stats['risk_distribution'].items():
        print(f"  {risk}: {count}")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  ✓ Context-aware detection prevents false positives")
    print("  ✓ Risk-based classification ensures safety")
    print("  ✓ Confidence scoring enables smart automation")
    print("  ✓ Security issues always require manual review")
    print()


async def main():
    """Run demo"""
    await demo_classification_engine()


if __name__ == '__main__':
    asyncio.run(main())
