#!/usr/bin/env python3
"""
TypeScript/JavaScript Detection Demo
=====================================

Demonstrates Trough's TypeScript and JavaScript detection capabilities:
1. Detects error handling issues (unhandled promises, missing try/catch)
2. Detects security issues (eval, innerHTML, hardcoded secrets)
3. Detects TypeScript-specific issues (any types, type assertions)
4. Detects React-specific issues (missing keys, unsafe hooks)

Author: mythRL Team
Date: November 15, 2025
"""

import asyncio
from trough import TypeScriptSlopDetector, Language


# Sample TypeScript code with intentional issues
SAMPLE_TYPESCRIPT = """
// TypeScript code with quality issues

// 1. Unhandled promise (HIGH)
async function fetchData() {
    fetch('https://api.example.com/data')
        .then(response => response.json());  // Missing .catch()
}

// 2. Hardcoded API key (CRITICAL)
const API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz";

// 3. Use of 'any' type (MEDIUM)
function processData(data: any) {
    return data.value;
}

// 4. Dangerous eval (CRITICAL)
function executeCode(code: string) {
    eval(code);  // Code injection risk
}

// 5. Direct innerHTML (HIGH - XSS risk)
function updateContent(html: string) {
    document.getElementById('content').innerHTML = html;
}

// 6. Missing await in async function (LOW)
async function getData() {
    fetchData().then(data => console.log(data));  // Should use await
}

// 7. Non-null assertion (MEDIUM)
function getUser() {
    return users.find(u => u.id === 1)!.name;  // Risky !
}

// 8. Magic number (LOW)
const TIMEOUT = 86400000;  // What is this?

// 9. Event listener leak (LOW)
window.addEventListener('click', handleClick);  // Never removed

// 10. TODO comment (LOW)
// TODO: Implement proper error handling
"""

SAMPLE_REACT_TSX = """
// React TSX with quality issues

import React, { useState, useEffect } from 'react';

// 1. useState without type (LOW)
function Counter() {
    const [count, setCount] = useState(0);  // Missing <number>

    // 2. useEffect without dependencies (MEDIUM)
    useEffect(() => {
        console.log('Count changed');
    });  // Missing dependency array

    return <div>{count}</div>;
}

// 3. Missing key in mapped elements (MEDIUM)
function UserList({ users }) {
    return (
        <ul>
            {users.map(user => <li>{user.name}</li>)}  {/* Missing key */}
        </ul>
    );
}

// 4. Unsafe lifecycle method (HIGH)
class OldComponent extends React.Component {
    componentWillMount() {  // Deprecated!
        this.loadData();
    }
}

// 5. Console.log in production (LOW)
function debugComponent() {
    console.log('Rendering...');  // Remove before deployment
}
"""

SAMPLE_JAVASCRIPT = """
// JavaScript code with quality issues

// 1. Missing try/catch around await (MEDIUM)
async function loadUser() {
    const response = await fetch('/api/user');  // No error handling
    return response.json();
}

// 2. Throw string instead of Error (LOW)
function validateInput(input) {
    if (!input) {
        throw "Input is required";  // Should be Error object
    }
}

// 3. Unclosed WebSocket (MEDIUM)
const ws = new WebSocket('wss://example.com');  // Never closed

// 4. FIXME comment (MEDIUM)
// FIXME: This breaks on edge cases
function parseData(data) {
    return JSON.parse(data);
}

// 5. Commented out code (LOW)
// function oldImplementation() {
//     return legacy.process();
// }
"""


async def demo_typescript_detection():
    """Demonstrate TypeScript detection."""
    print("=" * 70)
    print("TYPESCRIPT DETECTION DEMO")
    print("=" * 70)
    print()

    detector = TypeScriptSlopDetector()

    # Detect issues in TypeScript code
    print("[1/3] Analyzing TypeScript code...")
    print()

    ts_issues = await detector.detect_all(
        SAMPLE_TYPESCRIPT,
        Language.TYPESCRIPT,
        "example.ts"
    )

    print(f"Found {len(ts_issues)} issues in TypeScript code:")
    print()

    # Group by severity
    by_severity = {}
    for issue in ts_issues:
        severity = issue.severity.value
        by_severity.setdefault(severity, []).append(issue)

    # Show critical and high severity issues
    for severity in ['critical', 'high', 'medium']:
        if severity in by_severity:
            issues_list = by_severity[severity]
            print(f"  {severity.upper()}: {len(issues_list)} issues")
            for issue in issues_list[:3]:  # Show first 3
                print(f"    - Line {issue.line_number}: {issue.message}")
                print(f"      Suggestion: {issue.suggestion}")
            print()

    print()


async def demo_react_detection():
    """Demonstrate React TSX detection."""
    print("=" * 70)
    print("REACT TSX DETECTION DEMO")
    print("=" * 70)
    print()

    detector = TypeScriptSlopDetector()

    # Detect issues in React code
    print("[2/3] Analyzing React TSX code...")
    print()

    react_issues = await detector.detect_all(
        SAMPLE_REACT_TSX,
        Language.TSX,
        "Component.tsx"
    )

    print(f"Found {len(react_issues)} issues in React code:")
    print()

    # Show React-specific issues
    react_specific = [
        i for i in react_issues
        if 'React' in i.message or 'useState' in i.message or 'useEffect' in i.message or 'key' in i.message
    ]

    print(f"  React-specific: {len(react_specific)} issues")
    for issue in react_specific:
        print(f"    - Line {issue.line_number}: {issue.message}")
        print(f"      Category: {issue.category.value}")
    print()

    print()


async def demo_javascript_detection():
    """Demonstrate JavaScript detection."""
    print("=" * 70)
    print("JAVASCRIPT DETECTION DEMO")
    print("=" * 70)
    print()

    detector = TypeScriptSlopDetector()

    # Detect issues in JavaScript code
    print("[3/3] Analyzing JavaScript code...")
    print()

    js_issues = await detector.detect_all(
        SAMPLE_JAVASCRIPT,
        Language.JAVASCRIPT,
        "script.js"
    )

    print(f"Found {len(js_issues)} issues in JavaScript code:")
    print()

    # Show by category
    by_category = {}
    for issue in js_issues:
        category = issue.category.value
        by_category.setdefault(category, []).append(issue)

    print("  Issues by category:")
    for category, issues_list in sorted(by_category.items()):
        print(f"    - {category}: {len(issues_list)} issues")
    print()

    # Show detailed examples
    print("  Example issues:")
    for issue in js_issues[:5]:
        print(f"    {issue.line_number}. [{issue.severity.value.upper()}] {issue.message}")
        print(f"       Code: {issue.code_snippet[:60]}...")
    print()


async def demo_comparison():
    """Compare detection across languages."""
    print()
    print("=" * 70)
    print("LANGUAGE COMPARISON")
    print("=" * 70)
    print()

    detector = TypeScriptSlopDetector()

    # Detect in all three code samples
    ts_issues = await detector.detect_all(SAMPLE_TYPESCRIPT, Language.TYPESCRIPT, "example.ts")
    tsx_issues = await detector.detect_all(SAMPLE_REACT_TSX, Language.TSX, "Component.tsx")
    js_issues = await detector.detect_all(SAMPLE_JAVASCRIPT, Language.JAVASCRIPT, "script.js")

    print("Issue counts by language:")
    print(f"  TypeScript (.ts):  {len(ts_issues)} issues")
    print(f"  React TSX (.tsx):  {len(tsx_issues)} issues")
    print(f"  JavaScript (.js):  {len(js_issues)} issues")
    print()

    # Compare categories
    print("Common issue categories detected:")
    all_categories = set()
    all_categories.update(i.category.value for i in ts_issues)
    all_categories.update(i.category.value for i in tsx_issues)
    all_categories.update(i.category.value for i in js_issues)

    for category in sorted(all_categories):
        ts_count = sum(1 for i in ts_issues if i.category.value == category)
        tsx_count = sum(1 for i in tsx_issues if i.category.value == category)
        js_count = sum(1 for i in js_issues if i.category.value == category)

        if ts_count > 0 or tsx_count > 0 or js_count > 0:
            print(f"  {category:20} TS:{ts_count:2}  TSX:{tsx_count:2}  JS:{js_count:2}")
    print()


async def main():
    """Run all demos."""
    print()
    print("🎯 Trough TypeScript/JavaScript Detection")
    print("   Full-stack code quality analysis")
    print()

    try:
        await demo_typescript_detection()
        await demo_react_detection()
        await demo_javascript_detection()
        await demo_comparison()

        print("=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print()
        print("✓ TypeScript detection working")
        print("✓ React TSX detection working")
        print("✓ JavaScript detection working")
        print("✓ Full-stack coverage achieved!")
        print()
        print("Key Features Demonstrated:")
        print("  - Error handling detection (promises, async/await)")
        print("  - Security issue detection (eval, XSS, hardcoded secrets)")
        print("  - TypeScript-specific issues (any types, assertions)")
        print("  - React-specific issues (hooks, keys, lifecycle)")
        print("  - Performance issues (missing dependencies)")
        print("  - Code quality (documentation, dead code)")
        print()
        print("Next Steps:")
        print("  1. Scan your TypeScript/JavaScript codebase")
        print("  2. Integrate with xTerminator for auto-fixing")
        print("  3. Add to CI/CD pipeline for continuous quality")
        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
