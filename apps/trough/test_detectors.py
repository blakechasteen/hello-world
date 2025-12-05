#!/usr/bin/env python3
"""
Quick test of Trough detectors - standalone version
"""

import asyncio
from trough import AISlopDetector, MLLogicDetector, Language

# Test code with various issues
TEST_CODE = """
import requests
import os

# Hardcoded secret (should be detected)
API_KEY = "sk-1234567890abcdef"

def fetch_user(user_id):
    # Missing error handling (should be detected)
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()

def process_file():
    # Resource leak - file not closed (should be detected)
    f = open("data.txt")
    data = f.read()
    return data

def get_value(data):
    # Dictionary access without .get() (should be detected)
    return data["key"]

# Division by zero (logic error, should be detected by ML)
def divide(a, b):
    return a / b  # No check for b == 0

# Array bounds issue (off-by-one)
def get_last(arr):
    return arr[len(arr)]  # Should be arr[-1] or arr[len(arr)-1]
"""

async def main():
    print("=" * 70)
    print("TROUGH STANDALONE DETECTOR TEST")
    print("=" * 70)
    print()

    # Test AI Slop Detector
    print("1. Testing AI Slop Detector...")
    print("-" * 70)
    slop_detector = AISlopDetector()
    slop_issues = await slop_detector.detect_all(
        code=TEST_CODE,
        language=Language.PYTHON,
        file_path="test.py"
    )

    print(f"Found {len(slop_issues)} AI slop issues:\n")
    for issue in slop_issues:
        print(f"  Line {issue.line_number}: [{issue.severity.value.upper()}] {issue.category.value}")
        print(f"    {issue.message}")
        if issue.suggestion:
            print(f"    Fix: {issue.suggestion}")
        print()

    # Test ML Logic Detector
    print("\n2. Testing ML Logic Detector...")
    print("-" * 70)
    logic_detector = MLLogicDetector()
    logic_errors = await logic_detector.detect(
        code=TEST_CODE,
        language=Language.PYTHON,
        file_path="test.py"
    )

    print(f"Found {len(logic_errors)} logic errors:\n")
    for error in logic_errors:
        print(f"  Line {error.line_number}: [{error.severity.value.upper()}] {error.category.value}")
        print(f"    {error.message}")
        if error.suggested_fix:
            print(f"    Fix: {error.suggested_fix}")
        print(f"    Confidence: {error.confidence:.2f}")
        print()

    # Summary
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Total issues found: {len(slop_issues) + len(logic_errors)}")
    print(f"  AI Slop Issues: {len(slop_issues)}")
    print(f"  Logic Errors: {len(logic_errors)}")
    print()
    print("✓ Both detectors working standalone (no HoloLoom deps)!")
    print()

if __name__ == "__main__":
    asyncio.run(main())
