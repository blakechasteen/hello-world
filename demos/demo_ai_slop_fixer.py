#!/usr/bin/env python3
"""
Demo: AI Slop Fixer
===================
Demonstrates the complete AI Slop Fixer system.

This script:
1. Starts a test server
2. Indexes a sample codebase
3. Tests hallucination detection
4. Tests code verification
5. Shows the full fix cycle

Run:
    PYTHONPATH=. python demos/demo_ai_slop_fixer.py
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.agentic.codebase_ingestion import CodebaseIndexer, Language
from HoloLoom.agentic.hallucination_detector import HallucinationDetector
from HoloLoom.agentic.code_verification import CodeVerifier


# Sample "codebase" - functions that actually exist
SAMPLE_CODEBASE = """
# User management module
def get_user_by_username(username: str) -> dict:
    '''Fetch user from database by username.'''
    return {"username": username, "id": 123}

def check_password(password: str, password_hash: str) -> bool:
    '''Verify password against hash.'''
    import bcrypt
    return bcrypt.checkpw(password.encode(), password_hash.encode())

def generate_jwt_token(user: dict) -> str:
    '''Generate JWT token for authenticated user.'''
    import jwt
    return jwt.encode(user, 'secret', algorithm='HS256')

class UserSession:
    '''User session management.'''
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.token = None

    def create_session(self) -> str:
        '''Create new session token.'''
        return generate_jwt_token({"user_id": self.user_id})
"""


# AI-generated broken code (with hallucinations and type errors)
AI_BROKEN_CODE = """
def authenticate_user(username, password):
    # These functions don't exist (hallucinations)!
    user = fetch_user_from_database(username)
    if verify_password_hash(password, user.hash):
        return create_session_token(user)
    return None
"""


async def demo_ingestion():
    """Demo 1: Codebase Ingestion"""
    print("=" * 80)
    print("DEMO 1: Codebase Ingestion")
    print("=" * 80)
    print()

    # Create indexer
    indexer = CodebaseIndexer()

    # Index sample code
    print("📚 Indexing sample codebase...")
    code_file = await indexer.ingest_file("sample_users.py", Language.PYTHON)

    print(f"✅ Found {len(code_file.entities)} entities:")
    for entity in code_file.entities[:10]:
        print(f"  - {entity.entity_type.value:10s} {entity.name:30s} (line {entity.line_number})")

    print()
    print(f"📊 Statistics:")
    stats = indexer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print()
    return indexer


async def demo_hallucination_detection(indexer: CodebaseIndexer):
    """Demo 2: Hallucination Detection"""
    print("=" * 80)
    print("DEMO 2: Hallucination Detection")
    print("=" * 80)
    print()

    # Create detector
    detector = HallucinationDetector(indexer)

    print("🤖 AI-Generated Code (with hallucinations):")
    print("-" * 40)
    print(AI_BROKEN_CODE)
    print("-" * 40)
    print()

    print("🔍 Detecting hallucinations...")
    result = await detector.detect_and_explain(
        AI_BROKEN_CODE,
        Language.PYTHON,
        "test.py"
    )

    print(f"\n❌ Found {result['summary']['total_hallucinations']} hallucination(s):")
    print()

    for halluc in result['hallucinations']:
        print(f"Line {halluc['line']}: '{halluc['reference']}'")
        print(f"  Type: {halluc['type']}")
        print(f"  Confidence: {halluc['confidence']:.2f}")
        print(f"  Reason: {halluc['reason']}")

        if halluc['suggestions']:
            print(f"  💡 Suggestions:")
            for suggestion in halluc['suggestions'][:3]:
                print(f"     - {suggestion}")

        if halluc['fix']:
            print(f"  🔧 Fix: {halluc['fix']}")

        print()

    return result


async def demo_verification():
    """Demo 3: Code Verification"""
    print("=" * 80)
    print("DEMO 3: Code Verification")
    print("=" * 80)
    print()

    # Create verifier
    verifier = CodeVerifier()

    # Test with syntax error
    broken_python = """
def foo():
    return bar(  # Missing closing paren
"""

    print("🐍 Python code with syntax error:")
    print("-" * 40)
    print(broken_python)
    print("-" * 40)
    print()

    print("🔍 Running verification...")
    result = await verifier.verify_comprehensive(
        broken_python,
        "python",
        "test.py"
    )

    print(f"\n{'✅' if result['success'] else '❌'} Verification Result:")
    print(f"  Errors: {result['error_count']}")
    print(f"  Warnings: {result['warning_count']}")
    print(f"  Checks run: {', '.join(result['checks_run'])}")
    print()

    if result['errors']:
        print("📋 Errors found:")
        for error in result['errors']:
            print(f"  Line {error['line']}: {error['message']}")

    print()
    print(result['summary'])
    print()


async def demo_fix_cycle():
    """Demo 4: Complete Fix Cycle (simulation)"""
    print("=" * 80)
    print("DEMO 4: Complete Fix Cycle (Simulation)")
    print("=" * 80)
    print()

    print("This demonstrates what happens when you run 'Squad: Fix AI Slop'")
    print()

    # Simulate fix iterations
    iterations = [
        {
            'iteration': 1,
            'hallucinations': 3,
            'errors': 5,
            'changes': 'Replaced hallucinated functions with real ones'
        },
        {
            'iteration': 2,
            'hallucinations': 0,
            'errors': 2,
            'changes': 'Fixed remaining type errors'
        },
        {
            'iteration': 3,
            'hallucinations': 0,
            'errors': 0,
            'changes': 'Added missing imports'
        }
    ]

    for iter_data in iterations:
        print(f"🔄 Iteration {iter_data['iteration']}/5...")
        print(f"   Hallucinations: {iter_data['hallucinations']}")
        print(f"   Type Errors: {iter_data['errors']}")
        print(f"   Changes: {iter_data['changes']}")

        if iter_data['hallucinations'] == 0 and iter_data['errors'] == 0:
            print()
            print("✅ Code fixed!")
            break

        print("   Attempting fix...")
        await asyncio.sleep(0.5)  # Simulate processing
        print()

    print()
    print("📝 Final Code:")
    print("-" * 40)

    fixed_code = """
def authenticate_user(username, password):
    # Fixed: using real functions from codebase
    user = get_user_by_username(username)
    if check_password(password, user.get('password_hash', '')):
        return generate_jwt_token(user)
    return None
"""
    print(fixed_code)
    print("-" * 40)
    print()


async def main():
    """Run all demos"""
    print()
    print("🚀 AI Slop Fixer - Complete Demo")
    print("=" * 80)
    print()

    # Demo 1: Ingestion
    indexer = await demo_ingestion()

    # Demo 2: Hallucination Detection
    await demo_hallucination_detection(indexer)

    # Demo 3: Verification
    await demo_verification()

    # Demo 4: Fix Cycle
    await demo_fix_cycle()

    print("=" * 80)
    print("✨ Demo Complete!")
    print()
    print("Next steps:")
    print("  1. Start HoloLoom server: uvicorn HoloLoom.server.agentic_api:app")
    print("  2. Open VS Code with Squad extension")
    print("  3. Run 'Squad: Index Workspace'")
    print("  4. Try 'Squad: Fix AI Slop' on broken code!")
    print()


if __name__ == "__main__":
    # Create temp file with sample codebase
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SAMPLE_CODEBASE)
        temp_file = f.name

    # Override the file path in the demo
    original_ingest = CodebaseIndexer.ingest_file

    async def ingest_file_override(self, file_path, language):
        return await original_ingest(self, temp_file, language)

    CodebaseIndexer.ingest_file = ingest_file_override

    try:
        asyncio.run(main())
    finally:
        # Cleanup
        os.unlink(temp_file)
