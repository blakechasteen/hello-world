#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick setup script for testing Promptly VS Code extension.
Creates test prompts in Promptly database.
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add Promptly to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Promptly"))

try:
    from promptly import Promptly
except ImportError:
    print("ERROR: Could not import Promptly")
    print("Make sure you're running from the project root")
    sys.exit(1)

def main():
    print("🚀 Setting up Promptly test data...")

    # Initialize Promptly
    p = Promptly()

    try:
        p.init()
        print("✅ Initialized Promptly repository")
    except Exception as e:
        print(f"ℹ️  Repository already exists: {e}")

    # Test prompts with various tags
    test_prompts = [
        {
            "name": "greeting",
            "content": "You are a friendly and welcoming assistant. Greet users warmly and make them feel comfortable asking questions.",
            "tags": ["assistant", "friendly", "general"]
        },
        {
            "name": "python_coder",
            "content": "You are an expert Python developer. Write clean, efficient, and well-documented code following PEP 8 standards.",
            "tags": ["code", "python", "expert"]
        },
        {
            "name": "creative_writer",
            "content": "You are a creative writing assistant. Help users craft compelling stories, vivid descriptions, and engaging narratives.",
            "tags": ["creative", "writing", "stories"]
        },
        {
            "name": "data_analyst",
            "content": "You are a data analysis expert. Help users understand data patterns, create visualizations, and derive insights.",
            "tags": ["data", "analysis", "insights"]
        },
        {
            "name": "debugger",
            "content": "You are a debugging specialist. Help identify and fix code issues systematically, explaining the root cause of bugs.",
            "tags": ["code", "debugging", "troubleshooting"]
        }
    ]

    print(f"\n📝 Adding {len(test_prompts)} test prompts...")

    for prompt in test_prompts:
        try:
            p.add(
                name=prompt["name"],
                content=prompt["content"],
                metadata={"tags": prompt["tags"]}
            )
            print(f"  ✅ Added: {prompt['name']} (tags: {', '.join(prompt['tags'])})")
        except Exception as e:
            print(f"  ⚠️  {prompt['name']}: {e}")

    # Verify
    prompts = p.list_prompts()
    print(f"\n✅ Setup complete! {len(prompts)} prompts in database")
    print("\nPrompts available:")
    for prompt in prompts:
        print(f"  - {prompt['name']}")

    print("\n🎉 Ready to test!")
    print("\nNext steps:")
    print("  1. cd promptly-vscode && npm install && npm run compile")
    print("  2. Open 'promptly-vscode' folder in VS Code")
    print("  3. Press F5 to launch Extension Development Host")
    print("  4. Check the Promptly sidebar!")

if __name__ == "__main__":
    main()
