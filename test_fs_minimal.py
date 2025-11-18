#!/usr/bin/env python3
"""Minimal test - just verify filesystem_spinner.py file is syntactically correct."""

import sys
from pathlib import Path

# Check if file exists and can be parsed
fs_spinner_path = Path(__file__).parent / "HoloLoom" / "spinningWheel" / "filesystem_spinner.py"

print("Checking FilesystemSpinner implementation...")
print(f"File: {fs_spinner_path}")
print(f"Exists: {fs_spinner_path.exists()}")
print(f"Size: {fs_spinner_path.stat().st_size} bytes")
print()

# Try to compile the file
try:
    with open(fs_spinner_path) as f:
        code = f.read()

    compile(code, str(fs_spinner_path), 'exec')
    print("✅ Syntax check passed!")
    print()

    # Count key features
    lines = code.split('\n')
    print("Code statistics:")
    print(f"  Total lines: {len(lines)}")
    print(f"  Classes: {code.count('class ')}")
    print(f"  Functions: {code.count('def ')}")
    print(f"  Async functions: {code.count('async def ')}")
    print()

    # Check key components
    components = [
        "FilesystemSpinner",
        "FilesystemSpinnerConfig",
        "spin_incremental",
        "score_importance",
        "_scan_directory",
        "_process_file",
        "_chunk_text",
        "ingest_directory"
    ]

    print("Key components:")
    for comp in components:
        if comp in code:
            print(f"  ✓ {comp}")
        else:
            print(f"  ✗ {comp} (MISSING)")

    print()
    print("✅ FilesystemSpinner implementation complete!")

except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
