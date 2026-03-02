#!/usr/bin/env python3
"""
Fix old documentation.types imports across the codebase.

The HoloLoom.documentation.types module was moved/split into:
- HoloLoom.protocols.types (Query, MemoryShard, Features)
- HoloLoom.fabric.spacetime (Spacetime)

This script updates all files to use the correct import paths.
"""

import re
from pathlib import Path

# Files that need fixing (from grep search)
FILES_TO_FIX = [
    "HoloLoom/convergence/recursive_engine.py",
    "HoloLoom/convergence/recursive_reasoner.py",
    "HoloLoom/mcp_server_promptly.py",
    "HoloLoom/protocols/recursive_reasoning.py",
    "HoloLoom/spinningWheel/workspace.py",
    "HoloLoom/voice/elle_bridge.py",
    "HoloLoom/voice/voice_agent.py",
    "HoloLoom/voice_first/core/unified_agent.py",
]

def fix_file(filepath):
    """Fix imports in a single file."""
    path = Path(filepath)
    if not path.exists():
        print(f"SKIP: {filepath} (doesn't exist)")
        return False

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Pattern 1: from hololoom.documentation.types import X, Y, Z
    # Need to split based on what's imported
    pattern = r'from hololoom\.documentation\.types import ([^\n]+)'

    def replace_import(match):
        imports = match.group(1).strip()

        # Parse imports
        import_list = [i.strip() for i in imports.split(',')]

        # Categorize imports
        protocol_imports = []
        spacetime_imports = []

        for item in import_list:
            if item in ['Query', 'MemoryShard', 'Features', 'ActionPlan']:
                protocol_imports.append(item)
            elif item == 'Spacetime':
                spacetime_imports.append(item)
            else:
                # Unknown, put in protocols
                protocol_imports.append(item)

        # Build replacement
        lines = []
        if protocol_imports:
            lines.append(f"from hololoom.protocols.types import {', '.join(protocol_imports)}")
        if spacetime_imports:
            lines.append(f"from hololoom.fabric.spacetime import {', '.join(spacetime_imports)}")

        return '\n'.join(lines)

    content = re.sub(pattern, replace_import, content)

    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"FIXED: {filepath}")
        return True
    else:
        print(f"SKIP: {filepath} (no changes)")
        return False

def main():
    print("Fixing documentation.types imports...\n")

    fixed_count = 0
    for filepath in FILES_TO_FIX:
        if fix_file(filepath):
            fixed_count += 1

    print(f"\nDONE: Fixed {fixed_count}/{len(FILES_TO_FIX)} files")

if __name__ == "__main__":
    main()
