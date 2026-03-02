#!/usr/bin/env python3
"""
Debug Toolkit - Practical utilities for debugging HoloLoom issues
==================================================================

Usage:
    python debug_toolkit.py <command> [args]

Commands:
    trace-import <module>     - Trace import chain to find hangs
    check-circular            - Detect circular imports
    profile-import <module>   - Time each import step
    test-minimal <file>       - Test file with minimal imports
    find-heavy-imports        - Find slow-loading modules

Examples:
    python debug_toolkit.py trace-import HoloLoom.agentic.ml_logic_detector
    python debug_toolkit.py check-circular HoloLoom
    python debug_toolkit.py profile-import HoloLoom.memory.graph
"""

import sys
import time
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Set, Optional


# ============================================================================
# Import Tracing
# ============================================================================

class ImportTracer:
    """Trace import chain with timing."""

    def __init__(self):
        self.imports: List[Dict] = []
        self.start_time = time.time()
        self.current_level = 0

    def trace_import(self, module_name: str, max_depth: int = 5):
        """Trace imports for a module with timing."""
        print(f"Tracing imports for: {module_name}")
        print("=" * 80)

        # Hook into import system
        original_import = __builtins__.__import__

        def traced_import(name, *args, **kwargs):
            indent = "  " * self.current_level
            start = time.time()

            try:
                print(f"{indent}→ Importing {name}...")
                sys.stdout.flush()

                self.current_level += 1
                if self.current_level > max_depth:
                    print(f"{indent}  (max depth reached, skipping)")
                    result = original_import(name, *args, **kwargs)
                else:
                    result = original_import(name, *args, **kwargs)
                self.current_level -= 1

                elapsed = (time.time() - start) * 1000
                print(f"{indent}✓ {name} ({elapsed:.1f}ms)")

                self.imports.append({
                    'module': name,
                    'time_ms': elapsed,
                    'level': self.current_level
                })

                return result

            except Exception as e:
                self.current_level -= 1
                elapsed = (time.time() - start) * 1000
                print(f"{indent}✗ {name} FAILED ({elapsed:.1f}ms): {e}")
                raise

        # Replace import
        __builtins__.__import__ = traced_import

        try:
            # Try to import
            start = time.time()
            importlib.import_module(module_name)
            total = (time.time() - start) * 1000

            print("\n" + "=" * 80)
            print(f"Total import time: {total:.1f}ms")
            print("\nSlowest imports:")

            sorted_imports = sorted(self.imports, key=lambda x: x['time_ms'], reverse=True)[:10]
            for imp in sorted_imports:
                print(f"  {imp['time_ms']:>8.1f}ms - {imp['module']}")

        except Exception as e:
            print(f"\n✗ Import failed: {e}")
            traceback.print_exc()

        finally:
            # Restore original import
            __builtins__.__import__ = original_import


# ============================================================================
# Circular Import Detection
# ============================================================================

class CircularImportDetector:
    """Detect circular imports in a package."""

    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.dependencies: Dict[str, Set[str]] = {}

    def scan_file(self, file_path: Path) -> Set[str]:
        """Extract import statements from a Python file."""
        imports = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # from X import Y
                    if line.startswith('from '):
                        parts = line.split()
                        if len(parts) >= 2:
                            module = parts[1]
                            if module.startswith('hololoom'):
                                imports.add(module)

                    # import X
                    elif line.startswith('import '):
                        parts = line.split()
                        if len(parts) >= 2:
                            module = parts[1].split('.')[0]
                            if module == 'hololoom':
                                imports.add(module)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        return imports

    def build_dependency_graph(self):
        """Build dependency graph for all Python files."""
        print("Scanning for imports...")

        for py_file in self.package_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue

            module_name = str(py_file.relative_to(self.package_path.parent)).replace('\\', '.').replace('/', '.')[:-3]
            imports = self.scan_file(py_file)

            if imports:
                self.dependencies[module_name] = imports

        print(f"Found {len(self.dependencies)} modules with imports")

    def find_cycles(self) -> List[List[str]]:
        """Find circular dependencies using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.dependencies.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for module in self.dependencies:
            if module not in visited:
                dfs(module)

        return cycles

    def detect(self):
        """Detect and print circular imports."""
        self.build_dependency_graph()
        cycles = self.find_cycles()

        if cycles:
            print(f"\n✗ Found {len(cycles)} circular import(s):")
            for i, cycle in enumerate(cycles, 1):
                print(f"\nCycle {i}:")
                for j, module in enumerate(cycle):
                    if j < len(cycle) - 1:
                        print(f"  {module}")
                        print(f"    ↓")
                    else:
                        print(f"  {module} (back to start)")
        else:
            print("\n✓ No circular imports detected")


# ============================================================================
# Minimal Import Tester
# ============================================================================

def test_minimal_import(file_path: str):
    """Test importing a file with minimal dependencies."""
    print(f"Testing minimal import: {file_path}")
    print("=" * 80)

    # Try importing just standard library
    print("\n1. Testing standard library imports...")
    try:
        import ast
        import re
        import logging
        from typing import List, Dict
        from dataclasses import dataclass
        from enum import Enum
        print("✓ Standard library imports OK")
    except Exception as e:
        print(f"✗ Standard library failed: {e}")
        return

    # Try importing file contents directly
    print("\n2. Reading file...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✓ File read OK ({len(content)} chars)")
    except Exception as e:
        print(f"✗ File read failed: {e}")
        return

    # Check for HoloLoom imports
    print("\n3. Checking for HoloLoom imports...")
    hololoom_imports = []
    for line in content.split('\n'):
        if 'from hololoom' in line or 'import HoloLoom' in line:
            hololoom_imports.append(line.strip())

    if hololoom_imports:
        print(f"✗ Found {len(hololoom_imports)} HoloLoom import(s):")
        for imp in hololoom_imports:
            print(f"  - {imp}")
    else:
        print("✓ No HoloLoom imports (file is standalone)")


# ============================================================================
# Heavy Import Finder
# ============================================================================

def find_heavy_imports(package_name: str = "hololoom"):
    """Find modules that take longest to import."""
    print(f"Profiling imports in {package_name}...")
    print("=" * 80)

    import pkgutil
    import importlib

    results = []

    try:
        package = importlib.import_module(package_name)

        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__,
            prefix=f"{package_name}."
        ):
            print(f"Testing {modname}...", end=" ")
            sys.stdout.flush()

            start = time.time()
            try:
                importlib.import_module(modname)
                elapsed = (time.time() - start) * 1000
                results.append((modname, elapsed, True))
                print(f"OK ({elapsed:.1f}ms)")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                results.append((modname, elapsed, False))
                print(f"FAILED ({elapsed:.1f}ms): {e}")

    except Exception as e:
        print(f"✗ Failed to import package: {e}")
        return

    # Show results
    print("\n" + "=" * 80)
    print("Slowest imports:")
    results.sort(key=lambda x: x[1], reverse=True)

    for modname, time_ms, success in results[:20]:
        status = "✓" if success else "✗"
        print(f"{status} {time_ms:>8.1f}ms - {modname}")


# ============================================================================
# CLI
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "trace-import":
        if len(sys.argv) < 3:
            print("Usage: python debug_toolkit.py trace-import <module>")
            sys.exit(1)

        module_name = sys.argv[2]
        tracer = ImportTracer()
        tracer.trace_import(module_name)

    elif command == "check-circular":
        if len(sys.argv) < 3:
            package_path = Path("hololoom")
        else:
            package_path = Path(sys.argv[2])

        detector = CircularImportDetector(package_path)
        detector.detect()

    elif command == "profile-import":
        if len(sys.argv) < 3:
            print("Usage: python debug_toolkit.py profile-import <module>")
            sys.exit(1)

        module_name = sys.argv[2]
        tracer = ImportTracer()
        tracer.trace_import(module_name, max_depth=10)

    elif command == "test-minimal":
        if len(sys.argv) < 3:
            print("Usage: python debug_toolkit.py test-minimal <file>")
            sys.exit(1)

        file_path = sys.argv[2]
        test_minimal_import(file_path)

    elif command == "find-heavy-imports":
        package = sys.argv[2] if len(sys.argv) >= 3 else "hololoom"
        find_heavy_imports(package)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
