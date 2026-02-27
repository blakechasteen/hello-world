#!/usr/bin/env python3
"""
Analyze weaving_orchestrator.py to identify extraction opportunities.
"""

import re
from pathlib import Path

def analyze_file(file_path):
    """Analyze file structure and identify large methods."""

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find all method definitions
    methods = []
    current_method = None
    current_indent = None

    for i, line in enumerate(lines, 1):
        # Check for method definition
        method_match = re.match(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(', line)
        if method_match:
            # Save previous method
            if current_method:
                current_method['end_line'] = i - 1
                current_method['lines'] = current_method['end_line'] - current_method['start_line'] + 1
                methods.append(current_method)

            # Start new method
            indent = len(method_match.group(1))
            is_async = method_match.group(2) is not None
            name = method_match.group(3)

            current_method = {
                'name': name,
                'start_line': i,
                'indent': indent,
                'is_async': is_async,
                'is_private': name.startswith('_'),
                'is_init': name == '__init__'
            }
            current_indent = indent

        # Check if we've moved to a new top-level definition
        elif current_method and line.strip() and not line.startswith(' ' * (current_indent + 1)):
            if not line.strip().startswith('#'):  # Ignore comments
                current_method['end_line'] = i - 1
                current_method['lines'] = current_method['end_line'] - current_method['start_line'] + 1
                methods.append(current_method)
                current_method = None

    # Save last method
    if current_method:
        current_method['end_line'] = len(lines)
        current_method['lines'] = current_method['end_line'] - current_method['start_line'] + 1
        methods.append(current_method)

    return methods, len(lines)

def main():
    file_path = Path('hololoom/weaving_orchestrator.py')
    methods, total_lines = analyze_file(file_path)

    print(f"Total lines: {total_lines}")
    print(f"Total methods: {len(methods)}")
    print()

    # Sort by size
    methods.sort(key=lambda m: m['lines'], reverse=True)

    print("Largest methods (>50 lines):")
    print("-" * 80)
    print(f"{'Name':<40} {'Lines':>8} {'Start':>8} {'Type':>10}")
    print("-" * 80)

    large_methods = [m for m in methods if m['lines'] > 50]
    for method in large_methods:
        method_type = 'async' if method['is_async'] else 'sync'
        if method['is_init']:
            method_type += ' (init)'
        elif method['is_private']:
            method_type += ' (pvt)'

        print(f"{method['name']:<40} {method['lines']:>8} {method['start_line']:>8} {method_type:>10}")

    print()
    print(f"Total lines in large methods (>50): {sum(m['lines'] for m in large_methods)}")
    print(f"Percentage: {sum(m['lines'] for m in large_methods) / total_lines * 100:.1f}%")

    # Class structure
    print()
    print("Class structure:")
    print("-" * 80)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    classes = []
    for i, line in enumerate(lines, 1):
        if re.match(r'^class\s+(\w+)', line):
            classes.append((i, line.strip()))

    for line_no, class_def in classes:
        print(f"{line_no:>5}: {class_def}")

if __name__ == '__main__':
    main()
