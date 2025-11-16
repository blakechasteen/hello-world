"""
TypeScript Auto-Fixer
=====================

TypeScript-specific code fixer mirroring SimpleLLMFixer architecture.

Supports:
- Unused imports (dead_code)
- Any types → proper types (type_safety)
- Non-null assertions (!.) → optional chaining (?.)
- Missing type annotations
- Console.log cleanup
- Interface vs Type consistency

Philosophy: "Keep it simple, use intelligence when available"

Author: mythRL Team
Date: November 16, 2025
"""

import re
from typing import Optional, Tuple, Dict, Any, List
from difflib import unified_diff


class TypeScriptFixer:
    """
    TypeScript-specific code fixer for common issues.

    Mirrors SimpleLLMFixer architecture for consistency.
    Falls back to rule-based fixes if LLM unavailable.
    """

    def __init__(self, use_llm: bool = True, llm_model: str = "llama3.2:3b"):
        """
        Initialize TypeScript fixer.

        Args:
            use_llm: Whether to use LLM for fix generation (default: True)
            llm_model: Ollama model to use (default: llama3.2:3b)
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.ollama_available = False

        # Try to import ollama
        if use_llm:
            try:
                import ollama
                self.ollama = ollama
                self.ollama_available = True
            except ImportError:
                self.ollama_available = False

    async def fix_issue(
        self,
        issue: Dict[str, Any],
        full_code: str,
        file_path: str
    ) -> Optional[Tuple[str, str]]:
        """
        Generate and apply fix for TypeScript issue.

        Args:
            issue: Issue dict with category, message, line_number, code_snippet
            full_code: Full file content
            file_path: Path to file

        Returns:
            (fixed_code, diff) or None if can't fix
        """
        category = issue.get('category', '')
        message = issue.get('message', '')
        line_number = issue.get('line_number', 0)
        code_snippet = issue.get('code_snippet', '')

        # Route to appropriate fixer
        if category == 'dead_code' and 'import' in message.lower():
            return self._fix_unused_import(full_code, message, code_snippet, line_number)

        elif category == 'type_safety' and 'any' in message.lower():
            return self._fix_any_type(full_code, message, code_snippet, line_number)

        elif category == 'type_safety' and 'non-null' in message.lower():
            return self._fix_non_null_assertion(full_code, message, code_snippet, line_number)

        elif category == 'missing_types':
            return self._fix_missing_type_annotation(full_code, message, code_snippet, line_number)

        elif category == 'code_quality' and 'console' in message.lower():
            return self._fix_console_log(full_code, message, code_snippet, line_number)

        else:
            # Try LLM for other cases
            if self.use_llm and self.ollama_available:
                return await self._fix_with_llm(full_code, issue, file_path)
            else:
                return None

    def _fix_unused_import(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Fix unused import by removing it.

        TypeScript patterns:
        1. Named imports: import { Foo, Bar } from './module'
        2. Namespace imports: import * as vscode from 'vscode'
        3. Default imports: import React from 'react'
        4. Type-only imports: import type { SomeType } from './types'

        Strategy:
        - Remove entire line if single import unused
        - Remove specific name if multiple imports
        """
        lines = full_code.split('\n')

        if line_number <= 0 or line_number > len(lines):
            return None

        # Get the import line (1-indexed)
        import_line = lines[line_number - 1]

        # Extract unused name from message
        # Examples:
        # - "Unused import 'HallucinationResult'"
        # - "'CodeContext' is defined but never used"
        match = re.search(r"'([^']+)'", message)
        if not match:
            return None
        unused_name = match.group(1)

        # Case 1: Namespace import unused (import * as name)
        if f"import * as {unused_name}" in import_line:
            # Remove entire line
            fixed_lines = lines[:line_number - 1] + lines[line_number:]
            fixed_code = '\n'.join(fixed_lines)

        # Case 2: Default import unused (import Name from)
        elif re.match(rf"import {unused_name} from", import_line.strip()):
            # Remove entire line
            fixed_lines = lines[:line_number - 1] + lines[line_number:]
            fixed_code = '\n'.join(fixed_lines)

        # Case 3: Named imports with destructuring
        elif '{' in import_line and '}' in import_line:
            # Extract imports between { }
            match = re.search(r'\{([^}]+)\}', import_line)
            if not match:
                return None

            imports_str = match.group(1)
            imports = [imp.strip() for imp in imports_str.split(',')]

            # Remove the unused one
            # Handle type-only imports: import type { Foo }
            imports = [imp for imp in imports if unused_name not in imp]

            if not imports:
                # No imports left, remove entire line
                fixed_lines = lines[:line_number - 1] + lines[line_number:]
                fixed_code = '\n'.join(fixed_lines)
            else:
                # Reconstruct import line
                # Preserve import/from structure
                before_brace = import_line[:import_line.index('{')]
                after_brace = import_line[import_line.index('}') + 1:]
                new_import_line = f"{before_brace}{{ {', '.join(imports)} }}{after_brace}"
                lines[line_number - 1] = new_import_line
                fixed_code = '\n'.join(lines)

        else:
            return None

        # Generate diff
        diff = self._generate_diff(full_code, fixed_code, "unused_import_typescript")

        return (fixed_code, diff)

    def _fix_any_type(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Replace 'any' type with proper type.

        Strategies:
        1. If LLM available: Infer type from context
        2. Otherwise: Replace with 'unknown' (safer than any)

        Examples:
        - function process(data: any) → (data: unknown)
        - const items: any[] → const items: unknown[]
        """
        if not self.use_llm or not self.ollama_available:
            # Simple rule: Replace 'any' with 'unknown'
            lines = full_code.split('\n')
            if line_number <= 0 or line_number > len(lines):
                return None

            original_line = lines[line_number - 1]

            # Replace ': any' with ': unknown'
            # Be careful with word boundaries
            fixed_line = re.sub(r':\s*any\b', ': unknown', original_line)

            # Replace 'any[]' with 'unknown[]'
            fixed_line = re.sub(r'\bany\[\]', 'unknown[]', fixed_line)

            if fixed_line == original_line:
                return None

            lines[line_number - 1] = fixed_line
            fixed_code = '\n'.join(lines)

            diff = self._generate_diff(full_code, fixed_code, "any_type_to_unknown")
            return (fixed_code, diff)

        else:
            # Use LLM to infer proper type
            return None  # TODO: Implement LLM type inference

    def _fix_non_null_assertion(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Replace non-null assertion operator (!) with optional chaining (?)

        Examples:
        - obj!.property → obj?.property
        - arr![0] → arr?.[0]
        - fn!() → fn?.()
        """
        lines = full_code.split('\n')
        if line_number <= 0 or line_number > len(lines):
            return None

        original_line = lines[line_number - 1]

        # Pattern: Replace '!.' with '?.'
        fixed_line = re.sub(r'!\.', '?.', original_line)

        # Pattern: Replace '![' with '?.[' (array indexing)
        fixed_line = re.sub(r'!\[', '?.[', fixed_line)

        # Pattern: Replace '!(' with '?.(' (function calls)
        fixed_line = re.sub(r'!\(', '?.(', fixed_line)

        if fixed_line == original_line:
            return None

        lines[line_number - 1] = fixed_line
        fixed_code = '\n'.join(lines)

        diff = self._generate_diff(full_code, fixed_code, "non_null_to_optional_chaining")
        return (fixed_code, diff)

    def _fix_missing_type_annotation(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Add type annotation to function or variable.

        Strategies:
        1. If LLM available: Infer type from context
        2. Otherwise: Use generic 'any' with TODO comment

        Examples:
        - function foo(a, b) → function foo(a: any, b: any): any
        - const value = → const value: any =
        """
        if not self.use_llm or not self.ollama_available:
            # Without LLM, add 'any' with TODO comment
            lines = full_code.split('\n')
            if line_number <= 0 or line_number > len(lines):
                return None

            original_line = lines[line_number - 1]

            # Detect function definition
            if 'function' in original_line or '=>' in original_line:
                # Add ': any' to parameters without types
                # This is complex - for now, add TODO comment
                indent_match = re.match(r'^(\s*)', original_line)
                indent = indent_match.group(1) if indent_match else ''
                todo_comment = f"{indent}// TODO: Add type annotations"

                lines.insert(line_number - 1, todo_comment)
                fixed_code = '\n'.join(lines)

                diff = self._generate_diff(full_code, fixed_code, "missing_type_annotation_todo")
                return (fixed_code, diff)

            else:
                return None

        else:
            # Use LLM to infer proper type
            return None  # TODO: Implement LLM type inference

    def _fix_console_log(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Remove or comment out console.log statements.

        Strategy:
        - Remove entire console.log line
        - Keep console.error and console.warn (they're useful)
        """
        lines = full_code.split('\n')
        if line_number <= 0 or line_number > len(lines):
            return None

        original_line = lines[line_number - 1]

        # Only remove console.log, not console.error/warn
        if 'console.log' in original_line:
            # Comment it out instead of removing (safer)
            indent_match = re.match(r'^(\s*)', original_line)
            indent = indent_match.group(1) if indent_match else ''
            commented_line = f"{indent}// {original_line.strip()}  // Removed by autofix"

            lines[line_number - 1] = commented_line
            fixed_code = '\n'.join(lines)

            diff = self._generate_diff(full_code, fixed_code, "console_log_removal")
            return (fixed_code, diff)

        else:
            return None

    async def _fix_with_llm(
        self,
        full_code: str,
        issue: Dict[str, Any],
        file_path: str
    ) -> Optional[Tuple[str, str]]:
        """
        Use LLM to generate fix for any TypeScript issue.

        Args:
            full_code: Full file content
            issue: Issue dict
            file_path: Path to file

        Returns:
            (fixed_code, diff) or None
        """
        category = issue.get('category', '')
        message = issue.get('message', '')
        line_number = issue.get('line_number', 0)
        code_snippet = issue.get('code_snippet', '')

        prompt = f"""Fix this TypeScript code issue:

File: {file_path}
Issue: {message}
Category: {category}
Line {line_number}:
{code_snippet}

Full file context:
{full_code}

Generate the COMPLETE fixed TypeScript file. Make ONLY the minimal change needed to fix the issue.
Output ONLY the fixed code, no explanations."""

        try:
            response = self.ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 2000}
            )

            fixed_code = response['message']['content'].strip()

            # Remove markdown code blocks if present
            fixed_code = re.sub(r'^```typescript\n', '', fixed_code)
            fixed_code = re.sub(r'^```ts\n', '', fixed_code)
            fixed_code = re.sub(r'^```\n', '', fixed_code)
            fixed_code = re.sub(r'\n```$', '', fixed_code)

            # Generate diff
            diff = self._generate_diff(full_code, fixed_code, category)

            return (fixed_code, diff)

        except Exception as e:
            # LLM failed, return None
            return None

    def _generate_diff(self, original: str, fixed: str, label: str) -> str:
        """Generate unified diff."""
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        diff = unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f'original ({label})',
            tofile=f'fixed ({label})',
            lineterm=''
        )

        return ''.join(diff)

    # Utility methods for type inference (future)

    def _infer_type_from_usage(self, var_name: str, full_code: str) -> Optional[str]:
        """
        Infer TypeScript type from variable usage.

        Examples:
        - const x = 5 → number
        - const name = 'foo' → string
        - const items = [] → any[] (needs context)
        """
        # Simple heuristics
        # Pattern: const name = value
        pattern = rf'{var_name}\s*=\s*(.+?)[;,\n]'
        match = re.search(pattern, full_code)

        if not match:
            return None

        value = match.group(1).strip()

        # Number literal
        if re.match(r'^\d+(\.\d+)?$', value):
            return 'number'

        # String literal
        if value.startswith('"') or value.startswith("'") or value.startswith('`'):
            return 'string'

        # Boolean literal
        if value in ['true', 'false']:
            return 'boolean'

        # Array literal
        if value.startswith('['):
            return 'any[]'  # Need more context for element type

        # Object literal
        if value.startswith('{'):
            return 'any'  # Need more context for object shape

        # Otherwise unknown
        return None

    def detect_unused_imports(self, full_code: str) -> List[Dict[str, Any]]:
        """
        Detect all unused imports in TypeScript file.

        Returns list of issues:
        [
            {
                'category': 'dead_code',
                'message': "Unused import 'Foo'",
                'line_number': 5,
                'code_snippet': "import { Foo } from './module'",
                'severity': 'low'
            }
        ]

        Note: This is a simple regex-based detection.
        For production, use TypeScript Language Server or ESLint.
        """
        issues = []
        lines = full_code.split('\n')

        # Extract all imports
        import_pattern = r"import\s+(?:type\s+)?(?:\{([^}]+)\}|\*\s+as\s+(\w+)|(\w+))\s+from\s+['\"](.+?)['\"]"

        for i, line in enumerate(lines, start=1):
            match = re.search(import_pattern, line)
            if not match:
                continue

            # Extract imported names
            named_imports = match.group(1)  # { Foo, Bar }
            namespace_import = match.group(2)  # * as vscode
            default_import = match.group(3)  # React

            imports_to_check = []

            if named_imports:
                # Parse named imports
                names = [name.strip() for name in named_imports.split(',')]
                imports_to_check.extend(names)
            elif namespace_import:
                imports_to_check.append(namespace_import)
            elif default_import:
                imports_to_check.append(default_import)

            # Check if each import is used
            for imported_name in imports_to_check:
                # Simple heuristic: Search for usage in rest of file
                # Exclude the import line itself
                code_without_import = '\n'.join(lines[:i-1] + lines[i:])

                # Pattern: Look for word boundary usage
                usage_pattern = rf'\b{re.escape(imported_name)}\b'
                if not re.search(usage_pattern, code_without_import):
                    issues.append({
                        'category': 'dead_code',
                        'message': f"Unused import '{imported_name}'",
                        'line_number': i,
                        'code_snippet': line.strip(),
                        'severity': 'low'
                    })

        return issues


# Example usage and testing
if __name__ == '__main__':
    import asyncio

    async def demo():
        """Demo TypeScript autofix"""

        # Example TypeScript code with issues
        ts_code = """import * as vscode from 'vscode';
import { HoloLoomBridge, CodeContext } from './HoloLoomBridge';

export class Example {
    private bridge: HoloLoomBridge;

    constructor(bridge: HoloLoomBridge) {
        this.bridge = bridge;
        console.log('Example created');  // Should be removed
    }

    async process(data: any) {  // Should fix 'any'
        const value = data!.value;  // Should use optional chaining
        return value;
    }
}
"""

        fixer = TypeScriptFixer(use_llm=False)

        # Detect unused imports
        print("=== Detecting Unused Imports ===")
        issues = fixer.detect_unused_imports(ts_code)
        for issue in issues:
            print(f"Line {issue['line_number']}: {issue['message']}")
            print(f"  {issue['code_snippet']}")
        print()

        # Fix unused import
        if issues:
            print("=== Fixing Unused Import ===")
            result = await fixer.fix_issue(issues[0], ts_code, 'example.ts')
            if result:
                fixed_code, diff = result
                print("Diff:")
                print(diff)
            print()

        # Fix any type
        print("=== Fixing 'any' Type ===")
        any_issue = {
            'category': 'type_safety',
            'message': "Parameter 'data' implicitly has an 'any' type",
            'line_number': 11,
            'code_snippet': 'async process(data: any) {'
        }
        result = await fixer.fix_issue(any_issue, ts_code, 'example.ts')
        if result:
            fixed_code, diff = result
            print("Diff:")
            print(diff)
        print()

        # Fix non-null assertion
        print("=== Fixing Non-Null Assertion ===")
        nonnull_issue = {
            'category': 'type_safety',
            'message': "Non-null assertion is risky",
            'line_number': 12,
            'code_snippet': "const value = data!.value;"
        }
        result = await fixer.fix_issue(nonnull_issue, ts_code, 'example.ts')
        if result:
            fixed_code, diff = result
            print("Diff:")
            print(diff)
        print()

        # Fix console.log
        print("=== Fixing console.log ===")
        console_issue = {
            'category': 'code_quality',
            'message': "Unexpected console.log",
            'line_number': 8,
            'code_snippet': "console.log('Example created');"
        }
        result = await fixer.fix_issue(console_issue, ts_code, 'example.ts')
        if result:
            fixed_code, diff = result
            print("Diff:")
            print(diff)

    asyncio.run(demo())
