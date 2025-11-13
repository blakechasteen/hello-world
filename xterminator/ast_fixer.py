"""
🐷 Wilbur - The AST Auto-Fixer Piglet
=====================================

"Some Pig wrote this code. I'm just making it prize-winning!" - Wilbur

Phase 2 of xTerminator: Automated AST transformations for safe code fixes.

Charlotte's Web of Wisdom:
- "A function extracted is better than a function duplicated!"
- "Dead code is like rotten food - throw it out!"
- "Magic numbers are like mud - best cleaned up!"
- "Templeton says: Unused imports are garbage - recycle them!"

Features:
- 6 AST transformations (extract function, remove dead code, etc.)
- Safety-first design (syntax validation, rollback support)
- Unified diff generation for human review
- Complete test coverage

Usage:
    from xterminator import ASTFixer

    fixer = ASTFixer()
    result = await fixer.fix_issue(proposal, full_code)

    if result:
        fixed_code, diff = result
        print(diff)
"""

import ast
import difflib
import re
from typing import Optional, Tuple, List, Set, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from .xterminator_types import (
    FixProposal,
    FixStrategy,
    RiskLevel,
    CodeContext,
    ContextType
)


@dataclass
class TransformationResult:
    """Result of an AST transformation"""
    success: bool
    transformed_code: Optional[str] = None
    diff: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ASTFixer:
    """
    🐷 AST Auto-Fixer - Transforms code safely using AST manipulations.

    "Some Pig wrote this code. I'm just making it prize-winning!" - Wilbur

    Implements 6 core transformations:
    1. Extract Function - Deduplicate copy-paste code
    2. Remove Dead Code - Delete unreachable statements
    3. Remove Unused Import - Clean up unused imports
    4. Extract Constant - Move magic numbers to constants
    5. Rename Variable - Fix naming inconsistencies
    6. Add Type Hint - Add missing type annotations
    """

    def __init__(self):
        self.transformations = {
            'copy_paste': self._extract_function,
            'dead_code': self._remove_dead_code,
            'unused_import': self._remove_unused_import,
            'magic_number': self._extract_constant,
            'naming': self._rename_variable,
            'missing_type_hint': self._add_type_hint
        }

    async def fix_issue(
        self,
        proposal: FixProposal,
        full_code: str
    ) -> Optional[Tuple[str, str]]:
        """
        Apply fix and return (fixed_code, diff).
        Returns None if fix failed or unsafe.

        Args:
            proposal: FixProposal from classification engine
            full_code: Complete file contents

        Returns:
            (fixed_code, diff) tuple or None if fix failed
        """
        # Charlotte's First Rule: Safety checks!
        if not proposal.safe_to_autofix:
            return None

        if proposal.fix_strategy != FixStrategy.AST:
            return None

        if proposal.risk_level not in {RiskLevel.LOW, RiskLevel.MEDIUM}:
            return None

        # Parse to AST
        try:
            tree = ast.parse(full_code)
        except SyntaxError as e:
            proposal.metadata['error'] = f"Syntax error in original code: {e}"
            return None

        # Apply transformation based on category
        result = await self._apply_transformation(tree, full_code, proposal)

        if not result.success:
            proposal.metadata['error'] = result.error
            return None

        # Generate diff with Charlotte's commentary
        diff = self._generate_diff(
            full_code,
            result.transformed_code,
            proposal
        )

        return (result.transformed_code, diff)

    async def _apply_transformation(
        self,
        tree: ast.AST,
        full_code: str,
        proposal: FixProposal
    ) -> TransformationResult:
        """Apply the appropriate AST transformation"""

        category = proposal.issue_category.lower()

        # Map categories to transformations
        if 'copy' in category or 'paste' in category or 'duplicate' in category:
            return await self._extract_function(tree, full_code, proposal)
        elif 'dead' in category or 'unreachable' in category:
            return await self._remove_dead_code(tree, full_code, proposal)
        elif 'unused' in category and 'import' in category:
            return await self._remove_unused_import(tree, full_code, proposal)
        elif 'magic' in category or 'hardcoded' in category:
            return await self._extract_constant(tree, full_code, proposal)
        elif 'naming' in category or 'inconsistent' in category:
            return await self._rename_variable(tree, full_code, proposal)
        elif 'type' in category and 'hint' in category:
            return await self._add_type_hint(tree, full_code, proposal)
        else:
            return TransformationResult(
                success=False,
                error=f"No transformation for category: {category}"
            )

    async def _extract_function(
        self,
        tree: ast.AST,
        full_code: str,
        proposal: FixProposal
    ) -> TransformationResult:
        """
        Extract duplicated code into a new function.

        "A function extracted is better than a function duplicated!" - Charlotte
        """
        try:
            # Find the duplicated code block
            lines = full_code.split('\n')
            context = proposal.context

            if not context:
                return TransformationResult(
                    success=False,
                    error="No context provided for extraction"
                )

            # Extract the code block (simplified - in production would use AST)
            start_line = max(0, context.line_number - 1)
            end_line = min(len(lines), start_line + 5)  # Extract ~5 lines

            code_block = lines[start_line:end_line]

            # Generate function name
            func_name = self._generate_function_name(code_block, proposal)

            # Detect variables used
            variables = self._detect_variables(code_block)

            # Build new function
            indent = len(code_block[0]) - len(code_block[0].lstrip())
            func_def = self._build_function_def(
                func_name,
                variables,
                code_block,
                indent
            )

            # Replace duplicates with function calls
            new_code = self._replace_with_function_call(
                lines,
                start_line,
                end_line,
                func_name,
                variables,
                func_def,
                indent
            )

            # Verify syntax
            try:
                ast.parse('\n'.join(new_code))
            except SyntaxError as e:
                return TransformationResult(
                    success=False,
                    error=f"Syntax error after transformation: {e}"
                )

            return TransformationResult(
                success=True,
                transformed_code='\n'.join(new_code),
                metadata={
                    'function_name': func_name,
                    'extracted_lines': end_line - start_line
                }
            )

        except Exception as e:
            return TransformationResult(
                success=False,
                error=f"Extract function failed: {e}"
            )

    async def _remove_dead_code(
        self,
        tree: ast.AST,
        full_code: str,
        proposal: FixProposal
    ) -> TransformationResult:
        """
        Remove unreachable or dead code.

        "Dead code is like rotten food - throw it out!" - Templeton
        """
        try:
            # Use AST visitor to find dead code
            visitor = DeadCodeVisitor()
            visitor.visit(tree)

            if not visitor.dead_lines:
                return TransformationResult(
                    success=False,
                    error="No dead code detected by AST analysis"
                )

            # Remove dead lines (0-indexed)
            lines = full_code.split('\n')
            new_lines = [
                line for i, line in enumerate(lines)
                if (i + 1) not in visitor.dead_lines  # AST uses 1-indexed line numbers
            ]

            # Verify syntax
            try:
                ast.parse('\n'.join(new_lines))
            except SyntaxError as e:
                return TransformationResult(
                    success=False,
                    error=f"Syntax error after removing dead code: {e}"
                )

            return TransformationResult(
                success=True,
                transformed_code='\n'.join(new_lines),
                metadata={
                    'removed_lines': len(visitor.dead_lines),
                    'dead_line_numbers': sorted(visitor.dead_lines)
                }
            )

        except Exception as e:
            return TransformationResult(
                success=False,
                error=f"Remove dead code failed: {e}"
            )

    async def _remove_unused_import(
        self,
        tree: ast.AST,
        full_code: str,
        proposal: FixProposal
    ) -> TransformationResult:
        """
        Remove unused import statements.

        "Unused imports are garbage - recycle them!" - Templeton
        """
        try:
            # Find all imports
            visitor = ImportVisitor()
            visitor.visit(tree)

            # Find all name references
            usage_visitor = NameUsageVisitor()
            usage_visitor.visit(tree)

            # Determine unused imports
            unused_imports = visitor.imports - usage_visitor.names

            if not unused_imports:
                return TransformationResult(
                    success=False,
                    error="No unused imports detected"
                )

            # Remove unused imports from code
            lines = full_code.split('\n')
            new_lines = []

            for i, line in enumerate(lines):
                # Check if this line imports something unused
                should_remove = False
                for unused in unused_imports:
                    if f"import {unused}" in line or f"from {unused}" in line:
                        should_remove = True
                        break

                if not should_remove:
                    new_lines.append(line)

            # Verify syntax
            try:
                ast.parse('\n'.join(new_lines))
            except SyntaxError as e:
                return TransformationResult(
                    success=False,
                    error=f"Syntax error after removing imports: {e}"
                )

            return TransformationResult(
                success=True,
                transformed_code='\n'.join(new_lines),
                metadata={
                    'removed_imports': list(unused_imports)
                }
            )

        except Exception as e:
            return TransformationResult(
                success=False,
                error=f"Remove unused import failed: {e}"
            )

    async def _extract_constant(
        self,
        tree: ast.AST,
        full_code: str,
        proposal: FixProposal
    ) -> TransformationResult:
        """
        Extract hardcoded values to module-level constants.

        "Magic numbers are like mud - best cleaned up!" - Wilbur
        """
        try:
            context = proposal.context
            if not context:
                return TransformationResult(
                    success=False,
                    error="No context provided"
                )

            # Extract the magic number/string from original code
            magic_value = self._extract_magic_value(proposal.original_code)

            if not magic_value:
                return TransformationResult(
                    success=False,
                    error="Could not identify magic value"
                )

            # Generate constant name
            const_name = self._generate_constant_name(magic_value, context)

            # Insert constant at top of file (after imports)
            lines = full_code.split('\n')
            insert_line = self._find_constant_insertion_point(lines)

            # Create constant definition
            const_def = f"{const_name} = {magic_value}"
            lines.insert(insert_line, const_def)

            # Find the line containing the magic value (search AFTER the inserted constant)
            search_value = magic_value.strip('"\'') if magic_value and magic_value[0] in '"\'' else magic_value
            target_line_idx = None
            for i in range(insert_line + 1, len(lines)):  # Skip the constant definition we just inserted
                if search_value in lines[i]:
                    target_line_idx = i
                    break

            # Replace magic value with constant
            if target_line_idx is not None:
                lines[target_line_idx] = lines[target_line_idx].replace(
                    search_value,
                    const_name,
                    1  # Only replace first occurrence
                )

            # Verify syntax
            new_code = '\n'.join(lines)
            try:
                ast.parse(new_code)
            except SyntaxError as e:
                return TransformationResult(
                    success=False,
                    error=f"Syntax error after extracting constant: {e}"
                )

            return TransformationResult(
                success=True,
                transformed_code=new_code,
                metadata={
                    'constant_name': const_name,
                    'magic_value': magic_value
                }
            )

        except Exception as e:
            return TransformationResult(
                success=False,
                error=f"Extract constant failed: {e}"
            )

    async def _rename_variable(
        self,
        tree: ast.AST,
        full_code: str,
        proposal: FixProposal
    ) -> TransformationResult:
        """
        Rename variable to follow conventions.

        "Consistency is key - even for a humble pig!" - Wilbur
        """
        try:
            # Extract old and new variable names from proposal
            hints = proposal.metadata.get('implementation_hints', {})
            old_name = hints.get('old_name')
            new_name = hints.get('new_name')

            if not old_name or not new_name:
                # Try to extract from original code
                match = re.search(r'(\w+)\s*=', proposal.original_code)
                if match:
                    old_name = match.group(1)
                    new_name = self._to_snake_case(old_name)
                else:
                    return TransformationResult(
                        success=False,
                        error="Could not determine variable names"
                    )

            # Use regex to replace (simple approach - AST approach would be more robust)
            new_code = re.sub(
                r'\b' + re.escape(old_name) + r'\b',
                new_name,
                full_code
            )

            # Verify syntax
            try:
                ast.parse(new_code)
            except SyntaxError as e:
                return TransformationResult(
                    success=False,
                    error=f"Syntax error after renaming: {e}"
                )

            return TransformationResult(
                success=True,
                transformed_code=new_code,
                metadata={
                    'old_name': old_name,
                    'new_name': new_name
                }
            )

        except Exception as e:
            return TransformationResult(
                success=False,
                error=f"Rename variable failed: {e}"
            )

    async def _add_type_hint(
        self,
        tree: ast.AST,
        full_code: str,
        proposal: FixProposal
    ) -> TransformationResult:
        """
        Add missing type hints to function definitions.

        "Types make code trustworthy!" - Charlotte
        """
        try:
            context = proposal.context
            if not context:
                return TransformationResult(
                    success=False,
                    error="No context provided"
                )

            # Find the function definition
            lines = full_code.split('\n')
            func_line_idx = context.line_number - 1

            if func_line_idx < 0 or func_line_idx >= len(lines):
                return TransformationResult(
                    success=False,
                    error="Invalid line number"
                )

            # Skip empty/whitespace lines to find actual function definition
            while func_line_idx < len(lines) and not lines[func_line_idx].strip():
                func_line_idx += 1

            if func_line_idx >= len(lines):
                return TransformationResult(
                    success=False,
                    error="Invalid line number - no code found"
                )

            func_line = lines[func_line_idx]

            # Parse function signature (handle multi-line too)
            match = re.match(r'(\s*)def\s+(\w+)\s*\((.*?)\)\s*:', func_line)
            if not match:
                return TransformationResult(
                    success=False,
                    error="Not a function definition"
                )

            indent, func_name, params = match.groups()

            # Add basic type hints (simplified - would infer from usage in production)
            typed_params = self._add_param_types(params)

            # Only add return type if not already present
            if '->' not in func_line:
                new_func_line = f"{indent}def {func_name}({typed_params}) -> None:"
            else:
                new_func_line = f"{indent}def {func_name}({typed_params}):"

            lines[func_line_idx] = new_func_line

            # Verify syntax
            new_code = '\n'.join(lines)
            try:
                ast.parse(new_code)
            except SyntaxError as e:
                return TransformationResult(
                    success=False,
                    error=f"Syntax error after adding type hints: {e}"
                )

            return TransformationResult(
                success=True,
                transformed_code=new_code,
                metadata={
                    'function_name': func_name
                }
            )

        except Exception as e:
            return TransformationResult(
                success=False,
                error=f"Add type hint failed: {e}"
            )

    # ==================== Helper Methods ====================

    def _generate_diff(
        self,
        original: str,
        transformed: str,
        proposal: FixProposal
    ) -> str:
        """
        Generate unified diff with Charlotte's commentary.
        """
        # Generate unified diff
        original_lines = original.splitlines(keepends=True)
        transformed_lines = transformed.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            transformed_lines,
            fromfile='original',
            tofile='fixed',
            lineterm=''
        )

        diff_text = ''.join(diff)

        # Add Charlotte's commentary
        header = self._get_charlotte_commentary(proposal)

        return f"{header}\n\n{diff_text}"

    def _get_charlotte_commentary(self, proposal: FixProposal) -> str:
        """Get Charlotte's wisdom for this fix"""
        category = proposal.issue_category.lower()

        if 'copy' in category or 'paste' in category:
            return "🕷️ Charlotte says: 'A function extracted is better than a function duplicated!'"
        elif 'dead' in category:
            return "🐀 Templeton's Rule: 'Dead code is like rotten food - throw it out!'"
        elif 'import' in category:
            return "🐀 Templeton says: 'Unused imports are garbage - recycle them!'"
        elif 'magic' in category or 'hardcoded' in category:
            return "🐷 Wilbur's Wisdom: 'Magic numbers are like mud - best cleaned up!'"
        elif 'naming' in category:
            return "🐷 Wilbur says: 'Consistency is key - even for a humble pig!'"
        elif 'type' in category:
            return "🕷️ Charlotte says: 'Types make code trustworthy!'"
        else:
            return "🐷 Some Pig made this code prize-winning!"

    def _generate_function_name(
        self,
        code_block: List[str],
        proposal: FixProposal
    ) -> str:
        """Generate meaningful function name from code block"""
        # Simple heuristic: extract verbs and nouns
        text = ' '.join(code_block).lower()

        # Look for common patterns
        if 'validate' in text:
            return 'validate_input'
        elif 'process' in text:
            return 'process_data'
        elif 'format' in text:
            return 'format_output'
        elif 'calculate' in text or 'compute' in text:
            return 'calculate_result'
        else:
            return 'extracted_function'

    def _detect_variables(self, code_block: List[str]) -> List[str]:
        """Detect variables used in code block"""
        # Simplified - would use AST in production
        variables = []
        for line in code_block:
            # Find assignments
            matches = re.findall(r'(\w+)\s*=', line)
            variables.extend(matches)
        return list(set(variables))

    def _build_function_def(
        self,
        func_name: str,
        variables: List[str],
        code_block: List[str],
        indent: int
    ) -> str:
        """Build function definition"""
        params = ', '.join(variables) if variables else ''

        func_lines = [
            f"def {func_name}({params}):",
            f'    """Extracted function"""'
        ]

        # Add code block with adjusted indent
        for line in code_block:
            func_lines.append('    ' + line.lstrip())

        return '\n'.join(func_lines)

    def _replace_with_function_call(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
        func_name: str,
        variables: List[str],
        func_def: str,
        indent: int
    ) -> List[str]:
        """Replace code block with function call"""
        new_lines = lines[:start_line]

        # Insert function definition
        new_lines.append(func_def)
        new_lines.append('')

        # Insert function call
        args = ', '.join(variables) if variables else ''
        call = ' ' * indent + f"{func_name}({args})"
        new_lines.append(call)

        # Add remaining lines
        new_lines.extend(lines[end_line:])

        return new_lines

    def _extract_magic_value(self, code: str) -> Optional[str]:
        """Extract magic value from code"""
        # Look for numbers
        match = re.search(r'\b(\d+\.?\d*)\b', code)
        if match:
            return match.group(1)

        # Look for strings
        match = re.search(r'["\']([^"\']+)["\']', code)
        if match:
            return f'"{match.group(1)}"'

        return None

    def _generate_constant_name(self, value: str, context: CodeContext) -> str:
        """Generate constant name from value"""
        # Remove quotes if string
        value_clean = value.strip('"\'')

        # Check if it's a number (int or float)
        try:
            float(value_clean)
            # It's a number - check for common values
            float_val = float(value_clean)
            if abs(float_val - 3.14159) < 0.00001 or abs(float_val - 3.14) < 0.01:
                return "PI"
            elif abs(float_val - 2.71828) < 0.00001:
                return "E"
            else:
                return "MAGIC_NUMBER"
        except ValueError:
            # Not a number - try to extract meaningful name from string
            words = re.findall(r'\w+', value_clean)
            if words:
                # Make sure first word doesn't start with a digit
                name_parts = []
                for word in words[:3]:
                    if word[0].isdigit():
                        name_parts.append(f"NUM_{word}")
                    else:
                        name_parts.append(word.upper())
                return '_'.join(name_parts)
            return "CONSTANT_VALUE"

    def _find_constant_insertion_point(self, lines: List[str]) -> int:
        """Find where to insert constant (after imports)"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('import') and not stripped.startswith('from'):
                return i
        return 0

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _add_param_types(self, params: str) -> str:
        """Add type hints to parameters (simplified)"""
        if not params or params.strip() in ('self', ''):
            return params

        # Split params
        param_list = [p.strip() for p in params.split(',')]

        # Add int type to each (simplified - would infer in production)
        # Using int instead of Any to avoid import issues
        typed_params = []
        for param in param_list:
            if param == 'self':
                typed_params.append('self')
            elif ':' not in param:
                typed_params.append(f"{param}: int")  # Simple default type
            else:
                typed_params.append(param)

        return ', '.join(typed_params)


# ==================== AST Visitors ====================


class DeadCodeVisitor(ast.NodeVisitor):
    """Visitor to detect dead/unreachable code"""

    def __init__(self):
        self.dead_lines = set()

    def _check_block_for_dead_code(self, statements):
        """Check a block of statements for dead code after returns"""
        seen_return = False
        for stmt in statements:
            if seen_return:
                # Everything after a return is dead
                self._mark_dead(stmt)
            elif isinstance(stmt, ast.Return):
                seen_return = True
            else:
                # Visit this statement to check nested blocks
                self.visit(stmt)

    def visit_FunctionDef(self, node):
        """Visit function and check for dead code"""
        self._check_block_for_dead_code(node.body)

    def visit_If(self, node):
        """Visit if-statement and check both branches"""
        self._check_block_for_dead_code(node.body)
        self._check_block_for_dead_code(node.orelse)

    def visit_For(self, node):
        """Visit for-loop and check body"""
        self._check_block_for_dead_code(node.body)
        self._check_block_for_dead_code(node.orelse)

    def visit_While(self, node):
        """Visit while-loop and check body"""
        self._check_block_for_dead_code(node.body)
        self._check_block_for_dead_code(node.orelse)

    def visit_With(self, node):
        """Visit with-statement and check body"""
        self._check_block_for_dead_code(node.body)

    def visit_Try(self, node):
        """Visit try-statement and check all blocks"""
        self._check_block_for_dead_code(node.body)
        for handler in node.handlers:
            self._check_block_for_dead_code(handler.body)
        self._check_block_for_dead_code(node.orelse)
        self._check_block_for_dead_code(node.finalbody)

    def _mark_dead(self, node):
        """Recursively mark node and all children as dead"""
        if hasattr(node, 'lineno'):
            self.dead_lines.add(node.lineno)
        # Walk all child nodes and mark them too
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                self.dead_lines.add(child.lineno)


class ImportVisitor(ast.NodeVisitor):
    """Visitor to collect all imports"""

    def __init__(self):
        self.imports = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)


class NameUsageVisitor(ast.NodeVisitor):
    """Visitor to collect all name references"""

    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Track module usage (e.g., os.path)
        if isinstance(node.value, ast.Name):
            self.names.add(node.value.id)
        self.generic_visit(node)
