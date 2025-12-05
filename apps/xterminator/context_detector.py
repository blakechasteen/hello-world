"""
Context Detector
================

Analyzes code context to determine where an issue appears:
- Is it in a comment? (Skip fixing)
- Is it in a string? (Skip fixing)
- Is it in executable code? (Safe to analyze)
- What's the AST context? (Function, class, module)
- What's the semantic context? (In try block, loop, etc.)

This prevents false positives from "fixing" SQL keywords in comments,
or modifying code inside string literals.
"""

import ast
import re
from typing import List, Optional, Tuple
from .xterminator_types import CodeContext, ContextType


class ContextDetector:
    """Detects code context for accurate issue classification"""

    def __init__(self):
        self.comment_pattern = re.compile(r'^\s*#')
        self.docstring_patterns = [
            re.compile(r'^\s*"""'),
            re.compile(r"^\s*'''"),
        ]

    def analyze_context(
        self,
        file_path: str,
        line_number: int,
        code_lines: List[str],
        full_code: Optional[str] = None
    ) -> CodeContext:
        """
        Analyze code context at given location.

        Args:
            file_path: Path to file
            line_number: Line number (1-indexed)
            code_lines: Lines of code around issue
            full_code: Full file content (for AST analysis)

        Returns:
            CodeContext with classification
        """
        # Adjust to 0-indexed
        line_idx = line_number - 1

        # Get surrounding lines
        lines_before = code_lines[max(0, line_idx - 3):line_idx]
        target_line = code_lines[line_idx] if line_idx < len(code_lines) else ""
        lines_after = code_lines[line_idx + 1:min(len(code_lines), line_idx + 4)]

        # Detect context type
        context_type = self._detect_context_type(
            target_line,
            lines_before,
            lines_after
        )

        # Build base context
        context = CodeContext(
            context_type=context_type,
            file_path=file_path,
            line_number=line_number,
            code_snippet=target_line,
            lines_before=lines_before,
            lines_after=lines_after
        )

        # Add AST context if available
        if full_code and context_type == ContextType.EXECUTABLE:
            self._add_ast_context(context, full_code, line_number)

        # Add semantic context
        self._add_semantic_context(context, lines_before, target_line, lines_after)

        return context

    def _detect_context_type(
        self,
        line: str,
        lines_before: List[str],
        lines_after: List[str]
    ) -> ContextType:
        """Detect what type of context this line is in"""

        # Check if it's a comment
        if self.comment_pattern.match(line):
            return ContextType.COMMENT

        # Check if we're inside a docstring
        if self._in_docstring(lines_before, line):
            return ContextType.DOCSTRING

        # Check if it's in a string literal (basic heuristic)
        if self._in_string_literal(line):
            return ContextType.STRING_LITERAL

        # Check if it's an import
        if line.strip().startswith(('import ', 'from ')):
            return ContextType.IMPORT

        # Check if it's a definition
        if re.match(r'^\s*(def|class|async def)\s+', line):
            return ContextType.DEFINITION

        # Check if it's a type annotation (basic heuristic)
        if '->' in line or ': ' in line and '=' not in line:
            # Might be annotation, but not conclusive
            pass

        # Default: executable code
        return ContextType.EXECUTABLE

    def _in_docstring(self, lines_before: List[str], current_line: str) -> bool:
        """Check if we're inside a docstring"""
        # Count triple quotes before current line
        triple_quote_count = 0
        for line in lines_before:
            triple_quote_count += line.count('"""')
            triple_quote_count += line.count("'''")

        # Odd count means we're inside a docstring
        if triple_quote_count % 2 == 1:
            return True

        # Check if current line starts a docstring
        for pattern in self.docstring_patterns:
            if pattern.match(current_line):
                return True

        return False

    def _in_string_literal(self, line: str) -> bool:
        """Check if line is inside a string literal (basic heuristic)"""
        # This is a simplified check - just looks for quotes
        # Real implementation would need proper parsing
        stripped = line.strip()

        # Check for f-strings, r-strings, etc.
        if any(stripped.startswith(prefix) for prefix in ['f"', "f'", 'r"', "r'", 'b"', "b'"]):
            return True

        # Check if starts with quote and doesn't have assignment
        if (stripped.startswith(('"', "'")) and '=' not in stripped):
            return True

        return False

    def _add_ast_context(
        self,
        context: CodeContext,
        full_code: str,
        line_number: int
    ) -> None:
        """Add AST-based context information"""
        try:
            tree = ast.parse(full_code)
            finder = ASTContextFinder(line_number)
            finder.visit(tree)

            context.parent_node_type = finder.parent_type
            context.parent_node_name = finder.parent_name
            context.scope_depth = finder.scope_depth

        except SyntaxError:
            # Can't parse - leave AST context as None
            pass

    def _add_semantic_context(
        self,
        context: CodeContext,
        lines_before: List[str],
        current_line: str,
        lines_after: List[str]
    ) -> None:
        """Add semantic context (try blocks, loops, etc.)"""

        # Combine lines for analysis
        all_lines = lines_before + [current_line] + lines_after
        combined = '\n'.join(all_lines)

        # Check for try block
        context.in_try_block = any(
            re.match(r'^\s*try:', line) for line in lines_before[-5:]
        )

        # Check for loop
        context.in_loop = any(
            re.match(r'^\s*(for|while)\s+', line) for line in lines_before[-5:]
        )

        # Check for conditional
        context.in_conditional = any(
            re.match(r'^\s*(if|elif)\s+', line) for line in lines_before[-5:]
        )

        # Check if file has tests (heuristic: test_ prefix or _test suffix)
        context.has_tests = (
            'test_' in context.file_path or
            '_test.py' in context.file_path or
            any('def test_' in line for line in all_lines)
        )


class ASTContextFinder(ast.NodeVisitor):
    """AST visitor to find context of a specific line"""

    def __init__(self, target_line: int):
        self.target_line = target_line
        self.parent_type: Optional[str] = None
        self.parent_name: Optional[str] = None
        self.scope_depth = 0
        self.current_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_node(node, 'FunctionDef', node.name)
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_node(node, 'AsyncFunctionDef', node.name)
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_node(node, 'ClassDef', node.name)
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def _check_node(self, node: ast.AST, node_type: str, node_name: str) -> None:
        """Check if target line is within this node"""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            if node.lineno <= self.target_line <= (node.end_lineno or node.lineno):
                # Target is in this node - update context
                self.parent_type = node_type
                self.parent_name = node_name
                self.scope_depth = self.current_depth
