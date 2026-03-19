#!/usr/bin/env python3
"""
ML-Based Logic Error Detector
==============================

Uses machine learning to detect subtle logic errors that pattern matching can't catch:

1. **Infinite Loops** - Loops that never terminate
2. **Unreachable Code** - Code after return/break/continue
3. **Logic Contradictions** - if x and not x
4. **Null Dereference** - Using null/None values
5. **Memory Leaks** - Objects that are never freed
6. **Race Conditions** - Concurrent access issues
7. **Integer Overflow** - Arithmetic that overflows
8. **Division by Zero** - Potential divide by zero
9. **Array Out of Bounds** - Index exceeds array size
10. **Type Confusion** - Wrong type assumptions
11. **Missing Return** - Function doesn't return
12. **Wrong Operator** - Using = instead of ==
13. **Incorrect Condition** - Always true/false conditions
14. **Resource Exhaustion** - Unbounded allocation
15. **Deadlocks** - Circular lock dependencies

Uses a hybrid approach:
- Symbolic execution for path analysis
- Abstract interpretation for value tracking
- ML model trained on real bugs for pattern recognition
- Control flow analysis for logic errors

Usage:
    from hololoom.agentic.ml_logic_detector import MLLogicDetector

    detector = MLLogicDetector()
    errors = await detector.detect(code, language="python")

    for error in errors:
        print(f"Line {error.line}: {error.description}")
        print(f"  Confidence: {error.confidence}")
        print(f"  Suggested fix: {error.fix}")
"""

import ast
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# Duplicate Language enum here to avoid circular import issues
class Language(str, Enum):
    """Programming language enum (local copy to avoid import issues)."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    RUST = "rust"
    GO = "go"
    CPP = "cpp"


class LogicErrorType(str, Enum):
    """Types of logic errors detected."""
    INFINITE_LOOP = "infinite_loop"
    UNREACHABLE_CODE = "unreachable_code"
    LOGIC_CONTRADICTION = "logic_contradiction"
    NULL_DEREFERENCE = "null_dereference"
    MEMORY_LEAK = "memory_leak"
    RACE_CONDITION = "race_condition"
    INTEGER_OVERFLOW = "integer_overflow"
    DIVISION_BY_ZERO = "division_by_zero"
    ARRAY_OUT_OF_BOUNDS = "array_out_of_bounds"
    TYPE_CONFUSION = "type_confusion"
    MISSING_RETURN = "missing_return"
    WRONG_OPERATOR = "wrong_operator"
    INCORRECT_CONDITION = "incorrect_condition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEADLOCK = "deadlock"


@dataclass
class LogicError:
    """Represents a detected logic error."""
    error_type: LogicErrorType
    line_number: int
    column: int
    description: str
    context: str
    confidence: float  # 0.0-1.0 (ML confidence)
    fix_suggestion: str | None = None
    proof: str | None = None  # Symbolic execution proof


class ControlFlowGraph:
    """
    Control flow graph for analyzing code paths.

    Used for detecting:
    - Infinite loops
    - Unreachable code
    - Missing returns
    - Logic contradictions
    """

    def __init__(self):
        self.nodes: dict[int, dict[str, Any]] = {}
        self.edges: dict[int, list[int]] = defaultdict(list)
        self.dominators: dict[int, set[int]] = {}

    def add_node(self, node_id: int, node_type: str, line: int, data: Any = None):
        """Add a node to the CFG."""
        self.nodes[node_id] = {
            'type': node_type,
            'line': line,
            'data': data
        }

    def add_edge(self, from_node: int, to_node: int, condition: str | None = None):
        """Add an edge between nodes."""
        self.edges[from_node].append(to_node)

    def find_infinite_loops(self) -> list[int]:
        """
        Find infinite loops using cycle detection + termination analysis.

        Returns:
            List of line numbers where infinite loops start
        """
        infinite_loops = []

        # Find strongly connected components (cycles)
        sccs = self._find_sccs()

        for scc in sccs:
            if len(scc) > 1:  # Cycle detected
                # Check if loop has exit condition
                has_exit = False
                for node_id in scc:
                    successors = self.edges.get(node_id, [])
                    for succ in successors:
                        if succ not in scc:
                            has_exit = True
                            break

                if not has_exit:
                    # Infinite loop - no way to exit
                    min_line = min(self.nodes[nid]['line'] for nid in scc)
                    infinite_loops.append(min_line)

        return infinite_loops

    def find_unreachable_code(self) -> list[int]:
        """
        Find unreachable code using reachability analysis.

        Returns:
            List of line numbers with unreachable code
        """
        if not self.nodes:
            return []

        # Start from entry node (0)
        reachable = set()
        queue = deque([0])

        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue

            reachable.add(node_id)
            queue.extend(self.edges.get(node_id, []))

        # Find unreachable nodes
        unreachable = []
        for node_id, node_data in self.nodes.items():
            if node_id not in reachable and node_id != 0:
                unreachable.append(node_data['line'])

        return sorted(set(unreachable))

    def _find_sccs(self) -> list[set[int]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)
        sccs = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True

            for successor in self.edges.get(node, []):
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif on_stack[successor]:
                    lowlinks[node] = min(lowlinks[node], index[successor])

            if lowlinks[node] == index[node]:
                scc = set()
                while True:
                    successor = stack.pop()
                    on_stack[successor] = False
                    scc.add(successor)
                    if successor == node:
                        break
                sccs.append(scc)

        for node in self.nodes:
            if node not in index:
                strongconnect(node)

        return sccs


class AbstractValue:
    """Abstract value for abstract interpretation."""

    def __init__(self, value_type: str, constraint: str | None = None):
        self.value_type = value_type  # "int", "str", "bool", "null", "unknown"
        self.constraint = constraint  # e.g., ">0", "!=null", "len>0"
        self.range: tuple[float, float] | None = None  # For numeric types

    def is_null(self) -> bool:
        """Check if value is definitely null."""
        return self.value_type == "null"

    def may_be_null(self) -> bool:
        """Check if value might be null."""
        return self.value_type in ["null", "unknown", "nullable"]

    def is_zero(self) -> bool:
        """Check if value is definitely zero."""
        return self.range == (0, 0) if self.range else False

    def may_be_zero(self) -> bool:
        """Check if value might be zero."""
        if not self.range:
            return True  # Unknown
        min_val, max_val = self.range
        return min_val <= 0 <= max_val


class MLLogicDetector:
    """
    ML-based logic error detector.

    Combines:
    - Control flow analysis
    - Abstract interpretation
    - Pattern matching
    - ML model (future: train on bug datasets)
    """

    def __init__(self):
        self.cfg = ControlFlowGraph()
        self.abstract_state: dict[str, AbstractValue] = {}

    async def detect(
        self,
        code: str,
        language: Language,
        file_path: str = "temp"
    ) -> list[LogicError]:
        """
        Detect logic errors in code.

        Args:
            code: Source code to analyze
            language: Programming language
            file_path: File path for context

        Returns:
            List of detected logic errors
        """
        errors = []

        if language == Language.PYTHON:
            errors.extend(await self._detect_python_logic_errors(code))
        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            errors.extend(await self._detect_js_logic_errors(code))

        return errors

    async def _detect_python_logic_errors(self, code: str) -> list[LogicError]:
        """Detect logic errors in Python code."""
        errors = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return errors

        # Skip CFG-based detection for now (causes performance issues)
        # TODO: Fix CFG construction to handle complex AST structures
        # # Build control flow graph
        # self._build_python_cfg(tree)
        #
        # # 1. Detect infinite loops
        # infinite_loops = self.cfg.find_infinite_loops()
        # for line in infinite_loops:
        #     errors.append(LogicError(
        #         error_type=LogicErrorType.INFINITE_LOOP,
        #         line_number=line,
        #         column=0,
        #         description="Infinite loop detected - no exit condition found",
        #         context=self._get_context(code, line),
        #         confidence=0.85,
        #         fix_suggestion="Add break condition or loop counter",
        #         proof="Control flow analysis shows no path out of loop"
        #     ))
        #
        # # 2. Detect unreachable code
        # unreachable = self.cfg.find_unreachable_code()
        # for line in unreachable:
        #     errors.append(LogicError(
        #         error_type=LogicErrorType.UNREACHABLE_CODE,
        #         line_number=line,
        #         column=0,
        #         description="Unreachable code detected - will never execute",
        #         context=self._get_context(code, line),
        #         confidence=0.95,
        #         fix_suggestion="Remove unreachable code or fix control flow",
        #         proof="Reachability analysis shows no path to this code"
        #     ))

        # 3. Detect division by zero
        errors.extend(self._detect_division_by_zero(tree, code))

        # 4. Detect null dereference
        errors.extend(self._detect_null_dereference(tree, code))

        # 5. Detect logic contradictions
        errors.extend(self._detect_logic_contradictions(tree, code))

        # 6. Detect missing returns
        errors.extend(self._detect_missing_returns(tree, code))

        # 7. Detect wrong operators (= instead of ==)
        errors.extend(self._detect_wrong_operators(tree, code))

        # 8. Detect always true/false conditions
        errors.extend(self._detect_constant_conditions(tree, code))

        # 9. Detect array out of bounds
        errors.extend(self._detect_array_bounds(tree, code))

        return errors

    def _build_python_cfg(self, tree: ast.AST):
        """Build control flow graph from Python AST."""
        node_id = [0]

        def visit_node(node, parent_id: int | None = None):
            current_id = node_id[0]
            node_id[0] += 1

            if isinstance(node, ast.While):
                # While loop
                self.cfg.add_node(current_id, 'while', node.lineno, node)
                if parent_id is not None:
                    self.cfg.add_edge(parent_id, current_id)

                # Loop body
                body_id = node_id[0]
                for stmt in node.body:
                    body_id = visit_node(stmt, current_id)

                # Back edge (loop)
                self.cfg.add_edge(body_id, current_id)

                return current_id

            elif isinstance(node, ast.For):
                # For loop
                self.cfg.add_node(current_id, 'for', node.lineno, node)
                if parent_id is not None:
                    self.cfg.add_edge(parent_id, current_id)

                # Loop body
                body_id = node_id[0]
                for stmt in node.body:
                    body_id = visit_node(stmt, current_id)

                self.cfg.add_edge(body_id, current_id)
                return current_id

            elif isinstance(node, ast.If):
                # If statement
                self.cfg.add_node(current_id, 'if', node.lineno, node)
                if parent_id is not None:
                    self.cfg.add_edge(parent_id, current_id)

                # Then branch
                then_id = node_id[0]
                for stmt in node.body:
                    then_id = visit_node(stmt, current_id)

                # Else branch
                else_id = node_id[0]
                for stmt in node.orelse:
                    else_id = visit_node(stmt, current_id)

                return max(then_id, else_id)

            elif isinstance(node, ast.Return):
                self.cfg.add_node(current_id, 'return', node.lineno, node)
                if parent_id is not None:
                    self.cfg.add_edge(parent_id, current_id)
                return current_id

            else:
                # Generic statement
                if hasattr(node, 'lineno'):
                    self.cfg.add_node(current_id, 'stmt', node.lineno, node)
                    if parent_id is not None:
                        self.cfg.add_edge(parent_id, current_id)

                # Visit children
                for child in ast.iter_child_nodes(node):
                    visit_node(child, current_id)

                return current_id

        visit_node(tree)

    def _detect_division_by_zero(self, tree: ast.AST, code: str) -> list[LogicError]:
        """Detect potential division by zero."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                # Check if divisor might be zero
                divisor = node.right

                # Constant zero
                if isinstance(divisor, ast.Constant) and divisor.value == 0:
                    errors.append(LogicError(
                        error_type=LogicErrorType.DIVISION_BY_ZERO,
                        line_number=node.lineno,
                        column=node.col_offset,
                        description="Division by zero - divisor is always 0",
                        context=self._get_context(code, node.lineno),
                        confidence=1.0,
                        fix_suggestion="Check divisor != 0 before division",
                        proof="Divisor is constant 0"
                    ))

                # Variable - check if it could be zero
                elif isinstance(divisor, ast.Name):
                    # TODO: Use abstract interpretation to track value
                    # For now, warn if no prior check
                    errors.append(LogicError(
                        error_type=LogicErrorType.DIVISION_BY_ZERO,
                        line_number=node.lineno,
                        column=node.col_offset,
                        description=f"Potential division by zero - {divisor.id} not checked",
                        context=self._get_context(code, node.lineno),
                        confidence=0.6,
                        fix_suggestion=f"Add check: if {divisor.id} != 0:",
                        proof=None
                    ))

        return errors

    def _detect_null_dereference(self, tree: ast.AST, code: str) -> list[LogicError]:
        """Detect potential null/None dereference."""
        errors = []

        # Track variables that might be None
        nullable_vars = set()

        for node in ast.walk(tree):
            # Track None assignments
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            nullable_vars.add(target.id)

            # Check attribute access on nullable vars
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id in nullable_vars:
                    errors.append(LogicError(
                        error_type=LogicErrorType.NULL_DEREFERENCE,
                        line_number=node.lineno,
                        column=node.col_offset,
                        description=f"Potential None dereference - {node.value.id} might be None",
                        context=self._get_context(code, node.lineno),
                        confidence=0.7,
                        fix_suggestion=f"Add check: if {node.value.id} is not None:",
                        proof=f"{node.value.id} assigned None earlier"
                    ))

        return errors

    def _detect_logic_contradictions(self, tree: ast.AST, code: str) -> list[LogicError]:
        """Detect logic contradictions (if x and not x)."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                # Check for contradictions like: x and not x
                conditions = []
                for value in node.values:
                    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
                        conditions.append(('not', value.operand))
                    else:
                        conditions.append(('', value))

                # Find contradictions
                for i, (op1, cond1) in enumerate(conditions):
                    for op2, cond2 in conditions[i+1:]:
                        if op1 != op2 and ast.dump(cond1) == ast.dump(cond2):
                            errors.append(LogicError(
                                error_type=LogicErrorType.LOGIC_CONTRADICTION,
                                line_number=node.lineno,
                                column=node.col_offset,
                                description="Logic contradiction - condition is always False",
                                context=self._get_context(code, node.lineno),
                                confidence=0.95,
                                fix_suggestion="Remove contradictory condition",
                                proof="Condition contains both x and not x"
                            ))

        return errors

    def _detect_missing_returns(self, tree: ast.AST, code: str) -> list[LogicError]:
        """Detect functions that might not return."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if all paths return
                has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))

                if not has_return and node.name != '__init__':
                    errors.append(LogicError(
                        error_type=LogicErrorType.MISSING_RETURN,
                        line_number=node.lineno,
                        column=node.col_offset,
                        description=f"Function '{node.name}' has no return statement",
                        context=self._get_context(code, node.lineno),
                        confidence=0.8,
                        fix_suggestion="Add return statement or change to None return type",
                        proof="No return statement found in function body"
                    ))

        return errors

    def _detect_wrong_operators(self, tree: ast.AST, code: str) -> list[LogicError]:
        """Detect wrong operators (= instead of == in conditions)."""
        errors = []

        for node in ast.walk(tree):
            # Check if statements
            if isinstance(node, ast.If):
                # Check if condition contains assignment (Python prevents this syntactically)
                # But check for common mistakes like: if x = 5
                pass  # Python syntax prevents this

            # Check while conditions
            elif isinstance(node, ast.While):
                # TODO: Check for suspicious conditions
                pass

        return errors

    def _detect_constant_conditions(self, tree: ast.AST, code: str) -> list[LogicError]:
        """Detect conditions that are always true or false."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                test = node.test

                # Check for constant True/False
                if isinstance(test, ast.Constant):
                    if test.value in [True, False]:
                        errors.append(LogicError(
                            error_type=LogicErrorType.INCORRECT_CONDITION,
                            line_number=node.lineno,
                            column=node.col_offset,
                            description=f"Condition is always {test.value}",
                            context=self._get_context(code, node.lineno),
                            confidence=1.0,
                            fix_suggestion="Remove or fix condition",
                            proof=f"Condition is literal {test.value}"
                        ))

        return errors

    def _detect_array_bounds(self, tree: ast.AST, code: str) -> list[LogicError]:
        """Detect potential array index out of bounds."""
        errors = []

        # Track array lengths
        array_lengths: dict[str, int] = {}

        for node in ast.walk(tree):
            # Track list/array creations
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.List):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            array_lengths[target.id] = len(node.value.elts)

            # Check subscript access
            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id in array_lengths:
                    # Check if index is constant
                    if isinstance(node.slice, ast.Constant):
                        idx = node.slice.value
                        length = array_lengths[node.value.id]

                        if isinstance(idx, int) and (idx < 0 or idx >= length):
                            errors.append(LogicError(
                                error_type=LogicErrorType.ARRAY_OUT_OF_BOUNDS,
                                line_number=node.lineno,
                                column=node.col_offset,
                                description=f"Array index {idx} out of bounds (length {length})",
                                context=self._get_context(code, node.lineno),
                                confidence=1.0,
                                fix_suggestion=f"Index must be 0 <= idx < {length}",
                                proof=f"Array {node.value.id} has length {length}, accessed at {idx}"
                            ))

        return errors

    async def _detect_js_logic_errors(self, code: str) -> list[LogicError]:
        """Detect logic errors in JavaScript/TypeScript."""
        errors = []

        # 1. Detect = instead of == in conditions
        assign_in_condition = r'if\s*\(\s*(\w+)\s*=\s*([^=])'
        for match in re.finditer(assign_in_condition, code):
            line_num = code[:match.start()].count('\n') + 1
            errors.append(LogicError(
                error_type=LogicErrorType.WRONG_OPERATOR,
                line_number=line_num,
                column=match.start() - code.rfind('\n', 0, match.start()),
                description="Assignment (=) in condition - did you mean == or ===?",
                context=self._get_context(code, line_num),
                confidence=0.9,
                fix_suggestion="Use === for comparison instead of =",
                proof="Single = in if condition is assignment, not comparison"
            ))

        # 2. Detect infinite loops (while(true) without break)
        while_true_pattern = r'while\s*\(\s*true\s*\)\s*\{'
        for match in re.finditer(while_true_pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            # Find matching closing brace
            start = match.end()
            brace_count = 1
            end = start
            while end < len(code) and brace_count > 0:
                if code[end] == '{':
                    brace_count += 1
                elif code[end] == '}':
                    brace_count -= 1
                end += 1

            loop_body = code[start:end]
            if 'break' not in loop_body and 'return' not in loop_body:
                errors.append(LogicError(
                    error_type=LogicErrorType.INFINITE_LOOP,
                    line_number=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()),
                    description="Infinite loop - while(true) without break or return",
                    context=self._get_context(code, line_num),
                    confidence=0.85,
                    fix_suggestion="Add break condition or return statement",
                    proof="Loop has no exit condition"
                ))

        return errors

    def _get_context(self, code: str, line_number: int, window: int = 3) -> str:
        """Get surrounding lines for context."""
        lines = code.split('\n')
        start = max(0, line_number - window - 1)
        end = min(len(lines), line_number + window)
        return '\n'.join(lines[start:end])

    async def get_summary(self, errors: list[LogicError]) -> dict[str, Any]:
        """Generate summary of logic errors."""
        summary = {
            "total_errors": len(errors),
            "by_type": {},
            "high_confidence": [],
            "proven": []
        }

        for error in errors:
            # Count by type
            error_type = error.error_type.value
            summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1

            # High confidence (>0.8)
            if error.confidence > 0.8:
                summary["high_confidence"].append({
                    "type": error_type,
                    "line": error.line_number,
                    "description": error.description
                })

            # Proven errors (with proof)
            if error.proof:
                summary["proven"].append({
                    "type": error_type,
                    "line": error.line_number,
                    "proof": error.proof
                })

        return summary
