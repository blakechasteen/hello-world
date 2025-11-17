"""
Calculator Tool - Mathematical Operations
==========================================
Safe mathematical expression evaluation.
"""

import ast
import operator
import math
from typing import Union


class SafeCalculator:
    """
    Safe calculator that evaluates mathematical expressions.

    Prevents arbitrary code execution by only allowing whitelisted operations.
    """

    # Allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Allowed functions
    functions = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }

    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Numeric result

        Raises:
            ValueError: If expression is invalid or unsafe
        """
        try:
            # Parse expression
            tree = ast.parse(expression, mode='eval')

            # Evaluate safely
            return self._eval_node(tree.body)

        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    def _eval_node(self, node):
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Num):  # Number
            return node.n

        elif isinstance(node, ast.BinOp):  # Binary operation
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):  # Unary operation
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

            operand = self._eval_node(node.operand)
            return op(operand)

        elif isinstance(node, ast.Call):  # Function call
            func_name = node.func.id
            func = self.functions.get(func_name)
            if func is None:
                raise ValueError(f"Unsupported function: {func_name}")

            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)

        elif isinstance(node, ast.Name):  # Variable (only constants allowed)
            const = self.functions.get(node.id)
            if const is None:
                raise ValueError(f"Undefined variable: {node.id}")
            return const

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def register_calculator(server):
    """Register calculator tool with MCP server."""

    calc = SafeCalculator()

    @server.tool(
        "calculator",
        "Perform safe mathematical calculations",
        category="math",
        timeout=5.0
    )
    async def calculator(expression: str, precision: int = 4) -> Union[int, float]:
        """
        Calculate mathematical expression.

        Args:
            expression: Math expression (e.g., "2 + 2 * 3")
            precision: Decimal precision for rounding

        Returns:
            Numeric result

        Examples:
            - "2 + 2" → 4
            - "sqrt(16)" → 4.0
            - "sin(pi/2)" → 1.0
        """
        result = calc.evaluate(expression)

        # Round if precision specified
        if precision >= 0 and isinstance(result, float):
            result = round(result, precision)

        return result


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    calc = SafeCalculator()

    print("="*80)
    print("Calculator Tool Demo")
    print("="*80 + "\n")

    test_cases = [
        "2 + 2",
        "10 - 3 * 2",
        "(5 + 3) * 2",
        "2 ** 8",
        "sqrt(16)",
        "sin(pi/2)",
        "abs(-42)",
        "max(1, 5, 3)",
    ]

    for expr in test_cases:
        try:
            result = calc.evaluate(expr)
            print(f"  {expr:20s} = {result}")
        except ValueError as e:
            print(f"  {expr:20s} = Error: {e}")

    print("\n✓ Demo complete!")
