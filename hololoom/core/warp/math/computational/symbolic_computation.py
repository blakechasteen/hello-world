"""
Symbolic Computation for HoloLoom Warp Drive
==============================================

Expression trees, symbolic differentiation, rewrite systems,
and pattern matching/unification.

No sympy dependency — pure Python expression trees with numpy evaluation.

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# SYMBOLIC EXPRESSION
# ============================================================================

class SymbolicExpr:
    """
    Symbolic expression tree.

    Nodes:
    - Constant: ('const', value)
    - Variable: ('var', name)
    - Binary op: ('add'|'mul'|'sub'|'div'|'pow', left, right)
    - Unary op: ('neg'|'sin'|'cos'|'exp'|'log'|'sqrt', arg)
    """

    def __init__(self, op: str, *args):
        self.op = op
        self.args = list(args)

    # --- Constructors ---

    @staticmethod
    def const(value: float) -> 'SymbolicExpr':
        return SymbolicExpr('const', value)

    @staticmethod
    def var(name: str) -> 'SymbolicExpr':
        return SymbolicExpr('var', name)

    # --- Arithmetic operators ---

    def __add__(self, other):
        other = other if isinstance(other, SymbolicExpr) else SymbolicExpr.const(other)
        return SymbolicExpr('add', self, other)

    def __radd__(self, other):
        return SymbolicExpr.const(other) + self

    def __mul__(self, other):
        other = other if isinstance(other, SymbolicExpr) else SymbolicExpr.const(other)
        return SymbolicExpr('mul', self, other)

    def __rmul__(self, other):
        return SymbolicExpr.const(other) * self

    def __sub__(self, other):
        other = other if isinstance(other, SymbolicExpr) else SymbolicExpr.const(other)
        return SymbolicExpr('sub', self, other)

    def __truediv__(self, other):
        other = other if isinstance(other, SymbolicExpr) else SymbolicExpr.const(other)
        return SymbolicExpr('div', self, other)

    def __pow__(self, other):
        other = other if isinstance(other, SymbolicExpr) else SymbolicExpr.const(other)
        return SymbolicExpr('pow', self, other)

    def __neg__(self):
        return SymbolicExpr('neg', self)

    # --- Unary functions ---

    def sin(self): return SymbolicExpr('sin', self)
    def cos(self): return SymbolicExpr('cos', self)
    def exp(self): return SymbolicExpr('exp', self)
    def log(self): return SymbolicExpr('log', self)
    def sqrt(self): return SymbolicExpr('sqrt', self)

    # --- Core operations ---

    def differentiate(self, var: str) -> 'SymbolicExpr':
        """
        Symbolic differentiation d/d(var).

        Chain rule, product rule, quotient rule applied recursively.
        """
        if self.op == 'const':
            return SymbolicExpr.const(0)

        if self.op == 'var':
            return SymbolicExpr.const(1.0 if self.args[0] == var else 0.0)

        if self.op == 'neg':
            return -self.args[0].differentiate(var)

        if self.op == 'add':
            return self.args[0].differentiate(var) + self.args[1].differentiate(var)

        if self.op == 'sub':
            return self.args[0].differentiate(var) - self.args[1].differentiate(var)

        if self.op == 'mul':
            # Product rule: (fg)' = f'g + fg'
            f, g = self.args
            return f.differentiate(var) * g + f * g.differentiate(var)

        if self.op == 'div':
            # Quotient rule: (f/g)' = (f'g - fg') / g²
            f, g = self.args
            return (f.differentiate(var) * g - f * g.differentiate(var)) / (g ** SymbolicExpr.const(2))

        if self.op == 'pow':
            f, g = self.args
            # General: (f^g)' = f^g (g' ln f + g f'/f)
            # Special case: constant exponent
            if g.op == 'const':
                n = g.args[0]
                return SymbolicExpr.const(n) * (f ** SymbolicExpr.const(n - 1)) * f.differentiate(var)
            # General power rule
            return self * (g.differentiate(var) * f.log() + g * f.differentiate(var) / f)

        if self.op == 'sin':
            # d/dx sin(f) = cos(f) f'
            f = self.args[0]
            return f.cos() * f.differentiate(var)

        if self.op == 'cos':
            # d/dx cos(f) = -sin(f) f'
            f = self.args[0]
            return -(f.sin()) * f.differentiate(var)

        if self.op == 'exp':
            f = self.args[0]
            return f.exp() * f.differentiate(var)

        if self.op == 'log':
            f = self.args[0]
            return f.differentiate(var) / f

        if self.op == 'sqrt':
            f = self.args[0]
            return f.differentiate(var) / (SymbolicExpr.const(2) * f.sqrt())

        raise ValueError(f"Unknown op: {self.op}")

    def simplify(self) -> 'SymbolicExpr':
        """
        Algebraic simplification (basic rules).

        0 + x = x, 1 * x = x, 0 * x = 0, x - x = 0, x/1 = x, x^0 = 1, x^1 = x.
        Constant folding when both operands are constants.
        """
        if self.op in ('const', 'var'):
            return self

        # Simplify children first
        simplified_args = [
            a.simplify() if isinstance(a, SymbolicExpr) else a
            for a in self.args
        ]

        # Constant folding
        if self.op in ('add', 'sub', 'mul', 'div', 'pow'):
            a, b = simplified_args
            if isinstance(a, SymbolicExpr) and a.op == 'const' and \
               isinstance(b, SymbolicExpr) and b.op == 'const':
                try:
                    val = SymbolicExpr(self.op, a, b).evaluate({})
                    return SymbolicExpr.const(val)
                except (ValueError, ZeroDivisionError):
                    pass

        if self.op == 'add':
            a, b = simplified_args
            if _is_zero(a): return b
            if _is_zero(b): return a

        elif self.op == 'sub':
            a, b = simplified_args
            if _is_zero(b): return a
            if _is_zero(a): return -b

        elif self.op == 'mul':
            a, b = simplified_args
            if _is_zero(a) or _is_zero(b): return SymbolicExpr.const(0)
            if _is_one(a): return b
            if _is_one(b): return a

        elif self.op == 'div':
            a, b = simplified_args
            if _is_zero(a): return SymbolicExpr.const(0)
            if _is_one(b): return a

        elif self.op == 'pow':
            a, b = simplified_args
            if _is_zero(b): return SymbolicExpr.const(1)
            if _is_one(b): return a

        elif self.op == 'neg':
            a = simplified_args[0]
            if isinstance(a, SymbolicExpr) and a.op == 'const':
                return SymbolicExpr.const(-a.args[0])
            if isinstance(a, SymbolicExpr) and a.op == 'neg':
                return a.args[0]

        return SymbolicExpr(self.op, *simplified_args)

    def evaluate(self, env: dict[str, float]) -> float:
        """
        Evaluate expression given variable bindings.

        Args:
            env: Mapping from variable names to values.
        """
        if self.op == 'const':
            return float(self.args[0])
        if self.op == 'var':
            name = self.args[0]
            if name not in env:
                raise ValueError(f"Unbound variable: {name}")
            return float(env[name])

        if self.op == 'neg':
            return -self.args[0].evaluate(env)
        if self.op == 'sin':
            return np.sin(self.args[0].evaluate(env))
        if self.op == 'cos':
            return np.cos(self.args[0].evaluate(env))
        if self.op == 'exp':
            return np.exp(self.args[0].evaluate(env))
        if self.op == 'log':
            return np.log(self.args[0].evaluate(env))
        if self.op == 'sqrt':
            return np.sqrt(self.args[0].evaluate(env))

        a = self.args[0].evaluate(env)
        b = self.args[1].evaluate(env)

        if self.op == 'add': return a + b
        if self.op == 'sub': return a - b
        if self.op == 'mul': return a * b
        if self.op == 'div': return a / b
        if self.op == 'pow': return a ** b

        raise ValueError(f"Unknown op: {self.op}")

    def variables(self) -> set[str]:
        """Return set of all variable names in the expression."""
        if self.op == 'var':
            return {self.args[0]}
        if self.op == 'const':
            return set()
        result = set()
        for a in self.args:
            if isinstance(a, SymbolicExpr):
                result |= a.variables()
        return result

    def __repr__(self):
        if self.op == 'const':
            return f"{self.args[0]:.4g}" if isinstance(self.args[0], float) else str(self.args[0])
        if self.op == 'var':
            return str(self.args[0])
        if self.op == 'neg':
            return f"(-{self.args[0]})"
        if self.op in ('sin', 'cos', 'exp', 'log', 'sqrt'):
            return f"{self.op}({self.args[0]})"
        symbols = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '^'}
        return f"({self.args[0]} {symbols.get(self.op, self.op)} {self.args[1]})"


def _is_zero(expr) -> bool:
    return isinstance(expr, SymbolicExpr) and expr.op == 'const' and abs(expr.args[0]) < 1e-14

def _is_one(expr) -> bool:
    return isinstance(expr, SymbolicExpr) and expr.op == 'const' and abs(expr.args[0] - 1.0) < 1e-14


# ============================================================================
# REWRITE SYSTEM
# ============================================================================

@dataclass
class RewriteRule:
    """A rewrite rule: pattern → replacement."""
    pattern: SymbolicExpr
    replacement: SymbolicExpr
    name: str = ""


class RewriteSystem:
    """
    Term rewriting system.

    Rules are applied to subexpressions until a normal form is reached
    (no more rules apply) or a step limit is hit.

    A system is confluent if the normal form is unique regardless of
    rule application order.
    """

    def __init__(self, rules: list[RewriteRule]):
        self.rules = rules
        logger.info(f"RewriteSystem with {len(rules)} rules")

    def apply(self, expr: SymbolicExpr) -> SymbolicExpr:
        """Apply one rewrite step (first matching rule)."""
        for rule in self.rules:
            match = PatternMatcher.match(rule.pattern, expr)
            if match is not None:
                return self._substitute(rule.replacement, match)

        # Try subexpressions
        if expr.op in ('const', 'var'):
            return expr

        new_args = []
        for a in expr.args:
            if isinstance(a, SymbolicExpr):
                new_args.append(self.apply(a))
            else:
                new_args.append(a)

        return SymbolicExpr(expr.op, *new_args)

    def normalize(self, expr: SymbolicExpr, max_steps: int = 100) -> SymbolicExpr:
        """Apply rules until normal form (no more rules apply)."""
        current = expr
        for step in range(max_steps):
            next_expr = self.apply(current)
            if self._equal(next_expr, current):
                logger.info(f"Normal form reached in {step + 1} steps")
                return current
            current = next_expr

        logger.warning(f"Did not converge in {max_steps} steps")
        return current

    def is_confluent(self, test_exprs: list[SymbolicExpr] = None) -> bool:
        """
        Test confluence: try applying rules in different orders
        and check if the same normal form is reached.

        This is undecidable in general — we test on given examples.
        """
        if test_exprs is None:
            return True  # No test cases, assume confluent

        for expr in test_exprs:
            # Apply rules in forward order
            nf1 = self.normalize(expr)

            # Apply rules in reverse order
            reversed_system = RewriteSystem(list(reversed(self.rules)))
            nf2 = reversed_system.normalize(expr)

            if not self._equal(nf1, nf2):
                return False

        return True

    def _substitute(self, template: SymbolicExpr,
                    bindings: dict[str, SymbolicExpr]) -> SymbolicExpr:
        """Substitute variables in template with matched expressions."""
        if template.op == 'var' and template.args[0] in bindings:
            return deepcopy(bindings[template.args[0]])
        if template.op == 'const':
            return deepcopy(template)

        new_args = []
        for a in template.args:
            if isinstance(a, SymbolicExpr):
                new_args.append(self._substitute(a, bindings))
            else:
                new_args.append(a)

        return SymbolicExpr(template.op, *new_args)

    def _equal(self, a: SymbolicExpr, b: SymbolicExpr) -> bool:
        """Structural equality of expressions."""
        if a.op != b.op:
            return False
        if len(a.args) != len(b.args):
            return False
        for x, y in zip(a.args, b.args):
            if isinstance(x, SymbolicExpr) and isinstance(y, SymbolicExpr):
                if not self._equal(x, y):
                    return False
            elif x != y:
                return False
        return True


# ============================================================================
# PATTERN MATCHER
# ============================================================================

class PatternMatcher:
    """
    Pattern matching and unification for symbolic expressions.

    Pattern variables (ordinary 'var' nodes) match any subexpression.
    Constants and operators must match exactly.
    """

    @staticmethod
    def match(pattern: SymbolicExpr,
              expr: SymbolicExpr) -> dict[str, SymbolicExpr] | None:
        """
        Match a pattern against an expression.

        Pattern variables bind to subexpressions. A variable appearing
        multiple times must bind to structurally equal subexpressions.

        Returns:
            Bindings dict if match succeeds, None otherwise.
        """
        bindings = {}
        if PatternMatcher._match_rec(pattern, expr, bindings):
            return bindings
        return None

    @staticmethod
    def _match_rec(pattern: SymbolicExpr, expr: SymbolicExpr,
                   bindings: dict[str, SymbolicExpr]) -> bool:
        # Pattern variable matches anything
        if pattern.op == 'var':
            name = pattern.args[0]
            if name in bindings:
                # Must match previous binding
                return PatternMatcher._structurally_equal(bindings[name], expr)
            bindings[name] = expr
            return True

        # Constant must match exactly
        if pattern.op == 'const':
            return expr.op == 'const' and abs(pattern.args[0] - expr.args[0]) < 1e-14

        # Operator must match
        if pattern.op != expr.op:
            return False

        if len(pattern.args) != len(expr.args):
            return False

        for p_arg, e_arg in zip(pattern.args, expr.args):
            if isinstance(p_arg, SymbolicExpr) and isinstance(e_arg, SymbolicExpr):
                if not PatternMatcher._match_rec(p_arg, e_arg, bindings):
                    return False
            elif p_arg != e_arg:
                return False

        return True

    @staticmethod
    def unify(expr1: SymbolicExpr,
              expr2: SymbolicExpr) -> dict[str, SymbolicExpr] | None:
        """
        Unification: find substitution σ such that σ(expr1) = σ(expr2).

        More general than matching — both expressions can contain variables.
        """
        bindings = {}
        if PatternMatcher._unify_rec(expr1, expr2, bindings):
            return bindings
        return None

    @staticmethod
    def _unify_rec(e1: SymbolicExpr, e2: SymbolicExpr,
                   bindings: dict[str, SymbolicExpr]) -> bool:
        # Apply existing bindings
        e1 = PatternMatcher._apply_bindings(e1, bindings)
        e2 = PatternMatcher._apply_bindings(e2, bindings)

        if e1.op == 'var':
            name = e1.args[0]
            if name in bindings:
                return PatternMatcher._structurally_equal(bindings[name], e2)
            # Occurs check
            if e2.op == 'var' and e2.args[0] == name:
                return True
            if PatternMatcher._occurs(name, e2):
                return False
            bindings[name] = e2
            return True

        if e2.op == 'var':
            return PatternMatcher._unify_rec(e2, e1, bindings)

        if e1.op != e2.op:
            return False

        if e1.op == 'const':
            return abs(e1.args[0] - e2.args[0]) < 1e-14

        if len(e1.args) != len(e2.args):
            return False

        for a1, a2 in zip(e1.args, e2.args):
            if isinstance(a1, SymbolicExpr) and isinstance(a2, SymbolicExpr):
                if not PatternMatcher._unify_rec(a1, a2, bindings):
                    return False

        return True

    @staticmethod
    def _occurs(var_name: str, expr: SymbolicExpr) -> bool:
        """Occurs check: does variable appear in expression?"""
        if expr.op == 'var':
            return expr.args[0] == var_name
        for a in expr.args:
            if isinstance(a, SymbolicExpr) and PatternMatcher._occurs(var_name, a):
                return True
        return False

    @staticmethod
    def _apply_bindings(expr: SymbolicExpr,
                        bindings: dict[str, SymbolicExpr]) -> SymbolicExpr:
        """Apply bindings to expression."""
        if expr.op == 'var' and expr.args[0] in bindings:
            return bindings[expr.args[0]]
        return expr

    @staticmethod
    def _structurally_equal(a: SymbolicExpr, b: SymbolicExpr) -> bool:
        """Check structural equality."""
        if a.op != b.op:
            return False
        if len(a.args) != len(b.args):
            return False
        for x, y in zip(a.args, b.args):
            if isinstance(x, SymbolicExpr) and isinstance(y, SymbolicExpr):
                if not PatternMatcher._structurally_equal(x, y):
                    return False
            elif x != y:
                return False
        return True
