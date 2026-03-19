"""
Automatic Differentiation for HoloLoom Warp Drive
====================================================

Forward mode (dual numbers), reverse mode (backpropagation),
Jacobian-vector products (JVP), and vector-Jacobian products (VJP).

AD computes exact derivatives (up to floating point) of programs,
not symbolic or numerical approximations.

Forward mode: efficient for f: ℝ → ℝⁿ (few inputs, many outputs)
Reverse mode: efficient for f: ℝⁿ → ℝ (many inputs, few outputs)
JVP = forward mode primitive, VJP = reverse mode primitive.

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DUAL NUMBER (forward mode primitive)
# ============================================================================

class DualNumber:
    """
    Dual number: a + bε where ε² = 0.

    The dual part automatically carries derivative information:
    f(a + bε) = f(a) + f'(a)·b·ε

    This is the algebraic basis for forward-mode AD.
    """

    def __init__(self, value: float, derivative: float = 0.0):
        self.value = float(value)
        self.derivative = float(derivative)

    def __repr__(self):
        return f"Dual({self.value:.6g}, {self.derivative:.6g})"

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value + other.value, self.derivative + other.derivative)
        return DualNumber(self.value + float(other), self.derivative)

    def __radd__(self, other):
        return DualNumber(float(other) + self.value, self.derivative)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value - other.value, self.derivative - other.derivative)
        return DualNumber(self.value - float(other), self.derivative)

    def __rsub__(self, other):
        return DualNumber(float(other) - self.value, -self.derivative)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # (a + bε)(c + dε) = ac + (ad + bc)ε
            return DualNumber(
                self.value * other.value,
                self.value * other.derivative + self.derivative * other.value
            )
        return DualNumber(self.value * float(other), self.derivative * float(other))

    def __rmul__(self, other):
        return DualNumber(float(other) * self.value, float(other) * self.derivative)

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            # (a + bε)/(c + dε) = a/c + (bc - ad)/c² ε
            return DualNumber(
                self.value / other.value,
                (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2)
            )
        return DualNumber(self.value / float(other), self.derivative / float(other))

    def __rtruediv__(self, other):
        # other / self
        return DualNumber(
            float(other) / self.value,
            -float(other) * self.derivative / (self.value ** 2)
        )

    def __pow__(self, other):
        if isinstance(other, DualNumber):
            # (a + bε)^(c + dε) = a^c + a^c(d·ln(a) + c·b/a)ε
            val = self.value ** other.value
            if self.value > 0:
                deriv = val * (other.derivative * np.log(self.value) +
                               other.value * self.derivative / self.value)
            else:
                deriv = other.value * self.value ** (other.value - 1) * self.derivative
            return DualNumber(val, deriv)
        n = float(other)
        return DualNumber(
            self.value ** n,
            n * self.value ** (n - 1) * self.derivative
        )

    def __rpow__(self, other):
        # other^self = e^{self * ln(other)}
        base = float(other)
        val = base ** self.value
        deriv = val * np.log(base) * self.derivative
        return DualNumber(val, deriv)

    def __neg__(self):
        return DualNumber(-self.value, -self.derivative)

    def __abs__(self):
        if self.value >= 0:
            return DualNumber(self.value, self.derivative)
        return DualNumber(-self.value, -self.derivative)

    # --- Comparison (by value only) ---

    def __lt__(self, other):
        other_val = other.value if isinstance(other, DualNumber) else float(other)
        return self.value < other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, DualNumber) else float(other)
        return self.value > other_val

    # --- Math functions ---

    def sin(self):
        return DualNumber(np.sin(self.value), np.cos(self.value) * self.derivative)

    def cos(self):
        return DualNumber(np.cos(self.value), -np.sin(self.value) * self.derivative)

    def exp(self):
        e = np.exp(self.value)
        return DualNumber(e, e * self.derivative)

    def log(self):
        return DualNumber(np.log(self.value), self.derivative / self.value)

    def sqrt(self):
        s = np.sqrt(self.value)
        return DualNumber(s, self.derivative / (2.0 * s))

    def tanh(self):
        t = np.tanh(self.value)
        return DualNumber(t, (1 - t ** 2) * self.derivative)


# Module-level dual math functions for use in user code
def dual_sin(x): return x.sin() if isinstance(x, DualNumber) else DualNumber(np.sin(x))
def dual_cos(x): return x.cos() if isinstance(x, DualNumber) else DualNumber(np.cos(x))
def dual_exp(x): return x.exp() if isinstance(x, DualNumber) else DualNumber(np.exp(x))
def dual_log(x): return x.log() if isinstance(x, DualNumber) else DualNumber(np.log(x))
def dual_sqrt(x): return x.sqrt() if isinstance(x, DualNumber) else DualNumber(np.sqrt(x))


# ============================================================================
# FORWARD MODE AD
# ============================================================================

class ForwardModeAD:
    """
    Forward-mode automatic differentiation via dual numbers.

    Computes df/dx by evaluating f(x + ε) where ε² = 0.
    The dual part of the result IS the derivative.

    Efficient for f: ℝ → ℝⁿ (one input, many outputs).
    For f: ℝⁿ → ℝ, need n forward passes (one per input).
    """

    @staticmethod
    def differentiate(f: Callable, x: float) -> float:
        """
        Compute f'(x) using forward-mode AD.

        Args:
            f: Function that accepts DualNumber arguments.
            x: Point at which to differentiate.

        Returns:
            f'(x).
        """
        result = f(DualNumber(x, 1.0))
        if isinstance(result, DualNumber):
            return result.derivative
        return 0.0

    @staticmethod
    def gradient(f: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇f(x) for f: ℝⁿ → ℝ.

        Requires n forward passes (one per input dimension).
        """
        n = len(x)
        grad = np.zeros(n)

        for i in range(n):
            # Seed the i-th component
            dual_x = [DualNumber(x[j], 1.0 if j == i else 0.0) for j in range(n)]
            result = f(*dual_x)
            if isinstance(result, DualNumber):
                grad[i] = result.derivative

        return grad

    @staticmethod
    def jacobian(f: Callable, x: np.ndarray, m: int) -> np.ndarray:
        """
        Compute Jacobian J[i,j] = df_i/dx_j for f: ℝⁿ → ℝᵐ.

        Args:
            f: Function returning array of DualNumbers or np.ndarray.
            x: Input point.
            m: Output dimension.
        """
        n = len(x)
        J = np.zeros((m, n))

        for j in range(n):
            dual_x = [DualNumber(x[k], 1.0 if k == j else 0.0) for k in range(n)]
            result = f(*dual_x)

            if isinstance(result, (list, tuple)):
                for i in range(m):
                    if isinstance(result[i], DualNumber):
                        J[i, j] = result[i].derivative
            elif isinstance(result, DualNumber):
                J[0, j] = result.derivative

        return J


# ============================================================================
# REVERSE MODE AD (computation graph)
# ============================================================================

class _Node:
    """Node in the computation graph for reverse-mode AD."""

    def __init__(self, value: float, children=None, grad_fns=None, name=""):
        self.value = float(value)
        self.children = children or []    # (child_node, local_gradient_fn)
        self.grad_fns = grad_fns or []
        self.grad = 0.0                   # Accumulated gradient
        self.name = name

    def __repr__(self):
        return f"Node({self.value:.4g}, grad={self.grad:.4g})"

    def backward(self, grad: float = 1.0):
        """Backpropagate gradient through the computation graph."""
        self.grad += grad
        for child, grad_fn in zip(self.children, self.grad_fns):
            child.backward(grad_fn(grad))

    # --- Arithmetic ---

    def __add__(self, other):
        other = other if isinstance(other, _Node) else _Node(float(other))
        result = _Node(self.value + other.value,
                       [self, other],
                       [lambda g: g, lambda g: g])
        return result

    def __radd__(self, other):
        return _Node(float(other)).__add__(self)

    def __mul__(self, other):
        other = other if isinstance(other, _Node) else _Node(float(other))
        s_val, o_val = self.value, other.value
        result = _Node(s_val * o_val,
                       [self, other],
                       [lambda g, ov=o_val: g * ov,
                        lambda g, sv=s_val: g * sv])
        return result

    def __rmul__(self, other):
        return _Node(float(other)).__mul__(self)

    def __sub__(self, other):
        other = other if isinstance(other, _Node) else _Node(float(other))
        result = _Node(self.value - other.value,
                       [self, other],
                       [lambda g: g, lambda g: -g])
        return result

    def __rsub__(self, other):
        return _Node(float(other)).__sub__(self)

    def __truediv__(self, other):
        other = other if isinstance(other, _Node) else _Node(float(other))
        s_val, o_val = self.value, other.value
        result = _Node(s_val / o_val,
                       [self, other],
                       [lambda g, ov=o_val: g / ov,
                        lambda g, sv=s_val, ov=o_val: -g * sv / (ov ** 2)])
        return result

    def __pow__(self, n):
        n = float(n)
        val = self.value ** n
        s_val = self.value
        result = _Node(val,
                       [self],
                       [lambda g, sv=s_val, nn=n: g * nn * sv ** (nn - 1)])
        return result

    def __neg__(self):
        return _Node(-self.value, [self], [lambda g: -g])

    def sin(self):
        sv = self.value
        return _Node(np.sin(sv), [self], [lambda g, s=sv: g * np.cos(s)])

    def cos(self):
        sv = self.value
        return _Node(np.cos(sv), [self], [lambda g, s=sv: -g * np.sin(s)])

    def exp(self):
        ev = np.exp(self.value)
        return _Node(ev, [self], [lambda g, e=ev: g * e])

    def log(self):
        sv = self.value
        return _Node(np.log(sv), [self], [lambda g, s=sv: g / s])


class ReverseModeAD:
    """
    Reverse-mode automatic differentiation (backpropagation).

    Builds a computation graph during forward pass, then traverses
    it backward to accumulate gradients.

    Efficient for f: ℝⁿ → ℝ (many inputs, one output).
    Single backward pass computes all partial derivatives.
    """

    @staticmethod
    def differentiate(f: Callable, x: float) -> float:
        """
        Compute f'(x) using reverse-mode AD.

        Args:
            f: Function that accepts _Node arguments.
            x: Point at which to differentiate.
        """
        node_x = _Node(x, name='x')
        result = f(node_x)
        if isinstance(result, _Node):
            result.backward(1.0)
            return node_x.grad
        return 0.0

    @staticmethod
    def gradient(f: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇f(x) in a single backward pass.

        Args:
            f: Function f(*nodes) → node.
            x: Input point.
        """
        nodes = [_Node(float(xi), name=f'x{i}') for i, xi in enumerate(x)]
        result = f(*nodes)

        if isinstance(result, _Node):
            result.backward(1.0)
            return np.array([n.grad for n in nodes])

        return np.zeros(len(x))

    @staticmethod
    def hessian_diagonal(f: Callable, x: np.ndarray) -> np.ndarray:
        """
        Approximate diagonal of the Hessian via double reverse-mode.

        H[i,i] ≈ (f(x+h·eᵢ) - 2f(x) + f(x-h·eᵢ)) / h²
        (finite difference on the gradient for simplicity).
        """
        h = 1e-5
        n = len(x)
        hess_diag = np.zeros(n)

        grad_center = ReverseModeAD.gradient(f, x)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            grad_plus = ReverseModeAD.gradient(f, x_plus)
            hess_diag[i] = (grad_plus[i] - grad_center[i]) / h

        return hess_diag


# ============================================================================
# JVP (Jacobian-Vector Product)
# ============================================================================

class JVP:
    """
    Jacobian-vector product (JVP): compute J·v without forming J.

    JVP is the natural output of forward-mode AD:
    f(x + v·ε) = f(x) + (J·v)·ε

    This gives one column (or linear combination of columns) of J.
    """

    @staticmethod
    def compute(f: Callable, x: np.ndarray,
                v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute (f(x), J(x)·v) in a single forward pass.

        Args:
            f: Function f(*dual_numbers) → dual_number or list.
            x: Primal point.
            v: Tangent vector.

        Returns:
            (f_x, Jv): Function value and JVP.
        """
        n = len(x)
        dual_x = [DualNumber(x[i], v[i]) for i in range(n)]
        result = f(*dual_x)

        if isinstance(result, DualNumber):
            return np.array([result.value]), np.array([result.derivative])

        if isinstance(result, (list, tuple)):
            f_x = np.array([r.value if isinstance(r, DualNumber) else float(r)
                            for r in result])
            Jv = np.array([r.derivative if isinstance(r, DualNumber) else 0.0
                           for r in result])
            return f_x, Jv

        return np.array([float(result)]), np.array([0.0])


# ============================================================================
# VJP (Vector-Jacobian Product)
# ============================================================================

class VJP:
    """
    Vector-Jacobian product (VJP): compute vᵀ·J without forming J.

    VJP is the natural output of reverse-mode AD.
    The gradient ∇f = 1ᵀ·J is a special case.

    This gives one row (or linear combination of rows) of J.
    """

    @staticmethod
    def compute(f: Callable, x: np.ndarray,
                v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute (f(x), vᵀ·J(x)) in a single backward pass.

        Args:
            f: Function f(*nodes) → node or list of nodes.
            x: Primal point.
            v: Cotangent vector (adjoint seed).

        Returns:
            (f_x, vJ): Function value and VJP.
        """
        n = len(x)
        nodes = [_Node(float(x[i]), name=f'x{i}') for i in range(n)]
        result = f(*nodes)

        if isinstance(result, _Node):
            f_x = np.array([result.value])
            result.backward(v[0] if len(v) > 0 else 1.0)
        elif isinstance(result, (list, tuple)):
            f_x = np.array([r.value if isinstance(r, _Node) else float(r)
                            for r in result])
            for i, r in enumerate(result):
                if isinstance(r, _Node):
                    r.backward(v[i] if i < len(v) else 0.0)
        else:
            return np.array([float(result)]), np.zeros(n)

        vJ = np.array([node.grad for node in nodes])
        return f_x, vJ
