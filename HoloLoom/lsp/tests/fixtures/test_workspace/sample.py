"""
Sample module for LSP integration testing.

This module contains representative Python code structures for testing:
- Class definitions with docstrings
- Method definitions with type hints
- Function definitions
- Import statements
- Comments (NOTE, TODO, FIXME)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import json


# NOTE: This is a test note comment
class Calculator:
    """A simple calculator for demonstrating LSP completion and hover.

    Supports basic arithmetic operations on integers and floating-point numbers.
    Uses a history list to track previous calculations.
    """

    def __init__(self, initial_value: float = 0.0) -> None:
        """Initialize calculator with optional initial value.

        Args:
            initial_value: Starting value for calculations (default: 0.0)
        """
        self.value = initial_value
        self.history: List[Tuple[str, float]] = []

    def add(self, a: int, b: int) -> int:
        """Add two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        result = a + b
        self.history.append(("add", result))
        return result

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Product of a and b
        """
        result = a * b
        self.history.append(("multiply", result))
        return result

    # TODO: Implement division with error handling
    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide two numbers (a / b).

        Args:
            a: Numerator
            b: Denominator

        Returns:
            Result of a / b, or None if b is zero

        Raises:
            ZeroDivisionError: If denominator is zero
        """
        if b == 0:
            return None
        result = a / b
        self.history.append(("divide", result))
        return result

    def get_history(self) -> List[Tuple[str, float]]:
        """Get calculation history.

        Returns:
            List of (operation, result) tuples
        """
        return self.history.copy()

    # FIXME: Optimize for large datasets
    def clear_history(self) -> None:
        """Clear the calculation history."""
        self.history.clear()


class BanditArm:
    """Represents a single arm in a multi-armed bandit problem.

    Uses Thompson Sampling with Beta distribution for exploration/exploitation.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Initialize a bandit arm with Beta prior.

        Args:
            alpha: Alpha parameter of Beta distribution
            beta: Beta parameter of Beta distribution
        """
        self.alpha = alpha
        self.beta = beta
        self.successes = 0
        self.failures = 0

    def update(self, reward: float) -> None:
        """Update arm statistics based on reward.

        Args:
            reward: Reward signal (0 or 1 for binary, or continuous value)
        """
        if reward > 0.5:
            self.successes += 1
            self.alpha += 1
        else:
            self.failures += 1
            self.beta += 1

    def sample_thompson(self) -> float:
        """Sample from Thompson Sampling posterior.

        Returns:
            Sample from Beta(alpha, beta) distribution
        """
        return np.random.beta(self.alpha, self.beta)


def helper_function() -> int:
    """A helper function for testing symbol resolution.

    Creates a Calculator instance and performs operations.

    Returns:
        Result of calculation (should be 3)
    """
    calc = Calculator()
    result = calc.add(1, 2)
    return result


def process_data(data: List[Dict[str, float]]) -> Dict[str, float]:
    """Process a list of data dictionaries.

    Args:
        data: List of dictionaries with numeric values

    Returns:
        Dictionary with summary statistics (min, max, mean)
    """
    if not data:
        return {"min": 0, "max": 0, "mean": 0}

    values = [item.get("value", 0) for item in data]
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values))
    }


async def async_helper_function() -> str:
    """An async helper function for testing async symbol resolution.

    Returns:
        A greeting message
    """
    return "Hello from async function"


if __name__ == "__main__":
    # Example usage
    calc = Calculator(0)
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(calc.get_history())
