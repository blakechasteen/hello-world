"""
Another module for testing cross-file relationships.

This module imports from sample.py to test:
- Go-to-definition across files
- Completion based on imported symbols
- Hover information for imported types
"""

from sample import Calculator, BanditArm, helper_function, process_data
from typing import List


# NOTE: Testing imports from sample module
class AdvancedCalculator(Calculator):
    """An extended calculator that inherits from the base Calculator.

    Adds support for more complex operations like power and square root.
    """

    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to exponent.

        Args:
            base: The base number
            exponent: The exponent

        Returns:
            base ** exponent
        """
        result = base ** exponent
        self.history.append(("power", result))
        return result

    def square_root(self, value: float) -> float:
        """Calculate square root of value.

        Args:
            value: The input value (must be non-negative)

        Returns:
            Square root of value

        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Cannot take square root of negative number")
        result = value ** 0.5
        self.history.append(("sqrt", result))
        return result


class BanditExperiment:
    """Run multi-armed bandit experiments using Thompson Sampling.

    Uses multiple BanditArm instances to explore/exploit trade-off.
    """

    def __init__(self, num_arms: int = 3) -> None:
        """Initialize bandit experiment.

        Args:
            num_arms: Number of arms in the bandit problem
        """
        self.arms = [BanditArm() for _ in range(num_arms)]
        self.num_trials = 0

    def select_arm(self) -> int:
        """Select arm using Thompson Sampling.

        Returns:
            Index of selected arm
        """
        samples = [arm.sample_thompson() for arm in self.arms]
        return int(__import__("numpy").argmax(samples))

    def run_trial(self, arm_idx: int, reward: float) -> None:
        """Run one trial and update the selected arm.

        Args:
            arm_idx: Index of arm to play
            reward: Reward received (0.0 to 1.0)
        """
        self.arms[arm_idx].update(reward)
        self.num_trials += 1


def create_calculators(count: int) -> List[AdvancedCalculator]:
    """Create multiple calculator instances.

    Args:
        count: Number of calculators to create

    Returns:
        List of AdvancedCalculator instances
    """
    return [AdvancedCalculator() for _ in range(count)]


def test_calculator_integration() -> None:
    """Test integration between modules.

    Uses helper_function from sample module and process_data.
    """
    # Use imported function
    result = helper_function()
    print(f"Helper result: {result}")

    # Use imported class
    calc = Calculator()
    calc.add(10, 20)
    print(f"Calculator history: {calc.get_history()}")

    # Use local extended class
    adv_calc = AdvancedCalculator()
    adv_calc.power(2, 8)
    adv_calc.square_root(16)
    print(f"Advanced calculator: {adv_calc.get_history()}")

    # Test process_data import
    data = [{"value": 10}, {"value": 20}, {"value": 30}]
    stats = process_data(data)
    print(f"Data stats: {stats}")


if __name__ == "__main__":
    test_calculator_integration()
