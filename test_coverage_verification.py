"""
Simple test to verify pytest-cov is working correctly.
This test doesn't require external dependencies.
"""
import pytest


def hello_world():
    """A simple function to test coverage."""
    return "Hello, World!"


def calculate_sum(a, b):
    """Another function for coverage."""
    return a + b


def untested_function():
    """This function is not tested - should show in coverage report."""
    return "This is not covered"


class TestCoverage:
    """Test class to verify coverage reporting."""

    def test_hello_world(self):
        """Test hello_world function."""
        result = hello_world()
        assert result == "Hello, World!"

    def test_calculate_sum(self):
        """Test calculate_sum function."""
        assert calculate_sum(2, 3) == 5
        assert calculate_sum(-1, 1) == 0

    def test_calculate_sum_float(self):
        """Test calculate_sum with floats."""
        assert calculate_sum(1.5, 2.5) == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
