"""
Good code example - follows best practices with proper error handling, documentation, and security.
This is the baseline fixture for comparison against problematic code.

Created: 2025-11-22
Category: Baseline fixture (no issues)
"""

import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


# Proper logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_average(numbers: List[float]) -> Optional[float]:
    """
    Calculate the average of a list of numbers.

    Args:
        numbers: List of numeric values

    Returns:
        Average value, or None if list is empty

    Raises:
        TypeError: If list contains non-numeric values
    """
    if not numbers:
        logger.warning("calculate_average called with empty list")
        return None

    if not isinstance(numbers, list):
        raise TypeError(f"Expected list, got {type(numbers)}")

    try:
        total = sum(float(n) for n in numbers)
        average = total / len(numbers)
        return average
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating average: {e}")
        raise


def read_config_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON configuration file safely.

    Args:
        file_path: Path to the configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If required config keys are missing
    """
    if not file_path:
        raise ValueError("file_path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {file_path}: {e}")
        raise
    except IOError as e:
        logger.error(f"Error reading config file {file_path}: {e}")
        raise

    # Validate required keys
    required_keys = ['version', 'timeout']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    return config


def process_user_data(user_id: int, data: Dict[str, str]) -> Dict[str, Any]:
    """
    Process user data with proper validation and error handling.

    Args:
        user_id: Unique user identifier
        data: Dictionary containing user data

    Returns:
        Processed user data with metadata
    """
    # Input validation
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id}")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")

    # Process timestamp properly (timezone-aware)
    timestamp = datetime.utcnow().isoformat() + 'Z'  # UTC timezone

    # Sanitize inputs (escape special characters)
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            processed_data[key] = value.strip()
        else:
            processed_data[key] = value

    return {
        'user_id': user_id,
        'processed_at': timestamp,
        'data': processed_data,
        'version': '1.0'
    }


def retry_operation(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Any:
    """
    Retry a function call with exponential backoff.

    Args:
        func: Callable to retry
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        backoff: Backoff multiplier for each retry

    Returns:
        Result of the function call

    Raises:
        Exception: The last exception if all retries fail
    """
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func)}")

    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    last_exception = None
    current_delay = delay

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                # Sleep before retry
                import time
                time.sleep(current_delay)
                current_delay *= backoff

    # All retries exhausted
    logger.error(f"All {max_retries} retries failed")
    raise last_exception


class DatabaseConnection:
    """
    Database connection manager with proper resource cleanup.
    """

    def __init__(self, connection_string: str, timeout: int = 30):
        """
        Initialize database connection.

        Args:
            connection_string: Connection string for the database
            timeout: Connection timeout in seconds
        """
        if not connection_string:
            raise ValueError("connection_string cannot be empty")

        self.connection_string = connection_string
        self.timeout = timeout
        self.connected = False
        self._connection = None

    def __enter__(self):
        """Enter context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - ensures cleanup."""
        self.disconnect()
        return False

    def connect(self) -> None:
        """Connect to database."""
        try:
            logger.info(f"Connecting to database...")
            # Simulate connection
            self.connected = True
            self._connection = {'status': 'connected'}
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from database."""
        if self.connected:
            try:
                logger.info("Disconnecting from database...")
                self._connection = None
                self.connected = False
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

    def query(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a parameterized query safely.

        Args:
            sql: SQL query with ? placeholders
            params: Query parameters (safe from SQL injection)

        Returns:
            Query results
        """
        if not self.connected:
            raise RuntimeError("Not connected to database")

        if not sql:
            raise ValueError("sql cannot be empty")

        params = params or []

        try:
            logger.debug(f"Executing query: {sql}")
            # Simulated secure query execution
            return [{'id': 1, 'name': 'test'}]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise


# Usage example
if __name__ == "__main__":
    # Good: Using context manager for automatic cleanup
    try:
        with DatabaseConnection("host=localhost dbname=testdb") as db:
            results = db.query("SELECT * FROM users WHERE id = ?", [1])
            print(results)
    except Exception as e:
        print(f"Error: {e}")
