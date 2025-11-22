"""
Bad code with AI slop - contains 15+ common AI-generated code pitfalls.
This is a fixture for testing Trough's detection capabilities.

Created: 2025-11-22
Category: AI slop examples
Expected issues: 15+
"""

# ISSUE 1: Missing imports (hallucination)
# from non_existent_library import some_function


def calculate_average(numbers):
    # ISSUE 2: No error handling
    total = sum(numbers)
    average = total / len(numbers)  # ISSUE 3: No zero check (IndexError risk)
    return average


# ISSUE 4: Hardcoded secrets
API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz"
DATABASE_PASSWORD = "admin123"
AWS_SECRET = "AKIAIOSFODNN7EXAMPLE/wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


def fetch_data():
    # ISSUE 5: SQL injection vulnerability - string concatenation
    user_id = "1 OR 1=1"
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    # Execute query...
    return None


def process_user_input(data):
    # ISSUE 6: Unvalidated input
    # ISSUE 7: XSS vulnerability (dangerouslySetInnerHTML equivalent)
    result = f"<div>{data}</div>"  # No escaping!
    return result


def read_file(file_path):
    # ISSUE 8: Resource leak - file never closed
    f = open(file_path, 'r')
    content = f.read()
    return content
    # File handle not closed, resource leak!


def process_config():
    # ISSUE 9: Hardcoded values (magic numbers)
    timeout = 30
    max_retries = 5
    delay = 1000  # milliseconds

    # ISSUE 10: Type mismatch
    config = {"timeout": timeout, "max_retries": "5"}  # Should be int
    return config


def dead_code_example():
    # ISSUE 11: Unused variable
    unused_var = "never used"

    # ISSUE 12: Unused import (in file)
    # import json

    # ISSUE 13: Dead code - unreachable after return
    print("This code is important")
    return True
    print("This will never execute")  # Dead code


# ISSUE 14: Inconsistent naming
def process_data():
    pass


def processUserData():  # camelCase - inconsistent!
    pass


def process_account_info():  # snake_case - inconsistent!
    pass


# ISSUE 15: Missing docstring for public function
def complex_calculation(x, y, z):
    # No docstring!
    result = (x + y) * z
    return result


# ISSUE 16: Copy-paste error (slightly different variable names)
def validate_email(email):
    if "@" not in email:
        return False
    return True


def validate_phone(phone):  # Copy-pasted with minimal changes
    if "@" not in phone:  # BUG: Should check for digits, not "@"
        return False
    return True


# ISSUE 17: Incomplete implementation
def advanced_feature():
    # TODO: Implement error handling
    # TODO: Add logging
    pass  # ISSUE 18: Pass statement in production code


class DataProcessor:
    # ISSUE 19: Missing docstring for class

    def __init__(self):
        # ISSUE 20: Uninitialized attributes used elsewhere
        self.data = None  # TODO: Initialize properly
        # self.logger = None  # Uncommented when needed

    def process(self):
        # ISSUE 21: Accessing potentially None attribute
        return self.data.transform()  # AttributeError if data is None


# ISSUE 22: Off-by-one error
def get_range_items(items, start, end):
    """Get items in range (off-by-one error)."""
    # Should be range(start, end + 1)
    return items[start:end]  # Excludes last item!


# ISSUE 23: Race condition (simplified)
import threading

counter = 0

def increment_counter():
    global counter
    # ISSUE 24: No locking - race condition!
    temp = counter
    temp += 1
    counter = temp  # Another thread could modify counter between read and write


# ISSUE 25: Timezone issues - naive datetime
def get_current_time():
    from datetime import datetime
    # ISSUE 26: Naive datetime (no timezone info)
    return datetime.now()  # Should use datetime.utcnow() or timezone-aware


# ISSUE 27: Performance anti-pattern - N+1 queries (simulated)
def get_all_users_and_posts():
    users = ["user1", "user2", "user3"]

    results = []
    for user in users:
        # ISSUE 28: Query inside loop (N+1 pattern)
        posts = fetch_posts_for_user(user)  # Called 3 times!
        results.append((user, posts))

    return results


def fetch_posts_for_user(user):
    """Simulates database query."""
    return ["post1", "post2"]


# ISSUE 29: Unnecessary list comprehension in loop
def filter_data(items):
    results = []
    for item in items:
        # ISSUE 30: Inefficient - creates new list each iteration
        filtered = [x for x in items if x != item]
        results.append(filtered)

    return results


# ISSUE 31: Vague variable names
def foo(bar, baz):
    """
    No docstring explanation of parameters.
    Variables named 'foo', 'bar', 'baz' - meaningless!
    """
    x = bar + baz
    y = x * 2
    z = y - bar
    return z


# ISSUE 32: Inconsistent error handling
def operation_a():
    try:
        result = 10 / 0
    except:  # Bare except - catches everything!
        pass
    return result


def operation_b():
    result = 10 / 0  # No error handling at all!
    return result


# ISSUE 33: Meaningless metrics/measurements
def measure_performance():
    import time
    start = time.time()
    # Do something
    end = time.time()

    duration = end - start
    # ISSUE 34: No unit documentation
    return duration  # Seconds? Milliseconds? Unclear!


# ISSUE 35: Unclear ownership/responsibility
def send_email(recipient, subject, body):
    """Send email (but to where? No config!)."""
    # No email service configured - how does this work?
    # email_service.send(...)  # Undefined
    pass


# ISSUE 36: Redundant code
def validate_input(text):
    if text is None:
        return False
    if text == "":
        return False
    if len(text) == 0:  # Redundant with above
        return False
    return True


# ISSUE 37: Weasel words in comments
# Should probably handle this error somehow
# This might fail in edge cases
# TODO: Make this more robust (very vague)


# ISSUE 38: Mixed return types (type mismatch)
def get_value(use_string=False):
    if use_string:
        return "42"
    else:
        return 42  # Mixed int and string returns!


if __name__ == "__main__":
    # ISSUE 39: No error handling in main
    result = calculate_average([1, 2, 3])
    print(result)

    # ISSUE 40: Using undefined function
    # data = fetch_from_api()  # fetch_from_api doesn't exist
