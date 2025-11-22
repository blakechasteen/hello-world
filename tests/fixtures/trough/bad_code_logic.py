"""
Bad code with logic errors - focuses on ML logic bugs, off-by-one, null references, etc.

Created: 2025-11-22
Category: Logic and algorithmic errors
Expected issues: 12+
"""

from typing import List, Optional


def divide_numbers(a: int, b: int) -> float:
    """
    ISSUE 1: Division by zero not handled
    """
    return a / b  # Will crash if b == 0


def find_maximum(numbers: List[int]) -> int:
    """
    ISSUE 2: No handling for empty list
    """
    max_val = numbers[0]  # IndexError if list is empty
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val


def get_list_item(items: List, index: int) -> Optional[object]:
    """
    ISSUE 3: No bounds checking
    """
    return items[index]  # IndexError if index out of bounds


def process_config(config: Optional[dict]) -> str:
    """
    ISSUE 4: Null dereference (accessing None)
    """
    return config['key']  # AttributeError if config is None


def array_slice_operation(arr: List, start: int, end: int) -> List:
    """
    ISSUE 5: Off-by-one error in slicing
    """
    # Typically want to include the end index
    return arr[start:end]  # Excludes last element!


def string_indexing(text: str) -> str:
    """
    ISSUE 6: Off-by-one in string access
    """
    # Trying to get last character
    return text[len(text)]  # IndexError! Should be len(text) - 1


def loop_off_by_one(n: int) -> int:
    """
    ISSUE 7: Loop boundary error
    """
    sum_val = 0
    for i in range(n + 1):  # Includes n, should be range(n)
        sum_val += i
    return sum_val


def constant_condition(data: List) -> List:
    """
    ISSUE 8: Constant condition (always true/false)
    """
    result = []
    for item in data:
        if item > 0:  # Always true for positive numbers
            result.append(item)
        elif item > 0:  # This condition is impossible
            result.append(item * 2)
    return result


def unreachable_code(value: int) -> str:
    """
    ISSUE 9: Unreachable code after return
    """
    if value > 0:
        return "positive"
    return "not positive"
    # This code never executes:
    return "this is unreachable"


def logical_contradiction(x: int) -> str:
    """
    ISSUE 10: Contradictory conditions
    """
    if x > 10:
        return "greater than 10"
    elif x > 5 and x < 10:
        return "between 5 and 10"
    elif x > 10:  # This condition is impossible (already checked x > 10)
        return "this is unreachable"
    return "10 or less"


def missing_return(should_return: bool) -> Optional[str]:
    """
    ISSUE 11: Missing return statement in branch
    """
    if should_return:
        return "result"
    # Missing else branch - implicitly returns None


def nested_null_access(user: Optional[dict]) -> str:
    """
    ISSUE 12: Nested null dereference
    """
    # No check if user is None
    return user['profile']['name']  # Multiple errors possible


def float_comparison(val: float) -> bool:
    """
    ISSUE 13: Floating point comparison (inexact)
    """
    return val == 0.1 + 0.2  # Due to float precision, may fail


def array_bounds_in_loop(arr: List) -> None:
    """
    ISSUE 14: Array access out of bounds in loop
    """
    for i in range(len(arr) + 1):  # Accesses arr[len(arr)], out of bounds!
        print(arr[i])


def condition_always_false(x: int) -> str:
    """
    ISSUE 15: Condition that can never be true
    """
    if x > 100 and x < 0:  # Impossible condition
        return "impossible state"
    return "normal"


def while_infinite_loop() -> None:
    """
    ISSUE 16: Infinite loop
    """
    i = 0
    while i >= 0:  # i starts at 0, never decrements, always >= 0
        print(i)
        # No break or increment that makes i < 0


def invalid_type_operation(data: str) -> int:
    """
    ISSUE 17: Type mismatch in operation
    """
    # Adding string and int
    return data + 5  # TypeError!


def wrong_operator_usage(a: int, b: int) -> int:
    """
    ISSUE 18: Wrong operator (likely logic error)
    """
    # Probably meant to use * instead of &
    return a & b  # Bitwise AND instead of multiplication


def empty_try_except(data):
    """
    ISSUE 19: Empty except that silently fails
    """
    try:
        return data[0]
    except:
        pass  # Silent failure - bug is hidden


def wrong_variable_in_loop(items: List[int]) -> int:
    """
    ISSUE 20: Using wrong variable in loop
    """
    total = 0
    for i in range(len(items)):
        total += i  # Should be items[i], not i!
    return total


def list_mutation_during_iteration(items: List) -> List:
    """
    ISSUE 21: Modifying list during iteration
    """
    for item in items:
        if item % 2 == 0:
            items.remove(item)  # Dangerous - modifies while iterating!
    return items


def copy_paste_variable_error(user_name: str, user_email: str) -> dict:
    """
    ISSUE 22: Copy-paste with forgotten variable rename
    """
    user1 = {"name": user_name, "email": user_email}
    user2 = {"name": user_name, "email": user_name}  # Should be user_email!
    return {"primary": user1, "secondary": user2}


def incorrect_loop_count(n: int) -> int:
    """
    ISSUE 23: Loop executes wrong number of times
    """
    result = 0
    for i in range(n):
        result += 1
    return result  # Returns n, but maybe should return n-1?


def state_not_initialized(use_default: bool) -> dict:
    """
    ISSUE 24: Using variable before initialization
    """
    if use_default:
        state = {"value": 0}
    # Missing else - state is undefined if not use_default
    return state  # NameError if use_default is False
