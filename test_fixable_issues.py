"""
Simple test module with fixable issues.
"""

def calculate_area(radius):
    """Calculate circle area."""
    return 3.14159 * radius * radius  # Magic number - should use constant

def calculate_timeout():
    """Get timeout value."""
    return 30  # Magic number - should use constant

def process_batch(items):
    """Process items in batches."""
    batch_size = 100  # Magic number
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        print(f"Processing batch: {batch}")

# Debug print
print("Module loaded")
