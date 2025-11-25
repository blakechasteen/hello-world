#!/usr/bin/env python3
"""Fix the Farm.__post_init__ method to remove super() call"""

# Read the farm.py file
with open('sous/models/farm.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace super().__post_init__() call
content = content.replace(
    '        super().__post_init__()\n        self.org_type = OrganizationType.FARM',
    '        # Set farm-specific organization type\n        self.org_type = OrganizationType.FARM'
)

# Write back
with open('sous/models/farm.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Farm.__post_init__ method - removed super() call!")
