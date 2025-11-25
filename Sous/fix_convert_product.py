#!/usr/bin/env python3
"""Fix convert_product_data function"""

# Read the demo file
with open('demos/demo_farm_networks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the old function
old_function = '''def convert_product_data(product_dict):
    """Convert product JSON to Product object"""
    return create_product(
        product_id=product_dict["product_id"],
        farm_id=product_dict["farm_id"],
        name=product_dict["name"],
        category=ProductCategory(product_dict["category"]),
        price=Decimal(product_dict["price"]),
        unit=product_dict["unit"],
        quantity_available=Decimal(product_dict["quantity_available"]),
        organic=product_dict["organic"]
    )'''

# Define the new function
new_function = '''def convert_product_data(product_dict):
    """Convert product JSON to Product object"""
    return create_product(
        farm_id=product_dict["farm_id"],
        name=product_dict["name"],
        description=product_dict["description"],
        category=ProductCategory(product_dict["category"]),
        price=Decimal(product_dict["price"]),
        unit=product_dict["unit"],
        quantity_available=Decimal(product_dict["quantity_available"]),
        organic=product_dict["organic"]
    )'''

# Replace the function
content = content.replace(old_function, new_function)

# Write back
with open('demos/demo_farm_networks.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed convert_product_data function!")
