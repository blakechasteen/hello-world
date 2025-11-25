#!/usr/bin/env python3
"""Fix order creation in demo to use cart system"""

# Read the demo file
with open('demos/demo_farm_networks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the order creation section
old_code = '''    # Get first 3 available products
    available_products = [p for p in marketplace.products.values() if p.is_in_stock()][:3]

    items = [
        OrderItem(
            product_id=p.product_id,
            product_name=p.name,
            quantity=Decimal("2"),
            unit_price=p.price
        )
        for p in available_products
    ]

    order = marketplace.create_order(
        customer_id=customer_id,
        farm_id=available_products[0].farm_id,
        items=items,
        fulfillment_method=FulfillmentMethod.PICKUP
    )'''

new_code = '''    # Get first 3 available products from same farm
    all_products = list(marketplace.products.values())
    farm_id = "farm_001"  # Use Green Valley Urban Farm
    available_products = [p for p in all_products if p.farm_id == farm_id and p.is_in_stock()][:3]

    # Add products to cart
    for p in available_products:
        marketplace.add_to_cart(
            customer_id=customer_id,
            farm_id=farm_id,
            product_id=p.product_id,
            quantity=Decimal("2")
        )

    # Create order from cart
    order = marketplace.create_order_from_cart(
        customer_id=customer_id,
        farm_id=farm_id,
        fulfillment_method=FulfillmentMethod.PICKUP
    )'''

# Replace
content = content.replace(old_code, new_code)

# Write back
with open('demos/demo_farm_networks.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed order creation to use cart system!")
