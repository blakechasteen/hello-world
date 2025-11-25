#!/usr/bin/env python3
"""Fix product categories in products.json"""

import json

# Read products.json
with open('data/products.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fix categories
for product in data['products']:
    if product['category'] == 'grains':
        product['category'] = 'other'

# Write back
with open('data/products.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Fixed product categories!")
