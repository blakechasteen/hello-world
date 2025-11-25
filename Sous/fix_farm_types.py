#!/usr/bin/env python3
"""Fix farm_type values in farms.json"""

import json

# Read farms.json
with open('data/farms.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fix farm_type values
for farm in data['farms']:
    if farm['farm_type'] == 'regenerative':
        farm['farm_type'] = 'regenerative_farm'

# Write back
with open('data/farms.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Fixed farm_type values!")
