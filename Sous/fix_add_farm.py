#!/usr/bin/env python3
"""Fix add_farm to register_farm in demo"""

# Read the demo file
with open('demos/demo_farm_networks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace add_farm with register_farm
content = content.replace('marketplace.add_farm(farm)', 'marketplace.register_farm(farm)')
content = content.replace('csa_coordinator.add_farm(farm)', 'csa_coordinator.register_farm(farm)')

# Write back
with open('demos/demo_farm_networks.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed add_farm -> register_farm!")
