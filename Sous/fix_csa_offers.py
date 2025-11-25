#!/usr/bin/env python3
"""Fix offers_csa flag in demo"""

# Read the demo file
with open('demos/demo_farm_networks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the demo_csa_coordinator function and update it
old_code = '''    # Add farms that offer CSA
    for farm_dict in farms_data["farms"]:
        farm = convert_farm_data(farm_dict)
        csa_coordinator.register_farm(farm)'''

new_code = '''    # Add farms that offer CSA (farms 002, 003, 005 have CSA seasons)
    csa_farm_ids = {"farm_002", "farm_003", "farm_005"}
    for farm_dict in farms_data["farms"]:
        if farm_dict["farm_id"] in csa_farm_ids:
            farm = convert_farm_data(farm_dict)
            farm.offers_csa = True  # Enable CSA for these farms
            csa_coordinator.register_farm(farm)'''

# Replace
content = content.replace(old_code, new_code)

# Write back
with open('demos/demo_farm_networks.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed offers_csa for CSA farms!")
