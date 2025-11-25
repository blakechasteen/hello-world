#!/usr/bin/env python3
"""Fix the convert_farm_data function in demo_farm_networks.py"""

import re

# Read the demo file
with open('demos/demo_farm_networks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the old function
old_function = '''def convert_farm_data(farm_dict):
    """Convert farm JSON to Farm object"""
    return Farm(
        farm_id=farm_dict["farm_id"],
        name=farm_dict["name"],
        farm_type=FarmType(farm_dict["farm_type"]),
        address=farm_dict["location"]["address"],
        city=farm_dict["location"]["city"],
        state=farm_dict["location"]["state"],
        zipcode=farm_dict["location"]["zipcode"],
        total_acres=Decimal(str(farm_dict["total_acres"])),
        organic_certified=farm_dict["organic_certified"],
        certifications=[
            CertificationType(cert) for cert in farm_dict["certifications"]
        ] if farm_dict["certifications"] else [],
        growing_zones=farm_dict["growing_zones"],
        year_established=farm_dict["year_established"],
        active=farm_dict["active"]
    )'''

# Define the new function
new_function = '''def convert_farm_data(farm_dict):
    """Convert farm JSON to Farm object"""
    # Create Location object
    location = Location(
        address=farm_dict["location"]["address"],
        city=farm_dict["location"]["city"],
        state=farm_dict["location"]["state"],
        zip_code=farm_dict["location"]["zipcode"],
        latitude=farm_dict["location"]["coordinates"].get("lat") if "coordinates" in farm_dict["location"] else None,
        longitude=farm_dict["location"]["coordinates"].get("lon") if "coordinates" in farm_dict["location"] else None
    )

    return Farm(
        # Organization base fields
        org_id=farm_dict["farm_id"],
        name=farm_dict["name"],
        org_type=OrganizationType.FARM,
        location=location,
        contact_name=f"{farm_dict['name']} Manager",
        contact_email=farm_dict["contact"]["email"],
        contact_phone=farm_dict["contact"]["phone"],

        # Farm-specific fields
        farm_type=FarmType(farm_dict["farm_type"]),
        total_acres=Decimal(str(farm_dict["total_acres"])),
        organic_certified=farm_dict["organic_certified"],
        certifications=[
            CertificationType(cert) for cert in farm_dict["certifications"]
        ] if farm_dict["certifications"] else [],
        growing_zones=farm_dict["growing_zones"],
        year_established=farm_dict["year_established"],
        active=farm_dict["active"]
    )'''

# Replace the function
content = content.replace(old_function, new_function)

# Write back
with open('demos/demo_farm_networks.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed convert_farm_data function!")
