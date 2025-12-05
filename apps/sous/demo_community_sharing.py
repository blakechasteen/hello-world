#!/usr/bin/env python3
"""
Community Food Sharing Demonstration

Demonstrates Phase 2 features:
- Multi-organization food sharing network
- Intelligent donor-recipient matching
- Community inventory management
- Network connections and trust
- AI-powered matching with HoloLoom
- Donation lifecycle from surplus to delivery

This showcases how grocery stores, restaurants, food banks, and churches
can coordinate to reduce food waste and feed communities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Sous imports
from sous.models.organization import (
    Organization, FoodBank, Restaurant, GroceryStore, Church,
    Location, OrganizationType
)
from sous.models.food_item import (
    FoodItem, DonationListing, SurplusAlert,
    FoodCategory, StorageRequirement, FoodSafetyRating
)
from sous.services.donation_coordinator import DonationCoordinator
from sous.services.community_inventory_manager import CommunityInventoryManager
from sous.services.community_network import (
    CommunityNetwork, ConnectionType, CommunicationChannel
)

# HoloLoom integration (optional)
try:
    from sous.services.hololoom_integration import create_hololoom_engine
    HOLOLOOM_AVAILABLE = True
except:
    HOLOLOOM_AVAILABLE = False
    print("Note: HoloLoom integration not available, running without AI features\n")


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")


async def load_sample_data() -> tuple:
    """Load sample organizations and inventory from JSON files"""
    print_section("Loading Sample Data")

    data_dir = Path(__file__).parent / "Sous" / "data"

    # Load organizations
    orgs_file = data_dir / "sample_organizations.json"
    print(f"Loading organizations from {orgs_file}")

    with open(orgs_file) as f:
        orgs_data = json.load(f)

    # Parse organizations (simplified - in production would use proper parsing)
    organizations = {}
    for org_dict in orgs_data["organizations"]:
        # Create location
        loc = Location(
            address=org_dict["location"]["address"],
            city=org_dict["location"]["city"],
            state=org_dict["location"]["state"],
            zip_code=org_dict["location"]["zip_code"],
            latitude=org_dict["location"]["latitude"],
            longitude=org_dict["location"]["longitude"]
        )

        # Create basic organization
        org = Organization(
            org_id=org_dict["org_id"],
            name=org_dict["name"],
            org_type=OrganizationType(org_dict["org_type"]),
            location=loc,
            contact_name=org_dict["contact"]["name"],
            contact_email=org_dict["contact"]["email"],
            contact_phone=org_dict["contact"]["phone"],
            can_donate=org_dict["capabilities"]["can_donate"],
            can_receive=org_dict["capabilities"]["can_receive"],
            has_refrigeration=org_dict["logistics"]["has_refrigeration"],
            has_freezer=org_dict["logistics"]["has_freezer"]
        )

        organizations[org_dict["org_id"]] = org

    print(f"✓ Loaded {len(organizations)} organizations")
    print(f"  - Food Banks: {sum(1 for o in organizations.values() if o.org_type == OrganizationType.FOOD_BANK)}")
    print(f"  - Restaurants: {sum(1 for o in organizations.values() if o.org_type == OrganizationType.RESTAURANT)}")
    print(f"  - Grocery Stores: {sum(1 for o in organizations.values() if o.org_type == OrganizationType.GROCERY_STORE)}")
    print(f"  - Churches: {sum(1 for o in organizations.values() if o.org_type == OrganizationType.CHURCH)}")

    # Load inventory
    inv_file = data_dir / "sample_inventory.json"
    print(f"\nLoading inventory from {inv_file}")

    with open(inv_file) as f:
        inv_data = json.load(f)

    # Parse inventory items (simplified)
    inventory = {}
    for org_id, items in inv_data["inventory"].items():
        inventory[org_id] = []
        for item_dict in items:
            # Parse dates
            expiration_date = None
            if item_dict.get("expiration_date"):
                expiration_date = datetime.fromisoformat(item_dict["expiration_date"])

            prepared_date = None
            if item_dict.get("prepared_date"):
                prepared_date = datetime.fromisoformat(item_dict["prepared_date"])

            use_by_date = None
            if item_dict.get("use_by_date"):
                use_by_date = datetime.fromisoformat(item_dict["use_by_date"])

            item = FoodItem(
                item_id=item_dict["item_id"],
                name=item_dict["name"],
                category=FoodCategory(item_dict["category"]),
                description=item_dict.get("description", ""),
                quantity=Decimal(str(item_dict["quantity"])),
                unit=item_dict["unit"],
                servings=item_dict.get("servings"),
                storage_requirement=StorageRequirement(item_dict["storage_requirement"]),
                expiration_date=expiration_date,
                prepared_date=prepared_date,
                use_by_date=use_by_date,
                estimated_value=Decimal(str(item_dict.get("estimated_value", 0))),
                is_vegetarian=item_dict.get("is_vegetarian", False),
                is_vegan=item_dict.get("is_vegan", False),
                allergens=item_dict.get("allergens", [])
            )

            inventory[org_id].append(item)

    total_items = sum(len(items) for items in inventory.values())
    print(f"✓ Loaded {total_items} food items across {len(inventory)} organizations")

    return organizations, inventory


async def demo_inventory_management(
    organizations: dict,
    inventory: dict
):
    """Demonstrate community inventory management"""
    print_section("Community Inventory Management")

    manager = CommunityInventoryManager()

    # Add inventory to manager
    print_subsection("Loading Inventory")
    for org_id, items in inventory.items():
        manager.bulk_add_items(org_id, items)
        print(f"✓ Loaded {len(items)} items for {organizations[org_id].name}")

    # Get inventory stats for a grocery store
    print_subsection("Fresh Market Grocery - Inventory Analysis")
    stats = manager.get_inventory_stats("store_fresh_market")
    print(f"Total items: {stats.total_items}")
    print(f"Total weight: {stats.total_weight_lbs} lbs")
    print(f"Total value: ${stats.total_value}")
    print(f"Items expiring today: {stats.items_expiring_today}")
    print(f"Items expiring this week: {stats.items_expiring_this_week}")

    # Identify surplus
    print_subsection("Identifying Surplus Food")
    surplus = manager.identify_surplus("store_fresh_market", days_threshold=2)
    print(f"Found {len(surplus)} surplus items approaching expiration:")
    for item in surplus[:3]:
        days_left = item.get_days_until_expiration()
        safety = item.get_safety_rating()
        print(f"  - {item.name}: {days_left} days left ({safety.value})")

    # Waste risk analysis
    print_subsection("Waste Risk Analysis")
    at_risk = manager.get_waste_risk_items("store_fresh_market", risk_threshold_days=2)
    print(f"Found {len(at_risk)} items at risk of waste:")
    for risk in at_risk[:3]:
        item = risk["item"]
        print(f"  - {item.name}")
        print(f"    Risk score: {risk['risk_score']:.2f}")
        print(f"    Recommendation: {risk['recommendation']}")

    # Category breakdown
    print_subsection("Category Breakdown (Fresh Market)")
    breakdown = manager.get_category_breakdown("store_fresh_market")
    for category, data in breakdown.items():
        print(f"  {category.value}: {data['count']} items, {data['weight_lbs']} lbs, ${data['value']}")

    return manager


async def demo_donation_coordination(
    organizations: dict,
    inventory: dict,
    manager: CommunityInventoryManager
):
    """Demonstrate donor-recipient matching"""
    print_section("Donation Coordination")

    coordinator = DonationCoordinator()

    # Register organizations
    print_subsection("Registering Organizations")
    for org in organizations.values():
        coordinator.register_organization(org)
    print(f"✓ Registered {len(organizations)} organizations")

    # Create donation listing from surplus
    print_subsection("Creating Donation Listing")
    surplus = manager.identify_surplus("store_fresh_market", days_threshold=2)

    listing = DonationListing(
        listing_id="demo_listing_001",
        donor_org_id="store_fresh_market",
        items=surplus[:3],  # First 3 surplus items
        available_from=datetime.now(),
        available_until=datetime.now() + timedelta(hours=4),
        pickup_address="567 Market Street, Springfield, IL",
        notes="Fresh produce and dairy approaching expiration"
    )

    listing_id = coordinator.create_listing(listing)
    print(f"✓ Created listing {listing_id}")
    print(f"  Items: {', '.join([item.name for item in listing.items])}")
    print(f"  Total weight: {listing.get_total_weight_lbs()} lbs")
    print(f"  Estimated value: ${listing.get_estimated_value()}")

    # Find matches
    print_subsection("Finding Best Recipients")
    matches = coordinator.find_matches_for_listing(listing, max_matches=5)

    print(f"Found {len(matches)} potential recipients:\n")
    for i, match in enumerate(matches, 1):
        recipient = coordinator.get_organization(match.recipient_org_id)
        print(f"{i}. {recipient.name} (score: {match.total_score:.3f})")
        print(f"   {match.reasoning}")
        print(f"   Subscores:")
        for key, value in match.subscores.items():
            print(f"     - {key}: {value:.2f}")
        print()

    # Claim listing
    print_subsection("Claiming Donation")
    top_match = matches[0]
    success = coordinator.claim_listing(listing_id, top_match.recipient_org_id)

    if success:
        recipient = coordinator.get_organization(top_match.recipient_org_id)
        print(f"✓ Listing claimed by {recipient.name}")
    else:
        print("✗ Failed to claim listing")

    return coordinator


async def demo_community_network(
    organizations: dict
):
    """Demonstrate community network features"""
    print_section("Community Network")

    network = CommunityNetwork()

    # Add organizations
    print_subsection("Building Network")
    for org in organizations.values():
        network.add_organization(org)
    print(f"✓ Added {len(organizations)} organizations to network")

    # Create connections
    print_subsection("Creating Connections")

    # Food bank <-> Restaurant
    network.create_connection(
        "rest_bella_cucina",
        "fb_community_food_bank",
        ConnectionType.REGULAR_DONOR,
        verified=True
    )
    print("✓ Bella Cucina → Community Food Bank (regular donor)")

    # Grocery store <-> Food bank
    network.create_connection(
        "store_fresh_market",
        "fb_community_food_bank",
        ConnectionType.REGULAR_DONOR,
        verified=True
    )
    print("✓ Fresh Market → Community Food Bank (regular donor)")

    # Food bank <-> Church
    network.create_connection(
        "fb_community_food_bank",
        "church_st_paul",
        ConnectionType.PARTNER,
        verified=True
    )
    print("✓ Community Food Bank ↔ St. Paul's Church (partners)")

    # Network analytics
    print_subsection("Network Analytics")
    stats = network.get_network_stats()
    print(f"Total organizations: {stats['total_organizations']}")
    print(f"Total connections: {stats['total_connections']}")
    print(f"Verified connections: {stats['verified_connections']}")
    print(f"Average trust score: {stats['average_trust_score']:.2f}")
    print(f"Network density: {stats['network_density']:.3f}")

    # Most connected
    print_subsection("Most Connected Organizations")
    most_connected = network.get_most_connected_orgs(limit=3)
    for org_id, count in most_connected:
        org = network.get_organization(org_id)
        print(f"  {org.name}: {count} connections")

    # Connection suggestions
    print_subsection("Connection Suggestions")
    suggestions = network.suggest_connections("rest_sunrise_diner", limit=3)
    print("Recommendations for Sunrise Diner:")
    for suggestion in suggestions:
        print(f"  {suggestion['org_name']} (score: {suggestion['score']:.2f})")
        print(f"    Reasons: {', '.join(suggestion['reasons'])}")

    return network


async def demo_hololoom_matching(
    coordinator: DonationCoordinator,
    inventory: dict,
    manager: CommunityInventoryManager
):
    """Demonstrate AI-powered matching with HoloLoom"""
    if not HOLOLOOM_AVAILABLE:
        print_section("HoloLoom AI Matching (Not Available)")
        print("HoloLoom integration not available - skipping AI demo")
        return

    print_section("AI-Powered Matching with HoloLoom")

    engine = await create_hololoom_engine(coordinator)

    try:
        # Get surplus from restaurant
        print_subsection("Learning from Past Donations")
        surplus = manager.identify_surplus("rest_bella_cucina", days_threshold=1)

        if surplus:
            listing = DonationListing(
                listing_id="demo_listing_ai_001",
                donor_org_id="rest_bella_cucina",
                items=surplus[:2],
                available_from=datetime.now(),
                available_until=datetime.now() + timedelta(hours=2)
            )

            # Teach HoloLoom about successful donation
            await engine.learn_from_donation(
                listing,
                "fb_community_food_bank",
                success=True,
                feedback="Excellent quality prepared food, perfect for evening distribution"
            )
            print("✓ Learned from successful Italian restaurant → food bank donation")

        # Find AI-enhanced matches
        print_subsection("AI-Enhanced Matching")
        listing = DonationListing(
            listing_id="demo_listing_ai_002",
            donor_org_id="store_fresh_market",
            items=manager.identify_surplus("store_fresh_market", days_threshold=2)[:3],
            available_from=datetime.now(),
            available_until=datetime.now() + timedelta(hours=4)
        )

        matches = await engine.find_enhanced_matches(listing, max_matches=3)

        print(f"Found {len(matches)} AI-enhanced matches:\n")
        for i, match in enumerate(matches, 1):
            recipient = coordinator.get_organization(match.recipient_org_id)
            print(f"{i}. {recipient.name}")
            print(f"   Total score: {match.total_score:.3f}")
            print(f"   AI prediction: {match.subscores.get('ai_prediction', 0.0):.2f}")
            print(f"   {match.reasoning}\n")

        # Get AI insights
        print_subsection("AI Insights for Organizations")
        for org_id in ["rest_bella_cucina", "store_fresh_market"]:
            org = coordinator.get_organization(org_id)
            insights = await engine.get_insights(org_id)

            print(f"\n{org.name}:")
            for insight in insights.get("insights", []):
                print(f"  • {insight['message']}")

            for rec in insights.get("recommendations", []):
                print(f"  → {rec['action']}")

    finally:
        await engine.close()


async def demo_complete_workflow():
    """Complete end-to-end demonstration"""
    print_section("SOUS Phase 2: Community Food Sharing")
    print("Demonstrating complete food sharing workflow")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load data
    organizations, inventory = await load_sample_data()

    # Demonstrate features
    manager = await demo_inventory_management(organizations, inventory)
    coordinator = await demo_donation_coordination(organizations, inventory, manager)
    network = await demo_community_network(organizations)

    # AI features (if available)
    await demo_hololoom_matching(coordinator, inventory, manager)

    # Final summary
    print_section("Summary")
    print("✓ Community food sharing system demonstrated successfully!")
    print(f"\nKey Features Showcased:")
    print(f"  • {len(organizations)} organizations in network")
    print(f"  • {sum(len(items) for items in inventory.values())} food items tracked")
    print(f"  • Intelligent donor-recipient matching")
    print(f"  • Surplus identification and waste prevention")
    print(f"  • Network connections and trust management")
    if HOLOLOOM_AVAILABLE:
        print(f"  • AI-powered matching with HoloLoom")

    print("\nPhase 2 Implementation Complete! 🎉")


if __name__ == "__main__":
    asyncio.run(demo_complete_workflow())
