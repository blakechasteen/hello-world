"""
Locations - Neighborhood Spaces

Defines the locations where residents can be found
and interactions can happen.

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import random


class LocationType(Enum):
    """Types of neighborhood locations."""
    HOME = "home"
    PUBLIC = "public"
    NATURE = "nature"
    COMMERCIAL = "commercial"
    COMMUNITY = "community"


class ActivityType(Enum):
    """Types of activities possible at locations."""
    RELAXING = "relaxing"
    WORKING = "working"
    SOCIALIZING = "socializing"
    EXERCISING = "exercising"
    CREATING = "creating"
    EATING = "eating"
    SHOPPING = "shopping"
    WALKING = "walking"
    GARDENING = "gardening"
    READING = "reading"
    MUSIC = "music"


@dataclass
class Location:
    """A location in the neighborhood."""
    location_id: str
    name: str
    location_type: LocationType
    description: str
    capacity: int = 10  # Max people at once
    activities: List[ActivityType] = field(default_factory=list)
    atmosphere: str = "pleasant"
    times_open: Dict[str, bool] = field(default_factory=lambda: {
        "morning": True, "afternoon": True, "evening": True, "night": False
    })
    current_occupants: Set[str] = field(default_factory=set)
    conversation_likelihood: float = 0.5  # Chance of conversation starting

    def is_open(self, time_of_day: str) -> bool:
        return self.times_open.get(time_of_day, False)

    def is_full(self) -> bool:
        return len(self.current_occupants) >= self.capacity

    def add_occupant(self, resident_id: str):
        if not self.is_full():
            self.current_occupants.add(resident_id)

    def remove_occupant(self, resident_id: str):
        self.current_occupants.discard(resident_id)

    def get_random_activity(self) -> ActivityType:
        if self.activities:
            return random.choice(self.activities)
        return ActivityType.RELAXING


def create_neighborhood_locations() -> Dict[str, Location]:
    """Create the neighborhood locations."""
    locations = {}

    # Community Garden - Central gathering place
    locations["community_garden"] = Location(
        location_id="community_garden",
        name="Community Garden",
        location_type=LocationType.COMMUNITY,
        description="A vibrant garden with vegetable plots, flower beds, and a small gazebo.",
        capacity=15,
        activities=[ActivityType.GARDENING, ActivityType.SOCIALIZING, ActivityType.RELAXING],
        atmosphere="peaceful and green",
        conversation_likelihood=0.7,
    )

    # Cafe
    locations["corner_cafe"] = Location(
        location_id="corner_cafe",
        name="Corner Cafe",
        location_type=LocationType.COMMERCIAL,
        description="A cozy cafe with mismatched furniture and the smell of fresh coffee.",
        capacity=12,
        activities=[ActivityType.EATING, ActivityType.SOCIALIZING, ActivityType.READING],
        atmosphere="warm and bustling",
        times_open={"morning": True, "afternoon": True, "evening": True, "night": False},
        conversation_likelihood=0.8,
    )

    # Park
    locations["oak_park"] = Location(
        location_id="oak_park",
        name="Oak Park",
        location_type=LocationType.NATURE,
        description="A small park with old oak trees, benches, and a walking path.",
        capacity=25,
        activities=[ActivityType.WALKING, ActivityType.RELAXING, ActivityType.EXERCISING, ActivityType.READING],
        atmosphere="quiet and shaded",
        conversation_likelihood=0.5,
    )

    # Street
    locations["main_street"] = Location(
        location_id="main_street",
        name="Main Street",
        location_type=LocationType.PUBLIC,
        description="The main street with shops, storefronts, and constant foot traffic.",
        capacity=50,
        activities=[ActivityType.WALKING, ActivityType.SHOPPING],
        atmosphere="busy and lively",
        conversation_likelihood=0.4,
    )

    # Library corner (outdoor reading area)
    locations["reading_corner"] = Location(
        location_id="reading_corner",
        name="Reading Corner",
        location_type=LocationType.COMMUNITY,
        description="A quiet corner with a little free library and comfortable seats.",
        capacity=6,
        activities=[ActivityType.READING, ActivityType.RELAXING],
        atmosphere="quiet and contemplative",
        conversation_likelihood=0.3,
    )

    # Community Center
    locations["community_center"] = Location(
        location_id="community_center",
        name="Community Center",
        location_type=LocationType.COMMUNITY,
        description="A hall for neighborhood gatherings, classes, and events.",
        capacity=40,
        activities=[ActivityType.SOCIALIZING, ActivityType.EXERCISING, ActivityType.CREATING],
        atmosphere="welcoming and active",
        conversation_likelihood=0.7,
    )

    # Music Corner (gazebo with instruments)
    locations["music_gazebo"] = Location(
        location_id="music_gazebo",
        name="Music Gazebo",
        location_type=LocationType.COMMUNITY,
        description="A small gazebo where neighbors gather to play music.",
        capacity=8,
        activities=[ActivityType.MUSIC, ActivityType.SOCIALIZING],
        atmosphere="melodic and joyful",
        times_open={"morning": True, "afternoon": True, "evening": True, "night": False},
        conversation_likelihood=0.6,
    )

    # Evening gathering spot
    locations["plaza_fountain"] = Location(
        location_id="plaza_fountain",
        name="Plaza Fountain",
        location_type=LocationType.PUBLIC,
        description="A small plaza with a fountain where people gather in the evening.",
        capacity=20,
        activities=[ActivityType.SOCIALIZING, ActivityType.WALKING, ActivityType.RELAXING],
        atmosphere="romantic and cool",
        times_open={"morning": True, "afternoon": True, "evening": True, "night": True},
        conversation_likelihood=0.6,
    )

    # Individual homes (one per resident)
    home_descriptions = {
        "maya_001": "Maya's colorful apartment filled with art supplies and plants.",
        "jorge_002": "Jorge's cozy house with a small herb garden in front.",
        "elena_003": "Elena's modern apartment with lots of books and tech.",
        "carlos_004": "Carlos's lived-in home full of photographs and memories.",
        "sofia_005": "Sofia's welcoming home that always smells like cooking.",
        "diego_006": "Diego's small apartment with instruments everywhere.",
    }

    for resident_id, description in home_descriptions.items():
        name = resident_id.split("_")[0].capitalize()
        locations[f"home_{resident_id}"] = Location(
            location_id=f"home_{resident_id}",
            name=f"{name}'s Home",
            location_type=LocationType.HOME,
            description=description,
            capacity=4,
            activities=[ActivityType.RELAXING, ActivityType.CREATING, ActivityType.EATING],
            atmosphere="personal",
            times_open={"morning": True, "afternoon": True, "evening": True, "night": True},
            conversation_likelihood=0.3,
        )

    return locations


class LocationManager:
    """Manages locations and resident movements."""

    def __init__(self, locations: Dict[str, Location] = None):
        self.locations = locations or create_neighborhood_locations()
        self.resident_locations: Dict[str, str] = {}  # resident_id -> location_id

    def get_location(self, location_id: str) -> Optional[Location]:
        return self.locations.get(location_id)

    def get_resident_location(self, resident_id: str) -> Optional[Location]:
        loc_id = self.resident_locations.get(resident_id)
        return self.locations.get(loc_id) if loc_id else None

    def move_resident(self, resident_id: str, to_location_id: str) -> bool:
        """Move a resident to a new location."""
        new_loc = self.locations.get(to_location_id)
        if not new_loc or new_loc.is_full():
            return False

        # Remove from current location
        current_loc_id = self.resident_locations.get(resident_id)
        if current_loc_id and current_loc_id in self.locations:
            self.locations[current_loc_id].remove_occupant(resident_id)

        # Add to new location
        new_loc.add_occupant(resident_id)
        self.resident_locations[resident_id] = to_location_id
        return True

    def get_available_locations(self, time_of_day: str) -> List[Location]:
        """Get locations open at given time."""
        return [
            loc for loc in self.locations.values()
            if loc.is_open(time_of_day) and not loc.is_full()
        ]

    def get_public_locations(self, time_of_day: str) -> List[Location]:
        """Get public locations open at given time."""
        return [
            loc for loc in self.locations.values()
            if loc.is_open(time_of_day)
            and loc.location_type != LocationType.HOME
            and not loc.is_full()
        ]

    def find_residents_at(self, location_id: str) -> Set[str]:
        """Find all residents at a location."""
        loc = self.locations.get(location_id)
        return loc.current_occupants if loc else set()

    def find_nearby_residents(self, resident_id: str) -> List[str]:
        """Find residents at the same location."""
        loc_id = self.resident_locations.get(resident_id)
        if not loc_id:
            return []
        loc = self.locations.get(loc_id)
        if not loc:
            return []
        return [r for r in loc.current_occupants if r != resident_id]

    def get_locations_by_activity(self, activity: ActivityType, time_of_day: str) -> List[Location]:
        """Find locations where an activity is possible."""
        return [
            loc for loc in self.locations.values()
            if activity in loc.activities and loc.is_open(time_of_day)
        ]

    def describe_location(self, location_id: str) -> str:
        """Get description of location with current occupants."""
        loc = self.locations.get(location_id)
        if not loc:
            return "Unknown location."

        desc = f"{loc.name} - {loc.description}"
        if loc.current_occupants:
            n = len(loc.current_occupants)
            desc += f" ({n} {'person' if n == 1 else 'people'} here)"
        else:
            desc += " (empty)"
        return desc


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("LOCATIONS SYSTEM DEMO")
    print("=" * 60)

    manager = LocationManager()

    print("\n--- NEIGHBORHOOD LOCATIONS ---\n")
    for loc_id, loc in manager.locations.items():
        if loc.location_type != LocationType.HOME:
            print(f"{loc.name}")
            print(f"  Type: {loc.location_type.value}")
            print(f"  Atmosphere: {loc.atmosphere}")
            print(f"  Activities: {', '.join(a.value for a in loc.activities)}")
            print(f"  Conversation chance: {loc.conversation_likelihood:.0%}")
            print()

    print("\n--- MOVEMENT SIMULATION ---\n")

    # Place some residents
    manager.move_resident("maya_001", "community_garden")
    manager.move_resident("jorge_002", "community_garden")
    manager.move_resident("elena_003", "corner_cafe")

    for loc_id in ["community_garden", "corner_cafe"]:
        loc = manager.get_location(loc_id)
        print(f"{loc.name}: {list(loc.current_occupants)}")

    print(f"\nResidents near Maya: {manager.find_nearby_residents('maya_001')}")
