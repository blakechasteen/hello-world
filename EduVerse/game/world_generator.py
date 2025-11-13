"""
Educational World Generator - Immersive Learning Environments

**Purpose**: Generate themed 3D worlds for educational quests
**Date**: November 13, 2025
**Integration**: DreamWeaver + EduVerse

This module generates 3 educational world types:
1. **School World** - Realistic school environment (all subjects)
2. **Fantasy World** - Magic realm (math, logic, AI as spells)
3. **Space Station** - Sci-fi environment (science, engineering)

Each world has:
- Themed locations (classrooms, labs, libraries)
- Educational NPCs (teachers, mentors, peers)
- Quest integration points
- Consistent visual theme
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4
import random

# DreamWeaver imports
from HoloLoom.dreamweaving import (
    WorldEntity,
    EntityType,
    NarrativeEvent,
    EventType,
    WorldState
)

# EduVerse imports
from EduVerse.education.curriculum import Subject


class WorldTheme(Enum):
    """Educational world themes"""
    SCHOOL = "school"           # Modern school building
    FANTASY = "fantasy"         # Magical realm
    SPACE = "space"             # Space station
    HISTORICAL = "historical"   # Time travel to historical periods
    UNDERWATER = "underwater"   # Ocean research station


@dataclass
class Location:
    """A location in the world"""
    id: str
    name: str
    description: str
    theme: WorldTheme
    subject: Optional[Subject] = None
    connected_to: List[str] = field(default_factory=list)  # Location IDs
    npcs: List[str] = field(default_factory=list)  # NPC IDs
    quests_available: List[str] = field(default_factory=list)  # Quest IDs
    atmosphere: str = "neutral"  # "quiet", "bustling", "mysterious", etc.
    visual_elements: Dict[str, str] = field(default_factory=dict)


class WorldGenerator:
    """
    Generate themed educational worlds

    Each world is a collection of:
    - Locations (classrooms, labs, libraries, etc.)
    - NPCs (teachers, mentors, peers)
    - Visual themes (colors, architecture, props)
    - Quest integration points
    """

    def __init__(self):
        self.worlds: Dict[WorldTheme, Dict[str, Location]] = {}

    def generate_world(self, theme: WorldTheme, subjects: List[Subject]) -> Dict[str, Location]:
        """
        Generate a complete world

        Args:
            theme: World theme (school, fantasy, space)
            subjects: Subjects to include locations for

        Returns:
            Dictionary of location_id → Location
        """
        if theme == WorldTheme.SCHOOL:
            return self._generate_school_world(subjects)
        elif theme == WorldTheme.FANTASY:
            return self._generate_fantasy_world(subjects)
        elif theme == WorldTheme.SPACE:
            return self._generate_space_world(subjects)
        else:
            raise ValueError(f"Unsupported theme: {theme}")

    # ========== School World ==========

    def _generate_school_world(self, subjects: List[Subject]) -> Dict[str, Location]:
        """Generate modern school environment"""
        locations = {}

        # Central hub
        lobby = Location(
            id="school_lobby",
            name="School Lobby",
            description="A bright, welcoming entrance with student art on the walls and a bulletin board showing upcoming events.",
            theme=WorldTheme.SCHOOL,
            atmosphere="bustling",
            visual_elements={
                "floor": "polished tile",
                "walls": "light blue paint with student artwork",
                "lighting": "bright fluorescent",
                "props": "trophy cases, bulletin boards, school mascot statue"
            }
        )
        locations[lobby.id] = lobby

        # Subject-specific classrooms
        if Subject.MATH in subjects:
            math_class = Location(
                id="school_math_classroom",
                name="Mathematics Classroom",
                description="A classroom filled with number charts, geometric shapes, and a whiteboard covered in equations. Desks arranged in groups for collaboration.",
                theme=WorldTheme.SCHOOL,
                subject=Subject.MATH,
                connected_to=["school_lobby"],
                atmosphere="focused",
                visual_elements={
                    "floor": "carpet tiles",
                    "walls": "covered in math posters (pi, Fibonacci, famous mathematicians)",
                    "lighting": "warm overhead lights",
                    "props": "graphing calculators, protractors, number lines, 3D geometric models"
                }
            )
            locations[math_class.id] = math_class
            lobby.connected_to.append(math_class.id)

        if Subject.SCIENCE in subjects:
            science_lab = Location(
                id="school_science_lab",
                name="Science Laboratory",
                description="A state-of-the-art lab with experiment stations, safety equipment, and microscopes. Chemical models and periodic table posters adorn the walls.",
                theme=WorldTheme.SCHOOL,
                subject=Subject.SCIENCE,
                connected_to=["school_lobby"],
                atmosphere="experimental",
                visual_elements={
                    "floor": "chemical-resistant tiles",
                    "walls": "white walls with periodic table, lab safety posters",
                    "lighting": "bright task lighting over benches",
                    "props": "beakers, microscopes, lab coats, safety goggles, skeleton model"
                }
            )
            locations[science_lab.id] = science_lab
            lobby.connected_to.append(science_lab.id)

        if Subject.ELA in subjects:
            library = Location(
                id="school_library",
                name="School Library",
                description="A quiet sanctuary of books, with cozy reading nooks, computers for research, and a librarian's desk. The smell of paper fills the air.",
                theme=WorldTheme.SCHOOL,
                subject=Subject.ELA,
                connected_to=["school_lobby"],
                atmosphere="quiet",
                visual_elements={
                    "floor": "soft carpet",
                    "walls": "floor-to-ceiling bookshelves",
                    "lighting": "warm reading lamps",
                    "props": "books, reading chairs, computers, book displays, famous quotes"
                }
            )
            locations[library.id] = library
            lobby.connected_to.append(library.id)

        if Subject.AI_READINESS in subjects:
            computer_lab = Location(
                id="school_computer_lab",
                name="Computer Lab & AI Studio",
                description="A modern tech lab with rows of powerful computers, VR headsets, and AI project displays. Posters show neural networks and robots.",
                theme=WorldTheme.SCHOOL,
                subject=Subject.AI_READINESS,
                connected_to=["school_lobby"],
                atmosphere="innovative",
                visual_elements={
                    "floor": "raised floor with cable management",
                    "walls": "whiteboards with code snippets, AI concept posters",
                    "lighting": "adjustable LED strips (RGB!)",
                    "props": "computers, VR headsets, 3D printer, robot models, AI diagrams"
                }
            )
            locations[computer_lab.id] = computer_lab
            lobby.connected_to.append(computer_lab.id)

        # Common areas
        cafeteria = Location(
            id="school_cafeteria",
            name="Cafeteria",
            description="A lively space where students gather to eat, chat, and collaborate. Lunch tables fill the room, and the food line offers healthy options.",
            theme=WorldTheme.SCHOOL,
            connected_to=["school_lobby"],
            atmosphere="social",
            visual_elements={
                "floor": "easy-clean tiles",
                "walls": "colorful murals of food groups and nutrition facts",
                "lighting": "bright overhead",
                "props": "tables, chairs, food service line, vending machines, recycling bins"
            }
        )
        locations[cafeteria.id] = cafeteria
        lobby.connected_to.append(cafeteria.id)

        return locations

    # ========== Fantasy World ==========

    def _generate_fantasy_world(self, subjects: List[Subject]) -> Dict[str, Location]:
        """Generate magical fantasy realm"""
        locations = {}

        # Central hub
        town_square = Location(
            id="fantasy_town_square",
            name="Mystical Town Square",
            description="A cobblestone plaza surrounded by magical shops and towers. A glowing fountain in the center shows swirling mathematical equations in the water.",
            theme=WorldTheme.FANTASY,
            atmosphere="mysterious",
            visual_elements={
                "floor": "ancient cobblestones with rune patterns",
                "walls": "stone buildings with glowing windows",
                "lighting": "floating magical orbs (RGB color-shifting)",
                "props": "fountain with equation water, street vendors, magical creatures"
            }
        )
        locations[town_square.id] = town_square

        # Math as magic
        if Subject.MATH in subjects:
            wizard_tower = Location(
                id="fantasy_wizard_tower",
                name="Tower of Numerical Wizardry",
                description="A tall tower where numbers float in the air and equations glow on ancient scrolls. The Wizard of Algebra teaches powerful spells here.",
                theme=WorldTheme.FANTASY,
                subject=Subject.MATH,
                connected_to=["fantasy_town_square"],
                atmosphere="mystical",
                visual_elements={
                    "floor": "spiraling stone stairs",
                    "walls": "covered in floating equation holograms",
                    "lighting": "magical blue glow from spellbooks",
                    "props": "floating numbers, magic wands (calculate!), crystal balls showing graphs"
                }
            )
            locations[wizard_tower.id] = wizard_tower
            town_square.connected_to.append(wizard_tower.id)

        # Science as alchemy
        if Subject.SCIENCE in subjects:
            alchemy_lab = Location(
                id="fantasy_alchemy_lab",
                name="Alchemist's Laboratory",
                description="A cluttered lab filled with bubbling potions, glowing crystals, and ancient chemistry equipment. The air smells of sulfur and magic.",
                theme=WorldTheme.FANTASY,
                subject=Subject.SCIENCE,
                connected_to=["fantasy_town_square"],
                atmosphere="experimental",
                visual_elements={
                    "floor": "worn wooden planks stained with chemicals",
                    "walls": "shelves of potion bottles and ingredient jars",
                    "lighting": "flickering candles and glowing potions",
                    "props": "cauldrons, test tubes, magical herbs, element crystals, dragons"
                }
            )
            locations[alchemy_lab.id] = alchemy_lab
            town_square.connected_to.append(alchemy_lab.id)

        # Library as ancient archive
        if Subject.ELA in subjects:
            ancient_library = Location(
                id="fantasy_ancient_library",
                name="Library of Eternal Stories",
                description="An impossibly vast library where books float to readers, stories come alive as holograms, and ancient tales whisper from the shelves.",
                theme=WorldTheme.FANTASY,
                subject=Subject.ELA,
                connected_to=["fantasy_town_square"],
                atmosphere="enchanted",
                visual_elements={
                    "floor": "soft magical carpet that shifts colors",
                    "walls": "endless bookshelves reaching into mist",
                    "lighting": "soft golden glow from floating candles",
                    "props": "flying books, reading spectral cats, story holograms, quill pens"
                }
            )
            locations[ancient_library.id] = ancient_library
            town_square.connected_to.append(ancient_library.id)

        # AI as oracle magic
        if Subject.AI_READINESS in subjects:
            oracle_chamber = Location(
                id="fantasy_oracle_chamber",
                name="Chamber of the AI Oracle",
                description="A crystalline chamber where an ancient AI entity speaks through glowing runes. Neural networks are visualized as magical webs of light.",
                theme=WorldTheme.FANTASY,
                subject=Subject.AI_READINESS,
                connected_to=["fantasy_town_square"],
                atmosphere="futuristic-mystical",
                visual_elements={
                    "floor": "transparent crystal showing circuitry beneath",
                    "walls": "holographic neural network visualizations",
                    "lighting": "pulsing blue AI core light",
                    "props": "AI orb, data crystals, training datasets (glowing books), robot familiars"
                }
            )
            locations[oracle_chamber.id] = oracle_chamber
            town_square.connected_to.append(oracle_chamber.id)

        # Quest hub
        guild_hall = Location(
            id="fantasy_guild_hall",
            name="Adventurer's Guild Hall",
            description="A rowdy tavern where heroes gather for quests. A large quest board shows available challenges, and the bartender knows all the rumors.",
            theme=WorldTheme.FANTASY,
            connected_to=["fantasy_town_square"],
            atmosphere="lively",
            visual_elements={
                "floor": "wooden planks",
                "walls": "weapon racks, trophy heads, quest board",
                "lighting": "warm fireplace glow",
                "props": "long tables, quest board, barrels, adventurer NPCs"
            }
        )
        locations[guild_hall.id] = guild_hall
        town_square.connected_to.append(guild_hall.id)

        return locations

    # ========== Space Station World ==========

    def _generate_space_world(self, subjects: List[Subject]) -> Dict[str, Location]:
        """Generate sci-fi space station"""
        locations = {}

        # Central hub
        command_deck = Location(
            id="space_command_deck",
            name="Command Deck",
            description="The nerve center of the station with holographic displays showing star charts, ship systems, and mission objectives. Windows offer breathtaking views of space.",
            theme=WorldTheme.SPACE,
            atmosphere="high-tech",
            visual_elements={
                "floor": "metallic grating with LED strips",
                "walls": "curved metal panels with holographic displays",
                "lighting": "cool blue ambient + spotlight panels",
                "props": "captain's chair, control consoles, holographic star map, AI assistant"
            }
        )
        locations[command_deck.id] = command_deck

        # Math as navigation/engineering
        if Subject.MATH in subjects:
            engineering_bay = Location(
                id="space_engineering_bay",
                name="Engineering Bay",
                description="A workshop filled with tools, schematics, and equations for calculating trajectories, fuel efficiency, and structural integrity. Math keeps the station alive!",
                theme=WorldTheme.SPACE,
                subject=Subject.MATH,
                connected_to=["space_command_deck"],
                atmosphere="technical",
                visual_elements={
                    "floor": "reinforced metal plates",
                    "walls": "technical schematics, equation displays",
                    "lighting": "bright work lights",
                    "props": "tools, 3D printers, holographic calculators, robot assistants"
                }
            )
            locations[engineering_bay.id] = engineering_bay
            command_deck.connected_to.append(engineering_bay.id)

        # Science as research
        if Subject.SCIENCE in subjects:
            research_lab = Location(
                id="space_research_lab",
                name="Scientific Research Laboratory",
                description="A cutting-edge lab for studying alien life, new materials, and physics experiments. Samples from distant planets line the shelves.",
                theme=WorldTheme.SPACE,
                subject=Subject.SCIENCE,
                connected_to=["space_command_deck"],
                atmosphere="scientific",
                visual_elements={
                    "floor": "sterile white tiles",
                    "walls": "lab benches, specimen containment units",
                    "lighting": "bright sterile white light",
                    "props": "microscopes, alien samples, chemistry equipment, containment fields"
                }
            )
            locations[research_lab.id] = research_lab
            command_deck.connected_to.append(research_lab.id)

        # Communications/data = ELA
        if Subject.ELA in subjects:
            comm_center = Location(
                id="space_comm_center",
                name="Communications Center",
                description="Where messages from across the galaxy arrive. Translators work to decode alien languages, and reports must be written clearly for command.",
                theme=WorldTheme.SPACE,
                subject=Subject.ELA,
                connected_to=["space_command_deck"],
                atmosphere="focused",
                visual_elements={
                    "floor": "padded acoustic tiles",
                    "walls": "communication terminals, translation databases",
                    "lighting": "soft ambient",
                    "props": "headsets, translation AIs, message terminals, universal translator"
                }
            )
            locations[comm_center.id] = comm_center
            command_deck.connected_to.append(comm_center.id)

        # AI as ship systems
        if Subject.AI_READINESS in subjects:
            ai_core = Location(
                id="space_ai_core",
                name="AI Core Chamber",
                description="The brain of the station - a massive AI system that controls life support, navigation, and defense. Neural networks visualized on every wall.",
                theme=WorldTheme.SPACE,
                subject=Subject.AI_READINESS,
                connected_to=["space_command_deck"],
                atmosphere="futuristic",
                visual_elements={
                    "floor": "transparent floor showing circuitry",
                    "walls": "flowing neural network visualizations",
                    "lighting": "pulsing blue/green AI heartbeat",
                    "props": "AI interface terminals, quantum computers, holographic assistants"
                }
            )
            locations[ai_core.id] = ai_core
            command_deck.connected_to.append(ai_core.id)

        # Social area
        observation_lounge = Location(
            id="space_observation_lounge",
            name="Observation Lounge",
            description="A relaxation area with massive windows overlooking the cosmos. Crew members gather here to unwind and share stories of their missions.",
            theme=WorldTheme.SPACE,
            connected_to=["space_command_deck"],
            atmosphere="peaceful",
            visual_elements={
                "floor": "soft carpet",
                "walls": "floor-to-ceiling windows with space views",
                "lighting": "ambient colored lights (customizable mood)",
                "props": "comfortable seating, beverage replicator, holographic games, plants"
            }
        )
        locations[observation_lounge.id] = observation_lounge
        command_deck.connected_to.append(observation_lounge.id)

        return locations

    # ========== Helper Methods ==========

    def get_location(self, world_theme: WorldTheme, location_id: str) -> Optional[Location]:
        """Get a location from a world"""
        world = self.worlds.get(world_theme, {})
        return world.get(location_id)

    def get_locations_by_subject(self, world_theme: WorldTheme, subject: Subject) -> List[Location]:
        """Get all locations for a subject in a world"""
        world = self.worlds.get(world_theme, {})
        return [loc for loc in world.values() if loc.subject == subject]

    def get_connected_locations(self, world_theme: WorldTheme, location_id: str) -> List[Location]:
        """Get locations connected to a given location"""
        location = self.get_location(world_theme, location_id)
        if not location:
            return []

        world = self.worlds.get(world_theme, {})
        return [world[loc_id] for loc_id in location.connected_to if loc_id in world]

    def to_world_entity(self, location: Location) -> WorldEntity:
        """Convert Location to DreamWeaver WorldEntity"""
        return WorldEntity(
            entity_id=location.id,
            entity_type=EntityType.LOCATION,
            name=location.name,
            description=location.description,
            attributes={
                "theme": location.theme.value,
                "subject": location.subject.value if location.subject else None,
                "atmosphere": location.atmosphere,
                "visual_elements": location.visual_elements
            },
            relationships=[
                {"target_id": loc_id, "relation_type": "CONNECTED_TO", "strength": 1.0}
                for loc_id in location.connected_to
            ],
            history=[],
            metadata={}
        )


# ========== Helper Functions ==========

def create_educational_worlds() -> WorldGenerator:
    """Create all 3 educational worlds"""
    generator = WorldGenerator()

    # Generate all 3 worlds
    all_subjects = [Subject.MATH, Subject.SCIENCE, Subject.ELA, Subject.AI_READINESS]

    generator.worlds[WorldTheme.SCHOOL] = generator.generate_world(
        WorldTheme.SCHOOL,
        all_subjects
    )

    generator.worlds[WorldTheme.FANTASY] = generator.generate_world(
        WorldTheme.FANTASY,
        all_subjects
    )

    generator.worlds[WorldTheme.SPACE] = generator.generate_world(
        WorldTheme.SPACE,
        all_subjects
    )

    return generator


if __name__ == "__main__":
    print("=== EduVerse World Generator Demo ===\n")

    # Generate all worlds
    generator = create_educational_worlds()

    # Show school world
    print("1. SCHOOL WORLD")
    print("-" * 70)
    school_world = generator.worlds[WorldTheme.SCHOOL]
    print(f"Total locations: {len(school_world)}\n")

    for location in school_world.values():
        print(f"[{location.id}] {location.name}")
        print(f"   {location.description[:100]}...")
        print(f"   Subject: {location.subject.value if location.subject else 'Common Area'}")
        print(f"   Connected to: {len(location.connected_to)} locations")
        print()

    # Show fantasy world
    print("\n2. FANTASY WORLD")
    print("-" * 70)
    fantasy_world = generator.worlds[WorldTheme.FANTASY]
    print(f"Total locations: {len(fantasy_world)}\n")

    for location in fantasy_world.values():
        print(f"[{location.id}] {location.name}")
        print(f"   {location.description[:100]}...")
        print(f"   Atmosphere: {location.atmosphere}")
        print()

    # Show space world
    print("\n3. SPACE STATION")
    print("-" * 70)
    space_world = generator.worlds[WorldTheme.SPACE]
    print(f"Total locations: {len(space_world)}\n")

    for location in space_world.values():
        print(f"[{location.id}] {location.name}")
        print(f"   {location.description[:100]}...")
        print(f"   Visual theme: {location.visual_elements.get('lighting', 'N/A')}")
        print()

    print("=" * 70)
    print(f"\nWorld generator ready! {sum(len(w) for w in generator.worlds.values())} locations across 3 worlds.")
