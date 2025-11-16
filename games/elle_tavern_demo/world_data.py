"""
Game World Data for Elle Tavern Demo

This module defines all locations, NPCs, and their relationships for the
Oakridge Tavern fantasy demo game.

Created: 2025-11-16
"""

from typing import Dict, List, Optional
from apps.elle_game_engine.emotion import EmotionalState


# ============================================================================
# LOCATIONS (4 total as requested)
# ============================================================================

LOCATIONS = {
    "tavern_inside": {
        "name": "The Rusty Mug Tavern",
        "description": (
            "A cozy tavern with worn wooden tables and a crackling fireplace. "
            "The smell of roasted meat and spilled ale mingles in the warm air. "
            "A dartboard hangs on one wall, well-used and slightly crooked. "
            "Lanterns cast dancing shadows across the room, and you can hear "
            "the murmur of quiet conversations from the few patrons scattered about. "
            "Behind the bar, bottles of various liquors line the shelves, "
            "glinting in the firelight."
        ),
        "npcs": ["innkeeper", "mysterious_stranger"],
        "exits": ["town_square"],
        "ambient_sounds": ["fire crackling", "glasses clinking", "quiet murmurs"],
        "time_descriptions": {
            "morning": "Morning sunlight streams through the windows, illuminating dancing dust motes.",
            "afternoon": "The afternoon light gives the tavern a warm, golden glow.",
            "evening": "Evening shadows lengthen as patrons begin to arrive for dinner.",
            "night": "Lantern light flickers as the tavern fills with evening revelers."
        }
    },

    "town_square": {
        "name": "Oakridge Town Square",
        "description": (
            "The heart of the small town of Oakridge. A stone fountain sits in "
            "the center, its water sparkling in the sunlight. Cobblestone paths "
            "radiate outward to the various shops and establishments. A weathered "
            "wooden notice board displays wanted posters and town announcements. "
            "Children often play near the fountain while merchants hawk their wares. "
            "To the west, the warm glow of the Rusty Mug beckons travelers. "
            "To the east, the vibrant colors of Market Street catch the eye. "
            "A stern-looking guard watches over everything with careful attention."
        ),
        "npcs": ["guard", "child"],
        "exits": ["tavern_inside", "market", "forest_path"],
        "ambient_sounds": ["fountain splashing", "distant merchants", "children playing"],
        "time_descriptions": {
            "morning": "Merchants are setting up their stalls as the town awakens.",
            "afternoon": "The square bustles with activity as townspeople go about their business.",
            "evening": "The crowds thin as people head home for dinner.",
            "night": "The square is quiet save for the guard's watchful presence and the fountain's gentle splash."
        }
    },

    "market": {
        "name": "Market Street",
        "description": (
            "Colorful stalls line the cobblestone street, their canvas awnings "
            "fluttering in the breeze. The air is thick with the scent of fresh bread, "
            "exotic spices, and tanned leather. Merchants call out their wares, "
            "competing for the attention of passersby. You can find everything here: "
            "vegetables from local farms, fine fabrics from distant lands, tools forged "
            "by the town blacksmith, and mysterious trinkets from traveling peddlers. "
            "At the center stands Marcus's stall, the largest and most well-stocked. "
            "Wooden crates and barrels are stacked everywhere, evidence of recent shipments."
        ),
        "npcs": ["merchant"],
        "exits": ["town_square"],
        "ambient_sounds": ["merchants calling", "coins jingling", "haggling voices"],
        "time_descriptions": {
            "morning": "Merchants are eagerly opening their stalls, displaying fresh goods.",
            "afternoon": "The market is at its busiest, with crowds of shoppers browsing.",
            "evening": "Some stalls begin to close, but a few merchants linger for late customers.",
            "night": "The market is closed, stalls shuttered and streets empty."
        }
    },

    "forest_path": {
        "name": "Shadowed Forest Path",
        "description": (
            "A narrow dirt path winds into the dark forest beyond Oakridge. "
            "Ancient oak trees loom overhead, their thick canopy blocking most sunlight. "
            "The air is cooler here, heavy with the scent of moss and damp earth. "
            "Strange mushrooms glow faintly at the base of trees, and you can hear "
            "the distant hoot of owls even during the day. The path is rarely traveled; "
            "most townspeople avoid it, whispering about strange lights and stranger sounds. "
            "Somehow, you feel as though the forest itself is watching you. "
            "Only those trusted by the mysterious stranger know this place exists."
        ),
        "npcs": [],
        "exits": ["town_square"],
        "unlocked": False,  # Must be unlocked through quest
        "unlock_flag": "forest_path_discovered",
        "ambient_sounds": ["rustling leaves", "distant owl hoots", "mysterious whispers"],
        "time_descriptions": {
            "morning": "Mist clings to the ground, making the path seem even more mysterious.",
            "afternoon": "Scattered sunbeams pierce the canopy, creating pockets of light.",
            "evening": "The forest grows darker, and the mushrooms begin to glow more brightly.",
            "night": "The glowing mushrooms illuminate the path with an eerie, otherworldly light."
        }
    }
}


# ============================================================================
# NPC DATA (5 NPCs as requested)
# ============================================================================

NPCS = {
    "innkeeper": {
        "id": "innkeeper",
        "name": "Bob the Innkeeper",
        "role": "innkeeper",
        "location": "tavern_inside",

        "personality": {
            "traits": ["friendly", "gossipy", "helpful", "observant", "fatherly"],
            "dialogue_style": "warm and welcoming, speaks in a folksy manner",
            "speech_patterns": [
                "Often starts sentences with 'Well now...'",
                "Uses expressions like 'bless my stars' and 'by the light'",
                "Tends to ramble when excited about gossip",
                "Calls customers 'friend' or 'traveler'"
            ],
            "backstory": (
                "Bob has run the Rusty Mug for twenty years, inheriting it from his father. "
                "He knows everyone in Oakridge and hears all the town gossip. Despite his "
                "friendly demeanor, he's observant and often notices things others miss. "
                "He's been worried lately about strange noises from the cellar—sounds that "
                "suggest something more troublesome than the usual rats."
            ),
            "sample_dialogue": [
                "Welcome, welcome! Pull up a chair and rest your weary bones!",
                "Well now, I haven't seen you around these parts before. New in town, are you?",
                "Bless my stars, you should hear what happened at the market yesterday!",
                "Can I get you some stew? Made it myself this morning. Best in Oakridge, if I do say so!"
            ]
        },

        "initial_emotion": {
            "valence": 0.0,      # Neutral
            "arousal": 0.5,      # Moderately energized (busy innkeeper)
            "dominance": 0.5,    # Average confidence
            "trust": 0.5,        # Neutral toward new travelers
            "baseline_valence": 0.1,   # Naturally slightly cheerful
            "baseline_trust": 0.6      # Generally trusting person
        },

        "voice_profile": {
            "voice_id": "alloy",  # OpenAI TTS voice
            "pitch": 1.0,
            "speed": 1.0,
            "emotion": "warm",
            "description": "Warm, friendly male voice with a slightly gravelly quality"
        },

        "quest_hooks": ["rat_problem", "cellar_mystery"],
        "shop_items": ["ale", "stew", "room_key"],
        "rumors": [
            "The guard captain has been on edge lately—more bandits on the roads.",
            "Marcus the merchant lost a shipment somewhere between here and Millbrook.",
            "That mysterious stranger in the corner? Been here three days, barely says a word.",
            "Little Lily is heartbroken—her cat Whiskers went missing two days ago."
        ]
    },

    "guard": {
        "id": "guard",
        "name": "Captain Sarah Ironhelm",
        "role": "town_guard_captain",
        "location": "town_square",

        "personality": {
            "traits": ["stern", "duty-bound", "fair", "protective", "honorable"],
            "dialogue_style": "formal and direct, military bearing",
            "speech_patterns": [
                "Often addresses people by title or 'citizen'",
                "Speaks in short, clipped sentences",
                "Uses military terminology ('patrol', 'duty', 'orders')",
                "Rarely shows emotion in her voice"
            ],
            "backstory": (
                "Captain Sarah Ironhelm earned her position through merit, not connections. "
                "She came to Oakridge five years ago after serving in the capital's guard. "
                "Though her stern exterior intimidates some, she genuinely cares about protecting "
                "the townspeople. Recently, bandit attacks on trade caravans have increased, "
                "and she's frustrated by her limited resources. She respects those who prove "
                "themselves capable and honorable."
            ),
            "sample_dialogue": [
                "State your business, traveler. Oakridge is a peaceful town, and I intend to keep it that way.",
                "I don't have time for idle chatter. If you've got information about the bandits, speak up.",
                "You want to help? Actions speak louder than words, citizen.",
                "The roads aren't safe. Travel with caution, and keep your wits about you."
            ]
        },

        "initial_emotion": {
            "valence": -0.1,     # Slightly negative (stress from bandit situation)
            "arousal": 0.6,      # Alert and energized
            "dominance": 0.8,    # High confidence (position of authority)
            "trust": 0.4,        # Naturally suspicious (comes with the job)
            "baseline_valence": 0.0,   # Naturally serious
            "baseline_trust": 0.5      # Professional skepticism
        },

        "voice_profile": {
            "voice_id": "nova",  # OpenAI TTS voice
            "pitch": 0.9,        # Slightly lower
            "speed": 0.95,       # Deliberate pace
            "emotion": "firm",
            "description": "Firm, authoritative female voice"
        },

        "quest_hooks": ["bandit_patrol", "caravan_escort", "town_defense"],
        "reputation_thresholds": {
            "suspicious": 0.0,    # Default
            "neutral": 0.5,       # After first positive interaction
            "trusted": 0.7,       # After completing a quest
            "deputy": 0.9         # After multiple quests
        }
    },

    "merchant": {
        "id": "merchant",
        "name": "Marcus the Magnificent",
        "role": "merchant",
        "location": "market",

        "personality": {
            "traits": ["shrewd", "enthusiastic", "greedy", "helpful", "optimistic"],
            "dialogue_style": "energetic sales pitch, always upbeat",
            "speech_patterns": [
                "Speaks quickly and enthusiastically",
                "Uses superlatives constantly ('finest', 'best', 'most amazing')",
                "Often mentions profits or good deals",
                "Exaggerates the quality of his goods"
            ],
            "backstory": (
                "Marcus came to Oakridge ten years ago with nothing but a cart and ambition. "
                "Through shrewd trading and relentless optimism, he built the largest stall "
                "in the market. He genuinely believes his goods are the finest (they're actually "
                "pretty good), and his enthusiasm is infectious. Recently, he lost a valuable "
                "shipment to bandits and is desperate to recover it. Despite his greed, he's "
                "fair in his dealings and remembers those who help him."
            ),
            "sample_dialogue": [
                "Welcome, welcome! You've come to the right place for the finest goods in all the realm!",
                "What'll it be today? Spices from the south? Tools from the finest smiths? I have it all!",
                "Ah, a discerning customer! I can see you have an eye for quality!",
                "Business is good, business is good! Well, it would be better if those cursed bandits hadn't taken my shipment..."
            ]
        },

        "initial_emotion": {
            "valence": 0.2,      # Generally positive (loves his work)
            "arousal": 0.7,      # High energy (enthusiastic salesman)
            "dominance": 0.6,    # Confident in his domain
            "trust": 0.6,        # Trusting enough to do business
            "baseline_valence": 0.3,   # Naturally optimistic
            "baseline_trust": 0.6      # Trusts customers (to a point)
        },

        "voice_profile": {
            "voice_id": "onyx",  # OpenAI TTS voice
            "pitch": 1.1,        # Slightly higher, energetic
            "speed": 1.1,        # Faster pace
            "emotion": "eager",
            "description": "Eager, enthusiastic male voice"
        },

        "quest_hooks": ["lost_shipment", "supplier_negotiation", "rare_ingredient"],
        "shop_items": ["healing_potion", "rope", "torch", "rations", "exotic_spice"],
        "price_moods": {
            "angry": 1.3,        # 30% markup when angry
            "neutral": 1.0,      # Normal prices
            "happy": 0.85,       # 15% discount when happy
            "grateful": 0.7      # 30% discount when grateful
        }
    },

    "mysterious_stranger": {
        "id": "mysterious_stranger",
        "name": "The Hooded Stranger",
        "role": "mysterious_traveler",
        "location": "tavern_inside",

        "personality": {
            "traits": ["cryptic", "wise", "enigmatic", "patient", "observant"],
            "dialogue_style": "speaks in riddles and metaphors, philosophical",
            "speech_patterns": [
                "Often speaks in the third person or passive voice",
                "Uses poetic, flowery language",
                "References 'threads of fate' and 'paths unseen'",
                "Rarely gives direct answers"
            ],
            "backstory": (
                "Nobody knows where the stranger came from or why they're in Oakridge. "
                "They appeared three days ago, cloaked and hooded, and have barely moved "
                "from their corner table. When they do speak, it's in cryptic riddles that "
                "somehow prove prophetic. Those who earn their trust learn that they guard "
                "secret knowledge—including the location of the hidden forest path and what "
                "lies within. Some whisper they might be a wizard, a sage, or something else entirely."
            ),
            "sample_dialogue": [
                "The threads of fate are curious things... they weave and twist in patterns we cannot see.",
                "You seek answers. But perhaps the question itself is the answer you need.",
                "Three paths lie before you: the path of light, the path of shadow, and the path forgotten by time.",
                "The forest remembers what the town has forgotten. But only the worthy may walk its shadowed ways."
            ]
        },

        "initial_emotion": {
            "valence": 0.0,      # Neutral, inscrutable
            "arousal": 0.3,      # Very calm, almost meditative
            "dominance": 0.7,    # Quiet confidence
            "trust": 0.3,        # Slow to trust, observant
            "baseline_valence": 0.0,   # Remains neutral
            "baseline_trust": 0.3      # Naturally cautious
        },

        "voice_profile": {
            "voice_id": "echo",  # OpenAI TTS voice
            "pitch": 0.85,       # Lower, mysterious
            "speed": 0.9,        # Slow, deliberate
            "emotion": "calm",
            "description": "Deep, calm, mysterious voice with a hint of otherworldliness"
        },

        "quest_hooks": ["forest_path_unlock", "ancient_knowledge", "prophecy_quest"],
        "trust_thresholds": {
            "stranger": 0.0,      # Default—won't share secrets
            "curious": 0.5,       # Begins to hint at mysteries
            "worthy": 0.75,       # Reveals forest path
            "chosen": 0.9         # Shares deeper secrets
        }
    },

    "child": {
        "id": "child",
        "name": "Lily",
        "role": "local_child",
        "location": "town_square",

        "personality": {
            "traits": ["innocent", "curious", "playful", "caring", "emotional"],
            "dialogue_style": "childlike, emotional, earnest",
            "speech_patterns": [
                "Simple, direct sentences",
                "Shows emotions openly (crying, laughing)",
                "Asks lots of questions",
                "Gets distracted easily"
            ],
            "backstory": (
                "Lily is a seven-year-old girl who lives in Oakridge with her mother, the baker. "
                "She's usually full of energy, playing by the fountain and asking endless questions "
                "of anyone who'll listen. But two days ago, her beloved cat Whiskers disappeared, "
                "and she's been heartbroken ever since. She's convinced Whiskers is lost and scared, "
                "and she desperately wants someone to help find her. Despite her sadness, she still "
                "tries to be brave. Finding Whiskers would mean the world to her."
            ),
            "sample_dialogue": [
                "Have you seen my cat? She's orange and white and her name is Whiskers!",
                "*sniffling* I miss Whiskers so much... Mama says she'll come home, but I'm worried...",
                "You'll help me find Whiskers? Really? Oh, thank you, thank you!",
                "Whiskers likes to chase butterflies... maybe she chased one too far and got lost?"
            ]
        },

        "initial_emotion": {
            "valence": -0.4,     # Sad (missing cat)
            "arousal": 0.6,      # Still energetic despite sadness
            "dominance": 0.2,    # Child, feels powerless
            "trust": 0.8,        # Trusting child
            "baseline_valence": 0.6,   # Naturally very happy
            "baseline_trust": 0.8      # Naturally trusting
        },

        "voice_profile": {
            "voice_id": "shimmer",  # OpenAI TTS voice
            "pitch": 1.3,           # Higher pitch (child)
            "speed": 1.1,           # Slightly faster (energetic)
            "emotion": "earnest",
            "description": "Young, earnest female voice"
        },

        "quest_hooks": ["find_whiskers"],
        "rewards": {
            "gratitude": "friendship",
            "item": "lucky_flower"  # Gives player a flower she picked
        }
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_location(location_id: str) -> Optional[Dict]:
    """Get location data by ID."""
    return LOCATIONS.get(location_id)


def get_npc(npc_id: str) -> Optional[Dict]:
    """Get NPC data by ID."""
    return NPCS.get(npc_id)


def get_npcs_at_location(location_id: str) -> List[Dict]:
    """Get all NPCs at a specific location."""
    location = get_location(location_id)
    if not location:
        return []

    npc_ids = location.get("npcs", [])
    return [get_npc(npc_id) for npc_id in npc_ids if get_npc(npc_id)]


def get_available_exits(location_id: str, world_flags: Dict[str, bool]) -> List[str]:
    """
    Get available exits from a location, considering locked areas.

    Args:
        location_id: Current location ID
        world_flags: Dict of world state flags

    Returns:
        List of accessible location IDs
    """
    location = get_location(location_id)
    if not location:
        return []

    exits = location.get("exits", [])
    available = []

    for exit_id in exits:
        exit_location = get_location(exit_id)
        if not exit_location:
            continue

        # Check if exit is locked
        if exit_location.get("unlocked", True):  # Unlocked by default
            available.append(exit_id)
        else:
            # Check if unlock flag is set
            unlock_flag = exit_location.get("unlock_flag")
            if unlock_flag and world_flags.get(unlock_flag, False):
                available.append(exit_id)

    return available


def get_location_description(location_id: str, time_of_day: str = "afternoon") -> str:
    """
    Get full location description including time-specific details.

    Args:
        location_id: Location ID
        time_of_day: Time of day (morning/afternoon/evening/night)

    Returns:
        Full description string
    """
    location = get_location(location_id)
    if not location:
        return "You are nowhere."

    base_description = location.get("description", "")
    time_descriptions = location.get("time_descriptions", {})
    time_detail = time_descriptions.get(time_of_day, "")

    if time_detail:
        return f"{base_description}\n\n{time_detail}"
    return base_description


def create_npc_emotional_state(npc_id: str) -> EmotionalState:
    """
    Create an EmotionalState object from NPC data.

    Args:
        npc_id: NPC identifier

    Returns:
        EmotionalState object initialized with NPC's default emotion
    """
    npc = get_npc(npc_id)
    if not npc:
        return EmotionalState()  # Default neutral state

    emotion_data = npc.get("initial_emotion", {})
    return EmotionalState(
        valence=emotion_data.get("valence", 0.0),
        arousal=emotion_data.get("arousal", 0.5),
        dominance=emotion_data.get("dominance", 0.5),
        trust=emotion_data.get("trust", 0.5),
        baseline_valence=emotion_data.get("baseline_valence", 0.0),
        baseline_trust=emotion_data.get("baseline_trust", 0.5)
    )


def get_npc_sample_dialogue(npc_id: str) -> List[str]:
    """Get sample dialogue lines for an NPC."""
    npc = get_npc(npc_id)
    if not npc:
        return []

    personality = npc.get("personality", {})
    return personality.get("sample_dialogue", [])


def get_all_location_ids() -> List[str]:
    """Get list of all location IDs."""
    return list(LOCATIONS.keys())


def get_all_npc_ids() -> List[str]:
    """Get list of all NPC IDs."""
    return list(NPCS.keys())


# ============================================================================
# LOCATION VALIDATION
# ============================================================================

def validate_world_data() -> Dict[str, List[str]]:
    """
    Validate world data integrity.

    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []

    # Check NPCs reference valid locations
    for npc_id, npc_data in NPCS.items():
        location = npc_data.get("location")
        if location not in LOCATIONS:
            errors.append(f"NPC '{npc_id}' references invalid location '{location}'")

    # Check locations reference valid NPCs
    for loc_id, loc_data in LOCATIONS.items():
        for npc_id in loc_data.get("npcs", []):
            if npc_id not in NPCS:
                errors.append(f"Location '{loc_id}' references invalid NPC '{npc_id}'")

    # Check exits are valid
    for loc_id, loc_data in LOCATIONS.items():
        for exit_id in loc_data.get("exits", []):
            if exit_id not in LOCATIONS:
                errors.append(f"Location '{loc_id}' has invalid exit '{exit_id}'")

    # Warnings for missing data
    for npc_id, npc_data in NPCS.items():
        if not npc_data.get("personality", {}).get("sample_dialogue"):
            warnings.append(f"NPC '{npc_id}' has no sample dialogue")
        if not npc_data.get("voice_profile"):
            warnings.append(f"NPC '{npc_id}' has no voice profile")

    return {"errors": errors, "warnings": warnings}


if __name__ == "__main__":
    # Run validation when executed directly
    validation = validate_world_data()

    if validation["errors"]:
        print("❌ ERRORS:")
        for error in validation["errors"]:
            print(f"  - {error}")
    else:
        print("✅ No errors found")

    if validation["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    print(f"\n📊 World Statistics:")
    print(f"  - Locations: {len(LOCATIONS)}")
    print(f"  - NPCs: {len(NPCS)}")
    print(f"  - Total dialogue samples: {sum(len(get_npc_sample_dialogue(npc)) for npc in NPCS)}")
