"""
Create Complete 500-Symbol Base Database (v2)

Expands from pilot (51 symbols) to full production database (500 symbols).
- 50 symbols per category x 10 categories
- Maintains consistency with pilot structure
- Ready for batch enrichment with real LLM

Key categories and expansion:
  - trapped: 6 pilot -> 50 total (add 44 variants)
  - loss: 5 pilot -> 50 total (add 45 variants)
  - fear: 5 pilot -> 50 total (add 45 variants)
  - transformation: 5 pilot -> 50 total (add 45 variants)
  - connection: 5 pilot -> 50 total (add 45 variants)
  - power: 5 pilot -> 50 total (add 45 variants)
  - guilt: 5 pilot -> 50 total (add 45 variants)
  - hope: 5 pilot -> 50 total (add 45 variants)
  - conflict: 5 pilot -> 50 total (add 45 variants)
  - mystery: 5 pilot -> 50 total (add 45 variants)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive variants for each category
CATEGORY_VARIANTS = {
    "trapped": {
        "caged_bird": 9,
        "prison_cell": 9,
        "quicksand": 9,
        "maze": 9,
        "chains": 9,
        "spider_web": 9,
    },
    "loss": {
        "broken_mirror": 9,
        "empty_room": 9,
        "fading_photograph": 9,
        "falling_leaves": 9,
        "abandoned_house": 9,
    },
    "fear": {
        "drowning": 9,
        "shadow": 9,
        "darkness": 9,
        "cliff_edge": 9,
        "void": 9,
    },
    "transformation": {
        "metamorphosis": 9,
        "cocoon": 9,
        "phoenix": 9,
        "broken_glass": 9,
        "sunset_sunrise": 9,
    },
    "connection": {
        "bridge": 9,
        "intertwined_trees": 9,
        "shared_heartbeat": 9,
        "golden_thread": 9,
        "ceremony": 9,
    },
    "power": {
        "throne": 9,
        "crown": 9,
        "mountain_peak": 9,
        "lightning": 9,
        "lion": 9,
    },
    "guilt": {
        "bloodstain": 9,
        "following_shadow": 9,
        "weight": 9,
        "crack_spreading": 9,
        "worm_eaten_fruit": 9,
    },
    "hope": {
        "sunrise": 9,
        "candle": 9,
        "star": 9,
        "green_shoot": 9,
        "door_ajar": 9,
    },
    "conflict": {
        "two_wolves": 9,
        "split_path": 9,
        "storm": 9,
        "battle": 9,
        "scales_unbalanced": 9,
    },
    "mystery": {
        "veil": 9,
        "fog": 9,
        "locked_box": 9,
        "mask": 9,
        "labyrinth": 9,
    },
}

# Variant name patterns for each symbol type
VARIANT_PATTERNS = {
    # Trapped variants
    "caged_bird": [
        "caged_nightingale", "imprisoned_eagle", "caged_falcon", "trapped_swallow",
        "confined_crow", "captive_sparrow", "caged_raven", "enclosed_phoenix",
        "locked_lark"
    ],
    "prison_cell": [
        "solitary_confinement", "dungeon_chamber", "stone_prison", "iron_cage",
        "locked_cell", "dark_chamber", "isolated_room", "barred_window",
        "concrete_cell"
    ],
    "quicksand": [
        "sinking_mud", "shifting_ground", "soft_sand_trap", "bog_entrapment",
        "mire_field", "muddy_depths", "swamp_pull", "sandy_grave",
        "moving_earth"
    ],
    "maze": [
        "labyrinth", "intricate_passage", "winding_corridors", "twisted_path",
        "mirror_maze", "garden_labyrinth", "underground_warren", "architectural_trap",
        "hedge_maze"
    ],
    "chains": [
        "iron_shackles", "binding_ropes", "weighted_chains", "mental_chains",
        "golden_chains", "invisible_chains", "ancestral_chains", "karmic_bonds",
        "metaphorical_shackles"
    ],
    "spider_web": [
        "silken_trap", "intricate_web", "sticky_threads", "trapped_fly",
        "interconnected_strands", "delicate_net", "entrapping_silk", "architectural_web",
        "jeweled_web"
    ],

    # Loss variants
    "broken_mirror": [
        "shattered_glass", "fractured_reflection", "cracked_surface", "fragmented_image",
        "mirror_shards", "broken_reflection", "splintered_glass", "ruined_mirror",
        "shattered_self"
    ],
    "empty_room": [
        "vacant_chamber", "barren_space", "hollow_room", "abandoned_interior",
        "stripped_walls", "echoing_emptiness", "unfurnished_space", "bare_walls",
        "desolate_room"
    ],
    "fading_photograph": [
        "yellowed_picture", "fading_image", "sepia_memory", "degraded_photo",
        "faded_colors", "ghostly_image", "blurred_memory", "worn_photograph",
        "disintegrating_picture"
    ],
    "falling_leaves": [
        "autumn_decay", "scattered_foliage", "drifting_leaves", "seasonal_death",
        "leaf_by_leaf", "cascading_fall", "spinning_descent", "brown_decay",
        "winter_approach"
    ],
    "abandoned_house": [
        "vacant_mansion", "decaying_structure", "crumbling_walls", "ghostly_interior",
        "neglected_home", "boarded_windows", "deserted_halls", "lonely_structure",
        "deteriorating_building"
    ],

    # Fear variants
    "drowning": [
        "sinking_water", "suffocating_depths", "water_lungs", "struggling_surface",
        "dark_deep", "panic_water", "weighted_down", "underwater_tomb",
        "rising_water"
    ],
    "shadow": [
        "dark_figure", "looming_presence", "creeping_darkness", "cast_shadow",
        "following_shadow", "expanding_shadow", "shadow_self", "darkness_shape",
        "ominous_silhouette"
    ],
    "darkness": [
        "absolute_black", "endless_night", "void_space", "consuming_black",
        "impenetrable_dark", "light_void", "starless_night", "deep_dark",
        "suffocating_black"
    ],
    "cliff_edge": [
        "precipice_fall", "sheer_drop", "dangerous_edge", "vertiginous_height",
        "final_step", "point_of_no_return", "looking_down", "teetering_edge",
        "chasm_mouth"
    ],
    "void": [
        "infinite_emptiness", "nothingness", "black_expanse", "empty_universe",
        "boundless_void", "cosmic_emptiness", "spiritual_void", "existential_void",
        "endless_nothing"
    ],

    # Transformation variants
    "metamorphosis": [
        "body_change", "alien_form", "unwilling_transformation", "insect_body",
        "grotesque_change", "monstrous_becoming", "shape_shifting", "forced_evolution",
        "self_dissolution"
    ],
    "cocoon": [
        "silken_chrysalis", "transformation_chamber", "protective_shell", "rebirth_chamber",
        "vulnerable_state", "wrapped_becoming", "temporary_tomb", "gestation_space",
        "chrysalis_stage"
    ],
    "phoenix": [
        "fiery_rebirth", "ashes_rising", "burning_renewal", "cyclical_death",
        "flame_rebirth", "rising_from_ruin", "fire_transformation", "immortal_cycle",
        "burning_hope"
    ],
    "broken_glass": [
        "sharp_fragments", "shattering_moment", "transformation_through_break", "cutting_change",
        "reformed_pieces", "transparent_breaking", "destructive_change", "beautiful_breaking",
        "mosaic_fragments"
    ],
    "sunset_sunrise": [
        "death_rebirth_cycle", "closing_opening", "twilight_moment", "transitional_light",
        "burning_horizon", "changing_sky", "day_night_boundary", "eternal_transition",
        "golden_moment"
    ],

    # Connection variants
    "bridge": [
        "spanning_gap", "connecting_path", "divided_shores", "crossing_point",
        "arching_span", "linked_worlds", "bridge_journey", "passage_between",
        "rope_bridge"
    ],
    "intertwined_trees": [
        "root_connection", "shared_growth", "entwined_branches", "loving_trees",
        "forest_web", "tree_couple", "intertwined_roots", "mutual_support",
        "reaching_branches"
    ],
    "shared_heartbeat": [
        "synchronized_rhythm", "two_hearts_one", "rhythmic_connection", "life_pulse",
        "beating_together", "heart_synchrony", "vital_rhythm", "mutual_beat",
        "shared_pulse"
    ],
    "golden_thread": [
        "fate_string", "invisible_link", "guiding_thread", "connecting_line",
        "golden_cord", "destiny_thread", "threads_entwined", "bright_connection",
        "thread_pulling"
    ],
    "ceremony": [
        "ritual_bond", "shared_sacred", "ceremonial_connection", "binding_ritual",
        "communal_gathering", "sacred_moment", "ritual_transformation", "ceremonial_space",
        "shared_sacred_space"
    ],

    # Power variants
    "throne": [
        "seat_authority", "golden_seat", "royal_chair", "power_position",
        "empty_throne", "contested_throne", "distant_throne", "claiming_throne",
        "inherited_throne"
    ],
    "crown": [
        "head_ornament", "symbol_rule", "royal_crown", "heavy_crown",
        "jeweled_crown", "iron_crown", "contested_crown", "inherited_crown",
        "ceremonial_crown"
    ],
    "mountain_peak": [
        "highest_point", "summit_struggle", "reaching_top", "perspective_height",
        "towering_majesty", "solitary_peak", "snow_crowned", "perspective_peak",
        "climbing_mountain"
    ],
    "lightning": [
        "sudden_strike", "destructive_power", "illuminating_strike", "electric_force",
        "fast_death", "sky_strike", "creative_destruction", "power_strike",
        "fear_lightning"
    ],
    "lion": [
        "fierce_ruler", "golden_majesty", "roaring_power", "predatory_strength",
        "proud_beast", "king_animal", "protective_lion", "noble_lion",
        "hunting_lion"
    ],

    # Guilt variants
    "bloodstain": [
        "crimson_mark", "indelible_stain", "guilty_mark", "shameful_sign",
        "spreading_stain", "dried_blood", "hidden_stain", "visible_guilt",
        "stubborn_mark"
    ],
    "following_shadow": [
        "pursuing_guilt", "attached_shadow", "following_presence", "constant_companion",
        "inescapable_shadow", "dark_follower", "guilt_shadow", "shame_shadow",
        "lengthening_shadow"
    ],
    "weight": [
        "burden_carry", "pressing_weight", "heavy_load", "accumulated_burden",
        "crushing_weight", "invisible_weight", "increasing_weight", "burden_increasing",
        "weight_bearing"
    ],
    "crack_spreading": [
        "growing_fracture", "spider_web_crack", "spreading_break", "deepening_crack",
        "fragile_surface", "deteriorating_structure", "breaking_apart", "cracking_foundation",
        "creeping_damage"
    ],
    "worm_eaten_fruit": [
        "internal_rot", "hidden_corruption", "poisoned_fruit", "infected_seed",
        "decay_inside", "beautiful_outside", "hidden_sickness", "contaminated_fruit",
        "spoiled_promise"
    ],

    # Hope variants
    "sunrise": [
        "morning_light", "dawn_breaking", "new_day", "golden_morning",
        "rising_sun", "light_returning", "darkness_ending", "new_beginning",
        "morning_hope"
    ],
    "candle": [
        "small_light", "bright_flame", "guiding_light", "warming_glow",
        "persistent_flame", "flickering_hope", "steadfast_light", "simple_light",
        "candlelight"
    ],
    "star": [
        "distant_light", "guiding_star", "constant_hope", "celestial_promise",
        "shining_point", "eternal_light", "hope_point", "stellar_constant",
        "wishing_star"
    ],
    "green_shoot": [
        "emerging_growth", "spring_sign", "new_life", "pushing_through",
        "tender_growth", "life_pushing", "growth_beginning", "renewal_sign",
        "hopeful_green"
    ],
    "door_ajar": [
        "opening_possibility", "space_entering", "partial_opening", "hopeful_opening",
        "invitation_implied", "chance_entering", "light_through", "hope_entrance",
        "opening_door"
    ],

    # Conflict variants
    "two_wolves": [
        "internal_battle", "dual_nature", "good_evil_fight", "opposing_forces",
        "feeding_choice", "wild_tame", "light_dark_battle", "nature_nurture",
        "instinct_reason"
    ],
    "split_path": [
        "diverging_road", "choose_direction", "impossible_choice", "fork_decision",
        "road_divergence", "path_choosing", "road_split", "decision_point",
        "multiple_futures"
    ],
    "storm": [
        "violent_weather", "turbulent_force", "destructive_rage", "conflicting_forces",
        "chaos_weather", "storm_raging", "wind_battle", "storm_center",
        "shelter_seeking"
    ],
    "battle": [
        "military_conflict", "warrior_clash", "force_meeting", "combat_violence",
        "sword_clash", "army_clash", "victory_defeat", "battle_line",
        "battlefield_stood"
    ],
    "scales_unbalanced": [
        "weight_imbalance", "unequal_sides", "scales_tipping", "justice_broken",
        "heavy_side", "light_side", "balanced_moment", "scales_falling",
        "weighed_judgment"
    ],

    # Mystery variants
    "veil": [
        "hidden_truth", "semi_transparent", "concealing_fabric", "revealing_concealing",
        "thin_veil", "thick_veil", "lifting_veil", "drawn_veil",
        "mysterious_veil"
    ],
    "fog": [
        "obscuring_mist", "hidden_landscape", "confused_vision", "mysterious_fog",
        "rolling_fog", "thick_fog", "clearing_fog", "fog_deepening",
        "fog_cutting"
    ],
    "locked_box": [
        "sealed_container", "hidden_contents", "mysterious_box", "locked_secret",
        "key_missing", "box_sealed", "contents_unknown", "box_forgotten",
        "box_opening"
    ],
    "mask": [
        "hidden_face", "false_appearance", "true_self_hidden", "mask_wearing",
        "mask_slipping", "mask_revealing", "masked_ball", "mask_society",
        "true_face_hidden"
    ],
    "labyrinth": [
        "intricate_maze", "path_confusing", "center_hidden", "escape_impossible",
        "winding_path", "lost_inside", "thread_guiding", "labyrinth_heart",
        "center_reaching"
    ],
}

def expand_pilot_to_500() -> List[Dict]:
    """Expand pilot symbols to full 500-symbol database."""

    # Load pilot symbols
    pilot_path = Path(__file__).parent / "symbol_database_pilot.json"
    with open(pilot_path, 'r') as f:
        pilot_symbols = json.load(f)

    logger.info(f"Loaded {len(pilot_symbols)} pilot symbols")

    # Create expanded database
    expanded_symbols = []
    category_counts = {}

    # Add all pilot symbols
    for sym in pilot_symbols:
        expanded_symbols.append(sym)
        cat = sym['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Expand variants
    for pilot_sym in pilot_symbols:
        symbol_id = pilot_sym['symbol_id']
        category = pilot_sym['category']

        # Get variants for this symbol
        variants = VARIANT_PATTERNS.get(symbol_id, [])

        # Add variant symbols
        for variant_id in variants:
            # Create new symbol based on pilot template
            new_symbol = {
                "symbol_id": variant_id,
                "description": f"{variant_id.replace('_', ' ').title()} - variant of {symbol_id}",
                "emotion_tags": pilot_sym['emotion_tags'],
                "category": category,
                "existing_references": pilot_sym['existing_references']
            }
            expanded_symbols.append(new_symbol)
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # Verify target counts
    logger.info("\nExpansion results by category:")
    total_symbols = 0
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        status = "OK" if count == 50 else "INCOMPLETE"
        logger.info(f"  {status} {cat}: {count} symbols")
        total_symbols += count

    logger.info(f"\nTotal symbols: {total_symbols}")

    return expanded_symbols

def main():
    """Create and save expanded database."""

    logger.info("=" * 60)
    logger.info("Creating Full 500-Symbol Base Database")
    logger.info("=" * 60)

    # Expand database
    expanded = expand_pilot_to_500()

    # Save to file
    output_path = Path(__file__).parent / "symbol_database_base_500.json"
    with open(output_path, 'w') as f:
        json.dump(expanded, f, indent=2)

    logger.info(f"\nSaved {len(expanded)} symbols to {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATABASE READY FOR BATCH ENRICHMENT")
    logger.info("=" * 60)
    logger.info(f"Symbols: {len(expanded)}")
    logger.info(f"Categories: 10")
    logger.info(f"Next: python enrich_symbols_batch.py --input symbol_database_base_500.json")

if __name__ == "__main__":
    main()
