## Data models for Elle Game Engine Godot integration.
## These models match the Python API exactly for seamless interop.
##
## Author: Elle Game Engine Team
## Date: 2025-11-16

extends RefCounted
class_name ElleModels

# ============================================================================
# Game State Models
# ============================================================================

## State of a non-player character in the game world
class NPCState:
	## Unique identifier for the NPC (e.g. "innkeeper", "guard_1")
	var id: String

	## Display name of the NPC (e.g. "Bob", "Captain Morgan")
	var name: String

	## Role/type of NPC (e.g. "merchant", "guard", "quest_giver")
	var role: String

	## Current mood (optional): "neutral", "annoyed", "grateful", "curious", "hostile"
	var mood: String = ""

	## Current location (e.g. "village_square", "inn", "forest_edge")
	var location: String = ""

	## Custom world flags for this NPC
	var flags: Dictionary = {}

	func _init(p_id: String, p_name: String, p_role: String, p_location: String = "", p_mood: String = ""):
		id = p_id
		name = p_name
		role = p_role
		location = p_location
		mood = p_mood

	## Convert to dictionary for JSON serialization
	func to_dict() -> Dictionary:
		return {
			"id": id,
			"name": name,
			"role": role,
			"mood": mood if mood != "" else null,
			"location": location,
			"flags": flags
		}


## State of the player character
class PlayerState:
	## Player character name
	var name: String

	## Current player location
	var location: String

	## Quest progression stage (optional): "intro", "mid", "late", "climax"
	var quest_stage: String = ""

	## Player reputation (optional): "hero", "outsider", "villain", "neutral"
	var reputation: String = ""

	## Character traits (e.g. {"brave": 3, "kind": 2, "cunning": 1})
	var traits: Dictionary = {}

	## Inventory item tags (e.g. ["healing_herb", "ancient_key", "sword"])
	var inventory_tags: Array = []

	func _init(p_name: String, p_location: String):
		name = p_name
		location = p_location

	## Convert to dictionary for JSON serialization
	func to_dict() -> Dictionary:
		return {
			"name": name,
			"location": location,
			"quest_stage": quest_stage if quest_stage != "" else null,
			"reputation": reputation if reputation != "" else null,
			"traits": traits,
			"inventory_tags": inventory_tags
		}


## Ambient world conditions
class WorldState:
	## Time of day: "morning", "afternoon", "evening", "night", "dawn", "dusk"
	var time_of_day: String

	## Weather (optional): "clear", "rain", "storm", "fog", "snow"
	var weather: String = ""

	## Narrative tension level (optional): "calm", "uneasy", "tense", "high", "critical"
	var tension_level: String = ""

	func _init(p_time_of_day: String):
		time_of_day = p_time_of_day

	## Convert to dictionary for JSON serialization
	func to_dict() -> Dictionary:
		return {
			"time_of_day": time_of_day,
			"weather": weather if weather != "" else null,
			"tension_level": tension_level if tension_level != "" else null
		}


## Complete snapshot of game state for Elle's decision-making
class GameStateSnapshot:
	## Current scene identifier (e.g. "village_square", "dark_forest", "throne_room")
	var scene_id: String

	## All NPCs present in the scene
	var npcs: Array = []  # Array of NPCState

	## Player character state
	var player: PlayerState

	## World/ambient conditions
	var world: WorldState

	## Custom scene tags (e.g. ["post_battle", "festival", "first_visit"])
	var tags: Array = []

	func _init(p_scene_id: String, p_player: PlayerState, p_world: WorldState):
		scene_id = p_scene_id
		player = p_player
		world = p_world

	## Number of NPCs in the scene
	func get_npc_count() -> int:
		return npcs.size()

	## Find NPC by ID
	func get_npc_by_id(npc_id: String):
		for npc in npcs:
			if npc.id == npc_id:
				return npc
		return null

	## Check if snapshot has a specific tag
	func has_tag(tag: String) -> bool:
		return tag in tags

	## Convert to dictionary for JSON serialization
	func to_dict() -> Dictionary:
		var npc_dicts = []
		for npc in npcs:
			npc_dicts.append(npc.to_dict())

		return {
			"scene_id": scene_id,
			"npcs": npc_dicts,
			"player": player.to_dict(),
			"world": world.to_dict(),
			"tags": tags
		}


# ============================================================================
# Player Intent Models
# ============================================================================

## High-level player intent categories
enum PlayerIntentType {
	TALK_TO_NPC,     ## Player initiating dialogue with an NPC
	ENTER_SCENE,     ## Player entering a new scene/location
	REQUEST_HINT,    ## Player asking for guidance/hint
	DEBUG_SUMMARY    ## Developer requesting narrative summary (debug)
}

## What the player is doing / what we want from Elle
class PlayerIntent:
	## Type of player action/intent
	var type: PlayerIntentType

	## Target NPC ID (required for TALK_TO_NPC intent)
	var target_npc_id: String = ""

	## Player's raw dialogue text or paraphrased question
	var raw_input: String = ""

	func _init(p_type: PlayerIntentType, p_target_npc_id: String = "", p_raw_input: String = ""):
		type = p_type
		target_npc_id = p_target_npc_id
		raw_input = p_raw_input

	## Convert enum to string for API
	static func type_to_string(intent_type: PlayerIntentType) -> String:
		match intent_type:
			PlayerIntentType.TALK_TO_NPC:
				return "talk_to_npc"
			PlayerIntentType.ENTER_SCENE:
				return "enter_scene"
			PlayerIntentType.REQUEST_HINT:
				return "request_hint"
			PlayerIntentType.DEBUG_SUMMARY:
				return "debug_summary"
			_:
				return "talk_to_npc"

	## Convert to dictionary for JSON serialization
	func to_dict() -> Dictionary:
		return {
			"type": PlayerIntent.type_to_string(type),
			"target_npc_id": target_npc_id if target_npc_id != "" else null,
			"raw_input": raw_input if raw_input != "" else null
		}


# ============================================================================
# Action Response Models
# ============================================================================

## Type of action Elle is returning
enum ActionMode {
	NPC_DIALOGUE,    ## NPC speaking
	HINT,            ## Non-spoilery guidance
	WORLD_REACTION,  ## Environmental/world response
	DEV_DEBUG        ## Developer-facing narrative notes
}

## A line of NPC dialogue
class DialogueLine:
	## ID of NPC speaking this line
	var npc_id: String

	## The dialogue text
	var text: String

	## Emotional tone (optional): "warm", "stern", "cryptic", "excited", "sad"
	var tone: String = ""

	func _init(p_npc_id: String, p_text: String, p_tone: String = ""):
		npc_id = p_npc_id
		text = p_text
		tone = p_tone

	## Get emoji representation of tone for UI display
	func get_tone_emoji() -> String:
		if tone == "":
			return ""

		match tone.to_lower():
			"warm":
				return "😊"
			"stern":
				return "😠"
			"cryptic":
				return "🤔"
			"excited":
				return "😃"
			"sad":
				return "😢"
			"neutral":
				return "😐"
			"curious":
				return "🧐"
			"grateful":
				return "🙏"
			"annoyed":
				return "😒"
			"hostile":
				return "😡"
			_:
				return ""

	## Create from dictionary (JSON deserialization)
	static func from_dict(data: Dictionary) -> DialogueLine:
		return DialogueLine.new(
			data.get("npc_id", ""),
			data.get("text", ""),
			data.get("tone", "")
		)


## A change to the world state triggered by Elle's decision
class WorldChange:
	## Human-readable description of the change
	var description: String

	## Flags to set/unset in the world state
	var flag_changes: Dictionary = {}

	func _init(p_description: String):
		description = p_description

	## Create from dictionary (JSON deserialization)
	static func from_dict(data: Dictionary) -> WorldChange:
		var change = WorldChange.new(data.get("description", ""))
		change.flag_changes = data.get("flag_changes", {})
		return change


## Complete structured action Elle returns to the game
class ElleGameAction:
	## Type of action being returned
	var mode: ActionMode

	## Priority level: "low", "medium", "high"
	var priority: String

	## Dialogue lines (for NPC_DIALOGUE mode)
	var dialogue: Array = []  # Array of DialogueLine

	## Hint text (for HINT mode)
	var hint_text: String = ""

	## World reaction (for WORLD_REACTION mode)
	var world_reaction = null  # WorldChange or null

	## Developer notes (not shown to players)
	var debug_notes: String = ""

	## Audio URL or data for voice synthesis (optional)
	var audio_url: String = ""
	var audio_data: PackedByteArray = []

	func _init(p_mode: ActionMode, p_priority: String):
		mode = p_mode
		priority = p_priority

	## Check if this action contains dialogue
	func has_dialogue() -> bool:
		return dialogue.size() > 0

	## Check if this action changes the world
	func has_world_changes() -> bool:
		return world_reaction != null

	## Check if this action has audio
	func has_audio() -> bool:
		return audio_url != "" or audio_data.size() > 0

	## Convert string mode to enum
	static func mode_from_string(mode_str: String) -> ActionMode:
		match mode_str:
			"npc_dialogue":
				return ActionMode.NPC_DIALOGUE
			"hint":
				return ActionMode.HINT
			"world_reaction":
				return ActionMode.WORLD_REACTION
			"dev_debug":
				return ActionMode.DEV_DEBUG
			_:
				return ActionMode.NPC_DIALOGUE

	## Create from dictionary (JSON deserialization)
	static func from_dict(data: Dictionary) -> ElleGameAction:
		var action = ElleGameAction.new(
			mode_from_string(data.get("mode", "npc_dialogue")),
			data.get("priority", "medium")
		)

		# Parse dialogue
		if data.has("dialogue"):
			for line_data in data["dialogue"]:
				action.dialogue.append(DialogueLine.from_dict(line_data))

		# Parse other fields
		action.hint_text = data.get("hint_text", "")

		if data.has("world_reaction") and data["world_reaction"] != null:
			action.world_reaction = WorldChange.from_dict(data["world_reaction"])

		action.debug_notes = data.get("debug_notes", "")
		action.audio_url = data.get("audio_url", "")

		# Parse audio data if present
		if data.has("audio_data") and data["audio_data"] != null:
			# Convert base64 string to bytes if needed
			if data["audio_data"] is String:
				action.audio_data = Marshalls.base64_to_raw(data["audio_data"])
			elif data["audio_data"] is PackedByteArray:
				action.audio_data = data["audio_data"]

		return action


# ============================================================================
# Request/Response Wrappers (for JSON serialization)
# ============================================================================

## Complete request body for /elle/game/action endpoint
class GameActionRequest:
	var game_state: GameStateSnapshot
	var player_intent: PlayerIntent

	func _init(p_game_state: GameStateSnapshot, p_player_intent: PlayerIntent):
		game_state = p_game_state
		player_intent = p_player_intent

	## Convert to dictionary for JSON serialization
	func to_dict() -> Dictionary:
		return {
			"game_state": game_state.to_dict(),
			"player_intent": player_intent.to_dict()
		}
