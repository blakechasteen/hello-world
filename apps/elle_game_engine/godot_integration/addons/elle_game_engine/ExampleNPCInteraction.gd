## Example NPC interaction using Elle Game Engine.
##
## This example demonstrates:
## - Setting up ElleClient
## - Creating game state snapshots
## - Handling NPC dialogue
## - Playing synthesized voice audio
## - Error handling and retries
##
## Usage:
##   1. Attach this script to a Node in your scene
##   2. Add child nodes: ElleClient, RichTextLabel (for dialogue), AudioStreamPlayer (for voice)
##   3. Assign the nodes in the Inspector
##   4. Make sure Elle service is running (http://localhost:8000)
##   5. Run the scene and press the interact buttons
##
## Author: Elle Game Engine Team
## Date: 2025-11-16

extends Node

# ============================================================================
# Node References
# ============================================================================

## ElleClient node (create as child or reference from Autoload)
@export var elle_client: ElleClient

## UI node to display dialogue
@export var dialogue_label: RichTextLabel

## AudioStreamPlayer for voice synthesis
@export var voice_player: AudioStreamPlayer

## Button to trigger NPC greeting
@export var greet_button: Button

## Button to trigger quest dialogue
@export var quest_button: Button

## Button to trigger hint request
@export var hint_button: Button

# ============================================================================
# Game State
# ============================================================================

# Current scene info
var current_scene_id: String = "village_square"

# NPCs in the scene
var innkeeper: ElleModels.NPCState
var guard: ElleModels.NPCState

# Player state
var player: ElleModels.PlayerState

# World state
var world: ElleModels.WorldState

# ============================================================================
# Lifecycle
# ============================================================================

func _ready():
	# Initialize ElleClient if not set
	if elle_client == null:
		elle_client = ElleClient.new()
		elle_client.debug_mode = true
		add_child(elle_client)

	# Connect signals
	elle_client.action_received.connect(_on_action_received)
	elle_client.error_occurred.connect(_on_error_occurred)

	# Connect buttons if set
	if greet_button:
		greet_button.pressed.connect(_on_greet_button_pressed)
	if quest_button:
		quest_button.pressed.connect(_on_quest_button_pressed)
	if hint_button:
		hint_button.pressed.connect(_on_hint_button_pressed)

	# Initialize game state
	_initialize_game_state()

	# Display initial message
	_display_message("[b]Elle Game Engine - Example NPC Interaction[/b]\n\n" +
		"Press buttons to interact with NPCs.\n" +
		"Make sure Elle service is running at: %s\n" % elle_client.base_url)

	# Check health
	await elle_client.check_health()


func _initialize_game_state():
	# Create NPCs
	innkeeper = ElleModels.NPCState.new(
		"innkeeper",
		"Bob the Innkeeper",
		"merchant",
		current_scene_id,
		"neutral"
	)
	innkeeper.flags["knows_player"] = false
	innkeeper.flags["quest_available"] = true

	guard = ElleModels.NPCState.new(
		"guard_captain",
		"Captain Morgan",
		"guard",
		current_scene_id,
		"stern"
	)
	guard.flags["on_duty"] = true

	# Create player
	player = ElleModels.PlayerState.new("Adventurer", current_scene_id)
	player.quest_stage = "intro"
	player.reputation = "outsider"
	player.traits["brave"] = 3
	player.traits["diplomatic"] = 2
	player.inventory_tags = ["healing_herb", "rusty_sword"]

	# Create world
	world = ElleModels.WorldState.new("afternoon")
	world.weather = "clear"
	world.tension_level = "calm"


# ============================================================================
# Button Handlers
# ============================================================================

func _on_greet_button_pressed():
	_display_message("[color=yellow]You approach the innkeeper...[/color]\n")

	# Create game state snapshot
	var game_state = _create_game_state([innkeeper])

	# Create player intent
	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.TALK_TO_NPC,
		innkeeper.id,
		"Hello! I'm new in town."
	)

	# Get action from Elle
	await elle_client.get_action(game_state, intent)


func _on_quest_button_pressed():
	_display_message("[color=yellow]You ask the innkeeper about work...[/color]\n")

	# Update game state
	innkeeper.flags["knows_player"] = true

	var game_state = _create_game_state([innkeeper])

	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.TALK_TO_NPC,
		innkeeper.id,
		"Do you have any work for an adventurer?"
	)

	await elle_client.get_action(game_state, intent)


func _on_hint_button_pressed():
	_display_message("[color=yellow]You look around for clues...[/color]\n")

	var game_state = _create_game_state([innkeeper, guard])

	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.REQUEST_HINT
	)

	await elle_client.get_action(game_state, intent)


# ============================================================================
# Signal Handlers
# ============================================================================

func _on_action_received(action: ElleModels.ElleGameAction):
	print("[Example] Action received: mode=%s, priority=%s" % [action.mode, action.priority])

	# Handle different action modes
	match action.mode:
		ElleModels.ActionMode.NPC_DIALOGUE:
			_handle_dialogue(action)

		ElleModels.ActionMode.HINT:
			_handle_hint(action)

		ElleModels.ActionMode.WORLD_REACTION:
			_handle_world_reaction(action)

		ElleModels.ActionMode.DEV_DEBUG:
			_display_message("[color=gray]DEBUG: %s[/color]\n" % action.debug_notes)

	# Play voice audio if available
	if action.has_audio() and voice_player:
		elle_client.play_action_audio(action, voice_player)


func _on_error_occurred(error_message: String):
	_display_message("[color=red]ERROR: %s[/color]\n" % error_message)
	push_error("Elle API error: %s" % error_message)


# ============================================================================
# Action Handlers
# ============================================================================

func _handle_dialogue(action: ElleModels.ElleGameAction):
	for line in action.dialogue:
		var npc = _get_npc_by_id(line.npc_id)
		var npc_name = npc.name if npc else line.npc_id

		# Get tone emoji
		var emoji = line.get_tone_emoji()
		var tone_text = " [%s %s]" % [emoji, line.tone] if line.tone != "" else ""

		# Display dialogue
		_display_message("[b]%s[/b]%s: \"%s\"\n" % [npc_name, tone_text, line.text])

	# Handle world changes
	if action.has_world_changes():
		_handle_world_reaction(action)


func _handle_hint(action: ElleModels.ElleGameAction):
	_display_message("[color=cyan]💡 Hint:[/color] %s\n" % action.hint_text)


func _handle_world_reaction(action: ElleModels.ElleGameAction):
	if not action.world_reaction:
		return

	_display_message("[color=green]🌍 %s[/color]\n" % action.world_reaction.description)

	# Apply flag changes
	for flag_name in action.world_reaction.flag_changes:
		var flag_value = action.world_reaction.flag_changes[flag_name]
		print("[Example] Flag changed: %s = %s" % [flag_name, flag_value])

		# Update NPC flags (simplified - real game would be more complex)
		if innkeeper and innkeeper.flags.has(flag_name):
			innkeeper.flags[flag_name] = flag_value
		if guard and guard.flags.has(flag_name):
			guard.flags[flag_name] = flag_value


# ============================================================================
# Helper Methods
# ============================================================================

func _create_game_state(npcs_in_scene: Array) -> ElleModels.GameStateSnapshot:
	var game_state = ElleModels.GameStateSnapshot.new(
		current_scene_id,
		player,
		world
	)

	game_state.npcs = npcs_in_scene.duplicate()

	# Add scene tags based on game state
	if player.quest_stage == "intro":
		game_state.tags.append("first_visit")

	return game_state


func _get_npc_by_id(npc_id: String) -> ElleModels.NPCState:
	if innkeeper and innkeeper.id == npc_id:
		return innkeeper
	if guard and guard.id == npc_id:
		return guard
	return null


func _display_message(message: String):
	if dialogue_label:
		dialogue_label.append_text(message)
		# Auto-scroll to bottom
		dialogue_label.scroll_to_line(dialogue_label.get_line_count())
	else:
		print(message)


# ============================================================================
# Advanced Example: Conversation Flow
# ============================================================================

## Example of a multi-turn conversation with state tracking
func start_quest_conversation():
	_display_message("\n[b]=== Starting Quest Conversation ===[/b]\n")

	# Turn 1: Greet
	await _conversation_turn(
		innkeeper.id,
		"Hello!",
		"Player greets innkeeper"
	)

	# Wait a bit
	await get_tree().create_timer(1.0).timeout

	# Turn 2: Ask about quest
	innkeeper.flags["knows_player"] = true
	await _conversation_turn(
		innkeeper.id,
		"I heard you might have work for adventurers?",
		"Player asks about quest"
	)

	# Wait a bit
	await get_tree().create_timer(1.0).timeout

	# Turn 3: Accept quest
	innkeeper.flags["quest_offered"] = true
	await _conversation_turn(
		innkeeper.id,
		"I'll help you with the rats in the cellar!",
		"Player accepts quest"
	)

	_display_message("\n[b]=== Quest Accepted! ===[/b]\n")


func _conversation_turn(npc_id: String, player_message: String, description: String):
	_display_message("[color=yellow]%s[/color]\n" % description)

	var game_state = _create_game_state([innkeeper])
	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.TALK_TO_NPC,
		npc_id,
		player_message
	)

	await elle_client.get_action(game_state, intent)


# ============================================================================
# Advanced Example: Voice Synthesis Integration
# ============================================================================

## Example of requesting dialogue with voice synthesis
func get_dialogue_with_voice(npc_id: String, player_message: String):
	_display_message("[color=yellow]Requesting dialogue with voice...[/color]\n")

	# Note: Voice synthesis integration would require the /voice/synthesize endpoint
	# For now, this demonstrates the workflow

	# Step 1: Get dialogue from Elle
	var game_state = _create_game_state([innkeeper])
	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.TALK_TO_NPC,
		npc_id,
		player_message
	)

	await elle_client.get_action(game_state, intent)

	# Step 2: Voice synthesis would be handled automatically if enabled in Elle service
	# The action.audio_data or action.audio_url would be populated
	# The _on_action_received handler would play it automatically


# ============================================================================
# Test Functions (call from console or buttons)
# ============================================================================

## Test health check
func test_health():
	print("[Example] Testing health check...")
	await elle_client.check_health()


## Test quick dialogue helper
func test_quick_dialogue():
	print("[Example] Testing quick dialogue...")
	await elle_client.quick_dialogue("Bob", "Tell me a joke!")


## Test error handling
func test_error_handling():
	print("[Example] Testing error handling (invalid URL)...")
	var old_url = elle_client.base_url
	elle_client.base_url = "http://invalid-url:9999"

	await elle_client.check_health()

	# Restore URL
	elle_client.base_url = old_url
