## Simple Godot Demo for Elle Game Engine
##
## This demo shows the simplest possible Elle integration - a single NPC
## conversation with voice synthesis support.
##
## Setup:
##   1. Copy addons/elle_game_engine/ to your Godot project
##   2. Enable the Elle plugin in Project Settings
##   3. Make sure Elle service is running (http://localhost:8000)
##   4. Attach this script to a Node in your scene
##   5. Add child nodes: ElleClient, RichTextLabel, AudioStreamPlayer
##
## Author: Elle Game Engine Team
## Date: 2025-11-16

extends Node

# ============================================================================
# Node References (assign in Inspector or create in code)
# ============================================================================

@export var dialogue_label: RichTextLabel
@export var voice_player: AudioStreamPlayer

var elle_client: ElleClient

# ============================================================================
# Lifecycle
# ============================================================================

func _ready():
	print("=== Elle Game Engine - Simple Demo ===")

	# Create ElleClient if not assigned
	if elle_client == null:
		elle_client = ElleClient.new()
		elle_client.base_url = "http://localhost:8000"
		elle_client.debug_mode = true
		add_child(elle_client)

	# Connect signals
	elle_client.action_received.connect(_on_action_received)
	elle_client.error_occurred.connect(_on_error_occurred)

	# Create UI if not assigned
	if dialogue_label == null:
		dialogue_label = RichTextLabel.new()
		add_child(dialogue_label)

	if voice_player == null:
		voice_player = AudioStreamPlayer.new()
		add_child(voice_player)

	# Display instructions
	display_message("[b]Elle Game Engine Demo[/b]\n\n")
	display_message("Press SPACE to talk to the innkeeper\n")
	display_message("Press H to ask for a hint\n")
	display_message("Press Q to quit\n\n")

	# Check health
	print("Checking Elle service health...")
	await elle_client.check_health()


func _process(_delta):
	# Simple keyboard controls
	if Input.is_action_just_pressed("ui_accept"):  # Space
		talk_to_innkeeper()

	if Input.is_key_pressed(KEY_H):
		request_hint()

	if Input.is_key_pressed(KEY_Q):
		get_tree().quit()


# ============================================================================
# Elle Integration Functions
# ============================================================================

func talk_to_innkeeper():
	display_message("\n[color=yellow]You approach the innkeeper...[/color]\n")

	# Simple approach using quick_dialogue
	await elle_client.quick_dialogue(
		"Bob the Innkeeper",
		"Hello! Do you have any quests for me?"
	)


func request_hint():
	display_message("\n[color=yellow]You look around for clues...[/color]\n")

	# Create minimal player state
	var player = ElleModels.PlayerState.new("Adventurer", "village_inn")
	player.quest_stage = "intro"

	await elle_client.get_hint("village_inn", player)


# ============================================================================
# Signal Handlers
# ============================================================================

func _on_action_received(action: ElleModels.ElleGameAction):
	print("[Demo] Action received: %s" % action.mode)

	# Handle dialogue
	if action.has_dialogue():
		for line in action.dialogue:
			var npc_name = line.npc_id.capitalize()
			var tone_emoji = line.get_tone_emoji()

			display_message("[b]%s[/b] %s: \"%s\"\n" % [npc_name, tone_emoji, line.text])

	# Handle hints
	if action.hint_text and action.hint_text != "":
		display_message("[color=cyan]💡 Hint:[/color] %s\n" % action.hint_text)

	# Play voice audio if available
	if action.has_audio() and voice_player:
		print("[Demo] Playing voice audio...")
		elle_client.play_action_audio(action, voice_player)


func _on_error_occurred(error_message: String):
	display_message("[color=red]ERROR:[/color] %s\n" % error_message)
	print("[Demo] Elle error: %s" % error_message)


# ============================================================================
# Helper Functions
# ============================================================================

func display_message(message: String):
	if dialogue_label:
		dialogue_label.append_text(message)
		# Auto-scroll to bottom
		dialogue_label.scroll_to_line(dialogue_label.get_line_count())
	else:
		print(message)


# ============================================================================
# Alternative Example: Complete Game State
# ============================================================================

func talk_to_innkeeper_advanced():
	"""
	Example of building complete game state for more contextual responses.
	"""

	# Create NPCs
	var innkeeper = ElleModels.NPCState.new(
		"innkeeper",
		"Bob the Innkeeper",
		"merchant",
		"village_inn",
		"neutral"
	)
	innkeeper.flags["has_quest"] = true
	innkeeper.flags["knows_player"] = false

	# Create player
	var player = ElleModels.PlayerState.new("Adventurer", "village_inn")
	player.quest_stage = "intro"
	player.reputation = "outsider"
	player.traits["brave"] = 3
	player.traits["diplomatic"] = 2

	# Create world
	var world = ElleModels.WorldState.new("afternoon")
	world.weather = "clear"
	world.tension_level = "calm"

	# Create game state
	var game_state = ElleModels.GameStateSnapshot.new(
		"village_inn",
		player,
		world
	)
	game_state.npcs.append(innkeeper)
	game_state.tags.append("first_visit")

	# Create player intent
	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.TALK_TO_NPC,
		"innkeeper",
		"Do you have any work for an adventurer?"
	)

	# Get action
	await elle_client.get_action(game_state, intent)
