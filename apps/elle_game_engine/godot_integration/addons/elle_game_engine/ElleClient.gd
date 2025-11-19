## Godot client for Elle Game Engine API.
## Handles HTTP communication with the Elle service using HTTPRequest.
##
## Usage:
##   1. Add this script as an Autoload singleton in Project Settings
##   2. Configure base_url and other settings in the Inspector or via code
##   3. Connect to the action_received, error_occurred signals
##   4. Call get_npc_dialogue(), get_hint(), or get_action()
##
## Author: Elle Game Engine Team
## Date: 2025-11-16

extends Node
class_name ElleClient

# ============================================================================
# Signals
# ============================================================================

## Emitted when an action is successfully received from Elle
signal action_received(action: ElleModels.ElleGameAction)

## Emitted when an error occurs during API call
signal error_occurred(error_message: String)

## Emitted when health check completes
signal health_check_completed(is_healthy: bool)

# ============================================================================
# Configuration
# ============================================================================

## Base URL of the Elle Game Engine service
@export var base_url: String = "http://localhost:8000"

## Request timeout in seconds
@export var timeout_seconds: float = 10.0

## Number of retry attempts on network failure
@export var max_retries: int = 3

## Enable debug logging
@export var debug_mode: bool = false

# ============================================================================
# Private State
# ============================================================================

var _http_request: HTTPRequest = null
var _pending_callback: Callable
var _retry_count: int = 0
var _current_request_data: Dictionary = {}

# ============================================================================
# Lifecycle
# ============================================================================

func _ready():
	_log_debug("ElleClient initialized. Service URL: %s" % base_url)

	# Create HTTPRequest node
	_http_request = HTTPRequest.new()
	add_child(_http_request)

	# Connect signals
	_http_request.request_completed.connect(_on_request_completed)

# ============================================================================
# Public API Methods
# ============================================================================

## Get NPC dialogue based on player message and game state.
##
## @param npc_id: ID of the NPC to talk to
## @param scene_id: Current scene ID
## @param player_message: What the player said/did
## @param game_state: Optional full game state (if null, creates minimal state)
func get_npc_dialogue(
	npc_id: String,
	scene_id: String,
	player_message: String,
	game_state: ElleModels.GameStateSnapshot = null
):
	# Build minimal game state if not provided
	if game_state == null:
		game_state = _create_minimal_game_state(scene_id)

	# Create player intent for NPC dialogue
	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.TALK_TO_NPC,
		npc_id,
		player_message
	)

	await get_action(game_state, intent)


## Get a hint for the player when they're stuck.
##
## @param scene_id: Current scene ID
## @param player_state: Current player state
func get_hint(scene_id: String, player_state: ElleModels.PlayerState):
	var game_state = ElleModels.GameStateSnapshot.new(
		scene_id,
		player_state,
		ElleModels.WorldState.new("day")
	)

	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.REQUEST_HINT
	)

	await get_action(game_state, intent)


## Get world/environmental reaction based on game state.
##
## @param scene_id: Current scene ID
## @param world_state: Current world state
func get_world_reaction(scene_id: String, world_state: ElleModels.WorldState):
	var game_state = ElleModels.GameStateSnapshot.new(
		scene_id,
		ElleModels.PlayerState.new("Player", scene_id),
		world_state
	)

	var intent = ElleModels.PlayerIntent.new(
		ElleModels.PlayerIntentType.ENTER_SCENE
	)

	await get_action(game_state, intent)


## Get action from Elle based on full game state and player intent.
## This is the main API method - all others are convenience wrappers.
##
## @param game_state: Complete game state snapshot
## @param player_intent: What the player is doing
func get_action(game_state: ElleModels.GameStateSnapshot, player_intent: ElleModels.PlayerIntent):
	var request = ElleModels.GameActionRequest.new(game_state, player_intent)

	_retry_count = 0
	_current_request_data = request.to_dict()

	await _post_request("/elle/game/action", _current_request_data)


## Check if Elle service is healthy and reachable.
##
## Returns true if service is healthy, false otherwise.
func check_health():
	_retry_count = 0
	_current_request_data = {}

	await _get_request("/health")


# ============================================================================
# HTTP Helper Methods
# ============================================================================

func _post_request(endpoint: String, payload: Dictionary):
	var url = base_url + endpoint

	# Convert payload to JSON
	var json_payload = JSON.stringify(payload)

	_log_debug("POST %s\nPayload: %s" % [url, json_payload])

	# Configure request
	var headers = ["Content-Type: application/json"]

	# Send request
	var error = _http_request.request(
		url,
		headers,
		HTTPClient.METHOD_POST,
		json_payload
	)

	if error != OK:
		_log_error("Failed to send request: %s" % error)
		error_occurred.emit("Failed to send request")


func _get_request(endpoint: String):
	var url = base_url + endpoint

	_log_debug("GET %s" % url)

	# Send request
	var error = _http_request.request(url)

	if error != OK:
		_log_error("Failed to send request: %s" % error)
		error_occurred.emit("Failed to send request")


func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	_log_debug("Request completed. Result: %d, Response code: %d" % [result, response_code])

	# Check for network errors
	if result != HTTPRequest.RESULT_SUCCESS:
		_handle_request_failure("Network error: %d" % result)
		return

	# Check for HTTP errors
	if response_code < 200 or response_code >= 300:
		var error_text = body.get_string_from_utf8()
		_handle_request_failure("HTTP %d: %s" % [response_code, error_text])
		return

	# Parse response
	var response_text = body.get_string_from_utf8()
	_log_debug("Response: %s" % response_text)

	var json = JSON.new()
	var parse_result = json.parse(response_text)

	if parse_result != OK:
		_log_error("Failed to parse JSON response")
		error_occurred.emit("Invalid JSON response")
		return

	var data = json.data

	# Handle different response types
	if _current_request_data.is_empty():
		# Health check response
		var is_healthy = data.get("status", "") == "healthy"
		health_check_completed.emit(is_healthy)
	else:
		# Action response
		var action = ElleModels.ElleGameAction.from_dict(data)
		action_received.emit(action)


func _handle_request_failure(error_message: String):
	_log_debug("Attempt %d/%d failed: %s" % [_retry_count + 1, max_retries, error_message])

	_retry_count += 1

	# Check if we should retry
	if _retry_count < max_retries:
		# Exponential backoff: 100ms, 200ms, 400ms...
		var delay_ms = int(pow(2, _retry_count - 1) * 100)
		_log_debug("Retrying in %d ms..." % delay_ms)

		await get_tree().create_timer(delay_ms / 1000.0).timeout

		# Retry the request
		if _current_request_data.is_empty():
			# Retry health check
			await _get_request("/health")
		else:
			# Retry action request
			await _post_request("/elle/game/action", _current_request_data)
	else:
		# All retries failed
		_log_error("Request failed after %d attempts" % max_retries)
		error_occurred.emit(error_message)


# ============================================================================
# Helper Methods
# ============================================================================

func _create_minimal_game_state(scene_id: String) -> ElleModels.GameStateSnapshot:
	return ElleModels.GameStateSnapshot.new(
		scene_id,
		ElleModels.PlayerState.new("Player", scene_id),
		ElleModels.WorldState.new("day")
	)


func _log_debug(message: String):
	if debug_mode:
		print("[ElleClient] %s" % message)


func _log_error(message: String):
	push_error("[ElleClient] %s" % message)


# ============================================================================
# Convenience Methods for Common Use Cases
# ============================================================================

## Quick dialogue query - minimal setup required
##
## @param npc_name: Name of the NPC
## @param player_message: What the player said
## @param scene: Scene name (defaults to "default_scene")
func quick_dialogue(npc_name: String, player_message: String, scene: String = "default_scene"):
	# Create minimal NPC
	var npc = ElleModels.NPCState.new(
		npc_name.to_lower().replace(" ", "_"),
		npc_name,
		"npc",
		scene
	)

	# Create game state with this NPC
	var game_state = _create_minimal_game_state(scene)
	game_state.npcs.append(npc)

	# Get dialogue
	await get_npc_dialogue(npc.id, scene, player_message, game_state)


## Play audio from action response (if available)
##
## @param action: The ElleGameAction with audio data
## @param audio_player: AudioStreamPlayer node to play the audio
func play_action_audio(action: ElleModels.ElleGameAction, audio_player: AudioStreamPlayer):
	if not action.has_audio():
		_log_debug("No audio available for this action")
		return

	# If audio_url is provided, we need to download it
	if action.audio_url != "":
		_log_debug("Downloading audio from: %s" % action.audio_url)
		# TODO: Implement audio download
		push_warning("Audio URL download not yet implemented")
		return

	# If audio_data is provided, play it directly
	if action.audio_data.size() > 0:
		_log_debug("Playing audio data (%d bytes)" % action.audio_data.size())

		# Create audio stream from bytes
		# Note: This assumes the audio is in a format Godot can play (WAV, OGG, MP3)
		var stream = AudioStreamOggVorbis.new()
		stream.data = action.audio_data

		audio_player.stream = stream
		audio_player.play()
