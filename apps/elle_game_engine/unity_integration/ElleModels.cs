using System;
using System.Collections.Generic;
using UnityEngine;

namespace Elle.GameEngine
{
    /// <summary>
    /// Data models for Elle Game Engine Unity integration.
    /// These models match the Python API exactly for seamless interop.
    /// </summary>

    // ============================================================================
    // Game State Models
    // ============================================================================

    [Serializable]
    public class NPCState
    {
        /// <summary>Unique identifier for the NPC (e.g. "innkeeper", "guard_1")</summary>
        public string id;

        /// <summary>Display name of the NPC (e.g. "Bob", "Captain Morgan")</summary>
        public string name;

        /// <summary>Role/type of NPC (e.g. "merchant", "guard", "quest_giver")</summary>
        public string role;

        /// <summary>Current mood (optional): "neutral", "annoyed", "grateful", "curious", "hostile"</summary>
        public string mood;

        /// <summary>Current location (e.g. "village_square", "inn", "forest_edge")</summary>
        public string location;

        /// <summary>Custom world flags for this NPC</summary>
        public Dictionary<string, bool> flags = new Dictionary<string, bool>();

        public NPCState(string id, string name, string role, string location = "", string mood = null)
        {
            this.id = id;
            this.name = name;
            this.role = role;
            this.location = location;
            this.mood = mood;
        }
    }

    [Serializable]
    public class PlayerState
    {
        /// <summary>Player character name</summary>
        public string name;

        /// <summary>Current player location</summary>
        public string location;

        /// <summary>Quest progression stage (optional): "intro", "mid", "late", "climax"</summary>
        public string quest_stage;

        /// <summary>Player reputation (optional): "hero", "outsider", "villain", "neutral"</summary>
        public string reputation;

        /// <summary>Character traits (e.g. {"brave": 3, "kind": 2, "cunning": 1})</summary>
        public Dictionary<string, int> traits = new Dictionary<string, int>();

        /// <summary>Inventory item tags (e.g. ["healing_herb", "ancient_key", "sword"])</summary>
        public List<string> inventory_tags = new List<string>();

        public PlayerState(string name, string location)
        {
            this.name = name;
            this.location = location;
        }
    }

    [Serializable]
    public class WorldState
    {
        /// <summary>Time of day: "morning", "afternoon", "evening", "night", "dawn", "dusk"</summary>
        public string time_of_day;

        /// <summary>Weather (optional): "clear", "rain", "storm", "fog", "snow"</summary>
        public string weather;

        /// <summary>Narrative tension level (optional): "calm", "uneasy", "tense", "high", "critical"</summary>
        public string tension_level;

        public WorldState(string time_of_day)
        {
            this.time_of_day = time_of_day;
        }
    }

    [Serializable]
    public class GameStateSnapshot
    {
        /// <summary>Current scene identifier (e.g. "village_square", "dark_forest", "throne_room")</summary>
        public string scene_id;

        /// <summary>All NPCs present in the scene</summary>
        public List<NPCState> npcs = new List<NPCState>();

        /// <summary>Player character state</summary>
        public PlayerState player;

        /// <summary>World/ambient conditions</summary>
        public WorldState world;

        /// <summary>Custom scene tags (e.g. ["post_battle", "festival", "first_visit"])</summary>
        public List<string> tags = new List<string>();

        public GameStateSnapshot(string scene_id, PlayerState player, WorldState world)
        {
            this.scene_id = scene_id;
            this.player = player;
            this.world = world;
        }
    }

    // ============================================================================
    // Player Intent Models
    // ============================================================================

    /// <summary>
    /// High-level player intent categories.
    /// </summary>
    public enum PlayerIntentType
    {
        /// <summary>Player initiating dialogue with an NPC</summary>
        talk_to_npc,

        /// <summary>Player entering a new scene/location</summary>
        enter_scene,

        /// <summary>Player asking for guidance/hint</summary>
        request_hint,

        /// <summary>Developer requesting narrative summary (debug)</summary>
        debug_summary
    }

    [Serializable]
    public class PlayerIntent
    {
        /// <summary>Type of player action/intent</summary>
        public string type;  // Will be serialized to match PlayerIntentType enum

        /// <summary>Target NPC ID (required for talk_to_npc intent)</summary>
        public string target_npc_id;

        /// <summary>Player's raw dialogue text or paraphrased question</summary>
        public string raw_input;

        public PlayerIntent(PlayerIntentType type, string target_npc_id = null, string raw_input = null)
        {
            this.type = type.ToString();
            this.target_npc_id = target_npc_id;
            this.raw_input = raw_input;
        }
    }

    // ============================================================================
    // Action Response Models
    // ============================================================================

    /// <summary>
    /// Type of action Elle is returning.
    /// </summary>
    public enum ActionMode
    {
        /// <summary>NPC speaking</summary>
        npc_dialogue,

        /// <summary>Non-spoilery guidance</summary>
        hint,

        /// <summary>Environmental/world response</summary>
        world_reaction,

        /// <summary>Developer-facing narrative notes</summary>
        dev_debug
    }

    [Serializable]
    public class DialogueLine
    {
        /// <summary>ID of NPC speaking this line</summary>
        public string npc_id;

        /// <summary>The dialogue text</summary>
        public string text;

        /// <summary>Emotional tone (optional): "warm", "stern", "cryptic", "excited", "sad"</summary>
        public string tone;

        /// <summary>Get emoji representation of tone for UI display</summary>
        public string GetToneEmoji()
        {
            if (string.IsNullOrEmpty(tone)) return "";

            return tone.ToLower() switch
            {
                "warm" => "😊",
                "stern" => "😠",
                "cryptic" => "🤔",
                "excited" => "😃",
                "sad" => "😢",
                "neutral" => "😐",
                "curious" => "🧐",
                "grateful" => "🙏",
                "annoyed" => "😒",
                "hostile" => "😡",
                _ => ""
            };
        }
    }

    [Serializable]
    public class WorldChange
    {
        /// <summary>Human-readable description of the change</summary>
        public string description;

        /// <summary>Flags to set/unset in the world state</summary>
        public Dictionary<string, bool> flag_changes = new Dictionary<string, bool>();
    }

    [Serializable]
    public class ElleGameAction
    {
        /// <summary>Type of action being returned</summary>
        public string mode;  // Will match ActionMode enum

        /// <summary>Priority level: "low", "medium", "high"</summary>
        public string priority;

        /// <summary>Dialogue lines (for npc_dialogue mode)</summary>
        public List<DialogueLine> dialogue = new List<DialogueLine>();

        /// <summary>Hint text (for hint mode)</summary>
        public string hint_text;

        /// <summary>World reaction (for world_reaction mode)</summary>
        public WorldChange world_reaction;

        /// <summary>Developer notes (not shown to players)</summary>
        public string debug_notes;

        /// <summary>Check if this action contains dialogue</summary>
        public bool HasDialogue => dialogue != null && dialogue.Count > 0;

        /// <summary>Check if this action changes the world</summary>
        public bool HasWorldChanges => world_reaction != null;

        /// <summary>Get the action mode as enum</summary>
        public ActionMode GetMode()
        {
            return Enum.TryParse<ActionMode>(mode, out var result) ? result : ActionMode.npc_dialogue;
        }
    }

    // ============================================================================
    // Request/Response Wrappers (for JSON serialization)
    // ============================================================================

    [Serializable]
    internal class GameActionRequest
    {
        public GameStateSnapshot game_state;
        public PlayerIntent player_intent;

        public GameActionRequest(GameStateSnapshot game_state, PlayerIntent player_intent)
        {
            this.game_state = game_state;
            this.player_intent = player_intent;
        }
    }
}
