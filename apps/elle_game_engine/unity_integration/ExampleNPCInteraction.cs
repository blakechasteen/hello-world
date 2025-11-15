using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;  // If using TextMeshPro (recommended), otherwise use UnityEngine.UI.Text

namespace Elle.GameEngine.Examples
{
    /// <summary>
    /// Example implementation showing how to integrate Elle Game Engine
    /// for NPC dialogue in Unity.
    ///
    /// Usage:
    /// 1. Attach to a GameObject in your scene
    /// 2. Assign UI elements in the Inspector
    /// 3. Make sure ElleClient is attached to same or another GameObject
    /// 4. Click on NPC to trigger dialogue
    /// </summary>
    [RequireComponent(typeof(ElleClient))]
    public class ExampleNPCInteraction : MonoBehaviour
    {
        [Header("NPC Configuration")]
        [SerializeField]
        private string npcId = "innkeeper";

        [SerializeField]
        private string npcName = "Bob the Innkeeper";

        [SerializeField]
        private string npcRole = "innkeeper";

        [SerializeField]
        private string npcLocation = "village_inn";

        [Header("Scene Configuration")]
        [SerializeField]
        private string sceneId = "village_square";

        [Header("UI References")]
        [Tooltip("Panel to show/hide dialogue UI")]
        [SerializeField]
        private GameObject dialoguePanel;

        [Tooltip("Text element for NPC name")]
        [SerializeField]
        private TextMeshProUGUI npcNameText;  // Or: Text if not using TextMeshPro

        [Tooltip("Text element for dialogue content")]
        [SerializeField]
        private TextMeshProUGUI dialogueText;

        [Tooltip("Text element for tone emoji")]
        [SerializeField]
        private TextMeshProUGUI toneEmoji;

        [Tooltip("Input field for player message")]
        [SerializeField]
        private TMP_InputField playerInputField;  // Or: InputField

        [Tooltip("Button to send message")]
        [SerializeField]
        private Button sendButton;

        [Tooltip("Loading indicator (optional)")]
        [SerializeField]
        private GameObject loadingIndicator;

        [Header("Player Configuration")]
        [SerializeField]
        private string playerName = "Hero";

        private ElleClient elleClient;
        private bool isProcessing = false;

        // ============================================================================
        // Unity Lifecycle
        // ============================================================================

        private void Awake()
        {
            elleClient = GetComponent<ElleClient>();

            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(false);
            }

            if (loadingIndicator != null)
            {
                loadingIndicator.SetActive(false);
            }
        }

        private void Start()
        {
            // Hook up send button
            if (sendButton != null)
            {
                sendButton.onClick.AddListener(OnSendMessage);
            }

            // Also allow Enter key to send
            if (playerInputField != null)
            {
                playerInputField.onSubmit.AddListener(_ => OnSendMessage());
            }
        }

        private void OnMouseDown()
        {
            // When player clicks this NPC, start conversation
            StartConversation();
        }

        // ============================================================================
        // Public API
        // ============================================================================

        /// <summary>
        /// Start a conversation with this NPC.
        /// </summary>
        public void StartConversation()
        {
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(true);
            }

            // Show initial greeting
            SendPlayerMessage("Hello!");
        }

        /// <summary>
        /// End the conversation and hide UI.
        /// </summary>
        public void EndConversation()
        {
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(false);
            }

            if (playerInputField != null)
            {
                playerInputField.text = "";
            }
        }

        // ============================================================================
        // Event Handlers
        // ============================================================================

        private void OnSendMessage()
        {
            if (playerInputField != null && !string.IsNullOrEmpty(playerInputField.text))
            {
                SendPlayerMessage(playerInputField.text);
                playerInputField.text = "";
            }
        }

        // ============================================================================
        // Core Logic
        // ============================================================================

        private async void SendPlayerMessage(string message)
        {
            if (isProcessing)
            {
                Debug.LogWarning("Already processing a request");
                return;
            }

            isProcessing = true;
            SetLoadingState(true);

            try
            {
                // Build complete game state
                var gameState = BuildGameState();

                // Get dialogue from Elle
                var action = await elleClient.GetNPCDialogue(
                    npcId: npcId,
                    sceneId: sceneId,
                    playerMessage: message,
                    gameState: gameState
                );

                // Display the response
                DisplayAction(action);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to get dialogue: {e.Message}");
                DisplayError("Failed to connect to Elle service. Is it running?");
            }
            finally
            {
                isProcessing = false;
                SetLoadingState(false);
            }
        }

        // ============================================================================
        // Game State Building
        // ============================================================================

        private GameStateSnapshot BuildGameState()
        {
            // Create NPC state
            var npc = new NPCState(
                id: npcId,
                name: npcName,
                role: npcRole,
                location: npcLocation,
                mood: "neutral"
            );

            // Create player state
            var player = new PlayerState(playerName, sceneId)
            {
                quest_stage = "intro",
                reputation = "neutral"
            };

            // Add some example traits
            player.traits.Add("brave", 2);
            player.traits.Add("kind", 3);

            // Add some example inventory items
            player.inventory_tags.Add("healing_potion");
            player.inventory_tags.Add("old_map");

            // Create world state
            var world = new WorldState("afternoon")
            {
                weather = "clear",
                tension_level = "calm"
            };

            // Build complete snapshot
            var gameState = new GameStateSnapshot(sceneId, player, world);
            gameState.npcs.Add(npc);

            // Add scene tags
            gameState.tags.Add("first_visit");

            return gameState;
        }

        // ============================================================================
        // UI Display
        // ============================================================================

        private void DisplayAction(ElleGameAction action)
        {
            if (action.HasDialogue && action.dialogue.Count > 0)
            {
                var dialogue = action.dialogue[0];  // Get first dialogue line

                // Update NPC name
                if (npcNameText != null)
                {
                    npcNameText.text = npcName;
                }

                // Update dialogue text
                if (dialogueText != null)
                {
                    dialogueText.text = dialogue.text;
                }

                // Update tone emoji
                if (toneEmoji != null)
                {
                    toneEmoji.text = dialogue.GetToneEmoji();
                }

                // Log debug info
                Debug.Log($"[Elle] {npcName} ({dialogue.tone ?? "neutral"}): {dialogue.text}");

                if (!string.IsNullOrEmpty(action.debug_notes))
                {
                    Debug.Log($"[Elle Debug] {action.debug_notes}");
                }
            }
            else if (!string.IsNullOrEmpty(action.hint_text))
            {
                // Display hint
                if (dialogueText != null)
                {
                    dialogueText.text = $"💡 {action.hint_text}";
                }
            }
            else
            {
                Debug.LogWarning("Received action with no displayable content");
            }
        }

        private void DisplayError(string errorMessage)
        {
            if (dialogueText != null)
            {
                dialogueText.text = $"⚠️ {errorMessage}";
            }
        }

        private void SetLoadingState(bool loading)
        {
            if (loadingIndicator != null)
            {
                loadingIndicator.SetActive(loading);
            }

            if (sendButton != null)
            {
                sendButton.interactable = !loading;
            }

            if (playerInputField != null)
            {
                playerInputField.interactable = !loading;
            }
        }

        // ============================================================================
        // Public Helper Methods (for other scripts)
        // ============================================================================

        /// <summary>
        /// Send a custom message to the NPC (callable from other scripts).
        /// </summary>
        public void TriggerDialogue(string message)
        {
            SendPlayerMessage(message);
        }

        /// <summary>
        /// Get a hint from Elle (callable from other scripts).
        /// </summary>
        public async void RequestHint()
        {
            if (isProcessing) return;

            isProcessing = true;
            SetLoadingState(true);

            try
            {
                var player = new PlayerState(playerName, sceneId);
                var action = await elleClient.GetHint(sceneId, player);
                DisplayAction(action);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to get hint: {e.Message}");
                DisplayError("Failed to get hint from Elle service.");
            }
            finally
            {
                isProcessing = false;
                SetLoadingState(false);
            }
        }
    }
}
