using System;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace Elle.GameEngine
{
    /// <summary>
    /// Unity client for Elle Game Engine API.
    /// Handles HTTP communication with the Elle service using UnityWebRequest.
    /// </summary>
    public class ElleClient : MonoBehaviour
    {
        [Header("Configuration")]
        [Tooltip("Base URL of the Elle Game Engine service")]
        [SerializeField]
        private string baseUrl = "http://localhost:8000";

        [Tooltip("Request timeout in seconds")]
        [SerializeField]
        private int timeoutSeconds = 10;

        [Tooltip("Number of retry attempts on network failure")]
        [SerializeField]
        private int maxRetries = 3;

        [Tooltip("Enable debug logging")]
        [SerializeField]
        private bool debugMode = false;

        private string ActionEndpoint => $"{baseUrl}/elle/game/action";
        private string HealthEndpoint => $"{baseUrl}/health";

        // ============================================================================
        // Public API Methods
        // ============================================================================

        /// <summary>
        /// Get NPC dialogue based on player message and game state.
        /// </summary>
        /// <param name="npcId">ID of the NPC to talk to</param>
        /// <param name="sceneId">Current scene ID</param>
        /// <param name="playerMessage">What the player said/did</param>
        /// <param name="gameState">Optional full game state (if null, creates minimal state)</param>
        /// <returns>Elle's response action</returns>
        public async Task<ElleGameAction> GetNPCDialogue(
            string npcId,
            string sceneId,
            string playerMessage,
            GameStateSnapshot gameState = null)
        {
            // Build minimal game state if not provided
            if (gameState == null)
            {
                gameState = CreateMinimalGameState(sceneId);
            }

            // Create player intent for NPC dialogue
            var intent = new PlayerIntent(
                PlayerIntentType.talk_to_npc,
                target_npc_id: npcId,
                raw_input: playerMessage
            );

            return await GetAction(gameState, intent);
        }

        /// <summary>
        /// Get a hint for the player when they're stuck.
        /// </summary>
        /// <param name="sceneId">Current scene ID</param>
        /// <param name="playerState">Current player state</param>
        /// <returns>Elle's hint response</returns>
        public async Task<ElleGameAction> GetHint(string sceneId, PlayerState playerState)
        {
            var gameState = new GameStateSnapshot(
                scene_id: sceneId,
                player: playerState,
                world: new WorldState("day")
            );

            var intent = new PlayerIntent(PlayerIntentType.request_hint);

            return await GetAction(gameState, intent);
        }

        /// <summary>
        /// Get world/environmental reaction based on game state.
        /// </summary>
        /// <param name="sceneId">Current scene ID</param>
        /// <param name="worldState">Current world state</param>
        /// <returns>Elle's world reaction</returns>
        public async Task<ElleGameAction> GetWorldReaction(string sceneId, WorldState worldState)
        {
            var gameState = new GameStateSnapshot(
                scene_id: sceneId,
                player: new PlayerState("Player", sceneId),
                world: worldState
            );

            var intent = new PlayerIntent(PlayerIntentType.enter_scene);

            return await GetAction(gameState, intent);
        }

        /// <summary>
        /// Get action from Elle based on full game state and player intent.
        /// This is the main API method - all others are convenience wrappers.
        /// </summary>
        /// <param name="gameState">Complete game state snapshot</param>
        /// <param name="playerIntent">What the player is doing</param>
        /// <returns>Elle's response action</returns>
        public async Task<ElleGameAction> GetAction(GameStateSnapshot gameState, PlayerIntent playerIntent)
        {
            var request = new GameActionRequest(gameState, playerIntent);
            return await PostRequest<ElleGameAction>(ActionEndpoint, request);
        }

        /// <summary>
        /// Check if Elle service is healthy and reachable.
        /// </summary>
        /// <returns>True if service is healthy, false otherwise</returns>
        public async Task<bool> CheckHealth()
        {
            try
            {
                await GetRequest<HealthResponse>(HealthEndpoint);
                return true;
            }
            catch (Exception e)
            {
                LogError($"Health check failed: {e.Message}");
                return false;
            }
        }

        // ============================================================================
        // HTTP Helper Methods
        // ============================================================================

        private async Task<T> PostRequest<T>(string url, object payload)
        {
            string json = JsonUtility.ToJson(payload);
            LogDebug($"POST {url}\nPayload: {json}");

            Exception lastException = null;

            for (int attempt = 0; attempt < maxRetries; attempt++)
            {
                try
                {
                    using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
                    {
                        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                        request.downloadHandler = new DownloadHandlerBuffer();
                        request.SetRequestHeader("Content-Type", "application/json");
                        request.timeout = timeoutSeconds;

                        var operation = request.SendWebRequest();

                        // Await the operation
                        while (!operation.isDone)
                        {
                            await Task.Yield();
                        }

                        if (request.result == UnityWebRequest.Result.Success)
                        {
                            string responseJson = request.downloadHandler.text;
                            LogDebug($"Response: {responseJson}");

                            T result = JsonUtility.FromJson<T>(responseJson);
                            return result;
                        }
                        else
                        {
                            lastException = new Exception(
                                $"HTTP {request.responseCode}: {request.error}\n{request.downloadHandler.text}"
                            );

                            if (request.responseCode >= 400 && request.responseCode < 500)
                            {
                                // Client errors (4xx) - don't retry
                                throw lastException;
                            }

                            LogDebug($"Attempt {attempt + 1}/{maxRetries} failed: {lastException.Message}");
                        }
                    }
                }
                catch (Exception e)
                {
                    lastException = e;
                    LogDebug($"Attempt {attempt + 1}/{maxRetries} failed: {e.Message}");
                }

                // Wait before retry (exponential backoff)
                if (attempt < maxRetries - 1)
                {
                    int delayMs = (int)Math.Pow(2, attempt) * 100; // 100ms, 200ms, 400ms...
                    await Task.Delay(delayMs);
                }
            }

            // All retries failed
            LogError($"Request failed after {maxRetries} attempts");
            throw lastException ?? new Exception("Request failed");
        }

        private async Task<T> GetRequest<T>(string url)
        {
            LogDebug($"GET {url}");

            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.timeout = timeoutSeconds;

                var operation = request.SendWebRequest();

                while (!operation.isDone)
                {
                    await Task.Yield();
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string responseJson = request.downloadHandler.text;
                    LogDebug($"Response: {responseJson}");
                    return JsonUtility.FromJson<T>(responseJson);
                }
                else
                {
                    throw new Exception($"HTTP {request.responseCode}: {request.error}");
                }
            }
        }

        // ============================================================================
        // Helper Methods
        // ============================================================================

        private GameStateSnapshot CreateMinimalGameState(string sceneId)
        {
            return new GameStateSnapshot(
                scene_id: sceneId,
                player: new PlayerState("Player", sceneId),
                world: new WorldState("day")
            );
        }

        private void LogDebug(string message)
        {
            if (debugMode)
            {
                Debug.Log($"[ElleClient] {message}");
            }
        }

        private void LogError(string message)
        {
            Debug.LogError($"[ElleClient] {message}");
        }

        // ============================================================================
        // Unity Lifecycle
        // ============================================================================

        private void Start()
        {
            LogDebug($"ElleClient initialized. Service URL: {baseUrl}");
        }

        // ============================================================================
        // Internal Models
        // ============================================================================

        [Serializable]
        private class HealthResponse
        {
            public string status;
            public string service;
        }
    }
}
