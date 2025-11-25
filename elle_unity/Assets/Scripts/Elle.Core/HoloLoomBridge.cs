/**
 * HoloLoomBridge.cs - WebSocket client for Unity → HoloLoom communication
 *
 * Connects Unity client to HoloLoom backend via WebSocket.
 * Sends queries with AR context, receives responses with visualizations.
 *
 * Created: 2025-11-24
 */

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;

namespace Elle.Core
{
    /// <summary>
    /// Main bridge between Unity client and HoloLoom backend.
    /// Handles WebSocket connection, query sending, and response handling.
    /// </summary>
    public class HoloLoomBridge : MonoBehaviour
    {
        [Header("Backend Configuration")]
        [SerializeField] private string backendUrl = "ws://localhost:8000/ws/ar";
        [SerializeField] private bool autoConnect = true;
        [SerializeField] private float reconnectDelay = 5f;

        [Header("Debug")]
        [SerializeField] private bool logMessages = true;

        // WebSocket connection
        private WebSocket ws;
        private bool isConnecting = false;

        // Events
        public event Action<HoloLoomResponse> OnResponseReceived;
        public event Action OnConnected;
        public event Action<string> OnError;
        public event Action OnDisconnected;

        // Properties
        public bool IsConnected => ws?.IsAlive ?? false;
        public string SessionId { get; private set; }

        // Cached references (avoid FindObjectOfType on every query)
        private Elle.Vision.ObjectDetector cachedObjectDetector;
        private Elle.Vision.HandTracker cachedHandTracker;

        void Start()
        {
            SessionId = SystemInfo.deviceUniqueIdentifier;

            // Cache references once on startup
            cachedObjectDetector = FindObjectOfType<Elle.Vision.ObjectDetector>();
            cachedHandTracker = FindObjectOfType<Elle.Vision.HandTracker>();

            if (autoConnect)
            {
                Connect();
            }
        }

        /// <summary>
        /// Connect to HoloLoom backend via WebSocket
        /// </summary>
        public void Connect()
        {
            if (isConnecting || IsConnected)
            {
                Debug.LogWarning("Already connecting or connected");
                return;
            }

            isConnecting = true;
            Debug.Log($"Connecting to HoloLoom backend at {backendUrl}");

            try
            {
                ws = new WebSocket(backendUrl);

                // Connection opened
                ws.OnOpen += (sender, e) =>
                {
                    isConnecting = false;
                    Debug.Log("✅ Connected to HoloLoom backend");
                    OnConnected?.Invoke();
                };

                // Message received
                ws.OnMessage += (sender, e) =>
                {
                    if (logMessages)
                    {
                        Debug.Log($"📩 Received: {e.Data.Substring(0, Math.Min(100, e.Data.Length))}...");
                    }

                    try
                    {
                        var response = JsonConvert.DeserializeObject<HoloLoomResponse>(e.Data);
                        OnResponseReceived?.Invoke(response);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"Failed to parse response: {ex.Message}");
                    }
                };

                // Error occurred
                ws.OnError += (sender, e) =>
                {
                    isConnecting = false;
                    Debug.LogError($"❌ WebSocket error: {e.Message}");
                    OnError?.Invoke(e.Message);
                };

                // Connection closed
                ws.OnClose += (sender, e) =>
                {
                    isConnecting = false;
                    Debug.Log($"Disconnected from HoloLoom (code: {e.Code}, reason: {e.Reason})");
                    OnDisconnected?.Invoke();

                    // Auto-reconnect
                    if (autoConnect)
                    {
                        Debug.Log($"Reconnecting in {reconnectDelay}s...");
                        Invoke(nameof(Connect), reconnectDelay);
                    }
                };

                ws.Connect();
            }
            catch (Exception ex)
            {
                isConnecting = false;
                Debug.LogError($"Failed to connect: {ex.Message}");
                OnError?.Invoke(ex.Message);
            }
        }

        /// <summary>
        /// Send query to HoloLoom and wait for response
        /// </summary>
        public async Task<HoloLoomResponse> SendQuery(
            string query,
            ARContext context = null,
            string mode = "verify",
            int maxSteps = 5
        )
        {
            if (!IsConnected)
            {
                throw new InvalidOperationException("Not connected to HoloLoom backend");
            }

            var request = new HoloLoomRequest
            {
                query = query,
                context = context ?? GetCurrentContext(),
                mode = mode,
                max_steps = maxSteps,
                session_id = SessionId
            };

            var json = JsonConvert.SerializeObject(request, Formatting.None);

            if (logMessages)
            {
                Debug.Log($"📤 Sending query: {query}");
            }

            // Set up one-time response handler
            var tcs = new TaskCompletionSource<HoloLoomResponse>();
            Action<HoloLoomResponse> handler = null;

            handler = (response) =>
            {
                OnResponseReceived -= handler;
                tcs.TrySetResult(response);
            };

            OnResponseReceived += handler;

            // Send query
            ws.Send(json);

            // Wait for response with timeout
            var timeoutTask = Task.Delay(10000); // 10 second timeout
            var completedTask = await Task.WhenAny(tcs.Task, timeoutTask);

            if (completedTask == timeoutTask)
            {
                OnResponseReceived -= handler;
                throw new TimeoutException("HoloLoom query timed out after 10 seconds");
            }

            return await tcs.Task;
        }

        /// <summary>
        /// Send query without waiting for response (fire and forget)
        /// </summary>
        public void SendQueryAsync(string query, ARContext context = null, string mode = "verify")
        {
            if (!IsConnected)
            {
                Debug.LogWarning("Not connected - query not sent");
                return;
            }

            var request = new HoloLoomRequest
            {
                query = query,
                context = context ?? GetCurrentContext(),
                mode = mode,
                session_id = SessionId
            };

            var json = JsonConvert.SerializeObject(request);
            ws.Send(json);
        }

        /// <summary>
        /// Get current AR context (camera position, rotation, gaze)
        /// </summary>
        private ARContext GetCurrentContext()
        {
            var camera = Camera.main;
            if (camera == null)
            {
                Debug.LogWarning("Main camera not found");
                return new ARContext
                {
                    sessionId = SessionId,
                    platform = "unity_quest"
                };
            }

            return new ARContext
            {
                userPosition = camera.transform.position,
                userRotation = camera.transform.rotation,
                gazeDirection = camera.transform.forward,
                sessionId = SessionId,
                platform = "unity_quest",
                visibleObjects = GetVisibleObjects(),
                handGestures = GetHandGestures(),
                timestamp = DateTime.UtcNow.ToString("o")
            };
        }

        /// <summary>
        /// Get currently visible objects (from ObjectDetector if available)
        /// </summary>
        private List<DetectedObject> GetVisibleObjects()
        {
            // Use cached reference (set on Start)
            return cachedObjectDetector?.CurrentDetections ?? new List<DetectedObject>();
        }

        /// <summary>
        /// Get current hand gestures (from HandTracker if available)
        /// </summary>
        private List<HandGesture> GetHandGestures()
        {
            // Use cached reference (set on Start)
            return cachedHandTracker?.CurrentGestures ?? new List<HandGesture>();
        }

        /// <summary>
        /// Disconnect from backend
        /// </summary>
        public void Disconnect()
        {
            if (ws != null)
            {
                ws.Close();
                ws = null;
            }
        }

        void OnDestroy()
        {
            Disconnect();
        }

        void OnApplicationQuit()
        {
            Disconnect();
        }
    }

    // ==================== Data Models ====================

    [Serializable]
    public class HoloLoomRequest
    {
        public string query;
        public ARContext context;
        public string mode;
        public int max_steps;
        public string session_id;
    }

    [Serializable]
    public class HoloLoomResponse
    {
        public string response;
        public List<Visualization> visualizations;
        public float confidence;
        public Dictionary<string, object> metadata;
        public List<string> steps_taken;
    }

    [Serializable]
    public class ARContext
    {
        public Vector3 userPosition;
        public Quaternion userRotation;
        public Vector3 gazeDirection;
        public string sessionId;
        public string platform;
        public List<DetectedObject> visibleObjects;
        public List<HandGesture> handGestures;
        public string timestamp;
    }

    [Serializable]
    public class Visualization
    {
        public string type;  // "overlay", "highlight", "path"
        public string id;
        public Vector3 position;
        public Dictionary<string, object> data;
    }

    [Serializable]
    public class DetectedObject
    {
        public string id;
        public string label;
        public float confidence;
        public Vector3 position;
        public string objectType;
    }

    [Serializable]
    public class HandGesture
    {
        public string handId;  // "left" or "right"
        public string gesture;  // "point", "pinch", "palm_up", etc.
        public float confidence;
        public Vector3 position;
    }
}
