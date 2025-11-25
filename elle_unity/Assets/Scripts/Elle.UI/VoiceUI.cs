/**
 * VoiceUI.cs - Voice interface for Unity Elle client
 *
 * Handles:
 * - Wake word detection ("Hey Elle")
 * - Voice-to-text dictation
 * - Query sending to HoloLoom
 * - Visual feedback (listening indicator)
 *
 * Created: 2025-11-24
 */

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows.Speech;

namespace Elle.UI
{
    /// <summary>
    /// Voice interface for Elle AR assistant.
    /// Uses Unity's Windows.Speech API for wake word and dictation.
    /// </summary>
    public class VoiceUI : MonoBehaviour
    {
        [Header("Configuration")]
        [SerializeField] private string wakeWord = "Hey Elle";
        [SerializeField] private float dictationTimeout = 5f;
        [SerializeField] private bool showVisualFeedback = true;

        [Header("References")]
        [SerializeField] private HoloLoomBridge bridge;
        [SerializeField] private VisualizationManager visualizationManager;

        [Header("UI Elements (Optional)")]
        [SerializeField] private GameObject listeningIndicator;
        [SerializeField] private TMPro.TextMeshPro transcriptText;

        // Speech recognition
        private KeywordRecognizer keywordRecognizer;
        private DictationRecognizer dictationRecognizer;

        // State
        private bool isListening = false;
        private string currentTranscript = "";

        void Start()
        {
            // Find references if not set
            if (bridge == null)
                bridge = FindObjectOfType<Elle.Core.HoloLoomBridge>();

            if (visualizationManager == null)
                visualizationManager = FindObjectOfType<VisualizationManager>();

            // Initialize wake word recognition
            InitializeWakeWord();

            Debug.Log($"VoiceUI initialized. Say '{wakeWord}' to activate.");
        }

        /// <summary>
        /// Initialize wake word (keyword) recognition
        /// </summary>
        void InitializeWakeWord()
        {
            try
            {
                keywordRecognizer = new KeywordRecognizer(new[] { wakeWord });
                keywordRecognizer.OnPhraseRecognized += OnWakeWordDetected;
                keywordRecognizer.Start();

                Debug.Log($"✅ Wake word recognition started: '{wakeWord}'");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to initialize wake word recognition: {ex.Message}");
            }
        }

        /// <summary>
        /// Wake word detected - start dictation
        /// </summary>
        void OnWakeWordDetected(PhraseRecognizedEventArgs args)
        {
            Debug.Log($"🎤 Wake word detected: {args.text}");

            if (isListening)
            {
                Debug.LogWarning("Already listening");
                return;
            }

            StartDictation();
        }

        /// <summary>
        /// Start voice dictation (speech-to-text)
        /// </summary>
        void StartDictation()
        {
            isListening = true;
            currentTranscript = "";

            // Stop wake word recognition while dictating
            keywordRecognizer.Stop();

            // Show visual feedback
            if (showVisualFeedback)
            {
                ShowListeningIndicator(true);
            }

            try
            {
                dictationRecognizer = new DictationRecognizer();

                // Partial results (real-time)
                dictationRecognizer.DictationHypothesis += (text) =>
                {
                    UpdateTranscript($"{text}...");
                };

                // Final result
                dictationRecognizer.DictationResult += (text, confidence) =>
                {
                    OnDictationResult(text, confidence);
                };

                // Dictation complete
                dictationRecognizer.DictationComplete += (cause) =>
                {
                    OnDictationComplete(cause);
                };

                // Error handling
                dictationRecognizer.DictationError += (error, hresult) =>
                {
                    Debug.LogError($"Dictation error: {error}");
                    StopDictation();
                };

                dictationRecognizer.Start();

                Debug.Log("🎤 Dictation started - speak your query");

                // Auto-stop after timeout
                Invoke(nameof(StopDictation), dictationTimeout);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to start dictation: {ex.Message}");
                StopDictation();
            }
        }

        /// <summary>
        /// Dictation result received
        /// </summary>
        async void OnDictationResult(string text, ConfidenceLevel confidence)
        {
            currentTranscript = text;
            UpdateTranscript(text);

            Debug.Log($"📝 Dictation result: '{text}' (confidence: {confidence})");

            // Send query to HoloLoom
            if (bridge != null && bridge.IsConnected)
            {
                try
                {
                    ShowListeningIndicator(false);
                    ShowProcessingIndicator(true);

                    var response = await bridge.SendQuery(text);

                    Debug.Log($"✅ HoloLoom response: {response.response}");

                    // Render visualizations
                    if (visualizationManager != null && response.visualizations != null)
                    {
                        visualizationManager.RenderVisualizations(response.visualizations);
                    }

                    // Speak response (optional - implement TTS)
                    SpeakResponse(response.response);

                    ShowProcessingIndicator(false);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error querying HoloLoom: {ex.Message}");
                    ShowProcessingIndicator(false);
                }
            }
            else
            {
                Debug.LogWarning("HoloLoom bridge not connected");
            }
        }

        /// <summary>
        /// Dictation complete
        /// </summary>
        void OnDictationComplete(DictationCompletionCause cause)
        {
            Debug.Log($"Dictation complete: {cause}");
            StopDictation();
        }

        /// <summary>
        /// Stop dictation and return to wake word mode
        /// </summary>
        void StopDictation()
        {
            CancelInvoke(nameof(StopDictation));

            if (dictationRecognizer != null)
            {
                dictationRecognizer.Stop();
                dictationRecognizer.Dispose();
                dictationRecognizer = null;
            }

            isListening = false;
            ShowListeningIndicator(false);

            // Resume wake word recognition
            if (keywordRecognizer != null)
            {
                keywordRecognizer.Start();
            }

            Debug.Log("🎤 Dictation stopped - wake word active");
        }

        /// <summary>
        /// Update transcript display
        /// </summary>
        void UpdateTranscript(string text)
        {
            if (transcriptText != null)
            {
                transcriptText.text = text;
            }
        }

        /// <summary>
        /// Show/hide listening indicator
        /// </summary>
        void ShowListeningIndicator(bool show)
        {
            if (listeningIndicator != null)
            {
                listeningIndicator.SetActive(show);
            }
        }

        /// <summary>
        /// Show/hide processing indicator
        /// </summary>
        void ShowProcessingIndicator(bool show)
        {
            // TODO: Implement processing spinner/indicator
            Debug.Log(show ? "Processing..." : "Processing complete");
        }

        /// <summary>
        /// Speak Elle's response using TTS (optional)
        /// </summary>
        void SpeakResponse(string text)
        {
            // TODO: Implement text-to-speech
            // Options:
            // 1. Unity's built-in TTS (Windows only)
            // 2. Microsoft Azure Speech SDK
            // 3. Google Cloud TTS
            // 4. Elevenlabs API

            Debug.Log($"🔊 Elle says: {text}");
        }

        /// <summary>
        /// Manual trigger for voice input (for testing)
        /// </summary>
        public void TriggerVoiceInput()
        {
            if (!isListening)
            {
                StartDictation();
            }
        }

        void OnDestroy()
        {
            if (keywordRecognizer != null)
            {
                keywordRecognizer.Stop();
                keywordRecognizer.Dispose();
            }

            if (dictationRecognizer != null)
            {
                dictationRecognizer.Stop();
                dictationRecognizer.Dispose();
            }
        }

        void OnApplicationQuit()
        {
            OnDestroy();
        }
    }
}
