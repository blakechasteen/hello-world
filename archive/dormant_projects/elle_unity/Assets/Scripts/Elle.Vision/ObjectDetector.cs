/**
 * ObjectDetector.cs - Real-time object detection for Unity
 *
 * Uses Unity Barracuda for on-device ML inference.
 * Supports YOLO, SSD, EfficientDet models.
 *
 * Created: 2025-11-24
 */

using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using Elle.Core;

namespace Elle.Vision
{
    /// <summary>
    /// Real-time object detection using Unity Barracuda (on-device ML).
    /// Detects objects in camera feed and provides 3D positions.
    /// </summary>
    public class ObjectDetector : MonoBehaviour
    {
        [Header("Model Configuration")]
        [SerializeField] private NNModel modelAsset;
        [SerializeField] private TextAsset labelsAsset;
        [SerializeField] private WorkerFactory.Type workerType = WorkerFactory.Type.ComputePrecompiled;

        [Header("Detection Configuration")]
        [SerializeField] private Camera arCamera;
        [SerializeField] private int inputWidth = 640;
        [SerializeField] private int inputHeight = 480;
        [SerializeField] private float confidenceThreshold = 0.5f;
        [SerializeField] private int maxDetections = 20;
        [SerializeField] private float detectionInterval = 0.1f; // 10 FPS

        [Header("Debug")]
        [SerializeField] private bool visualizeDetections = true;
        [SerializeField] private bool logDetections = false;

        // Barracuda inference
        private IWorker worker;
        private Model model;
        private RenderTexture renderTexture;
        private Texture2D inputTexture;

        // Labels
        private string[] labels;

        // Current detections
        public List<DetectedObject> CurrentDetections { get; private set; } = new List<DetectedObject>();

        // State
        private float lastDetectionTime;
        private bool isProcessing = false;

        // Events
        public event System.Action<List<DetectedObject>> OnObjectsDetected;

        void Start()
        {
            if (arCamera == null)
            {
                arCamera = Camera.main;
            }

            InitializeModel();
            InitializeRenderTexture();
            LoadLabels();
        }

        /// <summary>
        /// Initialize Barracuda model
        /// </summary>
        void InitializeModel()
        {
            if (modelAsset == null)
            {
                Debug.LogWarning("No model asset assigned - object detection disabled");
                return;
            }

            try
            {
                model = ModelLoader.Load(modelAsset);
                worker = WorkerFactory.CreateWorker(workerType, model);
                Debug.Log($"✅ Object detection model loaded: {model.name}");
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"Failed to load model: {ex.Message}");
            }
        }

        /// <summary>
        /// Initialize render texture for camera feed
        /// </summary>
        void InitializeRenderTexture()
        {
            if (arCamera == null)
                return;

            renderTexture = new RenderTexture(inputWidth, inputHeight, 24);
            arCamera.targetTexture = renderTexture;

            inputTexture = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

            Debug.Log($"Render texture initialized: {inputWidth}x{inputHeight}");
        }

        /// <summary>
        /// Load class labels from text file
        /// </summary>
        void LoadLabels()
        {
            if (labelsAsset == null)
            {
                Debug.LogWarning("No labels asset assigned - using default COCO labels");
                labels = GetDefaultCOCOLabels();
                return;
            }

            var lines = labelsAsset.text.Split('\n');
            labels = new string[lines.Length];

            for (int i = 0; i < lines.Length; i++)
            {
                labels[i] = lines[i].Trim();
            }

            Debug.Log($"Loaded {labels.Length} class labels");
        }

        void Update()
        {
            if (worker == null || isProcessing)
                return;

            // Throttle detection frequency
            if (Time.time - lastDetectionTime < detectionInterval)
                return;

            DetectObjectsAsync();
        }

        /// <summary>
        /// Run object detection on current camera frame
        /// </summary>
        async void DetectObjectsAsync()
        {
            isProcessing = true;
            lastDetectionTime = Time.time;

            try
            {
                // Capture camera frame
                CaptureFrame();

                // Convert to tensor
                var tensor = new Tensor(inputTexture, 3);

                // Run inference
                worker.Execute(tensor);
                var output = worker.PeekOutput();

                // Parse detections
                var detections = ParseDetections(output);

                // Update current detections
                CurrentDetections = detections;

                // Trigger event
                OnObjectsDetected?.Invoke(detections);

                if (logDetections && detections.Count > 0)
                {
                    Debug.Log($"Detected {detections.Count} objects");
                }

                // Cleanup
                tensor.Dispose();
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"Detection error: {ex.Message}");
            }
            finally
            {
                isProcessing = false;
            }
        }

        /// <summary>
        /// Capture current camera frame to texture
        /// </summary>
        void CaptureFrame()
        {
            if (renderTexture == null || inputTexture == null)
                return;

            RenderTexture.active = renderTexture;
            inputTexture.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
            inputTexture.Apply();
            RenderTexture.active = null;
        }

        /// <summary>
        /// Parse detection output from model
        /// Note: Format depends on model type (YOLO, SSD, etc.)
        /// </summary>
        List<DetectedObject> ParseDetections(Tensor output)
        {
            var detections = new List<DetectedObject>();

            // This is a simplified parser - actual implementation depends on model
            // YOLO format: [batch, num_detections, 85] where 85 = [x, y, w, h, conf, 80 classes]
            // SSD format: [batch, num_detections, 7] where 7 = [?, class, conf, x1, y1, x2, y2]

            int numDetections = output.shape[1];
            int numClasses = labels.Length;

            for (int i = 0; i < numDetections && detections.Count < maxDetections; i++)
            {
                // Extract detection (format depends on model)
                float confidence = output[0, i, 4]; // Confidence score

                if (confidence < confidenceThreshold)
                    continue;

                // Get class with highest probability
                int classId = 0;
                float maxProb = 0f;

                for (int c = 0; c < numClasses; c++)
                {
                    float prob = output[0, i, 5 + c];
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        classId = c;
                    }
                }

                // Bounding box (normalized 0-1)
                float x = output[0, i, 0];
                float y = output[0, i, 1];
                float w = output[0, i, 2];
                float h = output[0, i, 3];

                // Convert to 3D position (simplified - assumes flat plane)
                var position = BoundingBoxToWorldPosition(x, y, w, h);

                var detection = new DetectedObject
                {
                    id = $"obj_{i}_{Time.frameCount}",
                    label = classId < labels.Length ? labels[classId] : $"class_{classId}",
                    confidence = confidence * maxProb,
                    position = position,
                    objectType = InferObjectType(classId < labels.Length ? labels[classId] : "")
                };

                detections.Add(detection);
            }

            return detections;
        }

        /// <summary>
        /// Convert 2D bounding box to 3D world position (simplified)
        /// </summary>
        Vector3 BoundingBoxToWorldPosition(float x, float y, float w, float h)
        {
            if (arCamera == null)
                return Vector3.zero;

            // Center of bounding box in normalized coords (0-1)
            var centerX = x;
            var centerY = y;

            // Convert to viewport point
            var viewportPoint = new Vector3(centerX, centerY, 1.5f); // Assume 1.5m depth

            // Convert to world position
            var worldPos = arCamera.ViewportToWorldPoint(viewportPoint);

            return worldPos;
        }

        /// <summary>
        /// Infer object type from label (for HoloLoom context)
        /// </summary>
        string InferObjectType(string label)
        {
            // Map labels to semantic types
            var typeMap = new Dictionary<string, string>
            {
                { "person", "person" },
                { "chair", "furniture" },
                { "couch", "furniture" },
                { "bed", "furniture" },
                { "dining table", "furniture" },
                { "laptop", "electronics" },
                { "cell phone", "electronics" },
                { "tv", "electronics" },
                { "bottle", "container" },
                { "cup", "container" },
                { "bowl", "container" },
                { "book", "document" },
                { "scissors", "tool" },
                { "knife", "tool" },
                { "spoon", "utensil" },
                { "fork", "utensil" }
            };

            return typeMap.ContainsKey(label) ? typeMap[label] : "object";
        }

        /// <summary>
        /// Get default COCO dataset labels (80 classes)
        /// </summary>
        string[] GetDefaultCOCOLabels()
        {
            return new string[]
            {
                "person", "bicycle", "car", "motorcycle", "airplane",
                "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            };
        }

        /// <summary>
        /// Visualize detections (for debugging)
        /// </summary>
        void OnGUI()
        {
            if (!visualizeDetections || CurrentDetections.Count == 0)
                return;

            foreach (var detection in CurrentDetections)
            {
                var screenPos = arCamera.WorldToScreenPoint(detection.position);

                if (screenPos.z > 0)
                {
                    var label = $"{detection.label} ({detection.confidence:F2})";
                    var labelSize = GUI.skin.label.CalcSize(new GUIContent(label));

                    // Draw background box
                    var rect = new Rect(
                        screenPos.x - labelSize.x / 2,
                        Screen.height - screenPos.y - labelSize.y / 2,
                        labelSize.x + 10,
                        labelSize.y + 5
                    );

                    GUI.Box(rect, "");

                    // Draw label
                    GUI.Label(rect, label);
                }
            }
        }

        void OnDestroy()
        {
            worker?.Dispose();

            if (renderTexture != null)
            {
                arCamera.targetTexture = null;
                Destroy(renderTexture);
            }

            if (inputTexture != null)
            {
                Destroy(inputTexture);
            }
        }
    }
}
