/**
 * VisualizationManager.cs - Manages AR visualizations in Unity
 *
 * Converts HoloLoom JSON visualizations → Unity GameObjects
 * Supports: overlays (text), highlights (boxes), paths (arrows)
 *
 * Created: 2025-11-24
 */

using System.Collections.Generic;
using UnityEngine;
using Elle.Core;

namespace Elle.UI
{
    /// <summary>
    /// Manages creation and lifecycle of AR visualizations.
    /// Converts JSON from HoloLoom into Unity GameObjects.
    /// </summary>
    public class VisualizationManager : MonoBehaviour
    {
        [Header("Prefabs")]
        [SerializeField] private GameObject overlayPrefab;
        [SerializeField] private GameObject highlightPrefab;
        [SerializeField] private GameObject pathPrefab;

        [Header("Configuration")]
        [SerializeField] private float defaultLifetime = 5f;
        [SerializeField] private bool autoRemove = true;

        // Active visualizations
        private Dictionary<string, GameObject> activeVisualizations = new Dictionary<string, GameObject>();

        /// <summary>
        /// Render list of visualizations from HoloLoom response
        /// </summary>
        public void RenderVisualizations(List<Visualization> visualizations)
        {
            if (visualizations == null || visualizations.Count == 0)
            {
                Debug.Log("No visualizations to render");
                return;
            }

            Debug.Log($"Rendering {visualizations.Count} visualizations");

            foreach (var viz in visualizations)
            {
                RenderVisualization(viz);
            }
        }

        /// <summary>
        /// Render single visualization
        /// </summary>
        public GameObject RenderVisualization(Visualization viz)
        {
            // Check if already exists
            if (activeVisualizations.ContainsKey(viz.id))
            {
                Debug.LogWarning($"Visualization {viz.id} already exists, removing old");
                RemoveVisualization(viz.id);
            }

            GameObject obj = null;

            switch (viz.type)
            {
                case "overlay":
                    obj = CreateOverlay(viz);
                    break;

                case "highlight":
                    obj = CreateHighlight(viz);
                    break;

                case "path":
                    obj = CreatePath(viz);
                    break;

                default:
                    Debug.LogWarning($"Unknown visualization type: {viz.type}");
                    return null;
            }

            if (obj != null)
            {
                activeVisualizations[viz.id] = obj;

                // Auto-remove after lifetime
                if (autoRemove)
                {
                    Destroy(obj, defaultLifetime);
                    StartCoroutine(RemoveAfterDelay(viz.id, defaultLifetime));
                }

                Debug.Log($"✅ Created {viz.type} visualization: {viz.id}");
            }

            return obj;
        }

        /// <summary>
        /// Create text overlay visualization
        /// </summary>
        GameObject CreateOverlay(Visualization viz)
        {
            GameObject obj;

            if (overlayPrefab != null)
            {
                obj = Instantiate(overlayPrefab, viz.position, Quaternion.identity);
            }
            else
            {
                // Create default overlay (3D text)
                obj = new GameObject($"Overlay_{viz.id}");
                obj.transform.position = viz.position;

                // Create TextMeshPro (or TextMesh as fallback)
                var tmp = obj.AddComponent<TMPro.TextMeshPro>();
                if (tmp != null)
                {
                    tmp.text = GetData<string>(viz.data, "content", "");
                    tmp.fontSize = GetData<float>(viz.data, "size", 0.8f) * 10f;
                    tmp.color = ParseColor(GetData<string>(viz.data, "color", "#00ff00"));
                    tmp.alignment = TMPro.TextAlignmentOptions.Center;
                    tmp.enableAutoSizing = true;
                }
            }

            // Make it face camera (billboard)
            var billboard = obj.GetComponent<Billboard>();
            if (billboard == null)
            {
                billboard = obj.AddComponent<Billboard>();
            }

            return obj;
        }

        /// <summary>
        /// Create highlight (bounding box) visualization
        /// </summary>
        GameObject CreateHighlight(Visualization viz)
        {
            GameObject obj;

            if (highlightPrefab != null)
            {
                obj = Instantiate(highlightPrefab, viz.position, Quaternion.identity);
            }
            else
            {
                // Create default highlight (wireframe cube)
                obj = GameObject.CreatePrimitive(PrimitiveType.Cube);
                obj.name = $"Highlight_{viz.id}";
                obj.transform.position = viz.position;

                // Get size from data
                var size = GetData<Vector3>(viz.data, "size", Vector3.one * 0.5f);
                obj.transform.localScale = size;

                // Wireframe material
                var renderer = obj.GetComponent<Renderer>();
                var material = new Material(Shader.Find("UI/Default"));
                material.color = ParseColor(GetData<string>(viz.data, "color", "#ffaa00"));
                renderer.material = material;

                // Remove collider
                Destroy(obj.GetComponent<Collider>());
            }

            // Add pulsing animation
            var pulser = obj.AddComponent<PulseAnimation>();
            pulser.speed = 2f;
            pulser.amplitude = 0.1f;

            return obj;
        }

        /// <summary>
        /// Create path (navigation arrows) visualization
        /// </summary>
        GameObject CreatePath(Visualization viz)
        {
            GameObject obj;

            if (pathPrefab != null)
            {
                obj = Instantiate(pathPrefab, viz.position, Quaternion.identity);
            }
            else
            {
                // Create default path (line of arrows)
                obj = new GameObject($"Path_{viz.id}");
                obj.transform.position = viz.position;

                // Get waypoints
                var waypoints = GetData<List<Vector3>>(viz.data, "waypoints", new List<Vector3>());

                if (waypoints.Count == 0)
                {
                    Debug.LogWarning("Path has no waypoints");
                    return obj;
                }

                // Create line renderer
                var lineRenderer = obj.AddComponent<LineRenderer>();
                lineRenderer.startWidth = 0.05f;
                lineRenderer.endWidth = 0.05f;
                lineRenderer.positionCount = waypoints.Count;
                lineRenderer.SetPositions(waypoints.ToArray());
                lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
                lineRenderer.startColor = ParseColor(GetData<string>(viz.data, "color", "#00ffff"));
                lineRenderer.endColor = ParseColor(GetData<string>(viz.data, "color", "#00ffff"));

                // Add arrows along path
                for (int i = 0; i < waypoints.Count - 1; i++)
                {
                    var arrowPos = Vector3.Lerp(waypoints[i], waypoints[i + 1], 0.5f);
                    var arrowDir = (waypoints[i + 1] - waypoints[i]).normalized;
                    CreateArrow(obj.transform, arrowPos, arrowDir);
                }
            }

            return obj;
        }

        /// <summary>
        /// Create arrow indicator
        /// </summary>
        void CreateArrow(Transform parent, Vector3 position, Vector3 direction)
        {
            var arrow = GameObject.CreatePrimitive(PrimitiveType.Cone);
            arrow.name = "Arrow";
            arrow.transform.SetParent(parent);
            arrow.transform.position = position;
            arrow.transform.rotation = Quaternion.LookRotation(direction);
            arrow.transform.localScale = new Vector3(0.1f, 0.2f, 0.1f);

            var renderer = arrow.GetComponent<Renderer>();
            renderer.material.color = Color.cyan;

            Destroy(arrow.GetComponent<Collider>());
        }

        /// <summary>
        /// Remove visualization by ID
        /// </summary>
        public void RemoveVisualization(string id)
        {
            if (activeVisualizations.TryGetValue(id, out var obj))
            {
                Destroy(obj);
                activeVisualizations.Remove(id);
                Debug.Log($"Removed visualization: {id}");
            }
        }

        /// <summary>
        /// Remove all visualizations
        /// </summary>
        public void ClearAll()
        {
            foreach (var kvp in activeVisualizations)
            {
                if (kvp.Value != null)
                {
                    Destroy(kvp.Value);
                }
            }
            activeVisualizations.Clear();
            Debug.Log("Cleared all visualizations");
        }

        /// <summary>
        /// Delayed removal coroutine
        /// </summary>
        System.Collections.IEnumerator RemoveAfterDelay(string id, float delay)
        {
            yield return new WaitForSeconds(delay);
            if (activeVisualizations.ContainsKey(id))
            {
                activeVisualizations.Remove(id);
            }
        }

        /// <summary>
        /// Get data from visualization dictionary with default
        /// Safely handles type conversion with explicit checks
        /// </summary>
        T GetData<T>(Dictionary<string, object> data, string key, T defaultValue)
        {
            if (data == null || !data.ContainsKey(key))
            {
                return defaultValue;
            }

            try
            {
                var value = data[key];

                // Handle null
                if (value == null)
                {
                    return defaultValue;
                }

                // Direct type match
                if (value is T typedValue)
                {
                    return typedValue;
                }

                // Try conversion (for numeric types, strings, etc.)
                if (typeof(T).IsPrimitive || typeof(T) == typeof(string))
                {
                    return (T)Convert.ChangeType(value, typeof(T));
                }

                // Fallback to direct cast (for complex types like Vector3)
                return (T)value;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Failed to convert data[{key}] to {typeof(T).Name}: {ex.Message}");
                return defaultValue;
            }
        }

        /// <summary>
        /// Parse color from hex string
        /// </summary>
        Color ParseColor(string hex)
        {
            if (ColorUtility.TryParseHtmlString(hex, out var color))
            {
                return color;
            }
            return Color.white;
        }
    }

    /// <summary>
    /// Billboard component - makes object face camera
    /// </summary>
    public class Billboard : MonoBehaviour
    {
        private Camera mainCamera;

        void Start()
        {
            mainCamera = Camera.main;
        }

        void LateUpdate()
        {
            if (mainCamera != null)
            {
                transform.LookAt(mainCamera.transform);
                // Rotate 180° because LookAt makes object face *away* from target by default
                // This flips it to face toward the camera
                transform.Rotate(0, 180, 0);
            }
        }
    }

    /// <summary>
    /// Pulse animation component
    /// </summary>
    public class PulseAnimation : MonoBehaviour
    {
        public float speed = 2f;
        public float amplitude = 0.1f;

        private Vector3 originalScale;

        void Start()
        {
            originalScale = transform.localScale;
        }

        void Update()
        {
            var pulse = Mathf.Sin(Time.time * speed) * amplitude;
            transform.localScale = originalScale * (1f + pulse);
        }
    }
}
