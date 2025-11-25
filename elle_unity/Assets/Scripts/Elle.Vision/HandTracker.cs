/**
 * HandTracker.cs - Hand tracking and gesture recognition for Unity
 *
 * Uses Unity XR Hands subsystem for Quest 3 hand tracking.
 * Detects gestures: point, pinch, palm_up, grab, swipe
 *
 * Created: 2025-11-24
 */

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;
using Elle.Core;

namespace Elle.Vision
{
    /// <summary>
    /// Hand tracking and gesture recognition using Unity XR Hands.
    /// </summary>
    public class HandTracker : MonoBehaviour
    {
        [Header("Configuration")]
        [SerializeField] private bool enableGestureRecognition = true;
        [SerializeField] private float pinchThreshold = 0.03f;  // 3cm
        [SerializeField] private bool visualizeHands = true;

        [Header("Debug")]
        [SerializeField] private bool logGestures = true;

        // XR Hands subsystem
        private XRHandSubsystem handSubsystem;

        // Current state
        public List<HandGesture> CurrentGestures { get; private set; } = new List<HandGesture>();

        // Events
        public event System.Action<HandGesture> OnGestureDetected;

        // Gesture state tracking
        private Dictionary<string, string> lastGestures = new Dictionary<string, string>();
        private Dictionary<string, float> gestureStartTimes = new Dictionary<string, float>();

        void Start()
        {
            InitializeHandTracking();
        }

        /// <summary>
        /// Initialize Unity XR Hands subsystem
        /// </summary>
        void InitializeHandTracking()
        {
            var subsystems = new List<XRHandSubsystem>();
            SubsystemManager.GetSubsystems(subsystems);

            if (subsystems.Count > 0)
            {
                handSubsystem = subsystems[0];
                handSubsystem.Start();
                Debug.Log("✅ Hand tracking subsystem started");
            }
            else
            {
                Debug.LogWarning("❌ No hand tracking subsystem found");
            }
        }

        void Update()
        {
            if (handSubsystem == null || !enableGestureRecognition)
                return;

            CurrentGestures.Clear();

            // Process left hand
            var leftHand = handSubsystem.leftHand;
            if (leftHand.isTracked)
            {
                var gesture = DetectGesture(leftHand, "left");
                if (gesture != null)
                {
                    CurrentGestures.Add(gesture);
                }
            }

            // Process right hand
            var rightHand = handSubsystem.rightHand;
            if (rightHand.isTracked)
            {
                var gesture = DetectGesture(rightHand, "right");
                if (gesture != null)
                {
                    CurrentGestures.Add(gesture);
                }
            }
        }

        /// <summary>
        /// Detect gesture from hand pose
        /// </summary>
        HandGesture DetectGesture(XRHand hand, string handId)
        {
            // Get key joint positions
            if (!TryGetJointPose(hand, XRHandJointID.IndexTip, out var indexTipPose))
                return null;

            if (!TryGetJointPose(hand, XRHandJointID.ThumbTip, out var thumbTipPose))
                return null;

            if (!TryGetJointPose(hand, XRHandJointID.Palm, out var palmPose))
                return null;

            // Detect pinch gesture
            var pinchDistance = Vector3.Distance(indexTipPose.position, thumbTipPose.position);
            if (pinchDistance < pinchThreshold)
            {
                return CreateGesture(handId, "pinch", indexTipPose.position, 1f - (pinchDistance / pinchThreshold));
            }

            // Detect point gesture
            if (IsPointing(hand, indexTipPose, palmPose))
            {
                var pointDirection = (indexTipPose.position - palmPose.position).normalized;
                return CreateGesture(handId, "point", indexTipPose.position, 0.9f);
            }

            // Detect palm up gesture
            if (IsPalmUp(palmPose))
            {
                return CreateGesture(handId, "palm_up", palmPose.position, 0.8f);
            }

            // Detect grab gesture
            if (IsGrabbing(hand, indexTipPose, thumbTipPose, palmPose))
            {
                return CreateGesture(handId, "grab", palmPose.position, 0.85f);
            }

            // No gesture detected
            return null;
        }

        /// <summary>
        /// Check if hand is pointing
        /// </summary>
        bool IsPointing(XRHand hand, HandJointPose indexTip, HandJointPose palm)
        {
            // Index finger extended, other fingers curled
            if (!TryGetJointPose(hand, XRHandJointID.MiddleTip, out var middleTip))
                return false;

            var indexDist = Vector3.Distance(indexTip.position, palm.position);
            var middleDist = Vector3.Distance(middleTip.position, palm.position);

            // Index should be more extended than middle
            return indexDist > middleDist * 1.2f;
        }

        /// <summary>
        /// Check if palm is facing up
        /// </summary>
        bool IsPalmUp(HandJointPose palm)
        {
            // Palm normal should point upward
            var palmUp = Vector3.Dot(palm.up, Vector3.up);
            return palmUp > 0.8f; // 80% alignment with up vector
        }

        /// <summary>
        /// Check if hand is grabbing
        /// </summary>
        bool IsGrabbing(XRHand hand, HandJointPose indexTip, HandJointPose thumbTip, HandJointPose palm)
        {
            // All fingers close to palm
            if (!TryGetJointPose(hand, XRHandJointID.MiddleTip, out var middleTip))
                return false;

            if (!TryGetJointPose(hand, XRHandJointID.RingTip, out var ringTip))
                return false;

            var avgDist = (
                Vector3.Distance(indexTip.position, palm.position) +
                Vector3.Distance(thumbTip.position, palm.position) +
                Vector3.Distance(middleTip.position, palm.position) +
                Vector3.Distance(ringTip.position, palm.position)
            ) / 4f;

            return avgDist < 0.08f; // 8cm average distance
        }

        /// <summary>
        /// Create gesture object
        /// </summary>
        HandGesture CreateGesture(string handId, string gestureName, Vector3 position, float confidence)
        {
            // Check if this is a new gesture (debounce)
            var key = $"{handId}_{gestureName}";
            var isNewGesture = !lastGestures.ContainsKey(key) || lastGestures[key] != gestureName;

            if (isNewGesture)
            {
                lastGestures[key] = gestureName;
                gestureStartTimes[key] = Time.time;

                if (logGestures)
                {
                    Debug.Log($"👋 {handId} hand: {gestureName} (confidence: {confidence:F2})");
                }

                var gesture = new HandGesture
                {
                    handId = handId,
                    gesture = gestureName,
                    confidence = confidence,
                    position = position
                };

                OnGestureDetected?.Invoke(gesture);

                return gesture;
            }

            // Return existing gesture
            return new HandGesture
            {
                handId = handId,
                gesture = gestureName,
                confidence = confidence,
                position = position
            };
        }

        /// <summary>
        /// Try to get joint pose (with error handling)
        /// </summary>
        bool TryGetJointPose(XRHand hand, XRHandJointID jointId, out HandJointPose pose)
        {
            var joint = hand.GetJoint(jointId);
            return joint.TryGetPose(out pose);
        }

        /// <summary>
        /// Get current hand positions (for visualization)
        /// </summary>
        public Dictionary<string, Vector3> GetHandPositions()
        {
            var positions = new Dictionary<string, Vector3>();

            if (handSubsystem == null)
                return positions;

            // Left hand
            var leftHand = handSubsystem.leftHand;
            if (leftHand.isTracked && TryGetJointPose(leftHand, XRHandJointID.Palm, out var leftPalm))
            {
                positions["left_palm"] = leftPalm.position;
            }

            // Right hand
            var rightHand = handSubsystem.rightHand;
            if (rightHand.isTracked && TryGetJointPose(rightHand, XRHandJointID.Palm, out var rightPalm))
            {
                positions["right_palm"] = rightPalm.position;
            }

            return positions;
        }

        /// <summary>
        /// Check if specific gesture is currently active
        /// </summary>
        public bool IsGestureActive(string gestureName)
        {
            return CurrentGestures.Exists(g => g.gesture == gestureName);
        }

        /// <summary>
        /// Get gesture by hand ID
        /// </summary>
        public HandGesture GetGestureByHand(string handId)
        {
            return CurrentGestures.Find(g => g.handId == handId);
        }

        void OnDestroy()
        {
            if (handSubsystem != null)
            {
                handSubsystem.Stop();
            }
        }
    }

    /// <summary>
    /// Hand joint pose (position + rotation)
    /// </summary>
    public struct HandJointPose
    {
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 up => rotation * Vector3.up;
        public Vector3 forward => rotation * Vector3.forward;
    }
}
