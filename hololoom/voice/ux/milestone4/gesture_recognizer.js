/**
 * Gesture Recognizer - Touch/Mouse Gesture Detection
 * Milestone 4: Gesture + voice coordination
 * Date: November 2025
 *
 * Features:
 * - Touch gesture recognition (swipe, tap, pinch, rotate)
 * - Mouse gesture recognition (drag, click patterns)
 * - Gesture + voice fusion ("swipe right while saying 'next thread'")
 * - Continuous gesture tracking
 * - Velocity and acceleration analysis
 * - Multi-touch support
 * - Gesture trajectory analysis
 */

/**
 * Gesture Types
 */
const GestureType = {
    TAP: 'tap',
    DOUBLE_TAP: 'double_tap',
    LONG_PRESS: 'long_press',
    SWIPE_LEFT: 'swipe_left',
    SWIPE_RIGHT: 'swipe_right',
    SWIPE_UP: 'swipe_up',
    SWIPE_DOWN: 'swipe_down',
    PINCH_IN: 'pinch_in',
    PINCH_OUT: 'pinch_out',
    ROTATE_CW: 'rotate_cw',
    ROTATE_CCW: 'rotate_ccw',
    DRAG: 'drag',
    UNKNOWN: 'unknown'
};

/**
 * Touch Point
 */
class TouchPoint {
    constructor(data) {
        this.id = data.id;
        this.x = data.x || 0;
        this.y = data.y || 0;
        this.timestamp = data.timestamp || Date.now();
        this.pressure = data.pressure || 1.0;
    }

    /**
     * Calculate distance to another point
     */
    distanceTo(other) {
        const dx = this.x - other.x;
        const dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Calculate angle to another point (radians)
     */
    angleTo(other) {
        return Math.atan2(other.y - this.y, other.x - this.x);
    }
}

/**
 * Gesture Trajectory
 */
class GestureTrajectory {
    constructor() {
        this.points = [];
        this.startTime = null;
        this.endTime = null;
    }

    /**
     * Add point to trajectory
     */
    addPoint(point) {
        this.points.push(point);

        if (!this.startTime) {
            this.startTime = point.timestamp;
        }
        this.endTime = point.timestamp;
    }

    /**
     * Get total distance traveled
     */
    getTotalDistance() {
        let distance = 0;
        for (let i = 1; i < this.points.length; i++) {
            distance += this.points[i - 1].distanceTo(this.points[i]);
        }
        return distance;
    }

    /**
     * Get straight-line distance (start to end)
     */
    getStraightLineDistance() {
        if (this.points.length < 2) return 0;
        return this.points[0].distanceTo(this.points[this.points.length - 1]);
    }

    /**
     * Get duration (ms)
     */
    getDuration() {
        if (!this.startTime || !this.endTime) return 0;
        return this.endTime - this.startTime;
    }

    /**
     * Get velocity (pixels/second)
     */
    getVelocity() {
        const duration = this.getDuration();
        if (duration === 0) return 0;
        return (this.getTotalDistance() / duration) * 1000;
    }

    /**
     * Get direction vector (normalized)
     */
    getDirection() {
        if (this.points.length < 2) return { x: 0, y: 0 };

        const start = this.points[0];
        const end = this.points[this.points.length - 1];
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const magnitude = Math.sqrt(dx * dx + dy * dy);

        if (magnitude === 0) return { x: 0, y: 0 };

        return {
            x: dx / magnitude,
            y: dy / magnitude
        };
    }

    /**
     * Calculate linearity (0.0-1.0, 1.0 = perfectly straight)
     */
    getLinearity() {
        const totalDistance = this.getTotalDistance();
        const straightDistance = this.getStraightLineDistance();

        if (totalDistance === 0) return 0;

        return straightDistance / totalDistance;
    }
}

/**
 * Gesture Result
 */
class GestureResult {
    constructor(data) {
        this.type = data.type || GestureType.UNKNOWN;
        this.confidence = data.confidence || 0.0;
        this.trajectory = data.trajectory || null;

        // Gesture properties
        this.velocity = data.velocity || 0.0;
        this.direction = data.direction || { x: 0, y: 0 };
        this.duration = data.duration || 0;
        this.distance = data.distance || 0;

        // Multi-touch properties
        this.touchCount = data.touchCount || 1;
        this.scale = data.scale || 1.0; // For pinch gestures
        this.rotation = data.rotation || 0.0; // For rotation gestures (radians)

        // Metadata
        this.timestamp = data.timestamp || new Date();
    }
}

/**
 * Gesture Recognizer
 * Detects gestures from touch/mouse input
 */
class GestureRecognizer {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Detection thresholds
            swipeMinDistance: config.swipeMinDistance || 50, // pixels
            swipeMaxDuration: config.swipeMaxDuration || 500, // ms
            swipeMinVelocity: config.swipeMinVelocity || 200, // pixels/s

            tapMaxDuration: config.tapMaxDuration || 200, // ms
            tapMaxDistance: config.tapMaxDistance || 10, // pixels

            longPressMinDuration: config.longPressMinDuration || 500, // ms
            longPressMaxDistance: config.longPressMaxDistance || 10, // pixels

            doubleTapMaxInterval: config.doubleTapMaxInterval || 300, // ms

            pinchMinScaleChange: config.pinchMinScaleChange || 0.1, // 10% change
            rotateMinAngle: config.rotateMinAngle || 0.1, // radians (~6 degrees)

            // Gesture filtering
            minLinearity: config.minLinearity || 0.5, // For swipe gestures
            confidenceThreshold: config.confidenceThreshold || 0.7,

            ...config
        };

        // Gesture state
        this.isTracking = false;
        this.currentTouches = new Map(); // id -> TouchPoint
        this.trajectories = new Map(); // id -> GestureTrajectory

        // Multi-touch state
        this.initialDistance = null;
        this.initialAngle = null;

        // Tap detection state
        this.lastTapTime = null;
        this.lastTapPosition = null;

        // Long press detection
        this.longPressTimer = null;

        // Event handlers
        this.touchStartHandler = null;
        this.touchMoveHandler = null;
        this.touchEndHandler = null;

        // Callbacks
        this.onGestureDetectedCallback = null;
        this.onGestureStartCallback = null;
        this.onGestureEndCallback = null;

        // Metrics
        this.metrics = {
            totalGestures: 0,
            gesturesByType: {},
            avgConfidence: 0.0,
            lastGestureTime: null
        };

        console.log('[GestureRecognizer] Initialized');
    }

    /**
     * Start gesture tracking
     */
    startTracking(element = document.body) {
        if (!this.config.enabled || this.isTracking) {
            return false;
        }

        this.isTracking = true;
        this.targetElement = element;

        // Setup touch event listeners
        this.touchStartHandler = this._handleTouchStart.bind(this);
        this.touchMoveHandler = this._handleTouchMove.bind(this);
        this.touchEndHandler = this._handleTouchEnd.bind(this);

        element.addEventListener('touchstart', this.touchStartHandler, { passive: false });
        element.addEventListener('touchmove', this.touchMoveHandler, { passive: false });
        element.addEventListener('touchend', this.touchEndHandler);
        element.addEventListener('touchcancel', this.touchEndHandler);

        // Also support mouse events for desktop
        element.addEventListener('mousedown', this.touchStartHandler);
        element.addEventListener('mousemove', this.touchMoveHandler);
        element.addEventListener('mouseup', this.touchEndHandler);

        console.log('[GestureRecognizer] Tracking started');
        return true;
    }

    /**
     * Stop gesture tracking
     */
    stopTracking() {
        if (!this.isTracking) {
            return false;
        }

        this.isTracking = false;

        if (this.targetElement) {
            this.targetElement.removeEventListener('touchstart', this.touchStartHandler);
            this.targetElement.removeEventListener('touchmove', this.touchMoveHandler);
            this.targetElement.removeEventListener('touchend', this.touchEndHandler);
            this.targetElement.removeEventListener('touchcancel', this.touchEndHandler);

            this.targetElement.removeEventListener('mousedown', this.touchStartHandler);
            this.targetElement.removeEventListener('mousemove', this.touchMoveHandler);
            this.targetElement.removeEventListener('mouseup', this.touchEndHandler);

            this.targetElement = null;
        }

        this._clearLongPressTimer();

        console.log('[GestureRecognizer] Tracking stopped');
        return true;
    }

    /**
     * Handle touch/mouse start
     */
    _handleTouchStart(event) {
        event.preventDefault();

        const touches = this._extractTouches(event);

        for (const touch of touches) {
            const touchPoint = new TouchPoint({
                id: touch.identifier,
                x: touch.clientX,
                y: touch.clientY,
                timestamp: Date.now(),
                pressure: touch.force || 1.0
            });

            this.currentTouches.set(touch.identifier, touchPoint);

            const trajectory = new GestureTrajectory();
            trajectory.addPoint(touchPoint);
            this.trajectories.set(touch.identifier, trajectory);
        }

        // Emit gesture start event
        if (this.onGestureStartCallback) {
            this.onGestureStartCallback({
                type: 'gesture_start',
                touchCount: this.currentTouches.size,
                timestamp: new Date()
            });
        }

        // Start long press timer for single touch
        if (touches.length === 1) {
            this._startLongPressTimer(touches[0]);
        }

        // Initialize multi-touch state
        if (touches.length === 2) {
            const points = Array.from(this.currentTouches.values());
            this.initialDistance = points[0].distanceTo(points[1]);
            this.initialAngle = points[0].angleTo(points[1]);
        }

        console.log('[GestureRecognizer] Touch start:', touches.length, 'touches');
    }

    /**
     * Handle touch/mouse move
     */
    _handleTouchMove(event) {
        event.preventDefault();

        const touches = this._extractTouches(event);

        for (const touch of touches) {
            const touchPoint = new TouchPoint({
                id: touch.identifier,
                x: touch.clientX,
                y: touch.clientY,
                timestamp: Date.now(),
                pressure: touch.force || 1.0
            });

            this.currentTouches.set(touch.identifier, touchPoint);

            const trajectory = this.trajectories.get(touch.identifier);
            if (trajectory) {
                trajectory.addPoint(touchPoint);
            }
        }

        // Cancel long press if moved too much
        if (this.longPressTimer && touches.length === 1) {
            const trajectory = this.trajectories.get(touches[0].identifier);
            if (trajectory && trajectory.getTotalDistance() > this.config.longPressMaxDistance) {
                this._clearLongPressTimer();
            }
        }
    }

    /**
     * Handle touch/mouse end
     */
    _handleTouchEnd(event) {
        const touches = this._extractTouches(event, true); // changedTouches

        for (const touch of touches) {
            const trajectory = this.trajectories.get(touch.identifier);

            if (trajectory) {
                // Recognize gesture
                const gesture = this._recognizeGesture(trajectory);

                if (gesture && gesture.confidence >= this.config.confidenceThreshold) {
                    this._handleGestureDetected(gesture);
                }
            }

            // Cleanup
            this.currentTouches.delete(touch.identifier);
            this.trajectories.delete(touch.identifier);
        }

        // Emit gesture end event
        if (this.onGestureEndCallback) {
            this.onGestureEndCallback({
                type: 'gesture_end',
                touchCount: this.currentTouches.size,
                timestamp: new Date()
            });
        }

        // Clear multi-touch state
        if (this.currentTouches.size < 2) {
            this.initialDistance = null;
            this.initialAngle = null;
        }

        this._clearLongPressTimer();

        console.log('[GestureRecognizer] Touch end');
    }

    /**
     * Extract touch points from event
     */
    _extractTouches(event, changed = false) {
        if (event.type.startsWith('touch')) {
            return Array.from(changed ? event.changedTouches : event.touches);
        } else {
            // Mouse event - create synthetic touch
            return [{
                identifier: 0,
                clientX: event.clientX,
                clientY: event.clientY,
                force: 1.0
            }];
        }
    }

    /**
     * Recognize gesture from trajectory
     */
    _recognizeGesture(trajectory) {
        const duration = trajectory.getDuration();
        const distance = trajectory.getStraightLineDistance();
        const velocity = trajectory.getVelocity();
        const direction = trajectory.getDirection();
        const linearity = trajectory.getLinearity();

        // TAP detection
        if (duration <= this.config.tapMaxDuration &&
            distance <= this.config.tapMaxDistance) {

            // Check for double tap
            const now = Date.now();
            if (this.lastTapTime &&
                now - this.lastTapTime <= this.config.doubleTapMaxInterval &&
                this.lastTapPosition) {

                const tapPoint = trajectory.points[0];
                const lastTapDist = Math.sqrt(
                    Math.pow(tapPoint.x - this.lastTapPosition.x, 2) +
                    Math.pow(tapPoint.y - this.lastTapPosition.y, 2)
                );

                if (lastTapDist <= this.config.tapMaxDistance) {
                    this.lastTapTime = null; // Reset for next double tap
                    return this._createGestureResult(GestureType.DOUBLE_TAP, trajectory, 0.95);
                }
            }

            // Single tap
            this.lastTapTime = now;
            this.lastTapPosition = trajectory.points[0];
            return this._createGestureResult(GestureType.TAP, trajectory, 0.9);
        }

        // SWIPE detection
        if (distance >= this.config.swipeMinDistance &&
            duration <= this.config.swipeMaxDuration &&
            velocity >= this.config.swipeMinVelocity &&
            linearity >= this.config.minLinearity) {

            // Determine swipe direction
            let gestureType;
            if (Math.abs(direction.x) > Math.abs(direction.y)) {
                gestureType = direction.x > 0 ? GestureType.SWIPE_RIGHT : GestureType.SWIPE_LEFT;
            } else {
                gestureType = direction.y > 0 ? GestureType.SWIPE_DOWN : GestureType.SWIPE_UP;
            }

            const confidence = Math.min(1.0, linearity * (velocity / 500));
            return this._createGestureResult(gestureType, trajectory, confidence);
        }

        // DRAG detection (slow movement)
        if (distance > this.config.tapMaxDistance &&
            velocity < this.config.swipeMinVelocity) {
            return this._createGestureResult(GestureType.DRAG, trajectory, 0.8);
        }

        // PINCH/ROTATE detection (multi-touch)
        if (this.currentTouches.size === 2 && this.initialDistance) {
            const points = Array.from(this.currentTouches.values());
            const currentDistance = points[0].distanceTo(points[1]);
            const currentAngle = points[0].angleTo(points[1]);

            const scale = currentDistance / this.initialDistance;
            const rotation = currentAngle - this.initialAngle;

            // PINCH detection
            if (Math.abs(1.0 - scale) >= this.config.pinchMinScaleChange) {
                const gestureType = scale > 1.0 ? GestureType.PINCH_OUT : GestureType.PINCH_IN;
                const confidence = Math.min(1.0, Math.abs(1.0 - scale) / 0.5);

                const result = this._createGestureResult(gestureType, trajectory, confidence);
                result.scale = scale;
                return result;
            }

            // ROTATE detection
            if (Math.abs(rotation) >= this.config.rotateMinAngle) {
                const gestureType = rotation > 0 ? GestureType.ROTATE_CW : GestureType.ROTATE_CCW;
                const confidence = Math.min(1.0, Math.abs(rotation) / Math.PI);

                const result = this._createGestureResult(gestureType, trajectory, confidence);
                result.rotation = rotation;
                return result;
            }
        }

        return null; // No gesture recognized
    }

    /**
     * Create gesture result from trajectory
     */
    _createGestureResult(type, trajectory, confidence) {
        return new GestureResult({
            type: type,
            confidence: confidence,
            trajectory: trajectory,
            velocity: trajectory.getVelocity(),
            direction: trajectory.getDirection(),
            duration: trajectory.getDuration(),
            distance: trajectory.getStraightLineDistance(),
            touchCount: this.currentTouches.size
        });
    }

    /**
     * Handle detected gesture
     */
    _handleGestureDetected(gesture) {
        // Update metrics
        this.metrics.totalGestures++;
        this.metrics.gesturesByType[gesture.type] =
            (this.metrics.gesturesByType[gesture.type] || 0) + 1;
        this.metrics.avgConfidence =
            (this.metrics.avgConfidence * (this.metrics.totalGestures - 1) + gesture.confidence) /
            this.metrics.totalGestures;
        this.metrics.lastGestureTime = gesture.timestamp;

        // Emit event
        if (this.onGestureDetectedCallback) {
            this.onGestureDetectedCallback({
                type: 'gesture_detected',
                gesture: gesture,
                timestamp: new Date()
            });
        }

        console.log('[GestureRecognizer] Gesture detected:', gesture.type,
            'confidence:', gesture.confidence.toFixed(2));
    }

    /**
     * Start long press timer
     */
    _startLongPressTimer(touch) {
        this._clearLongPressTimer();

        this.longPressTimer = setTimeout(() => {
            const trajectory = this.trajectories.get(touch.identifier);

            if (trajectory &&
                trajectory.getTotalDistance() <= this.config.longPressMaxDistance) {

                const gesture = this._createGestureResult(
                    GestureType.LONG_PRESS,
                    trajectory,
                    0.95
                );

                this._handleGestureDetected(gesture);
            }

            this.longPressTimer = null;
        }, this.config.longPressMinDuration);
    }

    /**
     * Clear long press timer
     */
    _clearLongPressTimer() {
        if (this.longPressTimer) {
            clearTimeout(this.longPressTimer);
            this.longPressTimer = null;
        }
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            isTracking: this.isTracking,
            activeTouches: this.currentTouches.size
        };
    }

    /**
     * Cleanup
     */
    async destroy() {
        this.stopTracking();
        this.currentTouches.clear();
        this.trajectories.clear();
        console.log('[GestureRecognizer] Destroyed');
    }

    // Public API for callbacks

    onGestureDetected(callback) {
        this.onGestureDetectedCallback = callback;
    }

    onGestureStart(callback) {
        this.onGestureStartCallback = callback;
    }

    onGestureEnd(callback) {
        this.onGestureEndCallback = callback;
    }
}


/**
 * Integration Example
 * Shows how to use GestureRecognizer with MultimodalRouter
 */
class GestureVoiceSystem {
    constructor(gestureRecognizer, multimodalRouter) {
        this.gestureRecognizer = gestureRecognizer;
        this.multimodalRouter = multimodalRouter;

        this._setupIntegration();
    }

    _setupIntegration() {
        // Listen for gestures
        this.gestureRecognizer.onGestureDetected((event) => {
            console.log('[GestureVoiceSystem] Gesture detected:', event.gesture.type);

            // Send to multimodal router
            this.multimodalRouter.processInput(
                'gesture',
                event.gesture.type,
                {
                    confidence: event.gesture.confidence,
                    velocity: event.gesture.velocity,
                    direction: event.gesture.direction
                }
            );
        });

        // Register gesture modality handler
        this.multimodalRouter.registerHandler(
            'gesture',
            {
                interpret: async (content) => {
                    // Gesture content is gesture type
                    const action = this._mapGestureToAction(content);

                    return {
                        action: action,
                        confidence: 0.9
                    };
                }
            }
        );
    }

    _mapGestureToAction(gestureType) {
        const mapping = {
            [GestureType.SWIPE_LEFT]: 'previous',
            [GestureType.SWIPE_RIGHT]: 'next',
            [GestureType.SWIPE_UP]: 'scroll_up',
            [GestureType.SWIPE_DOWN]: 'scroll_down',
            [GestureType.TAP]: 'select',
            [GestureType.DOUBLE_TAP]: 'open',
            [GestureType.LONG_PRESS]: 'context_menu',
            [GestureType.PINCH_IN]: 'zoom_out',
            [GestureType.PINCH_OUT]: 'zoom_in'
        };

        return mapping[gestureType] || 'unknown';
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        GestureRecognizer,
        GestureResult,
        GestureTrajectory,
        TouchPoint,
        GestureType,
        SpatialRelation,
        GestureVoiceSystem
    };
}
