/**
 * Facial Emotion Analyzer - Visual Emotion Detection
 * Milestone 3 Enhancement: Face analysis integration
 * Date: November 2025
 *
 * Features:
 * - Facial expression recognition (7 universal emotions)
 * - Action Unit (AU) detection (FACS - Facial Action Coding System)
 * - Head pose estimation (attention detection)
 * - Eye gaze tracking (engagement)
 * - Micro-expression detection
 * - Continuous face tracking
 * - Integration stubs for face-api.js, MediaPipe, TensorFlow.js
 *
 * Supported Libraries (optional integration):
 * - face-api.js: https://github.com/justadudewhohacks/face-api.js
 * - MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh
 * - TensorFlow.js FaceLandmarkDetection: https://github.com/tensorflow/tfjs-models
 */

/**
 * Facial Emotions (Ekman's 7 Universal Emotions)
 */
const FacialEmotion = {
    HAPPY: 'happy',
    SAD: 'sad',
    ANGRY: 'angry',
    FEARFUL: 'fearful',
    DISGUSTED: 'disgusted',
    SURPRISED: 'surprised',
    NEUTRAL: 'neutral'
};

/**
 * Action Units (FACS - subset of common expressions)
 */
const ActionUnit = {
    AU1: { id: 1, name: 'Inner Brow Raiser', emotion: 'sad/fearful' },
    AU2: { id: 2, name: 'Outer Brow Raiser', emotion: 'surprised' },
    AU4: { id: 4, name: 'Brow Lowerer', emotion: 'angry' },
    AU5: { id: 5, name: 'Upper Lid Raiser', emotion: 'surprised/fearful' },
    AU6: { id: 6, name: 'Cheek Raiser', emotion: 'happy' },
    AU7: { id: 7, name: 'Lid Tightener', emotion: 'angry' },
    AU9: { id: 9, name: 'Nose Wrinkler', emotion: 'disgusted' },
    AU10: { id: 10, name: 'Upper Lip Raiser', emotion: 'disgusted' },
    AU12: { id: 12, name: 'Lip Corner Puller', emotion: 'happy' },
    AU15: { id: 15, name: 'Lip Corner Depressor', emotion: 'sad' },
    AU17: { id: 17, name: 'Chin Raiser', emotion: 'sad' },
    AU20: { id: 20, name: 'Lip Stretcher', emotion: 'fearful' },
    AU23: { id: 23, name: 'Lip Tightener', emotion: 'angry' },
    AU25: { id: 25, name: 'Lips Part', emotion: 'surprised' },
    AU26: { id: 26, name: 'Jaw Drop', emotion: 'surprised' }
};

/**
 * Facial Emotion Result
 */
class FacialEmotionResult {
    constructor(data) {
        // Primary emotion
        this.emotion = data.emotion || FacialEmotion.NEUTRAL;
        this.confidence = data.confidence || 0.0;

        // All emotion probabilities
        this.emotionProbabilities = data.emotionProbabilities || {};

        // Action Units detected
        this.actionUnits = data.actionUnits || [];

        // Face metrics
        this.faceDetected = data.faceDetected !== false;
        this.faceConfidence = data.faceConfidence || 0.0;

        // Head pose (rotation in degrees)
        this.headPose = data.headPose || { yaw: 0, pitch: 0, roll: 0 };

        // Gaze direction
        this.gazeDirection = data.gazeDirection || { x: 0, y: 0 };

        // Engagement metrics
        this.eyeContact = data.eyeContact || false;
        this.attention = data.attention || 0.5; // 0.0-1.0

        // Micro-expressions
        this.microExpression = data.microExpression || null;

        // Metadata
        this.timestamp = data.timestamp || new Date();
        this.processingTimeMs = data.processingTimeMs || 0;
    }

    /**
     * Get top N emotions by probability
     */
    getTopEmotions(n = 3) {
        return Object.entries(this.emotionProbabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, n)
            .map(([emotion, prob]) => ({ emotion, probability: prob }));
    }

    /**
     * Check if specific emotion is present (above threshold)
     */
    hasEmotion(emotion, threshold = 0.3) {
        return (this.emotionProbabilities[emotion] || 0) >= threshold;
    }
}

/**
 * Facial Emotion Analyzer
 * Detects emotions from facial expressions using computer vision
 */
class FacialEmotionAnalyzer {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Detection settings
            detectionIntervalMs: config.detectionIntervalMs || 100, // 10 FPS
            minFaceConfidence: config.minFaceConfidence || 0.7,
            minEmotionConfidence: config.minEmotionConfidence || 0.5,

            // CV library selection
            library: config.library || 'stub', // 'stub', 'face-api', 'mediapipe', 'tfjs'

            // Model settings
            modelPath: config.modelPath || null,
            inputSize: config.inputSize || 224, // pixels

            // Feature flags
            enableActionUnits: config.enableActionUnits !== false,
            enableHeadPose: config.enableHeadPose !== false,
            enableGazeTracking: config.enableGazeTracking || false,
            enableMicroExpressions: config.enableMicroExpressions || false,

            // Performance
            useWebGL: config.useWebGL !== false,

            ...config
        };

        // Detection state
        this.isAnalyzing = false;
        this.videoElement = null;
        this.canvasElement = null;
        this.detectionTimer = null;

        // CV model (will be loaded based on library)
        this.model = null;
        this.modelLoaded = false;

        // Recent detections (for micro-expression analysis)
        this.detectionHistory = [];
        this.maxHistorySize = 10;

        // Callbacks
        this.onEmotionDetectedCallback = null;
        this.onFaceDetectedCallback = null;
        this.onFaceLostCallback = null;

        // Metrics
        this.metrics = {
            totalDetections: 0,
            avgProcessingTimeMs: 0,
            emotionCounts: {},
            faceDetectionRate: 0.0,
            lastDetectionTime: null
        };

        console.log('[FacialEmotionAnalyzer] Initialized with library:', this.config.library);
    }

    /**
     * Initialize video capture and load model
     */
    async initialize(videoElement = null) {
        if (!this.config.enabled) {
            console.log('[FacialEmotionAnalyzer] Disabled, skipping initialization');
            return false;
        }

        try {
            // Setup video element
            if (videoElement) {
                this.videoElement = videoElement;
            } else {
                // Create video element for camera
                this.videoElement = document.createElement('video');
                this.videoElement.autoplay = true;
                this.videoElement.style.display = 'none';
                document.body.appendChild(this.videoElement);

                // Request camera access
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                });

                this.videoElement.srcObject = stream;
                await this.videoElement.play();
            }

            // Create canvas for processing
            this.canvasElement = document.createElement('canvas');
            this.canvasElement.width = this.config.inputSize;
            this.canvasElement.height = this.config.inputSize;

            // Load CV model based on library
            await this._loadModel();

            console.log('[FacialEmotionAnalyzer] Initialized successfully');
            return true;

        } catch (error) {
            console.error('[FacialEmotionAnalyzer] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Start continuous emotion analysis
     */
    async startAnalysis() {
        if (!this.config.enabled || this.isAnalyzing || !this.modelLoaded) {
            return false;
        }

        this.isAnalyzing = true;

        // Start detection loop
        this._startDetectionLoop();

        console.log('[FacialEmotionAnalyzer] Analysis started');
        return true;
    }

    /**
     * Stop emotion analysis
     */
    stopAnalysis() {
        if (!this.isAnalyzing) {
            return false;
        }

        this.isAnalyzing = false;

        if (this.detectionTimer) {
            clearInterval(this.detectionTimer);
            this.detectionTimer = null;
        }

        console.log('[FacialEmotionAnalyzer] Analysis stopped');
        return true;
    }

    /**
     * Analyze single frame
     */
    async analyzeFrame() {
        if (!this.videoElement || !this.modelLoaded) {
            return null;
        }

        const startTime = Date.now();

        // Capture frame to canvas
        const ctx = this.canvasElement.getContext('2d');
        ctx.drawImage(
            this.videoElement,
            0, 0,
            this.canvasElement.width,
            this.canvasElement.height
        );

        // Run detection based on library
        const result = await this._detectEmotions(this.canvasElement);

        if (result) {
            result.processingTimeMs = Date.now() - startTime;

            // Add to history
            this.detectionHistory.push(result);
            if (this.detectionHistory.length > this.maxHistorySize) {
                this.detectionHistory.shift();
            }

            // Detect micro-expressions
            if (this.config.enableMicroExpressions) {
                result.microExpression = this._detectMicroExpression();
            }

            // Update metrics
            this._updateMetrics(result);

            // Emit events
            this._emitDetectionEvents(result);
        }

        return result;
    }

    /**
     * Start detection loop
     */
    _startDetectionLoop() {
        if (!this.isAnalyzing) {
            return;
        }

        const detect = async () => {
            if (!this.isAnalyzing) {
                return; // Stop loop
            }

            await this.analyzeFrame();
        };

        // Initial detection
        detect();

        // Periodic detection
        this.detectionTimer = setInterval(detect, this.config.detectionIntervalMs);
    }

    /**
     * Load CV model based on library
     */
    async _loadModel() {
        console.log('[FacialEmotionAnalyzer] Loading model for library:', this.config.library);

        switch (this.config.library) {
            case 'face-api':
                await this._loadFaceApiModel();
                break;
            case 'mediapipe':
                await this._loadMediaPipeModel();
                break;
            case 'tfjs':
                await this._loadTensorFlowModel();
                break;
            case 'stub':
            default:
                this._loadStubModel();
                break;
        }

        this.modelLoaded = true;
        console.log('[FacialEmotionAnalyzer] Model loaded successfully');
    }

    /**
     * STUB: Load face-api.js model
     * In production: import face-api.js and load models
     */
    async _loadFaceApiModel() {
        // STUB: In production, use actual face-api.js:
        /*
        await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
        await faceapi.nets.faceExpressionNet.loadFromUri('/models');
        await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
        this.model = faceapi;
        */

        console.log('[FacialEmotionAnalyzer] face-api.js stub loaded (replace with actual library)');
        this.model = { type: 'face-api-stub' };
    }

    /**
     * STUB: Load MediaPipe model
     * In production: import MediaPipe Face Mesh
     */
    async _loadMediaPipeModel() {
        // STUB: In production, use actual MediaPipe:
        /*
        const FaceMesh = window.FaceMesh;
        this.model = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        await this.model.initialize();
        */

        console.log('[FacialEmotionAnalyzer] MediaPipe stub loaded (replace with actual library)');
        this.model = { type: 'mediapipe-stub' };
    }

    /**
     * STUB: Load TensorFlow.js model
     * In production: load TF.js FaceLandmarkDetection
     */
    async _loadTensorFlowModel() {
        // STUB: In production, use actual TensorFlow.js:
        /*
        const faceLandmarksDetection = require('@tensorflow-models/face-landmarks-detection');
        this.model = await faceLandmarksDetection.load(
            faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
        );
        */

        console.log('[FacialEmotionAnalyzer] TensorFlow.js stub loaded (replace with actual library)');
        this.model = { type: 'tfjs-stub' };
    }

    /**
     * Load stub model (for testing without CV libraries)
     */
    _loadStubModel() {
        console.log('[FacialEmotionAnalyzer] Using stub model (no actual face detection)');
        this.model = { type: 'stub' };
    }

    /**
     * Detect emotions from frame
     */
    async _detectEmotions(canvas) {
        if (!this.model) {
            return null;
        }

        switch (this.model.type) {
            case 'face-api-stub':
                return this._detectWithFaceApi(canvas);
            case 'mediapipe-stub':
                return this._detectWithMediaPipe(canvas);
            case 'tfjs-stub':
                return this._detectWithTensorFlow(canvas);
            case 'stub':
            default:
                return this._detectWithStub(canvas);
        }
    }

    /**
     * STUB: Detect with face-api.js
     */
    async _detectWithFaceApi(canvas) {
        // STUB: In production:
        /*
        const detections = await faceapi
            .detectSingleFace(canvas, new faceapi.TinyFaceDetectorOptions())
            .withFaceExpressions()
            .withFaceLandmarks();

        if (!detections) {
            return this._createEmptyResult();
        }

        const expressions = detections.expressions;
        const emotion = expressions.asSortedArray()[0];

        return new FacialEmotionResult({
            emotion: emotion.expression,
            confidence: emotion.probability,
            emotionProbabilities: {
                happy: expressions.happy,
                sad: expressions.sad,
                angry: expressions.angry,
                fearful: expressions.fearful,
                disgusted: expressions.disgusted,
                surprised: expressions.surprised,
                neutral: expressions.neutral
            },
            faceDetected: true,
            faceConfidence: detections.detection.score
        });
        */

        // STUB: Return simulated result
        return this._detectWithStub(canvas);
    }

    /**
     * STUB: Detect with MediaPipe
     */
    async _detectWithMediaPipe(canvas) {
        // STUB: In production, use MediaPipe face mesh + emotion classifier
        return this._detectWithStub(canvas);
    }

    /**
     * STUB: Detect with TensorFlow.js
     */
    async _detectWithTensorFlow(canvas) {
        // STUB: In production, use TF.js face landmarks + emotion model
        return this._detectWithStub(canvas);
    }

    /**
     * Stub detection (simulated emotions for testing)
     */
    _detectWithStub(canvas) {
        // Simulate random emotions for demo purposes
        const emotions = Object.values(FacialEmotion);
        const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];

        const probabilities = {};
        let remainingProb = 1.0;

        // Assign high probability to primary emotion
        probabilities[randomEmotion] = 0.5 + Math.random() * 0.3;
        remainingProb -= probabilities[randomEmotion];

        // Distribute remaining probability
        for (const emotion of emotions) {
            if (emotion !== randomEmotion) {
                const prob = Math.random() * remainingProb / (emotions.length - 1);
                probabilities[emotion] = prob;
            }
        }

        // Simulate Action Units
        const actionUnits = [];
        if (randomEmotion === FacialEmotion.HAPPY) {
            actionUnits.push(ActionUnit.AU6, ActionUnit.AU12);
        } else if (randomEmotion === FacialEmotion.SAD) {
            actionUnits.push(ActionUnit.AU1, ActionUnit.AU15);
        } else if (randomEmotion === FacialEmotion.ANGRY) {
            actionUnits.push(ActionUnit.AU4, ActionUnit.AU7);
        }

        return new FacialEmotionResult({
            emotion: randomEmotion,
            confidence: probabilities[randomEmotion],
            emotionProbabilities: probabilities,
            actionUnits: actionUnits,
            faceDetected: Math.random() > 0.1, // 90% face detection rate
            faceConfidence: 0.8 + Math.random() * 0.2,
            headPose: {
                yaw: (Math.random() - 0.5) * 30,
                pitch: (Math.random() - 0.5) * 20,
                roll: (Math.random() - 0.5) * 15
            },
            gazeDirection: {
                x: (Math.random() - 0.5) * 2,
                y: (Math.random() - 0.5) * 2
            },
            eyeContact: Math.random() > 0.3,
            attention: 0.5 + Math.random() * 0.5
        });
    }

    /**
     * Detect micro-expressions (rapid facial changes)
     */
    _detectMicroExpression() {
        if (this.detectionHistory.length < 3) {
            return null;
        }

        // Check for rapid emotion changes
        const recent = this.detectionHistory.slice(-3);
        const emotions = recent.map(r => r.emotion);

        // Micro-expression: emotion changes within <500ms and returns
        if (emotions[0] !== emotions[1] && emotions[1] !== emotions[2]) {
            return {
                detected: true,
                emotion: emotions[1],
                duration: recent[2].timestamp - recent[0].timestamp,
                confidence: recent[1].confidence
            };
        }

        return null;
    }

    /**
     * Create empty result when no face detected
     */
    _createEmptyResult() {
        return new FacialEmotionResult({
            emotion: FacialEmotion.NEUTRAL,
            confidence: 0.0,
            faceDetected: false
        });
    }

    /**
     * Update metrics
     */
    _updateMetrics(result) {
        this.metrics.totalDetections++;

        this.metrics.avgProcessingTimeMs =
            (this.metrics.avgProcessingTimeMs * (this.metrics.totalDetections - 1) +
                result.processingTimeMs) / this.metrics.totalDetections;

        if (result.faceDetected) {
            this.metrics.emotionCounts[result.emotion] =
                (this.metrics.emotionCounts[result.emotion] || 0) + 1;
        }

        const detectedCount = this.detectionHistory.filter(r => r.faceDetected).length;
        this.metrics.faceDetectionRate = detectedCount / this.detectionHistory.length;

        this.metrics.lastDetectionTime = result.timestamp;
    }

    /**
     * Emit detection events
     */
    _emitDetectionEvents(result) {
        if (result.faceDetected && this.onFaceDetectedCallback) {
            this.onFaceDetectedCallback({
                type: 'face_detected',
                result: result,
                timestamp: new Date()
            });
        } else if (!result.faceDetected && this.onFaceLostCallback) {
            this.onFaceLostCallback({
                type: 'face_lost',
                timestamp: new Date()
            });
        }

        if (result.confidence >= this.config.minEmotionConfidence &&
            this.onEmotionDetectedCallback) {
            this.onEmotionDetectedCallback({
                type: 'emotion_detected',
                result: result,
                timestamp: new Date()
            });
        }
    }

    /**
     * Get recent emotion (smoothed over last N frames)
     */
    getRecentEmotion(frames = 5) {
        if (this.detectionHistory.length === 0) {
            return null;
        }

        const recent = this.detectionHistory.slice(-frames);
        const emotionCounts = {};

        for (const result of recent) {
            if (result.faceDetected) {
                emotionCounts[result.emotion] = (emotionCounts[result.emotion] || 0) + 1;
            }
        }

        // Return most common emotion
        return Object.entries(emotionCounts)
            .sort((a, b) => b[1] - a[1])[0]?.[0] || FacialEmotion.NEUTRAL;
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            isAnalyzing: this.isAnalyzing,
            modelLoaded: this.modelLoaded,
            detectionHistorySize: this.detectionHistory.length
        };
    }

    /**
     * Cleanup
     */
    async destroy() {
        this.stopAnalysis();

        // Stop video stream
        if (this.videoElement && this.videoElement.srcObject) {
            const stream = this.videoElement.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            this.videoElement.srcObject = null;
        }

        this.detectionHistory = [];

        console.log('[FacialEmotionAnalyzer] Destroyed');
    }

    // Public API for callbacks

    onEmotionDetected(callback) {
        this.onEmotionDetectedCallback = callback;
    }

    onFaceDetected(callback) {
        this.onFaceDetectedCallback = callback;
    }

    onFaceLost(callback) {
        this.onFaceLostCallback = callback;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        FacialEmotionAnalyzer,
        FacialEmotionResult,
        FacialEmotion,
        ActionUnit
    };
}
