/**
 * Multimodal Input Router - Visual + Voice Fusion
 * Milestone 4: Integrated multimodal interaction
 * Date: November 2025
 *
 * Features:
 * - Visual + voice input fusion
 * - Context-aware modality prioritization
 * - Conflict resolution between modalities
 * - State management across inputs
 * - Gesture + voice coordination
 * - Screen context + voice commands
 * - Attention tracking integration
 */

/**
 * Input Modality Types
 */
const InputModality = {
    VOICE: 'voice',
    VISUAL: 'visual',
    GESTURE: 'gesture',
    GAZE: 'gaze',
    TOUCH: 'touch'
};

/**
 * Multimodal Input Event
 */
class MultimodalInput {
    constructor(data) {
        this.id = data.id || `input_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.modality = data.modality;
        this.content = data.content;
        this.confidence = data.confidence || 1.0;
        this.timestamp = data.timestamp || new Date();
        this.metadata = data.metadata || {};
    }
}

/**
 * Fused Input (combined modalities)
 */
class FusedInput {
    constructor(data) {
        this.id = data.id || `fused_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.inputs = data.inputs || []; // MultimodalInput[]
        this.primaryModality = data.primaryModality;
        this.fusionStrategy = data.fusionStrategy; // 'early', 'late', 'hybrid'
        this.intent = data.intent || null;
        this.confidence = data.confidence || 0.0;
        this.timestamp = data.timestamp || new Date();
    }
}

/**
 * Multimodal Router
 * Routes and fuses inputs from multiple modalities
 */
class MultimodalRouter {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Fusion strategy
            fusionStrategy: config.fusionStrategy || 'hybrid', // 'early', 'late', 'hybrid'
            fusionWindow: config.fusionWindow || 1000, // ms to wait for multi-modal inputs

            // Modality priorities (higher = more important)
            modalityPriorities: config.modalityPriorities || {
                voice: 1.0,
                visual: 0.8,
                gesture: 0.7,
                gaze: 0.6,
                touch: 0.9
            },

            // Conflict resolution
            conflictResolution: config.conflictResolution || 'confidence', // 'confidence', 'priority', 'fusion'

            // Context awareness
            enableContextualPriority: config.enableContextualPriority !== false,

            ...config
        };

        // Input buffers
        this.inputBuffer = [];
        this.fusionQueue = [];

        // Current state
        this.activeModalities = new Set();
        this.lastFusedInput = null;

        // Modality handlers
        this.modalityHandlers = new Map();

        // Callbacks
        this.onInputReceivedCallback = null;
        this.onInputFusedCallback = null;
        this.onConflictDetectedCallback = null;

        // Metrics
        this.metrics = {
            totalInputs: 0,
            inputsByModality: {},
            totalFusions: 0,
            fusionsByStrategy: { early: 0, late: 0, hybrid: 0 },
            conflictsDetected: 0,
            avgFusionConfidence: 0.0
        };

        console.log('[MultimodalRouter] Initialized with strategy:', this.config.fusionStrategy);
    }

    /**
     * Register modality handler
     */
    registerHandler(modality, handler) {
        this.modalityHandlers.set(modality, handler);
        console.log('[MultimodalRouter] Registered handler for:', modality);
    }

    /**
     * Process input from any modality
     */
    async processInput(modality, content, metadata = {}) {
        if (!this.config.enabled) {
            return null;
        }

        const input = new MultimodalInput({
            modality: modality,
            content: content,
            confidence: metadata.confidence || 1.0,
            metadata: metadata
        });

        // Add to buffer
        this.inputBuffer.push(input);

        // Update metrics
        this.metrics.totalInputs++;
        this.metrics.inputsByModality[modality] =
            (this.metrics.inputsByModality[modality] || 0) + 1;

        // Track active modalities
        this.activeModalities.add(modality);

        // Emit event
        if (this.onInputReceivedCallback) {
            this.onInputReceivedCallback({
                type: 'input_received',
                input: input,
                timestamp: new Date()
            });
        }

        // Attempt fusion
        const fusedInput = await this._attemptFusion(input);

        // Clean old inputs from buffer
        this._cleanBuffer();

        console.log('[MultimodalRouter] Input processed:', modality,
            fusedInput ? '(fused)' : '(buffered)');

        return fusedInput;
    }

    /**
     * Attempt to fuse inputs
     */
    async _attemptFusion(triggerInput) {
        const now = Date.now();

        // Get recent inputs within fusion window
        const recentInputs = this.inputBuffer.filter(
            input => now - input.timestamp.getTime() <= this.config.fusionWindow
        );

        // Check if we have multiple modalities
        const modalities = new Set(recentInputs.map(i => i.modality));

        if (modalities.size < 2) {
            // Not enough modalities to fuse yet
            return null;
        }

        // Apply fusion strategy
        let fusedInput;

        switch (this.config.fusionStrategy) {
            case 'early':
                fusedInput = await this._earlyFusion(recentInputs);
                break;
            case 'late':
                fusedInput = await this._lateFusion(recentInputs);
                break;
            case 'hybrid':
                fusedInput = await this._hybridFusion(recentInputs);
                break;
            default:
                fusedInput = await this._hybridFusion(recentInputs);
        }

        if (fusedInput) {
            // Update metrics
            this.metrics.totalFusions++;
            this.metrics.fusionsByStrategy[this.config.fusionStrategy]++;
            this.metrics.avgFusionConfidence =
                (this.metrics.avgFusionConfidence * (this.metrics.totalFusions - 1) + fusedInput.confidence) /
                this.metrics.totalFusions;

            // Update last fused input
            this.lastFusedInput = fusedInput;

            // Remove fused inputs from buffer
            for (const input of recentInputs) {
                const index = this.inputBuffer.indexOf(input);
                if (index > -1) {
                    this.inputBuffer.splice(index, 1);
                }
            }

            // Emit event
            if (this.onInputFusedCallback) {
                this.onInputFusedCallback({
                    type: 'input_fused',
                    fusedInput: fusedInput,
                    timestamp: new Date()
                });
            }

            console.log('[MultimodalRouter] Fused input created:', fusedInput.intent);
        }

        return fusedInput;
    }

    /**
     * Early fusion (feature-level fusion)
     */
    async _earlyFusion(inputs) {
        // Combine features from all modalities before interpretation
        const voiceInputs = inputs.filter(i => i.modality === InputModality.VOICE);
        const visualInputs = inputs.filter(i => i.modality === InputModality.VISUAL);
        const gestureInputs = inputs.filter(i => i.modality === InputModality.GESTURE);

        // Example: "open that" (voice) + pointing at object (visual/gesture)
        if (voiceInputs.length > 0 && (visualInputs.length > 0 || gestureInputs.length > 0)) {
            const voiceCommand = voiceInputs[0].content;
            const visualTarget = visualInputs[0]?.content || gestureInputs[0]?.content;

            // Resolve deictic references ("that", "this", "there")
            const resolvedIntent = this._resolveDeicticReferences(voiceCommand, visualTarget);

            return new FusedInput({
                inputs: inputs,
                primaryModality: InputModality.VOICE,
                fusionStrategy: 'early',
                intent: resolvedIntent,
                confidence: this._calculateFusionConfidence(inputs)
            });
        }

        return null;
    }

    /**
     * Late fusion (decision-level fusion)
     */
    async _lateFusion(inputs) {
        // Interpret each modality independently, then combine interpretations
        const interpretations = [];

        for (const input of inputs) {
            const handler = this.modalityHandlers.get(input.modality);
            if (handler) {
                const interpretation = await handler.interpret(input.content);
                interpretations.push({
                    modality: input.modality,
                    interpretation: interpretation,
                    confidence: input.confidence
                });
            }
        }

        // Combine interpretations
        if (interpretations.length > 1) {
            // Check for conflicts
            const hasConflict = this._detectConflicts(interpretations);

            if (hasConflict) {
                return this._resolveConflict(interpretations, inputs);
            }

            // No conflict - combine interpretations
            const combinedIntent = this._combineInterpretations(interpretations);

            return new FusedInput({
                inputs: inputs,
                primaryModality: this._selectPrimaryModality(inputs),
                fusionStrategy: 'late',
                intent: combinedIntent,
                confidence: this._calculateFusionConfidence(inputs)
            });
        }

        return null;
    }

    /**
     * Hybrid fusion (combination of early and late)
     */
    async _hybridFusion(inputs) {
        // Try early fusion first (for deictic references)
        const earlyResult = await this._earlyFusion(inputs);
        if (earlyResult && earlyResult.confidence >= 0.7) {
            earlyResult.fusionStrategy = 'hybrid';
            return earlyResult;
        }

        // Fall back to late fusion
        const lateResult = await this._lateFusion(inputs);
        if (lateResult) {
            lateResult.fusionStrategy = 'hybrid';
            return lateResult;
        }

        return null;
    }

    /**
     * Resolve deictic references ("that", "this", "there")
     */
    _resolveDeicticReferences(voiceCommand, visualTarget) {
        const command = voiceCommand.toLowerCase();

        // Deictic patterns
        const deicticWords = ['that', 'this', 'there', 'here', 'it', 'them'];

        for (const word of deicticWords) {
            if (command.includes(word)) {
                // Replace with visual target
                return {
                    action: this._extractAction(command),
                    target: visualTarget,
                    resolved: true
                };
            }
        }

        // No deictic reference
        return {
            action: this._extractAction(command),
            target: null,
            resolved: false
        };
    }

    /**
     * Extract action from voice command
     */
    _extractAction(command) {
        const actionWords = ['open', 'close', 'select', 'delete', 'move', 'create'];

        for (const action of actionWords) {
            if (command.includes(action)) {
                return action;
            }
        }

        return 'unknown';
    }

    /**
     * Detect conflicts between interpretations
     */
    _detectConflicts(interpretations) {
        if (interpretations.length < 2) {
            return false;
        }

        // Simple conflict detection: different actions for same intent
        const actions = interpretations.map(i => i.interpretation.action);
        const uniqueActions = new Set(actions);

        return uniqueActions.size > 1;
    }

    /**
     * Resolve conflict between interpretations
     */
    _resolveConflict(interpretations, inputs) {
        this.metrics.conflictsDetected++;

        // Emit conflict event
        if (this.onConflictDetectedCallback) {
            this.onConflictDetectedCallback({
                type: 'conflict_detected',
                interpretations: interpretations,
                timestamp: new Date()
            });
        }

        let resolved;

        switch (this.config.conflictResolution) {
            case 'confidence':
                // Choose interpretation with highest confidence
                resolved = interpretations.reduce((best, current) =>
                    current.confidence > best.confidence ? current : best
                );
                break;

            case 'priority':
                // Choose interpretation from highest priority modality
                resolved = interpretations.reduce((best, current) => {
                    const currentPriority = this.config.modalityPriorities[current.modality] || 0;
                    const bestPriority = this.config.modalityPriorities[best.modality] || 0;
                    return currentPriority > bestPriority ? current : best;
                });
                break;

            case 'fusion':
                // Attempt to fuse conflicting interpretations
                resolved = this._fuseConflictingInterpretations(interpretations);
                break;

            default:
                resolved = interpretations[0];
        }

        console.log('[MultimodalRouter] Conflict resolved via:', this.config.conflictResolution);

        return new FusedInput({
            inputs: inputs,
            primaryModality: resolved.modality,
            fusionStrategy: 'late',
            intent: resolved.interpretation,
            confidence: resolved.confidence
        });
    }

    /**
     * Fuse conflicting interpretations
     */
    _fuseConflictingInterpretations(interpretations) {
        // Simple fusion: weighted average of confidences
        let totalWeight = 0;
        const fused = {};

        for (const interp of interpretations) {
            const weight = interp.confidence * (this.config.modalityPriorities[interp.modality] || 1.0);
            totalWeight += weight;

            // Combine interpretation fields
            for (const [key, value] of Object.entries(interp.interpretation)) {
                if (!fused[key]) {
                    fused[key] = { value: value, weight: weight };
                } else {
                    // Keep value with higher weight
                    if (weight > fused[key].weight) {
                        fused[key] = { value: value, weight: weight };
                    }
                }
            }
        }

        // Extract values
        const result = {};
        for (const [key, data] of Object.entries(fused)) {
            result[key] = data.value;
        }

        return {
            modality: 'fused',
            interpretation: result,
            confidence: totalWeight / interpretations.length
        };
    }

    /**
     * Combine interpretations (no conflict)
     */
    _combineInterpretations(interpretations) {
        const combined = {};

        for (const interp of interpretations) {
            Object.assign(combined, interp.interpretation);
        }

        return combined;
    }

    /**
     * Select primary modality
     */
    _selectPrimaryModality(inputs) {
        let primary = inputs[0];

        for (const input of inputs) {
            const currentPriority = this.config.modalityPriorities[input.modality] || 0;
            const primaryPriority = this.config.modalityPriorities[primary.modality] || 0;

            if (currentPriority > primaryPriority) {
                primary = input;
            }
        }

        return primary.modality;
    }

    /**
     * Calculate fusion confidence
     */
    _calculateFusionConfidence(inputs) {
        if (inputs.length === 0) return 0.0;

        // Weighted average based on modality priorities
        let totalWeight = 0;
        let weightedSum = 0;

        for (const input of inputs) {
            const weight = this.config.modalityPriorities[input.modality] || 1.0;
            weightedSum += input.confidence * weight;
            totalWeight += weight;
        }

        return totalWeight > 0 ? weightedSum / totalWeight : 0.0;
    }

    /**
     * Clean old inputs from buffer
     */
    _cleanBuffer() {
        const now = Date.now();
        const cutoff = this.config.fusionWindow * 2; // Keep 2x fusion window

        this.inputBuffer = this.inputBuffer.filter(
            input => now - input.timestamp.getTime() <= cutoff
        );
    }

    /**
     * Get active modalities
     */
    getActiveModalities() {
        return Array.from(this.activeModalities);
    }

    /**
     * Get last fused input
     */
    getLastFusedInput() {
        return this.lastFusedInput;
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            bufferSize: this.inputBuffer.length,
            activeModalities: this.activeModalities.size,
            fusionRate: this.metrics.totalInputs > 0
                ? this.metrics.totalFusions / this.metrics.totalInputs
                : 0.0,
            conflictRate: this.metrics.totalFusions > 0
                ? this.metrics.conflictsDetected / this.metrics.totalFusions
                : 0.0
        };
    }

    // Public API for callbacks

    onInputReceived(callback) {
        this.onInputReceivedCallback = callback;
    }

    onInputFused(callback) {
        this.onInputFusedCallback = callback;
    }

    onConflictDetected(callback) {
        this.onConflictDetectedCallback = callback;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MultimodalRouter,
        MultimodalInput,
        FusedInput,
        InputModality
    };
}
