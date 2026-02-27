/**
 * Voice Profile System - Speaker Identification
 * Milestone 2: Multi-user support with speaker recognition
 * Date: November 2025
 *
 * Features:
 * - Speaker enrollment (voice print creation)
 * - Speaker recognition (who is speaking)
 * - Multi-user support
 * - Speaker-specific preferences
 * - Voice biometric features
 * - Confidence scoring
 * - Profile persistence
 */

/**
 * Voice Print (Biometric Features)
 */
class VoicePrint {
    constructor(data) {
        this.id = data.id || `print_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.speakerId = data.speakerId;

        // Acoustic features
        this.pitchMean = data.pitchMean || 0.0;
        this.pitchVariance = data.pitchVariance || 0.0;
        this.energyMean = data.energyMean || 0.0;
        this.energyVariance = data.energyVariance || 0.0;
        this.speechRate = data.speechRate || 0.0;
        this.formants = data.formants || []; // F1, F2, F3 frequencies

        // Statistical features
        this.sampleCount = data.sampleCount || 0;
        this.createdAt = data.createdAt || new Date();
        this.updatedAt = data.updatedAt || new Date();
    }

    /**
     * Calculate similarity to another voice print
     */
    similarity(other) {
        // Euclidean distance in feature space (simplified)
        const pitchDist = Math.abs(this.pitchMean - other.pitchMean) / 100; // Normalize
        const energyDist = Math.abs(this.energyMean - other.energyMean);
        const rateDist = Math.abs(this.speechRate - other.speechRate) / 100;

        const distance = Math.sqrt(
            pitchDist * pitchDist +
            energyDist * energyDist +
            rateDist * rateDist
        );

        // Convert distance to similarity (0.0-1.0)
        return Math.max(0.0, 1.0 - distance);
    }

    /**
     * Update voice print with new sample
     */
    update(newFeatures) {
        // Incremental mean update
        const n = this.sampleCount;

        this.pitchMean = (this.pitchMean * n + newFeatures.pitchMean) / (n + 1);
        this.pitchVariance = (this.pitchVariance * n + newFeatures.pitchVariance) / (n + 1);
        this.energyMean = (this.energyMean * n + newFeatures.energyMean) / (n + 1);
        this.energyVariance = (this.energyVariance * n + newFeatures.energyVariance) / (n + 1);
        this.speechRate = (this.speechRate * n + newFeatures.speechRate) / (n + 1);

        this.sampleCount++;
        this.updatedAt = new Date();
    }
}

/**
 * Speaker Profile
 */
class SpeakerProfile {
    constructor(data) {
        this.id = data.id || `speaker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.name = data.name || 'Unknown Speaker';
        this.voicePrint = data.voicePrint || null;

        // Preferences
        this.preferences = data.preferences || {
            language: 'en',
            theme: 'light',
            activeThreads: [],
            customSettings: {}
        };

        // Metadata
        this.enrollmentDate = data.enrollmentDate || new Date();
        this.lastSeen = data.lastSeen || null;
        this.recognitionCount = data.recognitionCount || 0;
    }

    /**
     * Update last seen timestamp
     */
    updateLastSeen() {
        this.lastSeen = new Date();
        this.recognitionCount++;
    }
}

/**
 * Recognition Result
 */
class RecognitionResult {
    constructor(data) {
        this.recognized = data.recognized || false;
        this.speaker = data.speaker || null;
        this.confidence = data.confidence || 0.0;
        this.candidates = data.candidates || []; // Other possible speakers
        this.timestamp = data.timestamp || new Date();
    }
}

/**
 * Voice Profile Manager
 * Manages speaker enrollment and recognition
 */
class VoiceProfileManager {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Recognition settings
            recognitionThreshold: config.recognitionThreshold || 0.7,
            minEnrollmentSamples: config.minEnrollmentSamples || 3,
            maxProfiles: config.maxProfiles || 10,

            // Feature extraction
            minSpeechDuration: config.minSpeechDuration || 1000, // ms

            // Persistence
            persistProfiles: config.persistProfiles !== false,
            storageKey: config.storageKey || 'hololoom_voice_profiles',

            ...config
        };

        // Speaker profiles
        this.profiles = new Map();
        this.currentSpeaker = null;
        this.enrollmentInProgress = null;

        // Audio analysis
        this.audioContext = null;
        this.analyser = null;

        // Callbacks
        this.onSpeakerRecognizedCallback = null;
        this.onSpeakerChangedCallback = null;
        this.onEnrollmentCompleteCallback = null;

        // Metrics
        this.metrics = {
            totalRecognitions: 0,
            successfulRecognitions: 0,
            avgConfidence: 0.0,
            speakerSwitches: 0,
            enrolledSpeakers: 0
        };

        // Load persisted profiles
        if (this.config.persistProfiles) {
            this._loadProfiles();
        }

        console.log('[VoiceProfileManager] Initialized with', this.profiles.size, 'profiles');
    }

    /**
     * Initialize audio analysis
     */
    async initialize(audioStream) {
        if (!this.config.enabled) {
            console.log('[VoiceProfileManager] Disabled, skipping initialization');
            return false;
        }

        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create analyser for feature extraction
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            // Connect audio stream
            const source = this.audioContext.createMediaStreamSource(audioStream);
            source.connect(this.analyser);

            console.log('[VoiceProfileManager] Audio analysis initialized');
            return true;

        } catch (error) {
            console.error('[VoiceProfileManager] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Enroll new speaker
     */
    async enrollSpeaker(name) {
        if (this.profiles.size >= this.config.maxProfiles) {
            throw new Error(`Maximum ${this.config.maxProfiles} profiles reached`);
        }

        // Create new speaker profile
        const profile = new SpeakerProfile({ name: name });

        // Start enrollment process
        this.enrollmentInProgress = {
            profile: profile,
            samples: [],
            startTime: Date.now()
        };

        console.log('[VoiceProfileManager] Enrollment started for:', name);

        return profile.id;
    }

    /**
     * Add enrollment sample
     */
    async addEnrollmentSample(audioFeatures, transcript) {
        if (!this.enrollmentInProgress) {
            throw new Error('No enrollment in progress');
        }

        // Extract voice print features
        const features = this._extractVoiceFeatures(audioFeatures, transcript);

        // Add to samples
        this.enrollmentInProgress.samples.push(features);

        console.log('[VoiceProfileManager] Enrollment sample added',
            `(${this.enrollmentInProgress.samples.length}/${this.config.minEnrollmentSamples})`);

        // Check if we have enough samples
        if (this.enrollmentInProgress.samples.length >= this.config.minEnrollmentSamples) {
            return await this._completeEnrollment();
        }

        return null;
    }

    /**
     * Complete enrollment
     */
    async _completeEnrollment() {
        if (!this.enrollmentInProgress) {
            return null;
        }

        // Average all samples to create voice print
        const avgFeatures = this._averageSamples(this.enrollmentInProgress.samples);
        const voicePrint = new VoicePrint({
            speakerId: this.enrollmentInProgress.profile.id,
            ...avgFeatures,
            sampleCount: this.enrollmentInProgress.samples.length
        });

        // Set voice print
        this.enrollmentInProgress.profile.voicePrint = voicePrint;

        // Add to profiles
        const profile = this.enrollmentInProgress.profile;
        this.profiles.set(profile.id, profile);

        // Update metrics
        this.metrics.enrolledSpeakers = this.profiles.size;

        // Save profiles
        if (this.config.persistProfiles) {
            this._saveProfiles();
        }

        // Emit event
        if (this.onEnrollmentCompleteCallback) {
            this.onEnrollmentCompleteCallback({
                type: 'enrollment_complete',
                profile: profile,
                timestamp: new Date()
            });
        }

        console.log('[VoiceProfileManager] Enrollment complete for:', profile.name);

        // Clear enrollment state
        this.enrollmentInProgress = null;

        return profile;
    }

    /**
     * Recognize speaker from audio
     */
    async recognizeSpeaker(audioFeatures, transcript) {
        if (!this.config.enabled || this.profiles.size === 0) {
            return new RecognitionResult({ recognized: false });
        }

        // Extract voice features
        const features = this._extractVoiceFeatures(audioFeatures, transcript);
        const queryPrint = new VoicePrint({
            speakerId: null,
            ...features
        });

        // Compare with all enrolled speakers
        const similarities = [];
        for (const [id, profile] of this.profiles.entries()) {
            if (!profile.voicePrint) continue;

            const similarity = queryPrint.similarity(profile.voicePrint);
            similarities.push({
                speaker: profile,
                similarity: similarity
            });
        }

        // Sort by similarity
        similarities.sort((a, b) => b.similarity - a.similarity);

        // Check if top match exceeds threshold
        const topMatch = similarities[0];
        const recognized = topMatch && topMatch.similarity >= this.config.recognitionThreshold;

        // Update metrics
        this.metrics.totalRecognitions++;
        if (recognized) {
            this.metrics.successfulRecognitions++;
            this.metrics.avgConfidence =
                (this.metrics.avgConfidence * (this.metrics.successfulRecognitions - 1) + topMatch.similarity) /
                this.metrics.successfulRecognitions;
        }

        // Create result
        const result = new RecognitionResult({
            recognized: recognized,
            speaker: recognized ? topMatch.speaker : null,
            confidence: topMatch ? topMatch.similarity : 0.0,
            candidates: similarities.slice(0, 3).map(s => ({
                speaker: s.speaker.name,
                confidence: s.similarity
            }))
        });

        // Update current speaker
        if (recognized) {
            const prevSpeaker = this.currentSpeaker;
            this.currentSpeaker = topMatch.speaker;
            topMatch.speaker.updateLastSeen();

            // Check for speaker change
            if (!prevSpeaker || prevSpeaker.id !== topMatch.speaker.id) {
                this.metrics.speakerSwitches++;

                // Emit speaker changed event
                if (this.onSpeakerChangedCallback) {
                    this.onSpeakerChangedCallback({
                        type: 'speaker_changed',
                        previousSpeaker: prevSpeaker,
                        currentSpeaker: topMatch.speaker,
                        timestamp: new Date()
                    });
                }
            }

            // Emit recognized event
            if (this.onSpeakerRecognizedCallback) {
                this.onSpeakerRecognizedCallback({
                    type: 'speaker_recognized',
                    result: result,
                    timestamp: new Date()
                });
            }

            // Optionally update voice print (adaptive learning)
            if (result.confidence >= 0.8) {
                topMatch.speaker.voicePrint.update(features);
            }
        }

        console.log('[VoiceProfileManager] Recognition result:',
            recognized ? `${topMatch.speaker.name} (${topMatch.similarity.toFixed(2)})` : 'Unknown');

        return result;
    }

    /**
     * Extract voice features from audio
     */
    _extractVoiceFeatures(audioFeatures, transcript) {
        // If audio features provided, use them
        if (audioFeatures) {
            return {
                pitchMean: audioFeatures.pitch || 0.0,
                pitchVariance: audioFeatures.pitchVariation || 0.0,
                energyMean: audioFeatures.energy || 0.0,
                energyVariance: audioFeatures.energyVariance || 0.0,
                speechRate: this._estimateSpeechRate(transcript)
            };
        }

        // Otherwise extract from analyser
        if (!this.analyser) {
            return {
                pitchMean: 0.0,
                pitchVariance: 0.0,
                energyMean: 0.0,
                energyVariance: 0.0,
                speechRate: this._estimateSpeechRate(transcript)
            };
        }

        // Extract features (placeholder - real implementation would use DSP)
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);

        // Calculate mean and variance
        let sum = 0;
        let sumSquares = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
            sumSquares += dataArray[i] * dataArray[i];
        }

        const mean = sum / dataArray.length;
        const variance = (sumSquares / dataArray.length) - (mean * mean);

        return {
            pitchMean: mean,
            pitchVariance: variance,
            energyMean: mean / 255.0,
            energyVariance: variance / (255.0 * 255.0),
            speechRate: this._estimateSpeechRate(transcript)
        };
    }

    /**
     * Estimate speech rate from transcript
     */
    _estimateSpeechRate(transcript) {
        if (!transcript) return 0.0;

        const wordCount = transcript.split(/\s+/).length;
        const estimatedDuration = 1.0; // Placeholder - would need actual duration

        // Words per second
        return wordCount / estimatedDuration;
    }

    /**
     * Average enrollment samples
     */
    _averageSamples(samples) {
        const avg = {
            pitchMean: 0.0,
            pitchVariance: 0.0,
            energyMean: 0.0,
            energyVariance: 0.0,
            speechRate: 0.0
        };

        for (const sample of samples) {
            avg.pitchMean += sample.pitchMean;
            avg.pitchVariance += sample.pitchVariance;
            avg.energyMean += sample.energyMean;
            avg.energyVariance += sample.energyVariance;
            avg.speechRate += sample.speechRate;
        }

        const n = samples.length;
        avg.pitchMean /= n;
        avg.pitchVariance /= n;
        avg.energyMean /= n;
        avg.energyVariance /= n;
        avg.speechRate /= n;

        return avg;
    }

    /**
     * Get speaker profile by ID
     */
    getProfile(speakerId) {
        return this.profiles.get(speakerId);
    }

    /**
     * Get all profiles
     */
    getAllProfiles() {
        return Array.from(this.profiles.values());
    }

    /**
     * Get current speaker
     */
    getCurrentSpeaker() {
        return this.currentSpeaker;
    }

    /**
     * Delete speaker profile
     */
    deleteProfile(speakerId) {
        const deleted = this.profiles.delete(speakerId);

        if (deleted) {
            this.metrics.enrolledSpeakers = this.profiles.size;

            // Clear current speaker if deleted
            if (this.currentSpeaker && this.currentSpeaker.id === speakerId) {
                this.currentSpeaker = null;
            }

            // Save profiles
            if (this.config.persistProfiles) {
                this._saveProfiles();
            }

            console.log('[VoiceProfileManager] Profile deleted:', speakerId);
        }

        return deleted;
    }

    /**
     * Load profiles from localStorage
     */
    _loadProfiles() {
        try {
            const stored = localStorage.getItem(this.config.storageKey);
            if (stored) {
                const data = JSON.parse(stored);

                for (const profileData of data) {
                    const profile = new SpeakerProfile(profileData);

                    // Reconstruct voice print
                    if (profileData.voicePrint) {
                        profile.voicePrint = new VoicePrint(profileData.voicePrint);
                    }

                    this.profiles.set(profile.id, profile);
                }

                this.metrics.enrolledSpeakers = this.profiles.size;

                console.log('[VoiceProfileManager] Loaded', this.profiles.size, 'profiles');
            }
        } catch (error) {
            console.error('[VoiceProfileManager] Failed to load profiles:', error);
        }
    }

    /**
     * Save profiles to localStorage
     */
    _saveProfiles() {
        try {
            const data = Array.from(this.profiles.values());
            localStorage.setItem(this.config.storageKey, JSON.stringify(data));
        } catch (error) {
            console.error('[VoiceProfileManager] Failed to save profiles:', error);
        }
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            currentSpeaker: this.currentSpeaker ? this.currentSpeaker.name : null,
            recognitionRate: this.metrics.totalRecognitions > 0
                ? this.metrics.successfulRecognitions / this.metrics.totalRecognitions
                : 0.0,
            enrollmentInProgress: this.enrollmentInProgress !== null
        };
    }

    /**
     * Cleanup
     */
    async destroy() {
        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        console.log('[VoiceProfileManager] Destroyed');
    }

    // Public API for callbacks

    onSpeakerRecognized(callback) {
        this.onSpeakerRecognizedCallback = callback;
    }

    onSpeakerChanged(callback) {
        this.onSpeakerChangedCallback = callback;
    }

    onEnrollmentComplete(callback) {
        this.onEnrollmentCompleteCallback = callback;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VoiceProfileManager,
        SpeakerProfile,
        VoicePrint,
        RecognitionResult
    };
}
