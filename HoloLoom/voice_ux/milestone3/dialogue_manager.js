/**
 * Dialogue Manager - Multi-turn Context Management
 * Milestone 3: Continuous cognition with context preservation
 * Date: November 2025
 *
 * Features:
 * - Conversation context preservation across turns
 * - Anaphora resolution ("it", "that", "them")
 * - Context window management (sliding window)
 * - Topic tracking and coherence scoring
 * - Memory of conversation history
 * - Automatic context summarization
 */

/**
 * Turn in conversation
 */
class ConversationTurn {
    constructor(data) {
        this.id = data.id || `turn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.speaker = data.speaker; // 'user' or 'system'
        this.text = data.text || '';
        this.timestamp = data.timestamp || new Date();
        this.entities = data.entities || []; // Named entities mentioned
        this.topics = data.topics || []; // Topics discussed
        this.metadata = data.metadata || {};
    }
}

/**
 * Conversation Context Window
 */
class ContextWindow {
    constructor(maxTurns = 10) {
        this.maxTurns = maxTurns;
        this.turns = [];
        this.entities = new Map(); // entity → mentions count
        this.topics = new Map(); // topic → mentions count
    }

    /**
     * Add turn to context window
     */
    addTurn(turn) {
        this.turns.push(turn);

        // Track entities
        for (const entity of turn.entities) {
            this.entities.set(entity, (this.entities.get(entity) || 0) + 1);
        }

        // Track topics
        for (const topic of turn.topics) {
            this.topics.set(topic, (this.topics.get(topic) || 0) + 1);
        }

        // Evict oldest turn if exceeds max
        if (this.turns.length > this.maxTurns) {
            const evicted = this.turns.shift();

            // Update entity counts
            for (const entity of evicted.entities) {
                const count = this.entities.get(entity) - 1;
                if (count <= 0) {
                    this.entities.delete(entity);
                } else {
                    this.entities.set(entity, count);
                }
            }

            // Update topic counts
            for (const topic of evicted.topics) {
                const count = this.topics.get(topic) - 1;
                if (count <= 0) {
                    this.topics.delete(topic);
                } else {
                    this.topics.set(topic, count);
                }
            }
        }
    }

    /**
     * Get recent turns (last N)
     */
    getRecentTurns(n = 5) {
        return this.turns.slice(-n);
    }

    /**
     * Get all turns
     */
    getAllTurns() {
        return this.turns;
    }

    /**
     * Get most mentioned entities
     */
    getTopEntities(limit = 5) {
        return Array.from(this.entities.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([entity, count]) => ({ entity, count }));
    }

    /**
     * Get most mentioned topics
     */
    getTopTopics(limit = 3) {
        return Array.from(this.topics.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([topic, count]) => ({ topic, count }));
    }

    /**
     * Clear context window
     */
    clear() {
        this.turns = [];
        this.entities.clear();
        this.topics.clear();
    }
}

/**
 * Dialogue Manager
 * Manages multi-turn conversations with context preservation
 */
class DialogueManager {
    constructor(config = {}) {
        this.config = {
            maxContextTurns: config.maxContextTurns || 10,
            enableAnaphoraResolution: config.enableAnaphoraResolution !== false,
            enableTopicTracking: config.enableTopicTracking !== false,
            enableContextSummarization: config.enableContextSummarization !== false,
            coherenceThreshold: config.coherenceThreshold || 0.5,
            summarizationTrigger: config.summarizationTrigger || 20, // Turns before summarizing
            ...config
        };

        // Conversation state
        this.contextWindow = new ContextWindow(this.config.maxContextTurns);
        this.conversationId = `conv_${Date.now()}`;
        this.conversationStartTime = new Date();
        this.currentTopic = null;
        this.topicHistory = [];

        // Anaphora resolution
        this.recentEntities = []; // Last mentioned entities for "it", "that", etc.

        // Callbacks
        this.onTopicChangeCallback = null;
        this.onCoherenceDropCallback = null;
        this.onContextSummarizedCallback = null;

        // Metrics
        this.metrics = {
            totalTurns: 0,
            userTurns: 0,
            systemTurns: 0,
            avgCoherenceScore: 0.0,
            topicChanges: 0,
            anaphoraResolutions: 0,
            contextSummarizations: 0
        };

        console.log('[DialogueManager] Initialized', this.config);
    }

    /**
     * Process user turn
     */
    async processTurn(userText, metadata = {}) {
        const turn = new ConversationTurn({
            speaker: 'user',
            text: userText,
            metadata: metadata
        });

        // Extract entities and topics
        turn.entities = this._extractEntities(userText);
        turn.topics = this._extractTopics(userText);

        // Resolve anaphora if enabled
        if (this.config.enableAnaphoraResolution) {
            turn.text = this._resolveAnaphora(turn.text);
        }

        // Add to context window
        this.contextWindow.addTurn(turn);

        // Update metrics
        this.metrics.totalTurns++;
        this.metrics.userTurns++;

        // Track topic changes
        if (this.config.enableTopicTracking) {
            this._trackTopicChange(turn.topics);
        }

        // Calculate coherence
        const coherence = this._calculateCoherence(turn);
        this.metrics.avgCoherenceScore =
            (this.metrics.avgCoherenceScore * (this.metrics.totalTurns - 1) + coherence) /
            this.metrics.totalTurns;

        // Check for low coherence
        if (coherence < this.config.coherenceThreshold && this.onCoherenceDropCallback) {
            this.onCoherenceDropCallback({
                type: 'coherence_drop',
                coherence: coherence,
                turn: turn,
                timestamp: new Date()
            });
        }

        // Check if we should summarize context
        if (this.config.enableContextSummarization &&
            this.metrics.totalTurns % this.config.summarizationTrigger === 0) {
            await this._summarizeContext();
        }

        // Update recent entities for anaphora resolution
        this.recentEntities = turn.entities.slice(0, 3); // Keep last 3 entities

        console.log('[DialogueManager] User turn processed:', turn.text.substring(0, 50));

        return {
            turn: turn,
            coherence: coherence,
            context: this._buildContextForQuery(turn)
        };
    }

    /**
     * Record system response turn
     */
    recordSystemTurn(responseText, metadata = {}) {
        const turn = new ConversationTurn({
            speaker: 'system',
            text: responseText,
            metadata: metadata
        });

        turn.entities = this._extractEntities(responseText);
        turn.topics = this._extractTopics(responseText);

        this.contextWindow.addTurn(turn);

        this.metrics.totalTurns++;
        this.metrics.systemTurns++;

        console.log('[DialogueManager] System turn recorded');
    }

    /**
     * Build enriched context for query
     */
    _buildContextForQuery(currentTurn) {
        const recentTurns = this.contextWindow.getRecentTurns(5);
        const topEntities = this.contextWindow.getTopEntities(3);
        const topTopics = this.contextWindow.getTopTopics(2);

        return {
            recentTurns: recentTurns,
            topEntities: topEntities,
            topTopics: topTopics,
            currentTopic: this.currentTopic,
            conversationAge: Date.now() - this.conversationStartTime.getTime(),
            summary: this._generateBriefSummary()
        };
    }

    /**
     * Extract entities from text (simple implementation)
     */
    _extractEntities(text) {
        const entities = [];

        // Capitalized words (proper nouns)
        const capitalizedWords = text.match(/\b[A-Z][a-z]+\b/g) || [];
        entities.push(...capitalizedWords);

        // Common technical terms
        const technicalTerms = [
            'Thompson Sampling', 'Python', 'JavaScript', 'API', 'database',
            'machine learning', 'neural network', 'transformer'
        ];

        for (const term of technicalTerms) {
            if (text.toLowerCase().includes(term.toLowerCase())) {
                entities.push(term);
            }
        }

        return [...new Set(entities)]; // Deduplicate
    }

    /**
     * Extract topics from text (simple keyword-based)
     */
    _extractTopics(text) {
        const topics = [];

        const topicKeywords = {
            'technology': ['code', 'programming', 'software', 'API', 'database'],
            'machine_learning': ['ML', 'model', 'training', 'neural', 'algorithm'],
            'data': ['data', 'analysis', 'statistics', 'metrics'],
            'design': ['UI', 'UX', 'interface', 'design', 'layout'],
            'voice': ['voice', 'speech', 'audio', 'TTS', 'STT', 'recognition']
        };

        const lowerText = text.toLowerCase();

        for (const [topic, keywords] of Object.entries(topicKeywords)) {
            for (const keyword of keywords) {
                if (lowerText.includes(keyword.toLowerCase())) {
                    topics.push(topic);
                    break; // Only add topic once
                }
            }
        }

        return [...new Set(topics)]; // Deduplicate
    }

    /**
     * Resolve anaphora ("it", "that", "them" → actual entities)
     */
    _resolveAnaphora(text) {
        if (this.recentEntities.length === 0) {
            return text; // No entities to resolve
        }

        let resolved = text;

        // Simple anaphora resolution
        const anaphoraPatterns = [
            { pattern: /\bit\b/gi, replacement: this.recentEntities[0] || 'it' },
            { pattern: /\bthat\b/gi, replacement: this.recentEntities[0] || 'that' },
            { pattern: /\bthis\b/gi, replacement: this.recentEntities[0] || 'this' },
            { pattern: /\bthem\b/gi, replacement: this.recentEntities.join(' and ') || 'them' }
        ];

        for (const { pattern, replacement } of anaphoraPatterns) {
            if (pattern.test(text)) {
                resolved = text.replace(pattern, replacement);
                this.metrics.anaphoraResolutions++;
                console.log('[DialogueManager] Resolved anaphora:', text, '→', resolved);
                break; // Only resolve first match
            }
        }

        return resolved;
    }

    /**
     * Track topic changes
     */
    _trackTopicChange(topics) {
        if (topics.length === 0) {
            return;
        }

        const newTopic = topics[0]; // Primary topic

        if (this.currentTopic && newTopic !== this.currentTopic) {
            // Topic changed
            this.topicHistory.push({
                topic: this.currentTopic,
                endTime: new Date()
            });

            this.metrics.topicChanges++;

            // Emit topic change event
            if (this.onTopicChangeCallback) {
                this.onTopicChangeCallback({
                    type: 'topic_change',
                    oldTopic: this.currentTopic,
                    newTopic: newTopic,
                    timestamp: new Date()
                });
            }

            console.log('[DialogueManager] Topic changed:', this.currentTopic, '→', newTopic);
        }

        this.currentTopic = newTopic;
    }

    /**
     * Calculate coherence score (0.0-1.0)
     */
    _calculateCoherence(currentTurn) {
        const recentTurns = this.contextWindow.getRecentTurns(3);

        if (recentTurns.length < 2) {
            return 1.0; // First turns are always coherent
        }

        // Calculate entity overlap
        const currentEntities = new Set(currentTurn.entities);
        const recentEntities = new Set(
            recentTurns.flatMap(turn => turn.entities)
        );

        const intersection = new Set(
            [...currentEntities].filter(e => recentEntities.has(e))
        );

        const entityOverlap = currentEntities.size > 0
            ? intersection.size / currentEntities.size
            : 0.0;

        // Calculate topic overlap
        const currentTopics = new Set(currentTurn.topics);
        const recentTopics = new Set(
            recentTurns.flatMap(turn => turn.topics)
        );

        const topicIntersection = new Set(
            [...currentTopics].filter(t => recentTopics.has(t))
        );

        const topicOverlap = currentTopics.size > 0
            ? topicIntersection.size / currentTopics.size
            : 0.0;

        // Weighted average
        const coherence = 0.6 * entityOverlap + 0.4 * topicOverlap;

        return coherence;
    }

    /**
     * Summarize context (stub for future LLM integration)
     */
    async _summarizeContext() {
        console.log('[DialogueManager] Summarizing context (placeholder)');

        this.metrics.contextSummarizations++;

        // Future: Use LLM to summarize long conversation
        const summary = this._generateBriefSummary();

        if (this.onContextSummarizedCallback) {
            this.onContextSummarizedCallback({
                type: 'context_summarized',
                summary: summary,
                turnCount: this.metrics.totalTurns,
                timestamp: new Date()
            });
        }

        return summary;
    }

    /**
     * Generate brief summary of conversation
     */
    _generateBriefSummary() {
        const topTopics = this.contextWindow.getTopTopics(2);
        const topEntities = this.contextWindow.getTopEntities(3);

        if (topTopics.length === 0 && topEntities.length === 0) {
            return 'New conversation';
        }

        const topicsStr = topTopics.map(t => t.topic).join(', ');
        const entitiesStr = topEntities.map(e => e.entity).join(', ');

        return `Discussion about ${topicsStr || 'various topics'}${
            entitiesStr ? `, focusing on ${entitiesStr}` : ''
        }`;
    }

    /**
     * Get conversation history
     */
    getHistory(limit = null) {
        const turns = limit
            ? this.contextWindow.getRecentTurns(limit)
            : this.contextWindow.getAllTurns();

        return {
            conversationId: this.conversationId,
            turns: turns,
            currentTopic: this.currentTopic,
            topTopics: this.contextWindow.getTopTopics(3),
            topEntities: this.contextWindow.getTopEntities(5),
            summary: this._generateBriefSummary()
        };
    }

    /**
     * Clear conversation history
     */
    clearHistory() {
        this.contextWindow.clear();
        this.recentEntities = [];
        this.currentTopic = null;
        this.topicHistory = [];

        console.log('[DialogueManager] Conversation history cleared');
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            conversationAge: Date.now() - this.conversationStartTime.getTime(),
            currentTopic: this.currentTopic,
            topicChanges: this.metrics.topicChanges,
            turnRatio: this.metrics.userTurns > 0
                ? this.metrics.systemTurns / this.metrics.userTurns
                : 0.0
        };
    }

    // Public API for callbacks

    onTopicChange(callback) {
        this.onTopicChangeCallback = callback;
    }

    onCoherenceDrop(callback) {
        this.onCoherenceDropCallback = callback;
    }

    onContextSummarized(callback) {
        this.onContextSummarizedCallback = callback;
    }
}


/**
 * Integration with StreamingEngine and InterruptionHandler
 */
class ContinuousDialogueSystem {
    constructor(streamingEngine, interruptionHandler, dialogueManager) {
        this.streamingEngine = streamingEngine;
        this.interruptionHandler = interruptionHandler;
        this.dialogueManager = dialogueManager;

        this._setupEventHandlers();
    }

    _setupEventHandlers() {
        // Listen for final transcripts
        this.streamingEngine.onSegment((event) => {
            if (event.type === 'final') {
                this._handleUserTurn(event.segment.transcript);
            }
        });

        // Listen for topic changes
        this.dialogueManager.onTopicChange((event) => {
            console.log('[ContinuousDialogueSystem] Topic changed:', event.oldTopic, '→', event.newTopic);
        });

        // Listen for coherence drops
        this.dialogueManager.onCoherenceDrop((event) => {
            console.log('[ContinuousDialogueSystem] Coherence drop detected:', event.coherence);
            // Could trigger clarification request
        });
    }

    async _handleUserTurn(transcript) {
        // Process turn through dialogue manager
        const result = await this.dialogueManager.processTurn(transcript);

        // Get enriched context
        const context = result.context;

        // Generate response with context
        const response = await this._generateContextualResponse(transcript, context);

        // Record system turn
        this.dialogueManager.recordSystemTurn(response.text);

        // Deliver response with interruption monitoring
        await this.interruptionHandler.startMonitoring(response);

        console.log('[ContinuousDialogueSystem] Turn complete');
    }

    async _generateContextualResponse(query, context) {
        // Stub for future LLM integration
        console.log('[ContinuousDialogueSystem] Generating response with context:', context.summary);

        return {
            text: `Response to: ${query} (with context: ${context.summary})`,
            estimatedDuration: 3000
        };
    }

    getConversationHistory() {
        return this.dialogueManager.getHistory();
    }

    getMetrics() {
        return {
            dialogue: this.dialogueManager.getMetrics(),
            streaming: this.streamingEngine.getMetrics(),
            interruption: this.interruptionHandler.getMetrics()
        };
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DialogueManager,
        ConversationTurn,
        ContextWindow,
        ContinuousDialogueSystem
    };
}
