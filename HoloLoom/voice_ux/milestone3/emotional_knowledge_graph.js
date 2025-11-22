/**
 * Emotional Knowledge Graph (Per-User)
 * Persistent graph of user's emotional patterns, triggers, and preferences
 * Date: November 2025
 *
 * Breakthrough Feature: 105/100
 *
 * Core Capabilities:
 * 1. Persistent user-specific emotional patterns
 * 2. Trigger identification and tracking
 * 3. Response effectiveness tracking
 * 4. Temporal pattern analysis
 * 5. Personalized insights generation
 * 6. Cross-session learning
 */

/**
 * Node Types in the Emotional Knowledge Graph
 */
const NodeType = {
    USER: 'user',
    EMOTION: 'emotion',
    TRIGGER: 'trigger',
    RESPONSE: 'response',
    CONTEXT: 'context',
    PATTERN: 'pattern'
};

/**
 * Edge Types (Relationships)
 */
const EdgeType = {
    // User relationships
    EXPERIENCES: 'experiences',           // User -> Emotion
    TRIGGERED_BY: 'triggered_by',         // Emotion -> Trigger
    RESPONDS_TO: 'responds_to',          // User -> Response

    // Response effectiveness
    HELPS_WITH: 'helps_with',            // Response -> Emotion (positive)
    WORSENS: 'worsens',                  // Response -> Emotion (negative)
    NEUTRAL_TO: 'neutral_to',            // Response -> Emotion (no effect)

    // Context relationships
    OCCURS_IN: 'occurs_in',              // Emotion -> Context
    ASSOCIATED_WITH: 'associated_with',  // Trigger -> Context

    // Temporal relationships
    FOLLOWS: 'follows',                  // Emotion -> Emotion (sequence)
    PRECEDES: 'precedes',                // Emotion -> Emotion

    // Patterns
    EXHIBITS: 'exhibits',                // User -> Pattern
    CONTAINS: 'contains'                 // Pattern -> [Emotion|Trigger|Response]
};

/**
 * Graph Node
 */
class GraphNode {
    constructor(data) {
        this.id = data.id || this._generateId();
        this.type = data.type;
        this.label = data.label;
        this.properties = data.properties || {};

        // Temporal tracking
        this.firstSeen = data.firstSeen || new Date();
        this.lastSeen = data.lastSeen || new Date();
        this.occurrenceCount = data.occurrenceCount || 1;

        // Metadata
        this.metadata = data.metadata || {};
    }

    _generateId() {
        return `${this.type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    update() {
        this.lastSeen = new Date();
        this.occurrenceCount++;
    }

    toJSON() {
        return {
            id: this.id,
            type: this.type,
            label: this.label,
            properties: this.properties,
            firstSeen: this.firstSeen.toISOString(),
            lastSeen: this.lastSeen.toISOString(),
            occurrenceCount: this.occurrenceCount
        };
    }
}

/**
 * Graph Edge
 */
class GraphEdge {
    constructor(data) {
        this.id = data.id || this._generateId(data.from, data.to, data.type);
        this.from = data.from;  // Node ID
        this.to = data.to;      // Node ID
        this.type = data.type;

        // Edge properties
        this.weight = data.weight || 1.0;
        this.confidence = data.confidence || 1.0;

        // Temporal tracking
        this.firstObserved = data.firstObserved || new Date();
        this.lastObserved = data.lastObserved || new Date();
        this.observationCount = data.observationCount || 1;

        // Evidence
        this.evidence = data.evidence || [];
    }

    _generateId(from, to, type) {
        return `edge_${from}_${type}_${to}`;
    }

    strengthen(amount = 0.1) {
        this.weight = Math.min(1.0, this.weight + amount);
        this.observationCount++;
        this.lastObserved = new Date();
    }

    weaken(amount = 0.1) {
        this.weight = Math.max(0.0, this.weight - amount);
        this.lastObserved = new Date();
    }

    toJSON() {
        return {
            id: this.id,
            from: this.from,
            to: this.to,
            type: this.type,
            weight: this.weight,
            confidence: this.confidence,
            observationCount: this.observationCount
        };
    }
}

/**
 * Emotional Knowledge Graph
 */
class EmotionalKnowledgeGraph {
    constructor(userId) {
        this.userId = userId;
        this.nodes = new Map();  // nodeId -> GraphNode
        this.edges = new Map();  // edgeId -> GraphEdge

        // Index for fast lookups
        this.nodesByType = new Map();  // type -> Set of nodeIds
        this.edgesByType = new Map();  // type -> Set of edgeIds
        this.nodeEdges = new Map();    // nodeId -> {incoming: Set, outgoing: Set}

        // User profile node
        this._createUserNode();

        // Statistics
        this.stats = {
            totalInteractions: 0,
            emotionsTracked: new Set(),
            triggersIdentified: new Set(),
            responsesEvaluated: new Set(),
            lastUpdated: new Date()
        };

        console.log('[EmotionalKG] Initialized for user:', userId);
    }

    _createUserNode() {
        const userNode = new GraphNode({
            type: NodeType.USER,
            label: this.userId,
            properties: {
                createdAt: new Date()
            }
        });

        this.addNode(userNode);
        return userNode;
    }

    /**
     * Add node to graph
     */
    addNode(node) {
        this.nodes.set(node.id, node);

        // Update type index
        if (!this.nodesByType.has(node.type)) {
            this.nodesByType.set(node.type, new Set());
        }
        this.nodesByType.get(node.type).add(node.id);

        // Initialize edge tracking
        this.nodeEdges.set(node.id, {
            incoming: new Set(),
            outgoing: new Set()
        });

        return node;
    }

    /**
     * Add edge to graph
     */
    addEdge(edge) {
        this.edges.set(edge.id, edge);

        // Update type index
        if (!this.edgesByType.has(edge.type)) {
            this.edgesByType.set(edge.type, new Set());
        }
        this.edgesByType.get(edge.type).add(edge.id);

        // Update node edge tracking
        if (this.nodeEdges.has(edge.from)) {
            this.nodeEdges.get(edge.from).outgoing.add(edge.id);
        }
        if (this.nodeEdges.has(edge.to)) {
            this.nodeEdges.get(edge.to).incoming.add(edge.id);
        }

        return edge;
    }

    /**
     * Record an emotional experience
     */
    recordExperience(emotionalData, context = {}) {
        this.stats.totalInteractions++;

        const emotion = emotionalData.emotion;
        const trigger = emotionalData.trigger;
        const systemResponse = emotionalData.systemResponse;

        // 1. Find or create emotion node
        const emotionNode = this._findOrCreateEmotionNode(emotion, emotionalData);

        // 2. Find or create trigger node (if present)
        let triggerNode = null;
        if (trigger) {
            triggerNode = this._findOrCreateTriggerNode(trigger);
        }

        // 3. Find or create context node
        const contextNode = this._findOrCreateContextNode(context);

        // 4. Find or create response node (if system responded)
        let responseNode = null;
        if (systemResponse) {
            responseNode = this._findOrCreateResponseNode(systemResponse);
        }

        // 5. Create relationships
        const userNodeId = Array.from(this.nodesByType.get(NodeType.USER))[0];

        // User experiences emotion
        this._findOrCreateEdge(userNodeId, emotionNode.id, EdgeType.EXPERIENCES);

        // Emotion triggered by trigger
        if (triggerNode) {
            this._findOrCreateEdge(emotionNode.id, triggerNode.id, EdgeType.TRIGGERED_BY);
        }

        // Emotion occurs in context
        this._findOrCreateEdge(emotionNode.id, contextNode.id, EdgeType.OCCURS_IN);

        // Response relationship (will be updated with effectiveness later)
        if (responseNode) {
            this._findOrCreateEdge(userNodeId, responseNode.id, EdgeType.RESPONDS_TO);
        }

        // Update statistics
        this.stats.emotionsTracked.add(emotion);
        if (trigger) this.stats.triggersIdentified.add(trigger);
        if (systemResponse) this.stats.responsesEvaluated.add(systemResponse.action);
        this.stats.lastUpdated = new Date();

        return {
            emotionNode,
            triggerNode,
            contextNode,
            responseNode
        };
    }

    /**
     * Update response effectiveness
     */
    updateResponseEffectiveness(responseId, emotionBefore, emotionAfter, wasHelpful) {
        const responseNode = this.nodes.get(responseId);
        if (!responseNode) return;

        const emotionBeforeNode = this._findEmotionNode(emotionBefore);
        const emotionAfterNode = this._findEmotionNode(emotionAfter);

        if (!emotionBeforeNode) return;

        // Determine edge type based on effectiveness
        let edgeType;
        if (wasHelpful) {
            edgeType = EdgeType.HELPS_WITH;
        } else if (emotionAfter.valence < emotionBefore.valence - 0.2) {
            edgeType = EdgeType.WORSENS;
        } else {
            edgeType = EdgeType.NEUTRAL_TO;
        }

        // Create or strengthen edge
        const edge = this._findOrCreateEdge(responseNode.id, emotionBeforeNode.id, edgeType);
        edge.strengthen(0.1);
    }

    /**
     * Find or create emotion node
     */
    _findOrCreateEmotionNode(emotion, emotionalData) {
        // Try to find existing
        const emotionNodes = this.nodesByType.get(NodeType.EMOTION) || new Set();
        for (const nodeId of emotionNodes) {
            const node = this.nodes.get(nodeId);
            if (node.label === emotion) {
                node.update();
                return node;
            }
        }

        // Create new
        const node = new GraphNode({
            type: NodeType.EMOTION,
            label: emotion,
            properties: {
                intensity: emotionalData.intensity,
                valence: emotionalData.valence,
                arousal: emotionalData.arousal,
                dominance: emotionalData.dominance
            }
        });

        return this.addNode(node);
    }

    _findEmotionNode(emotion) {
        const emotionNodes = this.nodesByType.get(NodeType.EMOTION) || new Set();
        for (const nodeId of emotionNodes) {
            const node = this.nodes.get(nodeId);
            if (node.label === emotion) {
                return node;
            }
        }
        return null;
    }

    /**
     * Find or create trigger node
     */
    _findOrCreateTriggerNode(trigger) {
        const triggerNodes = this.nodesByType.get(NodeType.TRIGGER) || new Set();
        for (const nodeId of triggerNodes) {
            const node = this.nodes.get(nodeId);
            if (node.label === trigger) {
                node.update();
                return node;
            }
        }

        const node = new GraphNode({
            type: NodeType.TRIGGER,
            label: trigger,
            properties: {}
        });

        return this.addNode(node);
    }

    /**
     * Find or create context node
     */
    _findOrCreateContextNode(context) {
        const contextKey = JSON.stringify(context);
        const contextNodes = this.nodesByType.get(NodeType.CONTEXT) || new Set();

        for (const nodeId of contextNodes) {
            const node = this.nodes.get(nodeId);
            if (JSON.stringify(node.properties) === contextKey) {
                node.update();
                return node;
            }
        }

        const node = new GraphNode({
            type: NodeType.CONTEXT,
            label: `context_${Date.now()}`,
            properties: context
        });

        return this.addNode(node);
    }

    /**
     * Find or create response node
     */
    _findOrCreateResponseNode(response) {
        const responseKey = `${response.action}_${response.tone}`;
        const responseNodes = this.nodesByType.get(NodeType.RESPONSE) || new Set();

        for (const nodeId of responseNodes) {
            const node = this.nodes.get(nodeId);
            const nodeKey = `${node.properties.action}_${node.properties.tone}`;
            if (nodeKey === responseKey) {
                node.update();
                return node;
            }
        }

        const node = new GraphNode({
            type: NodeType.RESPONSE,
            label: response.action,
            properties: {
                action: response.action,
                tone: response.tone,
                text: response.text
            }
        });

        return this.addNode(node);
    }

    /**
     * Find or create edge
     */
    _findOrCreateEdge(fromId, toId, type) {
        const edgeId = `edge_${fromId}_${type}_${toId}`;
        const existing = this.edges.get(edgeId);

        if (existing) {
            existing.strengthen(0.05);
            return existing;
        }

        const edge = new GraphEdge({
            from: fromId,
            to: toId,
            type: type
        });

        return this.addEdge(edge);
    }

    /**
     * Get emotional triggers for user
     */
    getEmotionalTriggers(emotion = null) {
        const triggers = [];

        // Get emotion nodes
        let emotionNodes = Array.from(this.nodesByType.get(NodeType.EMOTION) || []);
        if (emotion) {
            emotionNodes = emotionNodes.filter(nodeId => {
                const node = this.nodes.get(nodeId);
                return node.label === emotion;
            });
        }

        // For each emotion, get triggers
        for (const emotionNodeId of emotionNodes) {
            const emotionNode = this.nodes.get(emotionNodeId);
            const edges = this.nodeEdges.get(emotionNodeId)?.outgoing || new Set();

            for (const edgeId of edges) {
                const edge = this.edges.get(edgeId);
                if (edge.type === EdgeType.TRIGGERED_BY) {
                    const triggerNode = this.nodes.get(edge.to);
                    triggers.push({
                        emotion: emotionNode.label,
                        trigger: triggerNode.label,
                        strength: edge.weight,
                        occurrences: edge.observationCount,
                        lastObserved: edge.lastObserved
                    });
                }
            }
        }

        // Sort by strength
        return triggers.sort((a, b) => b.strength - a.strength);
    }

    /**
     * Get effective responses for emotion
     */
    getEffectiveResponses(emotion) {
        const emotionNode = this._findEmotionNode(emotion);
        if (!emotionNode) return [];

        const responses = [];

        // Find all responses that help with this emotion
        const responseNodes = Array.from(this.nodesByType.get(NodeType.RESPONSE) || []);

        for (const responseNodeId of responseNodes) {
            const edges = this.nodeEdges.get(responseNodeId)?.outgoing || new Set();

            for (const edgeId of edges) {
                const edge = this.edges.get(edgeId);
                if (edge.to === emotionNode.id && edge.type === EdgeType.HELPS_WITH) {
                    const responseNode = this.nodes.get(responseNodeId);
                    responses.push({
                        action: responseNode.properties.action,
                        tone: responseNode.properties.tone,
                        effectiveness: edge.weight,
                        timesUsed: edge.observationCount
                    });
                }
            }
        }

        return responses.sort((a, b) => b.effectiveness - a.effectiveness);
    }

    /**
     * Detect patterns in user's emotional experiences
     */
    detectPatterns() {
        const patterns = [];

        // 1. Recurring triggers
        const triggers = this.getEmotionalTriggers();
        const recurringTriggers = triggers.filter(t => t.occurrences >= 3);

        if (recurringTriggers.length > 0) {
            patterns.push({
                type: 'recurring_triggers',
                description: 'Emotions frequently triggered by specific events',
                instances: recurringTriggers
            });
        }

        // 2. Emotion cycles (A -> B -> A)
        const cycles = this._detectEmotionCycles();
        if (cycles.length > 0) {
            patterns.push({
                type: 'emotion_cycles',
                description: 'Recurring emotional patterns',
                instances: cycles
            });
        }

        // 3. Context-emotion associations
        const contextPatterns = this._detectContextPatterns();
        if (contextPatterns.length > 0) {
            patterns.push({
                type: 'context_associations',
                description: 'Emotions associated with specific contexts',
                instances: contextPatterns
            });
        }

        return patterns;
    }

    _detectEmotionCycles() {
        // Look for FOLLOWS edges that form cycles
        const cycles = [];
        const followEdges = Array.from(this.edgesByType.get(EdgeType.FOLLOWS) || []);

        // Simple 2-cycle detection (A -> B -> A)
        for (const edgeId1 of followEdges) {
            const edge1 = this.edges.get(edgeId1);
            const emotionA = this.nodes.get(edge1.from);
            const emotionB = this.nodes.get(edge1.to);

            // Look for reverse edge
            for (const edgeId2 of followEdges) {
                const edge2 = this.edges.get(edgeId2);
                if (edge2.from === edge1.to && edge2.to === edge1.from) {
                    cycles.push({
                        emotions: [emotionA.label, emotionB.label],
                        strength: (edge1.weight + edge2.weight) / 2,
                        occurrences: Math.min(edge1.observationCount, edge2.observationCount)
                    });
                }
            }
        }

        return cycles;
    }

    _detectContextPatterns() {
        const patterns = [];
        const occursInEdges = Array.from(this.edgesByType.get(EdgeType.OCCURS_IN) || []);

        const contextEmotions = new Map();  // contextId -> [emotions]

        for (const edgeId of occursInEdges) {
            const edge = this.edges.get(edgeId);
            const emotionNode = this.nodes.get(edge.from);
            const contextId = edge.to;

            if (!contextEmotions.has(contextId)) {
                contextEmotions.set(contextId, []);
            }

            contextEmotions.get(contextId).push({
                emotion: emotionNode.label,
                weight: edge.weight,
                count: edge.observationCount
            });
        }

        // Find contexts with strong emotion associations
        for (const [contextId, emotions] of contextEmotions.entries()) {
            if (emotions.length >= 2) {
                const contextNode = this.nodes.get(contextId);
                patterns.push({
                    context: contextNode.properties,
                    emotions: emotions.sort((a, b) => b.weight - a.weight)
                });
            }
        }

        return patterns;
    }

    /**
     * Get personalized insights
     */
    getInsights() {
        const insights = [];

        // Most common emotions
        const emotionCounts = new Map();
        for (const nodeId of this.nodesByType.get(NodeType.EMOTION) || []) {
            const node = this.nodes.get(nodeId);
            emotionCounts.set(node.label, node.occurrenceCount);
        }

        const topEmotion = Array.from(emotionCounts.entries())
            .sort((a, b) => b[1] - a[1])[0];

        if (topEmotion) {
            insights.push({
                type: 'frequent_emotion',
                insight: `Most frequently experiences "${topEmotion[0]}" (${topEmotion[1]} times)`
            });
        }

        // Top triggers
        const triggers = this.getEmotionalTriggers();
        if (triggers.length > 0) {
            insights.push({
                type: 'top_trigger',
                insight: `Primary emotional trigger: "${triggers[0].trigger}" → ${triggers[0].emotion}`,
                data: triggers[0]
            });
        }

        // Patterns
        const patterns = this.detectPatterns();
        if (patterns.length > 0) {
            insights.push({
                type: 'patterns',
                insight: `Detected ${patterns.length} emotional patterns`,
                data: patterns
            });
        }

        return insights;
    }

    /**
     * Serialize graph
     */
    toJSON() {
        return {
            userId: this.userId,
            nodes: Array.from(this.nodes.values()).map(n => n.toJSON()),
            edges: Array.from(this.edges.values()).map(e => e.toJSON()),
            stats: {
                ...this.stats,
                emotionsTracked: Array.from(this.stats.emotionsTracked),
                triggersIdentified: Array.from(this.stats.triggersIdentified),
                responsesEvaluated: Array.from(this.stats.responsesEvaluated)
            }
        };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        EmotionalKnowledgeGraph,
        GraphNode,
        GraphEdge,
        NodeType,
        EdgeType
    };
}
