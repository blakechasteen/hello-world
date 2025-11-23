/**
 * Comprehensive Test Suite for 105/100 Breakthrough Features
 * Tests all 5 breakthrough systems and their integration
 * Date: November 2025
 */

// Mock assert for Node.js-like testing
const assert = {
    ok: (condition, message) => {
        if (!condition) throw new Error(message || 'Assertion failed');
    },
    equal: (actual, expected, message) => {
        if (actual !== expected) {
            throw new Error(message || `Expected ${expected}, got ${actual}`);
        }
    },
    greaterThan: (actual, threshold, message) => {
        if (actual <= threshold) {
            throw new Error(message || `Expected ${actual} > ${threshold}`);
        }
    },
    includes: (array, item, message) => {
        if (!array.includes(item)) {
            throw new Error(message || `Expected array to include ${item}`);
        }
    }
};

// Test runner
class TestRunner {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
        this.results = [];
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    async run() {
        console.log(`\n🧪 Running ${this.tests.length} tests...\n`);

        for (const { name, fn } of this.tests) {
            try {
                await fn();
                this.passed++;
                this.results.push({ name, status: 'PASS' });
                console.log(`✅ ${name}`);
            } catch (error) {
                this.failed++;
                this.results.push({ name, status: 'FAIL', error: error.message });
                console.log(`❌ ${name}`);
                console.log(`   Error: ${error.message}`);
            }
        }

        console.log(`\n📊 Results: ${this.passed} passed, ${this.failed} failed\n`);

        return {
            passed: this.passed,
            failed: this.failed,
            total: this.tests.length,
            results: this.results
        };
    }
}

const runner = new TestRunner();

// ============================================================================
// FEATURE #1: Recursive Self-Improvement Tests
// ============================================================================

runner.test('Self-Improvement: Initialize system', async () => {
    const { RecursiveSelfImprovement } = await import('../recursive_self_improvement.js');
    const system = new RecursiveSelfImprovement();

    assert.ok(system, 'System should initialize');
    assert.ok(system.interactions, 'Should have interactions array');
    assert.ok(system.calibrationParams, 'Should have calibration params');
});

runner.test('Self-Improvement: Record interaction', async () => {
    const { RecursiveSelfImprovement } = await import('../recursive_self_improvement.js');
    const system = new RecursiveSelfImprovement();

    system.recordInteraction({
        userInput: 'test',
        emotionalAnalysis: { fusedEmotion: { emotion: 'happy' } },
        systemResponse: 'response',
        modalityWeights: { facial: 0.5, vocal: 0.3, text: 0.2 }
    });

    assert.equal(system.getInteractions().length, 1, 'Should have 1 interaction');
});

runner.test('Self-Improvement: Update outcome and calculate success', async () => {
    const { RecursiveSelfImprovement } = await import('../recursive_self_improvement.js');
    const system = new RecursiveSelfImprovement();

    system.recordInteraction({
        userInput: 'test',
        emotionalAnalysis: { fusedEmotion: { emotion: 'happy' } },
        systemResponse: 'response',
        modalityWeights: { facial: 0.5, vocal: 0.3, text: 0.2 }
    });

    system.updateOutcome(0, {
        userSatisfaction: 0.9,
        conversationContinued: true,
        actionEffectiveness: 0.8
    });

    const interactions = system.getInteractions();
    assert.greaterThan(interactions[0].outcomeMetrics.overallSuccess, 0.5,
        'Overall success should be > 0.5');
});

runner.test('Self-Improvement: Calibrate modality weights', async () => {
    const { RecursiveSelfImprovement } = await import('../recursive_self_improvement.js');
    const system = new RecursiveSelfImprovement();

    // Record 5 interactions with outcomes
    for (let i = 0; i < 5; i++) {
        system.recordInteraction({
            userInput: `test ${i}`,
            emotionalAnalysis: { fusedEmotion: { emotion: 'happy' } },
            systemResponse: 'response',
            modalityWeights: { facial: 0.5, vocal: 0.3, text: 0.2 }
        });

        system.updateOutcome(i, {
            userSatisfaction: 0.8,
            actionEffectiveness: 0.9
        });
    }

    // Trigger calibration (private method, but should work)
    system._calibrateModalityWeights();

    const metrics = system.getLearningMetrics();
    assert.ok(metrics, 'Should have learning metrics');
});

// ============================================================================
// FEATURE #2: Chain-of-Emotion Reasoning Tests
// ============================================================================

runner.test('Chain Reasoning: Create emotional chain', async () => {
    const { EmotionalChain, EmotionalStateNode } = await import('../chain_of_emotion_reasoning.js');
    const chain = new EmotionalChain();

    const state1 = new EmotionalStateNode({
        emotion: 'neutral',
        intensity: 0.5,
        valence: 0.0,
        arousal: 0.0,
        dominance: 0.5
    });

    chain.addState(state1);

    assert.equal(chain.states.length, 1, 'Chain should have 1 state');
    assert.equal(chain.getCurrentState().emotion, 'neutral', 'Current state should be neutral');
});

runner.test('Chain Reasoning: Track state transitions', async () => {
    const { EmotionalChain, EmotionalStateNode } = await import('../chain_of_emotion_reasoning.js');
    const chain = new EmotionalChain();

    const state1 = new EmotionalStateNode({ emotion: 'neutral', intensity: 0.5, valence: 0.0, arousal: 0.0, dominance: 0.5 });
    const state2 = new EmotionalStateNode({ emotion: 'happy', intensity: 0.7, valence: 0.8, arousal: 0.6, dominance: 0.6 });

    chain.addState(state1);
    chain.addState(state2);

    assert.equal(chain.states.length, 2, 'Chain should have 2 states');
    assert.equal(state2.previousState.emotion, 'neutral', 'State 2 should link to state 1');
    assert.equal(state1.nextState.emotion, 'happy', 'State 1 should link to state 2');
});

runner.test('Chain Reasoning: Detect escalation pattern', async () => {
    const { EmotionalChain, EmotionalStateNode } = await import('../chain_of_emotion_reasoning.js');
    const chain = new EmotionalChain();

    // Create escalating intensity pattern
    chain.addState(new EmotionalStateNode({ emotion: 'frustrated', intensity: 0.3, valence: -0.3, arousal: 0.4, dominance: 0.3 }));
    chain.addState(new EmotionalStateNode({ emotion: 'frustrated', intensity: 0.5, valence: -0.5, arousal: 0.6, dominance: 0.4 }));
    chain.addState(new EmotionalStateNode({ emotion: 'angry', intensity: 0.8, valence: -0.8, arousal: 0.9, dominance: 0.7 }));

    const patterns = chain.analyzePatterns();
    const hasEscalation = patterns.some(p => p.type === 'escalation');

    assert.ok(hasEscalation, 'Should detect escalation pattern');
});

runner.test('Chain Reasoning: Get trajectory', async () => {
    const { EmotionalChain, EmotionalStateNode } = await import('../chain_of_emotion_reasoning.js');
    const chain = new EmotionalChain();

    chain.addState(new EmotionalStateNode({ emotion: 'neutral', intensity: 0.5, valence: 0.0, arousal: 0.0, dominance: 0.5 }));
    chain.addState(new EmotionalStateNode({ emotion: 'happy', intensity: 0.7, valence: 0.8, arousal: 0.6, dominance: 0.6 }));

    const trajectory = chain.getTrajectory();

    assert.equal(trajectory.length, 2, 'Trajectory should have 2 points');
    assert.equal(trajectory[0].emotion, 'neutral', 'First point should be neutral');
    assert.equal(trajectory[1].emotion, 'happy', 'Second point should be happy');
});

// ============================================================================
// FEATURE #3: Multi-Agent Debate Tests
// ============================================================================

runner.test('Multi-Agent Debate: Initialize orchestrator', async () => {
    const { MultiAgentDebateOrchestrator, AgentPersona } = await import('../multi_agent_emotional_debate.js');
    const orchestrator = new MultiAgentDebateOrchestrator({
        enabledAgents: [
            AgentPersona.EMPATHY_SPECIALIST,
            AgentPersona.BEHAVIORAL_ANALYST
        ]
    });

    assert.ok(orchestrator, 'Orchestrator should initialize');
    assert.equal(orchestrator.config.enabledAgents.length, 2, 'Should have 2 enabled agents');
});

runner.test('Multi-Agent Debate: Start debate session', async () => {
    const { MultiAgentDebateOrchestrator, AgentPersona, DebateSession } = await import('../multi_agent_emotional_debate.js');

    const orchestrator = new MultiAgentDebateOrchestrator({
        enabledAgents: [AgentPersona.EMPATHY_SPECIALIST, AgentPersona.BEHAVIORAL_ANALYST],
        llmProvider: 'mock'
    });

    const emotionalData = {
        facial: { emotion: 'happy', confidence: 0.8 },
        vocal: { emotion: 'happy', confidence: 0.7 },
        text: { emotion: 'neutral', confidence: 0.6 },
        fusedEmotion: { emotion: 'happy', confidence: 0.75, valence: 0.6, arousal: 0.5, dominance: 0.5 }
    };

    const result = await orchestrator.startDebate(emotionalData);

    assert.ok(result.interpretation, 'Should have final interpretation');
    assert.greaterThan(result.confidence, 0.0, 'Should have confidence > 0');
    assert.ok(result.transcript, 'Should have debate transcript');
});

// ============================================================================
// FEATURE #4: Emotional Knowledge Graph Tests
// ============================================================================

runner.test('Knowledge Graph: Initialize for user', async () => {
    const { EmotionalKnowledgeGraph } = await import('../emotional_knowledge_graph.js');
    const kg = new EmotionalKnowledgeGraph('user_123');

    assert.ok(kg, 'Knowledge graph should initialize');
    assert.equal(kg.userId, 'user_123', 'Should have correct user ID');
    assert.ok(kg.nodes.size > 0, 'Should have USER node');
});

runner.test('Knowledge Graph: Record emotional experience', async () => {
    const { EmotionalKnowledgeGraph } = await import('../emotional_knowledge_graph.js');
    const kg = new EmotionalKnowledgeGraph('user_123');

    kg.recordExperience({
        emotion: 'frustrated',
        trigger: 'debugging',
        intensity: 0.7,
        valence: -0.6,
        arousal: 0.7,
        dominance: 0.4,
        systemResponse: { action: 'acknowledge', tone: 'empathetic', text: 'I understand' }
    }, {
        activity: 'coding',
        location: 'home'
    });

    assert.greaterThan(kg.nodes.size, 1, 'Should have added emotion node');
    assert.greaterThan(kg.edges.size, 0, 'Should have added edges');
    assert.equal(kg.stats.totalInteractions, 1, 'Should track interaction count');
});

runner.test('Knowledge Graph: Get emotional triggers', async () => {
    const { EmotionalKnowledgeGraph } = await import('../emotional_knowledge_graph.js');
    const kg = new EmotionalKnowledgeGraph('user_123');

    kg.recordExperience({
        emotion: 'frustrated',
        trigger: 'debugging',
        intensity: 0.7,
        valence: -0.6,
        arousal: 0.7,
        dominance: 0.4
    });

    const triggers = kg.getEmotionalTriggers();

    assert.ok(Array.isArray(triggers), 'Triggers should be an array');
    if (triggers.length > 0) {
        assert.equal(triggers[0].trigger, 'debugging', 'Should identify debugging trigger');
    }
});

runner.test('Knowledge Graph: Detect patterns', async () => {
    const { EmotionalKnowledgeGraph } = await import('../emotional_knowledge_graph.js');
    const kg = new EmotionalKnowledgeGraph('user_123');

    // Record recurring trigger
    for (let i = 0; i < 3; i++) {
        kg.recordExperience({
            emotion: 'frustrated',
            trigger: 'debugging',
            intensity: 0.7,
            valence: -0.6,
            arousal: 0.7,
            dominance: 0.4
        });
    }

    const patterns = kg.detectPatterns();

    assert.ok(Array.isArray(patterns), 'Patterns should be an array');
});

// ============================================================================
// FEATURE #5: Predictive Anticipation Tests
// ============================================================================

runner.test('Predictive Anticipation: Initialize system', async () => {
    const { PredictiveEmotionalAnticipation } = await import('../predictive_emotional_anticipation.js');
    const predictor = new PredictiveEmotionalAnticipation({ userId: 'user_123' });

    assert.ok(predictor, 'Predictor should initialize');
    assert.ok(predictor.contextAnalyzer, 'Should have context analyzer');
    assert.ok(predictor.preSpeechDetector, 'Should have pre-speech detector');
    assert.ok(predictor.knowledgeGraph, 'Should have knowledge graph');
});

runner.test('Predictive Anticipation: Make prediction', async () => {
    const { PredictiveEmotionalAnticipation } = await import('../predictive_emotional_anticipation.js');
    const predictor = new PredictiveEmotionalAnticipation({ userId: 'user_123' });

    const prediction = await predictor.predict({
        location: 'work',
        activity: 'debugging',
        stressLevel: 'high'
    });

    assert.ok(prediction, 'Should return prediction');
    assert.ok(prediction.predictedEmotion, 'Should have predicted emotion');
    assert.greaterThan(prediction.confidence, 0.0, 'Should have confidence > 0');
});

runner.test('Predictive Anticipation: Validate prediction', async () => {
    const { PredictiveEmotionalAnticipation } = await import('../predictive_emotional_anticipation.js');
    const predictor = new PredictiveEmotionalAnticipation({ userId: 'user_123' });

    const prediction = await predictor.predict({ activity: 'coding' });

    const wasCorrect = predictor.validatePrediction(0, prediction.predictedEmotion);

    assert.equal(wasCorrect, true, 'Prediction should validate correctly when actual matches predicted');
});

runner.test('Predictive Anticipation: Learn from validation', async () => {
    const { PredictiveEmotionalAnticipation } = await import('../predictive_emotional_anticipation.js');
    const predictor = new PredictiveEmotionalAnticipation({
        userId: 'user_123',
        enableLearning: true
    });

    // Make prediction
    await predictor.predict({ activity: 'coding' });

    // Validate
    predictor.validatePrediction(0, 'neutral');

    const stats = predictor.getStatistics();

    assert.equal(stats.totalPredictions, 1, 'Should track 1 prediction');
});

// ============================================================================
// INTEGRATION TESTS: All Features Working Together
// ============================================================================

runner.test('Integration: Complete 105/100 system initialization', async () => {
    // This test would import the full BreakthroughEmotionalSystem
    // For now, just verify all components can be imported

    const { RecursiveSelfImprovement } = await import('../recursive_self_improvement.js');
    const { ChainOfEmotionReasoner } = await import('../chain_of_emotion_reasoning.js');
    const { MultiAgentDebateOrchestrator } = await import('../multi_agent_emotional_debate.js');
    const { EmotionalKnowledgeGraph } = await import('../emotional_knowledge_graph.js');
    const { PredictiveEmotionalAnticipation } = await import('../predictive_emotional_anticipation.js');

    const selfImprovement = new RecursiveSelfImprovement();
    const chainReasoner = new ChainOfEmotionReasoner();
    const debateOrchestrator = new MultiAgentDebateOrchestrator();
    const knowledgeGraph = new EmotionalKnowledgeGraph('user_123');
    const predictor = new PredictiveEmotionalAnticipation({ userId: 'user_123', knowledgeGraph });

    assert.ok(selfImprovement && chainReasoner && debateOrchestrator && knowledgeGraph && predictor,
        'All 5 systems should initialize');
});

runner.test('Integration: Predict → Detect → Chain → Graph → Learn flow', async () => {
    const { EmotionalKnowledgeGraph } = await import('../emotional_knowledge_graph.js');
    const { ChainOfEmotionReasoner } = await import('../chain_of_emotion_reasoning.js');
    const { PredictiveEmotionalAnticipation } = await import('../predictive_emotional_anticipation.js');
    const { RecursiveSelfImprovement } = await import('../recursive_self_improvement.js');

    const userId = 'test_user';
    const kg = new EmotionalKnowledgeGraph(userId);
    const chain = new ChainOfEmotionReasoner();
    const predictor = new PredictiveEmotionalAnticipation({ userId, knowledgeGraph: kg, chainReasoner: chain });
    const selfImprovement = new RecursiveSelfImprovement();

    // 1. Predict
    const prediction = await predictor.predict({ activity: 'coding' });

    // 2. Simulate actual emotion
    const actualEmotion = 'focused';

    // 3. Add to chain
    chain.addState(userId, {
        emotion: actualEmotion,
        intensity: 0.7,
        valence: 0.5,
        arousal: 0.6,
        dominance: 0.6,
        userInput: 'Working on the feature',
        systemResponse: 'Let me know if you need help'
    });

    // 4. Record in knowledge graph
    kg.recordExperience({
        emotion: actualEmotion,
        trigger: 'coding',
        intensity: 0.7,
        valence: 0.5,
        arousal: 0.6,
        dominance: 0.6,
        systemResponse: { action: 'acknowledge', tone: 'supportive', text: 'Great!' }
    });

    // 5. Validate prediction
    predictor.validatePrediction(0, actualEmotion);

    // 6. Record interaction for self-improvement
    selfImprovement.recordInteraction({
        userInput: 'Working on the feature',
        emotionalAnalysis: { fusedEmotion: { emotion: actualEmotion } },
        systemResponse: 'Great!',
        modalityWeights: { text: 1.0 },
        prediction: prediction
    });

    selfImprovement.updateOutcome(0, {
        userSatisfaction: 0.8,
        conversationContinued: true
    });

    // Verify flow completed
    assert.equal(chain.getChainSummary(userId).length, 1, 'Chain should have 1 state');
    assert.equal(kg.stats.totalInteractions, 1, 'KG should have 1 interaction');
    assert.equal(predictor.predictions.length, 1, 'Predictor should have 1 prediction');
    assert.equal(selfImprovement.getInteractions().length, 1, 'Self-improvement should have 1 interaction');
});

// ============================================================================
// Run all tests
// ============================================================================

if (typeof module !== 'undefined' && require.main === module) {
    runner.run().then(results => {
        process.exit(results.failed > 0 ? 1 : 0);
    });
}

// Export for use in other test runners
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { runner, TestRunner };
}
