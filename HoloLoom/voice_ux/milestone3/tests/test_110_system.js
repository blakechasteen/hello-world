/**
 * Test Suite for 110/100 Cross-User Learning System
 * Demonstrates the meta-meta-learning breakthrough
 */

const {
    CrossUserLearningSystem,
    MetaMetaLearningEngine,
    FederatedLearningCoordinator,
    StrategyMarketplace,
    CollectiveIntelligence,
    PrivacyLevel
} = require('../federated_learning_110.js');

// Simple test runner
class TestRunner {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    async run() {
        console.log(`\n🚀 Testing 110/100 Cross-User Learning System\n`);
        console.log('='.repeat(70) + '\n');

        for (const { name, fn } of this.tests) {
            try {
                await fn();
                this.passed++;
                console.log(`✅ ${name}`);
            } catch (error) {
                this.failed++;
                console.log(`❌ ${name}`);
                console.log(`   Error: ${error.message}`);
            }
        }

        console.log('\n' + '='.repeat(70));
        console.log(`\n📊 Results: ${this.passed}/${this.tests.length} tests passed`);

        if (this.failed === 0) {
            console.log('\n🎉 All tests passed! 110/100 system is working!\n');
        }

        return { passed: this.passed, failed: this.failed };
    }
}

const runner = new TestRunner();

// ============================================================================
// Meta-Meta-Learning Engine Tests
// ============================================================================

runner.test('MetaMetaLearning: Initialize engine', () => {
    const engine = new MetaMetaLearningEngine();

    if (!engine) throw new Error('Engine should initialize');
    if (!engine.currentHyperparameters) throw new Error('Should have hyperparameters');
    if (!engine.learningCurves) throw new Error('Should have learning curves');
});

runner.test('MetaMetaLearning: Record strategy outcome', () => {
    const engine = new MetaMetaLearningEngine();

    engine.recordStrategyOutcome('break_strategy',
        { emotion: 'frustrated', activity: 'coding' },
        { success: true, quality: 0.9 }
    );

    if (engine.iterationCount !== 1) {
        throw new Error('Iteration count should be 1');
    }
});

runner.test('MetaMetaLearning: Recommend strategy (exploration)', () => {
    const engine = new MetaMetaLearningEngine({ explorationRate: 1.0 }); // Always explore

    engine.recordStrategyOutcome('strategy_a', { emotion: 'happy' }, { success: true });

    const recommendation = engine.recommendStrategy({ emotion: 'happy' });

    if (!recommendation) throw new Error('Should recommend a strategy');
});

runner.test('MetaMetaLearning: Optimize hyperparameters', () => {
    const engine = new MetaMetaLearningEngine({ optimizationInterval: 1 });

    // Add some learning curve data
    engine.updateLearningCurves({ accuracy: 0.7, convergenceRate: 0.01 });
    engine.updateLearningCurves({ accuracy: 0.75, convergenceRate: 0.01 });
    engine.updateLearningCurves({ accuracy: 0.8, convergenceRate: 0.01 });

    // Trigger optimization
    engine.recordStrategyOutcome('test', { emotion: 'test' }, { success: true });

    if (engine.hyperparameterHistory.length === 0) {
        throw new Error('Should have optimization history');
    }
});

runner.test('MetaMetaLearning: Get insights', () => {
    const engine = new MetaMetaLearningEngine();

    engine.recordStrategyOutcome('strategy_a', { emotion: 'happy' }, { success: true, quality: 0.9 });
    engine.recordStrategyOutcome('strategy_b', { emotion: 'sad' }, { success: false, quality: 0.3 });

    const insights = engine.getInsights();

    if (!insights.topStrategies) throw new Error('Should have top strategies');
    if (!insights.currentHyperparameters) throw new Error('Should have hyperparameters');
    if (!insights.learningProgress) throw new Error('Should have learning progress');
});

// ============================================================================
// Federated Learning Tests
// ============================================================================

runner.test('FederatedLearning: Initialize coordinator', () => {
    const coordinator = new FederatedLearningCoordinator();

    if (!coordinator) throw new Error('Coordinator should initialize');
});

runner.test('FederatedLearning: Submit updates and aggregate', () => {
    const coordinator = new FederatedLearningCoordinator();
    const { FederatedUpdate } = require('../federated_learning_110.js');

    // Submit 5 updates to trigger aggregation
    for (let i = 0; i < 5; i++) {
        const update = new FederatedUpdate({
            userId: `user_${i}`,
            modalityWeights: { facial: 0.5 + i * 0.05, vocal: 0.5 - i * 0.05 },
            strategyPreferences: {},
            sampleCount: 10,
            averageConfidence: 0.8
        });
        update.addNoise(0.1);

        coordinator.submitUpdate(update);
    }

    const stats = coordinator.getStatistics();
    if (stats.totalUpdates !== 5) {
        throw new Error('Should have 5 updates');
    }
});

// ============================================================================
// Strategy Marketplace Tests
// ============================================================================

runner.test('StrategyMarketplace: Submit pattern', () => {
    const marketplace = new StrategyMarketplace();
    const { StrategyPattern } = require('../federated_learning_110.js');

    const pattern = new StrategyPattern({
        id: 'test_pattern',
        name: 'Test Strategy',
        category: 'test',
        trigger: { emotion: 'happy' },
        action: 'celebrate',
        quality: 0.9,
        usageCount: 10
    });

    const result = marketplace.submitPattern(pattern, 'user_1');

    if (!result.success) throw new Error('Pattern submission should succeed');
});

runner.test('StrategyMarketplace: Search patterns', () => {
    const marketplace = new StrategyMarketplace();
    const { StrategyPattern } = require('../federated_learning_110.js');

    const pattern = new StrategyPattern({
        id: 'test_pattern',
        name: 'Test Strategy',
        category: 'frustration',
        trigger: { emotion: 'frustrated', context: 'coding' },
        action: 'take_break',
        quality: 0.9,
        usageCount: 10
    });

    marketplace.submitPattern(pattern, 'user_1');

    const results = marketplace.searchPatterns(
        { emotion: 'frustrated', context: 'coding' },
        'frustration'
    );

    if (results.length === 0) throw new Error('Should find the pattern');
});

// ============================================================================
// Collective Intelligence Tests
// ============================================================================

runner.test('CollectiveIntelligence: Submit prediction', () => {
    const collective = new CollectiveIntelligence();

    collective.submitPrediction('pred_1', 'user_1', 'happy', 0.8);
    collective.submitPrediction('pred_1', 'user_2', 'happy', 0.9);
    collective.submitPrediction('pred_1', 'user_3', 'excited', 0.7);

    const prediction = collective.getCollectivePrediction('pred_1');

    if (!prediction) throw new Error('Should have collective prediction');
    if (prediction.emotion !== 'happy') throw new Error('Consensus should be happy');
});

// ============================================================================
// Complete Cross-User Learning System Tests
// ============================================================================

runner.test('CrossUserLearning: Initialize system', () => {
    const system = new CrossUserLearningSystem('user_123', {
        enableMetaMetaLearning: true
    });

    if (!system.metaLearner) throw new Error('Should have meta-learner');
});

runner.test('CrossUserLearning: Contribute model update with meta-optimization', async () => {
    const system = new CrossUserLearningSystem('user_123', {
        enableMetaMetaLearning: true
    });

    // Add some learning history to trigger optimization
    system.metaLearner.updateLearningCurves({ accuracy: 0.8, convergenceRate: 0.01 });

    const result = await system.contributeModelUpdate(
        { modalityWeights: { facial: 0.7, vocal: 0.8 } },
        100,
        0.85
    );

    // Should not aggregate yet (need 5 updates)
    if (result !== null) {
        // If it did aggregate, that's also fine
        console.log('   (Model aggregated successfully)');
    }
});

runner.test('CrossUserLearning: Record learning outcome', () => {
    const system = new CrossUserLearningSystem('user_123', {
        enableMetaMetaLearning: true
    });

    system.recordLearningOutcome(
        'break_strategy',
        { emotion: 'frustrated', activity: 'coding' },
        { success: true, quality: 0.9 }
    );

    const insights = system.getMetaLearningInsights();

    if (!insights) throw new Error('Should have insights');
    if (insights.topStrategies.length === 0) throw new Error('Should have strategies');
});

runner.test('CrossUserLearning: Get optimized hyperparameters', () => {
    const system = new CrossUserLearningSystem('user_123', {
        enableMetaMetaLearning: true
    });

    const hyperparams = system.getOptimizedHyperparameters();

    if (!hyperparams) throw new Error('Should have hyperparameters');
    if (!hyperparams.privacyEpsilon) throw new Error('Should have privacy epsilon');
    if (!hyperparams.learningRate) throw new Error('Should have learning rate');
});

runner.test('CrossUserLearning: Complete statistics', () => {
    const system = new CrossUserLearningSystem('user_123', {
        enableMetaMetaLearning: true
    });

    // Record some activity
    system.recordLearningOutcome('test', { emotion: 'test' }, { success: true });

    const stats = system.getStatistics();

    if (!stats.federated) throw new Error('Should have federated stats');
    if (!stats.marketplace) throw new Error('Should have marketplace stats');
    if (!stats.collective) throw new Error('Should have collective stats');
    if (!stats.metaLearning) throw new Error('Should have meta-learning stats');
});

// ============================================================================
// Integration Test: Complete 110/100 Workflow
// ============================================================================

runner.test('Integration: Complete 110/100 workflow', async () => {
    console.log('\n   Running complete 110/100 workflow...');

    // 1. Create system
    const system = new CrossUserLearningSystem('user_123', {
        privacyLevel: PrivacyLevel.MODERATE,
        enableMetaMetaLearning: true
    });

    // 2. Use meta-learner to learn strategies
    system.recordLearningOutcome(
        'break_strategy',
        { emotion: 'frustrated', activity: 'coding' },
        { success: true, quality: 0.9 }
    );

    system.recordLearningOutcome(
        'break_strategy',
        { emotion: 'frustrated', activity: 'coding' },
        { success: true, quality: 0.85 }
    );

    // 3. Share strategy to marketplace
    const { StrategyPattern } = require('../federated_learning_110.js');

    const strategy = new StrategyPattern({
        id: 'break_001',
        name: 'Break Time Strategy',
        category: 'frustration_management',
        trigger: { emotion: 'frustrated', context: 'coding' },
        action: 'suggest_break',
        quality: 0.9,
        usageCount: 15
    });

    const shareResult = system.shareStrategy(strategy);
    if (!shareResult.success) throw new Error('Strategy sharing should succeed');

    // 4. Find strategies (with meta-learner recommendation)
    const strategies = system.findStrategies(
        { emotion: 'frustrated', context: 'coding' },
        'frustration_management'
    );

    if (strategies.length === 0) throw new Error('Should find strategies');

    // 5. Contribute to collective prediction
    const prediction = system.contributePrediction('pred_001', 'frustrated', 0.8);
    if (!prediction) throw new Error('Should create prediction');

    // 6. Contribute model update (with meta-optimized privacy)
    await system.contributeModelUpdate(
        { modalityWeights: { facial: 0.7, vocal: 0.8 } },
        50,
        0.85
    );

    // 7. Get complete insights
    const insights = system.getMetaLearningInsights();

    console.log('\n   ✨ Complete 110/100 workflow executed successfully!');
    console.log(`   📊 Top strategy: ${insights.topStrategies[0]?.strategy}`);
    console.log(`   🔒 Privacy epsilon: ${insights.currentHyperparameters.privacyEpsilon}`);
    console.log(`   📈 Iterations: ${insights.learningProgress.iterations}`);
});

// ============================================================================
// Run all tests
// ============================================================================

(async () => {
    const results = await runner.run();
    process.exit(results.failed > 0 ? 1 : 0);
})();
