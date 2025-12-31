/**
 * HoloLoom SDK - Basic Usage Examples
 * Shows common patterns for using the SDK
 */

import { createHoloLoomClient, type Spacetime, type RecallResult } from '../src/index';

// ============================================================================
// Example 1: Create Client
// ============================================================================

// Browser environment (auto-detects origin)
const client = createHoloLoomClient();

// Or with explicit configuration
const customClient = createHoloLoomClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key',
  timeout: 30000,
});

// ============================================================================
// Example 2: Experience - Store Memories
// ============================================================================

async function storeMemory() {
  const result = await client.experience({
    content: 'Thompson Sampling is a Bayesian approach to exploration-exploitation',
    metadata: {
      source: 'research',
      confidence: 0.95,
      tags: ['bayesian', 'bandits', 'learning'],
    },
  });

  if (result.success) {
    console.log(`Memory stored: ${result.data.memoryId}`);
    console.log(`Embedded: ${result.data.embedded}`);
    console.log(`Graph updated: ${result.data.graphUpdated}`);
  } else {
    console.error(`Error: ${result.error}`);
  }
}

// ============================================================================
// Example 3: Recall - Search Memories
// ============================================================================

async function searchMemories() {
  const result = await client.recall(
    'What did I learn about Thompson Sampling?',
    {
      strategy: 'CONNECTED', // vs RECENT, SIMILAR, RESONANT, BALANCED
      limit: 10,
      minRelevance: 0.3,
    },
  );

  if (result.success) {
    const { memories, totalCount, strategy } = result.data;

    console.log(`Found ${totalCount} memories using strategy: ${strategy}`);

    // Print top 3 results
    memories.slice(0, 3).forEach((mem, i) => {
      console.log(`\n${i + 1}. [${mem.id}]`);
      console.log(`   Content: ${mem.content.substring(0, 60)}...`);
      console.log(`   Relevance: ${mem.relevance?.toFixed(2)}`);
      console.log(`   Confidence: ${mem.confidence?.toFixed(2)}`);
      console.log(`   Activation: ${mem.activation?.toFixed(2)}`);
    });
  }
}

// ============================================================================
// Example 4: Reflect - Learn from Feedback
// ============================================================================

async function provideFeedback() {
  const result = await client.reflect({
    memoryIds: ['mem_123', 'mem_456', 'mem_789'],
    feedback: {
      helpful: true,
      accurate: true,
      relevant: true,
      quality: 0.95, // 0.0-1.0
    },
    tags: ['important', 'verified'],
  });

  if (result.success) {
    console.log(`Updated ${result.data.updated} memories`);
    console.log(`Feedback recorded:`, result.data.feedback);
  }
}

// ============================================================================
// Example 5: Reason - Multi-Step Reasoning
// ============================================================================

async function multiStepReasoning() {
  const result = await client.reason(
    'Compare Thompson Sampling vs UCB algorithm',
    {
      mode: 'VERIFY', // DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE
      maxSteps: 5,
      verificationThreshold: 0.7,
    },
  );

  if (result.success) {
    const { response, confidence, mode, stepsTaken, verification } = result.data;

    console.log(`\nResponse (${mode} mode):`);
    console.log(response);
    console.log(`\nConfidence: ${(confidence * 100).toFixed(1)}%`);

    console.log(`\nReasoning Steps:`);
    stepsTaken.forEach((step, i) => {
      console.log(`  ${i + 1}. [${step.type}] ${step.query}`);
      console.log(`     → ${step.result}`);
      console.log(`     Confidence: ${(step.confidence * 100).toFixed(1)}%`);
    });

    if (verification?.verified) {
      console.log(`\n✓ Verification successful`);
    } else if (verification?.issues?.length) {
      console.log(`\n⚠ Verification found issues:`);
      verification.issues.forEach(issue => console.log(`  - ${issue}`));

      if (verification.corrections) {
        console.log(`Corrections:`);
        verification.corrections.forEach(c => console.log(`  - ${c}`));
      }
    }
  }
}

// ============================================================================
// Example 6: Query - Full Orchestrator
// ============================================================================

async function fullQuery() {
  const result = await client.query(
    'Explain Thompson Sampling and its applications',
    {
      mode: 'RESEARCH', // Full multi-query reasoning
      maxSteps: 5,
      complexityLevel: 'COMPLEX',
      format: 'markdown',
      includeProvenance: true,
    },
  );

  if (result.success) {
    const spacetime: Spacetime = result.data;

    console.log(`\nQuery Result:`);
    console.log(`Response:\n${spacetime.response}\n`);
    console.log(`Confidence: ${(spacetime.confidence * 100).toFixed(1)}%`);
    console.log(`Tool used: ${spacetime.toolUsed}`);

    if (spacetime.sources) {
      console.log(`\nSources (${spacetime.sources.length}):`);
      spacetime.sources.forEach((src, i) => {
        console.log(`  ${i + 1}. ${src.content.substring(0, 50)}...`);
        console.log(`     Relevance: ${src.relevance?.toFixed(2)}`);
      });
    }

    if (spacetime.trace) {
      console.log(`\nExecution Trace:`);
      spacetime.trace.stages.forEach(stage => {
        const status = stage.status === 'success' ? '✓' : '⚠';
        console.log(`  ${status} ${stage.name}: ${stage.durationMs}ms`);
      });
      console.log(`  Total: ${spacetime.trace.totalMs}ms`);
    }

    if (spacetime.metadata) {
      console.log(`\nMetadata:`);
      console.log(`  Cache hit: ${spacetime.metadata.cacheHit ? 'Yes' : 'No'}`);
      console.log(`  Complexity: ${spacetime.metadata.complexity}`);
      console.log(`  Threads: ${spacetime.metadata.threadCount}`);
    }
  } else {
    console.error(`Query failed: ${result.error}`);
  }
}

// ============================================================================
// Example 7: Batch Experience
// ============================================================================

async function batchExperience() {
  const result = await client.experienceBatch({
    items: [
      {
        content: 'Machine learning is about learning patterns from data',
        metadata: { source: 'textbook', chapter: 1 },
      },
      {
        content: 'Thompson Sampling explores uncertain options proportionally to probability of being best',
        metadata: { source: 'research', year: 2023 },
      },
      {
        content: 'Neural networks are inspired by biological neurons',
        metadata: { source: 'course', instructor: 'Dr. LeCun' },
      },
    ],
  });

  if (result.success) {
    console.log(`Stored ${result.data.length} memories in batch`);
    result.data.forEach(item => {
      console.log(`  - ${item.memoryId} (${item.timestamp})`);
    });
  }
}

// ============================================================================
// Example 8: Error Handling
// ============================================================================

async function demonstrateErrorHandling() {
  try {
    const result = await client.query(
      'Invalid query request',
      undefined,
      { timeout: 5000, retries: 3 }, // Custom request options
    );

    if (!result.success) {
      console.error(`API returned error: ${result.error}`);
      if (result.metadata) {
        console.error(`Request ID: ${result.metadata.timestamp}`);
      }
    }
  } catch (error) {
    if (error instanceof Error) {
      console.error(`Client error: ${error.message}`);
    }
  }
}

// ============================================================================
// Example 9: Configuration & Runtime Updates
// ============================================================================

async function demonstrateConfiguration() {
  // Get current config
  const config = customClient.getConfig();
  console.log('Current config:', config);

  // Update at runtime
  customClient.configure({
    timeout: 60000,
    baseUrl: 'https://api.hololoom.io',
  });

  console.log('Updated config:', customClient.getConfig());
}

// ============================================================================
// Example 10: Streaming Context (Advanced)
// ============================================================================

async function streamingQuery() {
  // For streaming responses, wrap in event handler
  const result = await client.query(
    'Tell me about Thompson Sampling',
    { mode: 'DIRECT' },
  );

  if (result.success && result.data.response) {
    // Simulate streaming output
    const response = result.data.response;
    const chunkSize = 50;

    for (let i = 0; i < response.length; i += chunkSize) {
      console.log(response.substring(i, i + chunkSize));
      await new Promise(resolve => setTimeout(resolve, 10)); // 10ms per chunk
    }
  }
}

// ============================================================================
// Main: Run Examples
// ============================================================================

async function runExamples() {
  console.log('=== HoloLoom SDK Examples ===\n');

  try {
    // Check health first
    const health = await client.health();
    if (health.success) {
      console.log(`Connected to HoloLoom ${health.data.version}\n`);
    }

    // Run examples (uncomment to try)
    // await storeMemory();
    // await searchMemories();
    // await provideFeedback();
    // await multiStepReasoning();
    // await fullQuery();
    // await batchExperience();
    // await demonstrateErrorHandling();
    // await demonstrateConfiguration();
    // await streamingQuery();

    console.log('\nExamples complete!');
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Uncomment to run
// runExamples().catch(console.error);

export {
  storeMemory,
  searchMemories,
  provideFeedback,
  multiStepReasoning,
  fullQuery,
  batchExperience,
  demonstrateErrorHandling,
  demonstrateConfiguration,
  streamingQuery,
};
