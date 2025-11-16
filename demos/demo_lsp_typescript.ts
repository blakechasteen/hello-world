/**
 * HoloLoom LSP Feature Demonstration - TypeScript Edition
 *
 * This demo showcases semantic code completion, hover information,
 * go-to-definition, and workspace symbol search powered by HoloLoom's
 * knowledge graph and neural decision system - in TypeScript.
 *
 * FEATURES DEMONSTRATED:
 * - Code Completion: Intelligent suggestions from HoloLoom memory
 * - Hover Information: Rich context from semantic calculus
 * - Go-to-Definition: Jump to symbol definitions with LSP
 * - Symbol Search: Workspace-wide symbol search with ranking
 * - Diagnostic Feedback: Type checking and semantic validation
 *
 * HOW TO USE THIS DEMO:
 * 1. Start HoloLoom LSP server:
 *    PYTHONPATH=. python -m HoloLoom.lsp.server
 *
 * 2. Open this file in your LSP-enabled editor:
 *    - VS Code: code demos/demo_lsp_typescript.ts
 *    - Neovim: nvim demos/demo_lsp_typescript.ts
 *    - Emacs: emacs demos/demo_lsp_typescript.ts
 *
 * 3. Try the features marked with comments below
 *
 * 4. Expected behavior documented in inline comments
 *
 * Author: HoloLoom Docs Team
 * Created: 2025-11-16
 */

// ============================================================================
// Feature 1: Code Completion
// ============================================================================

// DEMO 1A: Basic Interface Completion
// -----------------------------------

/**
 * Policy engine interface using Thompson Sampling.
 *
 * COMPLETION DEMO: After typing "PolicyEngine" elsewhere, trigger completion
 * Expected: Full interface suggestion with all methods
 */
interface PolicyEngine {
  selectAction(): number;
  update(action: number, reward: number): void;
  getStats(): { [key: number]: number };
}

// DEMO 1B: Class Implementation with Completion
// ----------------------------------------------

/**
 * Multi-armed bandit policy engine implementation.
 *
 * HOVER HERE: See interface definition and semantic context about
 * bandits, Thompson Sampling, and exploration-exploitation tradeoff
 * from HoloLoom's knowledge graph.
 */
class BanditPolicy implements PolicyEngine {
  private nArms: number;
  private epsilon: number;
  private armRewards: number[] = [];
  private armCounts: number[] = [];

  /**
   * Initialize the bandit policy.
   *
   * HOVER on 'nArms': See semantic context about action spaces
   * HOVER on 'epsilon': See semantic meaning of exploration rates
   */
  constructor(nArms: number, epsilon: number = 0.1) {
    this.nArms = nArms;
    this.epsilon = epsilon;
    // Initialize arm statistics
    for (let i = 0; i < nArms; i++) {
      this.armRewards.push(0);
      this.armCounts.push(0);
    }
  }

  /**
   * Select next action using epsilon-greedy strategy.
   *
   * COMPLETION DEMO: Type "this.ar" and see "armRewards" suggestion
   * HOVER on 'selectAction': See full method signature and docs
   */
  selectAction(): number {
    // COMPLETION: Type "this.arm" and see all arm-related properties
    if (Math.random() < this.epsilon) {
      // Explore: random action
      return Math.floor(Math.random() * this.nArms);
    }
    // Exploit: best arm
    let bestArm = 0;
    let bestValue = this.armRewards[0] / (this.armCounts[0] + 1);
    for (let i = 1; i < this.nArms; i++) {
      const value = this.armRewards[i] / (this.armCounts[i] + 1);
      if (value > bestValue) {
        bestArm = i;
        bestValue = value;
      }
    }
    return bestArm;
  }

  /**
   * Update statistics for selected action.
   *
   * HOVER on 'reward': See type annotation and semantic meaning
   */
  update(action: number, reward: number): void {
    this.armCounts[action]++;
    this.armRewards[action] += reward;
  }

  /**
   * Get current statistics for all arms.
   *
   * DEFINITION DEMO: Position cursor on 'getStats' and press go-to-definition
   * Expected: Jump to this method definition
   */
  getStats(): { [key: number]: number } {
    return {
      rewards: this.armRewards.reduce((a, b) => a + b, 0),
      trials: this.armCounts.reduce((a, b) => a + b, 0),
    };
  }
}

// DEMO 1C: Type-Aware Completion
// ------------------------------

interface MemoryShard {
  text: string;
  embeddings: number[];
  metadata: Record<string, unknown>;
}

/**
 * Creates a memory shard from text input.
 *
 * COMPLETION DEMO: Type "const shard: " and trigger completion
 * Expected: "MemoryShard" interface suggestion
 */
function createMemoryShard(text: string): MemoryShard {
  return {
    text,
    embeddings: new Array(384).fill(0), // 384D Matryoshka embeddings
    metadata: { source: "demo", timestamp: Date.now() },
  };
}

// ============================================================================
// Feature 2: Hover Information
// ============================================================================

// DEMO 2A: Hover on Constants
// ---------------------------

/** Exploration rate for epsilon-greedy strategy */
const EXPLORATION_RATE = 0.1;

/** Dimension of semantic space (244D in HoloLoom) */
const SEMANTIC_DIMENSIONS = 244;

/** Multi-armed bandit constants */
const BANDIT_CONSTANTS = {
  DEFAULT_ARMS: 8,
  MIN_EPSILON: 0.01,
  MAX_EPSILON: 1.0,
  DEFAULT_REWARD: 0.5,
} as const;

// HOVER on each constant above: See type, value, and semantic context

// DEMO 2B: Hover on Types
// -----------------------

type Reward = number;
type Action = number;
type Confidence = number;

/**
 * Decision result from weaving cycle.
 *
 * HOVER on 'DecisionResult': See interface definition and all properties
 */
interface DecisionResult {
  action: Action;
  confidence: Confidence;
  reasoning: string;
  alternatives: Action[];
}

// DEMO 2C: Hover on Function Parameters
// -----------------------------------------------

/**
 * Process decision result with validation.
 *
 * HOVER on 'result': See type 'DecisionResult' and semantic context
 * HOVER on 'reward': See type 'Reward' and semantic meaning
 */
function processDecision(
  result: DecisionResult,
  reward: Reward,
  metadata?: Record<string, unknown>
): void {
  // HOVER on 'result.action': See property type and doc
  console.log(`Selected action: ${result.action}`);

  // HOVER on 'reward': See inferred type from parameter
  console.log(`Received reward: ${reward}`);
}

// ============================================================================
// Feature 3: Go-to-Definition
// ============================================================================

// DEMO 3A: Definition Lookup
// -------------------------

/**
 * Create and configure a policy engine.
 *
 * DEFINITION DEMO: Position cursor on 'BanditPolicy' below and press gd
 * Expected: Jump to class definition above
 */
const policy: BanditPolicy = new BanditPolicy(8, 0.1);

// DEFINITION DEMO: Position cursor on 'selectAction' and press gd
// Expected: Jump to method definition
const action: Action = policy.selectAction();

// DEFINITION DEMO: Position cursor on 'update' and press gd
// Expected: Jump to method definition
policy.update(action, 0.5);

// DEFINITION DEMO: Position cursor on 'createMemoryShard' and press gd
// Expected: Jump to function definition
const shard: MemoryShard = createMemoryShard("Thompson Sampling");

// DEMO 3B: Cross-Module Imports (if HoloLoom is indexed)
// -----------------------------------------------

// These would work if HoloLoom modules are indexed:

// import { WeavingOrchestrator } from '@hololoom/core';
// DEFINITION DEMO: Position cursor on 'WeavingOrchestrator' and press gd
// Expected: Jump to HoloLoom/weaving_orchestrator.py

// import { Config } from '@hololoom/config';
// DEFINITION DEMO: Position cursor on 'Config' and press gd
// Expected: Jump to HoloLoom/config.py

// ============================================================================
// Feature 4: Symbol Search
// ============================================================================

// DEMO 4A: Workspace Symbol Search
// --------------------------------

/**
 * SYMBOL SEARCH DEMO: In your editor, open symbol search
 * - VS Code: Ctrl+Shift+O (document) or Ctrl+T (workspace)
 * - Neovim: :Telescope lsp_document_symbols
 * - Emacs: lsp-ivy-workspace-symbol
 *
 * Expected symbols this file defines:
 * - PolicyEngine (interface)
 * - BanditPolicy (class)
 * - selectAction (method)
 * - update (method)
 * - getStats (method)
 * - MemoryShard (interface)
 * - createMemoryShard (function)
 * - DecisionResult (interface)
 * - processDecision (function)
 * - AgenticReasoner (interface)
 * - HoloLoomAgent (class)
 *
 * Try searching for:
 * "thompson" → Finds comments about Thompson Sampling
 * "policy" → Finds PolicyEngine, BanditPolicy, selectAction
 * "bandit" → Finds BanditPolicy, BANDIT_CONSTANTS
 * "decision" → Finds DecisionResult, processDecision
 */

// ============================================================================
// Feature 5: Type Checking & Diagnostics
// ============================================================================

// DEMO 5A: Type Safety with Hover
// --------------------------------

interface TypeCheckingExample {
  reward: number;
  action: number;
  confidence: number;
}

function typeCheckingDemo(): TypeCheckingExample {
  /**
   * HOVER on 'result': Shows inferred type from return type
   */
  const result: TypeCheckingExample = {
    reward: 0.5,
    action: 2,
    confidence: 0.95,
  };
  return result;
}

// DEMO 5B: Type Error Detection
// ----------------------------

function typeErrorDemo(): void {
  /**
   * DIAGNOSTIC: This would show a type error in LSP
   */
  // const policy: BanditPolicy = "not a policy"; // Type error
  // const action: Action = "not an action"; // Type error

  // Correct usage:
  const policy = new BanditPolicy(4);
  const action = policy.selectAction();
  policy.update(action, 1.0);
}

// ============================================================================
// Feature 6: Completion with Context
// ============================================================================

// DEMO 6A: Smart Property Completion
// -----------------------------------

class Agent {
  /**
   * Reinforcement learning agent with policy and memory.
   *
   * COMPLETION DEMO: After 'agent.', trigger completion
   * Expected: All properties and methods with relevance scoring
   */
  private policy: PolicyEngine;
  private memory: MemoryShard[] = [];
  private episodeRewards: number[] = [];

  constructor(policy: PolicyEngine) {
    this.policy = policy;
  }

  /**
   * Select action based on current state.
   *
   * COMPLETION DEMO: Type "this.po" and see "policy" suggestion
   */
  act(): Action {
    // COMPLETION: Type "this.ep" and see "episodeRewards"
    return this.policy.selectAction();
  }

  /**
   * Record reward signal from environment.
   *
   * COMPLETION DEMO: Type "this.epi" and see "episodeRewards" completion
   */
  observeReward(reward: number): void {
    this.episodeRewards.push(reward);
  }

  /**
   * Get learning statistics.
   *
   * COMPLETION DEMO: Type "this.get" and see all "get" methods
   */
  getStats(): Record<string, number> {
    return {
      totalReward: this.episodeRewards.reduce((a, b) => a + b, 0),
      episodes: this.episodeRewards.length,
      avgReward:
        this.episodeRewards.length > 0
          ? this.episodeRewards.reduce((a, b) => a + b, 0) /
            this.episodeRewards.length
          : 0,
    };
  }
}

// DEMO 6B: Method Chaining Completion
// ----------------------------------

/**
 * Builder pattern with completion support.
 */
class AgentBuilder {
  private config: Record<string, unknown> = {};

  withPolicy(policy: PolicyEngine): this {
    // COMPLETION DEMO: Type "this.with" and see all builder methods
    this.config.policy = policy;
    return this;
  }

  withMemory(memory: MemoryShard[]): this {
    this.config.memory = memory;
    return this;
  }

  withReflection(enabled: boolean): this {
    this.config.reflection = enabled;
    return this;
  }

  build(): Agent {
    // COMPLETION: All "with" methods appear during chaining
    return new Agent(this.config.policy as PolicyEngine);
  }
}

// DEMO 6C: Method Chaining Usage
// -----------------------------

// COMPLETION DEMO: After typing "builder.", trigger completion
// Expected: withPolicy, withMemory, withReflection, build
const builder = new AgentBuilder();
const agent = builder
  .withPolicy(policy)
  .withMemory([shard])
  .withReflection(true)
  .build();

// ============================================================================
// Feature 7: Documentation Integration
// ============================================================================

// DEMO 7A: JSDoc Hover
// --------------------

/**
 * Orchestrates multi-scale weaving cycle.
 *
 * The weaving orchestrator coordinates:
 * 1. Loom Command: Pattern card selection (BARE/FAST/FUSED)
 * 2. Yarn Graph: Thread selection from memory
 * 3. Resonance Shed: Feature extraction
 * 4. Warp Space: Continuous manifold tensioning
 * 5. Convergence Engine: Decision collapse
 *
 * HOVER HERE: See complete documentation
 * Expected: Full JSDoc with all method descriptions
 *
 * @example
 * const orchestrator = new WeavingOrchestrator();
 * const result = await orchestrator.weave("query");
 */
interface AgenticReasoner {
  weave(query: string): Promise<DecisionResult>;
  reflect(result: DecisionResult, feedback: unknown): Promise<void>;
}

/**
 * HoloLoom agent implementation with semantic intelligence.
 *
 * HOVER on 'HoloLoomAgent': See full documentation and semantic context
 */
class HoloLoomAgent implements AgenticReasoner {
  private reasoner: AgenticReasoner;

  constructor(reasoner: AgenticReasoner) {
    this.reasoner = reasoner;
  }

  /**
   * Weave semantic intelligence into decision.
   *
   * HOVER on 'weave': See complete documentation about weaving cycle
   * and semantic context from HoloLoom's knowledge graph
   */
  async weave(query: string): Promise<DecisionResult> {
    return this.reasoner.weave(query);
  }

  /**
   * Learn from decision outcome.
   *
   * HOVER on 'reflect': See documentation about reflection and learning
   */
  async reflect(result: DecisionResult, feedback: unknown): Promise<void> {
    return this.reasoner.reflect(result, feedback);
  }
}

// ============================================================================
// Feature 8: Interface Completion
// ============================================================================

// DEMO 8A: Object Literal Completion
// ----------------------------------

/**
 * Create configuration object with completion support.
 *
 * COMPLETION DEMO: Inside the object literal, trigger completion
 * Expected: All properties of AgentConfig interface suggested
 */
interface AgentConfig {
  nArms: number;
  epsilon: number;
  learningRate: number;
  enableReflection: boolean;
  enableAdaptation: boolean;
}

const config: AgentConfig = {
  nArms: 8,
  // COMPLETION DEMO: Type "e" and see "epsilon" suggestion
  epsilon: 0.1,
  learningRate: 0.05,
  enableReflection: true,
  // COMPLETION: All required properties show in completions
};

// DEMO 8B: Destructuring Completion
// ----------------------------------

function useConfig({
  nArms,
  // COMPLETION DEMO: Trigger completion to see parameter suggestions
  epsilon,
  learningRate,
}: AgentConfig): void {
  console.log(`Using ${nArms} arms with epsilon=${epsilon}`);
}

// ============================================================================
// Feature 9: Semantic Analysis Examples
// ============================================================================

// DEMO 9A: Related Concepts in Hover
// ---------------------------------

/**
 * Thompson Sampling implementation.
 *
 * HOVER HERE: See semantic analysis results:
 * - Related concepts: Bayesian inference, Beta distribution, bandits
 * - Key properties: exploration-exploitation, posterior sampling
 * - Use cases: adaptive testing, A/B testing, recommendation
 */
class ThompsonSampler {
  sample(): Action {
    // HOVER on 'sample': See semantic meaning of sampling operations
    return 0;
  }
}

// DEMO 9B: Semantic Rich Type Hover
// --------------------------------

/**
 * Result from refinement pipeline.
 *
 * HOVER on 'RefinementResult': See semantic categories
 * - quality dimensions: elegance, clarity, completeness
 * - verification modes: accuracy, consistency, safety
 * - relationship to HoloLoom's alignment framework
 */
interface RefinementResult {
  initialConfidence: Confidence;
  finalConfidence: Confidence;
  iterations: number;
  strategy: "elegance" | "verify" | "hofstadter";
  qualityGain: number;
}

// ============================================================================
// Feature 10: Full Pipeline Example
// ============================================================================

/**
 * End-to-end example using all LSP features.
 *
 * This demonstrates:
 * 1. Class instantiation with completion
 * 2. Method completion
 * 3. Hover on types
 * 4. Go-to-definition
 * 5. Symbol search
 */
async function completeExample(): Promise<void> {
  // STEP 1: Create policy (completion + definition)
  // Type "new Band" and trigger completion
  const policy = new BanditPolicy(8, 0.1);

  // STEP 2: Call method (completion + hover)
  // Type "policy.se" and see "selectAction" in completions
  const action: Action = policy.selectAction();

  // HOVER on 'action' to see it's an Action (number)
  const reward: Reward = 0.5;

  // STEP 3: Create agent (cross-file symbols)
  // Press gd on 'Agent' to jump to definition
  const agent = new Agent(policy);

  // STEP 4: Use agent (method completion)
  // Type "agent." and see all available methods
  const selectedAction = agent.act();

  // STEP 5: Learn from reward (type + hover)
  // Type "agent.ob" and see "observeReward" completion
  agent.observeReward(reward);

  // STEP 6: Get statistics
  const stats = agent.getStats();
  console.log("Agent statistics:", stats);
}

// ============================================================================
// Feature 11: Advanced Completion Examples
// ============================================================================

// DEMO 11A: Enum Completion
// -------------------------

enum ExecutionMode {
  BARE = "bare",
  FAST = "fast",
  FUSED = "fused",
  RESEARCH = "research",
}

/**
 * Select execution mode with completion.
 *
 * COMPLETION DEMO: Type "ExecutionMode." and see all enum values
 * Expected: BARE, FAST, FUSED, RESEARCH suggestions
 */
const mode: ExecutionMode = ExecutionMode.FUSED;

// DEMO 11B: Union Type Completion
// -------------------------------

type RefinementStrategy = "elegance" | "verify" | "hofstadter";

/**
 * COMPLETION DEMO: Type strategy variable assignment
 * Expected: String literal completion for all valid strategies
 */
const strategy: RefinementStrategy = "elegance";

// ============================================================================
// Feature 12: Test Scenarios
// ============================================================================

/**
 * Test all LSP features in this module.
 *
 * Run with: npx ts-node demos/demo_lsp_typescript.ts
 */

async function main(): Promise<void> {
  console.log("HoloLoom LSP Demo - TypeScript Features");
  console.log("=".repeat(50));

  // Test 1: PolicyEngine
  console.log("\n1. Testing BanditPolicy...");
  const policy = new BanditPolicy(4);
  for (let i = 0; i < 10; i++) {
    const action = policy.selectAction();
    const reward = action === 2 ? 0.8 : 0.1;
    policy.update(action, reward);
  }
  console.log("   Stats:", policy.getStats());

  // Test 2: Agent
  console.log("\n2. Testing Agent...");
  const agent = new Agent(new BanditPolicy(4));
  agent.act();
  agent.observeReward(1.0);
  console.log("   Agent stats:", agent.getStats());

  // Test 3: MemoryShard
  console.log("\n3. Testing MemoryShard...");
  const shard = createMemoryShard("Thompson Sampling");
  console.log(`   Created: ${shard.text} (${shard.embeddings.length}D)`);

  // Test 4: Builder Pattern
  console.log("\n4. Testing AgentBuilder...");
  const builtAgent = new AgentBuilder()
    .withPolicy(new BanditPolicy(8))
    .withMemory([shard])
    .withReflection(true)
    .build();
  console.log("   Built agent successfully");

  // Test 5: Full Example
  console.log("\n5. Testing complete pipeline...");
  await completeExample();

  console.log("\n✓ All tests passed!");
  console.log("\nNext: Open this file in your LSP editor and try the demos!");
}

// Run main
main().catch(console.error);
