#!/usr/bin/env python3
"""
Warp Drive Showcase - Complete Integration Demos
=================================================
Realistic examples showing the full power of HoloLoom's warp drive.

Scenarios:
1. Multi-document semantic search with curved manifolds
2. Quantum-inspired decision making with superposition
3. Real-time chat with GPU-accelerated attention
4. Knowledge graph exploration via tensor decomposition
5. Adaptive learning with Fisher information geometry
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import List, Dict

# Core imports
from HoloLoom.warp.space import WarpSpace
from HoloLoom.warp.advanced import (
    RiemannianManifold,
    TensorDecomposer,
    QuantumWarpOperations,
    FisherInformationGeometry
)
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.convergence.engine import ConvergenceEngine, CollapseStrategy
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Demo 1: Semantic Search with Riemannian Geometry
# ============================================================================

async def demo_1_semantic_search():
    """
    Demonstrate semantic search using curved manifolds.

    Use case: Finding related documents where similarity is measured
    via geodesic distance on a learned semantic manifold.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: Semantic Search with Riemannian Geometry")
    logger.info("="*80)

    # Document corpus
    documents = [
        "Thompson Sampling uses Bayesian exploration for multi-armed bandits",
        "Neural networks learn hierarchical feature representations",
        "Reinforcement learning agents optimize long-term rewards",
        "Graph neural networks process structured data",
        "Attention mechanisms enable context-aware transformers",
        "Bayesian optimization efficiently searches hyperparameter spaces",
        "Policy gradient methods learn stochastic policies",
        "Variational autoencoders learn latent representations",
        "Monte Carlo tree search explores game trees",
        "Proximal policy optimization improves RL stability"
    ]

    logger.info(f"\nDocument corpus: {len(documents)} documents")

    # Initialize embeddings and warp space
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    # Tension documents into warp space
    logger.info("\n1. Tensioning documents into Warp Space...")
    await warp.tension(documents)

    # Create Riemannian manifold (spherical geometry for semantic space)
    manifold = RiemannianManifold(dim=384, curvature=0.5)
    logger.info(f"   Manifold: dim=384, curvature=0.5 (spherical)")

    # Query
    query = "How do Bayesian methods help with exploration?"
    logger.info(f"\n2. Query: '{query}'")

    # Get query embedding
    query_emb_dict = embedder.encode_scales([query])
    query_emb = query_emb_dict[384][0]

    # Compute attention (standard)
    attention_standard = warp.apply_attention(query_emb)

    # Compute geodesic distances
    logger.info("\n3. Computing geodesic distances...")
    geodesic_distances = []
    for thread in warp.threads:
        dist = manifold.geodesic_distance(query_emb, thread.embedding)
        geodesic_distances.append(dist)

    # Sort by geodesic distance
    ranked_indices = np.argsort(geodesic_distances)

    logger.info("\n4. Top 3 results (by geodesic distance):")
    for i, idx in enumerate(ranked_indices[:3]):
        logger.info(f"   {i+1}. [{geodesic_distances[idx]:.3f}] {documents[idx]}")

    logger.info("\n5. Comparison with Euclidean (attention):")
    attention_ranked = np.argsort(attention_standard)[::-1]
    for i, idx in enumerate(attention_ranked[:3]):
        logger.info(f"   {i+1}. [{attention_standard[idx]:.3f}] {documents[idx]}")

    # Visualization of manifold structure
    logger.info("\n6. Manifold insights:")
    logger.info(f"   Geodesic distances respect semantic curvature")
    logger.info(f"   Related concepts cluster on curved manifold")

    warp.collapse()
    logger.info("\nDEMO 1 COMPLETE!")


# ============================================================================
# Demo 2: Quantum Decision Making
# ============================================================================

async def demo_2_quantum_decisions():
    """
    Demonstrate quantum-inspired decision making.

    Use case: AI agent must choose actions while maintaining superposition
    of multiple strategies, collapsing only when necessary.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: Quantum-Inspired Decision Making")
    logger.info("="*80)

    # Possible strategies
    strategies = [
        "Explore new territory aggressively",
        "Exploit known good areas conservatively",
        "Balance exploration and exploitation",
        "Use Thompson Sampling for adaptive exploration",
        "Follow epsilon-greedy with decreasing epsilon"
    ]

    logger.info(f"\nStrategies: {len(strategies)} options")

    # Encode strategies
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    embeddings_dict = embedder.encode_scales(strategies)
    strategy_embeddings = embeddings_dict[384]

    # Normalize to unit vectors (quantum states)
    strategy_states = [emb / (np.linalg.norm(emb) + 1e-10) for emb in strategy_embeddings]

    logger.info("\n1. Creating quantum superposition of strategies...")

    # Create superposition with adaptive amplitudes
    # (Simulate learned preferences)
    amplitudes = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    superposed_state = QuantumWarpOperations.superposition(
        strategy_states,
        amplitudes
    )

    logger.info(f"   Superposition created: |ψ⟩ = Σᵢ αᵢ|ψᵢ⟩")
    logger.info(f"   Amplitudes: {[f'{a:.3f}' for a in amplitudes]}")

    # Environment observation (changes the context)
    observation = "The environment has high uncertainty and unknown rewards"
    logger.info(f"\n2. Observation: '{observation}'")

    # Encode observation
    obs_emb = embedder.encode_scales([observation])[384][0]
    obs_state = obs_emb / (np.linalg.norm(obs_emb) + 1e-10)

    # Entangle observation with strategy superposition
    logger.info("\n3. Entangling observation with strategy state...")
    # (Simplified: use as measurement basis perturbation)

    # Measurement (collapse to decision)
    logger.info("\n4. Measuring quantum state (collapse to decision)...")

    # Modified basis based on observation
    perturbed_states = []
    for state in strategy_states:
        # Perturb based on observation similarity
        similarity = np.dot(state, obs_state)
        perturbation = obs_state * similarity * 0.2
        perturbed = state + perturbation
        perturbed = perturbed / (np.linalg.norm(perturbed) + 1e-10)
        perturbed_states.append(perturbed)

    measured_idx, probability, collapsed_state = QuantumWarpOperations.measure(
        superposed_state,
        perturbed_states,
        collapse=True
    )

    logger.info(f"\n5. DECISION COLLAPSED:")
    logger.info(f"   Selected strategy: '{strategies[measured_idx]}'")
    logger.info(f"   Measurement probability: {probability:.3f}")

    # Simulate decoherence over time
    logger.info("\n6. Simulating decoherence (environmental interaction)...")
    for t in [1, 2, 3]:
        decohered = QuantumWarpOperations.decoherence(
            collapsed_state,
            noise_level=0.05 * t
        )
        fidelity = abs(np.dot(collapsed_state, decohered))**2
        logger.info(f"   Time step {t}: fidelity = {fidelity:.3f}")

    logger.info("\nDEMO 2 COMPLETE!")


# ============================================================================
# Demo 3: Real-Time Chat with GPU Acceleration
# ============================================================================

async def demo_3_gpu_chat():
    """
    Demonstrate real-time chat with GPU-accelerated warp operations.

    Use case: High-throughput conversational AI with parallel attention.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: Real-Time Chat with GPU Acceleration")
    logger.info("="*80)

    # Check GPU availability
    try:
        from HoloLoom.warp.optimized import GPUWarpSpace
        import torch

        if torch.cuda.is_available():
            logger.info(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
            use_gpu = True
        else:
            logger.info("\nGPU not available, using CPU")
            use_gpu = False
    except ImportError:
        logger.info("\nPyTorch not available, using standard Warp Space")
        use_gpu = False

    # Conversation history (memory)
    memory = [
        "User: Tell me about Thompson Sampling",
        "AI: Thompson Sampling is a Bayesian approach to the exploration-exploitation dilemma",
        "User: How does it compare to UCB?",
        "AI: While UCB uses confidence bounds, Thompson Sampling samples from posterior distributions",
        "User: What are the regret bounds?",
        "AI: Thompson Sampling achieves logarithmic regret for many bandit problems",
        "User: Can it be used for contextual bandits?",
        "AI: Yes, it extends naturally to contextual settings with posterior updates",
    ]

    logger.info(f"\nConversation history: {len(memory)} messages")

    # New user query
    user_query = "How would I implement this in Python?"
    logger.info(f"\nUser: {user_query}")

    # Initialize warp space
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    if use_gpu:
        warp = GPUWarpSpace(embedder, use_gpu=True)
        logger.info("\n1. Using GPU-accelerated Warp Space")
    else:
        warp = WarpSpace(embedder, scales=[96, 192, 384])
        logger.info("\n1. Using standard Warp Space")

    # Tension conversation history
    logger.info("2. Tensioning conversation history...")
    start = time.time()
    await warp.tension(memory)
    tension_time = time.time() - start
    logger.info(f"   Tension time: {tension_time*1000:.2f}ms")

    # Encode query
    query_emb = embedder.encode_scales([user_query])[384][0]

    # Compute attention
    logger.info("3. Computing context-aware attention...")
    start = time.time()

    if use_gpu:
        attention = warp.compute_attention(query_emb, temperature=0.8)
        context = warp.weighted_context(attention)
    else:
        attention = warp.apply_attention(query_emb)
        context = warp.weighted_context(attention)

    attention_time = time.time() - start
    logger.info(f"   Attention time: {attention_time*1000:.2f}ms")

    # Get top context
    if use_gpu:
        attention_np = attention.cpu().numpy()
    else:
        attention_np = attention

    top_indices = np.argsort(attention_np)[::-1][:3]

    logger.info("\n4. Most relevant context:")
    for i, idx in enumerate(top_indices):
        logger.info(f"   {i+1}. [{attention_np[idx]:.3f}] {memory[idx]}")

    # Simulate response generation
    logger.info("\n5. Generating response with context...")
    response = "Here's a Python implementation of Thompson Sampling: [code based on context]"
    logger.info(f"   AI: {response}")

    # Performance stats
    logger.info("\n6. Performance:")
    logger.info(f"   Total latency: {(tension_time + attention_time)*1000:.2f}ms")
    logger.info(f"   Memory size: {len(memory)} messages")
    if use_gpu:
        logger.info(f"   Device: {warp.device}")

    logger.info("\nDEMO 3 COMPLETE!")


# ============================================================================
# Demo 4: Knowledge Graph via Tensor Decomposition
# ============================================================================

async def demo_4_knowledge_graph():
    """
    Demonstrate knowledge graph exploration via tensor decomposition.

    Use case: Extract latent patterns from multi-relational knowledge graphs.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: Knowledge Graph via Tensor Decomposition")
    logger.info("="*80)

    # Simulate knowledge graph as 3D tensor
    # Dimensions: (entities, entities, relations)
    n_entities = 10
    n_relations = 4

    # Relations: "similar_to", "part_of", "requires", "improves"
    logger.info(f"\nKnowledge graph: {n_entities} entities, {n_relations} relations")

    # Create knowledge tensor (random for demo)
    knowledge_tensor = np.random.rand(n_entities, n_entities, n_relations)

    # Make symmetric for "similar_to" relation
    knowledge_tensor[:, :, 0] = (knowledge_tensor[:, :, 0] + knowledge_tensor[:, :, 0].T) / 2

    logger.info("\n1. Performing Tucker decomposition...")

    # Tucker decomposition
    core, factors = TensorDecomposer.tucker_decomposition(
        knowledge_tensor,
        ranks=[5, 5, 3]
    )

    logger.info(f"   Original: {knowledge_tensor.shape}")
    logger.info(f"   Core: {core.shape}")
    logger.info(f"   Factors: {[f.shape for f in factors]}")

    # Analyze latent patterns
    logger.info("\n2. Analyzing latent patterns...")

    # Entity embeddings from first factor
    entity_embeddings = factors[0]  # (n_entities, latent_dim)

    logger.info(f"   Entity embeddings: {entity_embeddings.shape}")

    # Find entity clusters via similarity
    similarities = entity_embeddings @ entity_embeddings.T
    logger.info(f"   Entity similarity matrix computed")

    # Relation patterns from third factor
    relation_patterns = factors[2]  # (n_relations, latent_dim)
    logger.info(f"   Relation patterns: {relation_patterns.shape}")

    # Compression ratio
    original_size = knowledge_tensor.size
    compressed_size = core.size + sum(f.size for f in factors)
    compression = 1 - (compressed_size / original_size)

    logger.info(f"\n3. Compression:")
    logger.info(f"   Original: {original_size} parameters")
    logger.info(f"   Compressed: {compressed_size} parameters")
    logger.info(f"   Reduction: {compression*100:.1f}%")

    # CP decomposition for rank-based analysis
    logger.info("\n4. CP decomposition (rank=3)...")
    cp_factors = TensorDecomposer.cp_decomposition(knowledge_tensor, rank=3, max_iter=50)

    logger.info(f"   CP factors: {[f.shape for f in cp_factors]}")
    logger.info(f"   Each rank captures a latent interaction pattern")

    logger.info("\nDEMO 4 COMPLETE!")


# ============================================================================
# Demo 5: Adaptive Learning with Fisher Information
# ============================================================================

async def demo_5_adaptive_learning():
    """
    Demonstrate adaptive learning with Fisher information geometry.

    Use case: Natural gradient descent for efficient policy optimization.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 5: Adaptive Learning with Fisher Information Geometry")
    logger.info("="*80)

    # Simulate a policy that outputs action probabilities
    n_actions = 5
    n_params = 10

    logger.info(f"\nPolicy: {n_params} parameters → {n_actions} actions")

    # Current policy parameters
    params = np.random.randn(n_params)

    # Compute policy distribution (softmax over linear combination)
    def compute_policy(params):
        # Simplified: linear layer + softmax
        weights = params.reshape(n_params // n_actions, n_actions)
        logits = np.sum(weights, axis=0)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    distribution = compute_policy(params)
    logger.info(f"\nCurrent policy distribution: {[f'{p:.3f}' for p in distribution]}")

    # Compute parameter gradients (for Fisher matrix)
    logger.info("\n1. Computing Fisher information matrix...")

    # Approximate gradients via finite differences
    eps = 1e-4
    param_gradients = []

    for i in range(min(3, n_params)):  # Use subset for demo
        params_perturbed = params.copy()
        params_perturbed[i] += eps
        dist_perturbed = compute_policy(params_perturbed)
        gradient = (dist_perturbed - distribution) / eps
        param_gradients.append(gradient)

    # Compute Fisher information matrix
    fim = FisherInformationGeometry.fisher_information_matrix(
        distribution,
        param_gradients
    )

    logger.info(f"   Fisher matrix shape: {fim.shape}")
    logger.info(f"   Condition number: {np.linalg.cond(fim):.2f}")

    # Simulate loss gradient
    loss_gradient = np.random.randn(len(param_gradients))
    logger.info(f"\n2. Loss gradient: {[f'{g:.3f}' for g in loss_gradient]}")

    # Compute natural gradient
    logger.info("\n3. Computing natural gradient...")
    natural_gradient = FisherInformationGeometry.natural_gradient(
        loss_gradient,
        fim,
        damping=1e-3
    )

    logger.info(f"   Natural gradient: {[f'{g:.3f}' for g in natural_gradient]}")

    # Compare step sizes
    standard_step = 0.1 * loss_gradient
    natural_step = 0.1 * natural_gradient

    logger.info(f"\n4. Gradient comparison:")
    logger.info(f"   Standard gradient norm: {np.linalg.norm(standard_step):.4f}")
    logger.info(f"   Natural gradient norm:  {np.linalg.norm(natural_step):.4f}")

    # Geometric interpretation
    logger.info(f"\n5. Geometric insight:")
    logger.info(f"   Natural gradient follows steepest descent on statistical manifold")
    logger.info(f"   Adapts step size based on parameter space curvature")
    logger.info(f"   Converges faster in ill-conditioned parameter spaces")

    logger.info("\nDEMO 5 COMPLETE!")


# ============================================================================
# Demo 6: Full Weaving Cycle with Advanced Features
# ============================================================================

async def demo_6_full_weaving():
    """
    Demonstrate complete weaving cycle using all advanced features.

    Use case: Production-ready query processing with the full stack.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 6: Full Weaving Cycle with Advanced Warp Features")
    logger.info("="*80)

    query_text = "How can I use Thompson Sampling for A/B testing?"

    logger.info(f"\nQuery: '{query_text}'")

    # Knowledge base
    knowledge = [
        "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
        "A/B testing compares two versions to determine which performs better",
        "Bayesian methods update beliefs based on observed data",
        "Multi-armed bandits balance exploration and exploitation",
        "Thompson Sampling samples from posterior distributions of success rates",
        "A/B tests can use Thompson Sampling for adaptive allocation",
        "Traditional A/B testing uses fixed allocation between variants",
        "Bayesian A/B testing provides probabilistic conclusions",
        "Thompson Sampling minimizes regret in sequential decision making",
        "Contextual bandits extend basic bandits with context features"
    ]

    # 1. Initialize components
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])
    manifold = RiemannianManifold(dim=384, curvature=0.3)

    logger.info("\n1. Tensioning knowledge base...")
    start_time = time.time()
    await warp.tension(knowledge)
    stage1_time = time.time() - start_time

    # 2. Query embedding with manifold
    logger.info("2. Computing query embedding on curved manifold...")
    query_emb = embedder.encode_scales([query_text])[384][0]

    # 3. Geodesic attention
    logger.info("3. Computing geodesic-aware attention...")
    start = time.time()

    # Standard attention
    attention_std = warp.apply_attention(query_emb)

    # Adjust attention based on geodesic distances
    geodesic_weights = []
    for thread in warp.threads:
        dist = manifold.geodesic_distance(query_emb, thread.embedding)
        # Convert distance to similarity weight
        weight = np.exp(-dist / 2.0)  # Gaussian kernel
        geodesic_weights.append(weight)

    geodesic_weights = np.array(geodesic_weights)
    geodesic_weights = geodesic_weights / (np.sum(geodesic_weights) + 1e-10)

    # Blend standard and geodesic attention
    attention_blended = 0.7 * attention_std + 0.3 * geodesic_weights

    stage2_time = time.time() - start

    # 4. Create superposition of top contexts
    logger.info("4. Creating quantum superposition of top contexts...")
    top_k = 3
    top_indices = np.argsort(attention_blended)[::-1][:top_k]

    context_states = [warp.threads[i].embedding for i in top_indices]
    context_amplitudes = attention_blended[top_indices]
    context_amplitudes = context_amplitudes / np.linalg.norm(context_amplitudes)

    superposed_context = QuantumWarpOperations.superposition(
        context_states,
        context_amplitudes
    )

    # 5. Decision engine with convergence
    logger.info("5. Collapsing to tool decision...")
    tools = ["answer", "search", "clarify", "synthesize"]

    # Simulate tool embeddings
    tool_embeddings = [np.random.randn(384) for _ in tools]

    # Measure superposed context against tool basis
    measured_tool_idx, prob, _ = QuantumWarpOperations.measure(
        superposed_context,
        tool_embeddings,
        collapse=True
    )

    selected_tool = tools[measured_tool_idx]

    # 6. Generate response
    logger.info("6. Generating response...")
    response_text = f"Based on Thompson Sampling principles, you can use Bayesian posterior sampling for adaptive A/B test allocation."

    # 7. Create Spacetime trace
    logger.info("7. Creating Spacetime fabric...")
    trace = WeavingTrace(
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_ms=(stage1_time + stage2_time) * 1000,
        stage_durations={
            'tension': stage1_time * 1000,
            'attention': stage2_time * 1000
        },
        motifs_detected=["THOMPSON", "BAYESIAN", "TESTING"],
        embedding_scales_used=[96, 192, 384],
        threads_activated=[f"thread_{i}" for i in top_indices],
        context_shards_count=top_k,
        retrieval_mode="geodesic_blended",
        policy_adapter="quantum_collapse",
        tool_selected=selected_tool,
        tool_confidence=prob
    )

    spacetime = Spacetime(
        query_text=query_text,
        response=response_text,
        tool_used=selected_tool,
        confidence=prob,
        trace=trace,
        metadata={
            "used_riemannian_geometry": True,
            "used_quantum_superposition": True,
            "manifold_curvature": manifold.curvature
        },
        sources_used=[knowledge[i] for i in top_indices]
    )

    # Display results
    logger.info("\n" + "="*60)
    logger.info("WEAVING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Response: {spacetime.response}")
    logger.info(f"Tool: {spacetime.tool_used}")
    logger.info(f"Confidence: {spacetime.confidence:.3f}")
    logger.info(f"\nTop contexts:")
    for i, idx in enumerate(top_indices):
        logger.info(f"  {i+1}. [{attention_blended[idx]:.3f}] {knowledge[idx][:60]}...")

    logger.info(f"\nPerformance:")
    logger.info(f"  Total time: {trace.duration_ms:.1f}ms")
    logger.info(f"  Threads activated: {len(trace.threads_activated)}")

    warp.collapse()

    logger.info("\nDEMO 6 COMPLETE!")


# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Run all demos."""
    logger.info("\n" + "="*80)
    logger.info("HOLOLOOM WARP DRIVE SHOWCASE")
    logger.info("Complete Integration Demonstrations")
    logger.info("="*80)

    demos = [
        ("Semantic Search with Riemannian Geometry", demo_1_semantic_search),
        ("Quantum-Inspired Decision Making", demo_2_quantum_decisions),
        ("Real-Time Chat with GPU Acceleration", demo_3_gpu_chat),
        ("Knowledge Graph via Tensor Decomposition", demo_4_knowledge_graph),
        ("Adaptive Learning with Fisher Information", demo_5_adaptive_learning),
        ("Full Weaving Cycle with Advanced Features", demo_6_full_weaving)
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(0.5)  # Brief pause between demos
        except Exception as e:
            logger.error(f"\nDemo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "="*80)
    logger.info("ALL DEMOS COMPLETE!")
    logger.info("="*80)
    logger.info("\nThe HoloLoom Warp Drive demonstrates:")
    logger.info("  - Riemannian geometry for curved semantic spaces")
    logger.info("  - Quantum-inspired superposition and measurement")
    logger.info("  - GPU acceleration for real-time performance")
    logger.info("  - Tensor decomposition for knowledge graphs")
    logger.info("  - Fisher information for adaptive learning")
    logger.info("  - Complete integration in production workflows")
    logger.info("\n")


if __name__ == "__main__":
    asyncio.run(main())
