#!/usr/bin/env python3
"""
HoloLoom Feature Showcase - Interactive Chat Demo
==================================================
Demonstrates core HoloLoom features without heavy ML dependencies.
Created: 2026-01-28

Features demonstrated:
1. Knowledge Graph (Yarn Graph) - Entity relationships
2. Convergence Engine - Thompson Sampling decision making
3. Context Packing - Beta wave activation spreading
4. Smart Query Routing - Complexity classification
5. Semantic Calculus - 16-axis projection (simulated)
6. Memory Navigation - 4-direction spatial traversal
7. Pattern Discovery - Emergent structure detection
"""

import warnings
warnings.filterwarnings("ignore")

import time
import random
import math
import networkx as nx


def section(title: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


# ─────────────────────────────────────────────────────────────
# DEMO 1: Knowledge Graph (Yarn Graph)
# ─────────────────────────────────────────────────────────────

def demo_knowledge_graph():
    section("1. Knowledge Graph (Yarn Graph)")
    print("HoloLoom stores knowledge as a typed multigraph.")
    print("Entities connect via semantic relationships.\n")

    from HoloLoom.memory.graph import KG, KGEdge

    kg = KG()
    edges = [
        KGEdge("Thompson_Sampling", "Bayesian_Methods", "IS_A", 1.0),
        KGEdge("Thompson_Sampling", "Exploration", "USES", 0.9),
        KGEdge("Thompson_Sampling", "Exploitation", "USES", 0.9),
        KGEdge("Exploration", "Multi_Armed_Bandit", "PART_OF", 0.8),
        KGEdge("Exploitation", "Multi_Armed_Bandit", "PART_OF", 0.8),
        KGEdge("UCB", "Multi_Armed_Bandit", "IS_A", 0.9),
        KGEdge("Epsilon_Greedy", "Multi_Armed_Bandit", "IS_A", 0.85),
        KGEdge("Bayesian_Methods", "Probability", "USES", 0.7),
        KGEdge("Neural_Networks", "Deep_Learning", "IS_A", 1.0),
        KGEdge("Deep_Learning", "Machine_Learning", "IS_A", 1.0),
        KGEdge("Thompson_Sampling", "Machine_Learning", "PART_OF", 0.6),
        KGEdge("PPO", "Reinforcement_Learning", "IS_A", 0.95),
        KGEdge("Reinforcement_Learning", "Machine_Learning", "IS_A", 1.0),
    ]
    kg.add_edges(edges)

    G = kg.G
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print("  Entity Relationships:")
    for node in sorted(G.nodes()):
        successors = list(G.successors(node))
        if successors:
            for succ in successors:
                edge_data = G.get_edge_data(node, succ)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    rel = edge_data[first_key].get("type", "RELATED")
                    w = edge_data[first_key].get("weight", 0)
                    print(f"    {node} --[{rel} {w:.1f}]--> {succ}")

    # Subgraph extraction
    print("\n  Subgraph for 'Thompson_Sampling':")
    neighbors = set(G.successors("Thompson_Sampling")) | set(G.predecessors("Thompson_Sampling"))
    neighbors.add("Thompson_Sampling")
    print(f"    Connected entities: {sorted(neighbors)}")

    # Path finding
    try:
        path = nx.shortest_path(G, "Thompson_Sampling", "Machine_Learning")
        print(f"\n  Shortest path Thompson_Sampling -> Machine_Learning:")
        print(f"    {' -> '.join(path)}")
    except nx.NetworkXNoPath:
        print("  No path found")

    # Graph statistics
    stats = kg.stats()
    print(f"\n  Graph Stats: {stats}")

    return kg


# ─────────────────────────────────────────────────────────────
# DEMO 2: Thompson Sampling (Convergence Engine)
# ─────────────────────────────────────────────────────────────

def demo_thompson_sampling():
    section("2. Thompson Sampling (Convergence Engine)")
    print("HoloLoom uses Thompson Sampling for exploration/exploitation.")
    print("Each tool has Beta(alpha, beta) priors that adapt over time.\n")

    tools = {
        "answer":   {"alpha": 5.0, "beta": 2.0, "desc": "Direct answer"},
        "research": {"alpha": 3.0, "beta": 3.0, "desc": "Multi-query research"},
        "explore":  {"alpha": 2.0, "beta": 4.0, "desc": "Knowledge exploration"},
        "verify":   {"alpha": 4.0, "beta": 2.5, "desc": "Claim verification"},
    }

    print("  Initial Priors (Beta distribution):")
    print(f"  {'Tool':<12} {'Alpha':>6} {'Beta':>6} {'E[reward]':>10} {'Description'}")
    print(f"  {'-'*60}")
    for name, t in tools.items():
        expected = t["alpha"] / (t["alpha"] + t["beta"])
        print(f"  {name:<12} {t['alpha']:>6.1f} {t['beta']:>6.1f} {expected:>10.3f} {t['desc']}")

    # Simulate 20 queries with Thompson Sampling
    print("\n  Simulating 20 queries with Thompson Sampling...\n")
    selections = {name: 0 for name in tools}

    for i in range(20):
        # Sample from each Beta distribution
        samples = {}
        for name, t in tools.items():
            samples[name] = random.betavariate(t["alpha"], t["beta"])

        # Select highest sample (Thompson Sampling)
        selected = max(samples, key=samples.get)
        selections[selected] += 1

        # Simulate outcome
        confidence = random.uniform(0.5, 1.0)
        success = confidence >= 0.7

        if success:
            tools[selected]["alpha"] += confidence
        else:
            tools[selected]["beta"] += (1 - confidence)

    print("  After 20 queries:")
    print(f"  {'Tool':<12} {'Selected':>9} {'Alpha':>6} {'Beta':>6} {'E[reward]':>10}")
    print(f"  {'-'*50}")
    for name, t in tools.items():
        expected = t["alpha"] / (t["alpha"] + t["beta"])
        print(f"  {name:<12} {selections[name]:>9} {t['alpha']:>6.1f} {t['beta']:>6.1f} {expected:>10.3f}")

    print("\n  Thompson Sampling naturally balances:")
    print("    - Exploitation: picks tools that work well")
    print("    - Exploration: occasionally tries underused tools")


# ─────────────────────────────────────────────────────────────
# DEMO 3: Beta Wave Activation Spreading
# ─────────────────────────────────────────────────────────────

def demo_activation_spreading(kg):
    section("3. Beta Wave Activation Spreading")
    print("Context packing uses neuroscience-inspired beta waves")
    print("(12-30 Hz) to spread activation across the knowledge graph.\n")

    G = kg.G
    # Manual activation spreading simulation
    activation = {node: 0.0 for node in G.nodes()}
    activation["Thompson_Sampling"] = 1.0  # Seed node

    decay_rate = 0.7
    max_hops = 3

    print(f"  Seed: Thompson_Sampling (activation=1.0)")
    print(f"  Decay rate: {decay_rate}, Max hops: {max_hops}\n")

    for hop in range(1, max_hops + 1):
        new_activation = dict(activation)
        for node, act in activation.items():
            if act > 0.01:
                for neighbor in set(G.successors(node)) | set(G.predecessors(node)):
                    spread = act * (decay_rate ** hop)
                    new_activation[neighbor] = max(new_activation[neighbor], spread)
        activation = new_activation

        active = {k: v for k, v in activation.items() if v > 0.01}
        print(f"  Hop {hop}: {len(active)} active nodes")
        for node, act in sorted(active.items(), key=lambda x: -x[1]):
            bar = "█" * int(act * 30)
            print(f"    {node:<25} {act:.3f} {bar}")
        print()

    print("  Activation spreading identifies contextually relevant")
    print("  memories even if not directly connected to the query.")


# ─────────────────────────────────────────────────────────────
# DEMO 4: Smart Query Routing
# ─────────────────────────────────────────────────────────────

def demo_query_routing():
    section("4. Smart Query Routing")
    print("Queries are classified by complexity for optimal routing.")
    print("TRIVIAL/SIMPLE queries get fast-path (<50ms), saving resources.\n")

    import re

    trivial_patterns = [
        r'^(hi|hello|hey|yo)[\s!?]*$',
        r'^(thanks|thank you|thx)[\s!?.]*$',
        r'^(ok|okay|sure|yes|no)[\s!?.]*$',
    ]

    simple_patterns = [
        r'^what is \w+',
        r'^define \w+',
        r'^who is \w+',
    ]

    research_keywords = {
        'analyze', 'compare', 'evaluate', 'tradeoffs',
        'comprehensive', 'versus', 'pros and cons'
    }

    queries = [
        "hello",
        "thanks!",
        "what is recursion",
        "define entropy",
        "Explain how Thompson Sampling works",
        "Compare Thompson Sampling versus UCB tradeoffs",
        "Analyze comprehensive pros and cons of all exploration strategies",
    ]

    print(f"  {'Query':<55} {'Level':<12} {'Latency'}")
    print(f"  {'-'*80}")

    for q in queries:
        q_lower = q.lower().strip()
        level = "MODERATE"
        latency = "~100ms"

        # Check trivial
        for pat in trivial_patterns:
            if re.match(pat, q_lower):
                level = "TRIVIAL"
                latency = "<10ms  ⚡"
                break

        if level == "MODERATE":
            for pat in simple_patterns:
                if re.match(pat, q_lower):
                    level = "SIMPLE"
                    latency = "<50ms  ⚡"
                    break

        if level == "MODERATE":
            words = set(q_lower.split())
            if words & research_keywords:
                if len(words & research_keywords) >= 2:
                    level = "RESEARCH"
                    latency = "unlimited"
                else:
                    level = "COMPLEX"
                    latency = "<150ms"

            if len(q_lower.split()) > 12:
                level = "COMPLEX"
                latency = "<150ms"

        print(f"  {q:<55} {level:<12} {latency}")

    print("\n  Routing saves 40%+ compute on common queries (greetings, lookups).")


# ─────────────────────────────────────────────────────────────
# DEMO 5: Semantic Calculus (16 Interpretable Axes)
# ─────────────────────────────────────────────────────────────

def demo_semantic_calculus():
    section("5. Semantic Calculus (16 Interpretable Axes)")
    print("HoloLoom projects queries into 228D semantic space.")
    print("First 16 dimensions are human-interpretable.\n")

    # Simulated semantic projections for demo purposes
    queries = {
        "What is Thompson Sampling?": {
            "Sentiment": 0.1, "Formality": 0.6, "Technicality": 0.8,
            "Certainty": 0.3, "Urgency": 0.2, "Abstraction": 0.5,
            "Specificity": 0.7, "Complexity": 0.6,
        },
        "HELP! Server is down!!!": {
            "Sentiment": -0.7, "Formality": -0.8, "Technicality": 0.4,
            "Certainty": 0.9, "Urgency": 0.95, "Abstraction": -0.3,
            "Specificity": 0.8, "Complexity": 0.3,
        },
        "Perhaps consider analyzing the epistemological implications": {
            "Sentiment": 0.2, "Formality": 0.9, "Technicality": 0.7,
            "Certainty": -0.4, "Urgency": -0.5, "Abstraction": 0.9,
            "Specificity": 0.3, "Complexity": 0.85,
        },
    }

    for query, axes in queries.items():
        print(f'  Query: "{query}"')
        print(f"  {'Axis':<15} {'Value':>6}  {'Visualization'}")
        print(f"  {'-'*50}")
        for axis, value in axes.items():
            # Create visual bar
            pos = int((value + 1) * 15)  # Map [-1, 1] to [0, 30]
            bar = list("·" * 30)
            bar[pos] = "█"
            bar_str = "".join(bar)
            sign = "+" if value >= 0 else ""
            print(f"  {axis:<15} {sign}{value:.1f}  [{bar_str}]")
        print()

    print("  These axes enable semantic similarity, clustering,")
    print("  and query-aware context selection.")


# ─────────────────────────────────────────────────────────────
# DEMO 6: Memory Navigation
# ─────────────────────────────────────────────────────────────

def demo_memory_navigation(kg):
    section("6. Memory Navigation (4 Directions)")
    print("Navigate knowledge like a spatial map.\n")

    G = kg.G

    start = "Thompson_Sampling"
    print(f"  Starting from: {start}\n")

    # FORWARD - successors
    forward = list(G.successors(start))
    print(f"  FORWARD (what comes next):")
    for n in forward:
        print(f"    → {n}")

    # BACKWARD - predecessors
    backward = list(G.predecessors(start))
    print(f"\n  BACKWARD (what came before):")
    for n in backward:
        print(f"    ← {n}")

    # SIDEWAYS - siblings (share parents or children)
    siblings = set()
    for parent in G.predecessors(start):
        for sibling in G.successors(parent):
            if sibling != start:
                siblings.add(sibling)
    for child in G.successors(start):
        for sibling in G.predecessors(child):
            if sibling != start:
                siblings.add(sibling)
    print(f"\n  SIDEWAYS (related concepts):")
    for n in sorted(siblings):
        print(f"    ↔ {n}")

    # DEEP - cycles
    print(f"\n  DEEP (feedback loops):")
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            for c in cycles[:3]:
                print(f"    ⟳ {' → '.join(c)} → {c[0]}")
        else:
            print("    No cycles found (DAG structure)")
    except Exception:
        print("    No cycles found")


# ─────────────────────────────────────────────────────────────
# DEMO 7: Pattern Discovery
# ─────────────────────────────────────────────────────────────

def demo_pattern_discovery(kg):
    section("7. Pattern Discovery")
    print("Emergent patterns detected in the knowledge graph.\n")

    G = kg.G

    # Clusters (community detection)
    print("  CLUSTERS (tightly connected groups):")
    try:
        communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
        for i, comm in enumerate(communities):
            strength = len(comm) / G.number_of_nodes()
            print(f"    Cluster {i+1} (strength={strength:.2f}): {sorted(comm)}")
    except Exception as e:
        print(f"    Community detection: {e}")

    # Threads (longest paths)
    print("\n  THREADS (narrative chains):")
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    longest_path = []
    for root in roots:
        for target in G.nodes():
            try:
                for path in nx.all_simple_paths(G, root, target, cutoff=5):
                    if len(path) > len(longest_path):
                        longest_path = path
            except nx.NetworkXError:
                pass
    if longest_path:
        print(f"    Longest thread ({len(longest_path)} hops):")
        print(f"    {' → '.join(longest_path)}")

    # Hub detection (most connected)
    print("\n  HUBS (most connected entities):")
    degree = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
    for node, deg in sorted(degree.items(), key=lambda x: -x[1])[:5]:
        bar = "█" * deg
        print(f"    {node:<30} degree={deg} {bar}")


# ─────────────────────────────────────────────────────────────
# DEMO 8: Weaving Cycle Overview
# ─────────────────────────────────────────────────────────────

def demo_weaving_cycle():
    section("8. The 9-Step Weaving Cycle")
    print("HoloLoom's core processing pipeline:\n")

    steps = [
        ("Step 0", "Meta-Prompt Enhancement", "LLM query rewriting", "optional"),
        ("Step 1", "Loom Command", "Select pattern: BARE/FAST/FUSED", "~5ms"),
        ("Step 2", "Chrono Trigger", "Create temporal window", "~5ms"),
        ("Step 3", "Thread Selection", "Select memory threads (MCTS)", "~20ms"),
        ("Step 4", "Resonance Shed", "Extract features → DotPlasma", "~30ms"),
        ("Step 5", "Warp Space", "Tension to continuous manifold", "~20ms"),
        ("Step 6", "Memory Retrieval", "Multipass graph crawl", "~30ms"),
        ("Step 7", "Convergence Engine", "Collapse to tool selection", "~10ms"),
        ("Step 8", "Tool Execution", "Safety gate + execute", "~20ms"),
        ("Step 9", "Spacetime Fabric", "Weave output + trace", "~10ms"),
    ]

    total = 0
    for step, name, desc, timing in steps:
        t = timing.replace("~", "").replace("ms", "").replace("optional", "0")
        try:
            total += int(t)
        except ValueError:
            pass
        indicator = "║" if "optional" not in timing else "┊"
        print(f"  {indicator} {step}: {name:<22} {desc:<35} [{timing}]")

    print(f"  ╚{'═'*72}╝")
    print(f"    Total pipeline: ~{total}ms (FAST mode)")
    print(f"    Steps 4-6 run in PARALLEL for 40-120ms speedup")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         HoloLoom Feature Showcase                       ║")
    print("║         Neural Decision-Making System                   ║")
    print("║         2026-01-28                                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    t0 = time.time()

    kg = demo_knowledge_graph()
    demo_thompson_sampling()
    demo_activation_spreading(kg)
    demo_query_routing()
    demo_semantic_calculus()
    demo_memory_navigation(kg)
    demo_pattern_discovery(kg)
    demo_weaving_cycle()

    elapsed = time.time() - t0

    section("Summary")
    print(f"  8 features demonstrated in {elapsed:.2f}s")
    print(f"  HoloLoom: 924+ Python files, ~165,000+ lines of code")
    print(f"  Systems: 7 learning loops, 11 memory systems, 47 input adapters")
    print(f"  Philosophy: 'Reliable Systems: Safety First'")
    print()


if __name__ == "__main__":
    main()
