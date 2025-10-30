"""
Visual Spring Activation Demo
==============================
Shows how activation spreads through the graph like energy through springs.
"""

print("=" * 70)
print("SPRING-BASED MEMORY RETRIEVAL DEMO")
print("=" * 70)
print()

# Simple graph representation
edges = [
    ("Thompson Sampling", "Bandits", 1.0),
    ("Thompson Sampling", "Bayesian Inference", 0.9),
    ("Thompson Sampling", "Exploration", 0.8),
    ("Bandits", "Exploration", 1.0),
    ("Bandits", "Reward Optimization", 0.9),
    ("Bayesian Inference", "Prior Distribution", 1.0),
    ("Bayesian Inference", "Posterior Sampling", 0.9),
    ("Exploration", "Regret Bounds", 0.7),
]

# All nodes
nodes = set()
for src, dst, _ in edges:
    nodes.add(src)
    nodes.add(dst)

# Build adjacency
neighbors = {node: [] for node in nodes}
for src, dst, weight in edges:
    neighbors[src].append((dst, weight))
    neighbors[dst].append((src, weight))  # Bidirectional

# Physics params (tuned for visibility)
STIFFNESS = 0.25  # Higher stiffness
DAMPING = 0.75    # Lower damping (more oscillation)
DECAY = 0.995     # Much slower decay
DT = 0.05         # Larger timestep

# State
activation = {node: 0.0 for node in nodes}
velocity = {node: 0.0 for node in nodes}

print("Query: 'How does Thompson Sampling work?'")
print()
print("Knowledge Graph:")
print("  Thompson Sampling")
print("    |-- Bandits (1.0)")
print("    |-- Bayesian Inference (0.9)")
print("    |-- Exploration (0.8)")
print("  Bandits")
print("    |-- Exploration (1.0)")
print("    |-- Reward Optimization (0.9)")
print("  Bayesian Inference")
print("    |-- Prior Distribution (1.0)")
print("    |-- Posterior Sampling (0.9)")
print("  Exploration")
print("    |-- Regret Bounds (0.7)")
print()

# Activate seed
print("=" * 70)
print("SPRING PROPAGATION SIMULATION")
print("=" * 70)
print()
print("Activating seed: Thompson Sampling = 1.0")
print()

activation["Thompson Sampling"] = 1.0

# Show snapshots
snapshots = [0, 10, 30, 60, 100, 150]

for iteration in range(max(snapshots) + 1):
    # Compute forces
    forces = {node: 0.0 for node in nodes}

    for node in nodes:
        for neighbor, weight in neighbors[node]:
            # Spring force: F = k * weight * (neighbor_activation - my_activation)
            act_diff = activation[neighbor] - activation[node]
            force = STIFFNESS * weight * act_diff
            forces[node] += force

    # Update states
    for node in nodes:
        # Apply damping to velocity
        velocity[node] *= DAMPING

        # Add force (F = ma, assuming m=1)
        velocity[node] += forces[node] * DT

        # Update activation
        activation[node] += velocity[node] * DT

        # Apply decay
        activation[node] *= DECAY

        # Clamp
        activation[node] = max(0.0, min(1.0, activation[node]))

    # Show snapshot
    if iteration in snapshots:
        print(f"Step {iteration}:")
        active = [(n, activation[n]) for n in nodes if activation[n] > 0.01]
        active.sort(key=lambda x: x[1], reverse=True)

        for node, act in active:
            bars = int(act * 40)
            bar = "#" * bars
            print(f"  {node:30} {bar} {act:.3f}")
        print()

# Final results
print("=" * 70)
print("RETRIEVAL COMPARISON")
print("=" * 70)
print()

threshold = 0.05  # Lower threshold to show more transitive discoveries
final = [(n, activation[n]) for n in nodes if activation[n] > threshold]
final.sort(key=lambda x: x[1], reverse=True)

print("STATIC RETRIEVAL:")
print("  1. Thompson Sampling (direct query match)")
print()

print(f"SPRING ACTIVATION (threshold = {threshold}):")
for i, (node, act) in enumerate(final, 1):
    marker = " <- SEED" if node == "Thompson Sampling" else ""
    print(f"  {i}. {node:30} activation: {act:.3f}{marker}")

print()
print(f"Static:  1 result")
print(f"Spring:  {len(final)} results")
print()

transitive = [n for n, a in final if n != "Thompson Sampling"]
if transitive:
    print("TRANSITIVE DISCOVERIES (multi-hop):")
    for node in transitive:
        # Find shortest path
        if node in ["Bandits", "Bayesian Inference", "Exploration"]:
            print(f"  {node}")
            print(f"    1-hop: Thompson Sampling -> {node}")
        elif node in ["Prior Distribution", "Posterior Sampling"]:
            print(f"  {node}")
            print(f"    2-hop: Thompson Sampling -> Bayesian Inference -> {node}")
        elif node == "Regret Bounds":
            print(f"  {node}")
            print(f"    2-hop: Thompson Sampling -> Exploration -> {node}")
        elif node == "Reward Optimization":
            print(f"  {node}")
            print(f"    2-hop: Thompson Sampling -> Bandits -> {node}")

print()
print("=" * 70)
print("Key Insight: Spring activation finds concepts 2+ hops away!")
print("Static similarity can't do this - it only sees direct matches.")
print("=" * 70)
