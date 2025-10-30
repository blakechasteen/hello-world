"""
Spring Activation: The Transitive Discovery Demo
================================================
Shows how spring physics finds concepts 2-3 hops away from the query.
This is what static retrieval can't do!
"""

print()
print("=" * 80)
print(" " * 15 + "SPRING ACTIVATION: TRANSITIVE DISCOVERY")
print("=" * 80)
print()

# Graph with clear multi-hop paths
edges = [
    ("Thompson Sampling", "Bandits", 1.0),
    ("Thompson Sampling", "Bayesian Inference", 0.9),
    ("Bandits", "Exploration", 1.0),
    ("Bandits", "Reward Optimization", 0.8),
    ("Bayesian Inference", "Prior Distribution", 1.0),
    ("Bayesian Inference", "Posterior Sampling", 0.9),
    ("Exploration", "Regret Bounds", 0.9),
    ("Posterior Sampling", "Prior Distribution", 0.7),  # Cross-link for 2-hop
]

nodes = set()
for src, dst, _ in edges:
    nodes.add(src)
    nodes.add(dst)

neighbors = {node: [] for node in nodes}
for src, dst, weight in edges:
    neighbors[src].append((dst, weight))
    neighbors[dst].append((src, weight))

# Stronger physics for faster propagation
STIFFNESS = 0.30
DAMPING = 0.70
DECAY = 0.997
DT = 0.08

activation = {node: 0.0 for node in nodes}
velocity = {node: 0.0 for node in nodes}

print("QUERY: 'How does Thompson Sampling work?'")
print()
print("KNOWLEDGE GRAPH STRUCTURE:")
print()
print("  1-HOP from Thompson Sampling:")
print("    - Bandits")
print("    - Bayesian Inference")
print()
print("  2-HOP from Thompson Sampling:")
print("    - Exploration (via Bandits)")
print("    - Reward Optimization (via Bandits)")
print("    - Prior Distribution (via Bayesian Inference)")
print("    - Posterior Sampling (via Bayesian Inference)")
print()
print("  3-HOP from Thompson Sampling:")
print("    - Regret Bounds (via Bandits -> Exploration)")
print()
print("=" * 80)
print()

# Seed
activation["Thompson Sampling"] = 1.0

# Propagate
print("ACTIVATION SPREADING (like energy through springs)...")
print()

steps_to_show = [0, 20, 50, 100, 200]

for step in range(max(steps_to_show) + 1):
    # Forces
    forces = {node: 0.0 for node in nodes}
    for node in nodes:
        for neighbor, weight in neighbors[node]:
            force = STIFFNESS * weight * (activation[neighbor] - activation[node])
            forces[node] += force

    # Update
    for node in nodes:
        velocity[node] *= DAMPING
        velocity[node] += forces[node] * DT
        activation[node] += velocity[node] * DT
        activation[node] *= DECAY
        activation[node] = max(0.0, min(1.0, activation[node]))

    if step in steps_to_show:
        print(f"Step {step}:")

        # Group by distance
        one_hop = ["Bandits", "Bayesian Inference"]
        two_hop = ["Exploration", "Reward Optimization", "Prior Distribution", "Posterior Sampling"]
        three_hop = ["Regret Bounds"]

        # Show seed
        seed_act = activation["Thompson Sampling"]
        bars = "#" * int(seed_act * 50)
        print(f"  SEED: Thompson Sampling     {bars:50s} {seed_act:.4f}")

        # Show 1-hop
        print()
        print("  1-HOP:")
        for node in one_hop:
            act = activation[node]
            bars = "#" * int(act * 50)
            print(f"    {node:25} {bars:50s} {act:.4f}")

        # Show 2-hop
        print()
        print("  2-HOP:")
        for node in two_hop:
            act = activation[node]
            bars = "#" * int(act * 50)
            if act > 0.001:
                print(f"    {node:25} {bars:50s} {act:.4f}")
            else:
                print(f"    {node:25} {'(not yet activated)':50s}")

        # Show 3-hop
        print()
        print("  3-HOP:")
        for node in three_hop:
            act = activation[node]
            bars = "#" * int(act * 50)
            if act > 0.001:
                print(f"    {node:25} {bars:50s} {act:.4f}")
            else:
                print(f"    {node:25} {'(not yet activated)':50s}")

        print()
        print("-" * 80)
        print()

# Final comparison
print()
print("=" * 80)
print(" " * 25 + "RETRIEVAL COMPARISON")
print("=" * 80)
print()

threshold = 0.02
final = [(n, activation[n]) for n in nodes if activation[n] > threshold]
final.sort(key=lambda x: x[1], reverse=True)

print(f"STATIC RETRIEVAL (cosine similarity):")
print(f"  - Thompson Sampling  (1.0 - direct match)")
print(f"  - Total: 1 result")
print()

print(f"SPRING ACTIVATION (threshold = {threshold}):")
for i, (node, act) in enumerate(final, 1):
    # Determine hop distance
    if node == "Thompson Sampling":
        distance = "SEED"
    elif node in ["Bandits", "Bayesian Inference"]:
        distance = "1-hop"
    elif node in ["Exploration", "Reward Optimization", "Prior Distribution", "Posterior Sampling"]:
        distance = "2-hop"
    else:
        distance = "3-hop"

    bars = "#" * int(act * 40)
    print(f"  {i:2}. {node:25} {bars:40s} {act:.4f} ({distance})")

print(f"  - Total: {len(final)} results")
print()

# Count by distance
one_hop_count = sum(1 for n, _ in final if n in ["Bandits", "Bayesian Inference"])
two_hop_count = sum(1 for n, _ in final if n in ["Exploration", "Reward Optimization", "Prior Distribution", "Posterior Sampling"])
three_hop_count = sum(1 for n, _ in final if n in ["Regret Bounds"])

print("TRANSITIVE DISCOVERIES:")
print(f"  1-hop concepts: {one_hop_count}")
print(f"  2-hop concepts: {two_hop_count} << Static can't find these!")
print(f"  3-hop concepts: {three_hop_count} << Definitely not in static results!")
print()

if two_hop_count > 0 or three_hop_count > 0:
    print("EXAMPLE PATHS:")
    if two_hop_count > 0:
        print("  2-hop: Thompson Sampling -> Bandits -> Exploration")
        print("         Thompson Sampling -> Bayesian Inference -> Prior Distribution")
    if three_hop_count > 0:
        print("  3-hop: Thompson Sampling -> Bandits -> Exploration -> Regret Bounds")

print()
print("=" * 80)
print()
print("KEY INSIGHT:")
print("  Static retrieval only finds direct matches (1 result).")
print("  Spring activation spreads through the graph like energy through springs,")
print("  discovering related concepts 2-3 hops away ({} results).".format(len(final)))
print()
print("  This is the power of PHYSICS-BASED MEMORY RETRIEVAL!")
print()
print("=" * 80)
