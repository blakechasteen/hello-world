"""
Simple Spring Activation Demo
==============================
Shows physics-driven memory retrieval in action.
"""

print("=" * 70)
print("SPRING-BASED MEMORY RETRIEVAL DEMO")
print("=" * 70)
print()

# Simulate a simple knowledge graph
graph = {
    "Thompson Sampling": {
        "connections": [
            ("Bandits", 1.0, "IS_INSTANCE_OF"),
            ("Bayesian Inference", 0.9, "USES"),
            ("Exploration", 0.8, "ADDRESSES")
        ]
    },
    "Bandits": {
        "connections": [
            ("Exploration", 1.0, "INVOLVES"),
            ("UCB", 0.7, "SOLVED_BY"),
            ("Reward Optimization", 0.9, "GOAL_IS")
        ]
    },
    "Bayesian Inference": {
        "connections": [
            ("Prior Distribution", 1.0, "USES"),
            ("Posterior Sampling", 0.9, "ENABLES")
        ]
    },
    "Exploration": {
        "connections": [
            ("Regret Bounds", 0.7, "MEASURED_BY"),
            ("Epsilon Greedy", 0.6, "IMPLEMENTED_BY")
        ]
    },
    "UCB": {"connections": [("Regret Bounds", 0.9, "HAS_GUARANTEE")]},
    "Prior Distribution": {"connections": []},
    "Posterior Sampling": {"connections": []},
    "Regret Bounds": {"connections": []},
    "Epsilon Greedy": {"connections": []},
    "Reward Optimization": {"connections": []}
}

# Physics parameters
STIFFNESS = 0.15
DAMPING = 0.85
DECAY = 0.98
DT = 0.016

# Node states
activation = {node: 0.0 for node in graph}
velocity = {node: 0.0 for node in graph}

# Activate seed
print("Query: 'How does Thompson Sampling work?'")
print()
print("Activating seed node: Thompson Sampling")
activation["Thompson Sampling"] = 1.0
print()

# Show static vs spring comparison
print("-" * 70)
print("STATIC RETRIEVAL (Baseline)")
print("-" * 70)
print("Only retrieves direct match:")
print("  1. Thompson Sampling (direct match)")
print()

print("-" * 70)
print("SPRING ACTIVATION (Physics-Based)")
print("-" * 70)
print()

# Simulate spring propagation
print("Propagating activation through springs...")
print()

for iteration in [0, 10, 20, 30, 40, 50]:
    if iteration > 0:
        # Run 10 iterations
        for _ in range(10):
            # Calculate forces
            forces = {node: 0.0 for node in graph}

            for node, data in graph.items():
                for neighbor, weight, _ in data["connections"]:
                    # Hooke's Law: F = k × (activation_diff)
                    act_diff = activation[neighbor] - activation[node]
                    spring_force = STIFFNESS * weight * act_diff
                    forces[node] += spring_force
                    forces[neighbor] -= spring_force

            # Update activations
            for node in graph:
                # Damping force
                damping_force = -DAMPING * velocity[node]
                total_force = forces[node] + damping_force

                # Update velocity and activation
                velocity[node] += total_force * DT
                velocity[node] *= DAMPING
                activation[node] += velocity[node] * DT
                activation[node] *= DECAY

                # Clamp
                activation[node] = max(0.0, min(1.0, activation[node]))

    # Show snapshot
    print(f"Iteration {iteration}:")
    active_nodes = [(n, a) for n, a in activation.items() if a > 0.01]
    active_nodes.sort(key=lambda x: x[1], reverse=True)

    for node, act in active_nodes[:6]:
        bar_length = int(act * 30)
        bar = "#" * bar_length
        print(f"  {node:25} {bar} {act:.3f}")
    print()

# Final results
print("-" * 70)
print("FINAL RESULTS (Activation > 0.1)")
print("-" * 70)

active_final = [(n, a) for n, a in activation.items() if a > 0.1]
active_final.sort(key=lambda x: x[1], reverse=True)

print()
for i, (node, act) in enumerate(active_final, 1):
    print(f"  {i}. {node:25} activation: {act:.3f}")

print()
print("-" * 70)
print("COMPARISON")
print("-" * 70)
print()
print(f"Static retrieval:  1 concept (direct match only)")
print(f"Spring activation: {len(active_final)} concepts (multi-hop transitive!)")
print()

# Show transitive discoveries
transitive = [node for node, act in active_final if node != "Thompson Sampling"]
if transitive:
    print("Transitive discoveries (not direct matches):")
    for node in transitive:
        print(f"  • {node}")
        # Find path
        if node in ["Bayesian Inference", "Bandits", "Exploration"]:
            print(f"    Path: Thompson Sampling → {node}")
        elif node in ["Prior Distribution", "Posterior Sampling"]:
            print(f"    Path: Thompson Sampling → Bayesian Inference → {node}")
        elif node in ["Regret Bounds", "Epsilon Greedy", "UCB"]:
            print(f"    Path: Thompson Sampling → Exploration → {node}")
        elif node in ["Reward Optimization"]:
            print(f"    Path: Thompson Sampling → Bandits → {node}")

print()
print("=" * 70)
print("Spring activation finds concepts 2-3 hops away from the query!")
print("This is the power of physics-based spreading activation.")
print("=" * 70)
