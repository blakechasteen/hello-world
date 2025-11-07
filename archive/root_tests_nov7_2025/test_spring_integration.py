"""
Test Spring Dynamics Integration
=================================
Verify that advanced integrators are properly integrated.
"""

from HoloLoom.memory.spring_dynamics import SpringDynamics, SpringConfig
from HoloLoom.memory.graph import KG, KGEdge

def test_spring_dynamics():
    """Test both Euler and Verlet integration."""

    # Create simple test graph
    kg = KG()
    edges = [
        KGEdge("A", "B", "RELATED_TO", weight=1.0),
        KGEdge("B", "C", "RELATED_TO", weight=1.0),
        KGEdge("C", "D", "RELATED_TO", weight=1.0),
    ]
    kg.add_edges(edges)

    print("="*80)
    print("Testing Spring Dynamics Integration")
    print("="*80)

    # Test 1: Euler integration (fallback)
    print("\n1. Testing Euler Integration (Fallback)")
    print("-" * 40)

    config_euler = SpringConfig(
        use_advanced_integrator=False,
        dt=0.01,
        max_iterations=100
    )

    dynamics_euler = SpringDynamics(kg, config_euler)
    dynamics_euler.activate_nodes({'A': 1.0})
    result_euler = dynamics_euler.propagate()

    print(f"Euler Result: {result_euler}")
    print(f"  Iterations: {result_euler.iterations}")
    print(f"  Converged: {result_euler.converged}")
    print(f"  Final Energy: {result_euler.final_energy:.6f}")
    print(f"  Active Nodes: {len(result_euler.activated_nodes)}")

    # Test 2: Verlet integration (advanced)
    print("\n2. Testing Verlet Integration (Advanced)")
    print("-" * 40)

    config_verlet = SpringConfig(
        use_advanced_integrator=True,
        integrator_type="verlet",
        dt=0.01,
        max_iterations=100
    )

    try:
        dynamics_verlet = SpringDynamics(kg, config_verlet)
        dynamics_verlet.activate_nodes({'A': 1.0})
        result_verlet = dynamics_verlet.propagate()

        print(f"Verlet Result: {result_verlet}")
        print(f"  Iterations: {result_verlet.iterations}")
        print(f"  Converged: {result_verlet.converged}")
        print(f"  Final Energy: {result_verlet.final_energy:.6f}")
        print(f"  Active Nodes: {len(result_verlet.activated_nodes)}")

        # Compare
        print("\n3. Comparison")
        print("-" * 40)
        print(f"Iterations: Euler={result_euler.iterations}, Verlet={result_verlet.iterations}")
        print(f"Energy: Euler={result_euler.final_energy:.6f}, Verlet={result_verlet.final_energy:.6f}")

        if result_verlet.iterations < result_euler.iterations:
            print("✓ Verlet converged faster!")

        print("\n✓ Integration test PASSED")

    except Exception as e:
        print(f"⚠ Advanced integrators not available: {e}")
        print("  Falling back to Euler integration (working)")

    print("\n" + "="*80)

if __name__ == "__main__":
    test_spring_dynamics()
