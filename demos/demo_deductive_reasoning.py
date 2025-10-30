"""
Deductive Reasoning Demo

Demonstrates logical inference through classic examples:
1. Socrates Syllogism (All humans are mortal)
2. Family Relationships (parent/grandparent/ancestor)
3. Forward Chaining (derive all consequences)
4. Backward Chaining (prove specific goals)
5. Proof Generation (explain reasoning)

Shows the power of deductive logic for knowledge representation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.reasoning.deductive import (
    Fact, Rule, KnowledgeBase, DeductiveReasoner, create_fact, create_rule
)

# ============================================================================
# Example 1: Classic Socrates Syllogism
# ============================================================================

def demo_socrates():
    """The classic example of deductive reasoning."""

    print("=" * 80)
    print("EXAMPLE 1: SOCRATES SYLLOGISM".center(80))
    print("=" * 80)
    print()

    print("Knowledge Base:")
    print("-" * 80)

    # Create KB
    kb = KnowledgeBase()

    # Facts
    socrates = create_fact("human", "Socrates")
    plato = create_fact("human", "Plato")
    aristotle = create_fact("human", "Aristotle")

    kb.add_facts([socrates, plato, aristotle])

    print("Facts:")
    print(f"  {socrates}")
    print(f"  {plato}")
    print(f"  {aristotle}")
    print()

    # Rules
    mortality_rule = create_rule(
        premises=[create_fact("human", "?x")],
        conclusion=create_fact("mortal", "?x"),
        name="mortality"
    )
    kb.add_rule(mortality_rule)

    print("Rules:")
    print(f"  {mortality_rule}")
    print()

    # Reasoning
    print("Reasoning:")
    print("-" * 80)

    reasoner = DeductiveReasoner(kb)

    # Forward chaining
    print("Forward Chaining (derive all consequences):")
    derived = reasoner.forward_chain()
    print(f"  Derived {len(derived)} total facts:")
    for fact in sorted(derived, key=str):
        print(f"    {fact}")
    print()

    # Backward chaining
    print("Backward Chaining (prove specific goals):")
    goal = create_fact("mortal", "Socrates")
    print(f"  Goal: {goal}")
    print()

    proof = reasoner.backward_chain(goal)
    if proof:
        print("  ✓ Goal proven!")
        print()
        print(proof.to_string(indent=1))
    else:
        print("  ✗ Cannot prove goal")

    print()


# ============================================================================
# Example 2: Family Relationships
# ============================================================================

def demo_family():
    """More complex reasoning with transitive relationships."""

    print("=" * 80)
    print("EXAMPLE 2: FAMILY RELATIONSHIPS".center(80))
    print("=" * 80)
    print()

    print("Knowledge Base:")
    print("-" * 80)

    # Create KB
    kb = KnowledgeBase()

    # Facts: family tree
    #   Alice
    #     ├─ Bob
    #     │   ├─ Dave
    #     │   └─ Eve
    #     └─ Carol
    #         └─ Frank

    facts = [
        create_fact("parent", "Alice", "Bob"),
        create_fact("parent", "Alice", "Carol"),
        create_fact("parent", "Bob", "Dave"),
        create_fact("parent", "Bob", "Eve"),
        create_fact("parent", "Carol", "Frank"),
    ]
    kb.add_facts(facts)

    print("Facts (parent relationships):")
    for fact in facts:
        print(f"  {fact}")
    print()

    # Rules
    grandparent_rule = create_rule(
        premises=[
            create_fact("parent", "?x", "?y"),
            create_fact("parent", "?y", "?z")
        ],
        conclusion=create_fact("grandparent", "?x", "?z"),
        name="grandparent"
    )

    ancestor_base = create_rule(
        premises=[create_fact("parent", "?x", "?y")],
        conclusion=create_fact("ancestor", "?x", "?y"),
        name="ancestor_base"
    )

    ancestor_transitive = create_rule(
        premises=[
            create_fact("parent", "?x", "?y"),
            create_fact("ancestor", "?y", "?z")
        ],
        conclusion=create_fact("ancestor", "?x", "?z"),
        name="ancestor_transitive"
    )

    kb.add_rules([grandparent_rule, ancestor_base, ancestor_transitive])

    print("Rules:")
    print(f"  {grandparent_rule}")
    print(f"  {ancestor_base}")
    print(f"  {ancestor_transitive}")
    print()

    # Reasoning
    print("Reasoning:")
    print("-" * 80)

    reasoner = DeductiveReasoner(kb)

    # Forward chaining
    print("Forward Chaining (derive all relationships):")
    derived = reasoner.forward_chain(max_iterations=10)

    # Show grandparents
    grandparents = [f for f in derived if f.predicate == "grandparent"]
    print(f"\n  Grandparent relationships:")
    for gp in sorted(grandparents, key=str):
        print(f"    {gp}")

    # Show ancestors
    ancestors = [f for f in derived if f.predicate == "ancestor"]
    print(f"\n  Ancestor relationships:")
    for anc in sorted(ancestors, key=str):
        print(f"    {anc}")

    print()

    # Backward chaining
    print("Backward Chaining (prove specific relationships):")
    goals = [
        create_fact("grandparent", "Alice", "Dave"),
        create_fact("ancestor", "Alice", "Frank"),
        create_fact("ancestor", "Alice", "Eve"),
    ]

    for goal in goals:
        print(f"\n  Goal: {goal}")
        proof = reasoner.backward_chain(goal)
        if proof:
            print("    ✓ Proven!")
            print(f"    Proof depth: {proof.depth}")
            print(f"    Rules used: {len(proof.rules_applied)}")
        else:
            print("    ✗ Cannot prove")

    print()


# ============================================================================
# Example 3: Logic Puzzle
# ============================================================================

def demo_logic_puzzle():
    """Solve a logic puzzle using deduction."""

    print("=" * 80)
    print("EXAMPLE 3: LOGIC PUZZLE".center(80))
    print("=" * 80)
    print()

    print("Puzzle: Who is guilty?")
    print("-" * 80)
    print("""
    Facts:
    - If someone is at crime scene AND has motive, they are suspect
    - If someone is suspect AND no alibi, they are guilty
    - Alice was at crime scene
    - Alice has motive
    - Bob has motive
    - Alice has no alibi
    - Bob has alibi

    Question: Who is guilty?
    """)

    # Create KB
    kb = KnowledgeBase()

    # Facts
    facts = [
        create_fact("at_scene", "Alice"),
        create_fact("has_motive", "Alice"),
        create_fact("has_motive", "Bob"),
        create_fact("no_alibi", "Alice"),
        create_fact("has_alibi", "Bob"),
    ]
    kb.add_facts(facts)

    # Rules
    suspect_rule = create_rule(
        premises=[
            create_fact("at_scene", "?x"),
            create_fact("has_motive", "?x")
        ],
        conclusion=create_fact("suspect", "?x"),
        name="suspect_rule"
    )

    guilty_rule = create_rule(
        premises=[
            create_fact("suspect", "?x"),
            create_fact("no_alibi", "?x")
        ],
        conclusion=create_fact("guilty", "?x"),
        name="guilty_rule"
    )

    kb.add_rules([suspect_rule, guilty_rule])

    # Solve
    print("Solving:")
    print("-" * 80)

    reasoner = DeductiveReasoner(kb)

    # Forward chain to derive all conclusions
    derived = reasoner.forward_chain()

    # Check who is guilty
    guilty_facts = [f for f in derived if f.predicate == "guilty"]

    if guilty_facts:
        print("✓ Solution found!")
        print()
        for guilty in guilty_facts:
            person = guilty.arguments[0]
            print(f"  {person} is GUILTY")
            print()

            # Explain reasoning
            proof = reasoner.explain(guilty)
            if proof:
                print("  Reasoning:")
                print(proof.to_string(indent=2))
    else:
        print("✗ No guilty party found")

    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all deductive reasoning demos."""

    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + "DEDUCTIVE REASONING DEMONSTRATION".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run examples
    demo_socrates()
    print("\n\n")

    demo_family()
    print("\n\n")

    demo_logic_puzzle()
    print("\n\n")

    # Summary
    print("=" * 80)
    print("SUMMARY".center(80))
    print("=" * 80)
    print()

    print("What We Demonstrated:")
    print()
    print("1. ✅ Knowledge Representation")
    print("   - Facts: Atomic propositions (human(Socrates))")
    print("   - Rules: Logical implications (human(?x) → mortal(?x))")
    print("   - Variables: Pattern matching with unification")
    print()
    print("2. ✅ Forward Chaining (Data-Driven)")
    print("   - Start with facts")
    print("   - Apply rules to derive new facts")
    print("   - Repeat until fixed point")
    print("   - Derives ALL consequences")
    print()
    print("3. ✅ Backward Chaining (Goal-Driven)")
    print("   - Start with goal")
    print("   - Find rules that conclude goal")
    print("   - Recursively prove premises")
    print("   - Efficient for specific queries")
    print()
    print("4. ✅ Proof Generation")
    print("   - Complete reasoning trace")
    print("   - Shows which rules applied")
    print("   - Explains WHY conclusion holds")
    print("   - Foundation for explainable AI")
    print()
    print("5. ✅ Unification Algorithm")
    print("   - Pattern matching with variables")
    print("   - Finds consistent variable bindings")
    print("   - Enables general rules")
    print()

    print("Applications:")
    print("   - Expert systems (medical diagnosis, fault diagnosis)")
    print("   - Automated theorem proving")
    print("   - Knowledge graphs and semantic web")
    print("   - Planning precondition checking")
    print("   - Legal reasoning and compliance")
    print()

    print("Research Alignment:")
    print("   - Nilsson (1980): Principles of AI")
    print("   - Russell & Norvig (2020): AI: A Modern Approach")
    print("   - Kowalski (1974): Logic Programming")
    print("   - Forgy (1982): RETE Algorithm")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
