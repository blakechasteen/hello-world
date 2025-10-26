"""
Computability Theory - Turing Machines, Decidability, Complexity
===============================================================

Theory of what can and cannot be computed.

Classes:
    TuringMachine: Universal model of computation
    ChurchTuring: Thesis on computable functions
    Decidability: Decidable vs undecidable problems
    HaltingProblem: Classic undecidable problem
    ComplexityClasses: P, NP, PSPACE, EXPTIME
    Reductions: Polynomial-time reductions
    NPCompleteness: Cook-Levin theorem, SAT

Applications:
    - Limits of computation
    - Cryptography (one-way functions)
    - Algorithm design
    - Complexity theory
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """Tape head movement direction."""
    LEFT = -1
    RIGHT = 1
    STAY = 0


@dataclass
class TuringState:
    """State of Turing machine."""
    state: str
    tape: List[str]
    head_position: int
    is_accept: bool = False
    is_reject: bool = False


class TuringMachine:
    """
    Turing machine: universal model of computation.

    M = (Q, Σ, Γ, δ, q_0, q_accept, q_reject)
    """

    def __init__(self, states: Set[str], alphabet: Set[str], tape_alphabet: Set[str],
                 transition: Callable, start_state: str, accept_state: str, reject_state: str):
        """
        Args:
            states: Finite set of states Q
            alphabet: Input alphabet Σ
            tape_alphabet: Tape alphabet Γ (Σ ⊂ Γ)
            transition: δ: Q × Γ → Q × Γ × {L,R}
            start_state: q_0
            accept_state: q_accept
            reject_state: q_reject
        """
        self.states = states
        self.alphabet = alphabet
        self.tape_alphabet = tape_alphabet
        self.transition = transition
        self.start_state = start_state
        self.accept_state = accept_state
        self.reject_state = reject_state

    def run(self, input_string: str, max_steps: int = 1000) -> Tuple[bool, List[TuringState]]:
        """
        Simulate Turing machine on input.

        Returns: (accepted, computation trace)
        """
        # Initialize tape
        tape = list(input_string) + ['_'] * 100  # Blank symbol '_'
        state = self.start_state
        head = 0
        trace = [TuringState(state, tape.copy(), head)]

        for step in range(max_steps):
            # Check if halted
            if state == self.accept_state:
                trace[-1].is_accept = True
                return True, trace
            if state == self.reject_state:
                trace[-1].is_reject = True
                return False, trace

            # Read symbol
            symbol = tape[head]

            # Apply transition
            try:
                new_state, write_symbol, direction = self.transition(state, symbol)
            except:
                # No transition defined: reject
                return False, trace

            # Write and move
            tape[head] = write_symbol
            if direction == Direction.LEFT:
                head = max(0, head - 1)
            elif direction == Direction.RIGHT:
                head += 1
                if head >= len(tape):
                    tape.append('_')

            state = new_state
            trace.append(TuringState(state, tape.copy(), head))

        # Max steps exceeded: consider non-halting
        return False, trace

    @staticmethod
    def binary_increment() -> 'TuringMachine':
        """
        Turing machine that increments binary number.

        Example: 101 -> 110
        """
        def delta(state: str, symbol: str) -> Tuple[str, str, Direction]:
            if state == "q0":
                if symbol == '0' or symbol == '1':
                    return ("q0", symbol, Direction.RIGHT)
                elif symbol == '_':
                    return ("q1", '_', Direction.LEFT)
            elif state == "q1":
                if symbol == '1':
                    return ("q1", '0', Direction.LEFT)
                elif symbol == '0':
                    return ("q_accept", '1', Direction.STAY)
                elif symbol == '_':
                    return ("q_accept", '1', Direction.STAY)
            raise ValueError(f"No transition for ({state}, {symbol})")

        return TuringMachine(
            states={"q0", "q1", "q_accept", "q_reject"},
            alphabet={'0', '1'},
            tape_alphabet={'0', '1', '_'},
            transition=delta,
            start_state="q0",
            accept_state="q_accept",
            reject_state="q_reject"
        )


class ChurchTuringThesis:
    """
    Church-Turing thesis: Turing machines capture intuitive notion of algorithm.

    Equivalent models: lambda calculus, register machines, cellular automata.
    """

    @staticmethod
    def statement() -> str:
        """Statement of Church-Turing thesis."""
        return (
            "Church-Turing Thesis:\n\n"
            "Every effectively calculable function is computable by a Turing machine.\n\n"
            "Equivalent formulations:\n"
            "- Lambda calculus (Church)\n"
            "- Recursive functions (Gödel, Kleene)\n"
            "- Register machines\n"
            "- Cellular automata\n"
            "- Quantum computers (extended thesis)\n\n"
            "NOT a theorem (cannot be proved), but universally accepted.\n"
            "No counterexample found in 90+ years."
        )

    @staticmethod
    def universal_turing_machine() -> str:
        """Universal Turing machine: interpreter for Turing machines."""
        return (
            "Universal Turing Machine (UTM):\n\n"
            "A single Turing machine U that can simulate any TM M on input w:\n"
            "U(<M>, w) = M(w)\n\n"
            "Encodes both machine description and input on tape.\n"
            "Foundation of stored-program computer concept.\n\n"
            "Implications:\n"
            "- Programmable computers possible\n"
            "- Interpreters and compilers exist\n"
            "- Universal computation is achievable"
        )


class Decidability:
    """
    Decidability: problems that can be solved algorithmically.

    Decidable: algorithm always halts with yes/no answer.
    Undecidable: no algorithm can solve all instances.
    """

    @staticmethod
    def decidable_languages() -> List[str]:
        """Examples of decidable languages."""
        return [
            "Regular languages (DFA)",
            "Context-free languages (PDA)",
            "Presburger arithmetic (addition only)",
            "Finite graphs (many properties)",
            "Equivalence of DFAs"
        ]

    @staticmethod
    def undecidable_problems() -> List[str]:
        """Famous undecidable problems."""
        return [
            "Halting problem",
            "Post correspondence problem",
            "Hilbert's 10th problem (Diophantine equations)",
            "Tiling problem (Wang tiles)",
            "Entscheidungsproblem (general validity of FOL)",
            "Rice's theorem (non-trivial TM properties)",
            "Word problem for groups (general case)"
        ]

    @staticmethod
    def rices_theorem() -> str:
        """
        Rice's theorem: all non-trivial semantic properties of TMs undecidable.

        Semantic property: depends on language recognized, not machine encoding.
        """
        return (
            "Rice's Theorem:\n\n"
            "Any non-trivial property of Turing-recognizable languages is undecidable.\n\n"
            "Non-trivial: some TMs have property, some don't.\n\n"
            "Examples of undecidable properties:\n"
            "- Does TM accept empty language?\n"
            "- Does TM accept a finite language?\n"
            "- Does TM accept all strings?\n"
            "- Does TM compute a total function?\n\n"
            "Only trivial properties (always true or always false) are decidable."
        )


class HaltingProblem:
    """
    Halting problem: determine if TM M halts on input w.

    Classic undecidable problem (Turing 1936).
    """

    @staticmethod
    def undecidability_proof() -> str:
        """Proof that halting problem is undecidable."""
        return (
            "Halting Problem is Undecidable (Turing 1936):\n\n"
            "Problem: Given TM M and input w, does M halt on w?\n\n"
            "Proof by contradiction:\n"
            "1. Assume halting decider H exists:\n"
            "   H(M, w) = accept if M halts on w\n"
            "           = reject if M loops on w\n\n"
            "2. Construct diagonal machine D:\n"
            "   D(M) = loop if H(M, M) accepts\n"
            "        = halt if H(M, M) rejects\n\n"
            "3. Run D on itself: D(D)\n"
            "   - If D(D) halts, then H(D, D) rejects\n"
            "     => D(D) loops (by definition of D)\n"
            "     Contradiction!\n"
            "   - If D(D) loops, then H(D, D) accepts\n"
            "     => D(D) halts (by definition of D)\n"
            "     Contradiction!\n\n"
            "4. No such H can exist. QED.\n\n"
            "Uses diagonalization (similar to Cantor, Gödel)."
        )

    @staticmethod
    def semi_decidable() -> str:
        """Halting problem is semi-decidable (Turing-recognizable)."""
        return (
            "Halting is Semi-Decidable:\n\n"
            "Can recognize when TM halts (simulate until it halts).\n"
            "Cannot recognize when TM doesn't halt (simulation never ends).\n\n"
            "Language: A_TM = {<M, w> : M accepts w}\n"
            "- Turing-recognizable (can enumerate accepts)\n"
            "- NOT Turing-decidable (cannot decide all instances)\n\n"
            "Complement: ~A_TM is not even Turing-recognizable!"
        )


class ComplexityClasses:
    """
    Complexity classes: classification by resources needed.

    Time complexity: number of steps.
    Space complexity: tape cells used.
    """

    @staticmethod
    def class_P() -> str:
        """
        P: problems solvable in polynomial time.

        P = problems decidable by deterministic TM in O(n^k) time.
        """
        return (
            "Class P (Polynomial Time):\n\n"
            "Problems solvable in polynomial time on deterministic TM.\n"
            "P = ∪_{k≥0} TIME(n^k)\n\n"
            "Examples:\n"
            "- Sorting\n"
            "- Shortest path (Dijkstra)\n"
            "- Matrix multiplication\n"
            "- Linear programming\n"
            "- Primality testing (AKS algorithm)\n\n"
            "Considered 'efficiently solvable'"
        )

    @staticmethod
    def class_NP() -> str:
        """
        NP: problems verifiable in polynomial time.

        NP = problems solvable by non-deterministic TM in polynomial time.
        """
        return (
            "Class NP (Nondeterministic Polynomial):\n\n"
            "Problems where solution can be verified in polynomial time.\n"
            "Equivalently: solvable by nondeterministic TM in poly time.\n\n"
            "Examples:\n"
            "- SAT (Boolean satisfiability)\n"
            "- TSP (Traveling salesman - decision version)\n"
            "- Graph coloring\n"
            "- Subset sum\n"
            "- Hamiltonian path\n\n"
            "Key property: certificate verification is efficient.\n"
            "P ⊆ NP (can verify by just solving)"
        )

    @staticmethod
    def P_vs_NP() -> str:
        """
        P vs NP: most important open problem in computer science.

        $1 million Clay prize for solution.
        """
        return (
            "P vs NP Problem:\n\n"
            "Is P = NP?\n\n"
            "Does verification efficiency imply solving efficiency?\n\n"
            "Implications if P = NP:\n"
            "- Efficient algorithms for all NP problems\n"
            "- Cryptography collapses (RSA, etc. broken)\n"
            "- Many optimization problems become tractable\n"
            "- Automated theorem proving becomes efficient\n\n"
            "Implications if P ≠ NP:\n"
            "- Fundamental limits on computation exist\n"
            "- Some problems inherently hard\n"
            "- One-way functions exist (foundation of crypto)\n\n"
            "Consensus: P ≠ NP (but unproven)\n"
            "Clay Millennium Prize: $1,000,000"
        )

    @staticmethod
    def complexity_hierarchy() -> Dict[str, str]:
        """Complexity class hierarchy."""
        return {
            "P": "Polynomial time (deterministic)",
            "NP": "Nondeterministic polynomial time",
            "co-NP": "Complements of NP languages",
            "PSPACE": "Polynomial space",
            "EXPTIME": "Exponential time",
            "NEXPTIME": "Nondeterministic exponential time",
            "Relations": "P ⊆ NP ⊆ PSPACE ⊆ EXPTIME ⊆ NEXPTIME"
        }


class NPCompleteness:
    """
    NP-completeness: hardest problems in NP.

    If any NP-complete problem is in P, then P = NP.
    """

    @staticmethod
    def definition() -> str:
        """Definition of NP-completeness."""
        return (
            "NP-Complete:\n\n"
            "Language L is NP-complete if:\n"
            "1. L ∈ NP\n"
            "2. Every problem in NP reduces to L in polynomial time\n\n"
            "Hardest problems in NP.\n"
            "If any NP-complete problem is in P, then P = NP."
        )

    @staticmethod
    def cook_levin_theorem() -> str:
        """
        Cook-Levin theorem: SAT is NP-complete.

        First NP-completeness proof (1971).
        """
        return (
            "Cook-Levin Theorem (1971):\n\n"
            "Boolean satisfiability (SAT) is NP-complete.\n\n"
            "Proof sketch:\n"
            "1. SAT ∈ NP: given assignment, verify in poly time\n"
            "2. Every NP problem reduces to SAT:\n"
            "   - Any NP problem has poly-time verifier TM\n"
            "   - Encode TM computation as Boolean formula\n"
            "   - Formula satisfiable iff TM accepts\n\n"
            "First NP-complete problem discovered.\n"
            "Opened floodgates: thousands of NP-complete problems found."
        )

    @staticmethod
    def np_complete_problems() -> List[str]:
        """Famous NP-complete problems."""
        return [
            "SAT (Boolean satisfiability) - Cook-Levin",
            "3-SAT (3-CNF satisfiability) - Karp",
            "Clique (largest clique in graph)",
            "Vertex Cover (smallest vertex cover)",
            "Hamiltonian Path/Cycle",
            "Traveling Salesman Problem (decision)",
            "Graph Coloring (k-colorability)",
            "Subset Sum",
            "Knapsack (decision version)",
            "Integer Programming"
        ]

    @staticmethod
    def polynomial_reduction() -> str:
        """Polynomial-time reduction."""
        return (
            "Polynomial Reduction: A ≤_p B\n\n"
            "Function f computable in poly time such that:\n"
            "x ∈ A  ⟺  f(x) ∈ B\n\n"
            "If A ≤_p B and B ∈ P, then A ∈ P.\n"
            "Used to prove NP-completeness:\n"
            "To show C is NP-complete:\n"
            "1. Show C ∈ NP\n"
            "2. Show known NP-complete problem reduces to C"
        )


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_turing_machine():
    """Example: Binary increment Turing machine."""
    tm = TuringMachine.binary_increment()
    input_str = "101"  # Binary 5
    accepted, trace = tm.run(input_str)

    # Extract final tape content
    if trace:
        final_tape = ''.join(trace[-1].tape).strip('_')
        return accepted, final_tape


def example_halting_problem():
    """Example: Halting problem undecidability."""
    proof = HaltingProblem.undecidability_proof()
    return proof


if __name__ == "__main__":
    print("Computability Theory Module")
    print("=" * 60)

    # Test 1: Turing machine
    print("\n[Test 1] Turing machine (binary increment)")
    accepted, result = example_turing_machine()
    print(f"Input: 101 (binary 5)")
    print(f"Output: {result} (expect 110 = binary 6)")
    print(f"Accepted: {accepted}")

    # Test 2: Church-Turing thesis
    print("\n[Test 2] Church-Turing Thesis")
    print(ChurchTuringThesis.statement())

    # Test 3: Halting problem
    print("\n[Test 3] Halting Problem Undecidability")
    print(HaltingProblem.undecidability_proof())

    # Test 4: P vs NP
    print("\n[Test 4] P vs NP Problem")
    print(ComplexityClasses.P_vs_NP())

    # Test 5: NP-complete problems
    print("\n[Test 5] NP-Complete Problems")
    problems = NPCompleteness.np_complete_problems()
    print(f"Found {len(problems)} NP-complete problems:")
    for i, prob in enumerate(problems[:5], 1):
        print(f"  {i}. {prob}")

    # Test 6: Rice's theorem
    print("\n[Test 6] Rice's Theorem")
    print(Decidability.rices_theorem())

    print("\n" + "=" * 60)
    print("All computability theory tests complete!")
    print("Turing machines, undecidability, and complexity ready.")
