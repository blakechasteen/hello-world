"""
Mathematical Logic - Model Theory, Proof Theory, Godel's Theorems
================================================================

Formal foundations of mathematics and computation.

Classes:
    PropositionalLogic: Truth tables, SAT solvers
    FirstOrderLogic: Quantifiers, models, completeness
    ModelTheory: Structures, satisfaction, compactness
    ProofTheory: Formal proofs, consistency, completeness
    GodelTheorems: Incompleteness, undecidability
    SetTheory: ZFC axioms, ordinals, cardinals
    TypeTheory: Simply-typed lambda calculus, dependent types

Applications:
    - Automated theorem proving
    - Program verification
    - Type systems for programming languages
    - Foundations of mathematics
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class LogicOperator(Enum):
    """Logical connectives."""
    NOT = "¬"
    AND = "∧"
    OR = "∨"
    IMPLIES = "→"
    IFF = "↔"


@dataclass
class Proposition:
    """Propositional formula."""
    operator: Optional[LogicOperator]
    operands: List['Proposition']
    variable: Optional[str] = None

    def __str__(self):
        if self.variable:
            return self.variable
        elif self.operator == LogicOperator.NOT:
            return f"¬{self.operands[0]}"
        elif self.operator:
            op = self.operator.value
            return f"({self.operands[0]} {op} {self.operands[1]})"
        return ""

    @staticmethod
    def var(name: str) -> 'Proposition':
        """Atomic proposition."""
        return Proposition(None, [], variable=name)

    def __and__(self, other: 'Proposition') -> 'Proposition':
        return Proposition(LogicOperator.AND, [self, other])

    def __or__(self, other: 'Proposition') -> 'Proposition':
        return Proposition(LogicOperator.OR, [self, other])

    def __invert__(self) -> 'Proposition':
        return Proposition(LogicOperator.NOT, [self])


class PropositionalLogic:
    """
    Propositional logic: Boolean algebra of propositions.

    Complete via truth tables. Decidable via SAT solvers.
    """

    @staticmethod
    def evaluate(formula: Proposition, assignment: Dict[str, bool]) -> bool:
        """Evaluate formula under truth assignment."""
        if formula.variable:
            return assignment.get(formula.variable, False)

        if formula.operator == LogicOperator.NOT:
            return not PropositionalLogic.evaluate(formula.operands[0], assignment)
        elif formula.operator == LogicOperator.AND:
            return (PropositionalLogic.evaluate(formula.operands[0], assignment) and
                   PropositionalLogic.evaluate(formula.operands[1], assignment))
        elif formula.operator == LogicOperator.OR:
            return (PropositionalLogic.evaluate(formula.operands[0], assignment) or
                   PropositionalLogic.evaluate(formula.operands[1], assignment))
        elif formula.operator == LogicOperator.IMPLIES:
            p = PropositionalLogic.evaluate(formula.operands[0], assignment)
            q = PropositionalLogic.evaluate(formula.operands[1], assignment)
            return (not p) or q
        elif formula.operator == LogicOperator.IFF:
            p = PropositionalLogic.evaluate(formula.operands[0], assignment)
            q = PropositionalLogic.evaluate(formula.operands[1], assignment)
            return p == q

        return False

    @staticmethod
    def is_tautology(formula: Proposition, variables: List[str]) -> bool:
        """Check if formula is tautology (true under all assignments)."""
        # Try all 2^n assignments
        n = len(variables)
        for i in range(2**n):
            assignment = {}
            for j, var in enumerate(variables):
                assignment[var] = bool((i >> j) & 1)

            if not PropositionalLogic.evaluate(formula, assignment):
                return False

        return True

    @staticmethod
    def is_satisfiable(formula: Proposition, variables: List[str]) -> Optional[Dict[str, bool]]:
        """
        Check satisfiability (SAT problem).

        Returns satisfying assignment if exists, None otherwise.
        """
        n = len(variables)
        for i in range(2**n):
            assignment = {}
            for j, var in enumerate(variables):
                assignment[var] = bool((i >> j) & 1)

            if PropositionalLogic.evaluate(formula, assignment):
                return assignment

        return None

    @staticmethod
    def cnf_conversion(formula: Proposition) -> str:
        """
        Convert to Conjunctive Normal Form (CNF).

        Placeholder: full CNF conversion is complex.
        """
        return "CNF conversion: (A ∨ B) ∧ (¬A ∨ C) ∧ ..."


class FirstOrderLogic:
    """
    First-order logic (predicate logic): quantifiers ∀, ∃.

    More expressive than propositional logic.
    Semidecidable (complete but not decidable).
    """

    @staticmethod
    def universal_quantifier(predicate: Callable, domain: List) -> bool:
        """∀x P(x): predicate holds for all x in domain."""
        return all(predicate(x) for x in domain)

    @staticmethod
    def existential_quantifier(predicate: Callable, domain: List) -> bool:
        """∃x P(x): predicate holds for some x in domain."""
        return any(predicate(x) for x in domain)

    @staticmethod
    def de_morgan_laws() -> Dict[str, str]:
        """De Morgan's laws for quantifiers."""
        return {
            "¬∀x P(x)": "∃x ¬P(x)",
            "¬∃x P(x)": "∀x ¬P(x)",
            "¬(P ∧ Q)": "¬P ∨ ¬Q",
            "¬(P ∨ Q)": "¬P ∧ ¬Q"
        }

    @staticmethod
    def prenex_normal_form() -> str:
        """
        Prenex normal form: all quantifiers at front.

        ∀x ∃y (P(x,y) ∧ Q(x)) (prenex)
        vs (∀x P(x,y)) ∧ Q(x) (not prenex)
        """
        return "Prenex form: Q_1 x_1 Q_2 x_2 ... Q_n x_n [matrix]"


class ModelTheory:
    """
    Model theory: structures that satisfy formulas.

    Model M = (Domain, interpretation of symbols).
    """

    @staticmethod
    def satisfaction(model: Dict, formula: str) -> bool:
        """
        M ⊨ φ: model M satisfies formula φ.

        Placeholder: requires parsing and interpretation.
        """
        return True  # Simplified

    @staticmethod
    def compactness_theorem() -> str:
        """
        Compactness: if every finite subset of Γ has model, then Γ has model.

        Key theorem in model theory. Implies non-standard models.
        """
        return (
            "Compactness Theorem:\n"
            "If every finite subset of Γ has a model, then Γ has a model.\n\n"
            "Consequences:\n"
            "- Non-standard models of arithmetic exist\n"
            "- Infinite structures can be characterized by finite axioms\n"
            "- Used to prove König's lemma, ultraproduct theorem"
        )

    @staticmethod
    def lowenheim_skolem() -> str:
        """
        Löwenheim-Skolem: countable theory has countable model.

        Surprising: uncountable sets (R) can be axiomatized by countable theory,
        yet countable models exist!
        """
        return (
            "Löwenheim-Skolem Theorem:\n"
            "If first-order theory has infinite model, it has countable model.\n\n"
            "Upward: also has models of any larger cardinality.\n"
            "Downward: any model has countable elementary submodel."
        )


class ProofTheory:
    """
    Proof theory: formal proofs and syntactic properties.

    Proof systems: natural deduction, sequent calculus, Hilbert system.
    """

    @staticmethod
    def modus_ponens(p_implies_q: bool, p: bool) -> bool:
        """
        Modus ponens: P, P → Q ⊢ Q.

        Fundamental inference rule.
        """
        if p and p_implies_q:
            return True
        return False

    @staticmethod
    def deduction_theorem() -> str:
        """
        Deduction theorem: Γ, A ⊢ B iff Γ ⊢ A → B.

        Connects semantic entailment with syntactic provability.
        """
        return "Deduction Theorem: Γ, A ⊢ B  ⟺  Γ ⊢ (A → B)"

    @staticmethod
    def completeness_theorem() -> str:
        """
        Gödel's completeness: Γ ⊨ φ iff Γ ⊢ φ.

        Semantic entailment = syntactic provability (for FOL).
        """
        return (
            "Completeness Theorem (Gödel 1929):\n"
            "For first-order logic:\n"
            "Γ ⊨ φ  ⟺  Γ ⊢ φ\n\n"
            "Every semantically valid formula is provable.\n"
            "Proof system is complete (but undecidable)."
        )


class GodelTheorems:
    """
    Gödel's incompleteness theorems: limits of formal systems.

    Revolutionary results showing inherent limitations of mathematics.
    """

    @staticmethod
    def first_incompleteness() -> str:
        """
        Gödel's First Incompleteness Theorem (1931).

        Any consistent formal system containing arithmetic is incomplete.
        """
        return (
            "Gödel's First Incompleteness Theorem (1931):\n\n"
            "Any consistent formal system F that can express basic arithmetic contains\n"
            "a statement G such that:\n"
            "  1. G is true in the standard model of arithmetic\n"
            "  2. Neither G nor ¬G is provable in F\n\n"
            "Informally: G says 'I am not provable in F'.\n\n"
            "Consequences:\n"
            "- Mathematics cannot be completely formalized\n"
            "- No finite set of axioms can prove all truths about arithmetic\n"
            "- There exist true but unprovable statements\n\n"
            "Examples of unprovable statements:\n"
            "- Gödel sentence G\n"
            "- Consistency of system: Con(F)\n"
            "- Goodstein's theorem (provable in set theory, not in PA)"
        )

    @staticmethod
    def second_incompleteness() -> str:
        """
        Gödel's Second Incompleteness Theorem.

        No consistent system can prove its own consistency.
        """
        return (
            "Gödel's Second Incompleteness Theorem:\n\n"
            "No consistent formal system F (containing arithmetic) can prove\n"
            "its own consistency Con(F).\n\n"
            "Informally: 'I cannot prove that I am consistent'\n\n"
            "Consequences:\n"
            "- Hilbert's program (prove consistency of mathematics within math) fails\n"
            "- Must use stronger system to prove consistency of weaker system\n"
            "- ZFC cannot prove Con(ZFC) unless ZFC is inconsistent!\n\n"
            "Proof sketch:\n"
            "If F ⊢ Con(F), then F ⊢ G (Gödel sentence)\n"
            "But G is unprovable if F consistent (First Theorem)\n"
            "Contradiction => F cannot prove Con(F)"
        )

    @staticmethod
    def diagonal_lemma() -> str:
        """
        Diagonal lemma (fixed-point theorem for formulas).

        Foundation of Gödel's proof.
        """
        return (
            "Diagonal Lemma:\n"
            "For any formula φ(x), there exists sentence G such that:\n"
            "⊢ G ↔ φ(⌈G⌉)\n\n"
            "G is a 'fixed point' that refers to itself via Gödel numbering.\n"
            "Used to construct self-referential sentences."
        )


class SetTheory:
    """
    Axiomatic set theory: ZFC (Zermelo-Fraenkel with Choice).

    Foundation of modern mathematics.
    """

    @staticmethod
    def zfc_axioms() -> Dict[str, str]:
        """Axioms of ZFC set theory."""
        return {
            "Extensionality": "Sets with same elements are equal: ∀x(x∈A ↔ x∈B) → A=B",
            "Empty Set": "∃x ∀y (y ∉ x)  [empty set exists]",
            "Pairing": "∀x ∀y ∃z (z = {x, y})  [unordered pairs exist]",
            "Union": "∀F ∃A ∀x (x∈A ↔ ∃Y∈F (x∈Y))  [unions exist]",
            "Power Set": "∀x ∃y (y = P(x))  [power set exists]",
            "Infinity": "∃x (∅∈x ∧ ∀y∈x (y∪{y}∈x))  [infinite set exists]",
            "Replacement": "Image of set under function is set",
            "Regularity": "∀x (x≠∅ → ∃y∈x (y∩x=∅))  [no infinite descending chains]",
            "Choice": "∀x (∅∉x → ∃f: x → ∪x ∀A∈x (f(A)∈A))  [choice function exists]"
        }

    @staticmethod
    def continuum_hypothesis() -> str:
        """
        Continuum hypothesis: is there a set strictly between ℕ and ℝ?

        Proven independent of ZFC (Gödel, Cohen).
        """
        return (
            "Continuum Hypothesis (CH):\n"
            "There is no set with cardinality strictly between |ℕ| and |ℝ|.\n\n"
            "Formally: 2^(ℵ_0) = ℵ_1\n\n"
            "Status:\n"
            "- Proven independent of ZFC (Gödel 1940, Cohen 1963)\n"
            "- Can neither be proved nor disproved from ZFC axioms\n"
            "- Both ZFC + CH and ZFC + ¬CH are consistent (if ZFC is)"
        )

    @staticmethod
    def ordinal_arithmetic() -> str:
        """Ordinal numbers: extend natural numbers to transfinite."""
        return (
            "Ordinal Arithmetic:\n"
            "ω = {0, 1, 2, 3, ...}  (first infinite ordinal)\n"
            "ω + 1 = {0, 1, 2, ..., ω}\n"
            "ω + ω = ω·2\n"
            "ω^ω, ω^(ω^ω), ..., ε_0, ...\n\n"
            "Not commutative: 1 + ω = ω ≠ ω + 1"
        )

    @staticmethod
    def cardinal_arithmetic() -> str:
        """Cardinal numbers: sizes of infinite sets."""
        return (
            "Cardinal Arithmetic:\n"
            "ℵ_0 = |ℕ| (countable infinity)\n"
            "2^(ℵ_0) = |ℝ| (continuum)\n\n"
            "Cantor's theorem: |X| < |P(X)| for any set X\n"
            "Implies hierarchy: ℵ_0 < 2^(ℵ_0) < 2^(2^(ℵ_0)) < ..."
        )


class TypeTheory:
    """
    Type theory: alternative foundation to set theory.

    Used in proof assistants (Coq, Agda, Lean) and programming languages (Haskell, ML).
    """

    @staticmethod
    def simply_typed_lambda() -> str:
        """Simply-typed lambda calculus."""
        return (
            "Simply-Typed Lambda Calculus:\n\n"
            "Types: τ ::= Base | τ₁ → τ₂\n"
            "Terms: e ::= x | λx:τ.e | e₁ e₂\n\n"
            "Typing rules:\n"
            "Γ, x:τ₁ ⊢ e : τ₂\n"
            "─────────────────  (abstraction)\n"
            "Γ ⊢ λx:τ₁.e : τ₁→τ₂\n\n"
            "Properties:\n"
            "- Strong normalization: every term terminates\n"
            "- Type safety: well-typed programs don't go wrong"
        )

    @staticmethod
    def curry_howard() -> str:
        """
        Curry-Howard correspondence: proofs ≅ programs.

        Deep connection between logic and computation.
        """
        return (
            "Curry-Howard Correspondence:\n\n"
            "Logic          Type Theory       Programming\n"
            "─────────────  ───────────────  ─────────────────\n"
            "Proposition    Type             Type\n"
            "Proof          Term             Program\n"
            "Implication    Function type    Function\n"
            "Conjunction    Product type     Pair\n"
            "Disjunction    Sum type         Union/Either\n"
            "True           Unit type        Unit/void\n"
            "False          Empty type       Never/bottom\n"
            "∀ (universal)  Dependent type   Polymorphism\n"
            "∃ (existential) Σ-type          Dependent pair\n\n"
            "Proofs are programs, propositions are types!"
        )


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_propositional():
    """Example: Propositional tautology."""
    p = Proposition.var("P")
    q = Proposition.var("Q")

    # P → (Q → P) is tautology
    formula = Proposition(LogicOperator.IMPLIES, [
        p,
        Proposition(LogicOperator.IMPLIES, [q, p])
    ])

    is_taut = PropositionalLogic.is_tautology(formula, ["P", "Q"])
    return is_taut


def example_first_order():
    """Example: First-order quantifiers."""
    domain = [1, 2, 3, 4, 5]

    # ∀x (x > 0)
    all_positive = FirstOrderLogic.universal_quantifier(lambda x: x > 0, domain)

    # ∃x (x is even)
    exists_even = FirstOrderLogic.existential_quantifier(lambda x: x % 2 == 0, domain)

    return all_positive, exists_even


if __name__ == "__main__":
    print("Mathematical Logic Module")
    print("=" * 60)

    # Test 1: Propositional logic
    print("\n[Test 1] Propositional tautology")
    is_taut = example_propositional()
    print(f"P → (Q → P) is tautology: {is_taut}")

    # Test 2: Satisfiability
    print("\n[Test 2] SAT solving")
    p = Proposition.var("P")
    q = Proposition.var("Q")
    formula = p & ~p  # P ∧ ¬P (unsatisfiable)
    assignment = PropositionalLogic.is_satisfiable(formula, ["P", "Q"])
    print(f"P ∧ ¬P satisfiable: {assignment is not None}")
    print(f"Expected: False (contradiction)")

    # Test 3: First-order logic
    print("\n[Test 3] First-order quantifiers")
    all_pos, exists_even = example_first_order()
    print(f"All positive: {all_pos} (expect True)")
    print(f"Exists even: {exists_even} (expect True)")

    # Test 4: Gödel's theorems
    print("\n[Test 4] Gödel's First Incompleteness Theorem")
    print(GodelTheorems.first_incompleteness())

    # Test 5: ZFC axioms
    print("\n[Test 5] ZFC Set Theory Axioms")
    axioms = SetTheory.zfc_axioms()
    print(f"Number of axioms: {len(axioms)}")
    print(f"Sample - Infinity: {axioms['Infinity']}")

    # Test 6: Curry-Howard
    print("\n[Test 6] Curry-Howard Correspondence")
    print(TypeTheory.curry_howard())

    print("\n" + "=" * 60)
    print("All mathematical logic tests complete!")
    print("Propositional, first-order logic, and Gödel ready.")
