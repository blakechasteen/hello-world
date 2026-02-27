"""
Deductive Reasoning Engine

Implements logical inference through forward and backward chaining:
- Knowledge base (facts + rules)
- Forward chaining (data-driven: facts → derive new facts)
- Backward chaining (goal-driven: goal → find proof)
- Proof generation (explain reasoning chains)
- Unification (pattern matching with variables)

Research:
- Russell & Norvig (2020): "AI: A Modern Approach" (Ch. 7-9)
- Nilsson (1980): "Principles of Artificial Intelligence"
- Kowalski (1974): "Predicate Logic as Programming Language"
- Forgy (1982): RETE algorithm for forward chaining

Philosophy:
    "Deduction is the soul of mathematics and the foundation of reason.
     From axioms flow theorems, from facts flow conclusions,
     from knowledge flows wisdom."
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass(frozen=True, eq=True)
class Fact:
    """
    Atomic fact in knowledge base.

    Examples:
        Fact("human", ("Socrates",))           # Socrates is human
        Fact("mortal", ("Socrates",))          # Socrates is mortal
        Fact("parent", ("Alice", "Bob"))       # Alice is parent of Bob

    Variables start with '?' (e.g., "?x")
    """
    predicate: str              # Relation name (e.g., "human", "parent")
    arguments: tuple            # Arguments (e.g., ("Socrates",), ("Alice", "Bob"))

    def is_variable(self, arg: str) -> bool:
        """Check if argument is a variable (starts with ?)."""
        return isinstance(arg, str) and arg.startswith('?')

    def variables(self) -> Set[str]:
        """Get all variables in fact."""
        return {arg for arg in self.arguments if self.is_variable(arg)}

    def ground(self) -> bool:
        """Check if fact is fully ground (no variables)."""
        return len(self.variables()) == 0

    def substitute(self, bindings: Dict[str, Any]) -> 'Fact':
        """Apply variable substitutions."""
        new_args = tuple(
            bindings.get(arg, arg) if self.is_variable(arg) else arg
            for arg in self.arguments
        )
        return Fact(self.predicate, new_args)

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.arguments)
        return f"{self.predicate}({args_str})"

    def __hash__(self):
        return hash((self.predicate, self.arguments))


@dataclass
class Rule:
    """
    Logical rule: premises → conclusion.

    Examples:
        # All humans are mortal
        Rule(
            premises=[Fact("human", ("?x",))],
            conclusion=Fact("mortal", ("?x",)),
            name="mortality"
        )

        # Transitivity of ancestry
        Rule(
            premises=[
                Fact("parent", ("?x", "?y")),
                Fact("parent", ("?y", "?z"))
            ],
            conclusion=Fact("grandparent", ("?x", "?z")),
            name="grandparent"
        )
    """
    premises: List[Fact]
    conclusion: Fact
    confidence: float = 1.0     # Rule certainty (0-1)
    name: str = ""              # Rule identifier

    def variables(self) -> Set[str]:
        """Get all variables in rule."""
        vars_set = set()
        for premise in self.premises:
            vars_set |= premise.variables()
        vars_set |= self.conclusion.variables()
        return vars_set

    def substitute(self, bindings: Dict[str, Any]) -> 'Rule':
        """Apply variable substitutions to entire rule."""
        new_premises = [p.substitute(bindings) for p in self.premises]
        new_conclusion = self.conclusion.substitute(bindings)
        return Rule(new_premises, new_conclusion, self.confidence, self.name)

    def __repr__(self):
        premises_str = " ∧ ".join(str(p) for p in self.premises)
        return f"{premises_str} → {self.conclusion}" + (f" [{self.name}]" if self.name else "")


@dataclass
class Proof:
    """
    Proof chain showing how fact was derived.

    Represents complete reasoning trace from axioms to conclusion.
    """
    conclusion: Fact
    rules_applied: List[Tuple[Rule, Dict[str, Any]]]  # (rule, bindings) pairs
    premises_used: List[Fact]
    depth: int = 0

    def __repr__(self):
        return f"Proof({self.conclusion}, {len(self.rules_applied)} steps)"

    def to_string(self, indent: int = 0) -> str:
        """Human-readable proof explanation."""
        prefix = "  " * indent
        lines = [f"{prefix}PROOF: {self.conclusion}"]

        if self.rules_applied:
            lines.append(f"{prefix}Steps:")
            for i, (rule, bindings) in enumerate(self.rules_applied, 1):
                lines.append(f"{prefix}  {i}. Apply {rule.name or 'rule'}")
                if bindings:
                    bindings_str = ", ".join(f"{k}={v}" for k, v in bindings.items())
                    lines.append(f"{prefix}     Bindings: {bindings_str}")

        if self.premises_used:
            lines.append(f"{prefix}Using facts:")
            for fact in self.premises_used:
                lines.append(f"{prefix}  - {fact}")

        return "\n".join(lines)


# ============================================================================
# Unification (Pattern Matching with Variables)
# ============================================================================

class Unifier:
    """
    Unification algorithm for pattern matching.

    Finds variable bindings that make two facts identical.

    Example:
        Fact("human", ("?x",)) unifies with Fact("human", ("Socrates",))
        → Bindings: {?x: "Socrates"}
    """

    @staticmethod
    def unify(fact1: Fact, fact2: Fact,
              bindings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Unify two facts, returning variable bindings or None if impossible.

        Args:
            fact1: First fact (may have variables)
            fact2: Second fact (may have variables)
            bindings: Existing bindings to respect

        Returns:
            Updated bindings dict if unifiable, None otherwise
        """
        if bindings is None:
            bindings = {}

        # Different predicates → cannot unify
        if fact1.predicate != fact2.predicate:
            return None

        # Different arities → cannot unify
        if len(fact1.arguments) != len(fact2.arguments):
            return None

        # Try to unify each argument pair
        new_bindings = bindings.copy()
        for arg1, arg2 in zip(fact1.arguments, fact2.arguments):
            # Apply existing bindings
            if fact1.is_variable(arg1) and arg1 in new_bindings:
                arg1 = new_bindings[arg1]
            if fact2.is_variable(arg2) and arg2 in new_bindings:
                arg2 = new_bindings[arg2]

            # Both variables
            if fact1.is_variable(arg1) and fact2.is_variable(arg2):
                # Bind first to second
                new_bindings[arg1] = arg2

            # First is variable
            elif fact1.is_variable(arg1):
                new_bindings[arg1] = arg2

            # Second is variable
            elif fact2.is_variable(arg2):
                new_bindings[arg2] = arg1

            # Both constants
            else:
                if arg1 != arg2:
                    return None  # Cannot unify

        return new_bindings

    @staticmethod
    def can_unify(fact1: Fact, fact2: Fact) -> bool:
        """Quick check if two facts can unify."""
        return Unifier.unify(fact1, fact2) is not None


# ============================================================================
# Knowledge Base
# ============================================================================

class KnowledgeBase:
    """
    Repository of facts and rules.

    Supports efficient querying and inference.
    """

    def __init__(self):
        """Initialize empty knowledge base."""
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []

        # Indexing for efficient lookup
        self._fact_index: Dict[str, Set[Fact]] = defaultdict(set)  # predicate -> facts
        self._rule_index: Dict[str, List[Rule]] = defaultdict(list)  # conclusion predicate -> rules

        logger.info("Initialized KnowledgeBase")

    # ------------------------------------------------------------------------
    # Adding Knowledge
    # ------------------------------------------------------------------------

    def add_fact(self, fact: Fact):
        """Add fact to knowledge base."""
        if not fact.ground():
            raise ValueError(f"Cannot add non-ground fact: {fact}")

        self.facts.add(fact)
        self._fact_index[fact.predicate].add(fact)
        logger.debug(f"Added fact: {fact}")

    def add_facts(self, facts: List[Fact]):
        """Add multiple facts."""
        for fact in facts:
            self.add_fact(fact)

    def add_rule(self, rule: Rule):
        """Add inference rule."""
        self.rules.append(rule)
        self._rule_index[rule.conclusion.predicate].append(rule)
        logger.debug(f"Added rule: {rule}")

    def add_rules(self, rules: List[Rule]):
        """Add multiple rules."""
        for rule in rules:
            self.add_rule(rule)

    # ------------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------------

    def query(self, query_fact: Fact) -> bool:
        """
        Check if fact is in KB (exact match for ground facts).

        Args:
            query_fact: Fact to query (must be ground)

        Returns:
            True if fact exists in KB
        """
        if not query_fact.ground():
            raise ValueError(f"Query must be ground: {query_fact}")

        return query_fact in self.facts

    def query_with_unification(self, query_fact: Fact) -> List[Dict[str, Any]]:
        """
        Query with pattern matching (supports variables).

        Args:
            query_fact: Fact pattern (may have variables)

        Returns:
            List of variable bindings that satisfy query
        """
        results = []

        # Get candidate facts (same predicate)
        candidates = self._fact_index.get(query_fact.predicate, set())

        # Try to unify with each candidate
        for fact in candidates:
            bindings = Unifier.unify(query_fact, fact)
            if bindings is not None:
                results.append(bindings)

        return results

    def get_rules_for(self, predicate: str) -> List[Rule]:
        """Get rules that conclude given predicate."""
        return self._rule_index.get(predicate, [])

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    def size(self) -> Tuple[int, int]:
        """Return (num_facts, num_rules)."""
        return len(self.facts), len(self.rules)

    def __repr__(self):
        n_facts, n_rules = self.size()
        return f"KnowledgeBase({n_facts} facts, {n_rules} rules)"


# ============================================================================
# Deductive Reasoner
# ============================================================================

class DeductiveReasoner:
    """
    Deductive reasoning engine with forward and backward chaining.

    Forward Chaining (Data-Driven):
        Start with facts → Apply rules → Derive new facts → Repeat

    Backward Chaining (Goal-Driven):
        Start with goal → Find rules that conclude goal → Prove premises
    """

    def __init__(self, kb: KnowledgeBase):
        """
        Initialize reasoner with knowledge base.

        Args:
            kb: Knowledge base containing facts and rules
        """
        self.kb = kb
        self.unifier = Unifier()

        logger.info(f"Initialized DeductiveReasoner with {kb}")

    # ------------------------------------------------------------------------
    # Forward Chaining
    # ------------------------------------------------------------------------

    def forward_chain(self, max_iterations: int = 100) -> Set[Fact]:
        """
        Forward chaining inference.

        Algorithm:
        1. Start with known facts
        2. For each rule:
           - Check if all premises satisfied
           - If yes, add conclusion to facts
        3. Repeat until no new facts (fixed point)

        Args:
            max_iterations: Max iterations to prevent infinite loops

        Returns:
            All derivable facts (original + inferred)
        """
        logger.info("Starting forward chaining")

        derived_facts = self.kb.facts.copy()
        new_facts_added = True
        iteration = 0

        while new_facts_added and iteration < max_iterations:
            new_facts_added = False
            iteration += 1

            # Try each rule
            for rule in self.kb.rules:
                # Find all ways to satisfy rule premises
                satisfying_bindings = self._find_satisfying_bindings(rule, derived_facts)

                # For each way to satisfy premises
                for bindings in satisfying_bindings:
                    # Derive conclusion
                    conclusion = rule.conclusion.substitute(bindings)

                    # Add if new
                    if conclusion not in derived_facts:
                        derived_facts.add(conclusion)
                        new_facts_added = True
                        logger.debug(f"Iteration {iteration}: Derived {conclusion} using {rule.name}")

        logger.info(f"Forward chaining complete: {len(derived_facts)} total facts "
                   f"({len(derived_facts) - len(self.kb.facts)} new)")

        return derived_facts

    def _find_satisfying_bindings(self, rule: Rule,
                                  facts: Set[Fact]) -> List[Dict[str, Any]]:
        """
        Find all variable bindings that satisfy rule premises.

        Uses recursive backtracking to find all consistent bindings.
        """
        # Recursive helper
        def find_bindings_recursive(premises: List[Fact],
                                   bindings: Dict[str, Any]) -> List[Dict[str, Any]]:
            # Base case: all premises satisfied
            if not premises:
                return [bindings]

            # Try to satisfy first premise
            first_premise = premises[0].substitute(bindings)
            remaining = premises[1:]

            all_bindings = []

            # Try to unify with each fact
            for fact in facts:
                new_bindings = self.unifier.unify(first_premise, fact, bindings)
                if new_bindings is not None:
                    # Recurse on remaining premises
                    sub_bindings = find_bindings_recursive(remaining, new_bindings)
                    all_bindings.extend(sub_bindings)

            return all_bindings

        return find_bindings_recursive(rule.premises, {})

    # ------------------------------------------------------------------------
    # Backward Chaining
    # ------------------------------------------------------------------------

    def backward_chain(self, goal: Fact, max_depth: int = 10) -> Optional[Proof]:
        """
        Backward chaining to prove goal.

        Algorithm:
        1. Check if goal in facts → Success
        2. Find rules that conclude goal
        3. For each rule:
           - Recursively prove each premise
           - If all proven → Success
        4. If no rules work → Failure

        Args:
            goal: Fact to prove
            max_depth: Max recursion depth

        Returns:
            Proof if goal provable, None otherwise
        """
        logger.info(f"Starting backward chaining for goal: {goal}")

        # Track visited goals to prevent cycles
        visited = set()

        proof = self._backward_chain_recursive(goal, max_depth, visited)

        if proof:
            logger.info(f"✓ Goal proven: {goal}")
        else:
            logger.info(f"✗ Cannot prove goal: {goal}")

        return proof

    def _backward_chain_recursive(self,
                                  goal: Fact,
                                  depth: int,
                                  visited: Set[Fact]) -> Optional[Proof]:
        """Recursive backward chaining."""

        # Base case: max depth reached
        if depth <= 0:
            logger.debug(f"Max depth reached for {goal}")
            return None

        # Prevent cycles
        if goal in visited:
            logger.debug(f"Cycle detected for {goal}")
            return None

        visited.add(goal)

        # Case 1: Goal is a known fact
        if goal.ground() and goal in self.kb.facts:
            return Proof(goal, [], [goal], depth=depth)

        # Case 2: Try to prove using rules
        rules = self.kb.get_rules_for(goal.predicate)

        for rule in rules:
            # Try to unify rule conclusion with goal
            bindings = self.unifier.unify(rule.conclusion, goal)

            if bindings is None:
                continue  # This rule doesn't help

            # Apply bindings to rule
            instantiated_rule = rule.substitute(bindings)

            # Try to prove all premises
            premise_proofs = []
            all_proven = True

            for premise in instantiated_rule.premises:
                premise_proof = self._backward_chain_recursive(premise, depth - 1, visited)

                if premise_proof is None:
                    all_proven = False
                    break

                premise_proofs.append(premise_proof)

            # If all premises proven, we have a proof!
            if all_proven:
                # Collect all base facts used
                all_premises = []
                all_rules = [(rule, bindings)]

                for premise_proof in premise_proofs:
                    all_premises.extend(premise_proof.premises_used)
                    all_rules.extend(premise_proof.rules_applied)

                return Proof(
                    conclusion=goal,
                    rules_applied=all_rules,
                    premises_used=all_premises,
                    depth=depth
                )

        # No rule worked
        visited.remove(goal)
        return None

    # ------------------------------------------------------------------------
    # Proof Explanation
    # ------------------------------------------------------------------------

    def explain(self, fact: Fact) -> Optional[Proof]:
        """
        Generate proof explanation for fact.

        Args:
            fact: Fact to explain

        Returns:
            Proof chain or None if not provable
        """
        return self.backward_chain(fact)

    def explain_to_string(self, fact: Fact) -> str:
        """Get human-readable explanation."""
        proof = self.explain(fact)

        if proof is None:
            return f"Cannot prove: {fact}"

        return proof.to_string()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_fact(predicate: str, *args) -> Fact:
    """
    Convenience function to create facts.

    Example:
        create_fact("human", "Socrates")
        create_fact("parent", "Alice", "Bob")
    """
    return Fact(predicate, args)


def create_rule(premises: List[Fact], conclusion: Fact, name: str = "") -> Rule:
    """Convenience function to create rules."""
    return Rule(premises, conclusion, name=name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'Fact',
    'Rule',
    'Proof',
    'Unifier',
    'KnowledgeBase',
    'DeductiveReasoner',
    'create_fact',
    'create_rule',
]
