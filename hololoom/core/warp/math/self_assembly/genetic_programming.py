"""
Genetic Programming — Tree GP, NEAT, Grammar-Guided, Neural Induction
======================================================================

Program synthesis via evolutionary methods: evolving tree-structured
programs, neural network topologies, grammar-constrained derivations,
and differentiable program induction.

Core Concepts:
- Genetic Programming: Evolve tree-structured programs via crossover/mutation
- NEAT: Neuroevolution of Augmenting Topologies (complexify networks)
- Grammar-guided GP: Constrain search to valid programs via CFG
- Neural program induction: Differentiable interpreter for program learning

Mathematical Foundation:
Fitness-proportionate selection: P(i) = f(i) / Σ f(j)
Subtree crossover: swap random subtrees between two parent trees
NEAT speciation: δ = c₁E/N + c₂D/N + c₃W̄ (excess, disjoint, weight diff)

Applications to Warp Space:
- Evolving scoring term functions
- Discovering novel retrieval strategies
- Architecture search for embedding models

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# TREE GP
# ============================================================================

@dataclass
class GPNode:
    """Node in a GP tree (either function or terminal)."""
    value: Any                              # Function name or terminal value
    children: list['GPNode'] = field(default_factory=list)
    is_terminal: bool = False

    @property
    def depth(self) -> int:
        if self.is_terminal or not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        if self.is_terminal or not self.children:
            return 1
        return 1 + sum(c.size for c in self.children)


class GeneticProgramming:
    """
    Tree-based Genetic Programming.

    Evolves tree-structured programs using subtree crossover, point
    mutation, and tournament selection.
    """

    # Built-in function set with arities
    BUILTIN_FUNCTIONS = {
        'add': (2, lambda a, b: a + b),
        'sub': (2, lambda a, b: a - b),
        'mul': (2, lambda a, b: a * b),
        'div': (2, lambda a, b: a / b if abs(b) > 1e-10 else 1.0),
        'sin': (1, lambda a: np.sin(a)),
        'cos': (1, lambda a: np.cos(a)),
        'abs': (1, lambda a: np.abs(a)),
        'neg': (1, lambda a: -a),
        'max2': (2, lambda a, b: max(a, b)),
        'min2': (2, lambda a, b: min(a, b)),
    }

    @staticmethod
    def evolve(fitness_fn: Callable[[GPNode], float],
               terminal_set: list[Any],
               function_set: list[str] | None = None,
               n_gen: int = 50,
               pop_size: int = 100,
               max_depth: int = 6,
               crossover_rate: float = 0.7,
               mutation_rate: float = 0.2,
               tournament_size: int = 5) -> GPNode:
        """
        Evolve a GP tree to maximize fitness.

        Args:
            fitness_fn: Function mapping GPNode -> fitness (higher is better).
            terminal_set: List of terminal values (constants, variable names).
            function_set: List of function names (keys in BUILTIN_FUNCTIONS).
            n_gen: Number of generations.
            pop_size: Population size.
            max_depth: Maximum tree depth.
            crossover_rate: Probability of crossover.
            mutation_rate: Probability of mutation.
            tournament_size: Tournament selection size.

        Returns:
            Best tree found.
        """
        if function_set is None:
            function_set = ['add', 'sub', 'mul', 'div']

        arities = {}
        for fn in function_set:
            if fn in GeneticProgramming.BUILTIN_FUNCTIONS:
                arities[fn] = GeneticProgramming.BUILTIN_FUNCTIONS[fn][0]
            else:
                arities[fn] = 2  # Default arity

        # Initialize population (ramped half-and-half)
        population = []
        for i in range(pop_size):
            depth = 2 + (i % (max_depth - 1))
            method = 'full' if i % 2 == 0 else 'grow'
            tree = GeneticProgramming._random_tree(
                function_set, terminal_set, arities, depth, method
            )
            population.append(tree)

        best = None
        best_fitness = float('-inf')

        for gen in range(n_gen):
            # Evaluate fitness
            fitnesses = []
            for tree in population:
                try:
                    f = fitness_fn(tree)
                except Exception:
                    f = float('-inf')
                fitnesses.append(f)

                if f > best_fitness:
                    best_fitness = f
                    best = copy.deepcopy(tree)

            # Build next generation
            new_pop = []

            # Elitism: keep best
            if best is not None:
                new_pop.append(copy.deepcopy(best))

            while len(new_pop) < pop_size:
                r = np.random.random()

                if r < crossover_rate:
                    p1 = GeneticProgramming._tournament(
                        population, fitnesses, tournament_size
                    )
                    p2 = GeneticProgramming._tournament(
                        population, fitnesses, tournament_size
                    )
                    child = GeneticProgramming._crossover(
                        p1, p2, max_depth
                    )
                elif r < crossover_rate + mutation_rate:
                    parent = GeneticProgramming._tournament(
                        population, fitnesses, tournament_size
                    )
                    child = GeneticProgramming._mutate(
                        parent, function_set, terminal_set, arities, max_depth
                    )
                else:
                    child = copy.deepcopy(
                        GeneticProgramming._tournament(
                            population, fitnesses, tournament_size
                        )
                    )

                new_pop.append(child)

            population = new_pop[:pop_size]

        logger.info(f"GP evolved {n_gen} generations, best fitness: {best_fitness:.4f}")
        return best if best is not None else population[0]

    @staticmethod
    def evaluate(tree: GPNode, variables: dict[str, float]) -> float:
        """Evaluate a GP tree given variable bindings."""
        if tree.is_terminal:
            if isinstance(tree.value, str) and tree.value in variables:
                return variables[tree.value]
            return float(tree.value) if not isinstance(tree.value, str) else 0.0

        fn_name = tree.value
        if fn_name in GeneticProgramming.BUILTIN_FUNCTIONS:
            _, fn = GeneticProgramming.BUILTIN_FUNCTIONS[fn_name]
            args = [GeneticProgramming.evaluate(c, variables) for c in tree.children]
            try:
                return float(fn(*args))
            except (ValueError, OverflowError):
                return 0.0
        return 0.0

    @staticmethod
    def _random_tree(functions, terminals, arities, max_depth, method='grow'):
        if max_depth <= 0 or (method == 'grow' and np.random.random() < 0.3):
            return GPNode(value=terminals[np.random.randint(len(terminals))],
                          is_terminal=True)

        fn = functions[np.random.randint(len(functions))]
        arity = arities[fn]
        children = [
            GeneticProgramming._random_tree(
                functions, terminals, arities, max_depth - 1, method
            )
            for _ in range(arity)
        ]
        return GPNode(value=fn, children=children)

    @staticmethod
    def _tournament(population, fitnesses, k):
        indices = np.random.choice(len(population), min(k, len(population)),
                                   replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return copy.deepcopy(population[best_idx])

    @staticmethod
    def _get_random_node(tree, depth=0):
        nodes = []
        GeneticProgramming._collect_nodes(tree, nodes, depth)
        return nodes[np.random.randint(len(nodes))] if nodes else (tree, 0)

    @staticmethod
    def _collect_nodes(node, result, depth=0):
        result.append((node, depth))
        for child in node.children:
            GeneticProgramming._collect_nodes(child, result, depth + 1)

    @staticmethod
    def _crossover(p1, p2, max_depth):
        child = copy.deepcopy(p1)
        nodes1 = []
        GeneticProgramming._collect_nodes(child, nodes1)
        nodes2 = []
        GeneticProgramming._collect_nodes(p2, nodes2)

        if len(nodes1) < 2 or len(nodes2) < 2:
            return child

        # Pick random crossover points
        n1, _ = nodes1[np.random.randint(1, len(nodes1))]
        n2, _ = nodes2[np.random.randint(len(nodes2))]

        # Replace n1's content with n2's
        n1.value = n2.value
        n1.children = copy.deepcopy(n2.children)
        n1.is_terminal = n2.is_terminal

        return child

    @staticmethod
    def _mutate(tree, functions, terminals, arities, max_depth):
        child = copy.deepcopy(tree)
        nodes = []
        GeneticProgramming._collect_nodes(child, nodes)

        if not nodes:
            return child

        target, depth = nodes[np.random.randint(len(nodes))]
        remaining_depth = max(1, max_depth - depth)
        new_subtree = GeneticProgramming._random_tree(
            functions, terminals, arities, remaining_depth, 'grow'
        )
        target.value = new_subtree.value
        target.children = new_subtree.children
        target.is_terminal = new_subtree.is_terminal

        return child


# ============================================================================
# NEAT (NeuroEvolution of Augmenting Topologies)
# ============================================================================

@dataclass
class NEATGene:
    """A connection gene in a NEAT genome."""
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0


@dataclass
class NEATGenome:
    """A NEAT genome: collection of node and connection genes."""
    n_inputs: int
    n_outputs: int
    connections: list[NEATGene] = field(default_factory=list)
    n_hidden: int = 0
    fitness: float = 0.0

    @property
    def n_nodes(self):
        return self.n_inputs + self.n_outputs + self.n_hidden

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the network on given inputs."""
        n = self.n_nodes
        activations = np.zeros(n)
        activations[:self.n_inputs] = inputs[:self.n_inputs]

        # Topological sort via dependency tracking
        for _ in range(n):  # Max iterations
            for conn in self.connections:
                if conn.enabled:
                    activations[conn.out_node] += (
                        activations[conn.in_node] * conn.weight
                    )
            # Apply activation to hidden and output nodes
            for i in range(self.n_inputs, n):
                activations[i] = np.tanh(activations[i])

        return activations[self.n_inputs:self.n_inputs + self.n_outputs]


class NEAT:
    """
    NeuroEvolution of Augmenting Topologies.

    Evolves neural network topology and weights simultaneously.
    Key innovations: historical markings for crossover alignment,
    speciation to protect innovation.
    """

    _innovation_counter = 0

    @staticmethod
    def evolve(fitness_fn: Callable[[NEATGenome], float],
               n_inputs: int, n_outputs: int,
               n_gen: int = 100,
               pop_size: int = 50,
               weight_mutation_rate: float = 0.8,
               add_node_rate: float = 0.03,
               add_conn_rate: float = 0.05) -> NEATGenome:
        """
        Evolve a NEAT genome to maximize fitness.

        Args:
            fitness_fn: Maps genome -> fitness (higher better).
            n_inputs: Number of input nodes.
            n_outputs: Number of output nodes.
            n_gen: Number of generations.
            pop_size: Population size.

        Returns:
            Best genome found.
        """
        NEAT._innovation_counter = 0

        # Initialize: fully connected minimal topology
        population = []
        for _ in range(pop_size):
            genome = NEATGenome(n_inputs=n_inputs, n_outputs=n_outputs)
            for i in range(n_inputs):
                for j in range(n_outputs):
                    NEAT._innovation_counter += 1
                    genome.connections.append(NEATGene(
                        in_node=i,
                        out_node=n_inputs + j,
                        weight=np.random.randn() * 0.5,
                        innovation=NEAT._innovation_counter,
                    ))
            population.append(genome)

        best = None
        best_fitness = float('-inf')

        for gen in range(n_gen):
            # Evaluate
            for genome in population:
                try:
                    genome.fitness = fitness_fn(genome)
                except Exception:
                    genome.fitness = float('-inf')

                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best = copy.deepcopy(genome)

            # Selection and reproduction
            population.sort(key=lambda g: g.fitness, reverse=True)
            survivors = population[:pop_size // 2]

            new_pop = [copy.deepcopy(survivors[0])]  # Elitism

            while len(new_pop) < pop_size:
                parent = copy.deepcopy(
                    survivors[np.random.randint(len(survivors))]
                )

                # Weight mutation
                if np.random.random() < weight_mutation_rate:
                    for conn in parent.connections:
                        if np.random.random() < 0.9:
                            conn.weight += np.random.randn() * 0.1
                        else:
                            conn.weight = np.random.randn()

                # Add node
                if np.random.random() < add_node_rate:
                    enabled = [c for c in parent.connections if c.enabled]
                    if enabled:
                        conn = enabled[np.random.randint(len(enabled))]
                        conn.enabled = False
                        new_node = parent.n_inputs + parent.n_outputs + parent.n_hidden
                        parent.n_hidden += 1

                        NEAT._innovation_counter += 1
                        parent.connections.append(NEATGene(
                            conn.in_node, new_node, 1.0,
                            innovation=NEAT._innovation_counter
                        ))
                        NEAT._innovation_counter += 1
                        parent.connections.append(NEATGene(
                            new_node, conn.out_node, conn.weight,
                            innovation=NEAT._innovation_counter
                        ))

                # Add connection
                if np.random.random() < add_conn_rate:
                    n = parent.n_nodes
                    for _ in range(10):  # Try 10 times
                        i = np.random.randint(n)
                        j = np.random.randint(parent.n_inputs, n)
                        if i != j:
                            exists = any(
                                c.in_node == i and c.out_node == j
                                for c in parent.connections
                            )
                            if not exists:
                                NEAT._innovation_counter += 1
                                parent.connections.append(NEATGene(
                                    i, j, np.random.randn() * 0.5,
                                    innovation=NEAT._innovation_counter
                                ))
                                break

                new_pop.append(parent)

            population = new_pop[:pop_size]

        logger.info(f"NEAT: {n_gen} gens, best fitness {best_fitness:.4f}")
        return best if best is not None else population[0]


# ============================================================================
# GRAMMAR-GUIDED GP
# ============================================================================

class GrammarGP:
    """
    Grammar-guided genetic programming.

    Constrains the search space to syntactically valid programs
    defined by a context-free grammar.
    """

    @staticmethod
    def evolve(grammar: dict[str, list[list[str]]],
               fitness_fn: Callable[[str], float],
               n_gen: int = 50,
               pop_size: int = 100,
               start_symbol: str = 'S',
               max_depth: int = 10) -> str:
        """
        Evolve a derivation string from a grammar.

        Args:
            grammar: CFG as dict of production rules.
                     e.g., {'S': [['expr', '+', 'expr'], ['num']],
                            'expr': [['num', '*', 'num'], ['num']],
                            'num': [['1'], ['2'], ['x']]}
            fitness_fn: Maps derived string -> fitness (higher better).
            n_gen: Generations.
            pop_size: Population size.
            start_symbol: Start nonterminal.
            max_depth: Maximum derivation depth.

        Returns:
            Best derived string.
        """
        # Represent individuals as lists of production choices
        population = []
        for _ in range(pop_size):
            choices = [np.random.randint(100) for _ in range(50)]
            population.append(choices)

        best_string = ""
        best_fitness = float('-inf')

        for gen in range(n_gen):
            strings_and_fitness = []
            for choices in population:
                s = GrammarGP._derive(grammar, choices, start_symbol, max_depth)
                try:
                    f = fitness_fn(s)
                except Exception:
                    f = float('-inf')
                strings_and_fitness.append((s, f))

                if f > best_fitness:
                    best_fitness = f
                    best_string = s

            fitnesses = [sf[1] for sf in strings_and_fitness]

            # Tournament selection + crossover
            new_pop = [population[np.argmax(fitnesses)][:]]  # Elitism

            while len(new_pop) < pop_size:
                i1 = GrammarGP._tournament_idx(fitnesses, 5)
                i2 = GrammarGP._tournament_idx(fitnesses, 5)
                p1, p2 = population[i1][:], population[i2][:]

                # One-point crossover
                pt = np.random.randint(1, len(p1))
                child = p1[:pt] + p2[pt:]

                # Mutation
                for i in range(len(child)):
                    if np.random.random() < 0.05:
                        child[i] = np.random.randint(100)

                new_pop.append(child)

            population = new_pop[:pop_size]

        logger.info(f"GrammarGP: best fitness {best_fitness:.4f}")
        return best_string

    @staticmethod
    def _derive(grammar, choices, symbol, max_depth, choice_idx=None):
        if choice_idx is None:
            choice_idx = [0]

        if symbol not in grammar:
            return symbol  # Terminal

        if max_depth <= 0:
            # Pick shortest production
            prods = grammar[symbol]
            shortest = min(prods, key=len)
            return ' '.join(
                GrammarGP._derive(grammar, choices, s, 0, choice_idx)
                for s in shortest
            )

        prods = grammar[symbol]
        idx = choice_idx[0] % len(choices)
        choice_idx[0] += 1
        prod = prods[choices[idx] % len(prods)]

        parts = []
        for s in prod:
            parts.append(GrammarGP._derive(
                grammar, choices, s, max_depth - 1, choice_idx
            ))
        return ' '.join(parts)

    @staticmethod
    def _tournament_idx(fitnesses, k):
        indices = np.random.choice(len(fitnesses), min(k, len(fitnesses)),
                                   replace=False)
        return indices[np.argmax([fitnesses[i] for i in indices])]


# ============================================================================
# NEURAL PROGRAM INDUCTION
# ============================================================================

class NeuralProgramInduction:
    """
    Simple differentiable program induction.

    Learns a program (sequence of instructions) from input-output examples
    using a differentiable interpreter with soft attention over instructions.
    """

    @staticmethod
    def induce(examples: list[tuple[np.ndarray, np.ndarray]],
               max_steps: int = 10,
               n_registers: int = 4,
               n_iter: int = 200,
               lr: float = 0.01) -> list[tuple[str, int, int]]:
        """
        Induce a simple program from input-output examples.

        The program operates on registers. Available operations:
        ADD, SUB, MUL, COPY, ZERO.

        Args:
            examples: List of (input_registers, expected_output_registers).
            max_steps: Maximum program length.
            n_registers: Number of registers.
            n_iter: Training iterations.
            lr: Learning rate.

        Returns:
            Induced program as list of (operation, arg1, arg2) tuples.
        """
        n_ops = 5  # ADD, SUB, MUL, COPY, ZERO

        # Learnable parameters: attention weights over (op, arg1, arg2)
        # For each step: softmax over ops, softmax over arg1, softmax over arg2
        op_logits = np.random.randn(max_steps, n_ops) * 0.1
        arg1_logits = np.random.randn(max_steps, n_registers) * 0.1
        arg2_logits = np.random.randn(max_steps, n_registers) * 0.1

        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / (np.sum(e) + 1e-10)

        def execute_soft(inputs, op_probs, a1_probs, a2_probs):
            regs = inputs.copy()
            for step in range(max_steps):
                op_p = op_probs[step]
                a1_p = a1_probs[step]
                a2_p = a2_probs[step]

                # Soft read from registers
                v1 = np.dot(a1_p, regs)
                v2 = np.dot(a2_p, regs)

                # Soft operation
                results = np.array([
                    v1 + v2,       # ADD
                    v1 - v2,       # SUB
                    v1 * v2,       # MUL
                    v2,            # COPY
                    0.0,           # ZERO
                ])
                result = np.dot(op_p, results)

                # Soft write to registers (write to arg1 position)
                regs = regs * (1 - a1_p) + a1_p * result

            return regs

        for iteration in range(n_iter):
            total_loss = 0.0

            for inputs, expected in examples:
                regs = inputs.astype(float)
                op_probs = np.array([softmax(op_logits[s]) for s in range(max_steps)])
                a1_probs = np.array([softmax(arg1_logits[s]) for s in range(max_steps)])
                a2_probs = np.array([softmax(arg2_logits[s]) for s in range(max_steps)])

                output = execute_soft(regs, op_probs, a1_probs, a2_probs)
                loss = np.sum((output - expected) ** 2)
                total_loss += loss

                # Numerical gradient
                eps = 1e-4
                for s in range(max_steps):
                    for o in range(n_ops):
                        op_logits[s, o] += eps
                        op_p2 = np.array([softmax(op_logits[ss]) for ss in range(max_steps)])
                        out2 = execute_soft(regs, op_p2, a1_probs, a2_probs)
                        loss2 = np.sum((out2 - expected) ** 2)
                        grad = (loss2 - loss) / eps
                        op_logits[s, o] -= eps + lr * grad

                    for r in range(n_registers):
                        arg1_logits[s, r] += eps
                        a1_p2 = np.array([softmax(arg1_logits[ss]) for ss in range(max_steps)])
                        out2 = execute_soft(regs, op_probs, a1_p2, a2_probs)
                        loss2 = np.sum((out2 - expected) ** 2)
                        grad = (loss2 - loss) / eps
                        arg1_logits[s, r] -= eps + lr * grad

                        arg2_logits[s, r] += eps
                        a2_p2 = np.array([softmax(arg2_logits[ss]) for ss in range(max_steps)])
                        out2 = execute_soft(regs, op_probs, a1_probs, a2_p2)
                        loss2 = np.sum((out2 - expected) ** 2)
                        grad = (loss2 - loss) / eps
                        arg2_logits[s, r] -= eps + lr * grad

        # Extract hard program
        op_names = ['ADD', 'SUB', 'MUL', 'COPY', 'ZERO']
        program = []
        for s in range(max_steps):
            op = np.argmax(op_logits[s])
            a1 = np.argmax(arg1_logits[s])
            a2 = np.argmax(arg2_logits[s])
            program.append((op_names[op], a1, a2))

        return program
