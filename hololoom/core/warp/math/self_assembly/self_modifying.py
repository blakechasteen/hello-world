"""
Self-Modifying Systems — LSM, Echo State, Quine, Reflective Tower, Novelty Search
===================================================================================

Systems that modify their own structure or behavior during execution.

Core Concepts:
- Liquid State Machine: Spiking reservoir with temporal dynamics
- Echo State Property: Spectral condition ensuring fading memory
- Quine: Self-replicating programs
- Reflective Tower: Metacircular evaluators at multiple levels
- Open-Ended Evolution: Novelty search driving unbounded complexity

Mathematical Foundation:
LSM: x(t+1) = f(W_in u(t) + W x(t))       (reservoir state update)
ESP: ρ(W) < 1 ⟹ fading memory property     (spectral radius condition)
Novelty: ρ(x) = (1/k) Σ_{i=1}^{k} d(x, μ_i) (k-nearest behavior distance)

Applications to Warp Space:
- Reservoir computing for temporal pattern processing
- Self-modifying scoring functions that evolve during use
- Novelty-driven exploration of strategy space

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# LIQUID STATE MACHINE
# ============================================================================

class LiquidStateMachine:
    """
    Liquid State Machine: a reservoir computing model using spiking dynamics.

    The reservoir is a recurrently connected network of leaky integrate-and-fire
    (LIF) neurons. Input is encoded as spike trains, and the reservoir state
    is read out by a linear layer.

    Architecture:
        Input → [Reservoir (spiking)] → Linear readout → Output
    """

    def __init__(self, n_neurons: int = 100, connectivity: float = 0.1,
                 spectral_radius: float = 0.9, tau: float = 20.0,
                 threshold: float = 1.0, reset: float = 0.0,
                 dt: float = 1.0):
        """
        Args:
            n_neurons: Number of reservoir neurons.
            connectivity: Connection probability.
            spectral_radius: Target spectral radius of weight matrix.
            tau: Membrane time constant (ms).
            threshold: Spike threshold voltage.
            reset: Reset voltage after spike.
            dt: Time step (ms).
        """
        self.n = n_neurons
        self.tau = tau
        self.threshold = threshold
        self.reset = reset
        self.dt = dt

        # Initialize random sparse weight matrix
        W = np.random.randn(n_neurons, n_neurons) * 0.1
        mask = np.random.random((n_neurons, n_neurons)) < connectivity
        np.fill_diagonal(mask, False)
        W *= mask

        # Scale to desired spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 1e-10:
            W *= spectral_radius / rho

        self.W = W
        self.W_in = None  # Set during first process() call
        self.state = np.zeros(n_neurons)  # Membrane potentials
        self.spikes = np.zeros(n_neurons)

    def process(self, input_spike_train: np.ndarray) -> np.ndarray:
        """
        Process an input spike train through the reservoir.

        Args:
            input_spike_train: Binary spike train (n_timesteps, n_inputs).
                               Each row is a binary vector indicating which
                               input channels spike at that timestep.

        Returns:
            Reservoir state trajectory (n_timesteps, n_neurons).
            Contains membrane potentials at each timestep.
        """
        n_timesteps, n_inputs = input_spike_train.shape

        # Initialize input weights if needed
        if self.W_in is None or self.W_in.shape[1] != n_inputs:
            self.W_in = np.random.randn(self.n, n_inputs) * 0.5

        states = np.zeros((n_timesteps, self.n))
        v = self.state.copy()

        for t in range(n_timesteps):
            # Input current
            I_in = self.W_in @ input_spike_train[t]

            # Recurrent current from spikes
            I_rec = self.W @ self.spikes

            # LIF dynamics: τ dv/dt = -v + I
            dv = (-v + I_in + I_rec) * self.dt / self.tau
            v += dv

            # Spike detection
            self.spikes = (v >= self.threshold).astype(float)

            # Reset spiking neurons
            v[self.spikes > 0.5] = self.reset

            states[t] = v.copy()

        self.state = v
        return states

    def reset_state(self):
        """Reset reservoir state to zero."""
        self.state = np.zeros(self.n)
        self.spikes = np.zeros(self.n)

    def train_readout(self, states: np.ndarray,
                      targets: np.ndarray,
                      alpha: float = 1e-4) -> np.ndarray:
        """
        Train linear readout weights via ridge regression.

        Args:
            states: Reservoir states (n_timesteps, n_neurons).
            targets: Target outputs (n_timesteps, n_outputs).
            alpha: Regularization strength.

        Returns:
            Readout weights (n_neurons, n_outputs).
        """
        n = states.shape[1]
        W_out = np.linalg.solve(
            states.T @ states + alpha * np.eye(n),
            states.T @ targets
        )
        return W_out


# ============================================================================
# ECHO STATE PROPERTY
# ============================================================================

class EchoStateProperty:
    """
    Echo State Property (ESP): a reservoir has the ESP if the effect of
    initial conditions vanishes over time.

    Sufficient condition: spectral radius ρ(W) < 1.
    Necessary condition: ρ(W) < 1 for any input (with contractive nonlinearity).

    The ESP ensures the reservoir acts as a fading memory of the input,
    which is essential for temporal processing.
    """

    @staticmethod
    def check(weight_matrix: np.ndarray,
              strict: bool = True) -> bool:
        """
        Check if a weight matrix satisfies the echo state property.

        Args:
            weight_matrix: Reservoir weight matrix (n x n).
            strict: If True, requires ρ(W) < 1 (sufficient condition).
                    If False, allows ρ(W) = 1 (borderline).

        Returns:
            True if the ESP condition is satisfied.
        """
        eigvals = np.linalg.eigvals(weight_matrix)
        rho = float(np.max(np.abs(eigvals)))

        if strict:
            satisfied = rho < 1.0
        else:
            satisfied = rho <= 1.0 + 1e-10

        logger.info(
            f"Spectral radius ρ(W) = {rho:.6f}, "
            f"ESP {'satisfied' if satisfied else 'VIOLATED'}"
        )
        return satisfied

    @staticmethod
    def spectral_radius(weight_matrix: np.ndarray) -> float:
        """Compute the spectral radius ρ(W) = max|λ_i|."""
        return float(np.max(np.abs(np.linalg.eigvals(weight_matrix))))

    @staticmethod
    def rescale(weight_matrix: np.ndarray,
                target_radius: float = 0.9) -> np.ndarray:
        """
        Rescale weight matrix to achieve target spectral radius.

        W_scaled = W * (target / ρ(W))
        """
        rho = EchoStateProperty.spectral_radius(weight_matrix)
        if rho < 1e-10:
            return weight_matrix
        return weight_matrix * (target_radius / rho)

    @staticmethod
    def memory_capacity(weight_matrix: np.ndarray,
                        W_in: np.ndarray,
                        n_steps: int = 1000,
                        max_delay: int = 50) -> float:
        """
        Estimate the memory capacity of a reservoir.

        MC = Σ_k r²(k) where r(k) is the correlation between
        readout and input delayed by k steps.

        Theoretical maximum: MC ≤ n_neurons.

        Args:
            weight_matrix: Reservoir weights (n x n).
            W_in: Input weights (n x 1).
            n_steps: Length of test signal.
            max_delay: Maximum delay to test.

        Returns:
            Total memory capacity.
        """
        n = weight_matrix.shape[0]

        # Generate random input
        u = np.random.uniform(-1, 1, n_steps)

        # Run reservoir
        states = np.zeros((n_steps, n))
        x = np.zeros(n)
        for t in range(n_steps):
            x = np.tanh(weight_matrix @ x + W_in.ravel() * u[t])
            states[t] = x

        # Compute memory capacity for each delay
        mc = 0.0
        warmup = max_delay + 50

        for k in range(1, max_delay + 1):
            target = u[warmup - k:n_steps - k]
            state_slice = states[warmup:n_steps]

            if len(target) < 10:
                continue

            # Ridge regression readout
            alpha = 1e-4
            W_out = np.linalg.solve(
                state_slice.T @ state_slice + alpha * np.eye(n),
                state_slice.T @ target
            )
            prediction = state_slice @ W_out

            # Squared correlation
            corr = np.corrcoef(target, prediction)[0, 1]
            mc += corr ** 2

        return float(mc)


# ============================================================================
# QUINE
# ============================================================================

class Quine:
    """
    Quine: a program that outputs its own source code.

    Demonstrates self-reference and fixed-point constructions.
    The quine theorem (Kleene's fixed-point theorem) guarantees that
    for any computable transformation t, there exists a program p
    such that p outputs t(source(p)).
    """

    @staticmethod
    def generate(language: str = 'python') -> str:
        """
        Generate a quine in the specified language.

        Args:
            language: 'python', 'javascript', or 'c'.

        Returns:
            Source code of a quine in the given language.
        """
        if language == 'python':
            return Quine._python_quine()
        elif language == 'javascript':
            return Quine._javascript_quine()
        elif language == 'c':
            return Quine._c_quine()
        else:
            raise ValueError(f"Unknown language: {language}")

    @staticmethod
    def _python_quine() -> str:
        return 's=\'s=%r;print(s%%s)\';print(s%s)'

    @staticmethod
    def _javascript_quine() -> str:
        return '(function q(){console.log("("+q+")()")})()'

    @staticmethod
    def _c_quine() -> str:
        return (
            '#include<stdio.h>\n'
            'char*s="#include<stdio.h>%cchar*s=%c%s%c;%c'
            'int main(){printf(s,10,34,s,34,10);}%c";\\n'
            'int main(){printf(s,10,34,s,34,10);}'
        )

    @staticmethod
    def verify(source: str, language: str = 'python') -> bool:
        """
        Verify that a program is a quine (outputs its own source).

        Only works for Python quines in the current process.
        """
        if language != 'python':
            logger.warning("Can only verify Python quines")
            return False

        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            exec(source)
            output = captured.getvalue().rstrip('\n')
            return output == source
        except Exception as e:
            logger.warning(f"Quine verification failed: {e}")
            return False
        finally:
            sys.stdout = old_stdout


# ============================================================================
# REFLECTIVE TOWER
# ============================================================================

class ReflectiveTower:
    """
    Reflective tower: a metacircular evaluator that can reflect on its
    own evaluation process at multiple levels.

    Level 0: Object language — evaluate expressions
    Level 1: Meta language — evaluate the evaluator
    Level 2: Meta-meta — evaluate the meta-evaluator
    ...

    Each level has access to the representation of the level below.
    """

    def __init__(self):
        self._env: dict[str, Any] = {}

    def evaluate(self, expr: Any, level: int = 0) -> Any:
        """
        Evaluate an expression at a given reflective level.

        Level 0: Direct evaluation of S-expressions.
        Level 1+: The evaluator itself becomes data that can be inspected.

        Args:
            expr: Expression to evaluate. Supports:
                  - Numbers (return as-is)
                  - Strings (variable lookup)
                  - Lists ['op', args...]:
                    ['quote', x] -> x
                    ['if', cond, then, else]
                    ['lambda', params, body]
                    ['define', name, value]
                    ['add', a, b], ['mul', a, b], etc.
                    ['reflect'] -> returns the evaluator as data
            level: Reflective level (0 = object, 1+ = meta).

        Returns:
            Evaluated result.
        """
        if level > 0:
            # At higher levels, we evaluate the evaluator applied to expr
            # This creates a reification: the evaluator becomes data
            meta_expr = ['apply-evaluator', expr, level - 1]
            return self._eval_meta(meta_expr, level)

        return self._eval(expr)

    def _eval(self, expr: Any) -> Any:
        """Level-0 evaluator."""
        # Atoms
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, str):
            if expr in self._env:
                return self._env[expr]
            raise NameError(f"Undefined: {expr}")

        # Lists (compound expressions)
        if not isinstance(expr, list) or len(expr) == 0:
            return expr

        op = expr[0]

        if op == 'quote':
            return expr[1] if len(expr) > 1 else None

        if op == 'if':
            cond = self._eval(expr[1])
            if cond:
                return self._eval(expr[2])
            return self._eval(expr[3]) if len(expr) > 3 else None

        if op == 'define':
            name = expr[1]
            value = self._eval(expr[2])
            self._env[name] = value
            return value

        if op == 'lambda':
            params = expr[1]
            body = expr[2]
            return ('closure', params, body, dict(self._env))

        if op == 'reflect':
            # Return a representation of the evaluator itself
            return {
                'type': 'evaluator',
                'level': 0,
                'environment': dict(self._env),
                'operations': ['quote', 'if', 'define', 'lambda',
                               'add', 'sub', 'mul', 'reflect'],
            }

        # Arithmetic
        if op == 'add':
            return self._eval(expr[1]) + self._eval(expr[2])
        if op == 'sub':
            return self._eval(expr[1]) - self._eval(expr[2])
        if op == 'mul':
            return self._eval(expr[1]) * self._eval(expr[2])
        if op == 'div':
            b = self._eval(expr[2])
            return self._eval(expr[1]) / b if abs(b) > 1e-10 else 0

        # Function application
        fn = self._eval(op)
        if isinstance(fn, tuple) and fn[0] == 'closure':
            _, params, body, closure_env = fn
            args = [self._eval(a) for a in expr[1:]]
            old_env = dict(self._env)
            self._env.update(closure_env)
            for p, a in zip(params, args):
                self._env[p] = a
            result = self._eval(body)
            self._env = old_env
            return result

        raise ValueError(f"Unknown expression: {expr}")

    def _eval_meta(self, expr: Any, level: int) -> Any:
        """Meta-level evaluation: the evaluator is reified as data."""
        if isinstance(expr, list) and expr[0] == 'apply-evaluator':
            inner_expr = expr[1]
            target_level = expr[2]

            # Reify: capture the evaluation trace
            trace = {
                'expression': inner_expr,
                'level': level,
                'target_level': target_level,
            }

            result = self.evaluate(inner_expr, target_level)
            trace['result'] = result

            return trace

        return self._eval(expr)


# ============================================================================
# OPEN-ENDED EVOLUTION (Novelty Search)
# ============================================================================

class OpenEndedEvolution:
    """
    Open-ended evolution via novelty search.

    Instead of maximizing fitness, novelty search rewards behavioral
    novelty: how different an individual's behavior is from the archive
    of previously seen behaviors.

    ρ(x) = (1/k) Σ d(x, μ_i)  (average distance to k nearest behaviors)
    """

    @staticmethod
    def step(population: list[np.ndarray],
             novelty_archive: list[np.ndarray],
             behavior_fn: Callable[[np.ndarray], np.ndarray],
             mutation_fn: Callable[[np.ndarray], np.ndarray],
             k: int = 15,
             archive_threshold: float = 0.1,
             pop_size: int | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        One generation of novelty search.

        Args:
            population: Current population of genomes.
            novelty_archive: Archive of novel behaviors.
            behavior_fn: Maps genome -> behavior descriptor (feature vector).
            mutation_fn: Maps genome -> mutated genome.
            k: Number of nearest neighbors for novelty computation.
            archive_threshold: Minimum novelty to add to archive.
            pop_size: Target population size (defaults to current).

        Returns:
            (new_population, updated_archive).
        """
        if pop_size is None:
            pop_size = len(population)

        # Compute behaviors
        behaviors = []
        for genome in population:
            try:
                b = behavior_fn(genome)
            except Exception:
                b = np.zeros(1)
            behaviors.append(b)

        # Compute novelty scores
        all_behaviors = behaviors + novelty_archive
        novelty_scores = []

        for b in behaviors:
            if len(all_behaviors) <= k:
                score = float(np.mean([
                    np.linalg.norm(b - other)
                    for other in all_behaviors
                    if not np.array_equal(b, other)
                ]) if len(all_behaviors) > 1 else 1.0)
            else:
                distances = np.array([
                    np.linalg.norm(b - other) for other in all_behaviors
                ])
                distances.sort()
                # Skip distance 0 (self)
                knn_dists = distances[1:k + 1] if len(distances) > k else distances[1:]
                score = float(np.mean(knn_dists)) if len(knn_dists) > 0 else 0.0
            novelty_scores.append(score)

        # Update archive
        updated_archive = list(novelty_archive)
        for i, score in enumerate(novelty_scores):
            if score > archive_threshold:
                updated_archive.append(behaviors[i])

        # Selection: tournament based on novelty
        new_population = []

        # Elitism: keep most novel
        best_idx = np.argmax(novelty_scores)
        new_population.append(population[best_idx].copy())

        while len(new_population) < pop_size:
            # Tournament selection
            candidates = np.random.choice(
                len(population),
                min(5, len(population)),
                replace=False
            )
            winner = candidates[
                np.argmax([novelty_scores[c] for c in candidates])
            ]

            # Mutate
            child = mutation_fn(population[winner].copy())
            new_population.append(child)

        return new_population[:pop_size], updated_archive

    @staticmethod
    def novelty_score(behavior: np.ndarray,
                      archive: list[np.ndarray],
                      k: int = 15) -> float:
        """Compute novelty score for a single behavior."""
        if not archive:
            return float('inf')

        distances = np.array([
            np.linalg.norm(behavior - other) for other in archive
        ])
        distances.sort()
        knn = distances[:min(k, len(distances))]
        return float(np.mean(knn)) if len(knn) > 0 else 0.0
