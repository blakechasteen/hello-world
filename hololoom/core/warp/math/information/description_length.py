"""
Minimum Description Length — Model Selection via Compression
=============================================================

MDL formalizes Occam's razor: the best model is the one that provides
the shortest total description of the data.

Total cost = L(H) + L(D|H)
           = model complexity + data encoded given model

Core insight: compression IS learning. A model that compresses data well
has captured its regularities.

Classes:
    MDLResult: Description length breakdown (model + data cost)
    MinimumDescriptionLength: Two-part MDL, NCD, Kolmogorov structure function
    StochasticComplexity: Normalized maximum likelihood (NML)
    PrequentialMDL: Online/predictive MDL for streaming data

Mathematical Foundation:
    Two-part MDL: min_H { L(H) + L(D|H) }
    NCD: (C(xy) - min(C(x),C(y))) / max(C(x),C(y))
    Stochastic complexity: log sum_theta p(D|theta)
    Prequential: sum_t log p(x_t | x_1, ..., x_{t-1})

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from .information_theory import entropy
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _compress_length(data_bytes: bytes) -> int:
    """Approximate Kolmogorov complexity via zlib compression length."""
    return len(zlib.compress(data_bytes, level=9))


def _to_bytes(data) -> bytes:
    """Convert data to bytes for compression-based measures."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, np.ndarray):
        return data.tobytes()
    if isinstance(data, str):
        return data.encode('utf-8')
    return bytes(str(data), 'utf-8')


@dataclass
class MDLResult:
    """Description length breakdown for a model.

    Attributes:
        description_length: Total L(H) + L(D|H).
        model_cost: L(H) — bits to describe the model.
        data_cost: L(D|H) — bits to describe data given model.
        optimal_k: Index of the optimal model (for multi-model comparison).
    """
    description_length: float
    model_cost: float
    data_cost: float
    optimal_k: int


class MinimumDescriptionLength:
    """Two-part MDL and compression-based model selection.

    The fundamental principle: among competing models, choose the one
    whose total description (model + data-given-model) is shortest.

    This implements:
    - Two-part MDL for explicit model comparison
    - Normalized Compression Distance (NCD) for similarity
    - Kolmogorov structure function for complexity analysis
    """

    @staticmethod
    def two_part(data: np.ndarray, model_complexities: np.ndarray,
                 data_costs: np.ndarray) -> MDLResult:
        """Two-part MDL: L(H) + L(D|H), find model minimizing total.

        Given k candidate models with known complexities and data-fit costs,
        select the one minimizing total description length.

        Args:
            data: The observed data (used for metadata, not recomputed).
            model_complexities: Array of L(H_k) for each model k.
                Each entry is the cost in bits to describe model k.
            data_costs: Array of L(D|H_k) for each model k.
                Each entry is the cost in bits to encode data given model k.

        Returns:
            MDLResult with the optimal model's breakdown.

        Example:
            # Compare polynomial fits of degree 1, 2, 3
            complexities = np.array([2.0, 4.0, 6.0])  # more params = more bits
            data_costs = np.array([50.0, 20.0, 19.5])  # better fit = fewer bits
            result = MinimumDescriptionLength.two_part(data, complexities, data_costs)
            # result.optimal_k = 1 (degree 2 — good fit without overfitting)
        """
        model_complexities = np.asarray(model_complexities, dtype=np.float64)
        data_costs = np.asarray(data_costs, dtype=np.float64)

        total = model_complexities + data_costs
        optimal_k = int(np.argmin(total))

        return MDLResult(
            description_length=float(total[optimal_k]),
            model_cost=float(model_complexities[optimal_k]),
            data_cost=float(data_costs[optimal_k]),
            optimal_k=optimal_k,
        )

    @staticmethod
    def normalized_compression_distance(x, y) -> float:
        """Normalized Compression Distance (NCD).

        NCD approximates normalized information distance using a real
        compressor. Values near 0 mean similar; near 1 means dissimilar.

        NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

        where C(.) is the compressed length.

        Args:
            x: First object (bytes, str, or np.ndarray).
            y: Second object (bytes, str, or np.ndarray).

        Returns:
            NCD in approximately [0, 1]. Values slightly above 1 are
            possible due to compressor imperfections.
        """
        x_bytes = _to_bytes(x)
        y_bytes = _to_bytes(y)

        c_x = _compress_length(x_bytes)
        c_y = _compress_length(y_bytes)
        c_xy = _compress_length(x_bytes + y_bytes)

        denominator = max(c_x, c_y)
        if denominator == 0:
            return 0.0

        ncd = (c_xy - min(c_x, c_y)) / denominator
        return float(ncd)

    @staticmethod
    def kolmogorov_structure_function(data: np.ndarray,
                                       k_range: list[int] | None = None
                                       ) -> list[tuple[int, float]]:
        """Kolmogorov structure function: model complexity k vs minimum data cost.

        For each model complexity level k, estimates the minimum data cost
        achievable. The resulting curve characterizes the data's structure:
        - Steep drop: data has strong regularity at that complexity
        - Flat region: additional model complexity doesn't help
        - The "elbow" suggests optimal model complexity

        Approximated by fitting polynomial models of increasing degree
        and measuring residual description length.

        Args:
            data: 1D array of observed values.
            k_range: List of model complexities (polynomial degrees) to test.
                Defaults to [1, 2, ..., min(15, N//2)].

        Returns:
            [(k, data_cost)] — model complexity vs data description cost.
        """
        data = np.asarray(data, dtype=np.float64)
        N = len(data)

        if k_range is None:
            max_k = min(15, max(1, N // 2))
            k_range = list(range(1, max_k + 1))

        x = np.arange(N, dtype=np.float64)
        results = []

        for k in k_range:
            if k >= N:
                break

            # Fit polynomial of degree k-1 (k parameters)
            try:
                coeffs = np.polyfit(x, data, deg=min(k - 1, N - 1))
                fitted = np.polyval(coeffs, x)
                residuals = data - fitted

                # Data cost: encode residuals
                # Use sum of squared residuals as proxy for log-likelihood cost
                ssr = np.sum(residuals ** 2)
                if ssr < 1e-15:
                    # Perfect fit
                    data_cost = 0.0
                else:
                    # Gaussian assumption: L(D|H) = (N/2) * log(SSR/N) + N/2
                    variance_est = ssr / N
                    data_cost = (N / 2.0) * np.log2(max(variance_est, 1e-15)) + N / 2.0

                results.append((k, float(data_cost)))
            except (np.linalg.LinAlgError, ValueError):
                logger.debug(f"Polyfit failed for k={k}, skipping")
                continue

        return results


class StochasticComplexity:
    """Stochastic complexity / Normalized Maximum Likelihood (NML).

    The stochastic complexity is the shortest description length achievable
    by any model in the class, without knowing the true parameter values
    in advance. It penalizes model complexity via the parametric complexity
    term — no explicit penalty needed.

    This is the "refined" MDL approach, as opposed to two-part MDL.
    """

    @staticmethod
    def compute(data: np.ndarray, log_likelihood_fn: Callable,
                param_dim: int, n_grid: int = 50) -> float:
        """Stochastic complexity via NML approximation.

        The NML distribution is:
            p_NML(x) = p(x | theta_hat(x)) / C_n

        where C_n = integral of p(x | theta_hat(x)) over all possible data x.
        We approximate via grid integration over the parameter space.

        Args:
            data: Observed data array.
            log_likelihood_fn: Function(data, theta) -> log p(data | theta).
                theta is a 1D array of param_dim parameters.
            param_dim: Dimensionality of the parameter space.
            n_grid: Grid points per dimension for numerical integration.

        Returns:
            Stochastic complexity (in nats).
        """
        # Compute maximum log-likelihood for observed data
        # We approximate by evaluating on a grid and taking the max
        N = len(data)

        # Generate parameter grid
        if param_dim == 1:
            # 1D grid centered around data statistics
            theta_grid = np.linspace(
                np.min(data) - np.std(data),
                np.max(data) + np.std(data),
                n_grid
            ).reshape(-1, 1)
        else:
            # Multi-dimensional: use random sampling
            rng = np.random.RandomState(42)
            data_scale = max(np.std(data), 1e-6)
            theta_grid = rng.randn(n_grid ** min(param_dim, 3), param_dim) * data_scale

        # Find MLE for observed data
        best_ll = -np.inf
        for theta in theta_grid:
            try:
                ll = log_likelihood_fn(data, theta)
                if ll > best_ll:
                    best_ll = ll
            except (ValueError, RuntimeError, FloatingPointError):
                continue

        # Approximate parametric complexity using Rissanen's formula
        comp = StochasticComplexity.parametric_complexity(N, param_dim)

        # Stochastic complexity = -max_log_likelihood + parametric_complexity
        return float(-best_ll + comp)

    @staticmethod
    def parametric_complexity(n: int, k: int) -> float:
        """Rissanen's parametric complexity.

        COMP(k, n) = (k/2) * log(n / (2*pi)) + log(pi^{k/2} / Gamma(k/2))

        This measures the "richness" of a model class — how many distinct
        distributions it can express with n data points.

        Args:
            n: Number of data points.
            k: Number of parameters (model dimensionality).

        Returns:
            Parametric complexity in nats.
        """
        if k <= 0 or n <= 0:
            return 0.0

        # (k/2) * log(n / (2*pi))
        term1 = (k / 2.0) * np.log(max(n / (2.0 * np.pi), 1e-15))

        # log(pi^{k/2} / Gamma(k/2))
        # log(pi^{k/2}) = (k/2) * log(pi)
        log_pi_term = (k / 2.0) * np.log(np.pi)

        # log(Gamma(k/2)) via Stirling for large k, exact for small k
        half_k = k / 2.0
        if half_k <= 0:
            log_gamma = 0.0
        elif half_k == int(half_k) and half_k <= 20:
            # Exact: Gamma(n) = (n-1)!
            from math import lgamma
            log_gamma = lgamma(half_k)
        else:
            from math import lgamma
            log_gamma = lgamma(half_k)

        term2 = log_pi_term - log_gamma

        return float(term1 + term2)


class PrequentialMDL:
    """Prequential (predictive) MDL for online model selection.

    Instead of two-part coding, uses the prequential plug-in code:
        L(x_1, ..., x_n) = -sum_{t=1}^{n} log p(x_t | x_1, ..., x_{t-1}, H)

    This is the cumulative log-loss of sequential prediction. The model
    that predicts best (lowest cumulative log-loss) is preferred.

    Advantages over two-part MDL:
    - No need to separately encode the model
    - Works naturally with streaming data
    - Minimax optimal asymptotically
    """

    @staticmethod
    def online(data_stream: np.ndarray,
               models: list[Callable]) -> list[MDLResult]:
        """Prequential MDL for multiple competing models.

        Each model is a callable: model(history, x_next) -> log p(x_next | history).
        Models are evaluated on their cumulative predictive log-loss.

        Args:
            data_stream: Sequential observations, 1D array.
            models: List of model callables. Each takes:
                - history: np.ndarray of past observations
                - x_next: float, the next observation
                Returns: log probability (base 2) of x_next given history.

        Returns:
            List of MDLResult, one per model, sorted by description_length.
        """
        n = len(data_stream)
        n_models = len(models)
        cumulative_loss = np.zeros(n_models)

        for t in range(1, n):
            history = data_stream[:t]
            x_next = data_stream[t]

            for m_idx, model in enumerate(models):
                try:
                    log_prob = model(history, x_next)
                    # log_prob should be negative (log of probability)
                    cumulative_loss[m_idx] -= log_prob
                except (ValueError, RuntimeError, FloatingPointError):
                    # Penalize failures heavily
                    cumulative_loss[m_idx] += 10.0

        results = []
        for m_idx in range(n_models):
            # Model cost is implicit in prequential MDL (no separate term)
            # We report cumulative log-loss as the total description length
            results.append(MDLResult(
                description_length=float(cumulative_loss[m_idx]),
                model_cost=0.0,  # Absorbed into predictions
                data_cost=float(cumulative_loss[m_idx]),
                optimal_k=m_idx,
            ))

        # Sort by description length
        results.sort(key=lambda r: r.description_length)
        # Mark the best model
        if results:
            results[0] = MDLResult(
                description_length=results[0].description_length,
                model_cost=results[0].model_cost,
                data_cost=results[0].data_cost,
                optimal_k=results[0].optimal_k,
            )

        return results

    @staticmethod
    def cumulative_code_length(data: np.ndarray,
                                model_fn: Callable) -> float:
        """Cumulative code length for a single model.

        Computes the total prequential code length:
            L = -sum_{t=1}^{n} log2 p(x_t | x_1, ..., x_{t-1})

        Args:
            data: Sequential observations, 1D array.
            model_fn: Callable(history, x_next) -> log2 p(x_next | history).
                Must return log probability in base 2.

        Returns:
            Total code length in bits.
        """
        total_length = 0.0

        for t in range(1, len(data)):
            history = data[:t]
            x_next = data[t]

            try:
                log_prob = model_fn(history, x_next)
                total_length -= log_prob
            except (ValueError, RuntimeError, FloatingPointError):
                # Unknown observation — assign maximum cost
                total_length += 10.0

        return float(total_length)


__all__ = [
    'MDLResult',
    'MinimumDescriptionLength',
    'StochasticComplexity',
    'PrequentialMDL',
]
