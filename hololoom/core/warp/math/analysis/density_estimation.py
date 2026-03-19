"""
Density Estimation — ST6.5
===========================

Non-parametric estimation of probability density functions from data.
Three approaches with increasing sophistication: histograms (binning),
kernel density estimation (local smoothing), and wavelet density
estimation (multi-resolution).

Classes:
    HistogramEstimator: Adaptive bin-width selection (Freedman-Diaconis)
    KDEEstimator: Kernel density estimation with bandwidth selection
    WaveletDensity: Wavelet shrinkage density estimation

Mathematical Foundation:
    Histogram: f_hat(x) = n_j / (n * h) for x in bin j
    KDE: f_hat(x) = (1/nh) sum K((x - x_i)/h) with kernel K
    Bandwidth selection:
        - Silverman: h = 0.9 * min(sigma, IQR/1.34) * n^{-1/5}
        - Freedman-Diaconis: bin_width = 2 * IQR * n^{-1/3}
    Wavelet: threshold empirical wavelet coefficients for denoising

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DensityResult:
    """
    Result of density estimation.

    Attributes:
        grid: Evaluation points, shape (n_grid,)
        density: Estimated density at grid points, shape (n_grid,)
        bandwidth: Bandwidth/bin-width used
        method: Estimation method name
    """
    grid: np.ndarray
    density: np.ndarray
    bandwidth: float
    method: str

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Interpolate density at arbitrary points."""
        return np.interp(x, self.grid, self.density)

    def cdf(self) -> np.ndarray:
        """Cumulative distribution function via trapezoidal integration."""
        dx = np.diff(self.grid)
        cumulative = np.zeros_like(self.density)
        cumulative[1:] = np.cumsum(0.5 * (self.density[:-1] + self.density[1:]) * dx)
        # Normalize
        if cumulative[-1] > 0:
            cumulative /= cumulative[-1]
        return cumulative

    def mode(self) -> float:
        """Location of maximum density."""
        return float(self.grid[np.argmax(self.density)])

    def entropy(self) -> float:
        """Differential entropy H = -integral f(x) log f(x) dx."""
        dx = np.diff(self.grid).mean()
        mask = self.density > 1e-10
        return -float(np.sum(self.density[mask] * np.log(self.density[mask]) * dx))


class HistogramEstimator:
    """
    Histogram density estimation with automatic bin selection.

    Bin width methods:
        - Freedman-Diaconis: h = 2 * IQR * n^{-1/3} (robust to outliers)
        - Sturges: k = 1 + log2(n) bins (Gaussian-optimal)
        - Scott: h = 3.49 * sigma * n^{-1/3} (Gaussian-optimal width)
    """

    def fit(
        self,
        data: np.ndarray,
        n_bins: str = "auto",
        range: tuple[float, float] | None = None,
        n_grid: int = 200,
    ) -> DensityResult:
        """
        Fit histogram density estimator.

        Args:
            data: Sample data, shape (n,)
            n_bins: Number of bins or method ("auto", "freedman", "scott", "sturges")
                    or an integer for explicit count
            range: (min, max) range for histogram. Default: data range.
            n_grid: Number of grid points for output

        Returns:
            DensityResult with density estimate on uniform grid
        """
        data = np.asarray(data, dtype=float).ravel()
        n = len(data)

        if range is None:
            pad = 0.1 * (data.max() - data.min() + 1e-10)
            range = (data.min() - pad, data.max() + pad)

        # Determine bin width
        if isinstance(n_bins, str):
            h = self._auto_bandwidth(data, n_bins)
            n_bins_int = max(1, int(np.ceil((range[1] - range[0]) / h)))
        else:
            n_bins_int = int(n_bins)
            h = (range[1] - range[0]) / n_bins_int

        # Compute histogram
        counts, bin_edges = np.histogram(data, bins=n_bins_int, range=range, density=True)

        # Interpolate to uniform grid
        grid = np.linspace(range[0], range[1], n_grid)
        density = np.zeros(n_grid)
        for i in range(len(counts)):
            mask = (grid >= bin_edges[i]) & (grid < bin_edges[i + 1])
            density[mask] = counts[i]
        # Handle last edge
        density[grid == bin_edges[-1]] = counts[-1] if len(counts) > 0 else 0

        return DensityResult(grid=grid, density=density, bandwidth=h, method="histogram")

    def _auto_bandwidth(self, data: np.ndarray, method: str) -> float:
        """Compute optimal bin width."""
        n = len(data)
        sigma = np.std(data, ddof=1)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)

        if method in ("auto", "freedman"):
            # Freedman-Diaconis
            if iqr > 0:
                return 2.0 * iqr * n ** (-1.0 / 3)
            else:
                return 3.49 * sigma * n ** (-1.0 / 3)

        elif method == "scott":
            return 3.49 * sigma * n ** (-1.0 / 3)

        elif method == "sturges":
            n_bins = int(np.ceil(1 + np.log2(n)))
            return (data.max() - data.min()) / max(n_bins, 1)

        raise ValueError(f"Unknown method: {method}")


class KDEEstimator:
    """
    Kernel Density Estimation.

    f_hat(x) = (1/nh) sum_{i=1}^n K((x - x_i) / h)

    Kernels:
        - Gaussian: K(u) = (1/sqrt(2*pi)) exp(-u^2/2)
        - Epanechnikov: K(u) = 3/4 (1 - u^2) for |u| <= 1 (optimal MSE)
        - Triangular: K(u) = (1 - |u|) for |u| <= 1

    Bandwidth selection:
        - Silverman's rule: h = 0.9 * min(sigma, IQR/1.34) * n^{-1/5}
        - Scott's rule: h = 1.06 * sigma * n^{-1/5}
    """

    KERNELS = {
        "gaussian": lambda u: np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi),
        "epanechnikov": lambda u: np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0),
        "triangular": lambda u: np.where(np.abs(u) <= 1, 1.0 - np.abs(u), 0.0),
        "uniform": lambda u: np.where(np.abs(u) <= 1, 0.5, 0.0),
    }

    def fit(
        self,
        data: np.ndarray,
        bandwidth: str = "silverman",
        kernel: str = "gaussian",
        n_grid: int = 200,
        range: tuple[float, float] | None = None,
    ) -> DensityResult:
        """
        Fit KDE to data.

        Args:
            data: Sample data, shape (n,)
            bandwidth: Bandwidth value or method ("silverman", "scott", or float)
            kernel: Kernel function name
            n_grid: Number of grid points
            range: Evaluation range (default: data range + 3*bandwidth)

        Returns:
            DensityResult with smooth density estimate
        """
        data = np.asarray(data, dtype=float).ravel()
        n = len(data)

        # Select kernel
        if kernel not in self.KERNELS:
            raise ValueError(f"Unknown kernel: {kernel}. Choose from {list(self.KERNELS.keys())}")
        K = self.KERNELS[kernel]

        # Select bandwidth
        if isinstance(bandwidth, str):
            h = self._bandwidth_select(data, bandwidth)
        else:
            h = float(bandwidth)

        # Evaluation grid
        if range is None:
            pad = 3 * h
            range = (data.min() - pad, data.max() + pad)

        grid = np.linspace(range[0], range[1], n_grid)

        # Evaluate KDE
        density = np.zeros(n_grid)
        for xi in data:
            u = (grid - xi) / h
            density += K(u)
        density /= (n * h)

        return DensityResult(grid=grid, density=density, bandwidth=h, method=f"kde_{kernel}")

    def _bandwidth_select(self, data: np.ndarray, method: str) -> float:
        """Select bandwidth by rule of thumb."""
        n = len(data)
        sigma = np.std(data, ddof=1)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)

        if method == "silverman":
            A = min(sigma, iqr / 1.349) if iqr > 0 else sigma
            return 0.9 * A * n ** (-0.2)

        elif method == "scott":
            return 1.06 * sigma * n ** (-0.2)

        raise ValueError(f"Unknown bandwidth method: {method}")

    def fit_2d(
        self,
        data: np.ndarray,
        bandwidth: float | None = None,
        n_grid: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D kernel density estimation with Gaussian kernel.

        Args:
            data: Sample data, shape (n, 2)
            bandwidth: Isotropic bandwidth (default: Silverman 2D rule)
            n_grid: Grid points per dimension

        Returns:
            (X_grid, Y_grid, density) meshgrid arrays
        """
        data = np.asarray(data, dtype=float)
        n = data.shape[0]

        if bandwidth is None:
            # Silverman 2D rule
            sigma = data.std(axis=0)
            h = np.mean(sigma) * n ** (-1.0 / 6)
        else:
            h = bandwidth

        pad = 3 * h
        x_range = (data[:, 0].min() - pad, data[:, 0].max() + pad)
        y_range = (data[:, 1].min() - pad, data[:, 1].max() + pad)

        x_grid = np.linspace(x_range[0], x_range[1], n_grid)
        y_grid = np.linspace(y_range[0], y_range[1], n_grid)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.stack([X.ravel(), Y.ravel()], axis=1)

        density = np.zeros(positions.shape[0])
        for xi in data:
            diff = positions - xi
            density += np.exp(-0.5 * np.sum(diff ** 2, axis=1) / h ** 2)
        density /= (n * 2 * np.pi * h ** 2)

        return X, Y, density.reshape(n_grid, n_grid)


class WaveletDensity:
    """
    Wavelet density estimation via coefficient shrinkage.

    Estimates density by:
    1. Compute a histogram at fine resolution
    2. Apply wavelet transform (Haar or similar)
    3. Threshold small coefficients (denoising)
    4. Inverse transform to get smooth density

    Adapts to spatially varying smoothness — sharp peaks stay sharp,
    smooth regions stay smooth.
    """

    def fit(
        self,
        data: np.ndarray,
        wavelet: str = "haar",
        level: int = 3,
        threshold: str = "universal",
        n_grid: int = 256,
        range: tuple[float, float] | None = None,
    ) -> DensityResult:
        """
        Fit wavelet density estimator.

        Args:
            data: Sample data, shape (n,)
            wavelet: Wavelet type ("haar" supported)
            level: Decomposition level
            threshold: Thresholding method ("universal" or "sure")
            n_grid: Number of grid points (should be power of 2)
            range: Evaluation range

        Returns:
            DensityResult with wavelet-smoothed density
        """
        data = np.asarray(data, dtype=float).ravel()
        n = len(data)

        # Ensure n_grid is power of 2
        n_grid = 2 ** int(np.ceil(np.log2(n_grid)))

        if range is None:
            pad = 0.1 * (data.max() - data.min() + 1e-10)
            range = (data.min() - pad, data.max() + pad)

        grid = np.linspace(range[0], range[1], n_grid)
        h = (range[1] - range[0]) / n_grid

        # Fine histogram as starting point
        counts, _ = np.histogram(data, bins=n_grid, range=range)
        density = counts.astype(float) / (n * h)

        # Haar wavelet transform
        coeffs = self._haar_transform(density, level)

        # Threshold
        sigma = np.median(np.abs(coeffs)) / 0.6745  # MAD estimate
        if threshold == "universal":
            thresh = sigma * np.sqrt(2 * np.log(n_grid))
        else:
            thresh = sigma * np.sqrt(2 * np.log(n_grid))

        # Soft thresholding
        coeffs = np.sign(coeffs) * np.maximum(np.abs(coeffs) - thresh, 0)

        # Inverse transform
        density_smooth = self._haar_inverse(coeffs, level)

        # Ensure non-negativity and normalization
        density_smooth = np.maximum(density_smooth, 0)
        total = np.trapz(density_smooth, grid)
        if total > 0:
            density_smooth /= total

        bw = h * (2 ** level)  # Effective bandwidth

        return DensityResult(grid=grid, density=density_smooth, bandwidth=bw, method=f"wavelet_{wavelet}")

    def _haar_transform(self, signal: np.ndarray, level: int) -> np.ndarray:
        """Forward Haar wavelet transform."""
        result = signal.copy()
        n = len(result)

        for _ in range(level):
            if n < 2:
                break
            temp = np.zeros(n)
            half = n // 2

            for i in range(half):
                temp[i] = (result[2 * i] + result[2 * i + 1]) / np.sqrt(2)
                temp[half + i] = (result[2 * i] - result[2 * i + 1]) / np.sqrt(2)

            result[:n] = temp
            n = half

        return result

    def _haar_inverse(self, coeffs: np.ndarray, level: int) -> np.ndarray:
        """Inverse Haar wavelet transform."""
        result = coeffs.copy()
        total_n = len(result)
        n = total_n >> level

        for _ in range(level):
            n *= 2
            if n > total_n:
                break

            half = n // 2
            temp = np.zeros(n)

            for i in range(half):
                temp[2 * i] = (result[i] + result[half + i]) / np.sqrt(2)
                temp[2 * i + 1] = (result[i] - result[half + i]) / np.sqrt(2)

            result[:n] = temp

        return result
