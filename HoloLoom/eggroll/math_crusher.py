import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass

# ==========================================
# 1. Hyperbolic Geometry (Poincaré Ball)
# ==========================================

class HyperbolicGeometry:
    """
    Operations in the Poincaré Ball Model of Hyperbolic Geometry.
    Useful for representing hierarchical structures (entailment, taxonomy).
    """
    
    @staticmethod
    def exp_map(v: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Exponential map: Euclidean -> Hyperbolic (Poincaré Ball).
        Maps a vector v in the tangent space at origin to the manifold.
        """
        norm_v = np.linalg.norm(v) + 1e-9
        sqrt_c = np.sqrt(c)
        coeff = np.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
        return coeff * v

    @staticmethod
    def poincare_distance(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> float:
        """
        Distance in the Poincaré Ball.
        d(u, v) = arccosh(1 + 2 * ||u-v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
        """
        sq_norm_u = np.sum(u ** 2)
        sq_norm_v = np.sum(v ** 2)
        sq_dist = np.sum((u - v) ** 2)
        
        # Clip to avoid numerical instability near boundary
        sq_norm_u = np.clip(sq_norm_u, 0, 1 - 1e-5)
        sq_norm_v = np.clip(sq_norm_v, 0, 1 - 1e-5)
        
        val = 1 + 2 * c * sq_dist / ((1 - c * sq_norm_u) * (1 - c * sq_norm_v))
        return np.arccosh(np.maximum(val, 1.0 + 1e-7)) / np.sqrt(c)

# ==========================================
# 2. Topology (Persistent Homology)
# ==========================================

class TopologicalFeatures:
    """
    Topological Data Analysis (TDA) utilities.
    Computes Betti numbers (holes) from point clouds using a simplified filtration.
    """
    
    @staticmethod
    def compute_betti_0(point_cloud: np.ndarray, epsilon: float) -> int:
        """
        Compute Betti-0 (Connected Components) at scale epsilon.
        Uses a simple Union-Find (Disjoint Set) approach on the Rips graph.
        """
        n = len(point_cloud)
        if n == 0: return 0
        
        parent = list(range(n))
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False
            
        # Build edges < epsilon
        components = n
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
                if dist < epsilon:
                    if union(i, j):
                        components -= 1
                        
        return components

# ==========================================
# 3. Hyperdimensional Computing (VSA)
# ==========================================

class HyperdimensionalComputing:
    """
    Vector Symbolic Architecture (VSA) operations.
    Algebra of high-dimensional vectors (MAP: Multiply, Add, Permute).
    """
    
    @staticmethod
    def bind(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Binding operation (Element-wise XOR for binary, or Circular Convolution).
        Here we use element-wise multiplication for real-valued vectors (Holographic Reduced Representations).
        """
        return u * v

    @staticmethod
    def bundle(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundling (Superposition) operation.
        Element-wise sum, usually followed by normalization.
        """
        if not vectors: return np.array([])
        sum_vec = np.sum(vectors, axis=0)
        return sum_vec / (np.linalg.norm(sum_vec) + 1e-9)

# ==========================================
# 4. Differential Evolution (Optimization)
# ==========================================

class DifferentialEvolution:
    """
    Differential Evolution strategy for continuous optimization.
    x_new = x_a + F * (x_b - x_c)
    """
    
    @staticmethod
    def mutate(population: Any, F: float = 0.8) -> Any:
        """
        Create a mutant.
        If population is a list of vectors, performs DE mutation: a + F*(b-c).
        If population is a scalar (float), adds Gaussian noise (Mutation).
        """
        if isinstance(population, (float, int)):
            # Scalar mutation
            return population * (1.0 + np.random.normal(0, 0.1))
            
        if isinstance(population, list) and len(population) >= 3:
            indices = np.random.choice(len(population), 3, replace=False)
            a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
            return a + F * (b - c)
            
        return population

# ==========================================
# 5. Category Theory (Functors)
# ==========================================

class Functor:
    """
    Abstract Base Class for Functors.
    Maps objects and morphisms from Category C to Category D.
    """
    def map_object(self, obj: Any) -> Any:
        raise NotImplementedError
    
    def map_morphism(self, f: Any) -> Any:
        raise NotImplementedError

@dataclass
class LinearMap:
    """A morphism in the category of Vector Spaces (Vect_K)."""
    matrix: np.ndarray
    
    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.matrix @ v

class TangentFunctor(Functor):
    """
    Simplified Tangent Functor T.
    Maps a manifold point (weights) to its tangent space (gradients).
    """
    def map_object(self, weights: np.ndarray) -> np.ndarray:
        # T(M) -> (p, v)
        # Here we just return the zero vector at that point (tangent origin)
        return np.zeros_like(weights)
        
    def map_morphism(self, update_step: LinearMap) -> LinearMap:
        # Maps a function f: M -> M to df: TM -> TM
        # For linear updates, the derivative is the map itself
        return update_step

# ==========================================
# 6. Calculus (Control Theory)
# ==========================================

class PIDController:
    """
    Proportional-Integral-Derivative Controller.
    Uses Calculus to regulate a process variable (e.g., Learning Rate).
    
    u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}
    """
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, measurement: float, dt: float = 1.0) -> float:
        """
        Compute control output.
        """
        error = self.setpoint - measurement
        
        # Integral term (Area under curve)
        self.integral += error * dt
        
        # Derivative term (Slope/Rate of change)
        derivative = (error - self.prev_error) / dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        return output

# ==========================================
# 7. Advanced Statistics (Information Geometry)
# ==========================================

class StatisticalMeasures:
    """
    Statistical distances and information theoretic measures.
    """
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Kullback-Leibler Divergence: D_KL(P || Q).
        Measure of information lost when Q is used to approximate P.
        Assumes p and q are probability distributions (sum to 1).
        """
        # Clip to avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return np.sum(p * np.log(p / q))

    @staticmethod
    def entropy(p: np.ndarray) -> float:
        """
        Shannon Entropy: H(P) = - sum p(x) log p(x).
        Measure of uncertainty/diversity.
        """
        p = np.clip(p, 1e-10, 1.0)
        return -np.sum(p * np.log(p))

    @staticmethod
    def wasserstein_1d(u_values: np.ndarray, v_values: np.ndarray) -> float:
        """
        Wasserstein-1 Distance (Earth Mover's Distance) for 1D distributions.
        """
        u_sorted = np.sort(u_values)
        v_sorted = np.sort(v_values)
        return np.mean(np.abs(u_sorted - v_sorted))

# ==========================================
# 8. Advanced Calculus (Signal Processing)
# ==========================================

class CalculusTools:
    """
    Advanced Calculus and Signal Processing tools.
    """
    
    @staticmethod
    def exponential_moving_average(current: float, history: float, alpha: float = 0.9) -> float:
        """
        EMA: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        Smoothing noisy signals (gradients, rewards).
        """
        return alpha * current + (1 - alpha) * history

    @staticmethod
    def compute_curvature(gradients: List[np.ndarray]) -> float:
        """
        Estimate curvature (2nd derivative approximation) from a sequence of gradients.
        High curvature -> unstable landscape -> lower learning rate.
        """
        if len(gradients) < 2: return 0.0
        
        # Secant equation approximation for 2nd derivative
        # f''(x) ~ (f'(x) - f'(x-1)) / dx
        # Here we just look at the change in gradient direction/magnitude
        g_t = gradients[-1]
        g_prev = gradients[-2]
        
        diff = np.linalg.norm(g_t - g_prev)
        return diff

    @staticmethod
    def detect_periodicity(signal: np.ndarray) -> float:
        """
        Use FFT to detect dominant frequencies in a signal.
        Useful for detecting loops or oscillations in agent behavior.
        """
        if len(signal) < 4: return 0.0
        
        # Fast Fourier Transform
        fft_vals = np.fft.rfft(signal)
        # Power spectrum
        power = np.abs(fft_vals)**2
        # Ignore DC component (0 Hz)
        power[0] = 0
        
        # Return peak power (strength of periodicity)
        return np.max(power)

# ==========================================
# 9. Cybernetics (System Regulation)
# ==========================================

class CyberneticRegulator:
    """
    Cybernetic mechanisms for self-regulation and homeostasis.
    Based on Ashby's Law of Requisite Variety and Wiener's Feedback.
    """
    
    def __init__(self, target_variety: float = 1.0):
        self.internal_state = 1.0 # "Energy" or "Capacity"
        self.target_variety = target_variety
        self.homeostasis_bounds = (0.5, 1.5)
        
    def regulate(self, disturbance_variety: float) -> float:
        """
        Apply Ashby's Law: Only variety can destroy variety.
        Adjust internal state to match disturbance.
        """
        # Calculate variety gap
        gap = disturbance_variety - self.internal_state
        
        # Negative Feedback Loop to restore balance
        # If disturbance > internal, we need to increase internal capacity
        response = 0.1 * gap
        self.internal_state += response
        
        # Homeostatic clamping
        lower, upper = self.homeostasis_bounds
        self.internal_state = max(lower, min(upper, self.internal_state))
        
        return self.internal_state

    def check_viability(self) -> bool:
        """
        Is the system within viable bounds?
        """
        lower, upper = self.homeostasis_bounds
        return lower <= self.internal_state <= upper

# ==========================================
# 10. Graph Theory (Network Analysis)
# ==========================================

class GraphTheory:
    """
    Graph algorithms for analyzing lineage and knowledge structures.
    """
    @staticmethod
    def eigenvector_centrality(adjacency_matrix: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """
        Compute Eigenvector Centrality (influence) of nodes.
        Ax = \lambda x
        """
        n = adjacency_matrix.shape[0]
        if n == 0: return np.array([])
        x = np.ones(n) / np.sqrt(n)
        
        for _ in range(max_iter):
            prev_x = x.copy()
            x = adjacency_matrix @ x
            norm = np.linalg.norm(x)
            if norm == 0: break
            x = x / norm
            if np.allclose(x, prev_x): break
            
        return x

    @staticmethod
    def spectral_clustering(adjacency_matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Partition graph into k clusters using Laplacian eigenvalues.
        """
        if adjacency_matrix.shape[0] == 0: return np.array([])
        
        # Degree matrix
        degrees = np.sum(adjacency_matrix, axis=1)
        D = np.diag(degrees)
        # Laplacian
        L = D - adjacency_matrix
        
        # Eigen decomposition
        vals, vecs = np.linalg.eigh(L)
        
        # Use k smallest eigenvectors
        return vecs[:, :k]

# ==========================================
# 11. Combinatorics (Discrete Structures)
# ==========================================

class Combinatorics:
    """
    Combinatorial utilities.
    """
    @staticmethod
    def catalan_number(n: int) -> int:
        """
        C_n = (2n)! / ((n+1)!n!)
        Counts valid parenthesis expressions, binary trees, etc.
        """
        if n == 0: return 1
        import math
        return math.comb(2*n, n) // (n + 1)

    @staticmethod
    def stirling_second(n: int, k: int) -> int:
        """
        Stirling numbers of the second kind S(n,k).
        Ways to partition a set of n elements into k non-empty subsets.
        """
        if k < 0 or k > n: return 0
        if k == 0: return 1 if n == 0 else 0
        if k == 1 or k == n: return 1
        
        # Iterative DP for efficiency
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
        return dp[n][k]

# ==========================================
# 12. Advanced Information Theory
# ==========================================

class InformationTheory:
    """
    Advanced Information Theoretic measures.
    """
    @staticmethod
    def mutual_information(p_xy: np.ndarray) -> float:
        """
        I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))
        p_xy: Joint probability matrix (X rows, Y cols)
        """
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        mi = 0.0
        rows, cols = p_xy.shape
        for i in range(rows):
            for j in range(cols):
                if p_xy[i, j] > 1e-10:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi

# ==========================================
# 13. Numerical Calculus (Integration/ODEs)
# ==========================================

class NumericalCalculus:
    """
    Numerical methods for Calculus.
    """
    @staticmethod
    def runge_kutta_4(f, y0: float, t0: float, dt: float, steps: int) -> List[float]:
        """
        RK4 method for solving ODE: dy/dt = f(t, y)
        """
        y = y0
        t = t0
        history = [y]
        
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
            k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
            k4 = f(t + dt, y + dt*k3)
            
            y += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            history.append(y)
            
        return history

# ==========================================
# 14. Riemannian Geometry (Manifolds)
# ==========================================

class RiemannianGeometry:
    """
    Operations on Riemannian Manifolds.
    """
    @staticmethod
    def geodesic_distance_sphere(u: np.ndarray, v: np.ndarray, r: float = 1.0) -> float:
        """
        Great-circle distance on a Hypersphere S^n with radius r.
        d(u, v) = r * arccos(<u, v> / (||u|| ||v||))
        """
        dot = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        # Clip for numerical stability
        cos_theta = np.clip(dot / (norm_u * norm_v), -1.0, 1.0)
        return r * np.arccos(cos_theta)

    @staticmethod
    def parallel_transport_sphere(v: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """
        Parallel transport vector v from start to end along geodesic on Sphere.
        Simplified: Project v onto tangent space of end.
        """
        # Tangent space projection at 'end': I - n n^T
        n = end / np.linalg.norm(end)
        projection = np.eye(len(end)) - np.outer(n, n)
        return projection @ v

# ==========================================
# 15. Advanced Matrix Operations (Tensor Algebra)
# ==========================================

class AdvancedMatrixOps:
    """
    High-performance matrix and tensor operations.
    """
    @staticmethod
    def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Kronecker product (Tensor product) of two matrices.
        Result size: (r1*r2, c1*c2)
        """
        return np.kron(A, B)

    @staticmethod
    def khatri_rao_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Column-wise Kronecker product.
        Used in Tensor decompositions (CP).
        """
        if A.shape[1] != B.shape[1]:
            raise ValueError("Matrices must have same number of columns")
        
        cols = A.shape[1]
        result = []
        for i in range(cols):
            result.append(np.kron(A[:, i], B[:, i]))
            
        return np.column_stack(result)

# ==========================================
# 16. Advanced Regression Analysis (Bayesian)
# ==========================================

class AdvancedRegression:
    """
    Bayesian and Non-parametric Regression methods.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha # Precision of prior
        self.beta = beta   # Precision of noise
        self.m_N = None    # Posterior mean
        self.S_N = None    # Posterior covariance
        
    def fit_bayesian_linear(self, Phi: np.ndarray, t: np.ndarray):
        """
        Update posterior distribution p(w|t) = N(w | m_N, S_N).
        Phi: Design matrix (N x M)
        t: Target values (N)
        """
        N, M = Phi.shape
        
        # S_N^-1 = alpha * I + beta * Phi.T @ Phi
        S_N_inv = self.alpha * np.eye(M) + self.beta * (Phi.T @ Phi)
        self.S_N = np.linalg.inv(S_N_inv)
        
        # m_N = beta * S_N @ Phi.T @ t
        self.m_N = self.beta * (self.S_N @ Phi.T @ t)
        
    def predict(self, phi_new: np.ndarray) -> Tuple[float, float]:
        """
        Predict predictive distribution p(t|x, t_train) = N(t | m, sigma^2).
        Returns (mean, variance).
        """
        if self.m_N is None:
            return 0.0, 1.0/self.beta
            
        # Mean = m_N.T @ phi
        y_mean = self.m_N @ phi_new
        
        # Variance = 1/beta + phi.T @ S_N @ phi
        y_var = (1.0 / self.beta) + (phi_new.T @ self.S_N @ phi_new)
        
        return y_mean, y_var
