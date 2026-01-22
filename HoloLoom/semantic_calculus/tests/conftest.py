"""
Pytest Fixtures for Semantic Calculus Tests
============================================

Provides mock embeddings, sample dimensions, and test utilities.

Fixtures:
    - mock_embed_fn: Mock embedding function (384D)
    - mock_embed_fn_128: Mock embedding function (128D) for testing different sizes
    - sample_dimension: Single SemanticDimension for basic tests
    - sample_dimensions_16: 16 standard dimensions
    - sample_trajectory: Sample embedding trajectory for analysis
    - mock_spectrum: Pre-configured SemanticSpectrum
    - mock_integrator: Pre-configured GeometricIntegrator
"""

import pytest
import numpy as np
from typing import Callable, List, Dict, Optional
from unittest.mock import MagicMock
import hashlib


# ============================================================================
# Mock Embedding Functions
# ============================================================================

@pytest.fixture
def mock_embed_fn() -> Callable[[str], np.ndarray]:
    """
    Create a deterministic mock embedding function (384D).

    Uses word hash to generate consistent embeddings for the same word.
    This ensures test reproducibility.

    Returns:
        Function that maps string -> 384D normalized vector
    """
    def embed(text: str) -> np.ndarray:
        # Use hash to create deterministic but varied embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(384)
        # Normalize to unit sphere
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    return embed


@pytest.fixture
def mock_embed_fn_128() -> Callable[[str], np.ndarray]:
    """
    Create a deterministic mock embedding function (128D).

    Smaller dimension for faster tests and testing dimension compatibility.
    """
    def embed(text: str) -> np.ndarray:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(128)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    return embed


@pytest.fixture
def mock_embed_fn_batch() -> Callable[[List[str]], np.ndarray]:
    """
    Create a batch embedding function for testing batch operations.

    Returns:
        Function that maps List[str] -> (N, 384) array
    """
    def embed_batch(texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embeddings.append(vec)
        return np.array(embeddings)

    return embed_batch


@pytest.fixture
def mock_embed_fn_with_semantic_structure() -> Callable[[str], np.ndarray]:
    """
    Create mock embeddings with semantic structure for testing projections.

    Words like "warm", "friendly" will have similar embeddings.
    Words like "cold", "distant" will have opposite embeddings.
    """
    # Define semantic clusters
    POSITIVE_WARMTH = ["warm", "friendly", "kind", "loving", "caring", "gentle"]
    NEGATIVE_WARMTH = ["cold", "distant", "hostile", "unfriendly", "aloof"]
    POSITIVE_VALENCE = ["good", "happy", "joy", "positive", "excellent", "wonderful"]
    NEGATIVE_VALENCE = ["bad", "sad", "terrible", "negative", "awful", "horrible"]
    FORMAL = ["formal", "official", "professional", "proper", "respectful"]
    INFORMAL = ["casual", "relaxed", "informal", "chill", "laid-back"]

    # Pre-generate cluster centers
    rng = np.random.RandomState(42)
    warmth_positive_center = rng.randn(384)
    warmth_positive_center = warmth_positive_center / np.linalg.norm(warmth_positive_center)
    warmth_negative_center = -warmth_positive_center + 0.1 * rng.randn(384)
    warmth_negative_center = warmth_negative_center / np.linalg.norm(warmth_negative_center)

    valence_positive_center = rng.randn(384)
    valence_positive_center = valence_positive_center / np.linalg.norm(valence_positive_center)
    valence_negative_center = -valence_positive_center + 0.1 * rng.randn(384)
    valence_negative_center = valence_negative_center / np.linalg.norm(valence_negative_center)

    formal_center = rng.randn(384)
    formal_center = formal_center / np.linalg.norm(formal_center)
    informal_center = -formal_center + 0.1 * rng.randn(384)
    informal_center = informal_center / np.linalg.norm(informal_center)

    def embed(text: str) -> np.ndarray:
        text_lower = text.lower().strip()

        # Check cluster membership
        if text_lower in POSITIVE_WARMTH:
            base = warmth_positive_center.copy()
        elif text_lower in NEGATIVE_WARMTH:
            base = warmth_negative_center.copy()
        elif text_lower in POSITIVE_VALENCE:
            base = valence_positive_center.copy()
        elif text_lower in NEGATIVE_VALENCE:
            base = valence_negative_center.copy()
        elif text_lower in FORMAL:
            base = formal_center.copy()
        elif text_lower in INFORMAL:
            base = informal_center.copy()
        else:
            # Generic hash-based embedding
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            rng_local = np.random.RandomState(seed)
            base = rng_local.randn(384)

        # Add small noise for uniqueness
        noise_seed = int(hashlib.md5(text.encode()).hexdigest()[:4], 16)
        noise_rng = np.random.RandomState(noise_seed)
        noise = 0.05 * noise_rng.randn(384)

        vec = base + noise
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    return embed


# ============================================================================
# Sample Dimensions
# ============================================================================

@pytest.fixture
def sample_dimension():
    """
    Create a single SemanticDimension for basic testing.
    """
    try:
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension
    except ImportError:
        from ..dimensions import SemanticDimension

    return SemanticDimension(
        name="TestWarmth",
        positive_exemplars=["warm", "friendly", "kind"],
        negative_exemplars=["cold", "distant", "hostile"]
    )


@pytest.fixture
def sample_dimensions_list():
    """
    Create a list of 5 test dimensions covering different semantic axes.
    """
    try:
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension
    except ImportError:
        from ..dimensions import SemanticDimension

    return [
        SemanticDimension(
            name="Warmth",
            positive_exemplars=["warm", "friendly", "kind", "loving"],
            negative_exemplars=["cold", "distant", "hostile", "unfriendly"]
        ),
        SemanticDimension(
            name="Valence",
            positive_exemplars=["good", "happy", "positive", "excellent"],
            negative_exemplars=["bad", "sad", "negative", "terrible"]
        ),
        SemanticDimension(
            name="Formality",
            positive_exemplars=["formal", "official", "proper", "professional"],
            negative_exemplars=["casual", "informal", "relaxed", "chill"]
        ),
        SemanticDimension(
            name="Intensity",
            positive_exemplars=["intense", "strong", "powerful", "extreme"],
            negative_exemplars=["mild", "weak", "gentle", "subtle"]
        ),
        SemanticDimension(
            name="Certainty",
            positive_exemplars=["certain", "definite", "sure", "confident"],
            negative_exemplars=["uncertain", "doubtful", "unsure", "hesitant"]
        ),
    ]


@pytest.fixture
def standard_dimensions():
    """
    Get the actual STANDARD_DIMENSIONS from the module.
    """
    try:
        from HoloLoom.semantic_calculus.dimensions import STANDARD_DIMENSIONS
    except ImportError:
        from ..dimensions import STANDARD_DIMENSIONS

    return STANDARD_DIMENSIONS


@pytest.fixture
def extended_dimensions():
    """
    Get the actual EXTENDED_244_DIMENSIONS from the module.
    """
    try:
        from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS
    except ImportError:
        from ..dimensions import EXTENDED_244_DIMENSIONS

    return EXTENDED_244_DIMENSIONS


# ============================================================================
# Sample Trajectories
# ============================================================================

@pytest.fixture
def sample_trajectory() -> np.ndarray:
    """
    Create a sample embedding trajectory (10 steps, 384D).

    Simulates a path through embedding space.
    """
    rng = np.random.RandomState(42)

    # Start position
    start = rng.randn(384)
    start = start / np.linalg.norm(start)

    # Generate smooth trajectory
    trajectory = [start]
    velocity = 0.1 * rng.randn(384)

    for _ in range(9):
        # Add some acceleration
        acceleration = 0.02 * rng.randn(384)
        velocity = 0.9 * velocity + acceleration

        # Update position
        new_pos = trajectory[-1] + velocity
        new_pos = new_pos / np.linalg.norm(new_pos)  # Stay on unit sphere
        trajectory.append(new_pos)

    return np.array(trajectory)


@pytest.fixture
def sample_trajectory_short() -> np.ndarray:
    """
    Create a short trajectory (3 steps) for basic velocity/acceleration tests.
    """
    rng = np.random.RandomState(123)

    trajectory = []
    for i in range(3):
        vec = rng.randn(384)
        vec = vec / np.linalg.norm(vec)
        trajectory.append(vec)

    return np.array(trajectory)


@pytest.fixture
def sample_trajectory_with_shift() -> np.ndarray:
    """
    Create a trajectory with a clear semantic shift in the middle.

    First 5 steps: one direction
    Last 5 steps: different direction (simulates topic change)
    """
    rng = np.random.RandomState(42)

    # Direction 1
    dir1 = rng.randn(384)
    dir1 = dir1 / np.linalg.norm(dir1)

    # Direction 2 (orthogonal)
    dir2 = rng.randn(384)
    dir2 = dir2 - np.dot(dir2, dir1) * dir1  # Make orthogonal
    dir2 = dir2 / np.linalg.norm(dir2)

    trajectory = []

    # First half: move along dir1
    for i in range(5):
        pos = dir1 * (i * 0.1) + 0.01 * rng.randn(384)
        pos = pos / np.linalg.norm(pos)
        trajectory.append(pos)

    # Second half: move along dir2
    for i in range(5):
        pos = dir2 * (i * 0.1) + 0.01 * rng.randn(384)
        pos = pos / np.linalg.norm(pos)
        trajectory.append(pos)

    return np.array(trajectory)


# ============================================================================
# Mock Spectrum and Integrator
# ============================================================================

@pytest.fixture
def mock_spectrum(sample_dimensions_list, mock_embed_fn):
    """
    Create a pre-configured SemanticSpectrum with learned axes.
    """
    try:
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum
    except ImportError:
        from ..dimensions import SemanticSpectrum

    spectrum = SemanticSpectrum(dimensions=sample_dimensions_list)
    spectrum.learn_axes(mock_embed_fn)
    return spectrum


@pytest.fixture
def mock_integrator():
    """
    Create a GeometricIntegrator with a random projection matrix.
    """
    try:
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator
    except ImportError:
        from ..integrator import GeometricIntegrator

    rng = np.random.RandomState(42)

    # Create orthonormal projection matrix (16D -> 384D)
    P = rng.randn(16, 384)
    # Orthonormalize rows
    P, _ = np.linalg.qr(P.T)
    P = P.T[:16, :]

    return GeometricIntegrator(P, mass=1.0)


@pytest.fixture
def mock_flow():
    """
    Create a SemanticFlow with heat equation PDE.
    """
    try:
        from HoloLoom.semantic_calculus.flow import SemanticFlow
    except ImportError:
        pytest.skip("SemanticFlow requires PDE solvers which may not be available")

    return SemanticFlow(
        dimensions=16,
        pde_type="heat",
        dt=0.01
    )


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def random_vector_384() -> np.ndarray:
    """Create a random normalized 384D vector."""
    rng = np.random.RandomState(42)
    vec = rng.randn(384)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def random_vector_16() -> np.ndarray:
    """Create a random normalized 16D vector."""
    rng = np.random.RandomState(42)
    vec = rng.randn(16)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def zero_vector_384() -> np.ndarray:
    """Create a zero 384D vector for edge case testing."""
    return np.zeros(384)


@pytest.fixture
def near_zero_vector_384() -> np.ndarray:
    """Create a near-zero 384D vector for numerical stability testing."""
    return np.ones(384) * 1e-15


@pytest.fixture
def orthogonal_vectors_384():
    """Create two orthogonal 384D vectors."""
    rng = np.random.RandomState(42)

    v1 = rng.randn(384)
    v1 = v1 / np.linalg.norm(v1)

    v2 = rng.randn(384)
    v2 = v2 - np.dot(v2, v1) * v1  # Make orthogonal
    v2 = v2 / np.linalg.norm(v2)

    return v1, v2


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def large_trajectory() -> np.ndarray:
    """Create a large trajectory (1000 steps) for performance testing."""
    rng = np.random.RandomState(42)
    trajectory = rng.randn(1000, 384)
    # Normalize each vector
    norms = np.linalg.norm(trajectory, axis=1, keepdims=True)
    trajectory = trajectory / (norms + 1e-10)
    return trajectory


@pytest.fixture
def many_words() -> List[str]:
    """Generate many unique words for embedding cache testing."""
    words = []
    for i in range(1000):
        words.append(f"word_{i}")
    return words


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def dimension_selector_config():
    """Configuration for dimension selector testing."""
    return {
        'n_dimensions': 36,
        'strategy': 'hybrid',
        'domain': 'narrative',
        'weights': {
            'variance': 0.35,
            'balance': 0.30,
            'domain': 0.35
        }
    }


@pytest.fixture
def pde_configs():
    """Configuration options for PDE-based flow testing."""
    return {
        'heat': {'dt': 0.01, 'diffusivity': 1.0},
        'wave': {'dt': 0.005, 'wave_speed': 1.0},
        'reaction_diffusion': {'dt': 0.01, 'reaction_type': 'competitive'},
    }


# ============================================================================
# Validation Helpers
# ============================================================================

def assert_normalized(vec: np.ndarray, atol: float = 1e-6):
    """Assert that a vector is normalized (unit length)."""
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < atol, f"Vector not normalized: norm = {norm}"


def assert_orthogonal(v1: np.ndarray, v2: np.ndarray, atol: float = 1e-6):
    """Assert that two vectors are orthogonal."""
    dot = np.abs(np.dot(v1, v2))
    assert dot < atol, f"Vectors not orthogonal: dot product = {dot}"


def assert_valid_projection(projection: Dict[str, float], dimensions: List):
    """Assert that a projection dict has valid structure."""
    assert isinstance(projection, dict), "Projection must be a dictionary"
    for dim in dimensions:
        dim_name = dim.name if hasattr(dim, 'name') else dim
        assert dim_name in projection, f"Missing dimension: {dim_name}"
        assert isinstance(projection[dim_name], (int, float)), f"Invalid value type for {dim_name}"
        assert not np.isnan(projection[dim_name]), f"NaN value for {dim_name}"


# Make helpers available as fixtures too
@pytest.fixture
def validation_helpers():
    """Return validation helper functions."""
    return {
        'assert_normalized': assert_normalized,
        'assert_orthogonal': assert_orthogonal,
        'assert_valid_projection': assert_valid_projection
    }
