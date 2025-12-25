//! Python bindings for HoloLoom's high-performance algorithms
//!
//! Provides numpy-compatible functions for:
//! - Cosine similarity (batch)
//! - Vector normalization (batch)
//! - K-means clustering
//! - Silhouette score
//!
//! All functions accept and return numpy arrays for seamless integration.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use hololoom_core::{
    cosine_similarity_batch,
    normalize_batch_flat,
    kmeans,
    silhouette_score,
};

/// Compute cosine similarity between multiple vectors and a centroid
///
/// Args:
///     vectors: 2D numpy array of shape (n_vectors, dim)
///     centroid: 1D numpy array of shape (dim,)
///
/// Returns:
///     1D numpy array of cosine similarities, shape (n_vectors,)
///
/// Example:
///     >>> import numpy as np
///     >>> from hololoom_rust import cosine_similarity_batch
///     >>> vectors = np.random.randn(1000, 384).astype(np.float32)
///     >>> centroid = np.random.randn(384).astype(np.float32)
///     >>> similarities = cosine_similarity_batch(vectors, centroid)
#[pyfunction]
#[pyo3(name = "cosine_similarity_batch")]
fn py_cosine_similarity_batch<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f32>,
    centroid: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let vectors_arr = vectors.as_array();
    let centroid_arr = centroid.as_array();

    let (n_vectors, dim) = vectors_arr.dim();
    let centroid_len = centroid_arr.len();

    if dim != centroid_len {
        return Err(PyValueError::new_err(format!(
            "Dimension mismatch: vectors have dim={}, centroid has dim={}",
            dim, centroid_len
        )));
    }

    // Convert to contiguous slices
    let vectors_slice: Vec<f32> = vectors_arr.iter().cloned().collect();
    let centroid_slice: Vec<f32> = centroid_arr.iter().cloned().collect();

    // Call Rust implementation
    let result = cosine_similarity_batch(&vectors_slice, &centroid_slice, n_vectors, dim);

    Ok(result.into_pyarray(py))
}

/// Normalize vectors to unit length
///
/// Args:
///     vectors: 2D numpy array of shape (n_vectors, dim)
///
/// Returns:
///     2D numpy array of normalized vectors, shape (n_vectors, dim)
///
/// Example:
///     >>> import numpy as np
///     >>> from hololoom_rust import normalize_batch
///     >>> vectors = np.random.randn(1000, 384).astype(np.float32)
///     >>> normalized = normalize_batch(vectors)
///     >>> norms = np.linalg.norm(normalized, axis=1)  # All ~1.0
#[pyfunction]
#[pyo3(name = "normalize_batch")]
fn py_normalize_batch<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let vectors_arr = vectors.as_array();
    let (n_vectors, dim) = vectors_arr.dim();

    let vectors_slice: Vec<f32> = vectors_arr.iter().cloned().collect();
    let result = normalize_batch_flat(&vectors_slice, n_vectors, dim);

    // Reshape to 2D
    let result_2d: Vec<Vec<f32>> = result
        .chunks(dim)
        .map(|chunk| chunk.to_vec())
        .collect();

    // Convert to numpy
    let flat: Vec<f32> = result_2d.into_iter().flatten().collect();
    Ok(PyArray1::from_vec(py, flat)
        .reshape([n_vectors, dim])
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .to_owned())
}

/// K-means clustering
///
/// Args:
///     data: 2D numpy array of shape (n_samples, dim)
///     k: Number of clusters
///     max_iterations: Maximum iterations (default: 300)
///     tolerance: Convergence tolerance (default: 1e-4)
///     n_init: Number of initializations (default: 10)
///     seed: Random seed (default: 42)
///
/// Returns:
///     Tuple of (labels, centroids, inertia, n_iterations)
///     - labels: 1D array of cluster assignments (n_samples,)
///     - centroids: 2D array of centroids (k, dim)
///     - inertia: Sum of squared distances to centroids
///     - n_iterations: Number of iterations run
///
/// Example:
///     >>> import numpy as np
///     >>> from hololoom_rust import kmeans
///     >>> data = np.random.randn(1000, 384).astype(np.float32)
///     >>> labels, centroids, inertia, n_iter = kmeans(data, k=5)
#[pyfunction]
#[pyo3(name = "kmeans", signature = (data, k, max_iterations=300, tolerance=1e-4, n_init=10, seed=42))]
fn py_kmeans<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k: usize,
    max_iterations: usize,
    tolerance: f32,
    n_init: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
    f32,
    usize,
)> {
    let data_arr = data.as_array();
    let (n_samples, dim) = data_arr.dim();

    if k == 0 || k > n_samples {
        return Err(PyValueError::new_err(format!(
            "Invalid k={}: must be between 1 and n_samples={}",
            k, n_samples
        )));
    }

    let data_slice: Vec<f32> = data_arr.iter().cloned().collect();

    let result = kmeans(&data_slice, n_samples, dim, k, max_iterations, tolerance, n_init, seed);

    // Convert labels to i64 for numpy compatibility
    let labels: Vec<i64> = result.labels.iter().map(|&x| x as i64).collect();

    // Reshape centroids to 2D
    let centroids_flat: Vec<f32> = result.centroids;

    Ok((
        labels.into_pyarray(py),
        PyArray1::from_vec(py, centroids_flat)
            .reshape([k, dim])
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .to_owned(),
        result.inertia,
        result.n_iterations,
    ))
}

/// Compute silhouette score for clustering quality
///
/// Args:
///     data: 2D numpy array of shape (n_samples, dim)
///     labels: 1D numpy array of cluster assignments (n_samples,)
///
/// Returns:
///     Mean silhouette coefficient in range [-1, 1]
///     Higher is better (1 = perfect clustering, -1 = wrong clustering)
///
/// Example:
///     >>> import numpy as np
///     >>> from hololoom_rust import kmeans, silhouette_score
///     >>> data = np.random.randn(1000, 384).astype(np.float32)
///     >>> labels, _, _, _ = kmeans(data, k=5)
///     >>> score = silhouette_score(data, labels)
#[pyfunction]
#[pyo3(name = "silhouette_score")]
fn py_silhouette_score<'py>(
    data: PyReadonlyArray2<'py, f32>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<f32> {
    let data_arr = data.as_array();
    let labels_arr = labels.as_array();

    let (n_samples, dim) = data_arr.dim();

    if labels_arr.len() != n_samples {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: data has {} samples, labels has {} elements",
            n_samples, labels_arr.len()
        )));
    }

    let data_slice: Vec<f32> = data_arr.iter().cloned().collect();
    let labels_slice: Vec<usize> = labels_arr.iter().map(|&x| x as usize).collect();

    Ok(silhouette_score(&data_slice, &labels_slice, n_samples, dim))
}

/// HoloLoom Rust - High-performance algorithms for Python
///
/// This module provides optimized implementations of:
/// - cosine_similarity_batch: Fast batch cosine similarity
/// - normalize_batch: Fast batch vector normalization
/// - kmeans: K-means clustering with k-means++ init
/// - silhouette_score: Clustering quality metric
///
/// All functions work with numpy arrays and provide 10-50x speedup
/// over pure Python/NumPy implementations.
#[pymodule]
fn hololoom_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_cosine_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(py_silhouette_score, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
