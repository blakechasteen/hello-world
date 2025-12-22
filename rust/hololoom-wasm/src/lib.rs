//! WebAssembly bindings for HoloLoom's high-performance algorithms
//!
//! Provides browser-compatible functions for:
//! - Cosine similarity (batch)
//! - Vector normalization (batch)
//! - K-means clustering
//! - Silhouette score
//!
//! Usage (JavaScript):
//! ```javascript
//! import init, { cosine_similarity_batch, kmeans } from '@hololoom/clustering';
//!
//! await init();
//!
//! const vectors = new Float32Array([...]);
//! const centroid = new Float32Array([...]);
//! const similarities = cosine_similarity_batch(vectors, centroid, 1000, 384);
//! ```
//!
//! TODO: Implement in Phase 5 of Rust+WASM roadmap

use wasm_bindgen::prelude::*;

// Initialize panic hook for better error messages in browser console
#[cfg(feature = "console_error_panic_hook")]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Compute cosine similarity between multiple vectors and a centroid
///
/// # Arguments
/// * `vectors` - Flat Float32Array containing n_vectors × dim elements
/// * `centroid` - Float32Array of length dim
/// * `n_vectors` - Number of vectors
/// * `dim` - Dimension of each vector
///
/// # Returns
/// Float32Array of cosine similarities, length n_vectors
#[wasm_bindgen]
pub fn cosine_similarity_batch(
    vectors: &[f32],
    centroid: &[f32],
    n_vectors: usize,
    dim: usize,
) -> Vec<f32> {
    #[cfg(feature = "console_error_panic_hook")]
    set_panic_hook();

    hololoom_core::cosine_similarity_batch(vectors, centroid, n_vectors, dim)
}

/// Normalize vectors to unit length
///
/// # Arguments
/// * `vectors` - Flat Float32Array containing n_vectors × dim elements
/// * `n_vectors` - Number of vectors
/// * `dim` - Dimension of each vector
///
/// # Returns
/// Float32Array of normalized vectors (same shape)
#[wasm_bindgen]
pub fn normalize_batch(vectors: &[f32], n_vectors: usize, dim: usize) -> Vec<f32> {
    #[cfg(feature = "console_error_panic_hook")]
    set_panic_hook();

    hololoom_core::vector_ops::normalize_batch_flat(vectors, n_vectors, dim)
}

/// K-means clustering result for JavaScript
#[wasm_bindgen]
pub struct KMeansResultJS {
    labels: Vec<i32>,
    centroids: Vec<f32>,
    inertia: f32,
    n_iterations: usize,
}

#[wasm_bindgen]
impl KMeansResultJS {
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Vec<i32> {
        self.labels.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn centroids(&self) -> Vec<f32> {
        self.centroids.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn inertia(&self) -> f32 {
        self.inertia
    }

    #[wasm_bindgen(getter)]
    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }
}

/// K-means clustering
///
/// # Arguments
/// * `data` - Flat Float32Array containing n_samples × dim elements
/// * `n_samples` - Number of samples
/// * `dim` - Dimension of each sample
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum iterations (default: 300)
/// * `tolerance` - Convergence tolerance (default: 1e-4)
/// * `n_init` - Number of initializations (default: 10)
/// * `seed` - Random seed (default: 42)
///
/// # Returns
/// KMeansResultJS with labels, centroids, inertia, and n_iterations
#[wasm_bindgen]
pub fn kmeans(
    data: &[f32],
    n_samples: usize,
    dim: usize,
    k: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f32>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> KMeansResultJS {
    #[cfg(feature = "console_error_panic_hook")]
    set_panic_hook();

    let max_iterations = max_iterations.unwrap_or(300);
    let tolerance = tolerance.unwrap_or(1e-4);
    let n_init = n_init.unwrap_or(10);
    let seed = seed.unwrap_or(42);

    let result = hololoom_core::kmeans(
        data,
        n_samples,
        dim,
        k,
        max_iterations,
        tolerance,
        n_init,
        seed,
    );

    KMeansResultJS {
        labels: result.labels.iter().map(|&x| x as i32).collect(),
        centroids: result.centroids,
        inertia: result.inertia,
        n_iterations: result.n_iterations,
    }
}

/// Compute silhouette score for clustering quality
///
/// # Arguments
/// * `data` - Flat Float32Array containing n_samples × dim elements
/// * `labels` - Int32Array of cluster assignments (n_samples)
/// * `n_samples` - Number of samples
/// * `dim` - Dimension of each sample
///
/// # Returns
/// Mean silhouette coefficient in range [-1, 1]
#[wasm_bindgen]
pub fn silhouette_score(data: &[f32], labels: &[i32], n_samples: usize, dim: usize) -> f32 {
    #[cfg(feature = "console_error_panic_hook")]
    set_panic_hook();

    let labels_usize: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
    hololoom_core::silhouette_score(data, &labels_usize, n_samples, dim)
}

/// Initialize the WASM module (call once before using other functions)
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    set_panic_hook();
}
