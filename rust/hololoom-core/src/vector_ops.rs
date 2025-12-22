//! Vector operations for embedding computations
//!
//! Provides high-performance implementations of:
//! - Dot product
//! - Cosine similarity (single and batch)
//! - Vector normalization (single and batch)
//! - Euclidean distance
//!
//! All operations use SIMD when available with automatic fallback.

use rayon::prelude::*;

const EPSILON: f32 = 1e-9;

/// Compute dot product of two vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector (must be same length as a)
///
/// # Returns
/// Dot product as f32
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { crate::simd::avx2::dot_product_avx2(a, b) };
        }
    }

    // Scalar fallback with loop unrolling hint
    dot_product_scalar(a, b)
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Compute L2 norm of a vector
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

/// Normalize a vector to unit length
///
/// # Arguments
/// * `v` - Vector to normalize
///
/// # Returns
/// New vector with L2 norm of 1.0 (or original if near-zero)
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm < EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Normalize a vector in-place
///
/// # Arguments
/// * `v` - Mutable vector to normalize
pub fn normalize_inplace(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm < EPSILON {
        return;
    }
    let inv_norm = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

/// Normalize multiple vectors in parallel
///
/// # Arguments
/// * `vectors` - Slice of vectors (each as a slice of f32)
///
/// # Returns
/// Vector of normalized vectors
pub fn normalize_batch(vectors: &[&[f32]]) -> Vec<Vec<f32>> {
    vectors
        .par_iter()
        .map(|v| normalize(v))
        .collect()
}

/// Normalize a 2D array of vectors (row-major)
///
/// # Arguments
/// * `data` - Flattened 2D array (n_vectors * dim)
/// * `n_vectors` - Number of vectors
/// * `dim` - Dimension of each vector
///
/// # Returns
/// Flattened array of normalized vectors
pub fn normalize_batch_flat(data: &[f32], n_vectors: usize, dim: usize) -> Vec<f32> {
    assert_eq!(data.len(), n_vectors * dim, "Data length must equal n_vectors * dim");

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                crate::simd::avx2::normalize_batch_avx2(data, n_vectors, dim)
            };
        }
    }

    // Scalar fallback with Rayon parallelism
    normalize_batch_flat_scalar(data, n_vectors, dim)
}

fn normalize_batch_flat_scalar(data: &[f32], n_vectors: usize, dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; data.len()];

    result
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(i, chunk)| {
            let start = i * dim;
            let src = &data[start..start + dim];
            let norm = l2_norm(src);
            if norm < EPSILON {
                chunk.copy_from_slice(src);
            } else {
                let inv_norm = 1.0 / norm;
                for (j, x) in src.iter().enumerate() {
                    chunk[j] = x * inv_norm;
                }
            }
        });

    result
}

/// Compute cosine similarity between two vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity in range [-1, 1]
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b + EPSILON;
    dot / denom
}

/// Compute cosine similarity between multiple vectors and a single centroid
///
/// This is the HOT PATH for clustering - optimized for batch operations.
///
/// # Arguments
/// * `vectors` - Flattened 2D array of vectors (n_vectors * dim)
/// * `centroid` - Single centroid vector
/// * `n_vectors` - Number of vectors
/// * `dim` - Dimension of each vector
///
/// # Returns
/// Vector of cosine similarities
pub fn cosine_similarity_batch(
    vectors: &[f32],
    centroid: &[f32],
    n_vectors: usize,
    dim: usize,
) -> Vec<f32> {
    assert_eq!(vectors.len(), n_vectors * dim, "Data length must equal n_vectors * dim");
    assert_eq!(centroid.len(), dim, "Centroid must have same dimension");

    // Pre-compute centroid norm
    let centroid_norm = l2_norm(centroid);
    if centroid_norm < EPSILON {
        return vec![0.0; n_vectors];
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                crate::simd::avx2::cosine_similarity_batch_avx2(
                    vectors, centroid, n_vectors, dim, centroid_norm
                )
            };
        }
    }

    // Parallel scalar fallback
    cosine_similarity_batch_scalar(vectors, centroid, n_vectors, dim, centroid_norm)
}

fn cosine_similarity_batch_scalar(
    vectors: &[f32],
    centroid: &[f32],
    n_vectors: usize,
    dim: usize,
    centroid_norm: f32,
) -> Vec<f32> {
    (0..n_vectors)
        .into_par_iter()
        .map(|i| {
            let start = i * dim;
            let vec = &vectors[start..start + dim];

            let dot: f32 = vec.iter()
                .zip(centroid.iter())
                .map(|(a, b)| a * b)
                .sum();

            let vec_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

            dot / (vec_norm * centroid_norm + EPSILON)
        })
        .collect()
}

/// Compute pairwise cosine similarity matrix
///
/// # Arguments
/// * `vectors` - Flattened 2D array (n_vectors * dim)
/// * `n_vectors` - Number of vectors
/// * `dim` - Dimension
///
/// # Returns
/// Flattened upper triangular similarity matrix
pub fn cosine_similarity_matrix(
    vectors: &[f32],
    n_vectors: usize,
    dim: usize,
) -> Vec<f32> {
    assert_eq!(vectors.len(), n_vectors * dim);

    // Pre-compute all norms
    let norms: Vec<f32> = (0..n_vectors)
        .into_par_iter()
        .map(|i| {
            let start = i * dim;
            l2_norm(&vectors[start..start + dim])
        })
        .collect();

    // Compute upper triangular matrix (n * (n-1) / 2 elements)
    let n_pairs = n_vectors * (n_vectors - 1) / 2;
    let mut result = vec![0.0f32; n_pairs];

    // Parallel computation of pairs
    result
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, sim)| {
            // Convert linear index to (i, j) where i < j
            let i = ((2.0 * n_vectors as f64 - 1.0
                     - ((2.0 * n_vectors as f64 - 1.0).powi(2)
                        - 8.0 * idx as f64).sqrt()) / 2.0) as usize;
            let j = idx - i * (2 * n_vectors - i - 1) / 2 + i + 1;

            let start_i = i * dim;
            let start_j = j * dim;
            let vec_i = &vectors[start_i..start_i + dim];
            let vec_j = &vectors[start_j..start_j + dim];

            let dot: f32 = vec_i.iter()
                .zip(vec_j.iter())
                .map(|(a, b)| a * b)
                .sum();

            *sim = dot / (norms[i] * norms[j] + EPSILON);
        });

    result
}

/// Compute Euclidean distance between two vectors
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute squared Euclidean distance (faster, avoids sqrt)
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let norm = l2_norm(&v);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = normalize(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
        assert!((l2_norm(&normalized) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = normalize(&v);
        assert_eq!(normalized, v);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6); // Orthogonal vectors

        let c = vec![1.0, 1.0];
        let d = vec![1.0, 1.0];
        let sim2 = cosine_similarity(&c, &d);
        assert!((sim2 - 1.0).abs() < 1e-6); // Identical vectors
    }

    #[test]
    fn test_cosine_similarity_batch() {
        let vectors = vec![
            1.0, 0.0,  // Vector 0
            0.0, 1.0,  // Vector 1
            1.0, 1.0,  // Vector 2
        ];
        let centroid = vec![1.0, 0.0];

        let sims = cosine_similarity_batch(&vectors, &centroid, 3, 2);

        assert!((sims[0] - 1.0).abs() < 1e-6);  // Same direction
        assert!(sims[1].abs() < 1e-6);          // Orthogonal
        assert!((sims[2] - 0.7071068).abs() < 1e-5); // 45 degrees
    }

    #[test]
    fn test_normalize_batch_flat() {
        let data = vec![
            3.0, 4.0,  // Vector 0, norm = 5
            0.0, 5.0,  // Vector 1, norm = 5
        ];

        let normalized = normalize_batch_flat(&data, 2, 2);

        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
        assert!((normalized[2] - 0.0).abs() < 1e-6);
        assert!((normalized[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }
}
