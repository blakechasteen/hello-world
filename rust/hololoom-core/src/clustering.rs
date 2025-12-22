//! Clustering algorithms
//!
//! Provides:
//! - K-means clustering
//! - Silhouette score computation

use rayon::prelude::*;
use crate::vector_ops::{euclidean_distance_squared, cosine_similarity};

/// Result of K-means clustering
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster assignment for each point (0 to k-1)
    pub labels: Vec<usize>,
    /// Cluster centroids (k * dim flattened)
    pub centroids: Vec<f32>,
    /// Number of clusters
    pub k: usize,
    /// Dimension of each point
    pub dim: usize,
    /// Inertia (sum of squared distances to centroids)
    pub inertia: f32,
    /// Number of iterations run
    pub n_iterations: usize,
}

/// K-means clustering with k-means++ initialization
///
/// # Arguments
/// * `data` - Flattened data points (n_samples * dim)
/// * `n_samples` - Number of data points
/// * `dim` - Dimension of each point
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum iterations (default: 300)
/// * `tolerance` - Convergence tolerance (default: 1e-4)
/// * `n_init` - Number of initializations to try (default: 10)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// KMeansResult with best clustering (lowest inertia)
pub fn kmeans(
    data: &[f32],
    n_samples: usize,
    dim: usize,
    k: usize,
    max_iterations: usize,
    tolerance: f32,
    n_init: usize,
    seed: u64,
) -> KMeansResult {
    assert_eq!(data.len(), n_samples * dim);
    assert!(k > 0 && k <= n_samples);

    // Run multiple initializations and keep best
    let results: Vec<KMeansResult> = (0..n_init)
        .into_par_iter()
        .map(|init_idx| {
            let init_seed = seed.wrapping_add(init_idx as u64);
            kmeans_single(data, n_samples, dim, k, max_iterations, tolerance, init_seed)
        })
        .collect();

    // Return result with lowest inertia
    results.into_iter()
        .min_by(|a, b| a.inertia.partial_cmp(&b.inertia).unwrap())
        .unwrap()
}

/// Single K-means run
fn kmeans_single(
    data: &[f32],
    n_samples: usize,
    dim: usize,
    k: usize,
    max_iterations: usize,
    tolerance: f32,
    seed: u64,
) -> KMeansResult {
    // Initialize centroids using k-means++
    let mut centroids = kmeans_plusplus_init(data, n_samples, dim, k, seed);
    let mut labels = vec![0usize; n_samples];
    let mut prev_inertia = f32::MAX;

    for iteration in 0..max_iterations {
        // Assignment step: assign each point to nearest centroid
        let (new_labels, inertia) = assign_clusters(data, &centroids, n_samples, dim, k);
        labels = new_labels;

        // Check convergence
        if (prev_inertia - inertia).abs() < tolerance {
            return KMeansResult {
                labels,
                centroids,
                k,
                dim,
                inertia,
                n_iterations: iteration + 1,
            };
        }
        prev_inertia = inertia;

        // Update step: compute new centroids
        centroids = update_centroids(data, &labels, n_samples, dim, k);
    }

    KMeansResult {
        labels,
        centroids,
        k,
        dim,
        inertia: prev_inertia,
        n_iterations: max_iterations,
    }
}

/// K-means++ initialization
fn kmeans_plusplus_init(
    data: &[f32],
    n_samples: usize,
    dim: usize,
    k: usize,
    seed: u64,
) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * dim);
    let mut rng = SimpleRng::new(seed);

    // First centroid: random point
    let first_idx = rng.next_usize() % n_samples;
    centroids.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

    // Remaining centroids: weighted by squared distance to nearest centroid
    for _ in 1..k {
        let distances: Vec<f32> = (0..n_samples)
            .map(|i| {
                let point = &data[i * dim..(i + 1) * dim];
                // Find distance to nearest existing centroid
                (0..centroids.len() / dim)
                    .map(|c| {
                        let centroid = &centroids[c * dim..(c + 1) * dim];
                        euclidean_distance_squared(point, centroid)
                    })
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        // Sample proportional to squared distance
        let total: f32 = distances.iter().sum();
        let threshold = rng.next_f32() * total;

        let mut cumsum = 0.0;
        let mut selected = 0;
        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                selected = i;
                break;
            }
        }

        centroids.extend_from_slice(&data[selected * dim..(selected + 1) * dim]);
    }

    centroids
}

/// Assign each point to nearest centroid
fn assign_clusters(
    data: &[f32],
    centroids: &[f32],
    n_samples: usize,
    dim: usize,
    k: usize,
) -> (Vec<usize>, f32) {
    let results: Vec<(usize, f32)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let point = &data[i * dim..(i + 1) * dim];

            let (nearest, min_dist) = (0..k)
                .map(|c| {
                    let centroid = &centroids[c * dim..(c + 1) * dim];
                    let dist = euclidean_distance_squared(point, centroid);
                    (c, dist)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            (nearest, min_dist)
        })
        .collect();

    let labels: Vec<usize> = results.iter().map(|(l, _)| *l).collect();
    let inertia: f32 = results.iter().map(|(_, d)| *d).sum();

    (labels, inertia)
}

/// Update centroids based on cluster assignments
fn update_centroids(
    data: &[f32],
    labels: &[usize],
    n_samples: usize,
    dim: usize,
    k: usize,
) -> Vec<f32> {
    let mut centroids = vec![0.0f32; k * dim];
    let mut counts = vec![0usize; k];

    // Sum points in each cluster
    for i in 0..n_samples {
        let cluster = labels[i];
        counts[cluster] += 1;
        for j in 0..dim {
            centroids[cluster * dim + j] += data[i * dim + j];
        }
    }

    // Divide by count to get mean
    for c in 0..k {
        if counts[c] > 0 {
            let count = counts[c] as f32;
            for j in 0..dim {
                centroids[c * dim + j] /= count;
            }
        }
    }

    centroids
}

/// Compute silhouette score for clustering
///
/// # Arguments
/// * `data` - Flattened data points
/// * `labels` - Cluster assignments
/// * `n_samples` - Number of samples
/// * `dim` - Dimension
///
/// # Returns
/// Mean silhouette coefficient in range [-1, 1]
pub fn silhouette_score(
    data: &[f32],
    labels: &[usize],
    n_samples: usize,
    dim: usize,
) -> f32 {
    assert_eq!(data.len(), n_samples * dim);
    assert_eq!(labels.len(), n_samples);

    if n_samples < 2 {
        return 0.0;
    }

    // Find unique clusters
    let k = *labels.iter().max().unwrap_or(&0) + 1;
    if k < 2 {
        return 0.0;
    }

    // Compute silhouette for each sample in parallel
    let silhouettes: Vec<f32> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let point = &data[i * dim..(i + 1) * dim];
            let cluster = labels[i];

            // a(i) = mean distance to points in same cluster
            let (same_sum, same_count) = labels.iter()
                .enumerate()
                .filter(|(j, &c)| *j != i && c == cluster)
                .fold((0.0f32, 0usize), |(sum, count), (j, _)| {
                    let other = &data[j * dim..(j + 1) * dim];
                    let dist = euclidean_distance_squared(point, other).sqrt();
                    (sum + dist, count + 1)
                });

            let a = if same_count > 0 {
                same_sum / same_count as f32
            } else {
                0.0
            };

            // b(i) = min mean distance to points in other clusters
            let b = (0..k)
                .filter(|&c| c != cluster)
                .map(|c| {
                    let (sum, count) = labels.iter()
                        .enumerate()
                        .filter(|(_, &label)| label == c)
                        .fold((0.0f32, 0usize), |(sum, count), (j, _)| {
                            let other = &data[j * dim..(j + 1) * dim];
                            let dist = euclidean_distance_squared(point, other).sqrt();
                            (sum + dist, count + 1)
                        });
                    if count > 0 { sum / count as f32 } else { f32::MAX }
                })
                .fold(f32::MAX, f32::min);

            // s(i) = (b - a) / max(a, b)
            let max_ab = a.max(b);
            if max_ab < 1e-9 {
                0.0
            } else {
                (b - a) / max_ab
            }
        })
        .collect();

    // Return mean silhouette
    silhouettes.iter().sum::<f32>() / n_samples as f32
}

/// Simple PRNG for reproducibility (xorshift64)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self) -> usize {
        self.next() as usize
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_simple() {
        // Two clear clusters
        let data = vec![
            0.0, 0.0,  // Cluster 0
            0.1, 0.1,
            0.2, 0.0,
            10.0, 10.0,  // Cluster 1
            10.1, 10.1,
            10.2, 10.0,
        ];

        let result = kmeans(&data, 6, 2, 2, 100, 1e-4, 3, 42);

        assert_eq!(result.k, 2);
        assert_eq!(result.labels.len(), 6);

        // Check that points are grouped correctly
        // (first 3 should be in same cluster, last 3 in another)
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_silhouette_score() {
        // Perfect clustering
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            10.0, 10.0,
            10.1, 10.1,
        ];
        let labels = vec![0, 0, 1, 1];

        let score = silhouette_score(&data, &labels, 4, 2);

        // Should be close to 1.0 for well-separated clusters
        assert!(score > 0.9, "Expected high silhouette score, got {}", score);
    }

    #[test]
    fn test_silhouette_single_cluster() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0, 0];

        let score = silhouette_score(&data, &labels, 2, 2);
        assert_eq!(score, 0.0);
    }
}
