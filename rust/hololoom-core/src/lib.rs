//! HoloLoom Core - High-performance vector operations and clustering
//!
//! This crate provides optimized implementations of:
//! - Cosine similarity (batch operations)
//! - Vector normalization
//! - K-means clustering
//! - Silhouette scoring
//! - Mutual information calculations
//!
//! All operations support both scalar and SIMD implementations with
//! automatic runtime detection.

pub mod vector_ops;
pub mod clustering;
pub mod simd;

// Re-export main functions at crate root
pub use vector_ops::{
    cosine_similarity,
    cosine_similarity_batch,
    normalize,
    normalize_batch,
    normalize_batch_flat,
    dot_product,
    euclidean_distance,
};

pub use clustering::{
    kmeans,
    silhouette_score,
    KMeansResult,
};
