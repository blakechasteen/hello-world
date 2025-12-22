//! AVX2 SIMD implementations for x86_64
//!
//! Uses 256-bit vectors for 8x f32 parallel operations.
//! These functions are unsafe and require AVX2 feature detection.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::prelude::*;

const EPSILON: f32 = 1e-9;

/// AVX2-optimized dot product
///
/// # Safety
/// Caller must ensure AVX2 is available via `is_x86_feature_detected!("avx2")`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();

    // Process 8 floats at a time
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    let mut result = horizontal_sum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }

    result
}

/// Horizontal sum of 8 floats in AVX2 register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Sum pairs: [a0+a4, a1+a5, a2+a6, a3+a7, a0+a4, a1+a5, a2+a6, a3+a7]
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);

    // Sum to single value
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);

    _mm_cvtss_f32(sum32)
}

/// AVX2-optimized batch cosine similarity
///
/// # Safety
/// Caller must ensure AVX2 is available
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn cosine_similarity_batch_avx2(
    vectors: &[f32],
    centroid: &[f32],
    n_vectors: usize,
    dim: usize,
    centroid_norm: f32,
) -> Vec<f32> {
    // Parallel processing of vectors
    (0..n_vectors)
        .into_par_iter()
        .map(|i| {
            let start = i * dim;
            let vec = &vectors[start..start + dim];

            // Compute dot product and norm simultaneously
            let chunks = dim / 8;
            let mut dot_sum = _mm256_setzero_ps();
            let mut norm_sum = _mm256_setzero_ps();

            for j in 0..chunks {
                let idx = j * 8;
                let vv = _mm256_loadu_ps(vec.as_ptr().add(idx));
                let vc = _mm256_loadu_ps(centroid.as_ptr().add(idx));

                dot_sum = _mm256_fmadd_ps(vv, vc, dot_sum);
                norm_sum = _mm256_fmadd_ps(vv, vv, norm_sum);
            }

            let mut dot = horizontal_sum_avx2(dot_sum);
            let mut vec_norm_sq = horizontal_sum_avx2(norm_sum);

            // Handle remainder
            for j in (chunks * 8)..dim {
                dot += vec[j] * centroid[j];
                vec_norm_sq += vec[j] * vec[j];
            }

            let vec_norm = vec_norm_sq.sqrt();
            dot / (vec_norm * centroid_norm + EPSILON)
        })
        .collect()
}

/// AVX2-optimized batch normalization
///
/// # Safety
/// Caller must ensure AVX2 is available
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn normalize_batch_avx2(
    data: &[f32],
    n_vectors: usize,
    dim: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; data.len()];

    result
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(i, chunk)| {
            let start = i * dim;
            let src = &data[start..start + dim];

            // Compute norm with SIMD
            let chunks = dim / 8;
            let mut norm_sum = _mm256_setzero_ps();

            for j in 0..chunks {
                let idx = j * 8;
                let v = _mm256_loadu_ps(src.as_ptr().add(idx));
                norm_sum = _mm256_fmadd_ps(v, v, norm_sum);
            }

            let mut norm_sq = horizontal_sum_avx2(norm_sum);

            // Remainder
            for j in (chunks * 8)..dim {
                norm_sq += src[j] * src[j];
            }

            let norm = norm_sq.sqrt();
            if norm < EPSILON {
                chunk.copy_from_slice(src);
                return;
            }

            let inv_norm = 1.0 / norm;
            let inv_norm_vec = _mm256_set1_ps(inv_norm);

            // Normalize with SIMD
            for j in 0..chunks {
                let idx = j * 8;
                let v = _mm256_loadu_ps(src.as_ptr().add(idx));
                let normalized = _mm256_mul_ps(v, inv_norm_vec);
                _mm256_storeu_ps(chunk.as_mut_ptr().add(idx), normalized);
            }

            // Remainder
            for j in (chunks * 8)..dim {
                chunk[j] = src[j] * inv_norm;
            }
        });

    result
}
