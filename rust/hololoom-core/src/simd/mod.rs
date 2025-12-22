//! SIMD-optimized implementations
//!
//! Platform-specific SIMD implementations:
//! - AVX2 for x86_64
//! - NEON for ARM (future)
//! - WASM SIMD for browsers (future)

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
