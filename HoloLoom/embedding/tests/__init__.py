"""
HoloLoom Embedding Module Tests
===============================

Comprehensive test suite for the embedding module covering:

1. MatryoshkaEmbeddings - Multi-scale embedding generation
   - Multi-scale extraction (96, 192, 384 dimensions)
   - Embedding normalization
   - Batch processing
   - Graceful degradation without sentence-transformers

2. ZeroCopyMatryoshkaEmbeddings - Memory-mapped storage
   - Memory-mapped storage creation/loading
   - Cache hit/miss behavior
   - Scale extraction via views (verify no copying)
   - Performance benchmarks
   - Cache invalidation

3. SpectralFusion - Graph-based embeddings
   - Graph Laplacian eigenvalue computation
   - SVD topic components
   - Feature concatenation
   - Multi-scale fusion

4. MultiScaleSpectralAnalyzer - Hierarchical analysis
   - Coarse-to-fine spectral analysis
   - Hierarchical spectral clustering
   - Multi-scale wavelet decomposition

Test Coverage Target: 2,000+ lines
Test File Structure:
    - test_spectral.py: Matryoshka + SpectralFusion (~1,200 lines)
    - test_zero_copy.py: Zero-copy embeddings (~800 lines)

Created: 2026-01-22
Author: Claude Code
"""

__all__ = [
    'test_spectral',
    'test_zero_copy',
]
