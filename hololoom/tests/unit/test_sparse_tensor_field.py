from __future__ import annotations

import numpy as np

from hololoom.warp.optimized import SparseTensorField


def _dense_to_sparse(matrix: np.ndarray, threshold: float = 0.0) -> SparseTensorField:
    field = SparseTensorField()
    field.from_dense(matrix, threshold=threshold)
    return field


def test_sparse_matmul_empty_returns_empty_array():
    field = SparseTensorField()
    result = field @ np.array([])
    assert result.size == 0


def test_sparse_matmul_with_vector_rhs():
    dense = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    field = _dense_to_sparse(dense)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = field @ vec

    expected = dense @ vec
    assert np.allclose(result, expected)


def test_sparse_matmul_with_sparse_rhs():
    left_dense = np.array(
        [
            [0.0, 4.0],
            [5.0, 0.0],
        ],
        dtype=np.float32,
    )
    right_dense = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    left = _dense_to_sparse(left_dense)
    right = _dense_to_sparse(right_dense)

    result = left @ right

    expected = left_dense @ right_dense
    assert np.allclose(result, expected)
