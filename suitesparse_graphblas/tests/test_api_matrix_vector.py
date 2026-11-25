"""Matrices and vectors of the functional API: building and extracting each type."""

import numpy as np
import pytest

from suitesparse_graphblas import ffi, lib, libget, supports_complex
from suitesparse_graphblas.api import matrix, vector

needs_complex = pytest.mark.skipif(not supports_complex(), reason="no complex types")

# suffix, NumPy dtype, three values
TYPES = [
    ("bool", np.bool_, [True, False, True]),
    ("int8", np.int8, [-(2**7), 0, 2**7 - 1]),
    ("int16", np.int16, [-(2**15), 0, 2**15 - 1]),
    ("int32", np.int32, [-(2**31), 0, 2**31 - 1]),
    ("int64", np.int64, [-(2**63), 0, 2**63 - 1]),
    ("uint8", np.uint8, [0, 1, 2**8 - 1]),
    ("uint16", np.uint16, [0, 1, 2**16 - 1]),
    ("uint32", np.uint32, [0, 1, 2**32 - 1]),
    ("uint64", np.uint64, [0, 1, 2**64 - 1]),
    ("fp32", np.float32, [-1.5, 0.0, 2.25]),
    ("fp64", np.float64, [-1.5, 0.0, 1e300]),
    pytest.param("fc32", np.complex64, [1.5 - 2j, 0j, -3 + 0.5j], marks=needs_complex),
    pytest.param("fc64", np.complex128, [1.5 - 2j, 0j, -3 + 1e300j], marks=needs_complex),
]


@pytest.mark.parametrize(("suffix", "dtype", "values"), TYPES)
def test_matrix_build_and_extract_tuples(suffix, dtype, values):
    extract = getattr(matrix, f"matrix_extract_tuples_{suffix}")
    A = matrix.matrix_new(libget(f"GrB_{suffix.upper()}"), 3, 4)
    for array, expected in zip(extract(A), [np.uint64, np.uint64, dtype]):
        assert array.dtype == expected
        assert array.size == 0

    rows = np.array([2, 0, 1], dtype=np.uint64)
    cols = np.array([1, 3, 0], dtype=np.uint64)
    build = getattr(matrix, f"matrix_build_{suffix}")
    build(A, rows, cols, np.array(values, dtype=dtype), 3, libget(f"GrB_FIRST_{suffix.upper()}"))
    assert matrix.matrix_nvals(A) == 3
    rows, cols, vals = extract(A)
    assert vals.dtype == dtype
    entries = sorted(zip(rows.tolist(), cols.tolist(), vals.tolist()))
    assert entries == [(0, 3, values[1]), (1, 0, values[2]), (2, 1, values[0])]


@pytest.mark.parametrize(("suffix", "dtype", "values"), TYPES)
def test_vector_build_and_extract_tuples(suffix, dtype, values):
    extract = getattr(vector, f"vector_extract_tuples_{suffix}")
    v = vector.vector_new(libget(f"GrB_{suffix.upper()}"), 5)
    for array, expected in zip(extract(v), [np.uint64, dtype]):
        assert array.dtype == expected
        assert array.size == 0

    indices = np.array([4, 0, 2], dtype=np.uint64)
    build = getattr(vector, f"vector_build_{suffix}")
    build(v, indices, np.array(values, dtype=dtype), 3, libget(f"GrB_FIRST_{suffix.upper()}"))
    assert vector.vector_nvals(v) == 3
    indices, vals = extract(v)
    assert vals.dtype == dtype
    entries = sorted(zip(indices.tolist(), vals.tolist()))
    assert entries == [(0, values[1]), (2, values[2]), (4, values[0])]


def test_matrix_without_gc():
    A = matrix.matrix_new(lib.GrB_INT64, 2, 3, free=None)
    matrix.set_int64(A, 7, 1, 2)
    B = matrix.matrix_dup(A, free=None)
    matrix.matrix_free(A)
    assert A[0] == ffi.NULL
    # The copy has its own data
    assert matrix.matrix_shape(B) == (2, 3)
    assert matrix.get_int64(B, 1, 2) == 7
    matrix.matrix_free(B)
    assert B[0] == ffi.NULL


def test_vector_without_gc():
    v = vector.vector_new(lib.GrB_INT64, 3, free=None)
    vector.set_int64(v, 7, 2)
    w = vector.vector_dup(v, free=None)
    vector.vector_free(v)
    assert v[0] == ffi.NULL
    # The copy has its own data
    assert vector.vector_size(w) == 3
    assert vector.get_int64(w, 2) == 7
    vector.vector_free(w)
    assert w[0] == ffi.NULL
