import platform
import pytest

if platform.system() == "Windows":
    pytest.skip("skipping windows-only tests", allow_module_level=True)

import gzip
import bz2
import lzma
from pathlib import Path

from suitesparse_graphblas import (
    ffi,
    lib,
    check_status,
    grb_types,
    bool_types,
    signed_integer_types,
    unsigned_integer_types,
    real_types,
    complex_types,
    supports_complex,
    matrix,
)

from suitesparse_graphblas.io import binary

NULL = ffi.NULL


def _test_elements(T):
    if T in bool_types:
        return [True, False], [0, 0], [1, 1]
    elif T in signed_integer_types:
        return [1, -42], [0, 0], [1, 1]
    elif T in unsigned_integer_types:
        return [1, 42], [0, 0], [1, 1]
    elif T in real_types:
        return [1.0, -42.42], [0, 0], [1, 1]
    elif T in complex_types:
        return [complex(1.0, 1.0), complex(42.0, -42.0)], [0, 0], [1, 1]


_element_setters = {
    lib.GrB_BOOL: lib.GrB_Matrix_setElement_BOOL,
    lib.GrB_INT8: lib.GrB_Matrix_setElement_INT8,
    lib.GrB_INT16: lib.GrB_Matrix_setElement_INT16,
    lib.GrB_INT32: lib.GrB_Matrix_setElement_INT32,
    lib.GrB_INT64: lib.GrB_Matrix_setElement_INT64,
    lib.GrB_UINT8: lib.GrB_Matrix_setElement_UINT8,
    lib.GrB_UINT16: lib.GrB_Matrix_setElement_UINT16,
    lib.GrB_UINT32: lib.GrB_Matrix_setElement_UINT32,
    lib.GrB_UINT64: lib.GrB_Matrix_setElement_UINT64,
    lib.GrB_FP32: lib.GrB_Matrix_setElement_FP32,
    lib.GrB_FP64: lib.GrB_Matrix_setElement_FP64,
}

if supports_complex():
    _element_setters.update(
        {
            lib.GxB_FC32: lib.GxB_Matrix_setElement_FC32,
            lib.GxB_FC64: lib.GxB_Matrix_setElement_FC64,
        }
    )


_eq_ops = {
    lib.GrB_BOOL: lib.GrB_EQ_BOOL,
    lib.GrB_INT8: lib.GrB_EQ_INT8,
    lib.GrB_INT16: lib.GrB_EQ_INT16,
    lib.GrB_INT32: lib.GrB_EQ_INT32,
    lib.GrB_INT64: lib.GrB_EQ_INT64,
    lib.GrB_UINT8: lib.GrB_EQ_UINT8,
    lib.GrB_UINT16: lib.GrB_EQ_UINT16,
    lib.GrB_UINT32: lib.GrB_EQ_UINT32,
    lib.GrB_UINT64: lib.GrB_EQ_UINT64,
    lib.GrB_FP32: lib.GrB_EQ_FP32,
    lib.GrB_FP64: lib.GrB_EQ_FP64,
}

if supports_complex():
    _eq_ops.update(
        {
            lib.GxB_FC32: lib.GxB_EQ_FC32,
            lib.GxB_FC64: lib.GxB_EQ_FC64,
        }
    )


def test_matrix_binfile_read_write(tmp_path):
    for opener in (Path.open, gzip.open, bz2.open, lzma.open):
        for format in (lib.GxB_BY_ROW, lib.GxB_BY_COL):
            for T in grb_types:
                for sparsity in (lib.GxB_HYPERSPARSE, lib.GxB_SPARSE, lib.GxB_BITMAP, lib.GxB_FULL):

                    A = matrix.new(T, 2, 2)

                    if T is not lib.GxB_FULL:
                        for args in zip(*_test_elements(T)):
                            f = _element_setters[T]
                            check_status(A, f(A[0], *args))
                    else:
                        Tone = _test_elements(T)[0][0]
                        check_status(
                            A[0],
                            lib.GrB_assign(
                                A,
                                NULL,
                                NULL,
                                Tone,
                                lib.GrB_ALL,
                                0,
                                lib.GrB_ALL,
                                0,
                                NULL,
                            ),
                        )
                    matrix.set_sparsity_control(A, sparsity)
                    matrix.set_format(A, format)

                    binfilef = tmp_path / "binfilewrite_test.binfile"
                    binary.binwrite(A, binfilef, opener=opener)
                    B = binary.binread(binfilef, opener=opener)

                    assert matrix.type(A) == matrix.type(B)
                    assert matrix.nrows(A) == matrix.nrows(B)
                    assert matrix.ncols(A) == matrix.ncols(B)
                    assert matrix.hyper_switch(A) == matrix.hyper_switch(B)
                    assert matrix.bitmap_switch(A) == matrix.bitmap_switch(B)
                    # assert matrix.sparsity_control(A) == matrix.sparsity_control(B)

                    C = matrix.new(lib.GrB_BOOL, 2, 2)

                    check_status(
                        C,
                        lib.GrB_Matrix_eWiseAdd_BinaryOp(
                            C[0], NULL, NULL, _eq_ops[T], A[0], B[0], NULL
                        ),
                    )

                    assert matrix.nvals(A) == matrix.nvals(B) == matrix.nvals(C)

                    is_eq = ffi.new("bool*")
                    check_status(
                        C,
                        lib.GrB_Matrix_reduce_BOOL(
                            is_eq, NULL, lib.GrB_LAND_MONOID_BOOL, C[0], NULL
                        ),
                    )

                    assert is_eq[0]
