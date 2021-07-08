import gzip, bz2, lzma
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
    lib.GxB_FC32: lib.GxB_Matrix_setElement_FC32,
    lib.GxB_FC64: lib.GxB_Matrix_setElement_FC64,
}

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
    lib.GxB_FC32: lib.GxB_EQ_FC32,
    lib.GxB_FC64: lib.GxB_EQ_FC64,
}


def test_matrix_binfile_read_write(tmp_path):
    for opener in (Path.open, gzip.open, bz2.open, lzma.open):
        for format in (lib.GxB_BY_ROW, lib.GxB_BY_COL):
            for T in grb_types:
                for sparsity in (lib.GxB_HYPERSPARSE, lib.GxB_SPARSE, lib.GxB_BITMAP, lib.GxB_FULL):

                    A = ffi.new("GrB_Matrix*")
                    check_status(A, lib.GrB_Matrix_new(A, T, 2, 2))

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
                    check_status(
                        A[0],
                        lib.GxB_Matrix_Option_set(
                            A[0], lib.GxB_SPARSITY_CONTROL, ffi.cast("int", sparsity)
                        ),
                    )
                    check_status(
                        A[0],
                        lib.GxB_Matrix_Option_set(
                            A[0], lib.GxB_FORMAT, ffi.cast("GxB_Format_Value", format)
                        ),
                    )
                    binfilef = tmp_path / "binfilewrite_test.binfile"
                    binary.binwrite(A, binfilef, opener=opener)
                    B = binary.binread(binfilef, opener=opener)

                    Atype = ffi.new("GrB_Type*")
                    Anrows = ffi.new("GrB_Index*")
                    Ancols = ffi.new("GrB_Index*")
                    Anvals = ffi.new("GrB_Index*")
                    Asparsity_control = ffi.new("int32_t*")
                    Ahyper_switch = ffi.new("double*")
                    Abitmap_switch = ffi.new("double*")
                    check_status(A, lib.GxB_Matrix_type(Atype, A[0]))
                    check_status(A, lib.GrB_Matrix_nrows(Anrows, A[0]))
                    check_status(A, lib.GrB_Matrix_ncols(Ancols, A[0]))
                    check_status(A, lib.GrB_Matrix_nvals(Anvals, A[0]))
                    check_status(
                        A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_HYPER_SWITCH, Ahyper_switch)
                    )
                    check_status(
                        A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_BITMAP_SWITCH, Abitmap_switch)
                    )
                    check_status(
                        A,
                        lib.GxB_Matrix_Option_get(
                            A[0], lib.GxB_SPARSITY_CONTROL, Asparsity_control
                        ),
                    )

                    Btype = ffi.new("GrB_Type*")
                    Bnrows = ffi.new("GrB_Index*")
                    Bncols = ffi.new("GrB_Index*")
                    Bnvals = ffi.new("GrB_Index*")
                    Bsparsity_control = ffi.new("int32_t*")
                    Bhyper_switch = ffi.new("double*")
                    Bbitmap_switch = ffi.new("double*")
                    check_status(B, lib.GxB_Matrix_type(Btype, B[0]))
                    check_status(B, lib.GrB_Matrix_nrows(Bnrows, B[0]))
                    check_status(B, lib.GrB_Matrix_ncols(Bncols, B[0]))
                    check_status(B, lib.GrB_Matrix_nvals(Bnvals, B[0]))
                    check_status(
                        B, lib.GxB_Matrix_Option_get(B[0], lib.GxB_HYPER_SWITCH, Bhyper_switch)
                    )
                    check_status(
                        B, lib.GxB_Matrix_Option_get(B[0], lib.GxB_BITMAP_SWITCH, Bbitmap_switch)
                    )
                    check_status(
                        B,
                        lib.GxB_Matrix_Option_get(
                            B[0], lib.GxB_SPARSITY_CONTROL, Bsparsity_control
                        ),
                    )

                    assert Atype[0] == Btype[0]
                    assert Anrows[0] == Bnrows[0]
                    assert Ancols[0] == Bncols[0]
                    assert Ahyper_switch[0] == Bhyper_switch[0]
                    assert Abitmap_switch[0] == Bbitmap_switch[0]
                    # assert Asparsity_control[0] == Bsparsity_control[0]

                    C = ffi.new("GrB_Matrix*")
                    check_status(C, lib.GrB_Matrix_new(C, lib.GrB_BOOL, 2, 2))

                    check_status(
                        C,
                        lib.GrB_Matrix_eWiseAdd_BinaryOp(
                            C[0], NULL, NULL, _eq_ops[T], A[0], B[0], NULL
                        ),
                    )

                    Cnvals = ffi.new("GrB_Index*")
                    check_status(C, lib.GrB_Matrix_nvals(Cnvals, C[0]))

                    assert Cnvals[0] == Anvals[0] == Bnvals[0]

                    is_eq = ffi.new("bool*")
                    check_status(
                        C,
                        lib.GrB_Matrix_reduce_BOOL(
                            is_eq, NULL, lib.GrB_LAND_MONOID_BOOL, C[0], NULL
                        ),
                    )

                    assert is_eq[0]

                    check_status(A, lib.GrB_Matrix_free(A))
                    check_status(B, lib.GrB_Matrix_free(B))
                    check_status(C, lib.GrB_Matrix_free(C))
