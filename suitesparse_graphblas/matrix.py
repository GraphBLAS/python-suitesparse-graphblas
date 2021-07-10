from suitesparse_graphblas import (
    lib,
    ffi,
    check_status,
)


def free(A):
    check_status(A, lib.GrB_Matrix_free(A))


def new(T, nrows=lib.GxB_INDEX_MAX, ncols=lib.GxB_INDEX_MAX, *, free=free):
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, T, nrows, ncols))
    if free:
        return ffi.gc(A, free)
    return A


def type(A):
    T = ffi.new("GrB_Type*")
    check_status(A, lib.GxB_Matrix_type(T, A[0]))
    return T[0]


def nrows(A):
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nrows(n, A[0]))
    return n[0]


def ncols(A):
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_ncols(n, A[0]))
    return n[0]


def nvals(A):
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nvals(n, A[0]))
    return n[0]


def sparsity_status(A):
    sparsity_status = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_SPARSITY_STATUS, sparsity_status))
    return sparsity_status[0]


def sparsity_control(A):
    sparsity_control = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_SPARSITY_CONTROL, sparsity_control))
    return sparsity_control[0]


def set_sparsity_control(A, sparsity):
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_SPARSITY_CONTROL, ffi.cast("int", sparsity))
    )


def hyper_switch(A):
    hyper_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_HYPER_SWITCH, hyper_switch))
    return hyper_switch[0]


def set_hyper_switch(A, hyper_switch):
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_HYPER_SWITCH, ffi.cast("double", hyper_switch))
    )


def bitmap_switch(A):
    bitmap_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_BITMAP_SWITCH, bitmap_switch))
    return bitmap_switch[0]


def set_bitmap_switch(A, bitmap_switch):
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_BITMAP_SWITCH, ffi.cast("double", bitmap_switch))
    )


def format(A):
    format = ffi.new("GxB_Format_Value*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_FORMAT, format))
    return format[0]


def set_format(A, format):
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_FORMAT, ffi.cast("GxB_Format_Value", format))
    )
