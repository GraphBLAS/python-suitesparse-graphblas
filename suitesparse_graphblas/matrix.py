from suitesparse_graphblas import (
    lib,
    ffi,
    check_status,
)


def free(A):
    """Free a matrix."""
    check_status(A, lib.GrB_Matrix_free(A))


def new(T, nrows=lib.GxB_INDEX_MAX, ncols=lib.GxB_INDEX_MAX, *, free=free):
    """Create a new `GrB_Matrix` of type `T` and initialize it.  The
    following example creates an eight bit unsigned 2x2 matrix:

    >>> A = new(lib.GrB_UINT8, 2, 2)
    >>> shape(A)
    (2, 2)

    The default value for `nrows` and `ncols` is `lib.GxB_INDEX_MAX`
    which creates a Matrix with maximal bounds:

    >>> A = new(lib.GrB_UINT8)
    >>> shape(A) == (lib.GxB_INDEX_MAX, lib.GxB_INDEX_MAX)
    True

    The `free` argument is called when the object is garbage
    collected, the default is `matrix.free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the matrix.

    """
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, T, nrows, ncols))
    if free:
        return ffi.gc(A, free)
    return A


def type(A):
    """Return the GraphBLAS type of the vector.

    >>> A = new(lib.GrB_UINT8)
    >>> type(A) == lib.GrB_UINT8
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(A, lib.GxB_Matrix_type(T, A[0]))
    return T[0]


def nrows(A):
    """Return the number of rows in the matrix.

    >>> A = new(lib.GrB_UINT8, 2, 3)
    >>> nrows(A)
    2

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nrows(n, A[0]))
    return n[0]


def ncols(A):
    """Return the number of columns in the matrix.

    >>> A = new(lib.GrB_UINT8, 2, 3)
    >>> ncols(A)
    3

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_ncols(n, A[0]))
    return n[0]


def nvals(A):
    """Return the number of stored elements in the matrix.

    >>> A = new(lib.GrB_UINT8, 2, 3)
    >>> nvals(A)
    0

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nvals(n, A[0]))
    return n[0]


def shape(A):
    """Return the shape of the matrix as a two tuple `(nrows, ncols)`

    >>> A = new(lib.GrB_UINT8, 2, 2)
    >>> shape(A)
    (2, 2)

    """
    return (nrows(A), ncols(A))


def format(A):
    """Return the format of the matrix.

    >>> A = new(lib.GrB_UINT8, 2, 2)
    >>> format(A) == lib.GxB_BY_ROW
    True

    """
    format = ffi.new("GxB_Format_Value*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_FORMAT, format))
    return format[0]


def set_format(A, format):
    """Set the format of the matrix.

    >>> A = new(lib.GrB_UINT8, 2, 2)
    >>> set_format(A, lib.GxB_BY_COL)
    >>> format(A) == lib.GxB_BY_COL
    True

    """
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_FORMAT, ffi.cast("GxB_Format_Value", format))
    )


def sparsity_status(A):
    """Get the sparsity status of the matrix."""
    sparsity_status = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_SPARSITY_STATUS, sparsity_status))
    return sparsity_status[0]


def sparsity_control(A):
    """Get the sparsity control of the matrix."""
    sparsity_control = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_SPARSITY_CONTROL, sparsity_control))
    return sparsity_control[0]


def set_sparsity_control(A, sparsity):
    """Set the sparsity control of the matrix."""
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_SPARSITY_CONTROL, ffi.cast("int", sparsity))
    )


def hyper_switch(A):
    """Get the hyper switch of the matrix."""
    hyper_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_HYPER_SWITCH, hyper_switch))
    return hyper_switch[0]


def set_hyper_switch(A, hyper_switch):
    """Set the hyper switch of the matrix."""
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_HYPER_SWITCH, ffi.cast("double", hyper_switch))
    )


def bitmap_switch(A):
    """Get the bitmap switch of the matrix."""
    bitmap_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_BITMAP_SWITCH, bitmap_switch))
    return bitmap_switch[0]


def set_bitmap_switch(A, bitmap_switch):
    """Set the bitmap switch of the matrix."""
    check_status(
        A, lib.GxB_Matrix_Option_set(A[0], lib.GxB_BITMAP_SWITCH, ffi.cast("double", bitmap_switch))
    )


def set_bool(A, value, i, j):
    """Set a boolean value to the matrix at row `i` column `j`.

    >>> A = new(lib.GrB_BOOL, 3, 3)
    >>> set_bool(A, True, 2, 2)
    >>> bool(A, 2, 2) == True
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_BOOL(A[0], value, i, j))


def bool(A, i, j):
    """Get a boolean value from the matrix at row `i` column `j`.

    >>> A = new(lib.GrB_BOOL, 3, 3)
    >>> set_bool(A, True, 2, 2)
    >>> bool(A, 2, 2) == True
    True

    """
    value = ffi.new("bool*")
    check_status(A, lib.GrB_Matrix_extractElement_BOOL(value, A[0], i, j))
    return value[0]
