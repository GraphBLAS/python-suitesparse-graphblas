from suitesparse_graphblas import check_status, ffi, lib, supports_complex

from .io.serialize import deserialize_matrix as deserialize  # noqa: F401
from .io.serialize import serialize_matrix as serialize  # noqa: F401


def matrix_free(A):
    """Free a matrix."""
    check_status(A, lib.GrB_Matrix_free(A))


def matrix_new(T, nrows=lib.GxB_INDEX_MAX, ncols=lib.GxB_INDEX_MAX, *, free=matrix_free):
    """Create a new `GrB_Matrix` of type `T` and initialize it.  The
    following example creates an eight bit unsigned 2x2 matrix:

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_shape(A)
    (2, 2)

    The default value for `nrows` and `ncols` is `lib.GxB_INDEX_MAX`
    which creates a Matrix with maximal bounds:

    >>> A = matrix_new(lib.GrB_UINT8)
    >>> matrix_shape(A) == (lib.GxB_INDEX_MAX, lib.GxB_INDEX_MAX)
    True

    The `free` argument is called when the object is garbage
    collected, the default is `matrix.matrix_free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the matrix.

    """
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, T, nrows, ncols))
    if free:
        return ffi.gc(A, free)
    return A


def matrix_type(A):
    """Return the GraphBLAS type of the vector.

    >>> A = matrix_new(lib.GrB_UINT8)
    >>> matrix_type(A) == lib.GrB_UINT8
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(A, lib.GxB_Matrix_type(T, A[0]))
    return T[0]


def matrix_nrows(A):
    """Return the number of rows in the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 3)
    >>> matrix_nrows(A)
    2

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nrows(n, A[0]))
    return n[0]


def matrix_ncols(A):
    """Return the number of columns in the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 3)
    >>> matrix_ncols(A)
    3

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_ncols(n, A[0]))
    return n[0]


def matrix_nvals(A):
    """Return the number of stored elements in the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 3)
    >>> matrix_nvals(A)
    0

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nvals(n, A[0]))
    return n[0]


def matrix_shape(A):
    """Return the shape of the matrix as a two tuple `(nrows, ncols)`

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_shape(A)
    (2, 2)

    """
    return (matrix_nrows(A), matrix_ncols(A))


def matrix_format(A):
    """Return the format of the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_format(A) == lib.GxB_BY_ROW
    True

    """
    format = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get_INT32(A[0], lib.GxB_FORMAT, format))
    return format[0]


def matrix_set_format(A, format):
    """Set the format of the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_set_format(A, lib.GxB_BY_COL)
    >>> matrix_format(A) == lib.GxB_BY_COL
    True

    """
    format_val = ffi.cast("int32_t", format)
    check_status(A, lib.GxB_Matrix_Option_set_INT32(A[0], lib.GxB_FORMAT, format_val))


def matrix_sparsity_status(A):
    """Get the sparsity status of the matrix."""
    sparsity_status = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get_INT32(A[0], lib.GxB_SPARSITY_STATUS, sparsity_status))
    return sparsity_status[0]


def matrix_sparsity_control(A):
    """Get the sparsity control of the matrix."""
    sparsity_control = ffi.new("int32_t*")
    check_status(
        A, lib.GxB_Matrix_Option_get_INT32(A[0], lib.GxB_SPARSITY_CONTROL, sparsity_control)
    )
    return sparsity_control[0]


def matrix_set_sparsity_control(A, sparsity):
    """Set the sparsity control of the matrix."""
    sparsity_control = ffi.cast("int32_t", sparsity)
    check_status(
        A, lib.GxB_Matrix_Option_set_INT32(A[0], lib.GxB_SPARSITY_CONTROL, sparsity_control)
    )


def matrix_hyper_switch(A):
    """Get the hyper switch of the matrix."""
    hyper_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get_FP64(A[0], lib.GxB_HYPER_SWITCH, hyper_switch))
    return hyper_switch[0]


def matrix_set_hyper_switch(A, hyper_switch):
    """Set the hyper switch of the matrix."""
    hyper_switch = ffi.cast("double", hyper_switch)
    check_status(A, lib.GxB_Matrix_Option_set_FP64(A[0], lib.GxB_HYPER_SWITCH, hyper_switch))


def matrix_bitmap_switch(A):
    """Get the bitmap switch of the matrix."""
    bitmap_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get_FP64(A[0], lib.GxB_BITMAP_SWITCH, bitmap_switch))
    return bitmap_switch[0]


def matrix_set_bitmap_switch(A, bitmap_switch):
    """Set the bitmap switch of the matrix."""
    bitmap_switch = ffi.cast("double", bitmap_switch)
    check_status(A, lib.GxB_Matrix_Option_set_FP64(A[0], lib.GxB_BITMAP_SWITCH, bitmap_switch))


def set_bool(A, value, i, j):
    """Set a boolean value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_BOOL, 3, 3)
    >>> set_bool(A, True, 2, 2)
    >>> get_bool(A, 2, 2) == True
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_BOOL(A[0], value, i, j))


def get_bool(A, i, j):
    """Get a boolean value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_BOOL, 3, 3)
    >>> set_bool(A, True, 2, 2)
    >>> get_bool(A, 2, 2) == True
    True

    """
    value = ffi.new("bool*")
    check_status(A, lib.GrB_Matrix_extractElement_BOOL(value, A[0], i, j))
    return value[0]


def set_int8(A, value, i, j):
    """Set an int8 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT8, 3, 3)
    >>> set_int8(A, 7, 2, 2)
    >>> get_int8(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT8(A[0], value, i, j))


def get_int8(A, i, j):
    """Get an int8 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT8, 3, 3)
    >>> set_int8(A, 7, 2, 2)
    >>> get_int8(A, 2, 2) == 7
    True

    """
    value = ffi.new("int8_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT8(value, A[0], i, j))
    return value[0]


def set_int16(A, value, i, j):
    """Set an int16 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT16, 3, 3)
    >>> set_int16(A, 7, 2, 2)
    >>> get_int16(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT16(A[0], value, i, j))


def get_int16(A, i, j):
    """Get an int16 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT16, 3, 3)
    >>> set_int16(A, 7, 2, 2)
    >>> get_int16(A, 2, 2) == 7
    True

    """
    value = ffi.new("int16_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT16(value, A[0], i, j))
    return value[0]


def set_int32(A, value, i, j):
    """Set an int32 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT32, 3, 3)
    >>> set_int32(A, 7, 2, 2)
    >>> get_int32(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT32(A[0], value, i, j))


def get_int32(A, i, j):
    """Get an int32 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT32, 3, 3)
    >>> set_int32(A, 7, 2, 2)
    >>> get_int32(A, 2, 2) == 7
    True

    """
    value = ffi.new("int32_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT32(value, A[0], i, j))
    return value[0]


def set_int64(A, value, i, j):
    """Set an int64 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT64, 3, 3)
    >>> set_int64(A, 7, 2, 2)
    >>> get_int64(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT64(A[0], value, i, j))


def get_int64(A, i, j):
    """Get an int64 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT64, 3, 3)
    >>> set_int64(A, 7, 2, 2)
    >>> get_int64(A, 2, 2) == 7
    True

    """
    value = ffi.new("int64_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT64(value, A[0], i, j))
    return value[0]


def set_uint8(A, value, i, j):
    """Set a uint8 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT8, 3, 3)
    >>> set_uint8(A, 7, 2, 2)
    >>> get_uint8(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT8(A[0], value, i, j))


def get_uint8(A, i, j):
    """Get a uint8 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT8, 3, 3)
    >>> set_uint8(A, 7, 2, 2)
    >>> get_uint8(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint8_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT8(value, A[0], i, j))
    return value[0]


def set_uint16(A, value, i, j):
    """Set a uint16 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT16, 3, 3)
    >>> set_uint16(A, 7, 2, 2)
    >>> get_uint16(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT16(A[0], value, i, j))


def get_uint16(A, i, j):
    """Get a uint16 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT16, 3, 3)
    >>> set_uint16(A, 7, 2, 2)
    >>> get_uint16(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint16_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT16(value, A[0], i, j))
    return value[0]


def set_uint32(A, value, i, j):
    """Set a uint32 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT32, 3, 3)
    >>> set_uint32(A, 7, 2, 2)
    >>> get_uint32(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT32(A[0], value, i, j))


def get_uint32(A, i, j):
    """Get a uint32 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT32, 3, 3)
    >>> set_uint32(A, 7, 2, 2)
    >>> get_uint32(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint32_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT32(value, A[0], i, j))
    return value[0]


def set_uint64(A, value, i, j):
    """Set a uint64 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT64, 3, 3)
    >>> set_uint64(A, 7, 2, 2)
    >>> get_uint64(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT64(A[0], value, i, j))


def get_uint64(A, i, j):
    """Get a uint64 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT64, 3, 3)
    >>> set_uint64(A, 7, 2, 2)
    >>> get_uint64(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint64_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT64(value, A[0], i, j))
    return value[0]


def set_fp32(A, value, i, j):
    """Set an fp32 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP32, 3, 3)
    >>> set_fp32(A, 1.5, 2, 2)
    >>> get_fp32(A, 2, 2) == 1.5
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_FP32(A[0], value, i, j))


def get_fp32(A, i, j):
    """Get an fp32 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP32, 3, 3)
    >>> set_fp32(A, 1.5, 2, 2)
    >>> get_fp32(A, 2, 2) == 1.5
    True

    """
    value = ffi.new("float*")
    check_status(A, lib.GrB_Matrix_extractElement_FP32(value, A[0], i, j))
    return value[0]


def set_fp64(A, value, i, j):
    """Set an fp64 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP64, 3, 3)
    >>> set_fp64(A, 1.5, 2, 2)
    >>> get_fp64(A, 2, 2) == 1.5
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_FP64(A[0], value, i, j))


def get_fp64(A, i, j):
    """Get an fp64 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP64, 3, 3)
    >>> set_fp64(A, 1.5, 2, 2)
    >>> get_fp64(A, 2, 2) == 1.5
    True

    """
    value = ffi.new("double*")
    check_status(A, lib.GrB_Matrix_extractElement_FP64(value, A[0], i, j))
    return value[0]


if supports_complex():

    def set_fc32(A, value, i, j):
        """Set an fc32 value to the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC32, 3, 3)
        >>> set_fc32(A, 2+3j, 2, 2)
        >>> get_fc32(A, 2, 2) == 2+3j
        True

        """
        check_status(A, lib.GxB_Matrix_setElement_FC32(A[0], value, i, j))

    def get_fc32(A, i, j):
        """Get an fc32 value from the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC32, 3, 3)
        >>> set_fc32(A, 2+3j, 2, 2)
        >>> get_fc32(A, 2, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC32_t*")
        check_status(A, lib.GxB_Matrix_extractElement_FC32(value, A[0], i, j))
        return value[0]

    def set_fc64(A, value, i, j):
        """Set an fc64 value to the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC64, 3, 3)
        >>> set_fc64(A, 2+3j, 2, 2)
        >>> get_fc64(A, 2, 2) == 2+3j
        True

        """
        check_status(A, lib.GxB_Matrix_setElement_FC64(A[0], value, i, j))

    def get_fc64(A, i, j):
        """Get an fc64 value from the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC64, 3, 3)
        >>> set_fc64(A, 2+3j, 2, 2)
        >>> get_fc64(A, 2, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC64_t*")
        check_status(A, lib.GxB_Matrix_extractElement_FC64(value, A[0], i, j))
        return value[0]
